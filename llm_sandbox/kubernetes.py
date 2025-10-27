import io
import logging
import shlex
import tarfile
import time
import uuid
from pathlib import Path
from typing import Any

from kubernetes import client as k8s_client
from kubernetes.client import CoreV1Api
from kubernetes.client.exceptions import ApiException
from kubernetes.stream import stream

from llm_sandbox.const import DefaultImage, SupportedLanguage
from llm_sandbox.core.config import SessionConfig
from llm_sandbox.core.session_base import BaseSession
from llm_sandbox.data import ConsoleOutput
from llm_sandbox.exceptions import CommandEmptyError, ContainerError, NotOpenSessionError
from llm_sandbox.security import SecurityPolicy

SH_SHELL = "/bin/sh"
POD_STARTUP_TIMEOUT = 300  # 5 minutes
KUBERNETES_POD_NOT_FOUND_ERROR_CODE = 404
KUBERNETES_POD_STATUS_POLL_INTERVAL = 2


class KubernetesContainerAPI:
    """Kubernetes implementation of the ContainerAPI protocol."""

    def __init__(self, client: CoreV1Api, namespace: str = "default") -> None:
        """Initialize Kubernetes container API."""
        self.client = client
        self.namespace = namespace

    def create_container(self, config: Any) -> Any:
        """Create Kubernetes pod."""
        pod_manifest = config["pod_manifest"]
        self.client.create_namespaced_pod(namespace=self.namespace, body=pod_manifest)

        # Wait for pod to be running
        pod_name = pod_manifest["metadata"]["name"]
        start_time = time.time()
        while True:
            pod = self.client.read_namespaced_pod(name=pod_name, namespace=self.namespace)
            if pod.status.phase == "Running":
                break
            time.sleep(1)
            if time.time() - start_time > POD_STARTUP_TIMEOUT:
                msg = f"Pod {pod_name} did not start within {POD_STARTUP_TIMEOUT} seconds"
                raise TimeoutError(msg)

        return pod_name

    def start_container(self, container: Any) -> None:
        """Start container (no-op for Kubernetes as pod is already running)."""

    def stop_container(self, container: Any) -> None:
        """Stop Kubernetes pod."""
        try:
            self.client.delete_namespaced_pod(
                name=container,
                namespace=self.namespace,
                body=k8s_client.V1DeleteOptions(),
            )
        except Exception as e:  # noqa: BLE001
            # Pod might already be deleted, log but don't raise
            import logging

            logging.getLogger(__name__).debug("Failed to delete pod %s: %s", container, e)

    def execute_command(self, container: Any, command: str, **kwargs: Any) -> tuple[int, Any]:
        """Execute command in Kubernetes pod."""
        workdir = kwargs.get("workdir")
        container_name = kwargs.get("container_name")  # Get the specific container name

        exec_command = [SH_SHELL, "-c", f"cd {workdir} && {command}"] if workdir else [SH_SHELL, "-c", command]

        resp = stream(
            self.client.connect_get_namespaced_pod_exec,
            container,
            self.namespace,
            command=exec_command,
            container=container_name,  # Specify which container to execute in
            stderr=True,
            stdin=False,
            stdout=True,
            tty=False,
            _preload_content=False,
        )

        stdout_output = ""
        stderr_output = ""

        while resp.is_open():
            resp.update(timeout=1)

            if resp.peek_stdout():
                chunk = resp.read_stdout()
                stdout_output += chunk

            if resp.peek_stderr():
                chunk = resp.read_stderr()
                stderr_output += chunk

        # Ensure we wait for the command to complete properly
        resp.close()
        exit_code = resp.returncode or 0
        return exit_code, (stdout_output, stderr_output)

    def copy_to_container(self, container: Any, src: str, dest: str, **kwargs: Any) -> None:
        """Copy file to Kubernetes pod."""
        # Validate source path exists and is accessible
        src_path = Path(src)
        if not (src_path.exists() and (src_path.is_file() or src_path.is_dir())):
            msg = f"Source path {src} does not exist or is not accessible"
            raise FileNotFoundError(msg)

        dest_dir = str(Path(dest).parent)
        container_name = kwargs.get("container_name")  # Get the specific container name

        # Create destination directory
        if dest_dir:
            exec_command = ["mkdir", "-p", dest_dir]
            resp = stream(
                self.client.connect_get_namespaced_pod_exec,
                container,
                self.namespace,
                command=exec_command,
                container=container_name,  # Specify which container to use
                stderr=True,
                stdin=False,
                stdout=True,
                tty=False,
                _preload_content=False,
            )

            stderr_output = ""
            while resp.is_open():
                resp.update(timeout=1)
                if resp.peek_stderr():
                    stderr_output += resp.read_stderr()

            if resp.returncode != 0:
                msg = f"Failed to create directory {dest_dir}: {stderr_output}"
                raise RuntimeError(msg)

        # Create tar archive
        tarstream = io.BytesIO()
        with tarfile.open(fileobj=tarstream, mode="w") as tar:
            tar.add(src, arcname=Path(dest).name)
        tarstream.seek(0)

        tar_size = len(tarstream.getvalue())
        tarstream.seek(0)

        exec_command = ["tar", "xf", "-", "-C", dest_dir]
        resp = stream(
            self.client.connect_get_namespaced_pod_exec,
            container,
            self.namespace,
            command=exec_command,
            container=container_name,
            stderr=True,
            stdin=True,
            stdout=True,
            tty=False,
            _preload_content=False,
        )

        chunk_size = 65536
        bytes_written = 0

        try:
            while True:
                chunk = tarstream.read(chunk_size)
                if not chunk:
                    break
                resp.write_stdin(chunk)
                bytes_written += len(chunk)

            resp.write_stdin("")

            stderr_output = ""
            for _ in range(10):
                resp.update(timeout=1)
                if resp.peek_stderr():
                    stderr_output += resp.read_stderr()
                if not resp.is_open():
                    break

        finally:
            resp.close()

        logger = logging.getLogger(__name__)
        logger.debug(
            "Copied %d bytes (%d tar size) from '%s' to pod '%s:%s'",
            bytes_written,
            tar_size,
            src,
            container,
            dest,
        )

        if stderr_output and "error" in stderr_output.lower():
            logger.warning("Tar extraction warnings for %s: %s", dest, stderr_output)

    def copy_from_container(self, container: Any, src: str, **kwargs: Any) -> tuple[bytes, dict]:
        """Copy file from Kubernetes pod."""
        container_name = kwargs.get("container_name")  # Get the specific container name

        stat_command = f"stat -c '%s %Y %n' {shlex.quote(str(src))} 2>/dev/null || echo 'NOT_FOUND'"
        exec_command = [SH_SHELL, "-c", stat_command]

        resp = stream(
            self.client.connect_get_namespaced_pod_exec,
            container,
            self.namespace,
            command=exec_command,
            container=container_name,
            stderr=True,
            stdin=False,
            stdout=True,
            tty=False,
            _preload_content=False,
        )

        stdout_output = ""
        while resp.is_open():
            resp.update(timeout=1)
            if resp.peek_stdout():
                stdout_output += resp.read_stdout()

        if stdout_output.strip() == "NOT_FOUND":
            return b"", {"size": 0}

        stat_parts = stdout_output.strip().split(" ", 2)
        if len(stat_parts) >= 3:  # noqa: PLR2004
            file_size = int(stat_parts[0])
            mtime = int(stat_parts[1])
            file_name = stat_parts[2]
        else:
            file_size = 0
            mtime = 0
            file_name = src

        src_path = Path(src)
        parent_dir = src_path.parent
        target_name = src_path.name
        base64_command = f"tar -C {shlex.quote(str(parent_dir))} -cf - {shlex.quote(target_name)} | base64 -w 0"
        exec_command = [SH_SHELL, "-c", base64_command]

        resp = stream(
            self.client.connect_get_namespaced_pod_exec,
            container,
            self.namespace,
            command=exec_command,
            container=container_name,
            stderr=True,
            stdin=False,
            stdout=True,
            tty=False,
            _preload_content=False,
        )

        stdout_output = ""
        while resp.is_open():
            resp.update(timeout=1)
            if resp.peek_stdout():
                stdout_output += resp.read_stdout()

        if not stdout_output.strip():
            return b"", {"size": 0}

        # Decode the base64 data
        try:
            import base64

            tar_data = base64.b64decode(stdout_output.strip())
        except Exception:  # noqa: BLE001
            return b"", {"size": 0}

        # Create stat dict
        stat_dict = {
            "name": file_name,
            "size": file_size,
            "mtime": mtime,
            "mode": 0o644,
            "linkTarget": "",
        }

        return tar_data, stat_dict


class SandboxKubernetesSession(BaseSession):
    r"""Sandbox session implemented using Kubernetes Pods.

    This class provides a sandboxed environment for code execution by leveraging Kubernetes.
    It handles Pod creation and lifecycle based on a provided or default manifest,
    code execution, library installation, and file operations within the Kubernetes Pod.
    """

    def __init__(
        self,  # NOSONAR (too many arguments)
        client: CoreV1Api | None = None,
        image: str | None = None,
        lang: str = SupportedLanguage.PYTHON,
        verbose: bool = False,
        kube_namespace: str = "default",
        env_vars: dict[str, str] | None = None,
        pod_manifest: dict | None = None,
        workdir: str = "/sandbox",
        security_policy: SecurityPolicy | None = None,
        default_timeout: float | None = None,
        execution_timeout: float | None = None,
        session_timeout: float | None = None,
        container_id: str | None = None,  # This will be pod_id for Kubernetes
        skip_environment_setup: bool = False,
        **kwargs: Any,
    ) -> None:
        r"""Initialize Kubernetes session.

        Args:
            client (CoreV1Api | None): The Kubernetes client to use.
            image (str | None): The image to use.
            lang (str): The language to use.
            verbose (bool): Whether to enable verbose output.
            kube_namespace (str): The Kubernetes namespace to use.
            env_vars (dict[str, str] | None): The environment variables to use.
            pod_manifest (dict | None): The Kubernetes pod manifest to use.
            workdir (str): The working directory to use.
            security_policy (SecurityPolicy | None): The security policy to use.
            default_timeout (float | None): The default timeout to use.
            execution_timeout (float | None): The execution timeout to use.
            session_timeout (float | None): The session timeout to use.
            container_id (str | None): ID of existing pod to connect to.
            skip_environment_setup (bool): Skip language-specific environment setup.
            **kwargs: Additional keyword arguments.

        Returns:
            None

        """
        config = SessionConfig(
            image=image,
            lang=SupportedLanguage(lang.upper()),
            verbose=verbose,
            workdir=workdir,
            security_policy=security_policy,
            default_timeout=default_timeout,
            execution_timeout=execution_timeout,
            session_timeout=session_timeout,
            container_id=container_id,
            skip_environment_setup=skip_environment_setup,
        )

        super().__init__(config=config, **kwargs)

        if not client:
            self._log("Using local Kubernetes context since client is not provided.")
            from kubernetes import config as k8s_config

            k8s_config.load_kube_config()
            self.client = CoreV1Api()
        else:
            self.client = client

        self.kube_namespace = kube_namespace
        self.container_api = KubernetesContainerAPI(self.client, kube_namespace)

        # Generate unique pod name (only if not using existing pod)
        if not self.using_existing_container:
            short_uuid = uuid.uuid4().hex[:8]
            self.pod_name = f"sandbox-{lang.lower()}-{short_uuid}"
            self.env_vars = env_vars
            self.pod_manifest = pod_manifest or self._default_pod_manifest()
            self._reconfigure_with_pod_manifest()

            # Extract container name from pod manifest for command execution
            containers = self.pod_manifest.get("spec", {}).get("containers", [])
            if containers:
                self.container_name = containers[0]["name"]
            else:
                self.container_name = "sandbox-container"  # fallback
        elif container_id:
            self.pod_name = container_id
            # For existing containers, we'll need to query the pod to get container name
            self.container_name = None  # Will be set when connecting

        # For compatibility with base class
        self.stream = False

    def _default_pod_manifest(self) -> dict:
        """Generate a default Kubernetes Pod manifest."""
        image = self.config.image or DefaultImage.__dict__[self.config.lang.upper()]

        pod_manifest = {
            "apiVersion": "v1",
            "kind": "Pod",
            "metadata": {
                "name": self.pod_name,
                "namespace": self.kube_namespace,
                "labels": {"app": "sandbox"},
            },
            "spec": {
                "containers": [
                    {
                        "name": "sandbox-container",
                        "image": image,
                        "tty": True,
                        "securityContext": {
                            "runAsUser": 0,
                            "runAsGroup": 0,
                        },
                    }
                ],
                "securityContext": {
                    "runAsUser": 0,
                    "runAsGroup": 0,
                },
            },
        }

        if self.env_vars:
            containers = pod_manifest["spec"]["containers"]  # type: ignore[index]
            containers[0]["env"] = [{"name": key, "value": value} for key, value in self.env_vars.items()]
        return pod_manifest

    def _reconfigure_with_pod_manifest(self) -> None:
        """Reconfigure session attributes based on the pod manifest."""
        self.pod_manifest["metadata"]["name"] = self.pod_name
        self.kube_namespace = self.pod_manifest.get("metadata", {}).get("namespace", self.kube_namespace)

    def _wait_for_pod_to_start(self, pod_id: str, timeout: int = POD_STARTUP_TIMEOUT) -> None:
        """Wait for a pending pod to start running.

        Args:
            pod_id (str): The name of the pod to wait for.
            timeout (int): Maximum time to wait in seconds.

        Raises:
            ContainerError: If pod doesn't start within timeout.

        """
        self._log(f"Pod {pod_id} is pending, waiting for it to start...")
        start_time = time.time()

        while time.time() - start_time < timeout:
            pod = self.client.read_namespaced_pod(name=pod_id, namespace=self.kube_namespace)
            if pod.status.phase == "Running":
                return
            time.sleep(KUBERNETES_POD_STATUS_POLL_INTERVAL)

        # If we get here, pod didn't start in time
        pod = self.client.read_namespaced_pod(name=pod_id, namespace=self.kube_namespace)
        msg = f"Pod {pod_id} is not running (status: {pod.status.phase})"
        raise ContainerError(msg)

    def _validate_pod_status(self, pod_id: str, pod: Any) -> None:
        """Validate that a pod is in running state.

        Args:
            pod_id (str): The name of the pod.
            pod: The pod object from Kubernetes API.

        Raises:
            ContainerError: If pod is not running.

        """
        if pod.status.phase == "Running":
            return

        if pod.status.phase == "Pending":
            self._wait_for_pod_to_start(pod_id)
        else:
            msg = f"Pod {pod_id} is not running (status: {pod.status.phase})"
            raise ContainerError(msg)

    def _connect_to_existing_container(self, pod_id: str) -> None:
        """Connect to an existing Kubernetes pod.

        Args:
            pod_id (str): The name of the existing pod to connect to.

        Raises:
            ContainerError: If the pod cannot be found or accessed.

        """
        try:
            # Verify pod exists and get its status
            pod = self.client.read_namespaced_pod(name=pod_id, namespace=self.kube_namespace)
            self._log(f"Connected to existing pod {pod_id}")

            # Validate pod is running or can be made running
            self._validate_pod_status(pod_id, pod)

            # Store pod name for operations
            self.container = pod_id

            # Extract container name from the existing pod
            containers = pod.spec.containers
            if containers:
                self.container_name = containers[0].name
            else:
                self.container_name = "sandbox-container"  # fallback

        except ApiException as e:
            if e.status == KUBERNETES_POD_NOT_FOUND_ERROR_CODE:
                msg = f"Pod {pod_id} not found in namespace {self.kube_namespace}"
                self._log(msg, "error")
                raise ContainerError(msg) from e
            msg = f"Failed to access pod {pod_id}: {e}"
            self._log(msg, "error")
            raise ContainerError(msg) from e
        except Exception as e:
            msg = f"Failed to connect to pod {pod_id}: {e}"
            self._log(msg, "error")
            raise ContainerError(msg) from e

    def _ensure_directory_exists(self, path: str) -> None:
        """Ensure directory exists in Kubernetes pod."""
        mkdir_result = self.container_api.execute_command(
            self.container, f"mkdir -p '{path}'", container_name=self.container_name
        )
        if mkdir_result[0] != 0:
            stdout_output, stderr_output = mkdir_result[1]
            error_msg = stderr_output if stderr_output else stdout_output
            self._log(f"Failed to create directory {path}: {error_msg}", "error")

    def _ensure_ownership(self, paths: list[str]) -> None:
        """Ensure correct ownership of paths in Kubernetes pod."""
        # Check if we're running as root
        user_check = self.execute_command("id -u")
        is_root = user_check.stdout.strip() == "0"

        if not is_root:
            # For non-root pods, ensure directories are owned by current user
            self.execute_command(f"chown -R $(id -u):$(id -g) {' '.join(paths)}")

    def _process_non_stream_output(self, output: Any) -> tuple[str, str]:
        """Process non-streaming Kubernetes output."""
        if isinstance(output, tuple) and len(output) == 2:  # noqa: PLR2004
            stdout_data, stderr_data = output
            return str(stdout_data), str(stderr_data)
        return "", ""

    def _process_stream_output(self, output: Any) -> tuple[str, str]:
        """Process streaming Kubernetes output (not used but required by mixin)."""
        return self._process_non_stream_output(output)

    def _handle_timeout(self) -> None:
        """Handle Kubernetes timeout cleanup."""
        if self.container:
            try:
                self.close()
            except Exception as e:  # noqa: BLE001
                self._log(f"Error during timeout cleanup: {e}", "error")

    def open(self) -> None:
        """Open Kubernetes session."""
        super().open()

        if self.using_existing_container and self.config.container_id:
            # Connect to existing pod
            self._connect_to_existing_container(self.config.container_id)
        else:
            # Create new pod
            container_config = {"pod_manifest": self.pod_manifest}
            self.container = self.container_api.create_container(container_config)

        # Setup environment only for newly-created pods
        if not self.using_existing_container:
            self.environment_setup()

    def close(self) -> None:
        """Close Kubernetes session."""
        super().close()

        if self.container:
            # Only delete pod if we created it (not existing pod)
            if not self.using_existing_container:
                try:
                    self.container_api.stop_container(self.container)
                    self._log("Deleted pod")
                except Exception as e:  # noqa: BLE001
                    self._log(f"Error cleaning up pod: {e}", "error")
            else:
                self._log("Disconnected from existing pod")

            self.container = None

    def get_archive(self, path: str) -> tuple[bytes, dict]:
        """Get archive from Kubernetes pod."""
        if not self.container:
            raise NotOpenSessionError

        return self.container_api.copy_from_container(self.container, path, container_name=self.container_name)

    def execute_command(self, command: str, workdir: str | None = None) -> ConsoleOutput:
        """Override to pass container name for Kubernetes."""
        if not command:
            raise CommandEmptyError

        if not self.container:
            raise NotOpenSessionError

        if self.verbose:
            self.logger.info("Executing command: %s", command)

        exit_code, output = self.container_api.execute_command(
            self.container, command, workdir=workdir, stream=self.stream, container_name=self.container_name
        )

        stdout, stderr = self._process_output(output)

        if self.verbose:
            if stdout:
                self.logger.info("STDOUT: %s", stdout)
            if stderr:
                self.logger.error("STDERR: %s", stderr)

        return ConsoleOutput(exit_code=exit_code or 0, stdout=stdout, stderr=stderr)

    def copy_to_runtime(self, src: str, dest: str) -> None:
        """Override to pass container name for Kubernetes."""
        if not self.container:
            raise NotOpenSessionError

        # Validate source path exists and is accessible (same as mixin)
        src_path = Path(src)
        if not (src_path.exists() and (src_path.is_file() or src_path.is_dir())):
            msg = f"Source path {src} does not exist or is not accessible"
            raise FileNotFoundError(msg)

        if self.verbose:
            self.logger.info("Copying %s to %s", src, dest)

        dest_dir = str(Path(dest).parent)
        if dest_dir:
            self._ensure_directory_exists(dest_dir)

        self.container_api.copy_to_container(self.container, src, dest, container_name=self.container_name)
        self._ensure_ownership([dest])

    def copy_from_runtime(self, src: str, dest: str) -> None:
        """Override to pass container name for Kubernetes."""
        if not self.container:
            raise NotOpenSessionError

        if self.verbose:
            self.logger.info("Copying %s to %s", src, dest)

        bits, stat = self.container_api.copy_from_container(self.container, src, container_name=self.container_name)
        if stat.get("size", 0) == 0:
            msg = f"File {src} not found in container"
            raise FileNotFoundError(msg)

        self._extract_archive_safely(bits, dest)
