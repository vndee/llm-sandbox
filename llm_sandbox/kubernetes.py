import io
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
from llm_sandbox.exceptions import ContainerError, NotOpenSessionError
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

        exec_command = [SH_SHELL, "-c", f"cd {workdir} && {command}"] if workdir else [SH_SHELL, "-c", command]

        resp = stream(
            self.client.connect_get_namespaced_pod_exec,
            container,
            self.namespace,
            command=exec_command,
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

        exit_code = resp.returncode or 0
        return exit_code, (stdout_output, stderr_output)

    def copy_to_container(self, container: Any, src: str, dest: str) -> None:
        """Copy file to Kubernetes pod."""
        # Validate source path exists and is accessible
        src_path = Path(src)
        if not (src_path.exists() and (src_path.is_file() or src_path.is_dir())):
            msg = f"Source path {src} does not exist or is not accessible"
            raise FileNotFoundError(msg)

        dest_dir = str(Path(dest).parent)

        # Create destination directory
        if dest_dir:
            exec_command = ["mkdir", "-p", dest_dir]
            resp = stream(
                self.client.connect_get_namespaced_pod_exec,
                container,
                self.namespace,
                command=exec_command,
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

        exec_command = ["tar", "xf", "-", "-C", dest_dir]
        resp = stream(
            self.client.connect_get_namespaced_pod_exec,
            container,
            self.namespace,
            command=exec_command,
            stderr=True,
            stdin=True,
            stdout=True,
            tty=False,
            _preload_content=False,
        )

        # Send tar data
        while resp.is_open():
            resp.update(timeout=1)

            chunk = tarstream.read(4096)
            if chunk:
                resp.write_stdin(chunk)
            else:
                break

        resp.close()

    def copy_from_container(self, container: Any, src: str) -> tuple[bytes, dict]:
        """Copy file from Kubernetes pod."""
        # First check if the path exists and get its stats
        stat_command = f"stat -c '%s %Y %n' {src} 2>/dev/null || echo 'NOT_FOUND'"
        exec_command = [SH_SHELL, "-c", stat_command]

        resp = stream(
            self.client.connect_get_namespaced_pod_exec,
            container,
            self.namespace,
            command=exec_command,
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

        # Parse stat output
        stat_parts = stdout_output.strip().split(" ", 2)
        if len(stat_parts) >= 3:  # noqa: PLR2004
            file_size = int(stat_parts[0])
            mtime = int(stat_parts[1])
            file_name = stat_parts[2]
        else:
            file_size = 0
            mtime = 0
            file_name = src

        # Get tar archive with base64 encoding
        src_path = Path(src)
        parent_dir = src_path.parent
        target_name = src_path.name
        base64_command = f"tar -C {parent_dir} -cf - {target_name} | base64 -w 0"
        exec_command = [SH_SHELL, "-c", base64_command]

        resp = stream(
            self.client.connect_get_namespaced_pod_exec,
            container,
            self.namespace,
            command=exec_command,
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
        elif container_id:
            self.pod_name = container_id

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
        mkdir_result = self.container_api.execute_command(self.container, f"mkdir -p '{path}'")
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

        return self.container_api.copy_from_container(self.container, path)
