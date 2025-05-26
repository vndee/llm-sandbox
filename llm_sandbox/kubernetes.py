import io
import tarfile
import tempfile
import time
import uuid
from pathlib import Path
from typing import Any

from kubernetes import client as k8s_client
from kubernetes import config
from kubernetes.stream import stream

from llm_sandbox.base import ConsoleOutput, Session
from llm_sandbox.const import DefaultImage, SupportedLanguage
from llm_sandbox.exceptions import NotOpenSessionError


class SandboxKubernetesSession(Session):
    """Sandbox session for Kubernetes."""

    def __init__(
        self,
        client: k8s_client.CoreV1Api | None = None,
        image: str | None = None,
        lang: str = SupportedLanguage.PYTHON,
        keep_template: bool = False,
        verbose: bool = False,
        kube_namespace: str | None = "default",
        env_vars: dict | None = None,
        pod_manifest: dict | None = None,
        workdir: str | None = "/sandbox",
        **kwargs: dict[str, Any],  # noqa: ARG002
    ) -> None:
        """Create a new sandbox session.

        :param client: Kubernetes client, if not provided, a new client will be created
                    based on local Kubernetes context
        :param image: Docker image to use
        :param lang: Language of the code
        :param keep_template: if True, the image and container will not be removed
                    after the session ends
        :param verbose: if True, print messages
        :param kube_namespace: Kubernetes namespace to use, default is 'default'
        :param env_vars: Environment variables to use
        :param pod_manifest: Pod manifest to use (ignores other settings: `image`,
                            `kube_namespace` and `env_vars`). By default runs as root user
                            for maximum compatibility. Advanced users can override security
                            context in custom pod_manifest.
        """
        super().__init__(lang, verbose)

        if not image:
            image = DefaultImage.__dict__[lang.upper()]

        if not client:
            if self.verbose:
                self.logger.info("Using local Kubernetes context since client is not provided..")

            config.load_kube_config()
            self.client = k8s_client.CoreV1Api()
        else:
            self.client = client

        self.image = image
        self.kube_namespace = kube_namespace
        self.pod_name = f"sandbox-{lang.lower()}-{uuid.uuid4().hex}"
        self.keep_template = keep_template
        self.container = None
        self.env_vars = env_vars
        self.pod_manifest = pod_manifest or self._default_pod_manifest()
        self.workdir = workdir
        self._reconfigure_with_pod_manifest()

    def _default_pod_manifest(self) -> dict:
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
                        "image": self.image,
                        "tty": True,
                        "securityContext": {
                            "runAsUser": 0,  # Run as root for maximum compatibility
                            "runAsGroup": 0,
                        },
                    }
                ],
                "securityContext": {
                    "runAsUser": 0,  # Pod-level security context for root access
                    "runAsGroup": 0,
                },
            },
        }

        if self.env_vars:
            pod_manifest["spec"]["containers"][0]["env"] = [  # type: ignore[index]
                {"name": key, "value": value} for key, value in self.env_vars.items()
            ]
        return pod_manifest

    def _reconfigure_with_pod_manifest(self) -> None:
        self.pod_name = self.pod_manifest.get("metadata", {}).get("name", self.pod_name)
        self.kube_namespace = self.pod_manifest.get("metadata", {}).get(
            "namespace", self.kube_namespace
        )

    def open(self) -> None:
        """Open the sandbox session."""
        self._create_kubernetes_pod()
        self.environment_setup()

    def _ensure_ownership(self, folders: list[str]) -> None:
        # For Kubernetes, check if we're running as root to handle ownership
        # If running as non-root user, the directories should already have correct ownership
        user_check = self.execute_command("id -u")
        is_root = user_check.stdout.strip() == "0"

        if not is_root:
            # For non-root pods, ensure cache directory is owned by current user
            self.execute_commands(
                [
                    (f"chown -R $(id -u):$(id -g) {' '.join(folders)}", None),
                ]
            )

    def _create_kubernetes_pod(self) -> None:
        self.client.create_namespaced_pod(namespace=self.kube_namespace, body=self.pod_manifest)

        while True:
            pod = self.client.read_namespaced_pod(name=self.pod_name, namespace=self.kube_namespace)
            if pod.status.phase == "Running":
                break
            time.sleep(1)

        self.container = self.pod_name

    def close(self) -> None:
        """Close the sandbox session."""
        self._delete_kubernetes_pod()

    def _delete_kubernetes_pod(self) -> None:
        self.client.delete_namespaced_pod(
            name=self.pod_name,
            namespace=self.kube_namespace,
            body=k8s_client.V1DeleteOptions(),
        )

    def run(self, code: str, libraries: list | None = None) -> ConsoleOutput:
        """Run the code in the sandbox session."""
        if not self.container:
            raise NotOpenSessionError

        self.install(libraries)

        with tempfile.TemporaryDirectory() as directory_name:
            code_file = str(Path(directory_name) / f"code.{self.language_handler.file_extension}")
            code_dest_file = f"{self.workdir}/code.{self.language_handler.file_extension}"

            with Path.open(code_file, "w", encoding="utf-8") as f:
                f.write(code)

            self.copy_to_runtime(code_file, code_dest_file)

            commands = self.language_handler.get_execution_commands(code_dest_file)
            return self.execute_commands(commands, workdir=self.workdir)

    def copy_from_runtime(self, src: str, dest: str) -> None:  # noqa: PLR0912
        """Copy a file from the runtime."""
        if not self.container:
            raise NotOpenSessionError

        if self.verbose:
            self.logger.info("Copying %s:%s to %s..", self.container, src, dest)

        exec_command = ["tar", "cf", "-", src]
        resp = stream(
            self.client.connect_get_namespaced_pod_exec,
            self.container,
            self.kube_namespace,
            command=exec_command,
            stderr=True,
            stdin=False,
            stdout=True,
            tty=False,
            _preload_content=False,
        )

        # Collect the tar archive data
        tar_data = io.BytesIO()
        while resp.is_open():
            resp.update(timeout=1)
            if resp.peek_stdout():
                stdout_chunk = resp.read_stdout()
                if isinstance(stdout_chunk, str):
                    tar_data.write(stdout_chunk.encode())
                else:
                    tar_data.write(stdout_chunk)
            if resp.peek_stderr():
                self.logger.error(resp.read_stderr())

        # Extract the file content from the tar archive
        tar_data.seek(0)
        with tarfile.open(fileobj=tar_data, mode="r") as tar:
            # Filter to prevent extraction of unsafe paths
            safe_members = [
                member
                for member in tar.getmembers()
                if not (member.name.startswith("/") or ".." in member.name)
            ]

            # Find the file we want to extract
            target_member = None
            src_name = Path(src).name
            for member in safe_members:
                if member.isfile() and (
                    member.name == src_name or member.name.endswith(f"/{src_name}")
                ):
                    target_member = member
                    break

            if target_member:
                # Extract the file content and write to destination
                file_obj = tar.extractfile(target_member)
                if file_obj:
                    Path(dest).parent.mkdir(parents=True, exist_ok=True)
                    with Path(dest).open("wb") as f:
                        f.write(file_obj.read())
                else:
                    raise FileNotFoundError(src)
            else:
                raise FileNotFoundError(src)

    def copy_to_runtime(self, src: str, dest: str) -> None:
        """Copy a file to the runtime."""
        if not self.container:
            raise NotOpenSessionError

        start_time = time.time()
        if self.verbose:
            self.logger.info("Copying %s to %s:%s..", src, self.container, dest)

        dest_dir = str(Path(dest).parent)
        dest_file = str(Path(dest).name)

        if dest_dir:
            self.execute_command(f"mkdir -p {dest_dir}")

        with Path(src).open("rb") as f:
            tarstream = io.BytesIO()
            with tarfile.open(fileobj=tarstream, mode="w") as tar:
                tarinfo = tarfile.TarInfo(name=dest_file)
                tarinfo.size = Path(src).stat().st_size
                tar.addfile(tarinfo, f)
            tarstream.seek(0)

            exec_command = ["tar", "xvf", "-", "-C", dest_dir]
            resp = stream(
                self.client.connect_get_namespaced_pod_exec,
                self.container,
                self.kube_namespace,
                command=exec_command,
                stderr=True,
                stdin=True,
                stdout=True,
                tty=False,
                _preload_content=False,
            )
            while resp.is_open():
                resp.update(timeout=1)
                if resp.peek_stdout():
                    self.logger.info(resp.read_stdout())
                if resp.peek_stderr():
                    self.logger.error(resp.read_stderr())
                resp.write_stdin(tarstream.read(4096))
            resp.close()

        end_time = time.time()
        if self.verbose:
            self.logger.info(
                "Copied %s to %s:%s in %.2f seconds",
                src,
                self.container,
                dest,
                end_time - start_time,
            )

        self._ensure_ownership([dest_dir])

    def execute_command(
        self, command: str, workdir: str | None = None, *, disable_logging: bool = False
    ) -> ConsoleOutput:
        """Execute a command in the sandbox session."""
        if not self.container:
            raise NotOpenSessionError

        if self.verbose:
            self.logger.info("Executing command: %s", command)

        if workdir:
            exec_command = ["sh", "-c", f"cd {workdir} && {command}"]
        else:
            exec_command = ["/bin/sh", "-c", command]

        resp = stream(
            self.client.connect_get_namespaced_pod_exec,
            self.container,
            self.kube_namespace,
            command=exec_command,
            stderr=True,
            stdin=False,
            stdout=True,
            tty=False,
            _preload_content=False,
        )

        stdout_output = ""
        stderr_output = ""

        if self.verbose and not disable_logging:
            self.logger.info("Output:")

        while resp.is_open():
            resp.update(timeout=1)
            if resp.peek_stdout():
                chunk = resp.read_stdout()
                stdout_output += chunk
                if self.verbose and not disable_logging:
                    self.logger.info("Stdout: %s", chunk)

            if resp.peek_stderr():
                chunk = resp.read_stderr()
                stderr_output += chunk
                if self.verbose and not disable_logging:
                    self.logger.error("Stderr: %s", chunk)

        exit_code = resp.returncode

        return ConsoleOutput(
            exit_code=exit_code,
            stdout=stdout_output,
            stderr=stderr_output,
        )

    def get_archive(self, path: str) -> tuple[bytes, dict]:
        """Get archive of files from pod."""
        if not self.container:
            raise NotOpenSessionError

        if self.verbose:
            self.logger.info("Getting archive for path: %s", path)

        # First check if the path exists and get its stats
        stat_command = f"stat -c '%s %Y %n' {path} 2>/dev/null || echo 'NOT_FOUND'"
        stat_result = self.execute_command(stat_command, disable_logging=True)

        if stat_result.stdout.strip() == "NOT_FOUND" or stat_result.exit_code != 0:
            return b"", {}

        # Parse stat output (size, mtime, name)
        stat_parts = stat_result.stdout.strip().split(" ", 2)
        if len(stat_parts) >= 3:  # noqa:PLR2004
            file_size = int(stat_parts[0])
            mtime = int(stat_parts[1])
            file_name = stat_parts[2]
        else:
            file_size = 0
            mtime = 0
            file_name = path

        # Use base64 encoding to safely transfer binary data
        base64_command = f"tar cf - {path} | base64 -w 0"
        result = self.execute_command(base64_command, disable_logging=True)

        if result.exit_code:
            if self.verbose:
                self.logger.error(
                    "base64 tar command failed with exit code %d: %s",
                    result.exit_code,
                    result.stderr,
                )
            return b"", {}

        # Decode the base64 data back to binary
        try:
            import base64

            tar_data = base64.b64decode(result.stdout.strip())
        except Exception:
            if self.verbose:
                self.logger.exception("Failed to decode base64 data")
            return b"", {}

        # Create stat dict similar to Docker's format
        stat_dict = {
            "name": file_name,
            "size": file_size,
            "mtime": mtime,
            "mode": 0o644,  # Default file mode
            "linkTarget": "",
        }

        if self.verbose:
            self.logger.info("Retrieved archive for %s (%d bytes)", path, len(tar_data))

        return tar_data, stat_dict
