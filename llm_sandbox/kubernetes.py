import io
import tarfile
import tempfile
import time
import uuid
from pathlib import Path
from typing import Any

from kubernetes import client as k8s_client
from kubernetes import config
from kubernetes.client import CoreV1Api
from kubernetes.stream import stream

from llm_sandbox.base import ConsoleOutput, Session
from llm_sandbox.const import DefaultImage, SupportedLanguage
from llm_sandbox.exceptions import NotOpenSessionError
from llm_sandbox.security import SecurityPolicy


class SandboxKubernetesSession(Session):
    r"""Sandbox session implemented using Kubernetes Pods.

    This class provides a sandboxed environment for code execution by leveraging Kubernetes.
    It handles Pod creation and lifecycle based on a provided or default manifest,
    code execution, library installation, and file operations within the Kubernetes Pod.
    """

    def __init__(
        self,
        client: CoreV1Api | None = None,
        image: str | None = None,
        lang: str = SupportedLanguage.PYTHON,
        verbose: bool = False,
        kube_namespace: str | None = "default",
        env_vars: dict | None = None,
        pod_manifest: dict | None = None,
        workdir: str | None = "/sandbox",
        security_policy: SecurityPolicy | None = None,
        **kwargs: dict[str, Any],  # noqa: ARG002
    ) -> None:
        """
        Initializes a new sandbox session using a Kubernetes Pod for secure code execution.
        
        Creates or configures a Kubernetes Pod with the specified Docker image, language, namespace, environment variables, and security policy. If no image or client is provided, defaults are selected based on the language and local Kubernetes configuration. Allows advanced customization via a full Pod manifest.
        """
        super().__init__(
            lang=lang,
            verbose=verbose,
            image=image,
            workdir=workdir,
            security_policy=security_policy,
        )

        if not image:
            self.image = DefaultImage.__dict__[lang.upper()]

        if not client:
            if self.verbose:
                self.logger.info("Using local Kubernetes context since client is not provided..")

            config.load_kube_config()
            self.client = CoreV1Api()
        else:
            self.client = client

        self.container: str
        self.kube_namespace = kube_namespace
        self.pod_name = f"sandbox-{lang.lower()}-{uuid.uuid4().hex}"
        self.env_vars = env_vars
        self.pod_manifest = pod_manifest or self._default_pod_manifest()
        self._reconfigure_with_pod_manifest()

    def _default_pod_manifest(self) -> dict:
        r"""Generate a default Kubernetes Pod manifest.

        This manifest defines a simple Pod with a single container running the specified
        `self.image`. It includes basic labels and sets the security context to run as root
        for broad compatibility. Environment variables from `self.env_vars` are included.

        Returns:
            dict: A dictionary representing the Kubernetes Pod manifest.

        """
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
        r"""Reconfigure session attributes based on the provided or default pod_manifest.

        Ensures that `self.pod_name` and `self.kube_namespace` are consistent with the
        metadata specified in the `pod_manifest`.
        """
        self.pod_name = self.pod_manifest.get("metadata", {}).get("name", self.pod_name)
        self.kube_namespace = self.pod_manifest.get("metadata", {}).get("namespace", self.kube_namespace)

    def open(self) -> None:
        r"""Open the Kubernetes sandbox session.

        This method prepares the Kubernetes environment by:
        1. Creating a Kubernetes Pod based on `self.pod_manifest`.
        2. Waiting for the Pod to reach the "Running" phase.
        3. Setting `self.container` to the Pod name.
        4. Calling `self.environment_setup()` to prepare language-specific settings within the Pod.
        """
        self._create_kubernetes_pod()
        self.environment_setup()

    def _ensure_ownership(self, folders: list[str]) -> None:
        r"""Ensure correct file ownership for specified folders within the Kubernetes Pod.

        If the Pod is detected to be running as a non-root user, this method attempts
        to change the ownership of the listed folders to that user and group using `chown`.
        This is primarily for ensuring writable cache or venv directories.

        Args:
            folders (list[str]): A list of absolute paths to folders within the Pod.

        """
        # For Kubernetes, check if we're running as root to handle ownership
        # If running as non-root user, the directories should already have correct ownership
        user_check = self.execute_command("id -u")
        is_root = user_check.stdout.strip() == "0"

        if not is_root:
            # For non-root pods, ensure cache directory is owned by current user
            self.execute_commands([
                (f"chown -R $(id -u):$(id -g) {' '.join(folders)}", None),
            ])

    def _create_kubernetes_pod(self) -> None:
        r"""Create the Kubernetes Pod and wait for it to become ready.

        Uses the Kubernetes client to create a namespaced Pod defined by `self.pod_manifest`.
        It then polls the Pod's status until its phase is "Running".
        """
        self.client.create_namespaced_pod(namespace=self.kube_namespace, body=self.pod_manifest)

        while True:
            pod = self.client.read_namespaced_pod(name=self.pod_name, namespace=self.kube_namespace)
            if pod.status.phase == "Running":
                break
            time.sleep(1)

        self.container = self.pod_name

    def close(self) -> None:
        r"""Close the Kubernetes sandbox session.

        This method cleans up Kubernetes resources by deleting the created Pod.
        """
        self._delete_kubernetes_pod()

    def _delete_kubernetes_pod(self) -> None:
        r"""Delete the Kubernetes Pod associated with this session.

        Uses the Kubernetes client to delete the namespaced Pod identified by
        `self.pod_name` and `self.kube_namespace`.
        """
        self.client.delete_namespaced_pod(
            name=self.pod_name,
            namespace=self.kube_namespace,
            body=k8s_client.V1DeleteOptions(),
        )

    def run(self, code: str, libraries: list | None = None) -> ConsoleOutput:
        r"""Run the provided code within the Kubernetes Pod.

        This method performs the following steps:
        1. Ensures the session is open (Pod is running).
        2. Installs any specified `libraries` using the language-specific handler.
        3. Writes the `code` to a temporary file on the host system.
        4. Copies this temporary file into the Pod at the configured `workdir`.
        5. Retrieves execution commands from the language handler.
        6. Executes these commands in the Pod using `execute_commands`.

        Args:
            code (str): The code string to execute.
            libraries (list | None, optional): A list of libraries to install before running the code.
                                            Defaults to None.

        Returns:
            ConsoleOutput: An object containing the stdout, stderr, and exit code from the code execution.

        Raises:
            NotOpenSessionError: If the session (Pod) is not currently running.
            CommandFailedError: If any of the execution commands fail.

        """
        if not self.container:
            raise NotOpenSessionError

        self.install(libraries)

        with tempfile.TemporaryDirectory() as directory_name:
            code_file = str(Path(directory_name) / f"code.{self.language_handler.file_extension}")
            code_dest_file = f"{self.workdir}/code.{self.language_handler.file_extension}"

            with Path(code_file).open("w", encoding="utf-8") as f:
                f.write(code)

            self.copy_to_runtime(code_file, code_dest_file)

            commands = self.language_handler.get_execution_commands(code_dest_file)
            return self.execute_commands(commands, workdir=self.workdir)  # type: ignore[arg-type]

    def copy_from_runtime(self, src: str, dest: str) -> None:  # noqa: PLR0912
        r"""Copy a file or directory from the Kubernetes Pod to the local host filesystem.

        This method uses `kubectl exec` (via the Kubernetes API stream) to create a tar archive
        of the `src` path within the Pod, streams it to the host, and then extracts the
        target file to the `dest` path. Basic security filtering is applied to prevent path
        traversal attacks during extraction.

        Args:
            src (str): The absolute path to the source file or directory within the Pod.
            dest (str): The path on the host filesystem where the content should be copied.
                        The parent directory of `dest` will be created if it doesn't exist.

        Raises:
            NotOpenSessionError: If the session (Pod) is not currently running.
            FileNotFoundError: If the `src` path does not exist or is not found in the archive from the Pod.

        """
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
                member for member in tar.getmembers() if not (member.name.startswith("/") or ".." in member.name)
            ]

            # Find the file we want to extract
            target_member = None
            src_name = Path(src).name
            for member in safe_members:
                if member.isfile() and (member.name == src_name or member.name.endswith(f"/{src_name}")):
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
        r"""Copy a file or directory from the local host filesystem to the Kubernetes Pod.

        This method creates a tar archive of the `src` path on the host, then uses `kubectl exec`
        (via the Kubernetes API stream) to stream this archive into the Pod and extract it at
        the `dest_dir` (parent directory of `dest`). The destination directory is created if it
        doesn't exist. File ownership is ensured after copying if a non-root user is detected.

        Args:
            src (str): The path to the source file or directory on the host system.
            dest (str): The absolute destination path within the Pod.

        Raises:
            NotOpenSessionError: If the session (Pod) is not currently running.

        """
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
        r"""Execute an arbitrary command directly within the Kubernetes Pod.

        This method uses `kubectl exec` (via the Kubernetes API stream) to run the command.
        It captures stdout, stderr, and the exit code of the command.

        Args:
            command (str): The command string to execute (e.g., "ls -l", "pip install <package>").
            workdir (str | None, optional): The working directory within the Pod where the command
                                        should be executed. If provided, the command is wrapped
                                        with `cd <workdir> && <command>`. Defaults to None.
            disable_logging (bool, optional): If True, suppress verbose logging for this specific command's
                                            output. Useful for internal status checks. Defaults to False.

        Returns:
            ConsoleOutput: An object containing the stdout, stderr, and exit code of the command.

        Raises:
            NotOpenSessionError: If the session (Pod) is not currently running.

        """
        if not self.container:
            raise NotOpenSessionError

        if self.verbose:
            self.logger.info("Executing command: %s", command)

        exec_command = ["sh", "-c", f"cd {workdir} && {command}"] if workdir else ["/bin/sh", "-c", command]

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
        r"""Retrieve a file or directory from the Kubernetes Pod as a tar archive.

        This method first uses `execute_command` to run `stat` on the `path` within the Pod to get
        its metadata (size, mtime, name). Then, it runs `tar cf - <path> | base64 -w 0` to get a
        base64-encoded tar stream of the path's content. This stream is decoded, and a stat-like
        dictionary is constructed to mimic Docker's `get_archive` behavior.

        Args:
            path (str): The absolute path to the file or directory within the Pod.

        Returns:
            tuple[bytes, dict]: A tuple where the first element is the raw bytes of the tar archive,
                                and the second element is a dictionary containing stat-like metadata
                                (name, size, mtime, mode, linkTarget). Returns `(b"", {})` if the
                                path is not found or an error occurs.

        Raises:
            NotOpenSessionError: If the session (Pod) is not currently running.

        """
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
