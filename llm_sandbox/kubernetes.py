import io
import tarfile
import tempfile
import time
import uuid
from pathlib import Path
from typing import Any, cast

from kubernetes import client as k8s_client
from kubernetes import config
from kubernetes.client import CoreV1Api
from kubernetes.stream import stream

from llm_sandbox.base import ConsoleOutput, Session
from llm_sandbox.const import DefaultImage, SupportedLanguage
from llm_sandbox.exceptions import NotOpenSessionError, SandboxTimeoutError
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
        default_timeout: float | None = None,
        execution_timeout: float | None = None,
        session_timeout: float | None = None,
        **kwargs: dict[str, Any],  # noqa: ARG002
    ) -> None:
        r"""Initialize a new Kubernetes-based sandbox session.

        Args:
            client (CoreV1Api | None, optional): An existing Kubernetes `CoreV1Api` client instance.
                If None, a new client will be created based on the local Kubernetes configuration
                (e.g., from `~/.kube/config`). Defaults to None.
            image (str | None, optional): The name of the Docker image to use for the Pod's container
                (e.g., "ghcr.io/vndee/sandbox-python-311-bullseye"). If None and `pod_manifest` is not provided with an
                image, a default image for the specified `lang` is used. Defaults to None.
            lang (str, optional): The programming language of the code to be run (e.g., "python", "java").
                Determines default image and language-specific handlers. Defaults to SupportedLanguage.PYTHON.
            verbose (bool, optional): If True, print detailed log messages. Defaults to False.
            kube_namespace (str | None, optional): The Kubernetes namespace where the Pod will be created.
                Defaults to "default". This is overridden if `pod_manifest` specifies a namespace.
            env_vars (dict | None, optional): A dictionary of environment variables to set in the sandbox
                container (e.g., `{"MY_VAR": "value"}`). Defaults to None. This is overridden if
                `pod_manifest` specifies environment variables.
            pod_manifest (dict | None, optional): A complete Kubernetes Pod manifest (as a dictionary).
                If provided, this manifest is used directly, overriding `image`, `kube_namespace`, `env_vars`,
                and default security contexts. Allows for advanced customization of the Pod.
                If None, a default Pod manifest is generated. Defaults to None.
            workdir (str | None, optional): The working directory inside the Pod's container.
                Defaults to "/sandbox".
            security_policy (SecurityPolicy | None, optional): The security policy to use for the session.
                Defaults to None.
            default_timeout (float | None, optional): The default timeout for the session.
                Defaults to None.
            execution_timeout (float | None, optional): The execution timeout for the session.
                Defaults to None.
            session_timeout (float | None, optional): The session timeout for the session.
                Defaults to None.
            **kwargs: Catches unused keyword arguments.

        """
        super().__init__(
            lang=lang,
            verbose=verbose,
            image=image,
            workdir=workdir,
            security_policy=security_policy,
            default_timeout=default_timeout,
            execution_timeout=execution_timeout,
            session_timeout=session_timeout,
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

        self.container: str | None = None
        self.kube_namespace = kube_namespace
        short_uuid = uuid.uuid4().hex[:8]
        self.pod_name = f"sandbox-{lang.lower()}-{short_uuid}"
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
        metadata specified in the `pod_manifest`. Always ensures pod name is unique.
        """
        # Always ensure pod name is unique, but keep it under 63 characters
        additional_uuid = uuid.uuid4().hex[:8]
        unique_pod_name = f"{self.pod_name}-{additional_uuid}"
        self.pod_name = unique_pod_name
        self.pod_manifest["metadata"]["name"] = unique_pod_name
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

        super().open()

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
        super().close()

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

    def run(self, code: str, libraries: list | None = None, timeout: float | None = None) -> ConsoleOutput:
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
            timeout (float | None, optional): The timeout for the code execution.
                                            Defaults to None.

        Returns:
            ConsoleOutput: An object containing the stdout, stderr, and exit code from the code execution.

        Raises:
            NotOpenSessionError: If the session (Pod) is not currently running.
            CommandFailedError: If any of the execution commands fail.

        """
        if not self.container:
            raise NotOpenSessionError

        self._check_session_timeout()
        actual_timeout = timeout or self.execution_timeout or self.default_timeout

        def _run_code() -> ConsoleOutput:
            self.install(libraries)

            with tempfile.TemporaryDirectory() as directory_name:
                code_file = str(Path(directory_name) / f"code.{self.language_handler.file_extension}")
                code_dest_file = f"{self.workdir}/code.{self.language_handler.file_extension}"

                with Path(code_file).open("w", encoding="utf-8") as f:
                    f.write(code)

                self.copy_to_runtime(code_file, code_dest_file)

                commands = self.language_handler.get_execution_commands(code_dest_file)
                # Type cast needed because get_execution_commands returns list[str]
                # but execute_commands expects list[str | tuple[str, str | None]]
                return self.execute_commands(
                    cast("list[str | tuple[str, str | None]]", commands),
                    workdir=self.workdir,
                    timeout=actual_timeout,
                )

        try:
            result = self._execute_with_timeout(_run_code, actual_timeout)
            return cast("ConsoleOutput", result)
        except SandboxTimeoutError:
            try:
                self._delete_kubernetes_pod()
                self.logger.warning("Pod deleted due to timeout")
            except Exception:
                if self.verbose:
                    self.logger.exception("Failed to delete Kubernetes Pod")
            raise

    def copy_from_runtime(self, src: str, dest: str) -> None:  # noqa: PLR0912, PLR0915
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
            # Enhanced safety: filter out dangerous paths
            safe_members = []
            for member in tar.getmembers():
                # Skip absolute paths and path traversal attempts
                if member.name.startswith("/") or ".." in member.name:
                    if self.verbose:
                        self.logger.warning("Skipping unsafe path: %s", member.name)
                    continue
                # Skip symlinks pointing outside extraction directory
                if member.issym() or member.islnk():
                    if self.verbose:
                        self.logger.warning("Skipping symlink: %s", member.name)
                    continue
                safe_members.append(member)

            if not safe_members and tar.getmembers():
                # All members were filtered - extract anyway to prevent data loss
                self.logger.warning("All tar members were filtered - extracting anyway")
                safe_members = tar.getmembers()
            elif not safe_members:
                msg = f"No content found in {src}"
                if self.verbose:
                    self.logger.error(msg)
                raise FileNotFoundError(msg)

            # Create destination directory if needed
            Path(dest).parent.mkdir(parents=True, exist_ok=True)

            # Handle both files and directories - account for tar path stripping
            src_path = Path(src)
            src_name = src_path.name
            # tar removes leading '/' so absolute paths become relative
            expected_prefix = str(src_path).lstrip("/")
            extracted_any = False

            if self.verbose:
                self.logger.info("Looking for tar members matching: %s or %s", src_name, expected_prefix)
                self.logger.info("Available members: %s", [m.name for m in safe_members[:5]])

            # Check if we're dealing with a single file
            single_file_members = [
                m
                for m in safe_members
                if m.isfile() and (m.name in (src_name, expected_prefix) or m.name.endswith(f"/{src_name}"))
            ]

            if len(single_file_members) == 1 and not any(m.isdir() for m in safe_members):
                # Single file extraction
                member = single_file_members[0]
                file_obj = tar.extractfile(member)
                if file_obj:
                    with Path(dest).open("wb") as f:
                        f.write(file_obj.read())
                    extracted_any = True
            else:
                # Directory extraction - look for members that match our expected path
                dest_dir = Path(dest)
                dest_dir.mkdir(exist_ok=True)

                for member in safe_members:
                    if member.isfile():
                        member_path = member.name
                        target_path = None

                        # Match against expected prefix (e.g., "sandbox/output/file.txt")
                        if member_path.startswith(f"{expected_prefix}/"):
                            # File inside the directory: sandbox/output/file.txt -> file.txt
                            rel_path = Path(member_path).relative_to(expected_prefix)
                            target_path = dest_dir / rel_path
                        elif member_path == expected_prefix:
                            # Directory itself
                            target_path = dest_dir / src_name
                        elif member_path.startswith(f"{src_name}/"):
                            # Fallback: direct basename match
                            rel_path = Path(member_path).relative_to(src_name)
                            target_path = dest_dir / rel_path

                        if target_path:
                            target_path.parent.mkdir(parents=True, exist_ok=True)
                            file_obj = tar.extractfile(member)
                            if file_obj:
                                with target_path.open("wb") as f:
                                    f.write(file_obj.read())
                                extracted_any = True

            if not extracted_any:
                raise FileNotFoundError(src)

    def copy_to_runtime(self, src: str, dest: str) -> None:  # noqa: PLR0912
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

        # Validate source path exists and is accessible
        src_path = Path(src)
        if not (src_path.exists() and (src_path.is_file() or src_path.is_dir())):
            msg = f"Source path {src} does not exist or is not accessible"
            if self.verbose:
                self.logger.error(msg)
            raise FileNotFoundError(msg)

        start_time = time.time()
        if self.verbose:
            self.logger.info("Copying %s to %s:%s..", src, self.container, dest)

        dest_dir = str(Path(dest).parent)

        if dest_dir:
            # Use quoted path to handle special characters
            result = self.execute_command(f"mkdir -p '{dest_dir}'")
            if result.exit_code != 0:
                self.logger.error("Failed to create directory %s: %s", dest_dir, result.stderr)

        # Create tar archive with validated source
        try:
            tarstream = io.BytesIO()
            with tarfile.open(fileobj=tarstream, mode="w") as tar:
                tar.add(src, arcname=Path(dest).name)
        except Exception:
            self.logger.exception("Failed to create tar archive for %s", src)
            raise
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

        # Use interleaved reading/writing to prevent deadlock and infinite loops
        stdin_completed = False
        while resp.is_open():
            resp.update(timeout=1)

            # Read any available output first to prevent buffer buildup
            if resp.peek_stdout():
                self.logger.info(resp.read_stdout())
            if resp.peek_stderr():
                self.logger.error(resp.read_stderr())

            # Write data in chunks, but stop once we've sent everything
            if not stdin_completed:
                chunk = tarstream.read(4096)
                if chunk:
                    resp.write_stdin(chunk)
                else:
                    # Mark stdin as completed to avoid infinite loop
                    stdin_completed = True
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

    def _execute_single_command(  # noqa: PLR0912
        self,
        command: str,
        workdir: str | None = None,
        timeout: float | None = None,
        *,
        disable_logging: bool = False,
    ) -> ConsoleOutput:
        r"""Execute a single command within the Kubernetes Pod with optional timeout.

        This is the base method that handles the actual command execution logic
        with timeout support for Kubernetes pods.

        Args:
            command (str): The command string to execute.
            workdir (str | None, optional): The working directory within the Pod.
            timeout (float | None, optional): The timeout for the command execution.
            disable_logging (bool, optional): If True, suppress verbose logging.

        Returns:
            ConsoleOutput: An object containing the stdout, stderr, and exit code.

        Raises:
            NotOpenSessionError: If the session (Pod) is not currently running.
            SandboxTimeoutError: If the command execution times out.

        """
        if not self.container:
            raise NotOpenSessionError

        if self.verbose and not disable_logging:
            self.logger.info("Executing command: %s", command)

        # Build the exec command
        if timeout:
            exec_command = ["timeout", str(int(timeout)), "sh", "-c"]
            if workdir:
                exec_command.append(f"cd {workdir} && {command}")
            else:
                exec_command.append(command)
        elif workdir:
            exec_command = ["sh", "-c", f"cd {workdir} && {command}"]
        else:
            exec_command = ["/bin/sh", "-c", command]

        # Create the stream connection
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

        # Execute and monitor the command
        start_time = time.time()
        stdout_output = ""
        stderr_output = ""

        if self.verbose and not disable_logging:
            self.logger.info("Output:")

        while resp.is_open():
            # Check timeout if specified
            if timeout and time.time() - start_time > timeout:
                resp.close()
                msg = f"Command execution timed out after {timeout} seconds"
                raise SandboxTimeoutError(msg)

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

        exit_code = resp.returncode or 0

        return ConsoleOutput(
            exit_code=exit_code,
            stdout=stdout_output,
            stderr=stderr_output,
        )

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
        return self._execute_single_command(
            command=command,
            workdir=workdir,
            disable_logging=disable_logging,
        )

    def execute_commands(
        self, commands: list[str | tuple[str, str | None]], workdir: str | None = None, timeout: float | None = None
    ) -> ConsoleOutput:
        """Execute a sequence of commands within the Kubernetes Pod.

        This overrides the base class method to add timeout support for Kubernetes.
        """
        if not commands:
            return ConsoleOutput(exit_code=0, stdout="", stderr="")

        result = ConsoleOutput(exit_code=0, stdout="", stderr="")

        for command in commands:
            if isinstance(command, tuple):
                cmd_str, cmd_workdir = command
            else:
                cmd_str = command
                cmd_workdir = workdir or self.workdir

            # Execute the command using the timeout-aware method
            result = self._execute_single_command(
                command=cmd_str,
                workdir=cmd_workdir,
                timeout=timeout,
                disable_logging=False,
            )

            # Stop if any command fails - maintaining original behavior
            if result.exit_code != 0:
                break

        return result

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
