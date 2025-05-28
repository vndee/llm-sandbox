# ruff: noqa: E501

import io
import tarfile
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING, Any

from podman import PodmanClient
from podman.domain.images import Image
from podman.errors import ImageNotFound

from llm_sandbox.base import ConsoleOutput, Session
from llm_sandbox.const import DefaultImage, SupportedLanguage
from llm_sandbox.exceptions import (
    CommandEmptyError,
    ExtraArgumentsError,
    ImageNotFoundError,
    ImagePullError,
    NotOpenSessionError,
)
from llm_sandbox.security import SecurityPolicy

if TYPE_CHECKING:
    from podman.domain.containers import Container


class SandboxPodmanSession(Session):
    r"""Sandbox session implemented using Podman containers.

    This class provides a sandboxed environment for code execution by leveraging Podman.
    It handles Podman image management (pulling, building from Dockerfile/Containerfile),
    container creation and lifecycle, code execution, library installation, and file
    operations within the Podman container. It is designed to be a drop-in replacement
    for the Docker-based session where Podman is the preferred container runtime.
    """

    def __init__(
        self,
        client: PodmanClient | None = None,
        image: str | None = None,
        dockerfile: str | None = None,  # Can be a Containerfile
        lang: str = SupportedLanguage.PYTHON,
        keep_template: bool = False,
        commit_container: bool = False,
        verbose: bool = False,
        mounts: list | None = None,  # Type hint for mounts can be more specific if known, e.g., list[dict]
        stream: bool = True,
        runtime_configs: dict | None = None,
        workdir: str | None = "/sandbox",
        security_policy: SecurityPolicy | None = None,
        **kwargs: dict[str, Any],  # noqa: ARG002
    ) -> None:
        r"""Initialize a new Podman-based sandbox session.

        Args:
            client (PodmanClient | None, optional): An existing Podman client instance.
                If None, a new client will be created based on the local Podman environment
                (e.g., Podman socket). Defaults to None.
            image (str | None, optional): The name of the Podman image to use (e.g., "vndee/sandbox-python-311-bullseye").
                If None and `dockerfile` is also None, a default image for the specified `lang` is used.
                Defaults to None.
            dockerfile (str | None, optional): The path to a Dockerfile or Containerfile to build an image from.
                Cannot be used if `image` is also provided. Defaults to None.
            lang (str, optional): The programming language of the code to be run (e.g., "python", "java").
                Determines default image and language-specific handlers. Defaults to SupportedLanguage.PYTHON.
            keep_template (bool, optional): If True, the Podman image (built or pulled)
                will not be removed after the session ends. Defaults to False.
            commit_container (bool, optional): If True, the Podman container's state will be committed
                to a new image after the session ends. Defaults to False.
            verbose (bool, optional): If True, print detailed log messages. Defaults to False.
            mounts (list | None, optional): A list of mount configurations for the container.
                The exact structure depends on Podman client library expectations (often list of strings or dicts).
                Defaults to None.
            stream (bool, optional): If True, the output from `execute_command` will be streamed.
                Note: Enabling this option might affect how exit codes are retrieved for commands.
                Defaults to True.
            runtime_configs (dict | None, optional): Additional configurations for the container runtime,
                such as resource limits (e.g., `cpu_count`, `mem_limit`) or user (`user="1000:1000"`).
                By default, containers run as the root user for maximum compatibility.
                Defaults to None.
            workdir (str | None, optional): The working directory inside the container.
                Defaults to "/sandbox". Consider using "/tmp/sandbox" when running as a non-root user.
            security_policy (SecurityPolicy | None, optional): The security policy to use for the session.
                Defaults to None.
            **kwargs: Catches unused keyword arguments passed from `create_session`.

        Raises:
            ExtraArgumentsError: If both `image` and `dockerfile` are provided.
            ImagePullError: If pulling the specified Podman image fails.
            ImageNotFoundError: If the specified image is not found and cannot be pulled or built.

        """
        super().__init__(
            lang=lang,
            verbose=verbose,
            image=image,
            keep_template=keep_template,
            workdir=workdir,
            security_policy=security_policy,
        )
        self.dockerfile = dockerfile
        if self.image and self.dockerfile:
            msg = "Only one of `image` or `dockerfile` can be provided"
            raise ExtraArgumentsError(msg)

        if not self.image and not self.dockerfile:
            self.image = DefaultImage.__dict__[lang.upper()]

        self.client: PodmanClient
        if not client:
            if self.verbose:
                self.logger.info("Using local Podman context since client is not provided..")

            self.client = PodmanClient.from_env()
        else:
            self.client = client

        self.docker_path: str
        self.docker_image: Image
        self.commit_container: bool = commit_container
        self.is_create_template: bool = False
        self.mounts: list | None = mounts
        self.stream: bool = stream
        self.runtime_configs: dict | None = runtime_configs
        self.container: Container | None = None

    def _ensure_ownership(self, folders: list[str]) -> None:
        r"""Ensure correct file ownership for specified folders within the Podman container.

        This is particularly important when the container is configured to run as a non-root user.
        It changes the ownership of the listed folders to the user specified in `runtime_configs`.

        Args:
            folders (list[str]): A list of absolute paths to folders within the container.

        """
        current_user = self.runtime_configs.get("user") if self.runtime_configs else None
        if current_user and current_user != "root" and self.container:
            self.container.exec_run(f"chown -R {current_user} {' '.join(folders)}", user="root")

    def open(self) -> None:
        r"""Open the Podman sandbox session.

        This method prepares the Podman environment by:
        1. Building an image from a Dockerfile/Containerfile if `dockerfile` is provided.
        2. Pulling an image from a registry if `image` is specified and not found locally.
        3. Creating and starting a Podman container from the prepared image with specified
            configurations (mounts, runtime_configs, user).
        4. Calls `self.environment_setup()` to prepare language-specific settings.

        Raises:
            ImagePullError: If pulling the specified Podman image fails.
            ImageNotFoundError: If the specified image is not found and cannot be pulled or built,
                                or if `client.images.pull` returns an unexpected type.

        """
        warning_str = (
            "Since the `keep_template` flag is set to True, the Podman image will "
            "not be removed after the session ends and remains for future use."
        )

        if self.dockerfile:
            self.docker_path = str(Path(self.dockerfile).parent)
            if self.verbose:
                f_str = f"Building Podman image from {self.dockerfile}"
                f_str = f"{f_str}\n{warning_str}" if self.keep_template else f_str
                self.logger.info(f_str)

            self.docker_image, _ = self.client.images.build(
                path=self.docker_path,
                dockerfile=Path(self.dockerfile).name,
                tag=f"sandbox-{self.lang.lower()}-{Path(self.docker_path).name}",
            )
            self.is_create_template = True
        elif isinstance(self.image, str):
            try:
                self.docker_image = self.client.images.get(self.image)
                if self.verbose:
                    self.logger.info("Using image %s", self.docker_image.tags[-1])
            except ImageNotFound:
                if self.verbose:
                    self.logger.info("Image %s not found locally. Attempting to pull...", self.image)

                try:
                    pulled_image = self.client.images.pull(self.image)
                    if isinstance(pulled_image, Image):
                        self.docker_image = pulled_image
                    elif isinstance(pulled_image, list):
                        self.docker_image = pulled_image[0]
                    else:
                        raise ImageNotFoundError(self.image)  # noqa: TRY301

                    if self.verbose:
                        self.logger.info("Successfully pulled image %s", self.docker_image.tags[-1])
                    self.is_create_template = True
                except ImageNotFoundError:
                    raise
                except Exception as e:
                    raise ImagePullError(self.image, str(e)) from e

        self.container = self.client.containers.create(
            image=self.docker_image,
            tty=True,
            mounts=self.mounts or [],
            user=self.runtime_configs.get("user", "root") if self.runtime_configs else "root",
            **{k: v for k, v in self.runtime_configs.items() if k != "user"} if self.runtime_configs else {},
        )
        self.container.start()

        self.environment_setup()

    def close(self) -> None:
        r"""Close the Podman sandbox session.

        This method cleans up Podman resources by:
        1. Committing the container to a new image if `commit_container` is True.
        2. Stopping and removing the running Podman container.
        3. Removing the Podman image if `is_create_template` is True (image was built or pulled
            during this session), `keep_template` is False, and the image is not in use by
            other containers.
        """
        if self.container:
            if self.commit_container and isinstance(self.image, Image):
                if self.image.tags:
                    full_tag = self.image.tags[-1]
                    if ":" in full_tag:
                        repository, tag = full_tag.rsplit(":", 1)
                    else:
                        repository = full_tag
                        tag = "latest"
                try:
                    # Commit the container with repository and tag
                    self.container.commit(repository=repository, tag=tag)
                    if self.verbose:
                        self.logger.info("Committed container as image %s:%s", repository, tag)
                except Exception:
                    if self.verbose:
                        self.logger.exception("Failed to commit container")
                    raise

            self.container.stop()
            self.container.wait()
            self.container.remove(force=True)

        if self.is_create_template and not self.keep_template:
            # check if the image is used by any other container
            containers = self.client.containers.list(all=True)
            image_id = self.docker_image.id
            image_in_use = any(container.image.id == image_id for container in containers)

            if not image_in_use:
                self.docker_image.remove(force=True)
            elif self.verbose:
                self.logger.info(
                    "Image %s is in use by other containers. Skipping removal..",
                    self.docker_image.tags[-1],
                )

    def run(self, code: str, libraries: list | None = None) -> ConsoleOutput:
        r"""Run the provided code within the Podman sandbox session.

        This method performs the following steps:
        1. Ensures the session is open (container is running).
        2. Installs any specified `libraries` using the language-specific handler.
        3. Writes the `code` to a temporary file on the host.
        4. Copies this temporary file into the container at the configured `workdir`.
        5. Retrieves execution commands from the language handler.
        6. Executes these commands in the container using `execute_commands`.

        Args:
            code (str): The code string to execute.
            libraries (list | None, optional): A list of libraries to install before running the code.
                                            Defaults to None.

        Returns:
            ConsoleOutput: An object containing the stdout, stderr, and exit code from the code execution.

        Raises:
            NotOpenSessionError: If the session (container) is not currently open/running.
            CommandFailedError: If any of the execution commands fail.

        """
        if not self.container:
            raise NotOpenSessionError

        self.install(libraries)

        with tempfile.NamedTemporaryFile(delete=True, suffix=f".{self.language_handler.file_extension}") as code_file:
            code_file.write(code.encode("utf-8"))
            code_file.seek(0)

            code_dest_file = f"{self.workdir}/code.{self.language_handler.file_extension}"
            self.copy_to_runtime(code_file.name, code_dest_file)

            commands = self.language_handler.get_execution_commands(code_dest_file)
            return self.execute_commands(commands, workdir=self.workdir)  # type: ignore[arg-type]

    def copy_from_runtime(self, src: str, dest: str) -> None:
        r"""Copy a file or directory from the Podman container to the local host filesystem.

        The source path `src` is retrieved from the container as a tar archive, which is then
        extracted to the `dest` path on the host. Basic security filtering is applied to
        prevent path traversal attacks during extraction.

        Args:
            src (str): The absolute path to the source file or directory within the container.
            dest (str): The path on the host filesystem where the content should be copied.
                        If `dest` is a directory, the content will be placed inside it with its original name.
                        If `dest` is a file path, the extracted content will be named accordingly.

        Raises:
            NotOpenSessionError: If the session (container) is not currently open/running.
            FileNotFoundError: If the `src` path does not exist or is empty in the container.

        """
        if not self.container:
            raise NotOpenSessionError

        if self.verbose:
            self.logger.info("Copying %s:%s to %s..", self.container.short_id, src, dest)

        bits, stat = self.container.get_archive(src)
        if stat["size"] == 0:
            msg = f"File {src} not found in the container"
            self.logger.error(msg)
            raise FileNotFoundError(msg)

        tarstream = io.BytesIO(b"".join(bits))
        with tarfile.open(fileobj=tarstream, mode="r") as tar:
            # Filter to prevent extraction of unsafe paths
            safe_members = [
                member for member in tar.getmembers() if not (member.name.startswith("/") or ".." in member.name)
            ]

            # Rename the members to match the destination filename
            dest_name = Path(dest).name
            for member in safe_members:
                if member.isfile():
                    member.name = dest_name

            tar.extractall(str(Path(dest).parent), members=safe_members)  # noqa: S202

    def copy_to_runtime(self, src: str, dest: str) -> None:
        """Copy a file to the runtime."""
        if not self.container:
            raise NotOpenSessionError

        is_created_dir = False
        directory = Path(dest).parent
        if directory and self.container.exec_run(f"test -d {directory}")[0] != 0:
            self.container.exec_run(f"mkdir -p {directory}")
            is_created_dir = True

        if self.verbose:
            if is_created_dir:
                self.logger.info("Creating directory %s:%s", self.container.short_id, directory)
            self.logger.info("Copying %s to %s:%s..", src, self.container.short_id, dest)

        tarstream = io.BytesIO()
        with tarfile.open(fileobj=tarstream, mode="w") as tar:
            tar.add(src, arcname=Path(dest).name)

        tarstream.seek(0)
        self.container.put_archive(str(Path(dest).parent), tarstream.getvalue())

        # Change ownership to current user if running as non-root
        # This is sufficient because file owners can read/write their own files
        current_user = self.runtime_configs.get("user") if self.runtime_configs else None
        if current_user and current_user != "root":
            self.container.exec_run(f"chown {current_user} {dest}", user="root")
            if directory:
                self.container.exec_run(f"chown {current_user} {directory}", user="root")

    def execute_command(  # noqa: PLR0912, PLR0915
        self, command: str, workdir: str | None = None
    ) -> ConsoleOutput:
        r"""Execute an arbitrary command directly within the Podman container.

        This method uses Podman's `exec_run` to execute the command. It handles both
        streamed and non-streamed output based on the `self.stream` attribute, and decodes
        output chunks appropriately.

        Args:
            command (str): The command string to execute (e.g., "ls -l", "pip install <package>").
            workdir (str | None, optional): The working directory within the container where
                                        the command should be executed. If None, the container's
                                        default working directory is used. Defaults to None.

        Returns:
            ConsoleOutput: An object containing the stdout, stderr, and exit code of the command.

        Raises:
            CommandEmptyError: If the provided `command` string is empty.
            NotOpenSessionError: If the session (container) is not currently open/running.

        """
        if not command:
            raise CommandEmptyError

        if not self.container:
            raise NotOpenSessionError

        if self.verbose:
            self.logger.info("Executing command: %s", command)

        if workdir:
            result = self.container.exec_run(
                command,
                stream=self.stream,
                tty=False,
                workdir=workdir,
                stderr=True,
                stdout=True,
                demux=True,
            )
        else:
            result = self.container.exec_run(
                command,
                stream=self.stream,
                tty=False,
                stderr=True,
                stdout=True,
                demux=True,
            )

        exit_code, output = result

        stdout_output = ""
        stderr_output = ""

        if self.verbose:
            self.logger.info("Output:")

        if not self.stream:
            # When not streaming and demux=True, output is a tuple (stdout, stderr)
            if output:
                stdout_data, stderr_data = output

                if stdout_data:
                    if isinstance(stdout_data, (tuple, list)):
                        stdout_output = b"".join(stdout_data).decode("utf-8")
                    else:
                        stdout_output = stdout_data.decode("utf-8")

                    if self.verbose:
                        self.logger.info(stdout_output)

                if stderr_data:
                    if isinstance(stderr_data, (tuple, list)):
                        stderr_output = b"".join(stderr_data).decode("utf-8")
                    else:
                        stderr_output = stderr_data.decode("utf-8")

                    if self.verbose:
                        self.logger.error(stderr_output)
        else:
            # When streaming and demux=True, we get a generator of (stdout, stderr)
            for stdout_chunk, stderr_chunk in output:
                if stdout_chunk:
                    if isinstance(stdout_chunk, (tuple, list)):
                        chunk_str = b"".join(stdout_chunk).decode("utf-8")
                    elif isinstance(stdout_chunk, bytes):
                        chunk_str = stdout_chunk.decode("utf-8")
                    else:
                        chunk_str = str(stdout_chunk)

                    stdout_output += chunk_str
                    if self.verbose:
                        self.logger.info(chunk_str)

                if stderr_chunk:
                    if isinstance(stderr_chunk, (tuple, list)):
                        chunk_str = b"".join(stderr_chunk).decode("utf-8")
                    elif isinstance(stderr_chunk, bytes):
                        chunk_str = stderr_chunk.decode("utf-8")
                    else:
                        chunk_str = str(stderr_chunk)

                    stderr_output += chunk_str
                    if self.verbose:
                        self.logger.error(chunk_str)

        return ConsoleOutput(
            exit_code=exit_code or 0,
            stdout=stdout_output,
            stderr=stderr_output,
        )

    def get_archive(self, path: str) -> tuple[bytes, dict]:
        r"""Retrieve a file or directory from the Podman container as a tar archive.

        This method uses Podman's `get_archive` to fetch the content at the specified `path`.

        Args:
            path (str): The absolute path to the file or directory within the container.

        Returns:
            tuple[bytes, dict]: A tuple where the first element is the raw bytes of the
                                tar archive, and the second element is a dictionary containing
                                archive metadata (stat info).

        Raises:
            NotOpenSessionError: If the session (container) is not currently open/running.

        """
        if not self.container:
            raise NotOpenSessionError

        data, stat = self.container.get_archive(path)
        return b"".join(data), stat
