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

if TYPE_CHECKING:
    from podman.domain.containers import Container


class SandboxPodmanSession(Session):
    """Sandbox session for Podman."""

    def __init__(
        self,
        client: PodmanClient | None = None,
        image: str | None = None,
        dockerfile: str | None = None,
        lang: str = SupportedLanguage.PYTHON,
        keep_template: bool = False,
        commit_container: bool = False,
        verbose: bool = False,
        mounts: list | None = None,
        stream: bool = True,
        runtime_configs: dict | None = None,
        workdir: str | None = "/sandbox",
        **kwargs: dict[str, Any],  # noqa: ARG002
    ) -> None:
        """Create a new sandbox session.

        :param client: Podman client, if not provided, a new client will be created
                        based on local podman context
        :param image: Podman image to use
        :param dockerfile: Path to the Dockerfile, if image is not provided
        :param lang: Language of the code
        :param keep_template: if True, the image and container
                                will not be removed after the session ends
        :param commit_container: if True, the podman container will be commited
                                after the session ends
        :param verbose: if True, print messages
        :param mounts: List of mounts to be mounted to the container
        :param stream: if True, the output will be streamed (enabling this option
                        prevents obtaining an exit code of run command)
        :param runtime_configs: Additional configurations for the container,
                                i.e. resources limits (cpu_count, mem_limit),
                                user ("1000:1000"), etc. By default runs as root user
                                for maximum compatibility.
        :param workdir: Working directory inside the container. Defaults to "/sandbox".
                        Use "/tmp/sandbox" when running as non-root user.
        """
        super().__init__(lang, verbose)
        if image and dockerfile:
            raise ExtraArgumentsError

        if not image and not dockerfile:
            image = DefaultImage.__dict__[lang.upper()]

        self.client: PodmanClient | None = None
        if not client:
            if self.verbose:
                self.logger.info(
                    "Using local Podman context since client is not provided.."
                )

            self.client = PodmanClient.from_env()
        else:
            self.client = client

        self.image: Image | str = image
        self.dockerfile: str | None = dockerfile
        self.container: Container | None = None
        self.path = None
        self.keep_template = keep_template
        self.commit_container = commit_container
        self.is_create_template: bool = False
        self.verbose = verbose
        self.mounts = mounts
        self.stream = stream
        self.runtime_configs = runtime_configs
        self.workdir = workdir

    def _ensure_ownership(self, folders: list[str]) -> None:
        """For non-root users, ensure ownership of the resources."""
        current_user = (
            self.runtime_configs.get("user") if self.runtime_configs else None
        )
        if current_user and current_user != "root":
            self.container.exec_run(
                f"chown -R {current_user} {' '.join(folders)}", user="root"
            )

    def open(self) -> None:
        """Open the sandbox session."""
        warning_str = (
            "Since the `keep_template` flag is set to True, the Podman image will "
            "not be removed after the session ends and remains for future use."
        )

        if self.dockerfile:
            self.path = str(Path(self.dockerfile).parent)
            if self.verbose:
                f_str = f"Building Podman image from {self.dockerfile}"
                f_str = f"{f_str}\n{warning_str}" if self.keep_template else f_str
                self.logger.info(f_str)

            self.image, _ = self.client.images.build(
                path=self.path,
                dockerfile=Path.name(self.dockerfile),
                tag=f"sandbox-{self.lang.lower()}-{Path.name(self.path)}",
            )
            self.is_create_template = True

        if isinstance(self.image, str):
            try:
                self.image = self.client.images.get(self.image)
                if self.verbose:
                    self.logger.info("Using image %s", self.image.tags[-1])
            except ImageNotFound:
                if self.verbose:
                    self.logger.info(
                        "Image %s not found locally. Attempting to pull...", self.image
                    )

                try:
                    self.image = self.client.images.pull(self.image)
                    if self.verbose:
                        self.logger.info(
                            "Successfully pulled image %s", self.image.tags[-1]
                        )
                    self.is_create_template = True
                except Exception as e:
                    raise ImagePullError(self.image, str(e)) from e

        self.container = self.client.containers.create(
            image=self.image.id if isinstance(self.image, Image) else self.image,
            tty=True,
            mounts=self.mounts or [],
            user=self.runtime_configs.get("user", "root")
            if self.runtime_configs
            else "root",
            **{k: v for k, v in (self.runtime_configs or {}).items() if k != "user"},
        )
        self.container.start()

        self.environment_setup()

    def close(self) -> None:  # noqa: PLR0912
        """Close the sandbox session."""
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
                        self.logger.info(
                            "Committed container as image %s:%s", repository, tag
                        )
                except Exception:
                    if self.verbose:
                        self.logger.exception("Failed to commit container")
                    raise

            self.container.stop()
            self.container.wait()
            self.container.remove(force=True)
            self.container = None

        if self.is_create_template and not self.keep_template:
            # check if the image is used by any other container
            containers = self.client.containers.list(all=True)
            image_id = (
                self.image.id
                if isinstance(self.image, Image)
                else self.client.images.get(self.image).id
            )
            image_in_use = any(
                container.image.id == image_id for container in containers
            )

            if not image_in_use:
                if isinstance(self.image, str):
                    self.client.images.remove(self.image)
                elif isinstance(self.image, Image):
                    self.image.remove(force=True)
                else:
                    raise ImageNotFoundError(self.image)
            elif self.verbose:
                self.logger.info(
                    "Image %s is in use by other containers. Skipping removal..",
                    self.image.tags[-1],
                )

    def run(self, code: str, libraries: list | None = None) -> ConsoleOutput:
        """Run the code in the sandbox session."""
        if not self.container:
            raise NotOpenSessionError

        self.install(libraries)

        with tempfile.NamedTemporaryFile(
            delete=True, suffix=f".{self.language_handler.file_extension}"
        ) as code_file:
            code_file.write(code.encode("utf-8"))
            code_file.seek(0)

            code_dest_file = (
                f"{self.workdir}/code.{self.language_handler.file_extension}"
            )
            self.copy_to_runtime(code_file.name, code_dest_file)

            commands = self.language_handler.get_execution_commands(code_dest_file)
            return self.execute_commands(commands, workdir=self.workdir)

    def copy_from_runtime(self, src: str, dest: str) -> None:
        """Copy a file from the runtime to the local filesystem."""
        if not self.container:
            raise NotOpenSessionError

        if self.verbose:
            self.logger.info(
                "Copying %s:%s to %s..", self.container.short_id, src, dest
            )

        bits, stat = self.container.get_archive(src)
        if stat["size"] == 0:
            msg = f"File {src} not found in the container"
            self.logger.error(msg)
            raise FileNotFoundError(msg)

        tarstream = io.BytesIO(b"".join(bits))
        with tarfile.open(fileobj=tarstream, mode="r") as tar:
            # Filter to prevent extraction of unsafe paths
            safe_members = [
                member
                for member in tar.getmembers()
                if not (member.name.startswith("/") or ".." in member.name)
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
                self.logger.info(
                    "Creating directory %s:%s", self.container.short_id, directory
                )
            self.logger.info(
                "Copying %s to %s:%s..", src, self.container.short_id, dest
            )

        tarstream = io.BytesIO()
        with tarfile.open(fileobj=tarstream, mode="w") as tar:
            tar.add(src, arcname=Path(dest).name)

        tarstream.seek(0)
        self.container.put_archive(str(Path(dest).parent), tarstream)

        # Change ownership to current user if running as non-root
        # This is sufficient because file owners can read/write their own files
        current_user = (
            self.runtime_configs.get("user") if self.runtime_configs else None
        )
        if current_user and current_user != "root":
            self.container.exec_run(f"chown {current_user} {dest}", user="root")
            if directory:
                self.container.exec_run(
                    f"chown {current_user} {directory}", user="root"
                )

    def execute_command(  # noqa: PLR0912
        self, command: str, workdir: str | None = None
    ) -> ConsoleOutput:
        """Execute a command in the sandbox session."""
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
                    stdout_output = stdout_data.decode("utf-8")
                    if self.verbose:
                        self.logger.info(stdout_output)

                if stderr_data:
                    stderr_output = stderr_data.decode("utf-8")
                    if self.verbose:
                        self.logger.error(stderr_output)
        else:
            # When streaming and demux=True, we get a generator of (stdout, stderr)
            for stdout_chunk, stderr_chunk in output:
                if stdout_chunk:
                    chunk_str = stdout_chunk.decode("utf-8")
                    stdout_output += chunk_str
                    if self.verbose:
                        self.logger.info(chunk_str)

                if stderr_chunk:
                    chunk_str = stderr_chunk.decode("utf-8")
                    stderr_output += chunk_str
                    if self.verbose:
                        self.logger.error(chunk_str)

        return ConsoleOutput(
            exit_code=exit_code,
            stdout=stdout_output,
            stderr=stderr_output,
        )

    def get_archive(self, path: str) -> tuple[bytes, dict]:
        """Get archive of files from container."""
        if not self.container:
            raise NotOpenSessionError

        data, stat = self.container.get_archive(path)
        return b"".join(data), stat
