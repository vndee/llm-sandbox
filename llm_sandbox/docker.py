import warnings
from pathlib import Path
from typing import TYPE_CHECKING, Any

import docker
from docker.errors import ImageNotFound, NotFound

from llm_sandbox.const import DefaultImage, SupportedLanguage
from llm_sandbox.core.config import SessionConfig
from llm_sandbox.core.session_base import BaseSession
from llm_sandbox.exceptions import ContainerError, ImagePullError, NotOpenSessionError
from llm_sandbox.security import SecurityPolicy

if TYPE_CHECKING:
    from docker.models.images import Image

DOCKER_CONFLICT_ERROR_CODES = {404, 409}


class DockerContainerAPI:
    """Docker implementation of the ContainerAPI protocol."""

    def __init__(self, client: docker.DockerClient, stream: bool = False) -> None:
        """Initialize Docker container API."""
        self.client = client
        self.stream = stream

    def create_container(self, config: Any) -> Any:
        """Create Docker container."""
        return self.client.containers.create(**config)

    def start_container(self, container: Any) -> None:
        """Start Docker container."""
        container.start()

    def stop_container(self, container: Any) -> None:
        """Stop Docker container."""
        container.stop()

    def execute_command(self, container: Any, command: str, **kwargs: Any) -> tuple[int, Any]:
        """Execute command in Docker container."""
        workdir = kwargs.get("workdir")
        exec_kwargs: dict[str, Any] = {
            "cmd": command,
            "stream": self.stream,
            "tty": False,
            "stderr": True,
            "stdout": True,
            "demux": True,
        }
        if workdir:
            exec_kwargs["workdir"] = workdir

        result = container.exec_run(**exec_kwargs)
        return result.exit_code or 0, result.output

    def copy_to_container(self, container: Any, src: str, dest: str) -> None:
        """Copy file to Docker container."""
        import io
        import tarfile

        tar_stream = io.BytesIO()
        with tarfile.open(fileobj=tar_stream, mode="w") as tar:
            tar.add(src, arcname=Path(dest).name)

        tar_stream.seek(0)
        container.put_archive(str(Path(dest).parent), tar_stream.getvalue())

    def copy_from_container(self, container: Any, src: str) -> tuple[bytes, dict]:
        """Copy file from Docker container."""
        data, stat = container.get_archive(src)
        return b"".join(data), stat


class SandboxDockerSession(BaseSession):
    r"""Sandbox session implemented using Docker containers.

    This class provides a sandboxed environment for code execution by leveraging Docker.
    It handles Docker image management (pulling, building from Dockerfile), container
    creation and lifecycle, code execution, library installation, and file operations
    within the Docker container.
    """

    def __init__(
        self,  # NOSONAR
        client: docker.DockerClient | None = None,
        image: str | None = None,
        dockerfile: str | None = None,
        lang: str = SupportedLanguage.PYTHON,
        keep_template: bool = False,
        commit_container: bool = False,
        verbose: bool = False,
        stream: bool = False,
        runtime_configs: dict | None = None,
        workdir: str = "/sandbox",
        security_policy: SecurityPolicy | None = None,
        default_timeout: float | None = None,
        execution_timeout: float | None = None,
        session_timeout: float | None = None,
        container_id: str | None = None,
        **kwargs: Any,
    ) -> None:
        r"""Initialize Docker session.

        Args:
            client (docker.DockerClient | None): The Docker client to use.
            image (str | None): The image to use.
            dockerfile (str | None): The Dockerfile to use.
            lang (str): The language to use.
            keep_template (bool): Whether to keep the template image.
            commit_container (bool): Whether to commit the container to a new image.
            verbose (bool): Whether to enable verbose output.
            stream (bool): Whether to stream the output.
            runtime_configs (dict | None): The runtime configurations to use.
            workdir (str): The working directory to use.
            security_policy (SecurityPolicy | None): The security policy to use.
            default_timeout (float | None): The default timeout to use.
            execution_timeout (float | None): The execution timeout to use.
            session_timeout (float | None): The session timeout to use.
            container_id (str | None): ID of existing container to connect to.
            **kwargs: Additional keyword arguments.

        Returns:
            None

        """
        config = SessionConfig(
            image=image,
            dockerfile=dockerfile,
            lang=SupportedLanguage(lang.upper()),
            verbose=verbose,
            workdir=workdir,
            runtime_configs=runtime_configs or {},
            security_policy=security_policy,
            default_timeout=default_timeout,
            execution_timeout=execution_timeout,
            session_timeout=session_timeout,
            container_id=container_id,
        )

        super().__init__(config=config, **kwargs)

        self.client: docker.DockerClient

        if not client:
            self._log("Using local Docker context since client is not provided.")
            self.client = docker.from_env()
        else:
            self.client = client

        self.container_api = DockerContainerAPI(self.client, stream)

        self.docker_image: Image
        self.keep_template: bool = keep_template
        self.commit_container: bool = commit_container
        self.is_create_template: bool = False
        self.stream: bool = stream

        if mounts := kwargs.get("mounts"):
            warnings.warn(
                "The 'mounts' parameter is deprecated and will be removed in a future version. "
                "Put the mounts in 'runtime_configs' instead.",
                DeprecationWarning,
                stacklevel=2,
            )
            existing_mounts = self.config.runtime_configs.setdefault("mounts", [])
            if isinstance(mounts, list):
                existing_mounts.extend(mounts)
            else:
                existing_mounts.append(mounts)

    def _ensure_directory_exists(self, path: str) -> None:
        r"""Ensure the directory exists.

        Args:
            path (str): The path to ensure exists.

        """
        mkdir_result = self.container_api.execute_command(self.container, f"mkdir -p '{path}'")
        if mkdir_result[0] != 0:
            stdout_output, stderr_output = self._process_non_stream_output(mkdir_result[1])
            error_msg = stderr_output if stderr_output else stdout_output
            self._log(f"Failed to create directory {path}: {error_msg}", "error")

    def _ensure_ownership(self, paths: list[str]) -> None:
        r"""Ensure ownership of the given paths.

        This method changes the ownership of specified paths to the current user

        Args:
            paths (list[str]): The paths to ensure ownership of.

        """
        current_user = self.config.runtime_configs.get("user") if self.config.runtime_configs else None
        if current_user and current_user != "root":
            self.container.exec_run(f"chown -R {current_user} {' '.join(paths)}", user="root")

    def _process_non_stream_output(self, output: Any) -> tuple[str, str]:
        """Process non-streaming Docker output."""
        stdout_output = ""
        stderr_output = ""

        if output:
            stdout_data, stderr_data = output
            if stdout_data:
                stdout_output = stdout_data.decode("utf-8")
            if stderr_data:
                stderr_output = stderr_data.decode("utf-8")

        return stdout_output, stderr_output

    def _process_stream_output(self, output: Any) -> tuple[str, str]:
        """Process streaming Docker output."""
        stdout_output, stderr_output = "", ""

        try:
            for stdout_chunk, stderr_chunk in output:
                if stdout_chunk:
                    stdout_output += (
                        stdout_chunk.decode("utf-8") if isinstance(stdout_chunk, bytes) else str(stdout_chunk)
                    )
                if stderr_chunk:
                    stderr_output += (
                        stderr_chunk.decode("utf-8") if isinstance(stderr_chunk, bytes) else str(stderr_chunk)
                    )
        except Exception as e:
            from llm_sandbox.exceptions import SandboxTimeoutError

            if isinstance(e, SandboxTimeoutError):
                raise

            self._log(f"Error processing stream output: {e}", "warning")

        return stdout_output, stderr_output

    def _handle_timeout(self) -> None:
        """Handle Docker timeout cleanup."""
        if self.using_existing_container:
            try:
                self.close()
            except Exception as e:  # noqa: BLE001
                self._log(f"Error during timeout cleanup: {e}", "error")

    def _connect_to_existing_container(self, container_id: str) -> None:
        """Connect to an existing Docker container.

        Args:
            container_id (str): The ID of the existing container to connect to.

        Raises:
            ContainerError: If the container cannot be found or accessed.

        """
        try:
            self.container = self.client.containers.get(container_id)
            self._log(f"Connected to existing container {container_id}")

            # Verify container is running
            if self.container.status != "running":
                self._log(f"Container {container_id} is not running, attempting to start...")
                self.container.start()
                self._log(f"Started container {container_id}")

        except NotFound as e:
            msg = f"Container {container_id} not found"
            self._log(msg, "error")
            raise ContainerError(msg) from e
        except Exception as e:
            msg = f"Failed to connect to container {container_id}: {e}"
            self._log(msg, "error")
            raise ContainerError(msg) from e

    def _prepare_image(self) -> None:
        """Prepare Docker image."""
        if self.config.dockerfile:
            self._build_image_from_dockerfile()
        elif self.config.image:
            self._get_or_pull_image()
        else:
            self.config.image = DefaultImage.__dict__[self.config.lang.upper()]
            self._get_or_pull_image()

    def _build_image_from_dockerfile(self) -> None:
        """Build image from Dockerfile."""
        docker_path = str(Path(self.config.dockerfile).parent) if self.config.dockerfile else None
        self._log(f"Building Docker image from {self.config.dockerfile}")

        self.docker_image, _ = self.client.images.build(
            path=docker_path,
            dockerfile=Path(self.config.dockerfile).name if self.config.dockerfile else None,
            tag=f"sandbox-{self.config.lang.lower()}-{Path(docker_path).name if docker_path else None}",
        )
        self.is_create_template = True

    def _get_or_pull_image(self) -> None:
        """Get local image or pull from registry."""
        try:
            self.docker_image = self.client.images.get(self.config.image)
            self._log(f"Using local image {self.docker_image.tags[-1] if self.docker_image.tags else 'untagged'}")
        except ImageNotFound:
            self._log(f"Image {self.config.image} not found locally. Pulling...")
            try:
                self.docker_image = self.client.images.pull(self.config.image)
                self._log(
                    f"Successfully pulled image {self.docker_image.tags[-1] if self.docker_image.tags else 'untagged'}"
                )
                self.is_create_template = True
            except Exception as e:
                raise ImagePullError(self.config.image or "", str(e)) from e

    def open(self) -> None:
        r"""Open Docker session.

        This method prepares the Docker environment for code execution by:
        - Building or pulling the Docker image (if not using existing container)
        - Creating a container or connecting to existing one
        - Setting up the environment (if not using existing container)

        Raises:
            ImagePullError: If the image cannot be pulled.
            ImageNotFoundError: If the image cannot be found.
            ContainerError: If existing container cannot be found or accessed.

        """
        super().open()

        if self.using_existing_container and self.config.container_id:
            # Connect to existing container
            self._connect_to_existing_container(self.config.container_id)
        else:
            # Create new container
            self._prepare_image()

            container_config = {"image": self.docker_image, "detach": True, "tty": True, "user": "root"}
            container_config.update(self.config.runtime_configs)

            self.container = self.container_api.create_container(container_config)
            self.container_api.start_container(self.container)

        # Setup environment (skipped for existing containers)
        self.environment_setup()

    def close(self) -> None:
        r"""Close the Docker sandbox session.

        This method cleans up Docker resources by:
        1. Committing the container to a new image if `commit_container` is True.
        2. Stopping and removing the running Docker container (only if we created it).
        3. Removing the Docker image if `is_create_template` is True (image was built or pulled
            during this session), `keep_template` is False, and the image is not in use by
            other containers.

        Note: When using existing containers, we only disconnect but don't stop/remove the container.

        Raises:
            ImageNotFoundError: If the image to be removed is not found (should not typically occur).

        """
        super().close()

        if self.container:
            if self.keep_template and self.docker_image:
                self._commit_container()

            # Only stop/remove container if we created it (not existing container)
            if not self.using_existing_container:
                try:
                    self.container.stop()
                    self.container.wait()
                    self.container.remove(force=True)
                    self._log("Stopped and removed container")
                except Exception:  # noqa: BLE001
                    self._log("Error cleaning up container")
            else:
                self._log("Disconnected from existing container")

            self.container = None

        if self.is_create_template and not self.keep_template and self.docker_image:
            self._cleanup_image()

    def _commit_container(self) -> None:
        """Commit container to new image."""
        if self.docker_image and self.docker_image.tags:
            full_tag = self.docker_image.tags[-1]
            repository, tag = full_tag.rsplit(":", 1) if ":" in full_tag else (full_tag, "latest")
            try:
                self.container.commit(repository=repository, tag=tag)
                self._log(f"Committed container as image {repository}:{tag}")
            except Exception:
                self._log("Failed to commit container", "error")
                raise

    def _cleanup_image(self) -> None:
        """Clean up image if not in use."""
        containers = self.client.containers.list(all=True, filters={"ancestor": self.docker_image.id})

        if not containers:
            try:
                self.docker_image.remove(force=True)
            except Exception as e:  # noqa: BLE001
                self._log(f"Failed to remove image: {e}", "error")
        else:
            self._log("Image in use by other containers, skipping removal")

    def get_archive(self, path: str) -> tuple[bytes, dict]:
        """Get archive from container."""
        if not self.container:
            raise NotOpenSessionError

        data, stat = self.container.get_archive(path)
        return b"".join(data), stat
