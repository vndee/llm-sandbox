# ruff: noqa: E501
import re
from typing import TYPE_CHECKING, Any

from podman import PodmanClient
from podman.errors.exceptions import ImageNotFound as PodmanImageNotFound
from podman.errors.exceptions import NotFound as PodmanNotFound

from llm_sandbox.const import EncodingErrorsType, SupportedLanguage
from llm_sandbox.core.config import SessionConfig
from llm_sandbox.docker import DockerContainerAPI, SandboxDockerSession
from llm_sandbox.exceptions import ContainerError, ImagePullError
from llm_sandbox.security import SecurityPolicy

if TYPE_CHECKING:
    from podman.domain.images import Image


class PodmanContainerAPI(DockerContainerAPI):
    """Podman implementation of the ContainerAPI protocol."""

    def __init__(self, client: PodmanClient, stream: bool = True) -> None:
        """Initialize Podman container API."""
        self.client = client
        self.stream = stream

    def execute_command(self, container: Any, command: str, **kwargs: Any) -> tuple[int, Any]:
        """Execute command in Podman container."""
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
        # Podman returns (exit_code, output) tuple, Docker returns object with .exit_code and .output
        exit_code, output = result
        return exit_code or 0, output


class SandboxPodmanSession(SandboxDockerSession):
    r"""Sandbox session implemented using Podman containers.

    This class provides a sandboxed environment for code execution by leveraging Podman.
    It inherits from SandboxDockerSession since Podman is designed to be Docker-compatible,
    only overriding the differences in client initialization and API behavior.
    """

    def __init__(
        self,  # NOSONAR
        client: PodmanClient | None = None,
        image: str | None = None,
        dockerfile: str | None = None,
        lang: str = SupportedLanguage.PYTHON,
        keep_template: bool = False,
        commit_container: bool = False,
        verbose: bool = False,
        mounts: list | None = None,
        stream: bool = False,
        runtime_configs: dict | None = None,
        workdir: str | None = "/sandbox",
        security_policy: SecurityPolicy | None = None,
        default_timeout: float | None = None,
        execution_timeout: float | None = None,
        session_timeout: float | None = None,
        container_id: str | None = None,
        skip_environment_setup: bool = False,
        encoding_errors: EncodingErrorsType = "strict",
        **kwargs: dict[str, Any],
    ) -> None:
        r"""Initialize Podman session.

        Args:
            client (PodmanClient | None): The Podman client to use.
            image (str | None): The image to use.
            dockerfile (str | None): The Dockerfile to use.
            lang (str): The language to use.
            keep_template (bool): Whether to keep the template image.
            commit_container (bool): Whether to commit the container to a new image.
            verbose (bool): Whether to enable verbose output.
            mounts (list | None): The mounts to use.
            stream (bool): Whether to stream the output.
            runtime_configs (dict | None): The runtime configurations to use.
            workdir (str | None): The working directory to use.
            security_policy (SecurityPolicy | None): The security policy to use.
            default_timeout (float | None): The default timeout to use.
            execution_timeout (float | None): The execution timeout to use.
            session_timeout (float | None): The session timeout to use.
            container_id (str | None): ID of existing container to connect to.
            skip_environment_setup (bool): Skip language-specific environment setup.
            encoding_errors (EncodingErrorsType): Error handling for decoding command output.
            **kwargs: Additional keyword arguments.

        Returns:
            None

        """
        config = SessionConfig(
            image=image,
            dockerfile=dockerfile,
            lang=SupportedLanguage(lang.upper()),
            verbose=verbose,
            workdir=workdir or "/sandbox",
            runtime_configs=runtime_configs or {},
            security_policy=security_policy,
            default_timeout=default_timeout,
            execution_timeout=execution_timeout,
            session_timeout=session_timeout,
            container_id=container_id,
            skip_environment_setup=skip_environment_setup,
            encoding_errors=encoding_errors,
        )

        # Initialize BaseSession (skip Docker's __init__)
        from llm_sandbox.core.session_base import BaseSession

        BaseSession.__init__(self, config=config, **kwargs)

        if not client:
            self._log("Using local Podman context since client is not provided.")
            self.client = PodmanClient.from_env()
        else:
            self.client = client

        self.container_api = PodmanContainerAPI(self.client, stream)

        # Set other attributes
        self.docker_image: Image
        self.keep_template: bool = keep_template
        self.commit_container: bool = commit_container
        self.is_create_template: bool = False
        self.stream: bool = stream

        if mounts:
            import warnings

            warnings.warn(
                "The 'mounts' parameter is deprecated and will be removed in a future version. "
                "Put the mounts in 'runtime_configs' instead.",
                DeprecationWarning,
                stacklevel=2,
            )
            self.config.runtime_configs.setdefault("mounts", []).append(mounts)

    def _get_or_pull_image(self) -> None:
        """Get local image or pull from registry (Podman-specific implementation).

        This method overrides the Docker implementation to handle Podman's
        different exception types for image not found errors.

        Raises:
            ImagePullError: If the image cannot be pulled from the registry.

        """
        try:
            self.docker_image = self.client.images.get(self.config.image)
            self._log(f"Using local image {self.docker_image.tags[-1] if self.docker_image.tags else 'untagged'}")
        except PodmanImageNotFound:
            self._log(f"Image {self.config.image} not found locally. Pulling...")
            try:
                self.docker_image = self.client.images.pull(self.config.image)
                self._log(
                    f"Successfully pulled image {self.docker_image.tags[-1] if self.docker_image.tags else 'untagged'}"
                )
                self.is_create_template = True
            except Exception as e:
                raise ImagePullError(self.config.image or "", str(e)) from e

    def _normalize_memory_limit(self, memory: str) -> str:
        """Normalize memory limit format for Podman.

        Podman's Python library is more strict about memory format than Docker.
        This converts formats like '1GB' to '1024m' which Podman can parse.

        If the input is already in Podman format (e.g., '1024m'), it returns it
        unchanged for efficiency.

        Args:
            memory: Memory limit string (e.g., '1GB', '512m', '2g', '1024m').

        Returns:
            Normalized memory limit string in megabytes (e.g., '1024m').

        """
        memory = memory.strip()

        # Check if already in Podman format: digits followed by lowercase 'm'
        # Podman expects format like '1024m', '512m', etc.
        if re.match(r"^\d+m$", memory):
            # Already in correct Podman format, return as-is
            return memory

        # Parse the memory string - handle both single and multi-character units
        # Match pattern like: 1GB, 512m, 2g, 1024k, etc.
        match = re.match(r"^(\d+)([bBkKmMgGtT][bB]?)$", memory)
        if not match:
            # If it doesn't match expected format, return as-is and let Podman handle the error
            return memory

        value, unit = match.groups()
        value_int = int(value)
        unit_lower = unit.lower()

        # Convert to megabytes
        # Handle both single char (k, m, g, t) and double char (kb, mb, gb, tb) units
        multipliers = {
            "b": 1 / (1024**2),  # bytes
            "kb": 1 / 1024,  # kilobytes
            "k": 1 / 1024,  # kilobytes (alternative)
            "mb": 1,  # megabytes
            "m": 1,  # megabytes (alternative)
            "gb": 1024,  # gigabytes
            "g": 1024,  # gigabytes (alternative)
            "tb": 1024**2,  # terabytes
            "t": 1024**2,  # terabytes (alternative)
        }

        if unit_lower not in multipliers:
            # Unknown unit, return as-is
            return memory

        # Convert to megabytes and format as 'XXXm'
        megabytes = int(value_int * multipliers[unit_lower])
        # Ensure we have at least 1MB
        megabytes = max(megabytes, 1)
        return f"{megabytes}m"

    def _normalize_runtime_configs_for_podman(self, runtime_configs: dict[str, Any]) -> dict[str, Any]:
        """Normalize runtime configs for Podman compatibility.

        Podman has different requirements than Docker for some parameters.
        This method normalizes the config to ensure Podman compatibility.

        Args:
            runtime_configs: Runtime configuration dictionary.

        Returns:
            Normalized runtime configuration dictionary.

        """
        normalized = runtime_configs.copy()

        # Normalize memory limit if present
        if "mem_limit" in normalized and isinstance(normalized["mem_limit"], str):
            normalized["mem_limit"] = self._normalize_memory_limit(normalized["mem_limit"])

        # Podman uses 'memory' instead of 'mem_limit' in some contexts
        # But based on the error, it seems to use 'mem_limit' in container creation
        # So we keep 'mem_limit' but normalize its format

        return normalized

    def open(self) -> None:
        """Open the Podman session with normalized runtime configs.

        This method overrides the Docker implementation to normalize
        runtime configurations for Podman compatibility before opening.

        """
        # Normalize runtime configs before opening
        if self.config.runtime_configs:
            self.config.runtime_configs = self._normalize_runtime_configs_for_podman(self.config.runtime_configs)

        # Call parent's open method
        super().open()

    def _connect_to_existing_container(self, container_id: str) -> None:
        """Connect to an existing Podman container.

        This method overrides the Docker implementation to handle Podman's
        different exception types for container not found errors.

        Args:
            container_id: The ID of the existing container to connect to.

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

        except PodmanNotFound as e:
            msg = f"Container {container_id} not found"
            self._log(msg, "error")
            raise ContainerError(msg) from e
        except Exception as e:
            msg = f"Failed to connect to container {container_id}: {e}"
            self._log(msg, "error")
            raise ContainerError(msg) from e
