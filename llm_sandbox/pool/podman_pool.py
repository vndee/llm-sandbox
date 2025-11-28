"""Podman-specific container pool manager."""

from typing import Any

from podman import PodmanClient

from llm_sandbox.const import SupportedLanguage
from llm_sandbox.pool.config import PoolConfig
from llm_sandbox.pool.docker_pool import DockerPoolManager


class PodmanPoolManager(DockerPoolManager):
    """Container pool manager for Podman backend.

    Since Podman is Docker-compatible, this class inherits from
    DockerPoolManager and only overrides client initialization.
    The standard session logic handles all environment setup.
    """

    def __init__(
        self,
        config: PoolConfig,
        lang: SupportedLanguage,
        image: str | None = None,
        dockerfile: str | None = None,
        client: PodmanClient | None = None,
        runtime_configs: dict | None = None,
        **session_kwargs: Any,
    ) -> None:
        """Initialize Podman pool manager.

        Args:
            config: Pool configuration
            lang: Programming language
            image: Container image to use
            dockerfile: Path to Dockerfile (alternative to image)
            client: Podman client instance (creates default if None)
            runtime_configs: Podman runtime configurations
            **session_kwargs: Additional session arguments

        """
        # Initialize Podman client if not provided
        if client is None:
            client = PodmanClient.from_env()

        self.dockerfile = dockerfile
        self.runtime_configs = runtime_configs or {}

        # Resolve image using helper
        from llm_sandbox.pool.base import resolve_default_image

        image = resolve_default_image(lang, image, dockerfile)

        # Call parent init with proper parameters
        super().__init__(
            config=config,
            lang=lang,
            image=image,
            dockerfile=dockerfile,
            client=client,
            runtime_configs=runtime_configs,
            **session_kwargs,
        )

    def _create_session_for_container(self) -> Any:
        """Create a Podman session for initializing a container.

        Returns:
            PodmanSession instance (not yet opened)

        """
        from llm_sandbox.podman import SandboxPodmanSession

        # Create session with same configuration as the pool
        # The session handles all initialization automatically
        return SandboxPodmanSession(
            client=self.client,
            image=self.image,
            dockerfile=self.dockerfile,
            lang=self.lang.value,
            runtime_configs=self.runtime_configs,
            **self.session_kwargs,
        )
