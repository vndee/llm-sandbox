"""Docker-specific container pool manager."""

from typing import Any

import docker
from docker.errors import NotFound
from docker.models.containers import Container

from llm_sandbox.const import DefaultImage, SupportedLanguage
from llm_sandbox.pool.base import ContainerPoolManager
from llm_sandbox.pool.config import PoolConfig


class DockerPoolManager(ContainerPoolManager):
    """Container pool manager for Docker backend.

    This manager creates and manages a pool of Docker containers,
    reusing the standard session logic for container initialization and
    environment setup (venv, pip, library installation, etc.).
    """

    def __init__(
        self,
        config: PoolConfig,
        lang: SupportedLanguage,
        image: str | None = None,
        dockerfile: str | None = None,
        client: docker.DockerClient | None = None,
        runtime_configs: dict | None = None,
        **session_kwargs: Any,
    ) -> None:
        """Initialize Docker pool manager.

        Args:
            config: Pool configuration
            lang: Programming language
            image: Docker image to use
            dockerfile: Path to Dockerfile (alternative to image)
            client: Docker client instance (creates default if None)
            runtime_configs: Docker runtime configurations
            **session_kwargs: Additional session arguments

        """
        self.client = client or docker.from_env()
        self.dockerfile = dockerfile
        self.runtime_configs = runtime_configs or {}

        # Resolve image
        if not image and not dockerfile:
            image = DefaultImage.__dict__[lang.upper()]

        super().__init__(client=client, config=config, lang=lang, image=image, **session_kwargs)

    def _create_session_for_container(self) -> Any:
        """Create a Docker session for initializing a container.

        This creates a session that, when opened, will:
        1. Prepare/pull the Docker image
        2. Create a container
        3. Start the container
        4. Set up the environment (venv, pip, libraries, etc.)

        Returns:
            DockerSession instance (not yet opened)

        """
        from llm_sandbox.docker import SandboxDockerSession

        # Create session with same configuration as the pool
        # The session handles all initialization automatically
        return SandboxDockerSession(
            client=self.client,
            image=self.image,
            dockerfile=self.dockerfile,
            lang=self.lang.value,
            runtime_configs=self.runtime_configs,
            **self.session_kwargs,
        )

    def _destroy_container_impl(self, container: Container) -> None:
        """Destroy a Docker container.

        Args:
            container: Container to destroy

        """
        try:
            container.stop()
            container.wait()
            container.remove(force=True)
        except Exception:
            self.logger.exception("Failed to destroy container")

    def _get_container_id(self, container: Container) -> str:
        """Get Docker container ID.

        Args:
            container: Docker container

        Returns:
            Container ID

        """
        return str(container.id)

    def _health_check_impl(self, container: Container) -> bool:
        """Perform health check on Docker container.

        Args:
            container: Container to check

        Returns:
            True if healthy, False otherwise

        """
        try:
            container.reload()
            if container.status != "running":
                return False

            # Execute a simple command to verify container responsiveness
            exit_code, _ = container.exec_run("echo health_check", tty=False)
            return bool(exit_code == 0)

        except NotFound:
            return False
        except Exception:
            self.logger.exception("Health check error")
            return False
