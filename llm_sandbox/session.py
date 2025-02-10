"""Main session module for LLM Sandbox."""

from typing import Optional, TypeVar, cast, TYPE_CHECKING
from importlib.util import find_spec

from .const import SupportedLanguage, SandboxBackend
from .factory import UnifiedSessionFactory
from .exceptions import DependencyError
from .base import Session

# Type checking imports that won't be evaluated at runtime
if TYPE_CHECKING:
    pass

T = TypeVar("T", bound=Session)


def _check_dependency(backend: SandboxBackend) -> None:
    """Check if required dependency is installed for the given backend."""
    if backend == SandboxBackend.DOCKER or backend == SandboxBackend.MICROMAMBA:
        if not find_spec("docker"):
            raise DependencyError(
                "Docker backend requires 'docker' package. Install it with: pip install llm-sandbox[docker]"
            )
    elif backend == SandboxBackend.KUBERNETES:
        if not find_spec("kubernetes"):
            raise DependencyError(
                "Kubernetes backend requires 'kubernetes' package. Install it with: pip install llm-sandbox[k8s]"
            )
    elif backend == SandboxBackend.PODMAN:
        if not find_spec("podman"):
            raise DependencyError(
                "Podman backend requires 'podman' package. Install it with: pip install llm-sandbox[podman]"
            )


class ContextManagerMixin:
    """Mixin to add context manager support to session instances."""

    def __enter__(self):
        """Enter the context manager."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit the context manager."""
        if hasattr(self, "close"):
            self.close()


class SandboxSession:
    """Factory class for creating sandbox sessions with context manager support."""

    def __new__(
        cls,
        backend: SandboxBackend = SandboxBackend.DOCKER,
        image: Optional[str] = None,
        dockerfile: Optional[str] = None,
        lang: str = SupportedLanguage.PYTHON,
        keep_template: bool = False,
        commit_container: bool = True,
        verbose: bool = False,
        runtime_configs: Optional[dict] = None,
        **kwargs,
    ) -> "SandboxSession":
        """
        Create a new sandbox session.

        Args:
            backend: Docker, Kubernetes or Podman backend
            image: Container image to use
            dockerfile: Path to Dockerfile
            lang: Programming language
            keep_template: Whether to keep the container template
            commit_container: Whether to commit container changes
            verbose: Enable verbose logging
            runtime_configs: Additional runtime configurations, check the specific backend for more details

        Returns:
            A new sandbox session of the appropriate type (Docker, Kubernetes, Podman, or Micromamba)

        Raises:
            DependencyError: If the required dependency for the chosen backend is not installed
        """
        # Check if required dependency is installed
        _check_dependency(backend)

        factory = UnifiedSessionFactory()

        session = factory.create_session(
            backend=backend,
            image=image,
            dockerfile=dockerfile,
            lang=lang,
            keep_template=keep_template,
            commit_container=commit_container,
            verbose=verbose,
            runtime_configs=runtime_configs,
            **kwargs,
        )

        # Create a new class that inherits from both the session class and our mixin
        session.__class__ = type(
            "SandboxSession",
            (session.__class__, ContextManagerMixin, SandboxSession),
            {},
        )

        return cast("SandboxSession", session)
