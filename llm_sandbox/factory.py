"""Factory for creating sandbox sessions."""

from typing import Union, Type, TYPE_CHECKING
from abc import ABC, abstractmethod

from .const import SandboxBackend
from .exceptions import ValidationError

if TYPE_CHECKING:
    from .docker import SandboxDockerSession
    from .kubernetes import SandboxKubernetesSession
    from .podman import SandboxPodmanSession
    from .micromamba import MicromambaSession


class SessionFactory(ABC):
    """Abstract base factory for creating sandbox sessions."""

    @abstractmethod
    def create_session(
        self, **kwargs
    ) -> Union[
        "SandboxDockerSession",
        "SandboxKubernetesSession",
        "SandboxPodmanSession",
        "MicromambaSession",
    ]:
        """Create a new sandbox session."""
        raise NotImplementedError


class DockerSessionFactory(SessionFactory):
    """Factory for creating Docker-based sandbox sessions."""

    def create_session(self, **kwargs) -> "SandboxDockerSession":
        """Create a new Docker sandbox session."""
        from .docker import SandboxDockerSession

        return SandboxDockerSession(**kwargs)


class KubernetesSessionFactory(SessionFactory):
    """Factory for creating Kubernetes-based sandbox sessions."""

    def create_session(self, **kwargs) -> "SandboxKubernetesSession":
        """Create a new Kubernetes sandbox session."""
        from .kubernetes import SandboxKubernetesSession

        return SandboxKubernetesSession(**kwargs)


class PodmanSessionFactory(SessionFactory):
    """Factory for creating Podman-based sandbox sessions."""

    def create_session(self, **kwargs) -> "SandboxPodmanSession":
        """Create a new Podman sandbox session."""
        from .podman import SandboxPodmanSession

        return SandboxPodmanSession(**kwargs)


class MicromambaSessionFactory(SessionFactory):
    """Factory for creating Micromamba-based sandbox sessions."""

    def create_session(self, **kwargs) -> "MicromambaSession":
        """Create a new Micromamba sandbox session."""
        from .micromamba import MicromambaSession

        return MicromambaSession(**kwargs)


class UnifiedSessionFactory:
    """Unified factory for creating sandbox sessions of any type."""

    def create_session(
        self, backend: SandboxBackend = SandboxBackend.DOCKER, **kwargs
    ) -> Union[
        "SandboxDockerSession",
        "SandboxKubernetesSession",
        "SandboxPodmanSession",
        "MicromambaSession",
    ]:
        """Create a new sandbox session of the specified type."""
        factories: dict[SandboxBackend, Type[SessionFactory]] = {
            SandboxBackend.DOCKER: DockerSessionFactory,
            SandboxBackend.KUBERNETES: KubernetesSessionFactory,
            SandboxBackend.PODMAN: PodmanSessionFactory,
            SandboxBackend.MICROMAMBA: MicromambaSessionFactory,
        }

        factory_class = factories.get(backend)
        if factory_class is None:
            raise ValidationError(f"Unsupported backend: {backend}")

        return factory_class().create_session(**kwargs)
