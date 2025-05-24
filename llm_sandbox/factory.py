"""Factory for creating sandbox sessions."""

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Union

from .const import SandboxBackend
from .exceptions import UnsupportedBackendError

if TYPE_CHECKING:
    from .docker import SandboxDockerSession
    from .kubernetes import SandboxKubernetesSession
    from .micromamba import MicromambaSession
    from .podman import SandboxPodmanSession


class SessionFactory(ABC):
    """Abstract base factory for creating sandbox sessions."""

    @abstractmethod
    def create_session(
        self, **kwargs: dict[str, Any]
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

    def create_session(self, **kwargs: dict[str, Any]) -> "SandboxDockerSession":
        """Create a new Docker sandbox session."""
        from .docker import SandboxDockerSession

        return SandboxDockerSession(**kwargs)


class KubernetesSessionFactory(SessionFactory):
    """Factory for creating Kubernetes-based sandbox sessions."""

    def create_session(self, **kwargs: dict[str, Any]) -> "SandboxKubernetesSession":
        """Create a new Kubernetes sandbox session."""
        from .kubernetes import SandboxKubernetesSession

        return SandboxKubernetesSession(**kwargs)


class PodmanSessionFactory(SessionFactory):
    """Factory for creating Podman-based sandbox sessions."""

    def create_session(self, **kwargs: dict[str, Any]) -> "SandboxPodmanSession":
        """Create a new Podman sandbox session."""
        from .podman import SandboxPodmanSession

        return SandboxPodmanSession(**kwargs)


class MicromambaSessionFactory(SessionFactory):
    """Factory for creating Micromamba-based sandbox sessions."""

    def create_session(self, **kwargs: dict[str, Any]) -> "MicromambaSession":
        """Create a new Micromamba sandbox session."""
        from .micromamba import MicromambaSession

        return MicromambaSession(**kwargs)


class SessionFactory:
    """Factory for creating sandbox sessions of any type."""

    def create_session(
        self,
        backend: SandboxBackend = SandboxBackend.DOCKER,
        **kwargs: dict[str, Any],
    ) -> Union[
        "SandboxDockerSession",
        "SandboxKubernetesSession",
        "SandboxPodmanSession",
        "MicromambaSession",
    ]:
        """Create a new sandbox session of the specified type."""
        factories: dict[SandboxBackend, type[SessionFactory]] = {
            SandboxBackend.DOCKER: DockerSessionFactory,
            SandboxBackend.KUBERNETES: KubernetesSessionFactory,
            SandboxBackend.PODMAN: PodmanSessionFactory,
            SandboxBackend.MICROMAMBA: MicromambaSessionFactory,
        }

        factory_class = factories.get(backend)
        if factory_class is None:
            raise UnsupportedBackendError(backend)

        return factory_class().create_session(**kwargs)
