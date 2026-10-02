"""Built-in backend providers.

The backends llm-sandbox ships and tests in CI, expressed through the same
`SandboxBackendPlugin` interface third-party backends use, so the registry has one code
path instead of a special case.

Two details are load-bearing and must not be "tidied":

1. Session classes are imported **inside** each method, never at module scope. That keeps
   `import llm_sandbox` free of optional backend dependencies (see
   ``tests/test_lazy_imports.py``) and keeps ``@patch("llm_sandbox.docker.SandboxDockerSession")``
   working, which several tests rely on.
2. The capability declarations reproduce today's behaviour exactly, including its gaps --
   micromamba genuinely has no interactive or pooling support, and asking for either has
   always raised. See ``docs/design/plugin-system.md`` section 5.2.
"""

from typing import TYPE_CHECKING, Any, ClassVar

from llm_sandbox.backends.plugin import PLUGIN_API_VERSION, BackendCapability, SandboxBackendPlugin

if TYPE_CHECKING:
    from llm_sandbox.core.session_base import BaseSession
    from llm_sandbox.pool.base import ContainerPoolManager

_CONTAINER_CAPABILITIES = frozenset({
    BackendCapability.ARTIFACTS,
    BackendCapability.INTERACTIVE,
    BackendCapability.POOLING,
    BackendCapability.EXISTING_CONTAINER,
})


class BuiltinBackend(SandboxBackendPlugin):
    """Marker base for backends that ship with llm-sandbox itself."""

    PLUGIN_API_VERSION: ClassVar[int] = PLUGIN_API_VERSION


class DockerBackend(BuiltinBackend):
    """The Docker backend. The default, and the most widely exercised."""

    name: ClassVar[str] = "docker"
    capabilities: ClassVar[frozenset[BackendCapability]] = _CONTAINER_CAPABILITIES

    @classmethod
    def create_session(cls, *args: Any, **kwargs: Any) -> "BaseSession":
        """Create a Docker sandbox session."""
        from llm_sandbox.docker import SandboxDockerSession

        return SandboxDockerSession(*args, **kwargs)

    @classmethod
    def create_pool_manager(cls, **kwargs: Any) -> "ContainerPoolManager":
        """Create a Docker container pool manager."""
        from llm_sandbox.pool.docker_pool import DockerPoolManager

        return DockerPoolManager(**kwargs)

    @classmethod
    def create_interactive_session(cls, **kwargs: Any) -> "BaseSession":
        """Create the Docker session backing an interactive session."""
        from llm_sandbox.docker import SandboxDockerSession

        return SandboxDockerSession(**kwargs)


class PodmanBackend(BuiltinBackend):
    """The Podman backend, for rootless containers."""

    name: ClassVar[str] = "podman"
    capabilities: ClassVar[frozenset[BackendCapability]] = _CONTAINER_CAPABILITIES

    @classmethod
    def create_session(cls, *args: Any, **kwargs: Any) -> "BaseSession":
        """Create a Podman sandbox session."""
        from llm_sandbox.podman import SandboxPodmanSession

        return SandboxPodmanSession(*args, **kwargs)

    @classmethod
    def create_pool_manager(cls, **kwargs: Any) -> "ContainerPoolManager":
        """Create a Podman container pool manager."""
        from llm_sandbox.pool.podman_pool import PodmanPoolManager

        return PodmanPoolManager(**kwargs)

    @classmethod
    def create_interactive_session(cls, **kwargs: Any) -> "BaseSession":
        """Create the Podman session backing an interactive session."""
        from llm_sandbox.podman import SandboxPodmanSession

        return SandboxPodmanSession(**kwargs)


class KubernetesBackend(BuiltinBackend):
    """The Kubernetes backend, for cluster-orchestrated execution."""

    name: ClassVar[str] = "kubernetes"
    capabilities: ClassVar[frozenset[BackendCapability]] = _CONTAINER_CAPABILITIES

    @classmethod
    def create_session(cls, *args: Any, **kwargs: Any) -> "BaseSession":
        """Create a Kubernetes sandbox session."""
        from llm_sandbox.kubernetes import SandboxKubernetesSession

        return SandboxKubernetesSession(*args, **kwargs)

    @classmethod
    def create_pool_manager(cls, **kwargs: Any) -> "ContainerPoolManager":
        """Create a Kubernetes pod pool manager."""
        from llm_sandbox.pool.kubernetes_pool import KubernetesPoolManager

        return KubernetesPoolManager(**kwargs)

    @classmethod
    def create_interactive_session(cls, **kwargs: Any) -> "BaseSession":
        """Create the Kubernetes session backing an interactive session.

        The Kubernetes session does not accept ``runtime_configs``; it is dropped here
        rather than raising a ``TypeError``, matching long-standing behaviour.
        """
        from llm_sandbox.kubernetes import SandboxKubernetesSession

        return SandboxKubernetesSession(**{k: v for k, v in kwargs.items() if k != "runtime_configs"})


class MicromambaBackend(BuiltinBackend):
    """The Micromamba backend: a Docker container with commands wrapped in ``micromamba run``.

    Declares neither ``INTERACTIVE`` nor ``POOLING``. Neither has ever worked for this
    backend, and both have always raised `llm_sandbox.exceptions.UnsupportedBackendError`.
    """

    name: ClassVar[str] = "micromamba"
    capabilities: ClassVar[frozenset[BackendCapability]] = frozenset({
        BackendCapability.ARTIFACTS,
        BackendCapability.EXISTING_CONTAINER,
    })

    @classmethod
    def create_session(cls, *args: Any, **kwargs: Any) -> "BaseSession":
        """Create a Micromamba sandbox session."""
        from llm_sandbox.micromamba import MicromambaSession

        return MicromambaSession(*args, **kwargs)


#: Built-in backends, keyed by their normalised name. A plugin can never take one of these
#: names -- see ``registry._discover``.
BUILTIN_BACKENDS: dict[str, type[BuiltinBackend]] = {
    DockerBackend.name: DockerBackend,
    KubernetesBackend.name: KubernetesBackend,
    PodmanBackend.name: PodmanBackend,
    MicromambaBackend.name: MicromambaBackend,
}
