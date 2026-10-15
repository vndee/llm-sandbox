"""Backend registration and static adapter capabilities.

Discovery never imports a backend SDK or probes the host. Capabilities describe
features implemented by an adapter, not protections enabled for a particular
session or guarantees that the current host can run it.
"""

import re
from collections.abc import Callable
from dataclasses import dataclass
from threading import RLock
from typing import TYPE_CHECKING, Any

from llm_sandbox.const import SandboxBackend
from llm_sandbox.exceptions import UnsupportedBackendError

if TYPE_CHECKING:
    from llm_sandbox.core.session_base import BaseSession

BackendFactory = Callable[..., "BaseSession"]


@dataclass(frozen=True)
class BackendCapabilities:
    """Immutable features exposed by an adapter, subject to host support.

    ``platforms`` lists client platforms using ``sys.platform`` names; an empty
    tuple means unspecified. ``network_isolation`` and ``seccomp`` indicate
    that the adapter can configure these controls, not that they are enabled.
    ``daemon_required`` includes a remote container service or cluster API.
    For custom backends, ``pooling=True`` opts in to the existing supplied-pool
    behavior: a pooled session delegates execution to that pool's backend and
    does not call the registered factory.
    """

    platforms: tuple[str, ...] = ()
    container_images: bool = False
    interactive: bool = False
    pooling: bool = False
    filesystem_isolation: bool = False
    network_isolation: bool = False
    seccomp: bool = False
    daemon_required: bool = False

    def __post_init__(self) -> None:
        """Snapshot platform collections so callers cannot mutate metadata."""
        object.__setattr__(self, "platforms", tuple(self.platforms))


@dataclass(frozen=True)
class BackendDefinition:
    """A registered session factory and its static feature description."""

    name: str
    factory: BackendFactory
    capabilities: BackendCapabilities
    builtin: bool = False


def _docker_factory(*args: Any, **kwargs: Any) -> "BaseSession":
    """Import the Docker SDK only when its backend is selected."""
    from llm_sandbox.docker import SandboxDockerSession

    return SandboxDockerSession(*args, **kwargs)


def _kubernetes_factory(*args: Any, **kwargs: Any) -> "BaseSession":
    """Import the Kubernetes SDK only when its backend is selected."""
    from llm_sandbox.kubernetes import SandboxKubernetesSession

    return SandboxKubernetesSession(*args, **kwargs)


def _podman_factory(*args: Any, **kwargs: Any) -> "BaseSession":
    """Import the Podman SDK only when its backend is selected."""
    from llm_sandbox.podman import SandboxPodmanSession

    return SandboxPodmanSession(*args, **kwargs)


def _micromamba_factory(*args: Any, **kwargs: Any) -> "BaseSession":
    """Import the Docker-based Micromamba backend only when selected."""
    from llm_sandbox.micromamba import MicromambaSession

    return MicromambaSession(*args, **kwargs)


def _bubblewrap_factory(*args: Any, **kwargs: Any) -> "BaseSession":
    """Import the local Bubblewrap backend only when selected."""
    from llm_sandbox.bubblewrap import BubblewrapSession

    return BubblewrapSession(*args, **kwargs)


def _nsjail_factory(*args: Any, **kwargs: Any) -> "BaseSession":
    """Import the local NsJail backend only when selected."""
    from llm_sandbox.nsjail import NsjailSession

    return NsjailSession(*args, **kwargs)


def _windows_mxc_factory(*args: Any, **kwargs: Any) -> "BaseSession":
    """Load the native Windows adapter only when explicitly selected."""
    from llm_sandbox.windows_mxc import WindowsMxcSession

    return WindowsMxcSession(*args, **kwargs)


def _wsl2_bubblewrap_factory(*args: Any, **kwargs: Any) -> "BaseSession":
    """Load the Windows client for an explicitly selected WSL2 distribution."""
    from llm_sandbox.wsl2_bubblewrap import Wsl2BubblewrapSession

    return Wsl2BubblewrapSession(*args, **kwargs)


_CONTAINER_CAPABILITIES = BackendCapabilities(
    container_images=True,
    interactive=True,
    pooling=True,
    filesystem_isolation=True,
    network_isolation=True,
    seccomp=True,
    daemon_required=True,
)
_LOCAL_CAPABILITIES = BackendCapabilities(
    platforms=("linux",),
    filesystem_isolation=True,
    network_isolation=True,
    seccomp=True,
)
_BACKENDS = {
    definition.name: definition
    for definition in (
        BackendDefinition("docker", _docker_factory, _CONTAINER_CAPABILITIES, builtin=True),
        BackendDefinition(
            "kubernetes",
            _kubernetes_factory,
            BackendCapabilities(
                container_images=True,
                interactive=True,
                pooling=True,
                filesystem_isolation=True,
                daemon_required=True,
            ),
            builtin=True,
        ),
        BackendDefinition("podman", _podman_factory, _CONTAINER_CAPABILITIES, builtin=True),
        BackendDefinition(
            "micromamba",
            _micromamba_factory,
            BackendCapabilities(
                container_images=True,
                filesystem_isolation=True,
                network_isolation=True,
                seccomp=True,
                daemon_required=True,
            ),
            builtin=True,
        ),
        BackendDefinition("bubblewrap", _bubblewrap_factory, _LOCAL_CAPABILITIES, builtin=True),
        BackendDefinition("nsjail", _nsjail_factory, _LOCAL_CAPABILITIES, builtin=True),
        BackendDefinition(
            "windows-mxc",
            _windows_mxc_factory,
            BackendCapabilities(platforms=("win32",), filesystem_isolation=True, network_isolation=True),
            builtin=True,
        ),
        BackendDefinition(
            "wsl2-bubblewrap",
            _wsl2_bubblewrap_factory,
            BackendCapabilities(platforms=("win32",), filesystem_isolation=True, network_isolation=True, seccomp=True),
            builtin=True,
        ),
    )
}
_REGISTRY_LOCK = RLock()


def _normalize_name(name: str) -> str:
    """Return a canonical backend name without changing meaningful characters."""
    if not isinstance(name, str):
        msg = "Backend names must be strings"
        raise TypeError(msg)
    normalized = name.lower()
    if re.fullmatch(r"[a-z][a-z0-9_-]*", normalized) is None:
        msg = "Backend names must start with a letter and contain only letters, digits, underscores, or hyphens"
        raise ValueError(msg)
    return normalized


def register_backend(
    name: str,
    factory: BackendFactory,
    *,
    capabilities: BackendCapabilities | None = None,
) -> BackendDefinition:
    """Register a custom session factory without replacing existing backends.

    The factory receives the arguments supplied to ``SandboxSession`` and must
    return a ``BaseSession`` implementation. Its dependencies and host checks
    are the adapter's responsibility. Registration does not execute the factory.
    """
    normalized = _normalize_name(name)
    if not callable(factory):
        msg = "Backend factory must be callable"
        raise TypeError(msg)
    if capabilities is not None and not isinstance(capabilities, BackendCapabilities):
        msg = "capabilities must be a BackendCapabilities instance"
        raise TypeError(msg)
    with _REGISTRY_LOCK:
        if normalized in _BACKENDS:
            msg = f"Backend '{normalized}' is already registered and cannot be replaced"
            raise ValueError(msg)
        definition = BackendDefinition(normalized, factory, capabilities or BackendCapabilities())
        _BACKENDS[normalized] = definition
        return definition


def get_backend(name: SandboxBackend | str) -> BackendDefinition:
    """Get backend metadata without importing its runtime or checking the host."""
    try:
        normalized = _normalize_name(name)
    except (TypeError, ValueError) as exc:
        raise UnsupportedBackendError(backend=name) from exc
    with _REGISTRY_LOCK:
        try:
            return _BACKENDS[normalized]
        except KeyError as exc:
            raise UnsupportedBackendError(backend=name) from exc


def list_backends() -> tuple[BackendDefinition, ...]:
    """Return an immutable snapshot of registered backends in registration order."""
    with _REGISTRY_LOCK:
        return tuple(_BACKENDS.values())
