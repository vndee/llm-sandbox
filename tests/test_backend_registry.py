"""Test backend discovery and extension without requiring sandbox binaries."""

import subprocess
import sys
import textwrap
from dataclasses import FrozenInstanceError
from typing import TYPE_CHECKING
from unittest.mock import MagicMock, patch

import pytest

import llm_sandbox.backends as backend_registry
from llm_sandbox import (
    ArtifactSandboxSession,
    BackendCapabilities,
    InteractiveSandboxSession,
    SandboxBackend,
    SandboxSession,
    get_backend,
    list_backends,
    register_backend,
)
from llm_sandbox.exceptions import UnsupportedBackendError, ValidationError

if TYPE_CHECKING:
    from collections.abc import Callable


@pytest.fixture(autouse=True)
def isolated_registry(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep custom test registrations out of the process-wide registry."""
    monkeypatch.setattr(backend_registry, "_BACKENDS", backend_registry._BACKENDS.copy())


def test_discovery_describes_builtins_without_importing_backends() -> None:
    """Discovery lists static features without executing factories or host probes."""
    definitions = list_backends()
    assert {definition.name for definition in definitions if definition.builtin} == {
        backend.value for backend in SandboxBackend
    }
    bubblewrap = get_backend("BUBBLEWRAP")
    assert bubblewrap.capabilities.platforms == ("linux",)
    assert bubblewrap.capabilities.filesystem_isolation
    assert bubblewrap.capabilities.network_isolation
    assert bubblewrap.capabilities.seccomp
    assert not bubblewrap.capabilities.daemon_required
    assert not bubblewrap.capabilities.container_images
    assert not bubblewrap.capabilities.pooling
    assert not bubblewrap.capabilities.interactive


@pytest.mark.parametrize(("name", "seccomp"), [("windows-mxc", False), ("wsl2-bubblewrap", True)])
def test_windows_backend_capabilities(name: str, *, seccomp: bool) -> None:
    """Windows client adapters distinguish native PSEC from Linux syscall filtering."""
    capabilities = get_backend(name).capabilities
    assert capabilities.platforms == ("win32",)
    assert capabilities.filesystem_isolation
    assert capabilities.network_isolation
    assert capabilities.seccomp is seccomp
    assert not capabilities.container_images
    assert not capabilities.interactive
    assert not capabilities.pooling
    assert not capabilities.daemon_required


@pytest.mark.parametrize("plotting", [True, False])
def test_windows_mxc_rejects_posix_artifact_wrapper(*, plotting: bool) -> None:
    """The native adapter cannot execute the plotting wrapper's POSIX scripts."""
    with pytest.raises(ValidationError, match="does not support the artifact wrapper"):
        ArtifactSandboxSession(backend="windows-mxc", enable_plotting=plotting)


def test_custom_factory_receives_arguments_through_public_alias() -> None:
    """The existing SandboxSession alias can select an explicitly registered adapter."""
    factory = MagicMock()
    metadata = BackendCapabilities(platforms=("linux",), filesystem_isolation=True)
    definition = register_backend("Example", factory, capabilities=metadata)
    factory.assert_not_called()

    result = SandboxSession("example", None, "argument", lang="python", local_option=True)

    factory.assert_called_once_with("argument", lang="python", local_option=True)
    assert result is factory.return_value
    assert get_backend("EXAMPLE") is definition
    assert definition.capabilities is metadata
    assert not definition.builtin


def test_registration_and_capabilities_are_immutable_snapshots() -> None:
    """Returned discovery snapshots cannot mutate capabilities or registry entries."""
    before = list_backends()
    definition = register_backend("example", MagicMock())
    after = list_backends()
    assert definition not in before
    assert definition in after
    assert isinstance(before, tuple)
    with pytest.raises(FrozenInstanceError):
        definition.capabilities.seccomp = True  # type: ignore[misc]
    with pytest.raises(FrozenInstanceError):
        definition.name = "renamed"  # type: ignore[misc]


@pytest.mark.parametrize("name", ["docker", "DOCKER", "bubblewrap", "nsjail", "windows-mxc", "wsl2-bubblewrap"])
def test_registration_cannot_replace_builtins(name: str) -> None:
    """Plugins cannot shadow built-in security boundaries."""
    original = get_backend(name)
    factory = MagicMock()
    with pytest.raises(ValueError, match="already registered"):
        register_backend(name, factory)
    assert get_backend(name) is original


def test_duplicate_custom_registration_is_rejected() -> None:
    """Duplicate names fail explicitly, including case-only collisions."""
    original = register_backend("example", MagicMock())
    duplicate_factory = MagicMock()
    with pytest.raises(ValueError, match="already registered"):
        register_backend("EXAMPLE", duplicate_factory)
    assert get_backend("example") is original


@pytest.mark.parametrize("name", ["", " padded", "two words", "../backend", "1backend"])
def test_invalid_registration_names_are_rejected(name: str) -> None:
    """Names have a predictable syntax for configuration and discovery."""
    factory = MagicMock()
    with pytest.raises(ValueError, match="Backend names"):
        register_backend(name, factory)


def test_invalid_registration_types_are_rejected() -> None:
    """Invalid extension contracts fail at registration time."""
    with pytest.raises(TypeError, match="callable"):
        register_backend("example", None)  # type: ignore[arg-type]
    factory = MagicMock()
    with pytest.raises(TypeError, match="BackendCapabilities"):
        register_backend("example", factory, capabilities={})  # type: ignore[arg-type]


@pytest.mark.parametrize("name", ["missing-backend", "invalid name"])
def test_unknown_backend_never_falls_back(name: str) -> None:
    """Unsupported requests cannot select another isolation mechanism implicitly."""
    with pytest.raises(UnsupportedBackendError):
        SandboxSession(backend=name)


@pytest.mark.parametrize("name", ["bubblewrap", "nsjail", "BUBBLEWRAP", "windows-mxc", "wsl2-bubblewrap"])
@pytest.mark.parametrize("session_factory", [SandboxSession, ArtifactSandboxSession])
def test_local_backends_reject_container_pools(name: str, session_factory: "Callable[..., object]") -> None:
    """A supplied container pool must not bypass an explicit local backend request."""
    pool = MagicMock()
    with pytest.raises(ValidationError, match="does not support container pooling"):
        session_factory(backend=name, pool=pool)


@pytest.mark.parametrize("name", ["micromamba", "MICROMAMBA"])
@pytest.mark.parametrize("session_factory", [SandboxSession, ArtifactSandboxSession])
def test_micromamba_rejects_container_pools(name: str, session_factory: "Callable[..., object]") -> None:
    """A pool must not bypass a built-in backend's declared lack of pooling support."""
    pool = MagicMock()
    with pytest.raises(ValidationError, match="does not support container pooling"):
        session_factory(backend=name, pool=pool)


@pytest.mark.parametrize("session_factory", [SandboxSession, ArtifactSandboxSession])
def test_custom_backends_reject_pools_without_explicit_support(session_factory: "Callable[..., object]") -> None:
    """A pool must not silently replace a custom backend unless it opts in."""
    factory = MagicMock()
    register_backend("example", factory)
    pool = MagicMock()
    with pytest.raises(ValidationError, match="does not support container pooling"):
        session_factory(backend="example", pool=pool)
    factory.assert_not_called()


def test_custom_backend_can_opt_in_to_supplied_pool_semantics() -> None:
    """Explicit pooling support uses the supplied pool instead of the custom factory."""
    factory = MagicMock()
    register_backend("example", factory, capabilities=BackendCapabilities(pooling=True))
    pool = MagicMock()
    with patch("llm_sandbox.pool.session.PooledSandboxSession") as pooled_session:
        session = SandboxSession(backend="example", pool=pool, lang="python")
    assert session is pooled_session.return_value
    pooled_session.assert_called_once_with(pool_manager=pool, lang="python")
    factory.assert_not_called()


@pytest.mark.parametrize("session_factory", [SandboxSession, ArtifactSandboxSession])
def test_unknown_backend_cannot_be_replaced_by_pool(session_factory: "Callable[..., object]") -> None:
    """A misspelled or missing backend remains an error when a pool is supplied."""
    pool = MagicMock()
    with pytest.raises(UnsupportedBackendError):
        session_factory(backend="missing-backend", pool=pool)


@pytest.mark.parametrize(
    "backend",
    [SandboxBackend.BUBBLEWRAP, SandboxBackend.NSJAIL, SandboxBackend.WINDOWS_MXC, SandboxBackend.WSL2_BUBBLEWRAP],
)
def test_local_backends_reject_interactive_sessions(backend: SandboxBackend) -> None:
    """Process-per-command adapters cannot promise a persistent background kernel."""
    with pytest.raises(UnsupportedBackendError):
        InteractiveSandboxSession(backend=backend)


@pytest.mark.parametrize(
    ("name", "factory_path"),
    [
        ("bubblewrap", "llm_sandbox.bubblewrap.BubblewrapSession"),
        ("nsjail", "llm_sandbox.nsjail.NsjailSession"),
        ("windows-mxc", "llm_sandbox.windows_mxc.WindowsMxcSession"),
        ("wsl2-bubblewrap", "llm_sandbox.wsl2_bubblewrap.Wsl2BubblewrapSession"),
    ],
)
def test_local_backend_factory_is_lazy(name: str, factory_path: str) -> None:
    """The selected local factory receives options without Docker dependency checks."""
    with patch(factory_path) as factory, patch("llm_sandbox.session.find_spec") as dependency_probe:
        result = SandboxSession(backend=name, lang="python", skip_environment_setup=True)
    assert result is factory.return_value
    factory.assert_called_once_with(lang="python", skip_environment_setup=True)
    dependency_probe.assert_not_called()


@pytest.mark.parametrize("backend", ["bubblewrap", "nsjail", "windows-mxc", "wsl2-bubblewrap"])
def test_local_backend_imports_without_container_sdks(backend: str) -> None:
    """A clean Python process can load local adapters while container imports are blocked."""
    program = textwrap.dedent(
        """
        import builtins
        import sys

        original_import = builtins.__import__
        forbidden = {"docker", "podman", "kubernetes"}

        def guarded_import(name, *args, **kwargs):
            if name.split(".")[0] in forbidden or name in {
                "llm_sandbox.docker", "llm_sandbox.podman", "llm_sandbox.kubernetes",
            }:
                raise AssertionError("Unexpected container dependency: " + name)
            return original_import(name, *args, **kwargs)

        builtins.__import__ = guarded_import
        from unittest.mock import patch
        from llm_sandbox import SandboxSession, list_backends

        backend = sys.argv[1]
        module_name, class_name = {
            "bubblewrap": ("bubblewrap", "BubblewrapSession"),
            "nsjail": ("nsjail", "NsjailSession"),
            "windows-mxc": ("windows_mxc", "WindowsMxcSession"),
            "wsl2-bubblewrap": ("wsl2_bubblewrap", "Wsl2BubblewrapSession"),
        }[backend]
        list_backends()
        with patch("llm_sandbox." + module_name + "." + class_name) as factory:
            result = SandboxSession(backend=backend, lang="python")
            assert result is factory.return_value
            factory.assert_called_once_with(lang="python")
        assert not forbidden.intersection(sys.modules)
        """,
    )
    result = subprocess.run(  # noqa: S603 - fixed test program and the current Python interpreter
        [sys.executable, "-c", program, backend],
        capture_output=True,
        text=True,
        check=False,
        timeout=20,
    )
    assert result.returncode == 0, result.stderr
