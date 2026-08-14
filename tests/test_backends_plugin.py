"""Tests for the frozen plugin contract and the public backend base class."""

import dataclasses
import importlib
from typing import Any, ClassVar

import pytest

import llm_sandbox
from llm_sandbox.backends import (
    ENTRY_POINT_GROUP,
    PLUGIN_API_VERSION,
    SUPPORTED_PLUGIN_API_VERSIONS,
    BackendCapability,
    BackendInfo,
    SandboxBackendBase,
    SandboxBackendPlugin,
)
from llm_sandbox.core.config import SessionConfig
from llm_sandbox.exceptions import (
    BackendCapabilityError,
    BackendLoadError,
    BackendNameConflictError,
    BackendNotFoundError,
    SandboxError,
    UnsupportedBackendError,
)


class _MinimalPlugin(SandboxBackendPlugin):
    """The smallest thing that satisfies the contract."""

    PLUGIN_API_VERSION: ClassVar[int] = 1
    name: ClassVar[str] = "minimal"

    @classmethod
    def create_session(cls, *args: Any, **kwargs: Any) -> Any:
        return {"args": args, "kwargs": kwargs}


class _StubSession(SandboxBackendBase):
    """A concrete SandboxBackendBase used to exercise the bridges."""

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.calls: list[tuple[str, Any]] = []

    def open(self) -> None:
        super().open()

    def close(self) -> None:
        super().close()

    def handle_timeout(self) -> None:
        self.calls.append(("handle_timeout", None))

    def ensure_directory_exists(self, path: str) -> None:
        self.calls.append(("ensure_directory_exists", path))

    def ensure_ownership(self, paths: list[str]) -> None:
        self.calls.append(("ensure_ownership", paths))

    def process_non_stream_output(self, output: Any) -> tuple[str, str]:
        self.calls.append(("process_non_stream_output", output))
        return "out", "err"

    def process_stream_output(
        self,
        output: Any,
        on_stdout: Any = None,
        on_stderr: Any = None,
    ) -> tuple[str, str]:
        self.calls.append(("process_stream_output", (output, on_stdout, on_stderr)))
        return "sout", "serr"


class TestPluginApiVersion:
    """The version constant is the compatibility contract."""

    def test_current_version_is_supported(self) -> None:
        """Current version is supported."""
        assert PLUGIN_API_VERSION in SUPPORTED_PLUGIN_API_VERSIONS

    def test_supported_versions_is_immutable(self) -> None:
        """Supported versions is immutable."""
        assert isinstance(SUPPORTED_PLUGIN_API_VERSIONS, frozenset)

    def test_entry_point_group_is_stable(self) -> None:
        """Changing this breaks every published plugin, so pin it in a test."""
        assert ENTRY_POINT_GROUP == "llm_sandbox.backends"


class TestSandboxBackendPlugin:
    """The registered descriptor."""

    def test_create_session_is_abstract(self) -> None:
        """Create session is abstract."""
        class Incomplete(SandboxBackendPlugin):
            PLUGIN_API_VERSION: ClassVar[int] = 1
            name: ClassVar[str] = "incomplete"

        assert "create_session" in Incomplete.__abstractmethods__

    def test_forwards_positional_and_keyword_arguments(self) -> None:
        """Forwards positional and keyword arguments."""
        result = _MinimalPlugin.create_session("positional", lang="python")

        assert result == {"args": ("positional",), "kwargs": {"lang": "python"}}

    def test_capabilities_default_to_empty(self) -> None:
        """Capabilities default to empty."""
        assert _MinimalPlugin.capabilities == frozenset()

    def test_supports_and_require(self) -> None:
        """Supports and require."""
        class Capable(_MinimalPlugin):
            capabilities: ClassVar[frozenset[BackendCapability]] = frozenset({BackendCapability.ARTIFACTS})

        assert Capable.supports(BackendCapability.ARTIFACTS)
        assert not Capable.supports(BackendCapability.POOLING)
        Capable.require(BackendCapability.ARTIFACTS)

        with pytest.raises(BackendCapabilityError):
            Capable.require(BackendCapability.POOLING)

    def test_optional_factories_raise_by_default(self) -> None:
        """Optional factories raise by default."""
        with pytest.raises(BackendCapabilityError, match="pooling"):
            _MinimalPlugin.create_pool_manager()

        with pytest.raises(BackendCapabilityError, match="interactive"):
            _MinimalPlugin.create_interactive_session()

    def test_capability_values_are_stable(self) -> None:
        """Capability strings appear in plugin source; they are contract."""
        assert {capability.value for capability in BackendCapability} == {
            "artifacts",
            "interactive",
            "pooling",
            "existing_container",
        }


class TestSandboxBackendBase:
    """The public base class bridges BaseSession's private contract."""

    def test_builds_config_from_kwargs(self) -> None:
        """Builds config from kwargs."""
        session = _StubSession(lang="python", workdir="/tmp/example", verbose=True)

        assert isinstance(session.config, SessionConfig)
        assert session.config.workdir == "/tmp/example"
        assert session.config.verbose is True

    def test_accepts_a_prebuilt_config(self) -> None:
        """Accepts a prebuilt config."""
        config = SessionConfig(workdir="/tmp/prebuilt")
        session = _StubSession(config=config)

        assert session.config is config

    def test_unrecognised_kwargs_reach_base_session(self) -> None:
        """Unrecognised kwargs reach base session."""
        session = _StubSession(lang="python", libraries=["numpy"])

        assert session._initial_libraries == ["numpy"]

    def test_stream_defaults_to_false(self) -> None:
        """Stream defaults to false."""
        assert _StubSession().stream is False
        assert _StubSession(stream=True).stream is True

    @pytest.mark.parametrize(
        ("private", "public", "args"),
        [
            ("_handle_timeout", "handle_timeout", ()),
            ("_ensure_directory_exists", "ensure_directory_exists", ("/tmp/dir",)),
            ("_ensure_ownership", "ensure_ownership", (["/tmp/a"],)),
            ("_process_non_stream_output", "process_non_stream_output", ("payload",)),
        ],
    )
    def test_private_hooks_bridge_to_public_ones(self, private: str, public: str, args: tuple) -> None:
        """Private hooks bridge to public ones."""
        session = _StubSession()

        getattr(session, private)(*args)

        assert session.calls[-1][0] == public

    def test_stream_bridge_forwards_callbacks(self) -> None:
        """Stream bridge forwards callbacks."""
        session = _StubSession()

        def on_stdout(_chunk: str) -> None: ...

        result = session._process_stream_output("payload", on_stdout=on_stdout)

        assert result == ("sout", "serr")
        name, (output, stdout_cb, stderr_cb) = session.calls[-1]
        assert name == "process_stream_output"
        assert output == "payload"
        assert stdout_cb is on_stdout
        assert stderr_cb is None

    def test_optional_hooks_raise_when_not_overridden(self) -> None:
        """Optional hooks raise when not overridden."""
        session = _StubSession()

        with pytest.raises(BackendCapabilityError, match="artifacts"):
            session.get_archive("/tmp/whatever")

        with pytest.raises(BackendCapabilityError, match="existing_container"):
            session.connect_to_existing_container("abc123")

    def test_required_hooks_are_abstract(self) -> None:
        """Required hooks are abstract."""
        expected = {
            "handle_timeout",
            "ensure_directory_exists",
            "ensure_ownership",
            "process_non_stream_output",
            "process_stream_output",
            "open",
            "close",
        }
        assert expected <= set(SandboxBackendBase.__abstractmethods__)


class TestExceptionHierarchy:
    """New errors must not break `except UnsupportedBackendError` downstream."""

    @pytest.mark.parametrize(
        "error",
        [
            BackendNotFoundError("x", "message"),
            BackendLoadError("x", "message"),
            BackendNameConflictError("x", "message"),
            BackendCapabilityError("x", capability="pooling"),
        ],
    )
    def test_all_subclass_unsupported_backend_error(self, error: Exception) -> None:
        """All subclass unsupported backend error."""
        assert isinstance(error, UnsupportedBackendError)
        assert isinstance(error, SandboxError)

    def test_unsupported_backend_error_message_unchanged(self) -> None:
        """Pinned by tests/test_exceptions.py; restated here because plugins depend on it."""
        assert str(UnsupportedBackendError("invalid_backend")) == "Unsupported backend: invalid_backend"

    def test_backend_attribute_is_exposed(self) -> None:
        """Backend attribute is exposed."""
        assert UnsupportedBackendError("tenki").backend == "tenki"
        assert BackendNotFoundError("tenki", "msg").backend == "tenki"

    def test_load_error_records_distribution(self) -> None:
        """Load error records distribution."""
        error = BackendLoadError("tenki", "msg", "llm-sandbox-tenki")
        assert error.distribution == "llm-sandbox-tenki"

    def test_conflict_error_records_distributions(self) -> None:
        """Conflict error records distributions."""
        error = BackendNameConflictError("tenki", "msg", ("a", "b"))
        assert error.distributions == ("a", "b")


class TestPublicSurface:
    """Names plugin authors depend on must be importable from documented paths."""

    def test_backends_module_exports(self) -> None:
        """Backends module exports."""
        module = importlib.import_module("llm_sandbox.backends")

        for name in module.__all__:
            assert hasattr(module, name), f"llm_sandbox.backends.__all__ lists missing name {name!r}"

    def test_top_level_exports(self) -> None:
        """Top level exports."""
        for name in llm_sandbox.__all__:
            assert hasattr(llm_sandbox, name), f"llm_sandbox.__all__ lists missing name {name!r}"

    @pytest.mark.parametrize(
        "name",
        [
            "list_backends",
            "BackendNotFoundError",
            "BackendLoadError",
            "BackendNameConflictError",
            "BackendCapabilityError",
            "BackendShadowWarning",
            "BackendNameMismatchWarning",
            "UnsupportedBackendError",
            "SandboxTimeoutError",
            "NotOpenSessionError",
            "CommandFailedError",
            "MissingDependencyError",
            "LibraryInstallationNotSupportedError",
        ],
    )
    def test_plugin_facing_names_are_top_level(self, name: str) -> None:
        """Plugin facing names are top level."""
        assert name in llm_sandbox.__all__
        assert hasattr(llm_sandbox, name)

    @pytest.mark.parametrize(
        "name",
        [
            "SandboxBackendPlugin",
            "SandboxBackendBase",
            "BackendCapability",
            "BackendInfo",
            "BaseSession",
            "ContainerAPI",
            "PLUGIN_API_VERSION",
            "SUPPORTED_PLUGIN_API_VERSIONS",
            "ENTRY_POINT_GROUP",
            "normalize_backend_name",
        ],
    )
    def test_plugin_authoring_names_are_in_backends(self, name: str) -> None:
        """Plugin authoring names are in backends."""
        module = importlib.import_module("llm_sandbox.backends")
        assert name in module.__all__
        assert hasattr(module, name)

    def test_backend_info_is_frozen(self) -> None:
        """Backend info is frozen."""
        info = BackendInfo(name="x")
        with pytest.raises(dataclasses.FrozenInstanceError):
            info.name = "y"  # type: ignore[misc]
