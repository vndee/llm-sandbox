"""Reusable conformance suites a backend plugin can run against itself.

This is the main lever core has for keeping third-party backend quality up without
reviewing anyone's code, so the checks are the ones that actually catch broken backends:
resources released when a block raises, timeouts that really cancel work, exit codes that
survive, and file transfer that round-trips.
"""

import contextlib
import uuid
from pathlib import Path
from typing import Any, ClassVar

try:
    import pytest
except ImportError as exc:  # pragma: no cover - depends on the installing environment
    msg = (
        "llm_sandbox.testing requires pytest, which llm-sandbox does not depend on at runtime. "
        "Install it with: pip install llm-sandbox[testing]"
    )
    raise ImportError(msg) from exc

from llm_sandbox.backends.plugin import (
    SUPPORTED_PLUGIN_API_VERSIONS,
    BackendCapability,
    SandboxBackendPlugin,
    normalize_backend_name,
)
from llm_sandbox.data import ConsoleOutput
from llm_sandbox.exceptions import NotOpenSessionError, SandboxTimeoutError, UnsupportedBackendError
from llm_sandbox.registry import get_backend
from llm_sandbox.session import create_session

_REQUIRED_SESSION_METHODS = (
    "open",
    "close",
    "run",
    "execute_command",
    "copy_to_runtime",
    "copy_from_runtime",
)

_EXPECTED_EXIT_CODE = 3


class BackendInterfaceComplianceTests:
    """Static conformance checks. Creates no sessions and needs no infrastructure.

    Subclass it and set `backend`. Everything else has a working default.

    Example:
        ```python
        from llm_sandbox.testing import BackendInterfaceComplianceTests


        class TestMyServiceInterface(BackendInterfaceComplianceTests):
            backend = "myservice"
        ```

    """

    backend: ClassVar[str]
    """The backend name, as passed to ``backend=``. Required."""

    expect_registered_entry_point: ClassVar[bool] = True
    """Whether the backend must be discoverable through an installed entry point.

    Set False when testing a backend registered in-process (for example one injected by a
    fixture) rather than installed as a distribution.
    """

    @pytest.fixture
    def provider(self) -> type[SandboxBackendPlugin]:
        """Resolve the backend under test.

        Returns:
            type[SandboxBackendPlugin]: The provider registered for `backend`.

        """
        return get_backend(self.backend)

    def test_backend_resolves(self, provider: type[SandboxBackendPlugin]) -> None:
        """The backend name resolves to a SandboxBackendPlugin subclass."""
        assert isinstance(provider, type), (
            f"Backend {self.backend!r} resolved to an instance, not a class. The entry point must "
            f"point at the SandboxBackendPlugin subclass itself."
        )
        assert issubclass(provider, SandboxBackendPlugin), (
            f"Backend {self.backend!r} must subclass llm_sandbox.backends.SandboxBackendPlugin."
        )

    def test_declares_supported_plugin_api_version(self, provider: type[SandboxBackendPlugin]) -> None:
        """The backend declares a plugin API version this release accepts."""
        declared = getattr(provider, "PLUGIN_API_VERSION", None)
        assert declared is not None, "Backend must declare PLUGIN_API_VERSION."
        assert declared in SUPPORTED_PLUGIN_API_VERSIONS, (
            f"Backend targets plugin API version {declared}; this release accepts "
            f"{sorted(SUPPORTED_PLUGIN_API_VERSIONS)}."
        )

    def test_declares_matching_name(self, provider: type[SandboxBackendPlugin]) -> None:
        """The declared name matches the name the backend is addressed by."""
        declared = getattr(provider, "name", None)
        assert declared, "Backend must declare a `name` class attribute."
        assert normalize_backend_name(str(declared)) == normalize_backend_name(self.backend), (
            f"Backend declares name {declared!r} but is registered as {self.backend!r}."
        )

    def test_capabilities_are_recognised(self, provider: type[SandboxBackendPlugin]) -> None:
        """Every declared capability is a real BackendCapability."""
        valid = {capability.value for capability in BackendCapability}
        for capability in provider.capabilities:
            assert str(capability) in valid, f"Unknown capability {capability!r}. Valid capabilities: {sorted(valid)}."

    def test_name_normalisation_is_stable(self, provider: type[SandboxBackendPlugin]) -> None:
        """The backend resolves identically through hyphen, underscore, and case variants."""
        base = normalize_backend_name(self.backend)
        for variant in (base, base.replace("_", "-"), base.upper()):
            assert get_backend(variant) is provider, (
                f"Backend did not resolve identically for the name variant {variant!r}."
            )

    def test_registered_as_entry_point(self, provider: type[SandboxBackendPlugin]) -> None:
        """The backend is discoverable through the ``llm_sandbox.backends`` entry point group."""
        if not self.expect_registered_entry_point:
            pytest.skip("expect_registered_entry_point is False")

        import llm_sandbox

        names = {info.name for info in llm_sandbox.list_backends()}
        assert normalize_backend_name(self.backend) in names, (
            f"Backend {self.backend!r} is not listed by llm_sandbox.list_backends(). Check the "
            f'[project.entry-points."llm_sandbox.backends"] table in pyproject.toml, and that the '
            f"distribution is installed."
        )
        assert provider is not None

    def test_unsupported_capabilities_raise(self, provider: type[SandboxBackendPlugin]) -> None:
        """Undeclared optional factories raise rather than half-working."""
        if not provider.supports(BackendCapability.POOLING):
            with pytest.raises(UnsupportedBackendError) as excinfo:
                provider.create_pool_manager()
            assert "pool" in str(excinfo.value).lower()

        if not provider.supports(BackendCapability.INTERACTIVE):
            with pytest.raises(UnsupportedBackendError) as excinfo:
                provider.create_interactive_session()
            assert "interactive" in str(excinfo.value).lower()


class BackendComplianceTests(BackendInterfaceComplianceTests):
    """Full conformance suite: interface checks plus live session behaviour.

    Subclass it, set `backend`, and adjust `session_kwargs` if your backend needs
    configuration. The code snippets default to Python; override them for a backend whose
    default language is something else.

    Every test creates its own session and closes it, so a failure never leaks a container
    into the next test.

    Example:
        ```python
        from llm_sandbox.testing import BackendComplianceTests


        class TestMyServiceCompliance(BackendComplianceTests):
            backend = "myservice"
            session_kwargs = {"lang": "python", "api_key": "test-key"}
        ```

    """

    session_kwargs: ClassVar[dict[str, Any]] = {}
    """Extra keyword arguments passed to every session created by the suite."""

    hello_code: ClassVar[str] = "print('llm-sandbox-compliance')"
    """Code that writes `hello_expected` to stdout and exits 0."""

    hello_expected: ClassVar[str] = "llm-sandbox-compliance"
    """Text `hello_code` is expected to produce on stdout."""

    failing_code: ClassVar[str] = "import sys; sys.exit(3)"
    """Code that must terminate with exit code 3."""

    slow_code: ClassVar[str] = "import time; time.sleep(30)"
    """Code that runs long enough to be cancelled by a 1-second timeout."""

    timeout_seconds: ClassVar[float] = 1.0
    """Timeout used by the timeout test."""

    scratch_dir: ClassVar[str | None] = None
    """Directory inside the sandbox to write scratch files to.

    Defaults to the session's own ``workdir``. The kit deliberately does not assume ``/tmp``
    exists or is writable in your sandbox -- a minimal image or a non-POSIX runtime may have
    neither -- and for a backend that executes on the host, a predictable shared path is a
    collision between concurrent runs and a symlink target on a shared machine. Override this
    only if your sandbox needs scratch files somewhere specific.
    """

    def scratch_path(self, session: Any, name: str) -> str:
        """Build a unique scratch path inside the sandbox.

        Args:
            session (Any): The open session the path will be used with.
            name (str): A short label distinguishing this file from others.

        Returns:
            str: An absolute path inside the sandbox, unique to this call.

        """
        base = self.scratch_dir or getattr(getattr(session, "config", None), "workdir", "") or "."
        return f"{base.rstrip('/')}/llm_sandbox_{name}_{uuid.uuid4().hex}.txt"

    def make_session(self, **overrides: Any) -> Any:
        """Create an unopened session for the backend under test.

        Args:
            **overrides: Keyword arguments merged over `session_kwargs`.

        Returns:
            Any: The session, not yet opened.

        """
        return create_session(backend=self.backend, **{**self.session_kwargs, **overrides})

    # ------------------------------------------------------------------ #
    # Interface completeness
    # ------------------------------------------------------------------ #

    def test_session_exposes_required_methods(self) -> None:
        """The session implements every method in the required interface."""
        session = self.make_session()
        try:
            missing = [name for name in _REQUIRED_SESSION_METHODS if not callable(getattr(session, name, None))]
            assert not missing, (
                f"Session is missing required method(s): {', '.join(missing)}. See the required "
                f"interface in docs/plugins/authoring.md."
            )
        finally:
            with contextlib.suppress(Exception):
                session.close()

    # ------------------------------------------------------------------ #
    # Lifecycle
    # ------------------------------------------------------------------ #

    def test_context_manager_runs_code(self) -> None:
        """A session used as a context manager executes code and reports success."""
        with self.make_session() as session:
            result = session.run(self.hello_code)

        assert result.exit_code == 0, f"Expected exit code 0, got {result.exit_code}. stderr: {result.stderr}"
        assert self.hello_expected in result.stdout, (
            f"Expected {self.hello_expected!r} in stdout, got {result.stdout!r}"
        )

    def test_run_before_open_raises(self) -> None:
        """Running before the session is opened raises NotOpenSessionError."""
        session = self.make_session()
        try:
            with pytest.raises(NotOpenSessionError):
                session.run(self.hello_code)
        finally:
            with contextlib.suppress(Exception):
                session.close()

    def test_cleanup_on_exception(self) -> None:
        """An exception inside the context manager still closes the session.

        This is the check most worth having. A backend that leaks a container when the body
        raises will pass every happy-path test and quietly cost its users money.
        """
        session = self.make_session()

        def use_session_then_fail() -> None:
            with session:
                session.run(self.hello_code)
                msg = "compliance sentinel"
                raise RuntimeError(msg)

        with pytest.raises(RuntimeError, match="compliance sentinel"):
            use_session_then_fail()

        assert session.is_open is False, (
            "Session remained open after an exception propagated out of the context manager. "
            "close() must release resources on the error path too."
        )

    def test_close_is_idempotent(self) -> None:
        """Closing an already-closed session does not raise."""
        session = self.make_session()
        with session:
            pass
        session.close()

    def test_close_without_open_does_not_raise(self) -> None:
        """Closing a session that was never opened does not raise."""
        self.make_session().close()

    # ------------------------------------------------------------------ #
    # Result types
    # ------------------------------------------------------------------ #

    def test_run_returns_console_output(self) -> None:
        """run() returns a ConsoleOutput with correctly typed fields."""
        with self.make_session() as session:
            result = session.run(self.hello_code)

        assert isinstance(result, ConsoleOutput), (
            f"run() must return llm_sandbox.ConsoleOutput (or a subclass such as ExecutionResult), "
            f"got {type(result).__name__}."
        )
        assert isinstance(result.exit_code, int)
        assert isinstance(result.stdout, str)
        assert isinstance(result.stderr, str)
        assert result.success() is True

    def test_nonzero_exit_code_is_reported(self) -> None:
        """A failing program's exit code reaches the caller instead of being swallowed."""
        with self.make_session() as session:
            result = session.run(self.failing_code)

        assert result.exit_code == _EXPECTED_EXIT_CODE, (
            f"Expected exit code {_EXPECTED_EXIT_CODE}, got {result.exit_code}. Exit codes must be "
            f"propagated, not normalised to 0 or 1."
        )
        assert result.success() is False

    def test_execute_command_returns_console_output(self) -> None:
        """execute_command() returns a ConsoleOutput and reports the command's output."""
        with self.make_session() as session:
            result = session.execute_command("echo llm-sandbox-compliance")

        assert isinstance(result, ConsoleOutput)
        assert result.exit_code == 0
        assert "llm-sandbox-compliance" in result.stdout

    # ------------------------------------------------------------------ #
    # File transfer
    # ------------------------------------------------------------------ #

    def test_file_round_trip(self, tmp_path: Path) -> None:
        """A file copied into the sandbox can be copied back out unchanged."""
        payload = f"compliance-{uuid.uuid4().hex}"
        source = tmp_path / "payload.txt"
        source.write_text(payload)
        destination = tmp_path / "returned.txt"

        with self.make_session() as session:
            remote = self.scratch_path(session, "roundtrip")
            session.copy_to_runtime(str(source), remote)
            session.copy_from_runtime(remote, str(destination))

        assert destination.read_text() == payload, "File contents changed in transit."

    def test_copy_to_runtime_rejects_missing_source(self, tmp_path: Path) -> None:
        """Copying a nonexistent file raises FileNotFoundError."""
        with self.make_session() as session, pytest.raises(FileNotFoundError):
            session.copy_to_runtime(str(tmp_path / "does-not-exist.txt"), self.scratch_path(session, "missing"))

    # ------------------------------------------------------------------ #
    # Declared capabilities
    # ------------------------------------------------------------------ #

    def test_artifacts_capability_works(self, tmp_path: Path) -> None:
        """A backend declaring ARTIFACTS can actually read a file back out as a tar archive.

        Checked behaviourally rather than structurally: `create_session` is a factory, so
        there is no session class to introspect, and a method that exists but raises is not
        the same as artifact support.
        """
        provider = get_backend(self.backend)
        if not provider.supports(BackendCapability.ARTIFACTS):
            pytest.skip("Backend does not declare the artifacts capability")

        source = tmp_path / "artifact.txt"
        source.write_text("artifact-payload")

        with self.make_session() as session:
            remote = self.scratch_path(session, "artifact")
            session.copy_to_runtime(str(source), remote)
            data, stat = session.get_archive(remote)

        assert isinstance(data, bytes), "get_archive() must return tar bytes."
        assert data, "get_archive() returned no data for a file that exists."
        assert stat.get("size", 0) > 0, (
            "get_archive() must report a non-zero size; callers treat size 0 as 'not found'."
        )

    def test_existing_container_capability_implies_connect(self) -> None:
        """A backend declaring EXISTING_CONTAINER really implements the attach hook."""
        provider = get_backend(self.backend)
        if not provider.supports(BackendCapability.EXISTING_CONTAINER):
            pytest.skip("Backend does not declare the existing_container capability")

        from llm_sandbox.backends.base import SandboxBackendBase

        session = self.make_session()
        try:
            session_cls = type(session)
            override = getattr(session_cls, "connect_to_existing_container", None) or getattr(
                session_cls, "_connect_to_existing_container", None
            )
            assert override is not None, (
                "Declaring EXISTING_CONTAINER requires a connect_to_existing_container() method."
            )
            assert override is not SandboxBackendBase.connect_to_existing_container, (
                "Backend declares EXISTING_CONTAINER but inherits the default, which always raises."
            )
        finally:
            with contextlib.suppress(Exception):
                session.close()

    # ------------------------------------------------------------------ #
    # Timeout
    # ------------------------------------------------------------------ #

    def test_timeout_raises_sandbox_timeout_error(self) -> None:
        """Code exceeding its timeout raises SandboxTimeoutError."""
        with self.make_session() as session, pytest.raises(SandboxTimeoutError):
            session.run(self.slow_code, timeout=self.timeout_seconds)

    def test_session_usable_after_timeout_cleanup(self) -> None:
        """A timeout leaves the session closable without raising.

        Backends kill the runtime to cancel work, so the session need not still be usable --
        but tearing it down must not raise on top of the timeout.
        """
        session = self.make_session()
        session.open()
        try:
            with contextlib.suppress(SandboxTimeoutError):
                session.run(self.slow_code, timeout=self.timeout_seconds)
        finally:
            session.close()
