# ruff: noqa: PLR2004, ARG002, SLF001
"""Tests for llm_sandbox.base Session timeout functionality - simplified."""

import time

import pytest

from llm_sandbox.base import Session
from llm_sandbox.data import ConsoleOutput
from llm_sandbox.exceptions import SandboxTimeoutError


class SimpleSession(Session):
    """Concrete implementation for testing without complex features."""

    def open(self) -> None:
        """Open the session."""
        super().open()

    def close(self) -> None:
        """Close the session."""
        super().close()

    def run(self, code: str, libraries: list | None = None, timeout: float | None = None) -> ConsoleOutput:
        """Run code."""
        return ConsoleOutput(exit_code=0, stdout="output", stderr="")

    def copy_to_runtime(self, src: str, dest: str) -> None:
        """Copy to runtime."""

    def copy_from_runtime(self, src: str, dest: str) -> None:
        """Copy from runtime."""

    def execute_command(self, command: str, workdir: str | None = None) -> ConsoleOutput:
        """Execute command."""
        return ConsoleOutput(exit_code=0, stdout="command output", stderr="")

    def get_archive(self, path: str) -> tuple[bytes, dict]:
        """Get archive."""
        return b"archive_data", {"size": 100}

    def _ensure_ownership(self, folders: list[str]) -> None:
        """Ensure ownership."""


class TestSessionTimeout:
    """Test Session timeout functionality."""

    def test_default_timeout_fallback(self) -> None:
        """Test default timeout fallback when None is provided."""
        session = SimpleSession(lang="python", default_timeout=None)
        assert session.default_timeout == 30.0

    def test_execution_timeout_uses_default(self) -> None:
        """Test execution timeout uses default when not specified."""
        session = SimpleSession(lang="python", default_timeout=25.0, execution_timeout=None)
        assert session.execution_timeout == 25.0

    def test_timeout_context_none(self) -> None:
        """Test timeout context with None timeout."""
        session = SimpleSession(lang="python")

        with session._timeout_context(timeout=None):
            pass  # Should complete without timeout

    def test_session_timeout_check_no_timeout(self) -> None:
        """Test session timeout check when no timeout is set."""
        session = SimpleSession(lang="python", session_timeout=None)
        session._session_start_time = time.time()
        session._check_session_timeout()  # Should not raise

    def test_session_timeout_check_no_start_time(self) -> None:
        """Test session timeout check when no start time."""
        session = SimpleSession(lang="python", session_timeout=10.0)
        session._session_start_time = None
        session._check_session_timeout()  # Should not raise

    def test_session_timeout_exceeded(self) -> None:
        """Test session timeout when exceeded."""
        session = SimpleSession(lang="python", session_timeout=0.1)
        session._session_start_time = time.time() - 0.2  # Past timeout

        with pytest.raises(SandboxTimeoutError):
            session._check_session_timeout()

    def test_start_timer_no_timeout(self) -> None:
        """Test starting timer when no session timeout."""
        session = SimpleSession(lang="python", session_timeout=None)
        session._start_session_timer()

        assert session._session_start_time is not None
        assert session._session_timer is None

    def test_start_timer_with_timeout(self) -> None:
        """Test starting timer with session timeout."""
        session = SimpleSession(lang="python", session_timeout=60.0)
        session._start_session_timer()

        assert session._session_start_time is not None
        assert session._session_timer is not None

        session._stop_session_timer()

    def test_stop_timer_no_timer(self) -> None:
        """Test stopping timer when none exists."""
        session = SimpleSession(lang="python")
        session._session_timer = None
        session._stop_session_timer()  # Should not raise

    def test_verbose_logging_disabled(self) -> None:
        """Test logging when verbose is disabled."""
        session = SimpleSession(lang="python", verbose=False)
        # This should not cause any output
        session._log("test message")

    def test_execute_commands_basic(self) -> None:
        """Test execute_commands method."""
        session = SimpleSession(lang="python")
        result = session.execute_commands(["ls"])
        assert result.exit_code == 0

    def test_context_manager(self) -> None:
        """Test context manager functionality."""
        session = SimpleSession(lang="python")

        with session as s:
            assert s is session

        # Should complete without errors
