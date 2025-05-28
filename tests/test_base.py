# ruff: noqa: ARG002, SLF001, PLR2004

"""Tests for base session functionality."""

import logging
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from llm_sandbox.base import Session
from llm_sandbox.const import SupportedLanguage
from llm_sandbox.data import ConsoleOutput
from llm_sandbox.exceptions import (
    CommandFailedError,
    InvalidRegexPatternError,
    LanguageHandlerNotInitializedError,
    LibraryInstallationNotSupportedError,
    SecurityViolationError,
)
from llm_sandbox.security import DangerousModule, SecurityIssueSeverity, SecurityPattern, SecurityPolicy


class ConcreteSession(Session):
    """Concrete implementation of Session for testing."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """Initialize the session."""
        super().__init__(*args, **kwargs)
        self.is_open = False

    def open(self) -> None:
        """Open the session."""
        self.is_open = True

    def close(self) -> None:
        """Close the session."""
        self.is_open = False

    def run(self, code: str, libraries: list | None = None) -> ConsoleOutput:
        """Run the code."""
        return ConsoleOutput(exit_code=0, stdout="output", stderr="")

    def copy_to_runtime(self, src: str, dest: str) -> None:
        """Copy to runtime."""

    def copy_from_runtime(self, src: str, dest: str) -> None:
        """Copy from runtime."""

    def execute_command(self, command: str, workdir: str | None = None) -> ConsoleOutput:
        """Execute a command."""
        return ConsoleOutput(exit_code=0, stdout="command output", stderr="")

    def get_archive(self, path: str) -> tuple[bytes, dict]:
        """Get an archive."""
        return b"archive data", {"size": 100}


class TestSessionInit:
    """Test Session initialization."""

    @patch("llm_sandbox.language_handlers.factory.LanguageHandlerFactory.create_handler")
    def test_init_with_defaults(self, mock_create_handler: MagicMock) -> None:
        """Test Session initialization with default parameters."""
        mock_handler = MagicMock()
        mock_create_handler.return_value = mock_handler

        session = ConcreteSession(lang="python")

        assert session.lang == "python"
        assert session.verbose is True
        assert session.image is None
        assert session.keep_template is False
        assert session.workdir == "/sandbox"
        assert session.security_policy is None
        assert session.language_handler == mock_handler
        mock_create_handler.assert_called_once_with("python", session.logger)

    @patch("llm_sandbox.language_handlers.factory.LanguageHandlerFactory.create_handler")
    def test_init_with_custom_params(self, mock_create_handler: MagicMock) -> None:
        """Test Session initialization with custom parameters."""
        mock_handler = MagicMock()
        mock_create_handler.return_value = mock_handler
        custom_logger = logging.getLogger("custom")
        security_policy = SecurityPolicy(patterns=[], restricted_modules=[])

        session = ConcreteSession(
            lang="java",
            verbose=False,
            image="custom-image",
            keep_template=True,
            logger=custom_logger,
            workdir="/custom",
            security_policy=security_policy,
        )

        assert session.lang == "java"
        assert session.verbose is False
        assert session.image == "custom-image"
        assert session.keep_template is True
        assert session.workdir == "/custom"
        assert session.security_policy == security_policy
        assert session.logger == custom_logger
        mock_create_handler.assert_called_once_with("java", custom_logger)


class TestSessionLogging:
    """Test Session logging functionality."""

    @patch("llm_sandbox.language_handlers.factory.LanguageHandlerFactory.create_handler")
    def test_log_with_verbose_enabled(self, mock_create_handler: MagicMock) -> None:
        """Test logging when verbose is enabled."""
        mock_handler = MagicMock()
        mock_create_handler.return_value = mock_handler
        mock_logger = MagicMock()

        session = ConcreteSession(lang="python", verbose=True, logger=mock_logger)

        session._log("test message")
        mock_logger.info.assert_called_once_with("test message")

        session._log("warning message", "warning")
        mock_logger.warning.assert_called_once_with("warning message")

    @patch("llm_sandbox.language_handlers.factory.LanguageHandlerFactory.create_handler")
    def test_log_with_verbose_disabled(self, mock_create_handler: MagicMock) -> None:
        """Test logging when verbose is disabled."""
        mock_handler = MagicMock()
        mock_create_handler.return_value = mock_handler
        mock_logger = MagicMock()

        session = ConcreteSession(lang="python", verbose=False, logger=mock_logger)

        session._log("test message")
        mock_logger.info.assert_not_called()


class TestSessionCommands:
    """Test Session command execution functionality."""

    @patch("llm_sandbox.language_handlers.factory.LanguageHandlerFactory.create_handler")
    def test_execute_commands_success(self, mock_create_handler: MagicMock) -> None:
        """Test successful execution of multiple commands."""
        mock_handler = MagicMock()
        mock_create_handler.return_value = mock_handler

        session = ConcreteSession(lang="python")

        with patch.object(session, "execute_command") as mock_exec:
            mock_exec.return_value = ConsoleOutput(exit_code=0, stdout="output", stderr="")

            commands: list[str | tuple[str, str | None]] = ["ls -l", ("mkdir test", "/tmp")]
            result = session.execute_commands(commands)

            assert result.exit_code == 0
            assert mock_exec.call_count == 2
            mock_exec.assert_any_call("ls -l", workdir=None)
            mock_exec.assert_any_call("mkdir test", workdir="/tmp")

    @patch("llm_sandbox.language_handlers.factory.LanguageHandlerFactory.create_handler")
    def test_execute_commands_failure(self, mock_create_handler: MagicMock) -> None:
        """Test command execution failure."""
        mock_handler = MagicMock()
        mock_create_handler.return_value = mock_handler

        session = ConcreteSession(lang="python")

        with patch.object(session, "execute_command") as mock_exec:
            mock_exec.return_value = ConsoleOutput(exit_code=1, stdout="", stderr="error")

            commands: list[str | tuple[str, str | None]] = ["failing_command"]

            with pytest.raises(CommandFailedError):
                session.execute_commands(commands)


class TestSessionLibraryInstallation:
    """Test Session library installation functionality."""

    @patch("llm_sandbox.language_handlers.factory.LanguageHandlerFactory.create_handler")
    def test_install_with_empty_libraries(self, mock_create_handler: MagicMock) -> None:
        """Test install with empty or None libraries list."""
        mock_handler = MagicMock()
        mock_create_handler.return_value = mock_handler

        session = ConcreteSession(lang="python")

        # Should return early without doing anything
        session.install(None)
        session.install([])

    @patch("llm_sandbox.language_handlers.factory.LanguageHandlerFactory.create_handler")
    def test_install_unsupported_language(self, mock_create_handler: MagicMock) -> None:
        """Test install with language that doesn't support library installation."""
        mock_handler = MagicMock()
        mock_handler.is_support_library_installation = False
        mock_create_handler.return_value = mock_handler

        session = ConcreteSession(lang="unsupported")

        with pytest.raises(LibraryInstallationNotSupportedError):
            session.install(["numpy"])

    @patch("llm_sandbox.language_handlers.factory.LanguageHandlerFactory.create_handler")
    def test_install_with_security_violation(self, mock_create_handler: MagicMock) -> None:
        """Test install with libraries blocked by security policy."""
        mock_handler = MagicMock()
        mock_handler.is_support_library_installation = True
        mock_create_handler.return_value = mock_handler

        restricted_modules = [DangerousModule(name="os", description="OS module", severity=SecurityIssueSeverity.HIGH)]
        security_policy = SecurityPolicy(patterns=[], restricted_modules=restricted_modules)

        session = ConcreteSession(lang="python", security_policy=security_policy)

        with pytest.raises(SecurityViolationError, match="Library os is not allowed to be installed"):
            session.install(["os"])

    @patch("llm_sandbox.language_handlers.factory.LanguageHandlerFactory.create_handler")
    def test_install_success(self, mock_create_handler: MagicMock) -> None:
        """Test successful library installation."""
        mock_handler = MagicMock()
        mock_handler.is_support_library_installation = True
        mock_handler.get_library_installation_command.side_effect = lambda lib: f"pip install {lib}"
        mock_create_handler.return_value = mock_handler

        session = ConcreteSession(lang="python")

        with patch.object(session, "execute_commands") as mock_exec:
            session.install(["numpy", "pandas"])

            expected_commands = [
                ("pip install numpy", "/sandbox"),
                ("pip install pandas", "/sandbox"),
            ]
            mock_exec.assert_called_once_with(expected_commands)


class TestSessionEnvironmentSetup:
    """Test Session environment setup functionality."""

    @patch("llm_sandbox.language_handlers.factory.LanguageHandlerFactory.create_handler")
    def test_environment_setup_python(self, mock_create_handler: MagicMock) -> None:
        """Test environment setup for Python."""
        mock_handler = MagicMock()
        mock_handler.name = SupportedLanguage.PYTHON
        mock_create_handler.return_value = mock_handler

        session = ConcreteSession(lang="python")

        with (
            patch.object(session, "execute_commands") as mock_exec,
            patch.object(session, "_ensure_ownership") as mock_ownership,
        ):
            session.environment_setup()

            # Should call execute_commands multiple times for Python setup
            assert mock_exec.call_count >= 2
            mock_ownership.assert_called_once_with(["/tmp/venv", "/tmp/pip_cache"])

    @patch("llm_sandbox.language_handlers.factory.LanguageHandlerFactory.create_handler")
    def test_environment_setup_go(self, mock_create_handler: MagicMock) -> None:
        """Test environment setup for Go."""
        mock_handler = MagicMock()
        mock_handler.name = SupportedLanguage.GO
        mock_create_handler.return_value = mock_handler

        session = ConcreteSession(lang="go")

        with patch.object(session, "execute_commands") as mock_exec:
            session.environment_setup()

            # Should call execute_commands for Go setup
            assert mock_exec.call_count >= 2

    @patch("llm_sandbox.language_handlers.factory.LanguageHandlerFactory.create_handler")
    def test_environment_setup_other_language(self, mock_create_handler: MagicMock) -> None:
        """Test environment setup for other languages."""
        mock_handler = MagicMock()
        mock_handler.name = "java"
        mock_create_handler.return_value = mock_handler

        session = ConcreteSession(lang="java")

        with patch.object(session, "execute_commands") as mock_exec:
            session.environment_setup()

            # Should only create the workdir
            mock_exec.assert_called_once()


class TestSessionSecurity:
    """Test Session security functionality."""

    @patch("llm_sandbox.language_handlers.factory.LanguageHandlerFactory.create_handler")
    def test_check_security_policy_no_policy(self, mock_create_handler: MagicMock) -> None:
        """Test security check when no policy is set."""
        mock_handler = MagicMock()
        mock_create_handler.return_value = mock_handler

        session = ConcreteSession(lang="python")

        is_safe, violations = session._check_security_policy("print('hello')")

        assert is_safe is True
        assert violations == []

    @patch("llm_sandbox.language_handlers.factory.LanguageHandlerFactory.create_handler")
    def test_check_security_policy_no_handler(self, mock_create_handler: MagicMock) -> None:
        """Test security check when language handler is not initialized."""
        mock_handler = MagicMock()
        mock_create_handler.return_value = mock_handler
        security_policy = SecurityPolicy(patterns=[], restricted_modules=[])

        session = ConcreteSession(lang="python", security_policy=security_policy)
        session.language_handler = None  # type: ignore[assignment]

        with pytest.raises(LanguageHandlerNotInitializedError):
            session._check_security_policy("print('hello')")

    @patch("llm_sandbox.language_handlers.factory.LanguageHandlerFactory.create_handler")
    def test_check_security_policy_with_violations(self, mock_create_handler: MagicMock) -> None:
        """Test security check with policy violations."""
        mock_handler = MagicMock()
        mock_handler.filter_comments.return_value = "import os"
        mock_create_handler.return_value = mock_handler

        patterns = [
            SecurityPattern(
                pattern=r"import\s+os", description="OS import detected", severity=SecurityIssueSeverity.HIGH
            )
        ]
        security_policy = SecurityPolicy(
            patterns=patterns, restricted_modules=[], severity_threshold=SecurityIssueSeverity.MEDIUM
        )

        session = ConcreteSession(lang="python", security_policy=security_policy)

        is_safe, violations = session._check_security_policy("import os")

        assert is_safe is False
        assert len(violations) == 1
        assert violations[0].description == "OS import detected"

    @patch("llm_sandbox.language_handlers.factory.LanguageHandlerFactory.create_handler")
    def test_check_security_policy_with_restricted_modules(self, mock_create_handler: MagicMock) -> None:
        """Test security check with dangerous modules."""
        mock_handler = MagicMock()
        mock_handler.filter_comments.return_value = "import subprocess"
        mock_handler.get_import_patterns.return_value = r"import\s+subprocess"
        mock_create_handler.return_value = mock_handler

        restricted_modules = [
            DangerousModule(name="subprocess", description="Subprocess module", severity=SecurityIssueSeverity.HIGH)
        ]
        security_policy = SecurityPolicy(
            patterns=[], restricted_modules=restricted_modules, severity_threshold=SecurityIssueSeverity.MEDIUM
        )

        session = ConcreteSession(lang="python", security_policy=security_policy)

        is_safe, violations = session._check_security_policy("import subprocess")

        assert is_safe is False
        assert len(violations) == 1

    @patch("llm_sandbox.language_handlers.factory.LanguageHandlerFactory.create_handler")
    def test_check_security_policy_invalid_regex(self, mock_create_handler: MagicMock) -> None:
        """Test security check with invalid regex pattern."""
        mock_handler = MagicMock()
        mock_handler.filter_comments.return_value = "import os"
        mock_create_handler.return_value = mock_handler

        # Test that invalid regex pattern raises an error during pattern creation
        with pytest.raises(InvalidRegexPatternError, match="Invalid regex pattern: \\[invalid regex"):
            SecurityPattern(
                pattern="[invalid regex",  # Invalid regex
                description="Invalid pattern",
                severity=SecurityIssueSeverity.HIGH,
            )

    @patch("llm_sandbox.language_handlers.factory.LanguageHandlerFactory.create_handler")
    def test_is_safe_delegates_to_check_security_policy(self, mock_create_handler: MagicMock) -> None:
        """Test that is_safe method delegates to _check_security_policy."""
        mock_handler = MagicMock()
        mock_create_handler.return_value = mock_handler

        session = ConcreteSession(lang="python")

        with patch.object(session, "_check_security_policy") as mock_check:
            mock_check.return_value = (True, [])

            result = session.is_safe("print('hello')")

            assert result == (True, [])
            mock_check.assert_called_once_with("print('hello')")


class TestSessionContextManager:
    """Test Session context manager functionality."""

    @patch("llm_sandbox.language_handlers.factory.LanguageHandlerFactory.create_handler")
    def test_context_manager_enter(self, mock_create_handler: MagicMock) -> None:
        """Test Session context manager enter."""
        mock_handler = MagicMock()
        mock_create_handler.return_value = mock_handler

        session = ConcreteSession(lang="python")

        result = session.__enter__()

        assert result == session
        assert session.is_open is True

    @patch("llm_sandbox.language_handlers.factory.LanguageHandlerFactory.create_handler")
    def test_context_manager_exit(self, mock_create_handler: MagicMock) -> None:
        """Test Session context manager exit."""
        mock_handler = MagicMock()
        mock_create_handler.return_value = mock_handler

        session = ConcreteSession(lang="python")
        session.is_open = True

        session.__exit__(None, None, None)

        assert session.is_open is False

    @patch("llm_sandbox.language_handlers.factory.LanguageHandlerFactory.create_handler")
    def test_context_manager_with_statement(self, mock_create_handler: MagicMock) -> None:
        """Test Session used in with statement."""
        mock_handler = MagicMock()
        mock_create_handler.return_value = mock_handler

        session = ConcreteSession(lang="python")

        with session as s:
            assert s == session

        assert session.is_open is False
