# ruff: noqa: E501, SLF001, ARG002, PLR2004, FBT003
"""Test cases for the new architecture BaseSession."""

import logging
from typing import Any
from unittest.mock import MagicMock, Mock, call, patch

import pytest

from llm_sandbox.const import SupportedLanguage
from llm_sandbox.core.config import SessionConfig
from llm_sandbox.core.mixins import ContainerAPI
from llm_sandbox.core.session_base import BaseSession
from llm_sandbox.data import ConsoleOutput
from llm_sandbox.exceptions import (
    LanguageHandlerNotInitializedError,
    LibraryInstallationNotSupportedError,
    NotOpenSessionError,
    SandboxTimeoutError,
    SecurityViolationError,
)
from llm_sandbox.language_handlers.factory import LanguageHandlerFactory
from llm_sandbox.security import RestrictedModule, SecurityIssueSeverity, SecurityPattern, SecurityPolicy


class MockLanguageHandler:
    """Mock language handler for testing."""

    def __init__(self, name: str = "python", file_extension: str = "py", supports_installation: bool = True) -> None:
        """Initialize mock language handler."""
        self.name = name
        self.file_extension = file_extension
        self._supports_installation = supports_installation

    @property
    def is_support_library_installation(self) -> bool:
        """Check if library installation is supported."""
        return self._supports_installation

    def get_library_installation_command(self, library: str) -> str:
        """Get library installation command."""
        return f"pip install {library}"

    def get_execution_commands(self, filename: str) -> list[str]:
        """Get execution commands."""
        return [f"python {filename}"]

    def filter_comments(self, code: str) -> str:
        """Filter comments."""
        return code

    def get_import_patterns(self, module_name: str) -> str:
        """Get import patterns."""
        return f"import {module_name}"


class MockBaseSession(BaseSession):
    """Concrete implementation of BaseSession for testing."""

    def __init__(self, config: SessionConfig, **kwargs: Any) -> None:
        """Initialize mock base session."""
        super().__init__(config, **kwargs)
        self.container_api = Mock(spec=ContainerAPI)

    def _handle_timeout(self) -> None:
        """Mock implementation of timeout handler."""

    def open(self) -> None:
        """Mock implementation of open."""
        super().open()

    def close(self) -> None:
        """Mock implementation of close."""
        super().close()

    def _ensure_directory_exists(self, path: str) -> None:
        """Mock implementation."""

    def _ensure_ownership(self, paths: list[str]) -> None:
        """Mock implementation."""

    def _process_non_stream_output(self, output: str) -> tuple[str, str]:
        """Mock implementation."""
        return "stdout", "stderr"

    def _process_stream_output(self, output: str) -> tuple[str, str]:
        """Mock implementation."""
        return "stream_stdout", "stream_stderr"

    def _connect_to_existing_container(self, pod_id: str) -> None:
        """Mock implementation."""


class TestBaseSessionInit:
    """Test BaseSession initialization."""

    @patch.object(LanguageHandlerFactory, "create_handler")
    def test_init_basic(self, mock_create_handler: MagicMock) -> None:
        """Test basic initialization."""
        mock_handler = MockLanguageHandler()
        mock_create_handler.return_value = mock_handler

        config = SessionConfig(lang=SupportedLanguage.PYTHON)
        session = MockBaseSession(config)

        assert session.config == config
        assert session.verbose == config.verbose
        assert session.language_handler == mock_handler
        assert session.container is None
        assert session.is_open is False
        assert session._session_start_time is None
        assert session._session_timer is None

    @patch.object(LanguageHandlerFactory, "create_handler")
    def test_init_with_verbose(self, mock_create_handler: MagicMock) -> None:
        """Test initialization with verbose logging."""
        mock_handler = MockLanguageHandler()
        mock_create_handler.return_value = mock_handler

        config = SessionConfig(lang=SupportedLanguage.PYTHON, verbose=True)
        session = MockBaseSession(config)

        assert session.verbose is True

    @patch.object(LanguageHandlerFactory, "create_handler")
    def test_init_creates_logger(self, mock_create_handler: MagicMock) -> None:
        """Test that logger is created properly."""
        mock_handler = MockLanguageHandler()
        mock_create_handler.return_value = mock_handler

        config = SessionConfig(lang=SupportedLanguage.PYTHON)
        session = MockBaseSession(config)

        assert isinstance(session.logger, logging.Logger)


class TestBaseSessionLogging:
    """Test BaseSession logging functionality."""

    def setup_method(self) -> None:
        """Set up test fixtures."""
        with patch.object(LanguageHandlerFactory, "create_handler") as mock_create_handler:
            mock_handler = MockLanguageHandler()
            mock_create_handler.return_value = mock_handler

            config = SessionConfig(lang=SupportedLanguage.PYTHON, verbose=True)
            self.session = MockBaseSession(config)
            self.session.logger = Mock()

    def test_log_verbose_enabled(self) -> None:
        """Test logging when verbose is enabled."""
        self.session._log("test message", "info")
        self.session.logger.info.assert_called_once_with("test message")

    def test_log_verbose_disabled(self) -> None:
        """Test logging when verbose is disabled."""
        self.session.verbose = False
        self.session._log("test message", "info")
        self.session.logger.info.assert_not_called()

    def test_log_different_levels(self) -> None:
        """Test logging with different levels."""
        self.session._log("warning message", "warning")
        self.session.logger.warning.assert_called_once_with("warning message")

        self.session._log("error message", "error")
        self.session.logger.error.assert_called_once_with("error message")


class TestBaseSessionTimeout:
    """Test session timeout functionality."""

    def test_session_timeout_with_cleanup_error(self) -> None:
        """Test session timeout when cleanup fails."""
        config = SessionConfig(session_timeout=0.1)  # Very short timeout
        session = MockBaseSession(config)

        # Mock close method to raise an exception
        with (
            patch.object(session, "close", side_effect=Exception("Cleanup failed")),
            patch.object(session, "_log") as mock_log,
            patch("threading.Timer") as mock_timer,
        ):
            # Create a mock timer instance that will call the timeout handler immediately
            mock_timer_instance = Mock()
            mock_timer.return_value = mock_timer_instance

            session._start_session_timer()

            # Get the timeout handler that was passed to Timer
            timeout_handler = mock_timer.call_args[0][1]

            # Call the timeout handler directly to simulate timeout
            timeout_handler()

            # Should log the cleanup error
            mock_log.assert_any_call("Error during timeout cleanup: Cleanup failed", "error")

    def test_session_timeout_handler_logs_timeout(self) -> None:
        """Test that session timeout handler logs timeout message."""
        config = SessionConfig(session_timeout=0.1)  # Very short timeout
        session = MockBaseSession(config)

        with (
            patch.object(session, "close"),
            patch.object(session, "_log") as mock_log,
            patch("threading.Timer") as mock_timer,
        ):
            # Create a mock timer instance that will call the timeout handler immediately
            mock_timer_instance = Mock()
            mock_timer.return_value = mock_timer_instance

            session._start_session_timer()

            # Get the timeout handler that was passed to Timer
            timeout_handler = mock_timer.call_args[0][1]

            # Call the timeout handler directly to simulate timeout
            timeout_handler()

            # Should log timeout message
            mock_log.assert_any_call("Session timed out after 0.1 seconds", "warning")


class TestBaseSessionSecurity:
    """Test BaseSession security functionality."""

    def setup_method(self) -> None:
        """Set up test fixtures."""
        with patch.object(LanguageHandlerFactory, "create_handler") as mock_create_handler:
            mock_handler = MockLanguageHandler()
            mock_create_handler.return_value = mock_handler

            config = SessionConfig(lang=SupportedLanguage.PYTHON)
            self.session = MockBaseSession(config)

    def test_check_security_policy_no_policy(self) -> None:
        """Test security check when no policy is set."""
        self.session.config.security_policy = None

        is_safe, violations = self.session._check_security_policy("print('hello')")

        assert is_safe is True
        assert violations == []

    def test_check_security_policy_no_language_handler(self) -> None:
        """Test security check when no language handler is initialized."""
        self.session.language_handler = None  # type: ignore[assignment]
        self.session.config.security_policy = SecurityPolicy()

        with pytest.raises(LanguageHandlerNotInitializedError):
            self.session._check_security_policy("print('hello')")

    def test_check_security_policy_with_restricted_modules(self) -> None:
        """Test security check with restricted modules."""
        restricted_module = RestrictedModule(
            name="os", description="Operating system interface", severity=SecurityIssueSeverity.HIGH
        )

        policy = SecurityPolicy(restricted_modules=[restricted_module], severity_threshold=SecurityIssueSeverity.MEDIUM)
        self.session.config.security_policy = policy

        # Mock the language handler's get_import_patterns method
        with patch.object(self.session.language_handler, "get_import_patterns", return_value="import os"):
            is_safe, violations = self.session._check_security_policy("import os")

            assert is_safe is False
            assert len(violations) == 1
            assert violations[0].severity == SecurityIssueSeverity.HIGH

    def test_check_security_policy_with_patterns(self) -> None:
        """Test security check with custom patterns."""
        pattern = SecurityPattern(
            pattern="eval\\(", description="Dangerous eval function", severity=SecurityIssueSeverity.HIGH
        )

        policy = SecurityPolicy(patterns=[pattern], severity_threshold=SecurityIssueSeverity.MEDIUM)
        self.session.config.security_policy = policy

        is_safe, violations = self.session._check_security_policy("eval('malicious code')")

        assert is_safe is False
        assert len(violations) == 1

    def test_is_safe_wrapper(self) -> None:
        """Test the is_safe method wrapper."""
        self.session.config.security_policy = None

        is_safe, violations = self.session.is_safe("print('hello')")

        assert is_safe is True
        assert violations == []


class TestBaseSessionLibraryInstallation:
    """Test BaseSession library installation functionality."""

    def setup_method(self) -> None:
        """Set up test fixtures."""
        with patch.object(LanguageHandlerFactory, "create_handler") as mock_create_handler:
            mock_handler = MockLanguageHandler()
            mock_create_handler.return_value = mock_handler

            config = SessionConfig(lang=SupportedLanguage.PYTHON)
            self.session = MockBaseSession(config)

    @patch.object(MockBaseSession, "execute_commands")
    def test_install_no_libraries(self, mock_execute_commands: Mock) -> None:
        """Test install method with no libraries."""
        mock_execute_commands.return_value = ConsoleOutput(exit_code=0)

        self.session.install(None)
        mock_execute_commands.assert_not_called()

        self.session.install([])
        mock_execute_commands.assert_not_called()

    def test_install_language_not_supported(self) -> None:
        """Test install when language doesn't support installation."""
        # Create a handler that doesn't support installation
        with (
            patch.object(self.session.language_handler, "_supports_installation", False),
            pytest.raises(LibraryInstallationNotSupportedError),
        ):
            self.session.install(["numpy"])

    def test_install_restricted_library(self) -> None:
        """Test install with security policy blocking libraries."""
        restricted_module = RestrictedModule(
            name="os", description="Operating system interface", severity=SecurityIssueSeverity.HIGH
        )

        policy = SecurityPolicy(restricted_modules=[restricted_module])
        self.session.config.security_policy = policy

        with pytest.raises(SecurityViolationError):
            self.session.install(["os"])

    @patch.object(MockBaseSession, "execute_commands")
    def test_install_success(self, mock_execute_commands: Mock) -> None:
        """Test successful library installation."""
        mock_execute_commands.return_value = ConsoleOutput(exit_code=0)

        self.session.install(["numpy", "pandas"])

        expected_commands = [
            ("pip install numpy", self.session.config.workdir),
            ("pip install pandas", self.session.config.workdir),
        ]
        mock_execute_commands.assert_called_once_with(expected_commands)


class TestBaseSessionCommandExecution:
    """Test BaseSession command execution functionality."""

    def setup_method(self) -> None:
        """Set up test fixtures."""
        with patch.object(LanguageHandlerFactory, "create_handler") as mock_create_handler:
            mock_handler = MockLanguageHandler()
            mock_create_handler.return_value = mock_handler

            config = SessionConfig(lang=SupportedLanguage.PYTHON)
            self.session = MockBaseSession(config)

    @patch.object(MockBaseSession, "execute_command")
    def test_execute_commands_string_commands(self, mock_execute_command: Mock) -> None:
        """Test execute_commands with string commands."""
        mock_execute_command.side_effect = [
            ConsoleOutput(exit_code=0, stdout="output1"),
            ConsoleOutput(exit_code=0, stdout="output2"),
        ]

        result = self.session.execute_commands(["ls", "pwd"], workdir="/test")

        assert result.stdout == "output2"  # Last command's output
        expected_call_count = 2
        assert mock_execute_command.call_count == expected_call_count
        mock_execute_command.assert_has_calls([call("ls", workdir="/test"), call("pwd", workdir="/test")])

    @patch.object(MockBaseSession, "execute_command")
    def test_execute_commands_tuple_commands(self, mock_execute_command: Mock) -> None:
        """Test execute_commands with tuple commands."""
        mock_execute_command.side_effect = [
            ConsoleOutput(exit_code=0, stdout="output1"),
            ConsoleOutput(exit_code=0, stdout="output2"),
        ]

        commands: list[str | tuple[str, str | None]] = [("ls", "/custom1"), ("pwd", "/custom2")]
        result = self.session.execute_commands(commands)

        assert result.stdout == "output2"
        mock_execute_command.assert_has_calls([call("ls", workdir="/custom1"), call("pwd", workdir="/custom2")])

    @patch.object(MockBaseSession, "execute_command")
    def test_execute_commands_mixed_commands(self, mock_execute_command: Mock) -> None:
        """Test execute_commands with mixed command types."""
        mock_execute_command.side_effect = [
            ConsoleOutput(exit_code=0, stdout="output1"),
            ConsoleOutput(exit_code=0, stdout="output2"),
        ]

        commands: list[str | tuple[str, str | None]] = ["ls", ("pwd", "/custom")]
        result = self.session.execute_commands(commands, workdir="/default")

        assert result.stdout == "output2"
        mock_execute_command.assert_has_calls([call("ls", workdir="/default"), call("pwd", workdir="/custom")])

    @patch.object(MockBaseSession, "execute_command")
    def test_execute_commands_command_failure(self, mock_execute_command: Mock) -> None:
        """Test execute_commands when a command fails."""
        mock_execute_command.side_effect = [
            ConsoleOutput(exit_code=0, stdout="success"),
            ConsoleOutput(exit_code=1, stdout="", stderr="error"),
        ]

        result = self.session.execute_commands(["ls", "invalid_command"])
        assert result.exit_code == 1


class TestBaseSessionEnvironmentSetup:
    """Test BaseSession environment setup functionality."""

    def setup_method(self) -> None:
        """Set up test fixtures."""
        with patch.object(LanguageHandlerFactory, "create_handler") as mock_create_handler:
            mock_handler = MockLanguageHandler(name=SupportedLanguage.PYTHON)
            mock_create_handler.return_value = mock_handler

            config = SessionConfig(lang=SupportedLanguage.PYTHON, workdir="/sandbox")
            self.session = MockBaseSession(config)

    @patch.object(MockBaseSession, "execute_commands")
    def test_environment_setup_python(self, mock_execute_commands: Mock) -> None:
        """Test environment setup for Python."""
        self.session.environment_setup()

        # Should be called multiple times for Python setup
        min_expected_calls = 2
        assert mock_execute_commands.call_count >= min_expected_calls

    @patch.object(MockBaseSession, "execute_commands")
    def test_environment_setup_go(self, mock_execute_commands: Mock) -> None:
        """Test environment setup for Go."""
        with patch.object(LanguageHandlerFactory, "create_handler") as mock_create_handler:
            mock_handler = MockLanguageHandler(name=SupportedLanguage.GO)
            mock_create_handler.return_value = mock_handler

            config = SessionConfig(lang=SupportedLanguage.GO, workdir="/sandbox")
            session = MockBaseSession(config)

            session.environment_setup()

            # Should be called for Go setup
            assert mock_execute_commands.call_count >= 1


class TestBaseSessionCodeExecution:
    """Test BaseSession code execution functionality."""

    def setup_method(self) -> None:
        """Set up test fixtures."""
        with patch.object(LanguageHandlerFactory, "create_handler") as mock_create_handler:
            mock_handler = MockLanguageHandler()
            mock_create_handler.return_value = mock_handler

            config = SessionConfig(lang=SupportedLanguage.PYTHON, workdir="/sandbox")
            self.session = MockBaseSession(config)
            self.session.container = Mock()
            self.session.is_open = True

    def test_run_session_not_open(self) -> None:
        """Test run method when session is not open."""
        self.session.is_open = False

        with pytest.raises(NotOpenSessionError):
            self.session.run("print('hello')")

    def test_run_no_container(self) -> None:
        """Test run method when container is None."""
        self.session.container = None

        with pytest.raises(NotOpenSessionError):
            self.session.run("print('hello')")

    @patch("tempfile.NamedTemporaryFile")
    @patch.object(MockBaseSession, "install")
    @patch.object(MockBaseSession, "copy_to_runtime")
    @patch.object(MockBaseSession, "execute_commands")
    def test_run_success(
        self, mock_execute_commands: Mock, mock_copy_to_runtime: Mock, mock_install: Mock, mock_tempfile: MagicMock
    ) -> None:
        """Test successful code execution."""
        # Mock temporary file
        mock_file = Mock()
        mock_file.name = "/tmp/code.py"
        mock_file.write = Mock()
        mock_file.seek = Mock()
        mock_tempfile.return_value.__enter__.return_value = mock_file

        mock_execute_commands.return_value = ConsoleOutput(exit_code=0, stdout="success")

        result = self.session.run("print('hello')", libraries=["numpy"])

        # Check that methods were called
        mock_install.assert_called_once_with(["numpy"])
        mock_file.write.assert_called_once_with(b"print('hello')")

        assert isinstance(result, ConsoleOutput)
        assert result.exit_code == 0

    def test_run_with_timeout(self) -> None:
        """Test run method with timeout."""
        with patch.object(self.session, "_execute_with_timeout") as mock_timeout:
            mock_timeout.return_value = ConsoleOutput(exit_code=0, stdout="success")

            self.session.run("print('hello')", timeout=5.0)

            mock_timeout.assert_called_once()
            args, kwargs = mock_timeout.call_args
            assert len(args) >= 1  # function
            assert "timeout" in kwargs
            assert abs(kwargs["timeout"] - 5.0) < 0.0001  # timeout value

    @patch.object(MockBaseSession, "_handle_timeout")
    def test_run_timeout_exception(self, mock_handle_timeout: Mock) -> None:
        """Test run method when timeout occurs."""
        with patch.object(self.session, "_execute_with_timeout") as mock_timeout:
            mock_timeout.side_effect = SandboxTimeoutError("Timeout occurred")

            with pytest.raises(SandboxTimeoutError):
                self.session.run("print('hello')", timeout=1.0)

            mock_handle_timeout.assert_called_once()


class TestBaseSessionContextManager:
    """Test BaseSession context manager functionality."""

    def setup_method(self) -> None:
        """Set up test fixtures."""
        with patch.object(LanguageHandlerFactory, "create_handler") as mock_create_handler:
            mock_handler = MockLanguageHandler()
            mock_create_handler.return_value = mock_handler

            config = SessionConfig(lang=SupportedLanguage.PYTHON)
            self.session = MockBaseSession(config)

    def test_context_manager_success(self) -> None:
        """Test successful context manager usage."""
        with (
            patch.object(self.session, "open") as mock_open,
            patch.object(self.session, "close") as mock_close,
        ):
            with self.session as session:
                assert session is self.session
                mock_open.assert_called_once()

            mock_close.assert_called_once()

    def test_context_manager_exception(self) -> None:
        """Test context manager when exception occurs."""
        with (
            patch.object(self.session, "open") as mock_open,
            patch.object(self.session, "close") as mock_close,
        ):
            with pytest.raises(ValueError):  # noqa: SIM117, PT012, PT011
                with self.session:
                    mock_open.assert_called_once()
                    raise ValueError

            mock_close.assert_called_once()


class TestBaseSessionOpenClose:
    """Test BaseSession open/close functionality."""

    def setup_method(self) -> None:
        """Set up test fixtures."""
        with patch.object(LanguageHandlerFactory, "create_handler") as mock_create_handler:
            mock_handler = MockLanguageHandler()
            mock_create_handler.return_value = mock_handler

            config = SessionConfig(lang=SupportedLanguage.PYTHON)
            self.session = MockBaseSession(config)

    def test_open_sets_state(self) -> None:
        """Test that open method sets correct state."""
        with patch.object(self.session, "_start_session_timer") as mock_timer:
            self.session.open()

            assert self.session.is_open is True
            mock_timer.assert_called_once()

    def test_close_sets_state(self) -> None:
        """Test that close method sets correct state."""
        with patch.object(self.session, "_stop_session_timer") as mock_timer:
            self.session.close()

            assert self.session.is_open is False
            mock_timer.assert_called_once()


class TestBaseSessionSecurityChecks:
    """Test security check functionality."""

    def test_security_violation_should_fail_high_severity(self) -> None:
        """Test that high severity violations cause immediate failure."""
        pattern = SecurityPattern(
            pattern="dangerous", description="Dangerous code", severity=SecurityIssueSeverity.HIGH
        )
        policy = SecurityPolicy(patterns=[pattern], severity_threshold=SecurityIssueSeverity.MEDIUM)
        config = SessionConfig(security_policy=policy)
        session = MockBaseSession(config)

        is_safe, violations = session.is_safe("dangerous code")

        assert is_safe is False
        assert len(violations) == 1
        assert violations[0].pattern == "dangerous"

    def test_restricted_library_installation(self) -> None:
        """Test that restricted libraries cannot be installed."""
        restricted_module = RestrictedModule(
            name="restricted_lib", description="Restricted library", severity=SecurityIssueSeverity.HIGH
        )
        policy = SecurityPolicy(restricted_modules=[restricted_module])
        config = SessionConfig(security_policy=policy)
        session = MockBaseSession(config)

        with pytest.raises(SecurityViolationError, match="Library restricted_lib is not allowed"):
            session.install(["restricted_lib"])
