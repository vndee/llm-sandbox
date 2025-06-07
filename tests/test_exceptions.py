# ruff: noqa: PLR2004
"""Tests for llm_sandbox.exceptions module."""

from llm_sandbox.exceptions import (
    CommandEmptyError,
    CommandFailedError,
    ContainerError,
    ExtraArgumentsError,
    ImageNotFoundError,
    ImagePullError,
    InvalidRegexPatternError,
    LanguageHandlerNotInitializedError,
    LanguageNotSupportedError,
    LanguageNotSupportPlotError,
    LibraryInstallationNotSupportedError,
    MissingDependencyError,
    NotOpenSessionError,
    PackageManagerError,
    ResourceError,
    SandboxError,
    SandboxTimeoutError,
    SecurityError,
    SecurityViolationError,
    UnsupportedBackendError,
    ValidationError,
)


class TestSandboxError:
    """Test SandboxError base exception."""

    def test_sandbox_error_initialization(self) -> None:
        """Test SandboxError initialization."""
        error = SandboxError("test message")
        assert str(error) == "test message"
        assert isinstance(error, Exception)

    def test_sandbox_error_inheritance(self) -> None:
        """Test that SandboxError inherits from Exception."""
        error = SandboxError("test")
        assert isinstance(error, Exception)


class TestContainerError:
    """Test ContainerError exception."""

    def test_container_error(self) -> None:
        """Test ContainerError creation and inheritance."""
        error = ContainerError("container failed")
        assert str(error) == "container failed"
        assert isinstance(error, SandboxError)


class TestSecurityError:
    """Test SecurityError exception."""

    def test_security_error(self) -> None:
        """Test SecurityError creation and inheritance."""
        error = SecurityError("security violation")
        assert str(error) == "security violation"
        assert isinstance(error, SandboxError)


class TestResourceError:
    """Test ResourceError exception."""

    def test_resource_error(self) -> None:
        """Test ResourceError creation and inheritance."""
        error = ResourceError("resource limit exceeded")
        assert str(error) == "resource limit exceeded"
        assert isinstance(error, SandboxError)


class TestValidationError:
    """Test ValidationError exception."""

    def test_validation_error(self) -> None:
        """Test ValidationError creation and inheritance."""
        error = ValidationError("validation failed")
        assert str(error) == "validation failed"
        assert isinstance(error, SandboxError)


class TestExtraArgumentsError:
    """Test ExtraArgumentsError exception."""

    def test_extra_arguments_error(self) -> None:
        """Test ExtraArgumentsError creation and message."""
        error = ExtraArgumentsError("extra args provided")
        assert str(error) == "extra args provided"
        assert isinstance(error, SandboxError)


class TestLanguageNotSupportedError:
    """Test LanguageNotSupportedError exception."""

    def test_language_not_supported_error(self) -> None:
        """Test LanguageNotSupportedError creation and message formatting."""
        error = LanguageNotSupportedError("rust")
        assert str(error) == "Language rust is not supported"
        assert isinstance(error, SandboxError)


class TestImageNotFoundError:
    """Test ImageNotFoundError exception."""

    def test_image_not_found_error(self) -> None:
        """Test ImageNotFoundError creation and message formatting."""
        error = ImageNotFoundError("python:3.9")
        assert str(error) == "Image python:3.9 not found"
        assert isinstance(error, SandboxError)


class TestNotOpenSessionError:
    """Test NotOpenSessionError exception."""

    def test_not_open_session_error(self) -> None:
        """Test NotOpenSessionError creation and default message."""
        error = NotOpenSessionError()
        expected_msg = "Session is not open. Please call open() method before running code."
        assert str(error) == expected_msg
        assert isinstance(error, SandboxError)


class TestLibraryInstallationNotSupportedError:
    """Test LibraryInstallationNotSupportedError exception."""

    def test_library_installation_not_supported_error(self) -> None:
        """Test LibraryInstallationNotSupportedError creation and message formatting."""
        error = LibraryInstallationNotSupportedError("javascript")
        expected_msg = "Library installation is not supported for javascript"
        assert str(error) == expected_msg
        assert isinstance(error, SandboxError)


class TestCommandEmptyError:
    """Test CommandEmptyError exception."""

    def test_command_empty_error(self) -> None:
        """Test CommandEmptyError creation and default message."""
        error = CommandEmptyError()
        assert str(error) == "Command cannot be empty"
        assert isinstance(error, SandboxError)


class TestCommandFailedError:
    """Test CommandFailedError exception."""

    def test_command_failed_error(self) -> None:
        """Test CommandFailedError creation and message formatting."""
        error = CommandFailedError("ls -la", 1, "permission denied")
        expected_msg = "Command ls -la failed with exit code 1:\npermission denied"
        assert str(error) == expected_msg
        assert isinstance(error, SandboxError)


class TestPackageManagerError:
    """Test PackageManagerError exception."""

    def test_package_manager_error(self) -> None:
        """Test PackageManagerError creation and message formatting."""
        # This should cover line 89 which was missing
        error = PackageManagerError("pip")
        assert str(error) == "Package manager pip not found"
        assert isinstance(error, SandboxError)

    def test_package_manager_error_different_managers(self) -> None:
        """Test PackageManagerError with different package managers."""
        managers = ["npm", "conda", "yarn", "poetry"]
        for manager in managers:
            error = PackageManagerError(manager)
            assert str(error) == f"Package manager {manager} not found"
            assert isinstance(error, SandboxError)


class TestImagePullError:
    """Test ImagePullError exception."""

    def test_image_pull_error(self) -> None:
        """Test ImagePullError creation and message formatting."""
        error = ImagePullError("python:3.9", "network timeout")
        expected_msg = "Failed to pull image python:3.9: network timeout"
        assert str(error) == expected_msg
        assert isinstance(error, SandboxError)


class TestUnsupportedBackendError:
    """Test UnsupportedBackendError exception."""

    def test_unsupported_backend_error(self) -> None:
        """Test UnsupportedBackendError creation and message formatting."""
        error = UnsupportedBackendError("invalid_backend")
        assert str(error) == "Unsupported backend: invalid_backend"
        assert isinstance(error, SandboxError)


class TestMissingDependencyError:
    """Test MissingDependencyError exception."""

    def test_missing_dependency_error(self) -> None:
        """Test MissingDependencyError creation and message."""
        error = MissingDependencyError("Docker is not installed")
        assert str(error) == "Docker is not installed"
        assert isinstance(error, SandboxError)


class TestLanguageNotSupportPlotError:
    """Test LanguageNotSupportPlotError exception."""

    def test_language_not_support_plot_error(self) -> None:
        """Test LanguageNotSupportPlotError creation and message formatting."""
        error = LanguageNotSupportPlotError("java")
        expected_msg = "Language java does not support plot detection"
        assert str(error) == expected_msg
        assert isinstance(error, SandboxError)


class TestInvalidRegexPatternError:
    """Test InvalidRegexPatternError exception."""

    def test_invalid_regex_pattern_error(self) -> None:
        """Test InvalidRegexPatternError creation and message formatting."""
        error = InvalidRegexPatternError("[invalid")
        assert str(error) == "Invalid regex pattern: [invalid"
        assert isinstance(error, SandboxError)


class TestLanguageHandlerNotInitializedError:
    """Test LanguageHandlerNotInitializedError exception."""

    def test_language_handler_not_initialized_error(self) -> None:
        """Test LanguageHandlerNotInitializedError creation and message formatting."""
        error = LanguageHandlerNotInitializedError("python")
        expected_msg = "Language handler for python is not initialized."
        assert str(error) == expected_msg
        assert isinstance(error, SandboxError)


class TestSecurityViolationError:
    """Test SecurityViolationError exception."""

    def test_security_violation_error(self) -> None:
        """Test SecurityViolationError creation and message."""
        error = SecurityViolationError("Unsafe code detected")
        assert str(error) == "Unsafe code detected"
        assert isinstance(error, SandboxError)


class TestSandboxTimeoutError:
    """Test SandboxTimeoutError exception."""

    def test_sandbox_timeout_error_basic(self) -> None:
        """Test SandboxTimeoutError creation with message only."""
        error = SandboxTimeoutError("Operation timed out")
        assert str(error) == "Operation timed out"
        assert isinstance(error, SandboxError)
        assert error.timeout_duration is None

    def test_sandbox_timeout_error_with_duration(self) -> None:
        """Test SandboxTimeoutError creation with timeout duration."""
        # This should cover lines 153-154 which were missing
        error = SandboxTimeoutError("Operation timed out after 30 seconds", timeout_duration=30.0)
        assert str(error) == "Operation timed out after 30 seconds"
        assert error.timeout_duration is not None
        assert abs(error.timeout_duration - 30.0) < 1e-6
        assert isinstance(error, SandboxError)

    def test_sandbox_timeout_error_none_duration(self) -> None:
        """Test SandboxTimeoutError with explicit None duration."""
        error = SandboxTimeoutError("Timeout occurred", timeout_duration=None)
        assert str(error) == "Timeout occurred"
        assert error.timeout_duration is None

    def test_sandbox_timeout_error_zero_duration(self) -> None:
        """Test SandboxTimeoutError with zero duration."""
        error = SandboxTimeoutError("Immediate timeout", timeout_duration=0.0)
        assert str(error) == "Immediate timeout"
        assert error.timeout_duration is not None
        assert abs(error.timeout_duration - 0.0) < 1e-6

    def test_sandbox_timeout_error_inheritance(self) -> None:
        """Test SandboxTimeoutError inheritance chain."""
        error = SandboxTimeoutError("timeout")
        assert isinstance(error, SandboxError)
        assert isinstance(error, Exception)


class TestExceptionHierarchy:
    """Test exception inheritance hierarchy."""

    def test_all_exceptions_inherit_from_sandbox_error(self) -> None:
        """Test that all custom exceptions inherit from SandboxError."""
        exceptions = [
            ContainerError,
            SecurityError,
            ResourceError,
            ValidationError,
            ExtraArgumentsError,
            LanguageNotSupportedError,
            ImageNotFoundError,
            NotOpenSessionError,
            LibraryInstallationNotSupportedError,
            CommandEmptyError,
            CommandFailedError,
            PackageManagerError,
            ImagePullError,
            UnsupportedBackendError,
            MissingDependencyError,
            LanguageNotSupportPlotError,
            InvalidRegexPatternError,
            LanguageHandlerNotInitializedError,
            SecurityViolationError,
            SandboxTimeoutError,
        ]

        for exception_class in exceptions:
            assert issubclass(exception_class, SandboxError)
            assert issubclass(exception_class, Exception)

    def test_exception_instantiation(self) -> None:
        """Test that all exceptions can be instantiated properly."""
        # Test exceptions that don't require arguments
        no_arg_exceptions = [
            (NotOpenSessionError, ()),
            (CommandEmptyError, ()),
        ]

        for exception_class, args in no_arg_exceptions:
            error = exception_class(*args)
            assert isinstance(error, SandboxError)
            assert len(str(error)) > 0

        # Test exceptions that require arguments
        arg_exceptions = [
            (SandboxError, ("test",)),
            (ContainerError, ("test",)),
            (SecurityError, ("test",)),
            (ResourceError, ("test",)),
            (ValidationError, ("test",)),
            (ExtraArgumentsError, ("test",)),
            (LanguageNotSupportedError, ("python",)),
            (ImageNotFoundError, ("python:3.9",)),
            (LibraryInstallationNotSupportedError, ("javascript",)),
            (CommandFailedError, ("ls", 1, "error")),
            (PackageManagerError, ("pip",)),
            (ImagePullError, ("python:3.9", "error")),
            (UnsupportedBackendError, ("invalid",)),
            (MissingDependencyError, ("dependency missing",)),
            (LanguageNotSupportPlotError, ("java",)),
            (InvalidRegexPatternError, ("[invalid",)),
            (LanguageHandlerNotInitializedError, ("python",)),
            (SecurityViolationError, ("violation",)),
            (SandboxTimeoutError, ("timeout",)),
        ]

        for exception_class, args in arg_exceptions:  # type: ignore[assignment]
            error = exception_class(*args)
            assert isinstance(error, SandboxError)
            assert len(str(error)) > 0


class TestExceptionMessageFormatting:
    """Test exception message formatting."""

    def test_exception_message_contains_input(self) -> None:
        """Test that exception messages contain the input parameters."""
        test_cases = [
            (LanguageNotSupportedError("rust"), "rust"),
            (ImageNotFoundError("python:3.9"), "python:3.9"),
            (PackageManagerError("pip"), "pip"),
            (UnsupportedBackendError("invalid"), "invalid"),
            (LanguageNotSupportPlotError("java"), "java"),
            (InvalidRegexPatternError("[invalid"), "[invalid"),
            (LanguageHandlerNotInitializedError("python"), "python"),
            (ImagePullError("python:3.9", "error"), "python:3.9"),
            (ImagePullError("python:3.9", "error"), "error"),
        ]

        for error, expected_substring in test_cases:
            assert expected_substring in str(error)

    def test_command_failed_error_formatting(self) -> None:
        """Test CommandFailedError message formatting."""
        error = CommandFailedError("echo hello", 127, "command not found")
        message = str(error)

        assert "echo hello" in message
        assert "127" in message
        assert "command not found" in message
        assert "failed with exit code" in message
