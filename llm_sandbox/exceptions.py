"""Custom exceptions for LLM Sandbox."""


class SandboxError(Exception):
    """Base exception for all sandbox related errors."""

    def __init__(self, message: str) -> None:
        """Initialize the SandboxError."""
        super().__init__(message)


class ContainerError(SandboxError):
    """Raised when container operations fail."""


class SecurityError(SandboxError):
    """Raised for security related issues."""


class ResourceError(SandboxError):
    """Raised when resource limits are exceeded."""


class ValidationError(SandboxError):
    """Raised when input validation fails."""


class ExtraArgumentsError(SandboxError):
    """Raised when extra arguments are provided."""

    def __init__(self, message: str) -> None:
        """Initialize the ExtraArgumentsError."""
        super().__init__(message)


class LanguageNotSupportedError(SandboxError):
    """Raised when the language is not supported."""

    def __init__(self, lang: str) -> None:
        """Initialize the LanguageNotSupportedError."""
        super().__init__(f"Language {lang} is not supported")


class ImageNotFoundError(SandboxError):
    """Raised when the image is not found."""

    def __init__(self, image: str) -> None:
        """Initialize the ImageNotFoundError."""
        super().__init__(f"Image {image} not found")


class NotOpenSessionError(SandboxError):
    """Raised when the session is not open."""

    def __init__(self) -> None:
        """Initialize the NotOpenSessionError."""
        super().__init__("Session is not open. Please call open() method before running code.")


class LibraryInstallationNotSupportedError(SandboxError):
    """Raised when the library installation is not supported."""

    def __init__(self, lang: str) -> None:
        """Initialize the LibraryInstallationNotSupportedError."""
        super().__init__(f"Library installation is not supported for {lang}")


class CommandEmptyError(SandboxError):
    """Raised when the command is empty."""

    def __init__(self) -> None:
        """Initialize the CommandEmptyError."""
        super().__init__("Command cannot be empty")


class CommandFailedError(SandboxError):
    """Raised when a command fails."""

    def __init__(self, command: str, exit_code: int, output: str) -> None:
        """Initialize the CommandFailedError."""
        super().__init__(f"Command {command} failed with exit code {exit_code}:\n{output}")


class PackageManagerError(SandboxError):
    """Raised when a package manager is not found."""

    def __init__(self, package_manager: str) -> None:
        """Initialize the PackageManagerError."""
        super().__init__(f"Package manager {package_manager} not found")


class ImagePullError(SandboxError):
    """Raised when an image pull fails."""

    def __init__(self, image: str, error: str) -> None:
        """Initialize the ImagePullError."""
        super().__init__(f"Failed to pull image {image}: {error}")


class UnsupportedBackendError(SandboxError):
    """Raised when an unsupported backend is provided."""

    def __init__(self, backend: str) -> None:
        """Initialize the UnsupportedBackendError."""
        super().__init__(f"Unsupported backend: {backend}")


class MissingDependencyError(SandboxError):
    """Raised when a required dependency is not installed."""

    def __init__(self, message: str) -> None:
        """Initialize the MissingDependencyError."""
        super().__init__(message)


class LanguageNotSupportPlotError(SandboxError):
    """Raised when the language does not support plot detection."""

    def __init__(self, lang: str) -> None:
        """Initialize the LanguageNotSupportPlotError."""
        super().__init__(f"Language {lang} does not support plot detection")


class InvalidRegexPatternError(SandboxError):
    """Raised when a regex pattern is invalid."""

    def __init__(self, pattern: str) -> None:
        """Initialize the InvalidRegexPatternError."""
        super().__init__(f"Invalid regex pattern: {pattern}")


class LanguageHandlerNotInitializedError(SandboxError):
    """Raised when the language handler is not initialized."""

    def __init__(self, language: str) -> None:
        """Initialize the exception."""
        super().__init__(f"Language handler for {language} is not initialized.")


class SecurityViolationError(SandboxError):
    """Exception raised when a security policy is violated."""

    def __init__(self, message: str) -> None:
        """Initialize the SecurityViolationError."""
        super().__init__(message)


class SandboxTimeoutError(SandboxError):
    """Raised when an operation times out."""

    def __init__(self, message: str, timeout_duration: float | None = None) -> None:
        """Initialize the TimeoutError."""
        super().__init__(message)
        self.timeout_duration = timeout_duration
