"""Custom exceptions for LLM Sandbox."""


class SandboxException(Exception):
    """Base exception for all sandbox related errors."""

    pass


class ContainerError(SandboxException):
    """Raised when container operations fail."""

    pass


class SecurityError(SandboxException):
    """Raised for security related issues."""

    pass


class ResourceError(SandboxException):
    """Raised when resource limits are exceeded."""

    pass


class TimeoutError(SandboxException):
    """Raised when execution exceeds timeout limit."""

    pass


class ValidationError(SandboxException):
    """Raised when input validation fails."""

    pass


class DependencyError(SandboxException):
    """Raised when a required dependency is not installed."""

    pass
