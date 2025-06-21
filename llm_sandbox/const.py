"""Constants used throughout the LLM Sandbox application.

This module defines enumerations and dataclasses for constants such as
sandbox backend types, supported programming languages, and default
container image names.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Any


class StrEnum(str, Enum):
    """A string enumeration that combines str and Enum functionality.

    This implementation provides StrEnum functionality for Python versions < 3.11
    where StrEnum is not available natively.

    Members are strings and can be compared directly to string values.
    """

    def __new__(cls, value: str) -> "StrEnum":
        """Create a new StrEnum member."""
        if not isinstance(value, str):
            msg = f"StrEnum values must be strings, got {type(value).__name__}"
            raise TypeError(msg)

        # Create the string object
        obj = str.__new__(cls, value)
        obj._value_ = value
        return obj

    def __str__(self) -> str:
        """Return the string value."""
        return str(self.value)

    def __repr__(self) -> str:
        """Return a detailed representation."""
        return f"<{self.__class__.__name__}.{self.name}: '{self.value}'>"

    @classmethod
    def _missing_(cls, value: Any) -> "StrEnum":
        """Handle missing values during lookup."""
        # This allows for case-insensitive lookup if desired
        if isinstance(value, str):
            for member in cls:
                if member.value.lower() == value.lower():
                    return member

        msg = f"{value!r} is not a valid {cls.__name__}"
        raise ValueError(msg)


class SandboxBackend(StrEnum):
    r"""Enumeration of supported sandbox backend technologies.

    Each value represents a different containerization or virtualization technology
    that can be used to isolate code execution.
    """

    DOCKER = "docker"
    KUBERNETES = "kubernetes"
    PODMAN = "podman"
    MICROMAMBA = "micromamba"


class SupportedLanguage(StrEnum):
    r"""Dataclass defining constants for supported programming languages.

    Each attribute represents a language identifier string used by the sandbox
    to select appropriate language handlers and container images.
    """

    PYTHON = "python"
    JAVA = "java"
    JAVASCRIPT = "javascript"
    CPP = "cpp"
    GO = "go"
    RUBY = "ruby"
    R = "r"


@dataclass
class DefaultImage:
    r"""Dataclass defining constants for default container images for each language.

    These are the default Docker image names used when a specific image is not
    provided by the user for a given programming language.
    """

    PYTHON = "ghcr.io/vndee/sandbox-python-311-bullseye"
    JAVA = "ghcr.io/vndee/sandbox-java-11-bullseye"
    JAVASCRIPT = "ghcr.io/vndee/sandbox-node-22-bullseye"
    CPP = "ghcr.io/vndee/sandbox-cpp-11-bullseye"
    GO = "ghcr.io/vndee/sandbox-go-123-bullseye"
    RUBY = "ghcr.io/vndee/sandbox-ruby-302-bullseye"
    R = "ghcr.io/vndee/sandbox-r-451-bullseye"
