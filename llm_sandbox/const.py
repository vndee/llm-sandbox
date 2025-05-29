"""Constants used throughout the LLM Sandbox application.

This module defines enumerations and dataclasses for constants such as
sandbox backend types, supported programming languages, and default
container image names.
"""

from dataclasses import dataclass
from enum import Enum

try:
    from enum import StrEnum
except ImportError:
    # Python < 3.11 compatibility
    class StrEnum(str, Enum):
        """String enumeration for Python < 3.11 compatibility."""


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


@dataclass
class DefaultImage:
    r"""Dataclass defining constants for default container images for each language.

    These are the default Docker image names used when a specific image is not
    provided by the user for a given programming language.
    """

    PYTHON = "vndee/sandbox-python-311-bullseye"
    JAVA = "openjdk:11.0.12-jdk-bullseye"
    JAVASCRIPT = "node:22-bullseye"
    CPP = "gcc:11.2.0-bullseye"
    GO = "golang:1.23.4-bullseye"
    RUBY = "ruby:3.0.2-bullseye"
