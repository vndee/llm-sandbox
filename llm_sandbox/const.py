from dataclasses import dataclass
from enum import Enum


class SandboxBackend(str, Enum):
    """Sandbox backend."""

    DOCKER = "docker"
    KUBERNETES = "kubernetes"
    PODMAN = "podman"
    MICROMAMBA = "micromamba"


@dataclass
class SupportedLanguage:
    """Supported languages."""

    PYTHON = "python"
    JAVA = "java"
    JAVASCRIPT = "javascript"
    CPP = "cpp"
    GO = "go"
    RUBY = "ruby"


@dataclass
class DefaultImage:
    """Default images."""

    PYTHON = "python:3.11-bullseye"
    JAVA = "openjdk:11.0.12-jdk-bullseye"
    JAVASCRIPT = "node:22-bullseye"
    CPP = "gcc:11.2.0-bullseye"
    GO = "golang:1.23.4-bullseye"
    RUBY = "ruby:3.0.2-bullseye"
