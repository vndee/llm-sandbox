"""Base session functionality for LLM Sandbox."""

import logging
from abc import ABC, abstractmethod
from typing import Optional, Dict
from dataclasses import dataclass


@dataclass
class ExecutionResult:
    """Result of code execution in sandbox."""

    exit_code: int
    output: str
    error: str
    execution_time: float
    resource_usage: Dict


class ConsoleOutput:
    """Console output from code execution."""

    def __init__(self, text: str, exit_code: int = 0):
        self._text = text
        self._exit_code = exit_code

    @property
    def text(self) -> str:
        return self._text

    @property
    def exit_code(self) -> int:
        return self._exit_code

    def __repr__(self):
        return f"ConsoleOutput(text={self.text}, exit_code={self.exit_code})"

    def __str__(self):
        return self.text


class Session(ABC):
    """Abstract base class for sandbox sessions."""

    def __init__(
        self,
        lang: str,
        verbose: bool = True,
        strict_security: bool = True,
        runtime_configs: Optional[dict] = None,
        logger: Optional[logging.Logger] = None,
    ):
        self.lang = lang
        self.verbose = verbose
        self.runtime_configs = runtime_configs
        self.strict_security = strict_security
        self.logger = logger or logging.getLogger(__name__)

    def _log(self, message: str, level: str = "info"):
        """Log message if verbose is enabled."""
        if self.verbose:
            getattr(self.logger, level)(message)

    @abstractmethod
    def open(self):
        """Open the sandbox session."""
        raise NotImplementedError

    @abstractmethod
    def close(self):
        """Close the sandbox session."""
        raise NotImplementedError

    @abstractmethod
    def copy_to_runtime(self, src: str, dest: str):
        """Copy file to sandbox runtime."""
        raise NotImplementedError

    @abstractmethod
    def copy_from_runtime(self, src: str, dest: str):
        """Copy file from sandbox runtime."""
        raise NotImplementedError

    @abstractmethod
    def execute_command(self, command: str) -> ConsoleOutput:
        """Execute command in sandbox."""
        raise NotImplementedError

    def __enter__(self):
        """Context manager entry."""
        self.open()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()
