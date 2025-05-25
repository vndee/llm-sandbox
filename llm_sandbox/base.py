"""Base session functionality for LLM Sandbox."""

import logging
import types
import warnings
from abc import ABC, abstractmethod
from dataclasses import dataclass, field

from llm_sandbox.const import SupportedLanguage
from llm_sandbox.exceptions import (
    CommandFailedError,
    LibraryInstallationNotSupportedError,
)

from .artifact import FileOutput, PlotOutput


@dataclass
class ConsoleOutput:
    """Console output from code execution."""

    exit_code: int = 0
    stderr: str = ""
    stdout: str = ""

    @property
    def text(self) -> str:
        """Get the text representation of the console output.

        .. deprecated::
            The `text` property is deprecated and will be removed in a future version.
            Use `stdout` attribute directly instead.
        """
        warnings.warn(
            "The 'text' property is deprecated and will be removed in a future version. "  # noqa: E501
            "Use 'stdout' attribute directly instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.stdout

    @property
    def success(self) -> bool:
        """Check if the execution was successful."""
        return not self.exit_code


@dataclass
class ExecutionResult(ConsoleOutput):
    """Result of code execution in sandbox."""

    plots: list[PlotOutput] = field(default_factory=list)
    files: list[FileOutput] = field(default_factory=list)


class Session(ABC):
    """Abstract base class for sandbox sessions."""

    def __init__(
        self,
        lang: str,
        verbose: bool = True,
        strict_security: bool = True,
        runtime_configs: dict | None = None,
        logger: logging.Logger | None = None,
    ) -> None:
        """Initialize the session."""
        self.lang = lang
        self.verbose = verbose
        self.runtime_configs = runtime_configs
        self.strict_security = strict_security
        self.logger = logger or logging.getLogger(__name__)

    def _log(self, message: str, level: str = "info") -> None:
        """Log message if verbose is enabled."""
        if self.verbose:
            getattr(self.logger, level)(message)

    @abstractmethod
    def open(self) -> None:
        """Open the sandbox session."""
        raise NotImplementedError

    @abstractmethod
    def close(self) -> None:
        """Close the sandbox session."""
        raise NotImplementedError

    @abstractmethod
    def run(
        self, code: str, libraries: list | None = None
    ) -> ConsoleOutput | ExecutionResult:
        """Run the code in the sandbox session."""
        raise NotImplementedError

    @abstractmethod
    def copy_to_runtime(self, src: str, dest: str) -> None:
        """Copy file to sandbox runtime."""
        raise NotImplementedError

    @abstractmethod
    def copy_from_runtime(self, src: str, dest: str) -> None:
        """Copy file from sandbox runtime."""
        raise NotImplementedError

    @abstractmethod
    def execute_command(self, command: str) -> ConsoleOutput:
        """Execute command in sandbox."""
        raise NotImplementedError

    def execute_commands(
        self, commands: list[str | tuple[str, str]], workdir: str | None = None
    ) -> ConsoleOutput:
        """Execute a list of commands in the container."""
        output = ConsoleOutput(exit_code=0)
        for command in commands:
            if isinstance(command, tuple) or (
                isinstance(command, list)
                and len(command) == 2  # noqa: PLR2004
                and isinstance(command[0], str)
                and isinstance(command[1], str)
            ):
                cmd_str, cmd_workdir = command
                output = self.execute_command(cmd_str, workdir=cmd_workdir)
            else:
                cmd_str = command
                output = self.execute_command(command, workdir=workdir)

            if output.exit_code:
                raise CommandFailedError(cmd_str, output.exit_code, output.stdout)

        return output

    def install(self, libraries: list[str]) -> None:
        """Install packages in the sandbox."""
        if not libraries:
            return

        if not self.language_handler.is_support_library_installation:
            raise LibraryInstallationNotSupportedError

        library_installation_commands = [
            (
                self.language_handler.get_library_installation_command(library),
                self.workdir,
            )
            for library in libraries
        ]

        self.execute_commands(library_installation_commands)

    def environment_setup(self) -> None:
        """Set up the language environment."""
        self.execute_commands(
            [
                (f"mkdir -p {self.workdir}", None),
            ]
        )

        match self.language_handler.name:
            case SupportedLanguage.PYTHON:
                # Create venv and cache directory first
                self.execute_commands(
                    [
                        ("python -m venv /tmp/venv", None),
                        ("mkdir -p /tmp/pip_cache", None),
                    ]
                )

                self._ensure_ownership(["/tmp/venv", "/tmp/pip_cache"])

                # Now upgrade pip with proper ownership and cache
                self.execute_commands(
                    [
                        (
                            "/tmp/venv/bin/pip install --upgrade pip --cache-dir /tmp/pip_cache",  # noqa: E501
                            None,
                        ),
                    ]
                )
            case SupportedLanguage.GO:
                self.execute_commands(
                    [
                        ("go mod init sandbox", self.workdir),
                        ("go mod tidy", self.workdir),
                    ]
                )

    def __enter__(self) -> "Session":
        """Context manager entry."""
        self.open()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: types.TracebackType | None,
    ) -> None:
        """Context manager exit."""
        self.close()
