"""Base session functionality for LLM Sandbox."""

import logging
import types
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

from llm_sandbox.const import SupportedLanguage
from llm_sandbox.data import ConsoleOutput, ExecutionResult
from llm_sandbox.exceptions import CommandFailedError, LibraryInstallationNotSupportedError
from llm_sandbox.language_handlers.factory import LanguageHandlerFactory

if TYPE_CHECKING:
    from llm_sandbox.language_handlers.base import AbstractLanguageHandler


class Session(ABC):
    """Abstract base class for sandbox sessions."""

    def __init__(
        self,
        lang: str,
        verbose: bool = True,
        image: str | None = None,
        keep_template: bool = False,
        strict_security: bool = True,
        logger: logging.Logger | None = None,
        workdir: str | None = "/sandbox",
    ) -> None:
        """Initialize the session."""
        self.lang = lang
        self.verbose = verbose
        self.image = image
        self.keep_template = keep_template
        self.strict_security = strict_security
        self.logger = logger or logging.getLogger(__name__)
        self.workdir = workdir

        self.language_handler: AbstractLanguageHandler = LanguageHandlerFactory.create_handler(self.lang, self.logger)

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
    def run(self, code: str, libraries: list | None = None) -> ConsoleOutput | ExecutionResult:
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
    def execute_command(self, command: str, workdir: str | None = None) -> ConsoleOutput:
        """Execute command in sandbox."""
        raise NotImplementedError

    @abstractmethod
    def get_archive(self, path: str) -> tuple[bytes, dict]:
        """Get archive of files from container."""
        raise NotImplementedError

    def _ensure_ownership(self, folders: list[str]) -> None:
        """For non-root users, ensure ownership of the resources."""
        raise NotImplementedError

    def execute_commands(
        self, commands: list[str | tuple[str, str | None]], workdir: str | None = None
    ) -> ConsoleOutput:
        """Execute a list of commands in the container."""
        output = ConsoleOutput(exit_code=0)
        for command in commands:
            if isinstance(command, tuple) or (
                isinstance(command, list)
                and len(command) == 2  # noqa: PLR2004
                and isinstance(command[0], str)
                and (isinstance(command[1], str) or command[1] is None)
            ):
                cmd_str, cmd_workdir = command
                output = self.execute_command(cmd_str, workdir=cmd_workdir)
            else:
                cmd_str = command
                output = self.execute_command(command, workdir=workdir)

            if output.exit_code:
                raise CommandFailedError(cmd_str, output.exit_code, output.stdout)

        return output

    def install(self, libraries: list[str] | None = None) -> None:
        """Install packages in the sandbox."""
        if not libraries:
            return

        if not self.language_handler.is_support_library_installation:
            raise LibraryInstallationNotSupportedError(self.lang)

        library_installation_commands: list[str | tuple[str, str | None]] = [
            (
                self.language_handler.get_library_installation_command(library),
                self.workdir,
            )
            for library in libraries
        ]

        self.execute_commands(library_installation_commands)

    def environment_setup(self) -> None:
        """Set up the language environment."""
        self.execute_commands([
            (f"mkdir -p {self.workdir}", None),
        ])

        match self.language_handler.name:
            case SupportedLanguage.PYTHON:
                # Create venv and cache directory first
                self.execute_commands([
                    ("python -m venv /tmp/venv", None),
                    ("mkdir -p /tmp/pip_cache", None),
                ])

                self._ensure_ownership(["/tmp/venv", "/tmp/pip_cache"])

                # Now upgrade pip with proper ownership and cache
                self.execute_commands([
                    (
                        "/tmp/venv/bin/pip install --upgrade pip --cache-dir /tmp/pip_cache",
                        None,
                    ),
                ])
            case SupportedLanguage.GO:
                self.execute_commands([
                    ("go mod init sandbox", self.workdir),
                    ("go mod tidy", self.workdir),
                ])

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
