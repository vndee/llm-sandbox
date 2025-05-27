"""Base session functionality for LLM Sandbox."""

import logging
import re
import types
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

from llm_sandbox.const import SupportedLanguage
from llm_sandbox.data import ConsoleOutput, ExecutionResult
from llm_sandbox.exceptions import (
    CommandFailedError,
    LanguageHandlerNotInitializedError,
    LibraryInstallationNotSupportedError,
)
from llm_sandbox.language_handlers.factory import LanguageHandlerFactory
from llm_sandbox.security import SecurityPattern, SecurityPolicy

if TYPE_CHECKING:
    from llm_sandbox.language_handlers.base import AbstractLanguageHandler


class Session(ABC):
    r"""Abstract base class for sandbox sessions.

    This class defines the common interface for all sandbox sessions, regardless of the
    underlying container technology (Docker, Kubernetes, Podman) or programming language.
    It provides methods for opening and closing sessions, running code, installing libraries,
    managing files, and executing arbitrary commands within the sandboxed environment.

    Subclasses must implement the abstract methods to provide backend-specific functionality.
    """

    def __init__(
        self,
        lang: str,
        verbose: bool = True,
        image: str | None = None,
        keep_template: bool = False,
        logger: logging.Logger | None = None,
        workdir: str | None = "/sandbox",
        security_policy: SecurityPolicy | None = None,
    ) -> None:
        r"""Initialize the sandbox session.

        Args:
            lang (str): The programming language to use for the sandbox (e.g., "python", "java").
            verbose (bool, optional): Whether to log detailed messages. Defaults to True.
            image (str | None, optional): The container image to use. If None, a default image
                                            for the specified language may be used. Defaults to None.
            keep_template (bool, optional): Whether to keep the template image after session closure.
                                            Defaults to False.
            logger (logging.Logger | None, optional): The logger instance to use. If None, a default
                                                        logger is created. Defaults to None.
            workdir (str | None, optional): The working directory inside the container.
                                                Defaults to "/sandbox".
            security_policy (SecurityPolicy | None, optional): The security policy to apply to the container.
                                                    Defaults to None.

        Raises:
            CommandFailedError: If an internal command fails during session initialization.
            LibraryInstallationNotSupportedError: If library installation is attempted for a language
                                                that doesn't support it during setup.

        """
        self.lang = lang
        self.verbose = verbose
        self.image = image
        self.keep_template = keep_template
        self.logger = logger or logging.getLogger(__name__)
        self.workdir = workdir
        self.security_policy = security_policy

        self.language_handler: AbstractLanguageHandler = LanguageHandlerFactory.create_handler(self.lang, self.logger)

    def _log(self, message: str, level: str = "info") -> None:
        r"""Log a message if verbose logging is enabled.

        Args:
            message (str): The message to log.
            level (str, optional): The logging level (e.g., "info", "debug", "warning").
                                    Defaults to "info".

        """
        if self.verbose:
            getattr(self.logger, level)(message)

    @abstractmethod
    def open(self) -> None:
        r"""Open the sandbox session.

        This method prepares the sandbox environment for code execution. This may involve
        starting a container, setting up networking, and creating necessary directories.
        Must be implemented by subclasses.
        """
        raise NotImplementedError

    @abstractmethod
    def close(self) -> None:
        r"""Close the sandbox session.

        This method cleans up any resources used by the sandbox session. This may involve
        stopping and removing containers, deleting temporary files, and releasing network ports.
        Must be implemented by subclasses.
        """
        raise NotImplementedError

    @abstractmethod
    def run(self, code: str, libraries: list | None = None) -> ConsoleOutput | ExecutionResult:
        r"""Run the provided code within the sandbox session.

        This is the primary method for executing user code. It handles code execution,
        library installation (if supported and requested), and captures output.
        Must be implemented by subclasses.

        Args:
            code (str): The code string to execute in the sandbox.
            libraries (list | None, optional): A list of libraries to install before running the code.
                                            Defaults to None.

        Returns:
            ConsoleOutput | ExecutionResult: An object containing the execution results,
                                            including stdout, stderr, exit code, and potentially
                                            extracted plots or other artifacts.

        """
        raise NotImplementedError

    @abstractmethod
    def copy_to_runtime(self, src: str, dest: str) -> None:
        r"""Copy a file or directory from the host to the sandbox runtime.

        Must be implemented by subclasses.

        Args:
            src (str): The path to the source file or directory on the host system.
            dest (str): The destination path within the sandbox container.

        """
        raise NotImplementedError

    @abstractmethod
    def copy_from_runtime(self, src: str, dest: str) -> None:
        r"""Copy a file or directory from the sandbox runtime to the host.

        Must be implemented by subclasses.

        Args:
            src (str): The path to the source file or directory within the sandbox container.
            dest (str): The destination path on the host system.

        """
        raise NotImplementedError

    @abstractmethod
    def execute_command(self, command: str, workdir: str | None = None) -> ConsoleOutput:
        r"""Execute an arbitrary command directly within the sandbox container.

        This method is used for running system-level commands, not user code.
        Must be implemented by subclasses.

        Args:
            command (str): The command string to execute (e.g., "ls -l", "mkdir /data").
            workdir (str | None, optional): The working directory within the container where
                                        the command should be executed. If None, a default
                                        working directory may be used. Defaults to None.

        Returns:
            ConsoleOutput: An object containing the stdout, stderr, and exit code of the command.

        """
        raise NotImplementedError

    @abstractmethod
    def get_archive(self, path: str) -> tuple[bytes, dict]:
        r"""Retrieve a file or directory from the sandbox as a tar archive.

        Must be implemented by subclasses.

        Args:
            path (str): The path to the file or directory within the sandbox container.

        Returns:
            tuple[bytes, dict]: A tuple containing the raw bytes of the tar archive
                                and a dictionary with archive metadata.

        """
        raise NotImplementedError

    def _ensure_ownership(self, folders: list[str]) -> None:
        r"""Ensure correct file ownership for specified folders, especially for non-root users.

        This method is typically called during environment setup to ensure that the user
        inside the container has the necessary permissions to write to directories used for
        caching or virtual environments.
        May be a no-op if running as root or if ownership is not an issue for the backend.
        Can be overridden by subclasses if specific ownership logic is needed.

        Args:
            folders (list[str]): A list of absolute paths to folders within the container
                                whose ownership needs to be checked and potentially adjusted.

        """
        raise NotImplementedError

    def execute_commands(
        self, commands: list[str | tuple[str, str | None]], workdir: str | None = None
    ) -> ConsoleOutput:
        r"""Execute a sequence of commands within the sandbox container.

        This method iterates through a list of commands, executing them one by one.
        If any command fails (returns a non-zero exit code), it raises a CommandFailedError.

        Args:
            commands (list[str | tuple[str, str | None]]): A list of commands to execute.
                Each item can be either:
                - A string: The command will be run in the `workdir` specified for this method call.
                - A tuple (command_string, specific_workdir): The command will be run in the
                    `specific_workdir`. If `specific_workdir` is None, the container's default may be used.
            workdir (str | None, optional): The default working directory to use for commands
                                        that do not specify their own. Defaults to None.

        Returns:
            ConsoleOutput: The output of the last successfully executed command.

        Raises:
            CommandFailedError: If any command in the sequence returns a non-zero exit code.

        """
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
        r"""Install specified libraries or packages into the sandbox environment.

        This method uses the language-specific handler to determine the correct installation
        commands for the given libraries. It then executes these commands within the sandbox.

        Args:
            libraries (list[str] | None, optional): A list of library names to install.
                                                    If None or empty, no installation is performed.
                                                    Defaults to None.

        Raises:
            LibraryInstallationNotSupportedError: If the active language handler does not support
                                                on-the-fly library installation.
            CommandFailedError: If any library installation command fails.

        """
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
        r"""Set up the language-specific environment within the sandbox.

        This method is called during session initialization to prepare the environment
        for the selected programming language. This may include creating virtual environments,
        initializing package managers, setting up cache directories, etc.

        For Python, it creates a venv and pip cache directory, then upgrades pip.
        For Go, it initializes a Go module.
        We will support other languages in the future.

        Raises:
            CommandFailedError: If any setup command fails.

        """
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

    def _check_security_policy(self, code: str) -> tuple[bool, list[SecurityPattern]]:
        r"""Check the security policy.

        Args:
            code (str): The code to check.

        Returns:
            tuple[bool, list[SecurityPattern]]: A tuple containing a boolean indicating if the code is safe
                                                and a list of security patterns that were violated.

        """
        if not self.security_policy:
            return True, []

        if not self.language_handler:
            raise LanguageHandlerNotInitializedError(self.lang)

        if self.security_policy.dangerous_modules and not self.security_policy.patterns:
            for module in self.security_policy.dangerous_modules:
                self.security_policy.add_pattern(
                    SecurityPattern(
                        pattern=self.language_handler.get_import_patterns(module.name),
                        description=module.description,
                        severity=module.severity,
                    )
                )

        if self.security_policy.patterns:
            violations: list[SecurityPattern] = []
            for pattern_obj in self.security_policy.patterns:
                if pattern_obj.pattern:
                    try:
                        if re.search(pattern_obj.pattern, code):
                            if pattern_obj.severity >= self.security_policy.safety_level:
                                violations.append(pattern_obj)
                                return False, violations

                            violations.append(pattern_obj)
                    except re.error as e:
                        self._log(
                            f"Security alert: Invalid regex pattern '{pattern_obj.pattern}'. Error: {e}",
                            level="error",
                        )

        return True, []

    def is_safe(self, code: str) -> tuple[bool, list[SecurityPattern]]:
        r"""Check if the code is safe.

        Args:
            code (str): The code to check.

        Returns:
            tuple[bool, list[SecurityPattern]]: A tuple containing a boolean indicating if the code is safe

        """
        return self._check_security_policy(code)

    def __enter__(self) -> "Session":
        r"""Enter the runtime context for the session (invokes `open()`).

        This allows the session to be used with the `with` statement, ensuring
        that `open()` is called at the beginning of the block and `close()` is called
        at the end, even if errors occur.

        Returns:
            Session: The current session instance.

        """
        self.open()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: types.TracebackType | None,
    ) -> None:
        r"""Exit the runtime context for the session (invokes `close()`).

        Ensures that the session's `close()` method is called when the `with` block is exited,
        regardless of whether an exception occurred within the block.

        Args:
            exc_type (type[BaseException] | None): The type of the exception that caused the
                                                    context to be exited, if any.
            exc_val (BaseException | None): The exception instance that caused the context
                                            to be exited, if any.
            exc_tb (types.TracebackType | None): A traceback object associated with the
                                                exception, if any.

        """
        self.close()
