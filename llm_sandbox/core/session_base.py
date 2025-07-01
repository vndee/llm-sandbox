import logging
import re
import tempfile
import threading
import time
import types
import uuid
from abc import ABC, abstractmethod
from typing import Any, cast

from llm_sandbox.const import SupportedLanguage
from llm_sandbox.core.config import SessionConfig
from llm_sandbox.core.mixins import CommandExecutionMixin, ContainerAPI, FileOperationsMixin, TimeoutMixin
from llm_sandbox.data import ConsoleOutput
from llm_sandbox.exceptions import (
    LanguageHandlerNotInitializedError,
    LibraryInstallationNotSupportedError,
    NotOpenSessionError,
    SandboxTimeoutError,
    SecurityViolationError,
)
from llm_sandbox.language_handlers.factory import LanguageHandlerFactory
from llm_sandbox.security import SecurityIssueSeverity, SecurityPattern

PYTHON_VENV_DIR = "/tmp/venv"
PYTHON_PIP_CACHE_DIR = "/tmp/pip_cache"
PYTHON_CREATE_VENV_COMMAND = "python -m venv --system-site-packages /tmp/venv"
PYTHON_CREATE_PIP_CACHE_COMMAND = "mkdir -p /tmp/pip_cache"
PYTHON_UPGRADE_PIP_COMMAND = "/tmp/venv/bin/pip install --upgrade pip --cache-dir /tmp/pip_cache"

GO_CREATE_MODULE_COMMAND = "go mod init sandbox"
GO_TIDY_MODULE_COMMAND = "go mod tidy"


class BaseSession(
    ABC,
    TimeoutMixin,
    FileOperationsMixin,
    CommandExecutionMixin,
):
    """Base session implementation with common functionality."""

    def __init__(self, config: SessionConfig, **kwargs: Any) -> None:  # noqa: ARG002
        """Initialize base session."""
        self.config = config
        self.verbose = config.verbose
        self.logger = logging.getLogger(__name__)

        # Initialize language handler
        self.language_handler = LanguageHandlerFactory.create_handler(config.lang, self.logger)

        # Session timing
        self._session_start_time: float | None = None
        self._session_timer: threading.Timer | None = None

        # Container state
        self.container: Any = None
        self.is_open = False
        self.using_existing_container = config.is_using_existing_container()

        self.container_api: ContainerAPI

    def _log(self, message: str, level: str = "info") -> None:
        """Log message if verbose."""
        if self.verbose:
            getattr(self.logger, level)(message)

    def _check_session_timeout(self) -> None:
        """Check if session has exceeded maximum lifetime."""
        if self.config.session_timeout and self._session_start_time:
            elapsed = time.time() - self._session_start_time
            if elapsed > self.config.session_timeout:
                msg = f"Session exceeded maximum lifetime of {self.config.session_timeout} seconds"
                raise SandboxTimeoutError(msg)

    def _start_session_timer(self) -> None:
        """Start session timeout timer."""
        self._session_start_time = time.time()

        if self.config.session_timeout:

            def timeout_handler() -> None:
                self._log(f"Session timed out after {self.config.session_timeout} seconds", "warning")
                try:
                    self.close()
                except Exception as e:  # noqa: BLE001
                    self._log(f"Error during timeout cleanup: {e}", "error")

            self._session_timer = threading.Timer(self.config.session_timeout, timeout_handler)
            self._session_timer.daemon = True
            self._session_timer.start()

    def _stop_session_timer(self) -> None:
        """Stop session timeout timer."""
        if self._session_timer:
            self._session_timer.cancel()
            self._session_timer = None

    def _add_restricted_module_patterns(self) -> None:
        """Add patterns for restricted modules to the security policy."""
        if not self.config.security_policy or not self.config.security_policy.restricted_modules:
            return

        for module in self.config.security_policy.restricted_modules:
            pattern = SecurityPattern(
                pattern=self.language_handler.get_import_patterns(module.name),
                description=module.description,
                severity=module.severity,
            )
            self.config.security_policy.add_pattern(pattern)

    def _check_pattern_violations(self, filtered_code: str) -> tuple[bool, list[SecurityPattern]]:
        """Check for pattern violations in the filtered code.

        Args:
            filtered_code (str): Code with comments filtered out.

        Returns:
            tuple[bool, list[SecurityPattern]]: A tuple containing safety status and violations.

        """
        violations: list[SecurityPattern] = []

        if not self.config.security_policy or not self.config.security_policy.patterns:
            return True, []

        for pattern_obj in self.config.security_policy.patterns:
            if not pattern_obj.pattern:
                continue

            violation_found = self._check_single_pattern(pattern_obj, filtered_code)
            if violation_found:
                violations.append(pattern_obj)

                # Check if this violation should cause immediate failure
                if self._should_fail_on_violation(pattern_obj):
                    return False, violations

        return True, violations

    def _check_single_pattern(self, pattern_obj: SecurityPattern, filtered_code: str) -> bool:
        """Check a single security pattern against the code.

        Args:
            pattern_obj (SecurityPattern): The pattern to check.
            filtered_code (str): Code with comments filtered out.

        Returns:
            bool: True if pattern matches (violation found), False otherwise.

        """
        try:
            return bool(re.search(pattern_obj.pattern, filtered_code))
        except re.error as e:
            self._log(f"Invalid regex pattern '{pattern_obj.pattern}': {e}", "error")
            return False

    def _should_fail_on_violation(self, pattern_obj: SecurityPattern) -> bool:
        """Determine if a security violation should cause immediate failure.

        Args:
            pattern_obj (SecurityPattern): The pattern that was violated.

        Returns:
            bool: True if execution should fail immediately, False otherwise.

        """
        if not self.config.security_policy or not self.config.security_policy.severity_threshold:
            return False

        return (
            self.config.security_policy.severity_threshold > SecurityIssueSeverity.SAFE
            and pattern_obj.severity >= self.config.security_policy.severity_threshold
        )

    def _check_security_policy(self, code: str) -> tuple[bool, list[SecurityPattern]]:
        r"""Check code against security policy.

        Args:
            code (str): The code to check.

        Returns:
            tuple[bool, list[SecurityPattern]]: A tuple containing a boolean indicating if the code is safe
                                                and a list of security patterns that were violated.

        """
        if not self.config.security_policy:
            return True, []

        if not self.language_handler:
            raise LanguageHandlerNotInitializedError(self.config.lang)

        # Add patterns for restricted modules
        self._add_restricted_module_patterns()

        # If no patterns are configured, code is safe
        if not self.config.security_policy.patterns:
            return True, []

        # Check for pattern violations
        filtered_code = self.language_handler.filter_comments(code)
        return self._check_pattern_violations(filtered_code)

    def is_safe(self, code: str) -> tuple[bool, list[SecurityPattern]]:
        r"""Check if code is safe to execute.

        Args:
            code (str): The code to check.

        Returns:
            tuple[bool, list[SecurityPattern]]: A tuple containing a boolean indicating if the code is safe
                                                and a list of security patterns that were violated.

        """
        return self._check_security_policy(code)

    def install(self, libraries: list[str] | None = None) -> None:
        r"""Install libraries into the sandbox environment.

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
            raise LibraryInstallationNotSupportedError(self.config.lang)

        if self.config.security_policy and self.config.security_policy.restricted_modules:
            for library in libraries:
                if any(library == module.name for module in self.config.security_policy.restricted_modules):
                    msg = f"Library {library} is not allowed to be installed"
                    raise SecurityViolationError(msg)

        library_installation_commands: list[str | tuple[str, str | None]] = [
            (self.language_handler.get_library_installation_command(library), self.config.workdir)
            for library in libraries
        ]
        self.execute_commands(library_installation_commands)

    def execute_commands(
        self, commands: list[str | tuple[str, str | None]], workdir: str | None = None
    ) -> ConsoleOutput:
        r"""Execute a sequence of commands within the sandbox container.

        This method executes the commands in order and returns the
        ConsoleOutput of the first command that fails or, if all succeed,
        the output of the last command.

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
                output = self.execute_command(command, workdir=workdir)

            if output.exit_code:
                return output

        return output

    def environment_setup(self) -> None:
        r"""Set up the language-specific environment within the sandbox.

        This method is called during session initialization to prepare the environment
        for the selected programming language. This may include creating virtual environments,
        initializing package managers, setting up cache directories, etc.

        For Python, it creates a venv and pip cache directory, then upgrades pip.
        For Go, it initializes a Go module.
        We will support other languages in the future.

        Note: This method is skipped when using an existing container (container_id is provided).

        Raises:
            CommandFailedError: If any setup command fails.

        """
        # Skip environment setup when using existing container
        if self.using_existing_container:
            self._log("Skipping environment setup for existing container", "info")
            return

        self.execute_commands([
            (f"mkdir -p {self.config.workdir}", None),
        ])

        match self.language_handler.name:
            case SupportedLanguage.PYTHON:
                # Create venv and cache directory first
                self.execute_commands([
                    (PYTHON_CREATE_VENV_COMMAND, None),
                    (PYTHON_CREATE_PIP_CACHE_COMMAND, None),
                ])

                self._ensure_ownership([PYTHON_VENV_DIR, PYTHON_PIP_CACHE_DIR])

                # Now upgrade pip with proper ownership and cache
                self.execute_commands([
                    (
                        PYTHON_UPGRADE_PIP_COMMAND,
                        None,
                    ),
                ])
            case SupportedLanguage.GO:
                self.execute_commands([
                    (GO_CREATE_MODULE_COMMAND, self.config.workdir),
                    (GO_TIDY_MODULE_COMMAND, self.config.workdir),
                ])

    def run(self, code: str, libraries: list | None = None, timeout: float | None = None) -> ConsoleOutput:
        r"""Run the provided code within the Docker sandbox session.

        This method performs the following steps:
        1. Ensures the session is open (container is running).
        2. Installs any specified `libraries` using the language-specific handler.
        3. Writes the `code` to a temporary file on the host.
        4. Copies this temporary file into the container at the configured `workdir`.
        5. Retrieves execution commands from the language handler.
        6. Executes these commands in the container using `execute_commands`.

        Args:
            code (str): The code string to execute.
            libraries (list | None, optional): A list of libraries to install before running the code.
                                            Defaults to None.
            timeout (float | None, optional): The timeout for the execution of the code.
                Defaults to None.

        Returns:
            ConsoleOutput: An object containing the stdout, stderr, and exit code from the code execution.

        Raises:
            NotOpenSessionError: If the session (container) is not currently open/running.
            CommandFailedError: If any of the execution commands fail.

        """
        if not self.container or not self.is_open:
            raise NotOpenSessionError

        self._check_session_timeout()
        actual_timeout = timeout or self.config.get_execution_timeout()

        def _run_code() -> ConsoleOutput:
            self.install(libraries)

            with tempfile.NamedTemporaryFile(
                delete=True, suffix=f".{self.language_handler.file_extension}"
            ) as code_file:
                code_file.write(code.encode("utf-8"))
                code_file.seek(0)

                code_dest_file = f"{self.config.workdir}/{uuid.uuid4().hex}.{self.language_handler.file_extension}"
                self.copy_to_runtime(code_file.name, code_dest_file)

                commands = self.language_handler.get_execution_commands(code_dest_file)
                return self.execute_commands(
                    cast("list[str | tuple[str, str | None]]", commands),
                    workdir=self.config.workdir,
                )

        try:
            result = self._execute_with_timeout(_run_code, timeout=actual_timeout)
            return cast("ConsoleOutput", result)
        except SandboxTimeoutError:
            self._handle_timeout()
            raise

    @abstractmethod
    def _handle_timeout(self) -> None:
        r"""Handle timeout cleanup - backend specific.

        Must be implemented by subclasses.
        """
        raise NotImplementedError

    @abstractmethod
    def _connect_to_existing_container(self, container_id: str) -> None:
        r"""Connect to an existing container - backend specific.

        Must be implemented by subclasses.

        Args:
            container_id (str): The ID of the existing container to connect to.

        """
        raise NotImplementedError

    @abstractmethod
    def open(self) -> None:
        r"""Open session.

        Must be implemented by subclasses.
        """
        self._start_session_timer()
        self.is_open = True

    @abstractmethod
    def close(self) -> None:
        r"""Close session.

        Must be implemented by subclasses.
        """
        self._stop_session_timer()
        self.is_open = False

    def __enter__(self) -> "BaseSession":
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
