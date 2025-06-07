"""Mixins for common functionality."""

import io
import signal
import tarfile
import threading
import types
from abc import abstractmethod
from pathlib import Path
from typing import Any, Protocol

from llm_sandbox.data import ConsoleOutput
from llm_sandbox.exceptions import NotOpenSessionError, SandboxTimeoutError


class ContainerAPI(Protocol):
    """Protocol for container operations."""

    def create_container(self, config: Any) -> Any:
        """Create container."""
        ...

    def start_container(self, container: Any) -> None:
        """Start container."""
        ...

    def stop_container(self, container: Any) -> None:
        """Stop container."""
        ...

    def execute_command(self, container: Any, command: str, **kwargs: Any) -> tuple[int, Any]:
        """Execute command in container."""
        ...

    def copy_to_container(self, container: Any, src: str, dest: str) -> None:
        """Copy file to container."""
        ...

    def copy_from_container(self, container: Any, src: str) -> tuple[bytes, dict]:
        """Copy file from container."""
        ...


class TimeoutMixin:
    """Mixin for timeout functionality."""

    def _execute_with_timeout(self, func: Any, timeout: float | None = None, *args: Any, **kwargs: Any) -> Any:
        """Execute a function with timeout monitoring.

        Uses signal-based timeout on Unix systems and threading-based timeout on Windows.

        Args:
            func: The function to execute
            timeout: Timeout in seconds
            *args: Arguments to pass to the function
            **kwargs: Keyword arguments to pass to the function

        Returns:
            The result from the function execution

        Raises:
            SandboxTimeoutError: If the function execution times out

        """
        if timeout is None:
            return func(*args, **kwargs)

        # For Unix systems with SIGALRM, use signal-based timeout
        if hasattr(signal, "SIGALRM"):

            def timeout_handler(signum: int, frame: types.FrameType | None) -> None:  # noqa: ARG001
                msg = f"Operation timed out after {timeout} seconds"
                raise SandboxTimeoutError(msg)

            old_handler = signal.signal(signal.SIGALRM, timeout_handler)
            signal.alarm(int(timeout))

            try:
                return func(*args, **kwargs)
            finally:
                signal.alarm(0)
                signal.signal(signal.SIGALRM, old_handler)

        # For Windows or systems without SIGALRM, use threading
        result: list[Any] = [None]
        exception: list[Exception | None] = [None]

        def target() -> None:
            try:
                result[0] = func(*args, **kwargs)
            except Exception as e:  # noqa: BLE001
                exception[0] = e

        thread = threading.Thread(target=target)
        thread.daemon = True
        thread.start()
        thread.join(timeout)

        if thread.is_alive():
            # Note: We can't actually kill the thread, but we can timeout
            msg = f"Operation timed out after {timeout} seconds"
            raise SandboxTimeoutError(msg)

        if exception[0]:
            raise exception[0]

        return result[0]


class FileOperationsMixin:
    """Mixin for file operations."""

    container_api: ContainerAPI
    container: Any
    verbose: bool
    logger: Any

    def copy_to_runtime(self, src: str, dest: str) -> None:
        """Copy file to runtime with common validation."""
        if not self.container:
            raise NotOpenSessionError

        src_path = Path(src)
        if not (src_path.exists() and (src_path.is_file() or src_path.is_dir())):
            msg = f"Source path {src} does not exist or is not accessible"
            raise FileNotFoundError(msg)

        if self.verbose:
            self.logger.info("Copying %s to %s", src, dest)

        dest_dir = str(Path(dest).parent)
        if dest_dir:
            self._ensure_directory_exists(dest_dir)

        self.container_api.copy_to_container(self.container, src, dest)
        self._ensure_ownership([dest])

    def copy_from_runtime(self, src: str, dest: str) -> None:
        """Copy file from runtime with common security filtering."""
        if not self.container:
            raise NotOpenSessionError

        if self.verbose:
            self.logger.info("Copying %s to %s", src, dest)

        bits, stat = self.container_api.copy_from_container(self.container, src)
        if stat.get("size", 0) == 0:
            msg = f"File {src} not found in container"
            raise FileNotFoundError(msg)

        self._extract_archive_safely(bits, dest)

    def _extract_archive_safely(self, bits: bytes, dest: str) -> None:
        """Extract tar archive with security filtering and consistent structure."""
        tarstream = io.BytesIO(bits)
        with tarfile.open(fileobj=tarstream, mode="r") as tar:
            safe_members = []

            for member in tar.getmembers():
                if member.name.startswith("/") or ".." in member.name:
                    if self.verbose:
                        self.logger.warning("Skipping unsafe path: %s", member.name)
                    continue
                if member.issym() or member.islnk():
                    if self.verbose:
                        self.logger.warning("Skipping symlink: %s", member.name)
                    continue
                safe_members.append(member)

            if not safe_members:
                msg = "No safe content found in archive"
                raise FileNotFoundError(msg)

            Path(dest).parent.mkdir(parents=True, exist_ok=True)

            # Handle single file case
            if len(safe_members) == 1 and safe_members[0].isfile():
                safe_members[0].name = Path(dest).name
                extract_path = str(Path(dest).parent)
            else:
                # Handle directory case - extract to destination preserving structure
                extract_path = str(dest)

            for member in safe_members:
                tar.extract(member, path=extract_path)

    @abstractmethod
    def _ensure_directory_exists(self, path: str) -> None:
        """Ensure directory exists - backend specific."""
        ...

    @abstractmethod
    def _ensure_ownership(self, paths: list[str]) -> None:
        """Ensure correct ownership - backend specific."""
        ...


class CommandExecutionMixin:
    """Mixin for command execution."""

    container_api: ContainerAPI
    container: Any
    verbose: bool
    logger: Any
    stream: bool

    def execute_command(self, command: str, workdir: str | None = None) -> ConsoleOutput:
        """Execute command with common logic."""
        if not command:
            msg = "Command cannot be empty"
            raise ValueError(msg)

        if not self.container:
            raise NotOpenSessionError

        if self.verbose:
            self.logger.info("Executing command: %s", command)

        exit_code, output = self.container_api.execute_command(
            self.container, command, workdir=workdir, stream=self.stream
        )

        stdout, stderr = self._process_output(output)

        if self.verbose:
            if stdout:
                self.logger.info("STDOUT: %s", stdout)
            if stderr:
                self.logger.error("STDERR: %s", stderr)

        return ConsoleOutput(exit_code=exit_code or 0, stdout=stdout, stderr=stderr)

    def _process_output(self, output: Any) -> tuple[str, str]:
        """Process command output - backend specific implementation."""
        if not self.stream:
            return self._process_non_stream_output(output)
        return self._process_stream_output(output)

    @abstractmethod
    def _process_non_stream_output(self, output: Any) -> tuple[str, str]:
        """Process non-streaming output."""
        ...

    @abstractmethod
    def _process_stream_output(self, output: Any) -> tuple[str, str]:
        """Process streaming output."""
        ...
