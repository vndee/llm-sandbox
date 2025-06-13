# ruff: noqa: SLF001, PLR2004, ARG002, PT011, FBT003

"""Test cases for the new architecture mixins."""

import io
import tarfile
import time
from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

import pytest

from llm_sandbox.core.mixins import CommandExecutionMixin, FileOperationsMixin, TimeoutMixin
from llm_sandbox.data import ConsoleOutput
from llm_sandbox.exceptions import CommandEmptyError, NotOpenSessionError, SandboxTimeoutError


class MockContainerAPI:
    """Mock implementation of ContainerAPI for testing."""

    def __init__(self) -> None:
        """Initialize mock container API."""
        self.create_container = Mock()
        self.start_container = Mock()
        self.stop_container = Mock()
        self.execute_command = Mock()
        self.copy_to_container = Mock()
        self.copy_from_container = Mock()


class TestTimeoutMixin:
    """Test cases for TimeoutMixin."""

    def test_execute_with_timeout_no_timeout(self) -> None:
        """Test execution without timeout."""
        mixin = TimeoutMixin()

        def test_func(x: int, y: int) -> int:
            return x + y

        result = mixin._execute_with_timeout(test_func, 2, 3, timeout=None, force_kill_on_timeout=False)
        assert result == 5

    def test_execute_with_timeout_success(self) -> None:
        """Test successful execution within timeout."""
        mixin = TimeoutMixin()

        def fast_func() -> str:
            return "completed"

        result = mixin._execute_with_timeout(fast_func, timeout=2.0)
        assert result == "completed"

    def test_execute_with_timeout_exception_propagation(self) -> None:
        """Test that exceptions are properly propagated."""
        mixin = TimeoutMixin()

        def error_func() -> str:
            raise ValueError

        with pytest.raises(ValueError):
            mixin._execute_with_timeout(error_func, timeout=2.0)

    def test_execute_with_timeout_actual_timeout(self) -> None:
        """Test actual timeout scenario with SandboxTimeoutError."""
        mixin = TimeoutMixin()

        def slow_func() -> str:
            time.sleep(0.5)  # Simulated sleep
            return "should not complete"

        with pytest.raises(SandboxTimeoutError, match="Operation timed out after 0.1 seconds"):
            mixin._execute_with_timeout(slow_func, timeout=0.1)

    def test_execute_with_timeout_with_handler_success(self) -> None:
        """Test timeout with force_kill_on_timeout=True and successful handler call."""

        class TimeoutMixinWithHandler(TimeoutMixin):
            def __init__(self) -> None:
                self.logger = Mock()
                self.handler_called = False

            def _handle_timeout(self) -> None:
                self.handler_called = True

        mixin = TimeoutMixinWithHandler()

        def slow_func() -> str:
            time.sleep(0.5)  # Simulated sleep
            return "should not complete"

        with pytest.raises(SandboxTimeoutError, match="Operation timed out after 0.1 seconds"):
            mixin._execute_with_timeout(slow_func, timeout=0.1, force_kill_on_timeout=True)

        # Verify the handler was called
        assert mixin.handler_called

    def test_execute_with_timeout_with_handler_exception(self) -> None:
        """Test timeout with force_kill_on_timeout=True when handler raises exception."""

        class TimeoutMixinWithFailingHandler(TimeoutMixin):
            def __init__(self) -> None:
                self.logger = Mock()

            def _handle_timeout(self) -> None:
                msg = "Handler failed"
                raise RuntimeError(msg)

        mixin = TimeoutMixinWithFailingHandler()

        def slow_func() -> str:
            time.sleep(0.5)  # Simulated sleep
            return "should not complete"

        with pytest.raises(SandboxTimeoutError, match="Operation timed out after 0.1 seconds"):
            mixin._execute_with_timeout(slow_func, timeout=0.1, force_kill_on_timeout=True)

        # Verify the warning was logged when handler failed
        mixin.logger.warning.assert_called_once_with("Failed to cleanup container after timeout")

    def test_execute_with_timeout_no_handler(self) -> None:
        """Test timeout with force_kill_on_timeout=True but no _handle_timeout method."""
        mixin = TimeoutMixin()
        mixin.logger = Mock()

        def slow_func() -> str:
            time.sleep(0.5)  # Simulated sleep
            return "should not complete"

        with pytest.raises(SandboxTimeoutError, match="Operation timed out after 0.1 seconds"):
            mixin._execute_with_timeout(slow_func, timeout=0.1, force_kill_on_timeout=True)

        # Handler should not be called since it doesn't exist
        mixin.logger.warning.assert_not_called()

    def test_execute_with_timeout_force_kill_disabled(self) -> None:
        """Test timeout with force_kill_on_timeout=False."""

        class TimeoutMixinWithHandler(TimeoutMixin):
            def __init__(self) -> None:
                self.logger = Mock()
                self.handler_called = False

            def _handle_timeout(self) -> None:
                self.handler_called = True

        mixin = TimeoutMixinWithHandler()

        def slow_func() -> str:
            time.sleep(0.5)  # Simulated sleep
            return "should not complete"

        with pytest.raises(SandboxTimeoutError, match="Operation timed out after 0.1 seconds"):
            mixin._execute_with_timeout(slow_func, timeout=0.1, force_kill_on_timeout=False)

        # Handler should not be called when force_kill_on_timeout=False
        assert not mixin.handler_called


class TestFileOperationsMixin:
    """Test cases for FileOperationsMixin."""

    def setup_method(self) -> None:
        """Set up test fixtures."""

        # Create a concrete implementation for testing
        class ConcreteFileOperationsMixin(FileOperationsMixin):
            def _ensure_directory_exists(self, path: str) -> None:
                """Ensure directory exists."""

            def _ensure_ownership(self, paths: list[str]) -> None:
                """Ensure ownership."""

        self.mixin = ConcreteFileOperationsMixin()
        self.mixin.container_api = MockContainerAPI()
        self.mixin.container = Mock()
        self.mixin.verbose = True
        self.mixin.logger = Mock()

    def test_copy_to_runtime_no_container(self) -> None:
        """Test copy_to_runtime when no container is available."""
        self.mixin.container = None

        with pytest.raises(NotOpenSessionError):
            self.mixin.copy_to_runtime("src", "dest")

    @patch("llm_sandbox.core.mixins.Path")
    def test_copy_to_runtime_file_not_exists(self, mock_path: MagicMock) -> None:
        """Test copy_to_runtime when source file doesn't exist."""
        mock_path_instance = Mock()
        mock_path_instance.exists.return_value = False
        mock_path_instance.is_file.return_value = False
        mock_path_instance.is_dir.return_value = False
        mock_path.return_value = mock_path_instance

        with pytest.raises(FileNotFoundError):
            self.mixin.copy_to_runtime("/nonexistent", "/dest")

    @patch("llm_sandbox.core.mixins.Path")
    def test_copy_to_runtime_success(self, mock_path: MagicMock) -> None:
        """Test successful copy_to_runtime."""
        # Mock source path
        mock_src_path = Mock()
        mock_src_path.exists.return_value = True
        mock_src_path.is_file.return_value = True
        mock_src_path.is_dir.return_value = False

        # Mock destination path
        mock_dest_path = Mock()
        # Use a simple string to avoid Mock serialization issues
        mock_dest_path.parent = "/dest"

        mock_path.side_effect = [mock_src_path, mock_dest_path]

        self.mixin.copy_to_runtime("/src/file", "/dest/file")

        # Verify the container API was called correctly
        self.mixin.container_api.copy_to_container.assert_called_once_with(  # type: ignore[attr-defined]
            self.mixin.container, "/src/file", "/dest/file"
        )

    def test_copy_from_runtime_no_container(self) -> None:
        """Test copy_from_runtime when no container is available."""
        self.mixin.container = None

        with pytest.raises(NotOpenSessionError):
            self.mixin.copy_from_runtime("src", "dest")

    def test_copy_from_runtime_file_not_found(self) -> None:
        """Test copy_from_runtime when file is not found in container."""
        self.mixin.container_api.copy_from_container.return_value = (b"", {"size": 0})  # type: ignore[attr-defined]

        with pytest.raises(FileNotFoundError):
            self.mixin.copy_from_runtime("/src", "/dest")

    @patch.object(FileOperationsMixin, "_extract_archive_safely")
    def test_copy_from_runtime_success(self, mock_extract: Mock) -> None:
        """Test successful copy_from_runtime."""
        # Create a simple tar archive
        tar_buffer = io.BytesIO()
        with tarfile.open(fileobj=tar_buffer, mode="w") as tar:
            info = tarfile.TarInfo(name="test.txt")
            info.size = 5
            tar.addfile(info, io.BytesIO(b"hello"))

        self.mixin.container_api.copy_from_container.return_value = (tar_buffer.getvalue(), {"size": 100})  # type: ignore[attr-defined]

        self.mixin.copy_from_runtime("/src", "/dest")

        mock_extract.assert_called_once()

    def test_extract_archive_safely_single_file(self) -> None:
        """Test extracting a single file from archive."""
        # Create a simple tar archive with one file
        tar_buffer = io.BytesIO()
        with tarfile.open(fileobj=tar_buffer, mode="w") as tar:
            info = tarfile.TarInfo(name="test.txt")
            info.size = 5
            tar.addfile(info, io.BytesIO(b"hello"))

        # Use a real temporary directory to avoid Mock issues
        import tempfile

        with tempfile.TemporaryDirectory() as temp_dir:
            dest_file = f"{temp_dir}/dest.txt"
            self.mixin._extract_archive_safely(tar_buffer.getvalue(), dest_file)

            # Verify the file was created
            assert Path(dest_file).exists()
            with Path(dest_file).open() as f:
                assert f.read() == "hello"

    def test_extract_archive_safely_unsafe_paths(self) -> None:
        """Test that unsafe paths are filtered out."""
        # Create tar with unsafe paths
        tar_buffer = io.BytesIO()
        with tarfile.open(fileobj=tar_buffer, mode="w") as tar:
            # Absolute path
            info1 = tarfile.TarInfo(name="/absolute/path.txt")
            info1.size = 0
            tar.addfile(info1, io.BytesIO(b""))

            # Path with ..
            info2 = tarfile.TarInfo(name="../escape.txt")
            info2.size = 0
            tar.addfile(info2, io.BytesIO(b""))

        with pytest.raises(FileNotFoundError, match="No safe content found"):
            self.mixin._extract_archive_safely(tar_buffer.getvalue(), "/dest")

    def test_extract_archive_safely_symlinks(self) -> None:
        """Test that symlinks are filtered out."""
        tar_buffer = io.BytesIO()
        with tarfile.open(fileobj=tar_buffer, mode="w") as tar:
            # Symlink
            info = tarfile.TarInfo(name="link.txt")
            info.type = tarfile.SYMTYPE
            tar.addfile(info)

        with pytest.raises(FileNotFoundError, match="No safe content found"):
            self.mixin._extract_archive_safely(tar_buffer.getvalue(), "/dest")


class TestCommandExecutionMixin:
    """Test cases for CommandExecutionMixin."""

    def setup_method(self) -> None:
        """Set up test fixtures."""

        # Create a concrete implementation for testing
        class ConcreteCommandExecutionMixin(CommandExecutionMixin):
            def _process_non_stream_output(self, output: str) -> tuple[str, str]:
                return "stdout", "stderr"

            def _process_stream_output(self, output: str) -> tuple[str, str]:
                return "stream_stdout", "stream_stderr"

        self.mixin = ConcreteCommandExecutionMixin()
        self.mixin.container_api = MockContainerAPI()
        self.mixin.container = Mock()
        self.mixin.verbose = True
        self.mixin.logger = Mock()
        self.mixin.stream = False

    def test_execute_command_empty_command(self) -> None:
        """Test execute_command with empty command."""
        with pytest.raises(CommandEmptyError):
            self.mixin.execute_command("")

    def test_execute_command_no_container(self) -> None:
        """Test execute_command when no container is available."""
        self.mixin.container = None

        with pytest.raises(NotOpenSessionError):
            self.mixin.execute_command("ls")

    def test_execute_command_success(self) -> None:
        """Test successful command execution."""
        self.mixin.container_api.execute_command.return_value = (0, "mock_output")  # type: ignore[attr-defined]

        result = self.mixin.execute_command("ls", workdir="/test")

        assert isinstance(result, ConsoleOutput)
        assert result.exit_code == 0
        assert result.stdout == "stdout"
        assert result.stderr == "stderr"

        self.mixin.container_api.execute_command.assert_called_once_with(  # type: ignore[attr-defined]
            self.mixin.container, "ls", workdir="/test", stream=False
        )

    def test_execute_command_with_workdir(self) -> None:
        """Test execute_command with specific workdir."""
        self.mixin.container_api.execute_command.return_value = (0, "output")  # type: ignore[attr-defined]

        self.mixin.execute_command("pwd", workdir="/custom")

        self.mixin.container_api.execute_command.assert_called_once_with(  # type: ignore[attr-defined]
            self.mixin.container, "pwd", workdir="/custom", stream=False
        )

    def test_execute_command_verbose_logging(self) -> None:
        """Test that verbose logging works correctly."""
        self.mixin.container_api.execute_command.return_value = (0, "output")  # type: ignore[attr-defined]

        self.mixin.execute_command("test_command")

        # Check that logging was called
        self.mixin.logger.info.assert_called()
        self.mixin.logger.error.assert_called()

    def test_process_output_non_stream(self) -> None:
        """Test _process_output for non-streaming mode."""
        self.mixin.stream = False

        result = self.mixin._process_output("mock_output")

        assert result == ("stdout", "stderr")

    def test_process_output_stream(self) -> None:
        """Test _process_output for streaming mode."""
        self.mixin.stream = True

        result = self.mixin._process_output("mock_output")

        assert result == ("stream_stdout", "stream_stderr")


class TestContainerAPIProtocol:
    """Test the ContainerAPI protocol."""

    def test_container_api_protocol(self) -> None:
        """Test that ContainerAPI protocol methods are defined correctly."""
        # This test ensures the protocol is properly defined
        api = MockContainerAPI()

        # Test that all required methods exist
        assert hasattr(api, "create_container")
        assert hasattr(api, "start_container")
        assert hasattr(api, "stop_container")
        assert hasattr(api, "execute_command")
        assert hasattr(api, "copy_to_container")
        assert hasattr(api, "copy_from_container")

        # Test that methods are callable
        assert callable(api.create_container)
        assert callable(api.start_container)
        assert callable(api.stop_container)
        assert callable(api.execute_command)
        assert callable(api.copy_to_container)
        assert callable(api.copy_from_container)
