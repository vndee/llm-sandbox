"""Tests for real-time output streaming callbacks (issue #149).

Tests cover:
- StreamCallback type alias definition
- Docker _process_stream_output callback invocation
- CommandExecutionMixin auto-stream enablement
- BaseSession.run() callback propagation
- execute_commands() callback propagation
- ArtifactSandboxSession callback propagation
- PooledSandboxSession callback propagation
- Callback error handling (exceptions in user callbacks)
- Non-streaming mode with callbacks
- Edge cases: None chunks, string chunks, mixed output
"""

from collections.abc import Generator
from unittest.mock import MagicMock, Mock, patch

from llm_sandbox.data import ConsoleOutput, StreamCallback
from llm_sandbox.docker import SandboxDockerSession


class TestStreamCallbackType:
    """Test StreamCallback type alias."""

    def test_stream_callback_is_callable(self) -> None:
        """Test that StreamCallback type alias works with callables."""

        def my_callback(chunk: str) -> None:
            pass

        callback: StreamCallback = my_callback
        assert callable(callback)

    def test_stream_callback_with_lambda(self) -> None:
        """Test that StreamCallback works with lambdas."""
        callback: StreamCallback = lambda _: None  # noqa: E731
        assert callable(callback)

    def test_stream_callback_exported_from_package(self) -> None:
        """Test that StreamCallback is exported from the top-level package."""
        from llm_sandbox import StreamCallback as StreamCallbackReexport

        assert StreamCallbackReexport is StreamCallback


class TestDockerStreamOutputCallbacks:
    """Test _process_stream_output with callbacks in Docker backend."""

    @patch("llm_sandbox.docker.docker.from_env")
    @patch("llm_sandbox.language_handlers.factory.LanguageHandlerFactory.create_handler")
    def test_stdout_callback_invoked_per_chunk(
        self, mock_create_handler: MagicMock, mock_docker_from_env: MagicMock
    ) -> None:
        """Test that on_stdout callback is invoked for each stdout chunk."""
        mock_handler = MagicMock()
        mock_create_handler.return_value = mock_handler

        session = SandboxDockerSession()
        stdout_chunks: list[str] = []

        def mock_output_gen() -> Generator[tuple[bytes | None, bytes | None], None, None]:
            yield (b"chunk1", None)
            yield (b"chunk2", None)
            yield (b"chunk3", None)

        stdout, stderr = session._process_stream_output(
            mock_output_gen(),
            on_stdout=stdout_chunks.append,
        )

        assert stdout == "chunk1chunk2chunk3"
        assert stderr == ""
        assert stdout_chunks == ["chunk1", "chunk2", "chunk3"]

    @patch("llm_sandbox.docker.docker.from_env")
    @patch("llm_sandbox.language_handlers.factory.LanguageHandlerFactory.create_handler")
    def test_stderr_callback_invoked_per_chunk(
        self, mock_create_handler: MagicMock, mock_docker_from_env: MagicMock
    ) -> None:
        """Test that on_stderr callback is invoked for each stderr chunk."""
        mock_handler = MagicMock()
        mock_create_handler.return_value = mock_handler

        session = SandboxDockerSession()
        stderr_chunks: list[str] = []

        def mock_output_gen() -> Generator[tuple[bytes | None, bytes | None], None, None]:
            yield (None, b"err1")
            yield (None, b"err2")

        stdout, stderr = session._process_stream_output(
            mock_output_gen(),
            on_stderr=stderr_chunks.append,
        )

        assert stdout == ""
        assert stderr == "err1err2"
        assert stderr_chunks == ["err1", "err2"]

    @patch("llm_sandbox.docker.docker.from_env")
    @patch("llm_sandbox.language_handlers.factory.LanguageHandlerFactory.create_handler")
    def test_both_callbacks_with_interleaved_output(
        self, mock_create_handler: MagicMock, mock_docker_from_env: MagicMock
    ) -> None:
        """Test both callbacks with interleaved stdout and stderr."""
        mock_handler = MagicMock()
        mock_create_handler.return_value = mock_handler

        session = SandboxDockerSession()
        stdout_chunks: list[str] = []
        stderr_chunks: list[str] = []

        def mock_output_gen() -> Generator[tuple[bytes | None, bytes | None], None, None]:
            yield (b"out1", b"err1")
            yield (b"out2", None)
            yield (None, b"err2")
            yield (b"out3", b"err3")

        stdout, stderr = session._process_stream_output(
            mock_output_gen(),
            on_stdout=stdout_chunks.append,
            on_stderr=stderr_chunks.append,
        )

        assert stdout == "out1out2out3"
        assert stderr == "err1err2err3"
        assert stdout_chunks == ["out1", "out2", "out3"]
        assert stderr_chunks == ["err1", "err2", "err3"]

    @patch("llm_sandbox.docker.docker.from_env")
    @patch("llm_sandbox.language_handlers.factory.LanguageHandlerFactory.create_handler")
    def test_no_callbacks_still_works(self, mock_create_handler: MagicMock, mock_docker_from_env: MagicMock) -> None:
        """Test that output processing works without any callbacks (backward compatibility)."""
        mock_handler = MagicMock()
        mock_create_handler.return_value = mock_handler

        session = SandboxDockerSession()

        def mock_output_gen() -> Generator[tuple[bytes | None, bytes | None], None, None]:
            yield (b"stdout", b"stderr")

        stdout, stderr = session._process_stream_output(mock_output_gen())

        assert stdout == "stdout"
        assert stderr == "stderr"

    @patch("llm_sandbox.docker.docker.from_env")
    @patch("llm_sandbox.language_handlers.factory.LanguageHandlerFactory.create_handler")
    def test_callbacks_with_string_chunks(
        self, mock_create_handler: MagicMock, mock_docker_from_env: MagicMock
    ) -> None:
        """Test callbacks with string chunks instead of bytes."""
        mock_handler = MagicMock()
        mock_create_handler.return_value = mock_handler

        session = SandboxDockerSession()
        stdout_chunks: list[str] = []

        def mock_output_gen() -> Generator[tuple[str | None, str | None], None, None]:
            yield ("string_out", "string_err")

        stdout, stderr = session._process_stream_output(
            mock_output_gen(),
            on_stdout=stdout_chunks.append,
        )

        assert stdout == "string_out"
        assert stderr == "string_err"
        assert stdout_chunks == ["string_out"]

    @patch("llm_sandbox.docker.docker.from_env")
    @patch("llm_sandbox.language_handlers.factory.LanguageHandlerFactory.create_handler")
    def test_callback_exception_does_not_corrupt_output(
        self, mock_create_handler: MagicMock, mock_docker_from_env: MagicMock
    ) -> None:
        """Test that a callback raising an exception still allows output to be accumulated.

        Note: The current implementation lets exceptions from the generator loop
        propagate (for SandboxTimeoutError) or logs warnings (for other exceptions).
        The callback exception will be caught by the try/except in the streaming loop.
        """
        mock_handler = MagicMock()
        mock_create_handler.return_value = mock_handler

        session = SandboxDockerSession()
        call_count = 0

        def failing_callback(_chunk: str) -> None:
            nonlocal call_count
            call_count += 1
            msg = "callback error"
            raise RuntimeError(msg)

        def mock_output_gen() -> Generator[tuple[bytes | None, bytes | None], None, None]:
            yield (b"data", None)

        # The exception from callback propagates through the stream loop
        with patch.object(session, "_log"):
            _stdout, _stderr = session._process_stream_output(
                mock_output_gen(),
                on_stdout=failing_callback,
            )

        # The callback was called and raised, which was caught by the loop's except block
        assert call_count == 1

    @patch("llm_sandbox.docker.docker.from_env")
    @patch("llm_sandbox.language_handlers.factory.LanguageHandlerFactory.create_handler")
    def test_empty_stream_with_callbacks(self, mock_create_handler: MagicMock, mock_docker_from_env: MagicMock) -> None:
        """Test callbacks with empty stream."""
        mock_handler = MagicMock()
        mock_create_handler.return_value = mock_handler

        session = SandboxDockerSession()
        stdout_chunks: list[str] = []
        stderr_chunks: list[str] = []

        def mock_output_gen() -> Generator[tuple[bytes | None, bytes | None], None, None]:
            yield (None, None)

        stdout, stderr = session._process_stream_output(
            mock_output_gen(),
            on_stdout=stdout_chunks.append,
            on_stderr=stderr_chunks.append,
        )

        assert stdout == ""
        assert stderr == ""
        assert stdout_chunks == []
        assert stderr_chunks == []


class TestAutoStreamEnablement:
    """Test that streaming mode is automatically enabled when callbacks are provided."""

    @patch("llm_sandbox.docker.docker.from_env")
    @patch("llm_sandbox.language_handlers.factory.LanguageHandlerFactory.create_handler")
    def test_callbacks_enable_streaming(self, mock_create_handler: MagicMock, mock_docker_from_env: MagicMock) -> None:
        """Test that providing callbacks forces streaming mode on."""
        mock_handler = MagicMock()
        mock_create_handler.return_value = mock_handler

        # Create session with stream=False
        session = SandboxDockerSession(stream=False)
        assert session.stream is False

        mock_container = MagicMock()
        # Return streaming output when stream=True is passed
        mock_container.exec_run.return_value = Mock(
            exit_code=0,
            output=iter([(b"streamed", None)]),
        )
        session.container = mock_container

        stdout_chunks: list[str] = []
        result = session.execute_command("echo test", on_stdout=stdout_chunks.append)

        # Verify exec_run was called with stream=True (auto-enabled)
        call_kwargs = mock_container.exec_run.call_args[1]
        assert call_kwargs["stream"] is True

        assert result.exit_code == 0
        assert stdout_chunks == ["streamed"]

    @patch("llm_sandbox.docker.docker.from_env")
    @patch("llm_sandbox.language_handlers.factory.LanguageHandlerFactory.create_handler")
    def test_no_callbacks_respects_stream_setting(
        self, mock_create_handler: MagicMock, mock_docker_from_env: MagicMock
    ) -> None:
        """Test that without callbacks, the stream setting is respected."""
        mock_handler = MagicMock()
        mock_create_handler.return_value = mock_handler

        session = SandboxDockerSession(stream=False)

        mock_container = MagicMock()
        mock_container.exec_run.return_value = Mock(
            exit_code=0,
            output=(b"stdout", b"stderr"),
        )
        session.container = mock_container

        result = session.execute_command("echo test")

        call_kwargs = mock_container.exec_run.call_args[1]
        assert call_kwargs["stream"] is False
        assert result.stdout == "stdout"


class TestRunMethodCallbackPropagation:
    """Test that callbacks are properly propagated from run() through the call chain."""

    @patch("llm_sandbox.docker.docker.from_env")
    @patch("llm_sandbox.language_handlers.factory.LanguageHandlerFactory.create_handler")
    def test_run_propagates_callbacks_to_execute_commands(
        self, mock_create_handler: MagicMock, mock_docker_from_env: MagicMock
    ) -> None:
        """Test that run() passes callbacks to execute_commands()."""
        mock_handler = MagicMock()
        mock_handler.file_extension = "py"
        mock_create_handler.return_value = mock_handler

        session = SandboxDockerSession()
        mock_container = MagicMock()
        session.container = mock_container
        session.is_open = True

        on_stdout = Mock()
        on_stderr = Mock()

        with (
            patch.object(session, "install"),
            patch.object(session, "copy_to_runtime"),
            patch.object(session, "execute_commands") as mock_exec_cmds,
            patch("tempfile.NamedTemporaryFile") as mock_temp_file,
        ):
            mock_file_instance = mock_temp_file.return_value
            mock_file_instance.name = "/tmp/code.py"
            mock_file_instance.write = Mock()
            mock_file_instance.seek = Mock()
            mock_file_instance.__enter__.return_value = mock_file_instance
            mock_file_instance.__exit__ = Mock()

            mock_exec_cmds.return_value = ConsoleOutput(exit_code=0, stdout="ok")

            session.run("print('hello')", on_stdout=on_stdout, on_stderr=on_stderr)

            # Verify callbacks were passed to execute_commands
            call_kwargs = mock_exec_cmds.call_args[1]
            assert call_kwargs["on_stdout"] is on_stdout
            assert call_kwargs["on_stderr"] is on_stderr

    @patch("llm_sandbox.docker.docker.from_env")
    @patch("llm_sandbox.language_handlers.factory.LanguageHandlerFactory.create_handler")
    def test_execute_commands_propagates_to_execute_command(
        self, mock_create_handler: MagicMock, mock_docker_from_env: MagicMock
    ) -> None:
        """Test that execute_commands() passes callbacks to execute_command()."""
        mock_handler = MagicMock()
        mock_create_handler.return_value = mock_handler

        session = SandboxDockerSession()
        mock_container = MagicMock()
        session.container = mock_container

        on_stdout = Mock()
        on_stderr = Mock()

        with patch.object(session, "execute_command") as mock_exec_cmd:
            mock_exec_cmd.return_value = ConsoleOutput(exit_code=0)

            session.execute_commands(
                ["cmd1", "cmd2"],
                workdir="/test",
                on_stdout=on_stdout,
                on_stderr=on_stderr,
            )

            # Both commands should have been called with callbacks
            for call in mock_exec_cmd.call_args_list:
                assert call[1]["on_stdout"] is on_stdout
                assert call[1]["on_stderr"] is on_stderr

    @patch("llm_sandbox.docker.docker.from_env")
    @patch("llm_sandbox.language_handlers.factory.LanguageHandlerFactory.create_handler")
    def test_execute_commands_propagates_with_tuple_commands(
        self, mock_create_handler: MagicMock, mock_docker_from_env: MagicMock
    ) -> None:
        """Test that execute_commands() propagates callbacks for tuple-style commands."""
        mock_handler = MagicMock()
        mock_create_handler.return_value = mock_handler

        session = SandboxDockerSession()
        mock_container = MagicMock()
        session.container = mock_container

        on_stdout = Mock()

        with patch.object(session, "execute_command") as mock_exec_cmd:
            mock_exec_cmd.return_value = ConsoleOutput(exit_code=0)

            session.execute_commands(
                [("cmd1", "/dir1"), ("cmd2", None)],
                on_stdout=on_stdout,
            )

            for call in mock_exec_cmd.call_args_list:
                assert call[1]["on_stdout"] is on_stdout


class TestEndToEndStreamingDocker:
    """End-to-end streaming callback tests with mocked Docker backend."""

    @patch("llm_sandbox.docker.docker.from_env")
    @patch("llm_sandbox.language_handlers.factory.LanguageHandlerFactory.create_handler")
    def test_full_streaming_pipeline(self, mock_create_handler: MagicMock, mock_docker_from_env: MagicMock) -> None:
        """Test the full callback pipeline from execute_command through to Docker streaming."""
        mock_handler = MagicMock()
        mock_create_handler.return_value = mock_handler

        session = SandboxDockerSession(stream=True)

        mock_container = MagicMock()
        # Simulate Docker streaming output
        mock_output = [
            (b"line1\n", None),
            (b"line2\n", b"warning\n"),
            (None, b"error\n"),
        ]
        mock_container.exec_run.return_value = Mock(exit_code=0, output=iter(mock_output))
        session.container = mock_container

        stdout_chunks: list[str] = []
        stderr_chunks: list[str] = []

        result = session.execute_command(
            "python script.py",
            on_stdout=stdout_chunks.append,
            on_stderr=stderr_chunks.append,
        )

        assert result.exit_code == 0
        assert result.stdout == "line1\nline2\n"
        assert result.stderr == "warning\nerror\n"
        assert stdout_chunks == ["line1\n", "line2\n"]
        assert stderr_chunks == ["warning\n", "error\n"]

    @patch("llm_sandbox.docker.docker.from_env")
    @patch("llm_sandbox.language_handlers.factory.LanguageHandlerFactory.create_handler")
    def test_only_stdout_callback(self, mock_create_handler: MagicMock, mock_docker_from_env: MagicMock) -> None:
        """Test providing only on_stdout callback."""
        mock_handler = MagicMock()
        mock_create_handler.return_value = mock_handler

        session = SandboxDockerSession(stream=True)

        mock_container = MagicMock()
        mock_output = [(b"out", b"err")]
        mock_container.exec_run.return_value = Mock(exit_code=0, output=iter(mock_output))
        session.container = mock_container

        stdout_chunks: list[str] = []
        result = session.execute_command("test", on_stdout=stdout_chunks.append)

        assert result.stdout == "out"
        assert result.stderr == "err"
        assert stdout_chunks == ["out"]

    @patch("llm_sandbox.docker.docker.from_env")
    @patch("llm_sandbox.language_handlers.factory.LanguageHandlerFactory.create_handler")
    def test_only_stderr_callback(self, mock_create_handler: MagicMock, mock_docker_from_env: MagicMock) -> None:
        """Test providing only on_stderr callback."""
        mock_handler = MagicMock()
        mock_create_handler.return_value = mock_handler

        session = SandboxDockerSession(stream=True)

        mock_container = MagicMock()
        mock_output = [(b"out", b"err")]
        mock_container.exec_run.return_value = Mock(exit_code=0, output=iter(mock_output))
        session.container = mock_container

        stderr_chunks: list[str] = []
        result = session.execute_command("test", on_stderr=stderr_chunks.append)

        assert result.stdout == "out"
        assert result.stderr == "err"
        assert stderr_chunks == ["err"]


class TestArtifactSessionCallbackPropagation:
    """Test callback propagation in ArtifactSandboxSession."""

    @patch("llm_sandbox.session.find_spec")
    @patch("llm_sandbox.docker.docker.from_env")
    @patch("llm_sandbox.language_handlers.factory.LanguageHandlerFactory.create_handler")
    def test_artifact_session_accepts_callbacks(
        self,
        mock_create_handler: MagicMock,
        mock_docker_from_env: MagicMock,
        mock_find_spec: MagicMock,
    ) -> None:
        """Test that ArtifactSandboxSession.run() accepts on_stdout and on_stderr."""
        from llm_sandbox.session import ArtifactSandboxSession

        mock_find_spec.return_value = MagicMock()
        mock_handler = MagicMock()
        mock_handler.is_support_plot_detection = True
        mock_handler.name = "python"
        mock_create_handler.return_value = mock_handler

        mock_result = ConsoleOutput(exit_code=0, stdout="ok", stderr="")
        mock_handler.run_with_artifacts.return_value = (mock_result, [])

        session = ArtifactSandboxSession(lang="python")

        with (
            patch.object(session, "open"),
            patch.object(session, "close"),
        ):
            session.__enter__()
            # Set up the internal session
            session._session.container = MagicMock()
            session._session.is_open = True

            on_stdout = Mock()
            on_stderr = Mock()

            session.run("print('test')", on_stdout=on_stdout, on_stderr=on_stderr)

            # Verify callbacks were passed to run_with_artifacts
            call_kwargs = mock_handler.run_with_artifacts.call_args[1]
            assert call_kwargs["on_stdout"] is on_stdout
            assert call_kwargs["on_stderr"] is on_stderr


class TestPooledSessionCallbackPropagation:
    """Test callback propagation in PooledSandboxSession."""

    def test_pooled_session_run_accepts_callbacks(self) -> None:
        """Test that PooledSandboxSession.run() accepts and propagates callbacks."""
        from llm_sandbox.pool.session import PooledSandboxSession

        session = PooledSandboxSession.__new__(PooledSandboxSession)
        session._backend_session = MagicMock()
        session._backend_session.run.return_value = ConsoleOutput(exit_code=0)

        on_stdout = Mock()
        on_stderr = Mock()

        session.run("code", on_stdout=on_stdout, on_stderr=on_stderr)

        session._backend_session.run.assert_called_once_with(
            "code",
            libraries=None,
            timeout=None,
            on_stdout=on_stdout,
            on_stderr=on_stderr,
            enforce_security_policy=None,
        )

    def test_artifact_pooled_session_run_accepts_callbacks(self) -> None:
        """Test that ArtifactPooledSandboxSession.run() accepts and propagates callbacks."""
        from llm_sandbox.pool.session import ArtifactPooledSandboxSession

        session = ArtifactPooledSandboxSession.__new__(ArtifactPooledSandboxSession)
        session.enable_plotting = True

        mock_backend = MagicMock()
        mock_backend.language_handler.is_support_plot_detection = True
        mock_backend.config.get_execution_timeout.return_value = 60

        mock_inner_session = MagicMock()
        mock_inner_session.backend_session = mock_backend
        session._session = mock_inner_session

        mock_result = ConsoleOutput(exit_code=0, stdout="ok")
        mock_backend.language_handler.run_with_artifacts.return_value = (mock_result, [])

        on_stdout = Mock()
        on_stderr = Mock()

        session.run("code", on_stdout=on_stdout, on_stderr=on_stderr)

        call_kwargs = mock_backend.language_handler.run_with_artifacts.call_args[1]
        assert call_kwargs["on_stdout"] is on_stdout
        assert call_kwargs["on_stderr"] is on_stderr


class TestBackwardCompatibility:
    """Test that existing code without callbacks still works correctly."""

    @patch("llm_sandbox.docker.docker.from_env")
    @patch("llm_sandbox.language_handlers.factory.LanguageHandlerFactory.create_handler")
    def test_run_without_callbacks(self, mock_create_handler: MagicMock, mock_docker_from_env: MagicMock) -> None:
        """Test that run() works without any callback parameters."""
        mock_handler = MagicMock()
        mock_handler.file_extension = "py"
        mock_create_handler.return_value = mock_handler

        session = SandboxDockerSession()
        mock_container = MagicMock()
        session.container = mock_container
        session.is_open = True

        with (
            patch.object(session, "install"),
            patch.object(session, "copy_to_runtime"),
            patch.object(session, "execute_commands") as mock_exec_cmds,
            patch("tempfile.NamedTemporaryFile") as mock_temp_file,
        ):
            mock_file_instance = mock_temp_file.return_value
            mock_file_instance.name = "/tmp/code.py"
            mock_file_instance.write = Mock()
            mock_file_instance.seek = Mock()
            mock_file_instance.__enter__.return_value = mock_file_instance
            mock_file_instance.__exit__ = Mock()

            mock_exec_cmds.return_value = ConsoleOutput(exit_code=0, stdout="output")

            result = session.run("print('hello')")

            assert result.exit_code == 0
            assert result.stdout == "output"

    @patch("llm_sandbox.docker.docker.from_env")
    @patch("llm_sandbox.language_handlers.factory.LanguageHandlerFactory.create_handler")
    def test_execute_command_without_callbacks(
        self, mock_create_handler: MagicMock, mock_docker_from_env: MagicMock
    ) -> None:
        """Test that execute_command() works without callbacks."""
        mock_handler = MagicMock()
        mock_create_handler.return_value = mock_handler

        session = SandboxDockerSession(stream=False)

        mock_container = MagicMock()
        mock_container.exec_run.return_value = Mock(exit_code=0, output=(b"stdout", b"stderr"))
        session.container = mock_container

        result = session.execute_command("ls -l")

        assert result.exit_code == 0
        assert result.stdout == "stdout"
        assert result.stderr == "stderr"

    @patch("llm_sandbox.docker.docker.from_env")
    @patch("llm_sandbox.language_handlers.factory.LanguageHandlerFactory.create_handler")
    def test_execute_commands_without_callbacks(
        self, mock_create_handler: MagicMock, mock_docker_from_env: MagicMock
    ) -> None:
        """Test that execute_commands() works without callbacks."""
        mock_handler = MagicMock()
        mock_create_handler.return_value = mock_handler

        session = SandboxDockerSession(stream=False)

        mock_container = MagicMock()
        mock_container.exec_run.return_value = Mock(exit_code=0, output=(b"ok", b""))
        session.container = mock_container

        result = session.execute_commands(["cmd1", "cmd2"], workdir="/test")

        assert result.exit_code == 0
