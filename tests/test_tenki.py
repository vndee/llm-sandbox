"""Tests for Tenki backend implementation."""

import io
import tarfile
from collections.abc import Callable
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, Mock, patch

import pytest
from tenki_sandbox import CommandResult, FileInfo, MissingAuthTokenError, SandboxError

from llm_sandbox.const import SupportedLanguage
from llm_sandbox.data import ConsoleOutput
from llm_sandbox.exceptions import (
    CommandEmptyError,
    ContainerError,
    ExtraArgumentsError,
    NotOpenSessionError,
)
from llm_sandbox.security import SecurityPolicy
from llm_sandbox.tenki import SandboxTenkiSession, TenkiContainerAPI

MTIME_NS = 1_700_000_000_000_000_000
WORKDIR = "/home/tenki"


def make_command_result(exit_code: int | None = 0, stdout: bytes = b"", stderr: bytes = b"") -> CommandResult:
    """Build a real SDK CommandResult so tests fail if its shape changes."""
    return CommandResult(argv=["bash", "-lc", "cmd"], exit_code=exit_code, stdout=stdout, stderr=stderr)


@pytest.fixture
def mock_sandbox() -> MagicMock:
    """Build a stand-in for tenki_sandbox.Sandbox that returns real SDK result types."""
    sandbox = MagicMock()
    sandbox.id = "sbx-test"
    sandbox.state = "RUNNING"
    sandbox.shell.return_value = make_command_result()
    sandbox.fs.stat.return_value = FileInfo(
        path=f"{WORKDIR}/file.txt", size=5, mode=0o644, is_dir=False, modified_unix_ns=MTIME_NS
    )
    sandbox.fs.read_bytes.return_value = b"hello"
    return sandbox


@pytest.fixture
def mock_client(mock_sandbox: MagicMock) -> MagicMock:
    """Build a stand-in for tenki_sandbox.Client."""
    client = MagicMock()
    client.create.return_value = mock_sandbox
    client.get.return_value = mock_sandbox
    return client


@pytest.fixture
def tenki_session_factory(mock_client: MagicMock) -> Callable[..., SandboxTenkiSession]:
    """Create SandboxTenkiSession instances with a mocked client and language handler."""

    def _factory(**kwargs: Any) -> SandboxTenkiSession:
        kwargs.setdefault("client", mock_client)
        with patch("llm_sandbox.language_handlers.factory.LanguageHandlerFactory.create_handler") as mock_handler:
            handler = MagicMock()
            handler.name = "unused"  # keeps environment_setup out of the venv/go branches
            handler.file_extension = "py"
            handler.is_support_library_installation = True
            mock_handler.return_value = handler
            return SandboxTenkiSession(**kwargs)

    return _factory


@pytest.fixture
def opened_session_factory(
    tenki_session_factory: Callable[..., SandboxTenkiSession],
) -> Callable[..., SandboxTenkiSession]:
    """Create SandboxTenkiSession instances that have already been opened."""

    def _factory(**kwargs: Any) -> SandboxTenkiSession:
        session = tenki_session_factory(**kwargs)
        session.open()
        return session

    return _factory


class TestSandboxTenkiSessionInit:
    """Test SandboxTenkiSession initialization."""

    def test_init_with_defaults(self, tenki_session_factory: Callable[..., SandboxTenkiSession]) -> None:
        """Test initialization with default parameters."""
        session = tenki_session_factory()

        assert session.config.lang == SupportedLanguage.PYTHON
        assert session.config.verbose is False
        assert session.config.image is None
        assert session.config.workdir == WORKDIR
        assert session.stream is False
        assert session.is_open is False
        assert session.container is None

    def test_init_with_custom_client(
        self, tenki_session_factory: Callable[..., SandboxTenkiSession], mock_client: MagicMock
    ) -> None:
        """Test initialization with a caller-supplied client does not take ownership of it."""
        session = tenki_session_factory(client=mock_client)

        assert session._client is mock_client
        assert session._owns_client is False

    def test_init_without_client_defers_authentication(self) -> None:
        """Test that constructing a session needs no API key and builds no client."""
        with (
            patch("llm_sandbox.tenki.Client") as mock_client_cls,
            patch("llm_sandbox.language_handlers.factory.LanguageHandlerFactory.create_handler"),
        ):
            session = SandboxTenkiSession()

        assert session._client is None
        assert session._owns_client is True
        mock_client_cls.assert_not_called()

    def test_init_with_custom_params(self, tenki_session_factory: Callable[..., SandboxTenkiSession]) -> None:
        """Test initialization with custom parameters."""
        security_policy = SecurityPolicy(patterns=[], restricted_modules=[])

        session = tenki_session_factory(
            image="tenki/python:3.11",
            lang="java",
            verbose=False,
            stream=True,
            workdir="/custom",
            security_policy=security_policy,
            runtime_configs={"cpu_cores": 2},
            auth_token="test-token",  # noqa: S106
            base_url="https://api.example.tenki.cloud",
        )

        assert session.config.image == "tenki/python:3.11"
        assert session.config.lang == SupportedLanguage.JAVA
        assert session.config.workdir == "/custom"
        assert session.config.security_policy == security_policy
        assert session.config.runtime_configs == {"cpu_cores": 2}
        assert session.stream is True
        assert session._auth_token == "test-token"  # noqa: S105
        assert session._base_url == "https://api.example.tenki.cloud"

    def test_init_with_container_id_marks_existing(
        self, tenki_session_factory: Callable[..., SandboxTenkiSession]
    ) -> None:
        """Test that container_id puts the session in existing-sandbox mode."""
        session = tenki_session_factory(container_id="sbx-999")

        assert session.config.container_id == "sbx-999"
        assert session.using_existing_container is True

    def test_init_with_dockerfile_raises_error(self, tenki_session_factory: Callable[..., SandboxTenkiSession]) -> None:
        """Test initialization fails when a dockerfile is provided."""
        with pytest.raises(ExtraArgumentsError, match="does not build images from a Dockerfile"):
            tenki_session_factory(dockerfile="/path/to/Dockerfile")


class TestSandboxTenkiSessionClient:
    """Test lazy Tenki client construction."""

    def test_get_client_builds_client_from_auth_token(
        self, tenki_session_factory: Callable[..., SandboxTenkiSession]
    ) -> None:
        """Test the client is built from the supplied auth token and base URL."""
        session = tenki_session_factory(client=None, auth_token="test-token", base_url="https://example.invalid")  # noqa: S106

        with patch("llm_sandbox.tenki.Client") as mock_client_cls:
            client = session._get_client()

        mock_client_cls.assert_called_once_with(auth_token="test-token", base_url="https://example.invalid")  # noqa: S106
        assert client is mock_client_cls.return_value

    def test_get_client_is_cached(self, tenki_session_factory: Callable[..., SandboxTenkiSession]) -> None:
        """Test repeated calls reuse the same client."""
        session = tenki_session_factory(client=None, auth_token="test-token")  # noqa: S106

        with patch("llm_sandbox.tenki.Client") as mock_client_cls:
            first = session._get_client()
            second = session._get_client()

        mock_client_cls.assert_called_once()
        assert first is second

    def test_get_client_defers_credential_lookup_to_sdk(
        self, tenki_session_factory: Callable[..., SandboxTenkiSession], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Test no credential is resolved locally; the SDK owns the env fallback."""
        monkeypatch.delenv("TENKI_AUTH_TOKEN", raising=False)
        monkeypatch.delenv("TENKI_API_KEY", raising=False)
        session = tenki_session_factory(client=None)

        with patch("llm_sandbox.tenki.Client") as mock_client_cls, patch.object(session, "_log") as mock_log:
            session._get_client()

        mock_client_cls.assert_called_once_with(auth_token=None, base_url=None)
        assert not any(call.args[1:] == ("warning",) for call in mock_log.call_args_list)

    def test_get_client_propagates_sdk_missing_auth_error(
        self, tenki_session_factory: Callable[..., SandboxTenkiSession], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Test the SDK's own missing-credential error surfaces unwrapped."""
        monkeypatch.delenv("TENKI_AUTH_TOKEN", raising=False)
        monkeypatch.delenv("TENKI_API_KEY", raising=False)
        session = tenki_session_factory(client=None)

        with pytest.raises(MissingAuthTokenError, match="TENKI_AUTH_TOKEN or TENKI_API_KEY"):
            session._get_client()


class TestSandboxTenkiSessionOpen:
    """Test SandboxTenkiSession open functionality."""

    def test_open_creates_and_waits_for_sandbox(
        self,
        tenki_session_factory: Callable[..., SandboxTenkiSession],
        mock_client: MagicMock,
        mock_sandbox: MagicMock,
    ) -> None:
        """Test open creates a sandbox and waits until it is ready."""
        session = tenki_session_factory()

        session.open()

        mock_client.create.assert_called_once()
        mock_sandbox.wait_ready.assert_called_once()
        assert session.container is mock_sandbox
        assert session.is_open is True

    def test_open_passes_image(
        self, tenki_session_factory: Callable[..., SandboxTenkiSession], mock_client: MagicMock
    ) -> None:
        """Test the configured image is forwarded to Client.create."""
        session = tenki_session_factory(image="tenki/python:3.11")

        session.open()

        assert mock_client.create.call_args.kwargs["image"] == "tenki/python:3.11"

    def test_open_omits_image_when_not_configured(
        self, tenki_session_factory: Callable[..., SandboxTenkiSession], mock_client: MagicMock
    ) -> None:
        """Test no image key is sent when none is configured, so Tenki's default applies."""
        session = tenki_session_factory()

        session.open()

        assert "image" not in mock_client.create.call_args.kwargs

    def test_open_injects_pythonunbuffered(
        self, tenki_session_factory: Callable[..., SandboxTenkiSession], mock_client: MagicMock
    ) -> None:
        """Test PYTHONUNBUFFERED is set so stdout is not swallowed by buffering."""
        session = tenki_session_factory()

        session.open()

        assert mock_client.create.call_args.kwargs["env"]["PYTHONUNBUFFERED"] == "1"

    def test_open_forwards_runtime_configs(
        self, tenki_session_factory: Callable[..., SandboxTenkiSession], mock_client: MagicMock
    ) -> None:
        """Test runtime_configs are forwarded verbatim to Client.create."""
        session = tenki_session_factory(
            runtime_configs={"cpu_cores": 2, "memory_mb": 2048, "allow_outbound": False, "tags": ["ci"]}
        )

        session.open()

        create_kwargs = mock_client.create.call_args.kwargs
        assert create_kwargs["cpu_cores"] == 2
        assert create_kwargs["memory_mb"] == 2048
        assert create_kwargs["allow_outbound"] is False
        assert create_kwargs["tags"] == ["ci"]

    def test_open_preserves_user_env(
        self, tenki_session_factory: Callable[..., SandboxTenkiSession], mock_client: MagicMock
    ) -> None:
        """Test a user-supplied env dict is preserved alongside the injected default."""
        session = tenki_session_factory(runtime_configs={"env": {"MY_VAR": "value"}})

        session.open()

        env = mock_client.create.call_args.kwargs["env"]
        assert env["MY_VAR"] == "value"
        assert env["PYTHONUNBUFFERED"] == "1"

    def test_open_does_not_mutate_runtime_configs(
        self, tenki_session_factory: Callable[..., SandboxTenkiSession]
    ) -> None:
        """Test open leaves the caller's runtime_configs untouched."""
        runtime_configs: dict[str, Any] = {"env": {"MY_VAR": "value"}}
        session = tenki_session_factory(runtime_configs=runtime_configs)

        session.open()

        assert runtime_configs == {"env": {"MY_VAR": "value"}}

    def test_open_wraps_creation_errors(
        self, tenki_session_factory: Callable[..., SandboxTenkiSession], mock_client: MagicMock
    ) -> None:
        """Test SDK failures during creation surface as ContainerError."""
        mock_client.create.side_effect = SandboxError("quota exceeded")
        session = tenki_session_factory()

        with pytest.raises(ContainerError, match="Failed to create Tenki sandbox"):
            session.open()

    def test_open_runs_environment_setup(self, tenki_session_factory: Callable[..., SandboxTenkiSession]) -> None:
        """Test open sets up the language environment for a new sandbox."""
        session = tenki_session_factory()

        with patch.object(session, "environment_setup") as mock_setup:
            session.open()

        mock_setup.assert_called_once()

    def test_open_attaches_to_existing_sandbox(
        self, tenki_session_factory: Callable[..., SandboxTenkiSession], mock_client: MagicMock
    ) -> None:
        """Test open attaches to an existing sandbox instead of creating one."""
        session = tenki_session_factory(container_id="sbx-999")

        session.open()

        mock_client.get.assert_called_once_with("sbx-999")
        mock_client.create.assert_not_called()


class TestSandboxTenkiSessionClose:
    """Test SandboxTenkiSession close functionality."""

    def test_close_terminates_created_sandbox(
        self, opened_session_factory: Callable[..., SandboxTenkiSession], mock_sandbox: MagicMock
    ) -> None:
        """Test a sandbox this session created is terminated so no microVM is leaked."""
        session = opened_session_factory()

        session.close()

        mock_sandbox.terminate.assert_called_once()
        mock_sandbox.detach.assert_not_called()
        assert session.container is None
        assert session.is_open is False

    def test_close_detaches_existing_sandbox(
        self, opened_session_factory: Callable[..., SandboxTenkiSession], mock_sandbox: MagicMock
    ) -> None:
        """Test an attached sandbox is detached and left running."""
        session = opened_session_factory(container_id="sbx-999")

        session.close()

        mock_sandbox.detach.assert_called_once()
        mock_sandbox.terminate.assert_not_called()

    def test_close_is_idempotent(
        self, opened_session_factory: Callable[..., SandboxTenkiSession], mock_sandbox: MagicMock
    ) -> None:
        """Test closing twice terminates only once."""
        session = opened_session_factory()

        session.close()
        session.close()

        mock_sandbox.terminate.assert_called_once()

    def test_close_swallows_termination_errors(
        self, opened_session_factory: Callable[..., SandboxTenkiSession], mock_sandbox: MagicMock
    ) -> None:
        """Test a failed termination is logged rather than raised."""
        session = opened_session_factory()
        mock_sandbox.terminate.side_effect = SandboxError("already gone")

        with patch.object(session, "_log") as mock_log:
            session.close()

        assert any(call.args[1:] == ("error",) for call in mock_log.call_args_list)
        assert session.container is None

    def test_close_does_not_close_injected_client(
        self, opened_session_factory: Callable[..., SandboxTenkiSession], mock_client: MagicMock
    ) -> None:
        """Test a caller-supplied client outlives the session."""
        session = opened_session_factory(client=mock_client)

        session.close()

        mock_client.close.assert_not_called()
        assert session._client is mock_client

    def test_close_closes_owned_client(self, tenki_session_factory: Callable[..., SandboxTenkiSession]) -> None:
        """Test a client the session created is closed with it."""
        session = tenki_session_factory(client=None, auth_token="test-token")  # noqa: S106

        with patch("llm_sandbox.tenki.Client") as mock_client_cls:
            owned_client = mock_client_cls.return_value
            owned_client.create.return_value = MagicMock(shell=Mock(return_value=make_command_result()))
            session.open()
            session.close()

        owned_client.close.assert_called_once()
        assert session._client is None

    def test_close_swallows_client_errors(self, tenki_session_factory: Callable[..., SandboxTenkiSession]) -> None:
        """Test a failure closing the owned client is logged rather than raised."""
        session = tenki_session_factory(client=None, auth_token="test-token")  # noqa: S106

        with patch("llm_sandbox.tenki.Client") as mock_client_cls:
            owned_client = mock_client_cls.return_value
            owned_client.create.return_value = MagicMock(shell=Mock(return_value=make_command_result()))
            owned_client.close.side_effect = SandboxError("transport already closed")
            session.open()
            with patch.object(session, "_log") as mock_log:
                session.close()

        assert any(call.args[1:] == ("error",) for call in mock_log.call_args_list)
        assert session._client is None


class TestSandboxTenkiSessionCommands:
    """Test SandboxTenkiSession command execution."""

    def test_execute_command_success(
        self, opened_session_factory: Callable[..., SandboxTenkiSession], mock_sandbox: MagicMock
    ) -> None:
        """Test stdout, stderr and exit code are mapped from the SDK result."""
        session = opened_session_factory()
        mock_sandbox.shell.return_value = make_command_result(exit_code=1, stdout=b"out", stderr=b"err")

        result = session.execute_command("echo hi")

        assert isinstance(result, ConsoleOutput)
        assert result.exit_code == 1
        assert result.stdout == "out"
        assert result.stderr == "err"

    def test_execute_command_passes_workdir_as_cwd(
        self, opened_session_factory: Callable[..., SandboxTenkiSession], mock_sandbox: MagicMock
    ) -> None:
        """Test the workdir is translated to the SDK's cwd argument."""
        session = opened_session_factory()
        mock_sandbox.shell.reset_mock()

        session.execute_command("ls", workdir=WORKDIR)

        mock_sandbox.shell.assert_called_once_with("ls", cwd=WORKDIR)

    def test_execute_command_none_exit_code_becomes_zero(
        self, opened_session_factory: Callable[..., SandboxTenkiSession], mock_sandbox: MagicMock
    ) -> None:
        """Test a missing exit code is normalized to 0."""
        session = opened_session_factory()
        mock_sandbox.shell.return_value = make_command_result(exit_code=None)

        assert session.execute_command("true").exit_code == 0

    def test_execute_command_empty(self, opened_session_factory: Callable[..., SandboxTenkiSession]) -> None:
        """Test an empty command is rejected."""
        session = opened_session_factory()

        with pytest.raises(CommandEmptyError):
            session.execute_command("")

    def test_execute_command_no_container(self, tenki_session_factory: Callable[..., SandboxTenkiSession]) -> None:
        """Test executing a command without an open session is rejected."""
        session = tenki_session_factory()

        with pytest.raises(NotOpenSessionError):
            session.execute_command("ls")


class TestSandboxTenkiSessionOutputProcessing:
    """Test SandboxTenkiSession output decoding and streaming callbacks."""

    def test_process_non_stream_output(self, tenki_session_factory: Callable[..., SandboxTenkiSession]) -> None:
        """Test a CommandResult is decoded into stdout and stderr strings."""
        session = tenki_session_factory()

        stdout, stderr = session._process_non_stream_output(make_command_result(stdout=b"out", stderr=b"err"))

        assert (stdout, stderr) == ("out", "err")

    def test_process_non_stream_output_empty(self, tenki_session_factory: Callable[..., SandboxTenkiSession]) -> None:
        """Test empty output decodes to empty strings."""
        session = tenki_session_factory()

        assert session._process_non_stream_output(make_command_result()) == ("", "")

    def test_process_stream_output_invokes_callbacks_once(
        self, tenki_session_factory: Callable[..., SandboxTenkiSession]
    ) -> None:
        """Test callbacks receive the whole output in a single chunk."""
        session = tenki_session_factory()
        stdout_chunks: list[str] = []
        stderr_chunks: list[str] = []

        stdout, stderr = session._process_stream_output(
            make_command_result(stdout=b"out", stderr=b"err"),
            on_stdout=stdout_chunks.append,
            on_stderr=stderr_chunks.append,
        )

        assert stdout_chunks == ["out"]
        assert stderr_chunks == ["err"]
        assert (stdout, stderr) == ("out", "err")

    def test_process_stream_output_skips_empty_streams(
        self, tenki_session_factory: Callable[..., SandboxTenkiSession]
    ) -> None:
        """Test callbacks are not invoked for empty streams."""
        session = tenki_session_factory()
        on_stdout = Mock()
        on_stderr = Mock()

        session._process_stream_output(make_command_result(stdout=b"out"), on_stdout=on_stdout, on_stderr=on_stderr)

        on_stdout.assert_called_once_with("out")
        on_stderr.assert_not_called()

    def test_process_stream_output_without_callbacks(
        self, tenki_session_factory: Callable[..., SandboxTenkiSession]
    ) -> None:
        """Test the streaming path still accumulates output when no callbacks are given."""
        session = tenki_session_factory()

        assert session._process_stream_output(make_command_result(stdout=b"out", stderr=b"err")) == ("out", "err")

    def test_execute_command_with_callbacks(
        self, opened_session_factory: Callable[..., SandboxTenkiSession], mock_sandbox: MagicMock
    ) -> None:
        """Test callbacks passed to execute_command are wired through the streaming path."""
        session = opened_session_factory()
        mock_sandbox.shell.return_value = make_command_result(stdout=b"streamed")
        chunks: list[str] = []

        result = session.execute_command("echo streamed", on_stdout=chunks.append)

        assert chunks == ["streamed"]
        assert result.stdout == "streamed"

    def test_decode_honours_encoding_errors(self, tenki_session_factory: Callable[..., SandboxTenkiSession]) -> None:
        """Test invalid bytes are handled per the configured encoding_errors mode."""
        session = tenki_session_factory(encoding_errors="replace")

        stdout, _ = session._process_non_stream_output(make_command_result(stdout=b"\xff"))

        assert stdout == "�"

    def test_decode_strict_raises_on_invalid_bytes(
        self, tenki_session_factory: Callable[..., SandboxTenkiSession]
    ) -> None:
        """Test the default strict mode surfaces undecodable output."""
        session = tenki_session_factory()

        with pytest.raises(UnicodeDecodeError):
            session._process_non_stream_output(make_command_result(stdout=b"\xff"))


class TestSandboxTenkiSessionRun:
    """Test SandboxTenkiSession run functionality."""

    def test_run_success(self, opened_session_factory: Callable[..., SandboxTenkiSession]) -> None:
        """Test successful code execution."""
        session = opened_session_factory()

        with (
            patch.object(session, "install") as mock_install,
            patch.object(session, "copy_to_runtime"),
            patch.object(session, "execute_commands") as mock_execute,
        ):
            expected = ConsoleOutput(exit_code=0, stdout="hello")
            mock_execute.return_value = expected

            result = session.run("print('hello')", ["numpy"])

        assert result == expected
        mock_install.assert_called_once_with(["numpy"])

    def test_run_copies_code_into_sandbox(self, opened_session_factory: Callable[..., SandboxTenkiSession]) -> None:
        """Test the generated code file is copied into the configured workdir."""
        session = opened_session_factory()

        with (
            patch.object(session, "copy_to_runtime") as mock_copy,
            patch.object(session, "execute_commands", return_value=ConsoleOutput()),
        ):
            session.run("print('hello')")

        dest = mock_copy.call_args.args[1]
        assert dest.startswith(f"{WORKDIR}/")
        assert dest.endswith(".py")

    def test_run_without_open_session(self, tenki_session_factory: Callable[..., SandboxTenkiSession]) -> None:
        """Test run fails when the session is not open."""
        session = tenki_session_factory()

        with pytest.raises(NotOpenSessionError):
            session.run("print('hello')")


class TestSandboxTenkiSessionFileOperations:
    """Test SandboxTenkiSession file operations."""

    def test_copy_to_runtime_uploads_file(
        self,
        opened_session_factory: Callable[..., SandboxTenkiSession],
        mock_sandbox: MagicMock,
        tmp_path: Path,
    ) -> None:
        """Test copy_to_runtime creates the parent directory and uploads the file."""
        session = opened_session_factory()
        src = tmp_path / "code.py"
        src.write_text("print('hello')")

        session.copy_to_runtime(str(src), f"{WORKDIR}/code.py")

        # The parent directory is created over exec, not fs.mkdir: the guest FS API rejects
        # the workdir itself, which is exactly the parent when copying into WORKDIR.
        mock_sandbox.shell.assert_any_call(f"mkdir -p {WORKDIR}", cwd=None)
        mock_sandbox.fs.upload.assert_called_once_with(str(src), f"{WORKDIR}/code.py")

    def test_copy_to_runtime_no_container(self, tenki_session_factory: Callable[..., SandboxTenkiSession]) -> None:
        """Test copy_to_runtime fails when the session is not open."""
        session = tenki_session_factory()

        with pytest.raises(NotOpenSessionError):
            session.copy_to_runtime("/host/file.txt", f"{WORKDIR}/file.txt")

    def test_copy_from_runtime_writes_host_file(
        self, opened_session_factory: Callable[..., SandboxTenkiSession], tmp_path: Path
    ) -> None:
        """Test copy_from_runtime round-trips guest bytes onto the host."""
        session = opened_session_factory()
        dest = tmp_path / "out.txt"

        session.copy_from_runtime(f"{WORKDIR}/file.txt", str(dest))

        assert dest.read_bytes() == b"hello"

    def test_copy_from_runtime_file_not_found(
        self, opened_session_factory: Callable[..., SandboxTenkiSession], mock_sandbox: MagicMock, tmp_path: Path
    ) -> None:
        """Test a missing guest file surfaces as FileNotFoundError."""
        session = opened_session_factory()
        mock_sandbox.fs.stat.side_effect = SandboxError("no such file")

        with pytest.raises(FileNotFoundError):
            session.copy_from_runtime(f"{WORKDIR}/missing.txt", str(tmp_path / "out.txt"))

    def test_ensure_directory_exists(
        self, opened_session_factory: Callable[..., SandboxTenkiSession], mock_sandbox: MagicMock
    ) -> None:
        """Test directories are created recursively in the guest."""
        session = opened_session_factory()

        session._ensure_directory_exists(f"{WORKDIR}/nested/dir")

        mock_sandbox.shell.assert_any_call(f"mkdir -p {WORKDIR}/nested/dir", cwd=None)

    def test_ensure_directory_exists_logs_failure(
        self, opened_session_factory: Callable[..., SandboxTenkiSession], mock_sandbox: MagicMock
    ) -> None:
        """Test a failed mkdir is logged rather than raised."""
        session = opened_session_factory()
        mock_sandbox.shell.return_value = make_command_result(exit_code=1, stderr=b"permission denied")

        with patch.object(session, "_log") as mock_log:
            session._ensure_directory_exists("/forbidden")

        assert mock_log.call_args.args[1] == "error"
        assert "permission denied" in mock_log.call_args.args[0]


class TestSandboxTenkiSessionOwnership:
    """Test SandboxTenkiSession ownership handling."""

    def test_ensure_ownership_with_non_root_user(
        self, opened_session_factory: Callable[..., SandboxTenkiSession], mock_sandbox: MagicMock
    ) -> None:
        """Test paths are chowned when a non-root user is configured."""
        session = opened_session_factory(runtime_configs={"user": "sandbox"})
        mock_sandbox.shell.reset_mock()

        session._ensure_ownership([f"{WORKDIR}/venv"])

        mock_sandbox.shell.assert_called_once_with("chown -R sandbox /home/tenki/venv", privileged=True)

    def test_ensure_ownership_with_root_user(
        self, opened_session_factory: Callable[..., SandboxTenkiSession], mock_sandbox: MagicMock
    ) -> None:
        """Test no chown is issued for the root user."""
        session = opened_session_factory(runtime_configs={"user": "root"})
        mock_sandbox.shell.reset_mock()

        session._ensure_ownership([f"{WORKDIR}/venv"])

        mock_sandbox.shell.assert_not_called()

    def test_ensure_ownership_without_user(
        self, opened_session_factory: Callable[..., SandboxTenkiSession], mock_sandbox: MagicMock
    ) -> None:
        """Test no chown is issued when no user is configured."""
        session = opened_session_factory()
        mock_sandbox.shell.reset_mock()

        session._ensure_ownership([f"{WORKDIR}/venv"])

        mock_sandbox.shell.assert_not_called()


class TestSandboxTenkiExistingSandbox:
    """Test attaching to an existing Tenki sandbox."""

    def test_connect_waits_until_ready(
        self,
        tenki_session_factory: Callable[..., SandboxTenkiSession],
        mock_client: MagicMock,
        mock_sandbox: MagicMock,
    ) -> None:
        """Test the attached sandbox is confirmed ready before use."""
        session = tenki_session_factory(container_id="sbx-999")
        session.container_api = TenkiContainerAPI(mock_client)

        session._connect_to_existing_container("sbx-999")

        mock_sandbox.wait_ready.assert_called_once()
        assert session.container is mock_sandbox

    def test_connect_resumes_paused_sandbox(
        self,
        tenki_session_factory: Callable[..., SandboxTenkiSession],
        mock_client: MagicMock,
        mock_sandbox: MagicMock,
    ) -> None:
        """Test a paused sandbox is resumed before use."""
        session = tenki_session_factory(container_id="sbx-999")
        session.container_api = TenkiContainerAPI(mock_client)
        mock_sandbox.state = "PAUSED"

        session._connect_to_existing_container("sbx-999")

        mock_sandbox.resume.assert_called_once()

    def test_connect_does_not_resume_running_sandbox(
        self,
        tenki_session_factory: Callable[..., SandboxTenkiSession],
        mock_client: MagicMock,
        mock_sandbox: MagicMock,
    ) -> None:
        """Test a running sandbox is not resumed."""
        session = tenki_session_factory(container_id="sbx-999")
        session.container_api = TenkiContainerAPI(mock_client)

        session._connect_to_existing_container("sbx-999")

        mock_sandbox.resume.assert_not_called()

    def test_connect_wraps_errors(
        self, tenki_session_factory: Callable[..., SandboxTenkiSession], mock_client: MagicMock
    ) -> None:
        """Test lookup failures surface as ContainerError."""
        session = tenki_session_factory(container_id="sbx-missing")
        session.container_api = TenkiContainerAPI(mock_client)
        mock_client.get.side_effect = SandboxError("session not found")

        with pytest.raises(ContainerError, match="Failed to attach to Tenki sandbox sbx-missing"):
            session._connect_to_existing_container("sbx-missing")


class TestSandboxTenkiSessionTimeout:
    """Test SandboxTenkiSession timeout handling."""

    def test_handle_timeout_closes_session(
        self, opened_session_factory: Callable[..., SandboxTenkiSession], mock_sandbox: MagicMock
    ) -> None:
        """Test a timeout tears the sandbox down so no hung work keeps billing."""
        session = opened_session_factory()

        session._handle_timeout()

        mock_sandbox.terminate.assert_called_once()
        assert session.container is None

    def test_handle_timeout_swallows_errors(self, opened_session_factory: Callable[..., SandboxTenkiSession]) -> None:
        """Test cleanup errors during timeout handling are not propagated."""
        session = opened_session_factory()

        with patch.object(session, "close", side_effect=SandboxError("boom")), patch.object(session, "_log"):
            session._handle_timeout()


class TestSandboxTenkiSessionContextManager:
    """Test SandboxTenkiSession context manager behaviour."""

    def test_context_manager(
        self,
        tenki_session_factory: Callable[..., SandboxTenkiSession],
        mock_client: MagicMock,
        mock_sandbox: MagicMock,
    ) -> None:
        """Test the sandbox is created on enter and terminated on exit."""
        session = tenki_session_factory()

        with session as entered:
            assert entered is session
            assert session.is_open is True
            mock_client.create.assert_called_once()

        mock_sandbox.terminate.assert_called_once()
        assert session.is_open is False

    def test_context_manager_with_exception(
        self, tenki_session_factory: Callable[..., SandboxTenkiSession], mock_sandbox: MagicMock
    ) -> None:
        """Test the sandbox is still terminated when the block raises."""
        session = tenki_session_factory()
        error = ValueError("boom")

        with pytest.raises(ValueError, match="boom"), session:
            raise error

        mock_sandbox.terminate.assert_called_once()


class TestTenkiContainerAPI:
    """Test the TenkiContainerAPI adapter directly."""

    def test_execute_command_returns_exit_code_and_result(
        self, mock_client: MagicMock, mock_sandbox: MagicMock
    ) -> None:
        """Test execute_command returns the exit code alongside the raw result."""
        expected = make_command_result(exit_code=2, stdout=b"out")
        mock_sandbox.shell.return_value = expected

        exit_code, result = TenkiContainerAPI(mock_client).execute_command(mock_sandbox, "ls", workdir=WORKDIR)

        assert exit_code == 2
        assert result is expected

    def test_copy_from_container_builds_tar(self, mock_client: MagicMock, mock_sandbox: MagicMock) -> None:
        """Test copy_from_container wraps guest bytes in a Docker-shaped tar archive."""
        data, stat = TenkiContainerAPI(mock_client).copy_from_container(mock_sandbox, f"{WORKDIR}/file.txt")

        assert stat["size"] == 5
        assert stat["mode"] == 0o644
        assert stat["mtime"] == MTIME_NS // 1_000_000_000

        with tarfile.open(fileobj=io.BytesIO(data), mode="r") as tar:
            members = tar.getmembers()
            assert [m.name for m in members] == ["file.txt"]
            extracted = tar.extractfile(members[0])
            assert extracted is not None
            assert extracted.read() == b"hello"

    def test_copy_from_container_missing_file(self, mock_client: MagicMock, mock_sandbox: MagicMock) -> None:
        """Test a missing guest file yields an empty archive."""
        mock_sandbox.fs.stat.side_effect = SandboxError("no such file")

        assert TenkiContainerAPI(mock_client).copy_from_container(mock_sandbox, f"{WORKDIR}/missing.txt") == (
            b"",
            {"size": 0},
        )

    def test_copy_from_container_directory(self, mock_client: MagicMock, mock_sandbox: MagicMock) -> None:
        """Test directories are not archived by this backend."""
        mock_sandbox.fs.stat.return_value = FileInfo(
            path=f"{WORKDIR}/dir", size=0, mode=0o755, is_dir=True, modified_unix_ns=MTIME_NS
        )

        assert TenkiContainerAPI(mock_client).copy_from_container(mock_sandbox, f"{WORKDIR}/dir") == (b"", {"size": 0})
        mock_sandbox.fs.read_bytes.assert_not_called()
