"""Tests for Tenki backend implementation."""

import io
import tarfile
from collections.abc import Callable
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, Mock, patch

import pytest
from tenki import (
    Client,
    CommandResult,
    FileInfo,
    MissingAuthTokenError,
    SandboxError,
    SessionNotFoundError,
    SessionTerminatedError,
)

from llm_sandbox.data import ConsoleOutput
from llm_sandbox.exceptions import ContainerError, ExtraArgumentsError
from llm_sandbox.tenki import SandboxTenkiSession, TenkiContainerAPI

MTIME_NS = 1_700_000_000_000_000_000
WORKDIR = "/home/tenki"


def make_command_result(
    exit_code: int | None = 0,
    stdout: bytes = b"",
    stderr: bytes = b"",
    signal: str | None = None,
    reason: str | None = None,
    errno: int | None = None,
) -> CommandResult:
    """Build a real SDK CommandResult so tests fail if its shape changes."""
    return CommandResult(
        argv=["bash", "-lc", "cmd"],
        exit_code=exit_code,
        stdout=stdout,
        stderr=stderr,
        signal=signal,
        reason=reason,
        errno=errno,
    )


@pytest.fixture
def mock_sandbox() -> MagicMock:
    """Build a stand-in for tenki.Sandbox that returns real SDK result types."""
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
    """Build a stand-in for tenki.Client."""
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

    def test_init_with_custom_client(
        self, tenki_session_factory: Callable[..., SandboxTenkiSession], mock_client: MagicMock
    ) -> None:
        """Test a caller-supplied client is not owned by the session."""
        session = tenki_session_factory(client=mock_client)

        assert session._client is mock_client
        assert session._owns_client is False

    def test_init_without_client_defers_authentication(self) -> None:
        """Test constructing a session builds no client yet."""
        with (
            patch("llm_sandbox.tenki.Client") as mock_client_cls,
            patch("llm_sandbox.language_handlers.factory.LanguageHandlerFactory.create_handler"),
        ):
            session = SandboxTenkiSession()

        assert session._client is None
        assert session._owns_client is True
        mock_client_cls.assert_not_called()

    def test_init_with_container_id_marks_existing(
        self, tenki_session_factory: Callable[..., SandboxTenkiSession]
    ) -> None:
        """Test container_id puts the session in existing-sandbox mode."""
        session = tenki_session_factory(container_id="sbx-999")

        assert session.using_existing_container is True

    def test_init_with_dockerfile_raises_error(self, tenki_session_factory: Callable[..., SandboxTenkiSession]) -> None:
        """Test dockerfile= is rejected."""
        with pytest.raises(ExtraArgumentsError, match="does not build images from a Dockerfile"):
            tenki_session_factory(dockerfile="/path/to/Dockerfile")

    @pytest.mark.parametrize("lang", ["python", "javascript", "cpp"])
    def test_init_allows_default_guest_languages(
        self, tenki_session_factory: Callable[..., SandboxTenkiSession], lang: str
    ) -> None:
        """Test languages on the default guest do not require a custom image."""
        session = tenki_session_factory(lang=lang)

        assert session.config.lang.value == lang
        assert session.config.image is None

    @pytest.mark.parametrize("lang", ["java", "go", "ruby", "r"])
    def test_init_rejects_unsupported_default_guest_languages(
        self, tenki_session_factory: Callable[..., SandboxTenkiSession], lang: str
    ) -> None:
        """Test langs absent from the default guest fail fast without image=."""
        with pytest.raises(ExtraArgumentsError, match="default guest image does not include"):
            tenki_session_factory(lang=lang)

    def test_init_allows_unsupported_language_with_custom_image(
        self, tenki_session_factory: Callable[..., SandboxTenkiSession]
    ) -> None:
        """Test a custom image opts into languages the default guest lacks."""
        session = tenki_session_factory(lang="java", image="tenki/java:17")

        assert session.config.lang.value == "java"
        assert session.config.image == "tenki/java:17"

    def test_init_allows_unsupported_language_when_attaching(
        self, tenki_session_factory: Callable[..., SandboxTenkiSession]
    ) -> None:
        """Test attaching to an existing sandbox skips the default-guest language gate."""
        session = tenki_session_factory(lang="go", container_id="sbx-999")

        assert session.config.lang.value == "go"
        assert session.using_existing_container is True


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

    def test_get_client_propagates_sdk_missing_auth_error(
        self, tenki_session_factory: Callable[..., SandboxTenkiSession], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Test the SDK's missing-credential error surfaces unwrapped."""
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
        """Test open creates with wait=False, waits ready, and injects PYTHONUNBUFFERED."""
        session = tenki_session_factory(runtime_configs={"env": {"MY_VAR": "value"}, "cpu_cores": 2})

        session.open()

        create_kwargs = mock_client.create.call_args.kwargs
        assert create_kwargs["wait"] is False
        assert create_kwargs["cpu_cores"] == 2
        assert create_kwargs["env"]["MY_VAR"] == "value"
        assert create_kwargs["env"]["PYTHONUNBUFFERED"] == "1"
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

    def test_open_wraps_creation_errors(
        self, tenki_session_factory: Callable[..., SandboxTenkiSession], mock_client: MagicMock
    ) -> None:
        """Test SDK failures during creation surface as ContainerError."""
        mock_client.create.side_effect = SandboxError("quota exceeded")
        session = tenki_session_factory()

        with pytest.raises(ContainerError, match="Failed to create Tenki sandbox"):
            session.open()

    def test_open_attaches_to_existing_sandbox(
        self, tenki_session_factory: Callable[..., SandboxTenkiSession], mock_client: MagicMock
    ) -> None:
        """Test open attaches instead of creating when container_id is set."""
        session = tenki_session_factory(container_id="sbx-999")

        session.open()

        mock_client.get.assert_called_once_with("sbx-999")
        mock_client.create.assert_not_called()


class TestSandboxTenkiSessionPythonBootstrap:
    """Test guest Python alias and venv verification gating."""

    def _bootstrap_calls(self, mock_sandbox: MagicMock) -> list[str]:
        return [c.args[0] for c in mock_sandbox.shell.call_args_list if "command -v python3" in str(c.args[0])]

    def test_interpreter_is_aliased_for_new_python_sessions(
        self, tenki_session_factory: Callable[..., SandboxTenkiSession], mock_sandbox: MagicMock
    ) -> None:
        """Test the python3 alias runs for new Python sandboxes (venv or skip)."""
        for kwargs in ({}, {"skip_environment_setup": True}):
            mock_sandbox.shell.reset_mock()
            session = tenki_session_factory(lang="python", **kwargs)

            session.open()

            assert self._bootstrap_calls(mock_sandbox)

    def test_interpreter_is_not_touched_on_an_attached_sandbox(
        self, tenki_session_factory: Callable[..., SandboxTenkiSession], mock_sandbox: MagicMock
    ) -> None:
        """Test attaching does not modify the guest interpreter."""
        session = tenki_session_factory(lang="python", container_id="sbx-999")

        session.open()

        assert not self._bootstrap_calls(mock_sandbox)

    def test_venv_verification_follows_environment_setup(
        self, tenki_session_factory: Callable[..., SandboxTenkiSession]
    ) -> None:
        """Test venv verify runs only when environment_setup builds one."""
        session = tenki_session_factory(lang="python")
        with patch.object(session, "_verify_python_environment") as mock_verify:
            session.open()
        mock_verify.assert_called_once()

        session = tenki_session_factory(lang="python", skip_environment_setup=True)
        with patch.object(session, "_verify_python_environment") as mock_verify:
            session.open()
        mock_verify.assert_not_called()


class TestSandboxTenkiSessionCloseWhenSandboxAlreadyGone:
    """Test close succeeds when the sandbox is already gone elsewhere."""

    @pytest.mark.parametrize(
        "error",
        [
            SessionNotFoundError("session does not exist"),
            SessionTerminatedError("session already terminated"),
        ],
        ids=["not-found", "already-terminated"],
    )
    def test_close_succeeds_when_the_sandbox_is_already_gone(
        self,
        opened_session_factory: Callable[..., SandboxTenkiSession],
        mock_sandbox: MagicMock,
        error: Exception,
    ) -> None:
        """Test SDK 'already gone' errors clear the handle instead of failing forever."""
        session = opened_session_factory()
        mock_sandbox.terminate.side_effect = error

        session.close()

        assert session.container is None


class TestSandboxTenkiSessionCloseIsRetryable:
    """Test failed teardown raises and stays retryable."""

    def test_failed_termination_raises_and_keeps_the_handle(
        self, opened_session_factory: Callable[..., SandboxTenkiSession], mock_sandbox: MagicMock
    ) -> None:
        """Test a live sandbox that will not die raises and retains the handle."""
        session = opened_session_factory()
        mock_sandbox.terminate.side_effect = SandboxError("503 Service Unavailable")

        with pytest.raises(ContainerError, match="Failed to release Tenki sandbox"):
            session.close()

        assert session.container is mock_sandbox

    def test_failed_termination_keeps_the_client_alive_for_the_retry(
        self, tenki_session_factory: Callable[..., SandboxTenkiSession]
    ) -> None:
        """Test the owned client is not closed while a sandbox still needs releasing."""
        session = tenki_session_factory(client=None, auth_token="test-token")  # noqa: S106

        with patch("llm_sandbox.tenki.Client") as mock_client_cls:
            owned_client = mock_client_cls.return_value
            sandbox = MagicMock(shell=Mock(return_value=make_command_result()))
            sandbox.terminate.side_effect = SandboxError("already gone")
            owned_client.create.return_value = sandbox
            session.open()

            with pytest.raises(ContainerError):
                session.close()

        owned_client.close.assert_not_called()
        assert session._client is owned_client

    def test_retrying_close_releases_the_sandbox(
        self, opened_session_factory: Callable[..., SandboxTenkiSession], mock_sandbox: MagicMock
    ) -> None:
        """Test a second close() after a transient failure frees the microVM."""
        session = opened_session_factory()
        mock_sandbox.terminate.side_effect = [SandboxError("transient"), None]

        with pytest.raises(ContainerError):
            session.close()
        session.close()

        assert mock_sandbox.terminate.call_count == 2
        assert session.container is None

    def test_failed_detach_of_attached_sandbox_also_raises(
        self, opened_session_factory: Callable[..., SandboxTenkiSession], mock_sandbox: MagicMock
    ) -> None:
        """Test detach failures raise and keep the handle."""
        session = opened_session_factory(container_id="sbx-999")
        mock_sandbox.detach.side_effect = SandboxError("connection reset")

        with pytest.raises(ContainerError, match="Failed to release Tenki sandbox"):
            session.close()

        assert session.container is mock_sandbox

    def test_exit_does_not_let_cleanup_failure_mask_the_block_error(
        self, tenki_session_factory: Callable[..., SandboxTenkiSession], mock_sandbox: MagicMock
    ) -> None:
        """Test the caller's exception wins, with the cleanup failure chained onto it."""
        mock_sandbox.terminate.side_effect = SandboxError("terminate failed")
        session = tenki_session_factory()
        body_error = ValueError("user code failed")

        with pytest.raises(ValueError, match="user code failed") as raised, session:
            raise body_error

        assert isinstance(raised.value.__context__, ContainerError)
        assert "Failed to release Tenki sandbox" in str(raised.value.__context__)
        assert session.container is mock_sandbox

        mock_sandbox.terminate.side_effect = None
        session.close()
        assert session.container is None

    def test_exit_reports_failed_cleanup_while_preserving_the_block_error(
        self, tenki_session_factory: Callable[..., SandboxTenkiSession], mock_sandbox: MagicMock
    ) -> None:
        """Test a failed __exit__ cleanup is logged with the sandbox id."""
        mock_sandbox.terminate.side_effect = SandboxError("terminate failed")
        session = tenki_session_factory()
        session.verbose = True
        body_error = ValueError("user code failed")

        with (
            patch.object(session, "_log") as mock_log,
            pytest.raises(ValueError, match="user code failed"),
            session,
        ):
            raise body_error

        mock_log.assert_any_call(
            f"Tenki sandbox {mock_sandbox.id} is STILL RUNNING: "
            f"CLEANUP FAILED (Failed to release Tenki sandbox {mock_sandbox.id}: terminate failed). "
            "Call close() again to retry or terminate directly.",
            "error",
        )
        assert session.container is mock_sandbox

    def test_exit_raises_cleanup_failure_when_the_block_succeeded(
        self, tenki_session_factory: Callable[..., SandboxTenkiSession], mock_sandbox: MagicMock
    ) -> None:
        """Test a clean block still surfaces a leaked sandbox."""
        mock_sandbox.terminate.side_effect = SandboxError("already gone")
        session = tenki_session_factory()

        with pytest.raises(ContainerError, match="Failed to release Tenki sandbox"), session:
            pass


class TestSandboxTenkiSessionOpenIsFailureAtomic:
    """Test that a failed open() never silently abandons a running microVM."""

    def test_open_terminates_sandbox_when_readiness_fails(
        self, tenki_session_factory: Callable[..., SandboxTenkiSession], mock_sandbox: MagicMock
    ) -> None:
        """Test a sandbox that never becomes ready is terminated."""
        mock_sandbox.wait_ready.side_effect = SandboxError("boot timed out")
        session = tenki_session_factory()

        with pytest.raises(ContainerError, match="Failed to create Tenki sandbox"):
            session.open()

        mock_sandbox.terminate.assert_called_once()
        assert session.container is None

    def test_open_terminates_sandbox_when_preparation_fails(
        self, tenki_session_factory: Callable[..., SandboxTenkiSession], mock_sandbox: MagicMock
    ) -> None:
        """Test post-create failures release the microVM and chain cleanup errors."""
        mock_sandbox.terminate.side_effect = SandboxError("terminate failed")
        session = tenki_session_factory()

        with (
            patch.object(session, "environment_setup", side_effect=SandboxError("setup blew up")),
            pytest.raises(SandboxError, match="setup blew up") as raised,
        ):
            session.open()

        assert isinstance(raised.value.__context__, ContainerError)
        assert session.container is mock_sandbox

        mock_sandbox.terminate.side_effect = None
        session.close()
        assert session.container is None

    def test_open_terminates_sandbox_on_cancellation(
        self, tenki_session_factory: Callable[..., SandboxTenkiSession], mock_sandbox: MagicMock
    ) -> None:
        """Test Ctrl-C mid-provision terminates and propagates."""
        mock_sandbox.wait_ready.side_effect = KeyboardInterrupt
        session = tenki_session_factory()

        with pytest.raises(KeyboardInterrupt):
            session.open()

        mock_sandbox.terminate.assert_called_once()
        assert session.container is None

    def test_open_detaches_rather_than_terminates_an_attached_sandbox(
        self, tenki_session_factory: Callable[..., SandboxTenkiSession], mock_sandbox: MagicMock
    ) -> None:
        """Test cleanup never destroys a sandbox this session did not create."""
        session = tenki_session_factory(container_id="sbx-999")

        with (
            patch.object(session, "environment_setup", side_effect=SandboxError("setup blew up")),
            pytest.raises(SandboxError),
        ):
            session.open()

        mock_sandbox.detach.assert_called_once()
        mock_sandbox.terminate.assert_not_called()

    def test_a_failed_cleanup_is_reported_with_the_sandbox_id(
        self, tenki_session_factory: Callable[..., SandboxTenkiSession], mock_sandbox: MagicMock
    ) -> None:
        """Test a sandbox left running after failed open is logged with its id."""
        mock_sandbox.terminate.side_effect = SandboxError("terminate failed")
        session = tenki_session_factory()
        logged: list[tuple[str, str]] = []

        with (
            patch.object(session, "environment_setup", side_effect=SandboxError("setup blew up")),
            patch.object(session, "_log", side_effect=lambda m, lvl="info": logged.append((lvl, m))),
            pytest.raises(SandboxError),
        ):
            session.open()

        leaks = [m for lvl, m in logged if lvl == "error" and "STILL RUNNING" in m]
        assert leaks
        assert mock_sandbox.id in leaks[0]

    def test_reopening_raises_instead_of_orphaning_the_first_sandbox(
        self, tenki_session_factory: Callable[..., SandboxTenkiSession], mock_client: MagicMock
    ) -> None:
        """Test a second open() is rejected rather than dropping the running sandbox."""
        session = tenki_session_factory()
        session.open()

        with pytest.raises(ContainerError, match="already open"):
            session.open()

        mock_client.create.assert_called_once()


class TestSandboxTenkiSessionClose:
    """Test SandboxTenkiSession close functionality."""

    def test_close_terminates_created_sandbox(
        self, opened_session_factory: Callable[..., SandboxTenkiSession], mock_sandbox: MagicMock
    ) -> None:
        """Test a created sandbox is terminated on close."""
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

    def test_close_does_not_close_injected_client(
        self, opened_session_factory: Callable[..., SandboxTenkiSession], mock_client: MagicMock
    ) -> None:
        """Test a caller-supplied client outlives the session."""
        session = opened_session_factory(client=mock_client)

        session.close()

        mock_client.close.assert_not_called()

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
        """Test workdir is translated to the SDK's cwd argument."""
        session = opened_session_factory()
        mock_sandbox.shell.reset_mock()

        session.execute_command("ls", workdir=WORKDIR)

        mock_sandbox.shell.assert_called_once_with("ls", cwd=WORKDIR)

    def test_execute_command_unknown_exit_code_is_not_reported_as_success(
        self, opened_session_factory: Callable[..., SandboxTenkiSession], mock_sandbox: MagicMock
    ) -> None:
        """Test an absent exit code fails closed rather than passing as 0."""
        session = opened_session_factory()
        mock_sandbox.shell.return_value = make_command_result(exit_code=None)

        assert session.execute_command("true").exit_code != 0


class TestSandboxTenkiSessionSignalledCommands:
    """Test signalled commands are not mistaken for success."""

    def test_sigkill_with_zero_exit_code_is_a_failure(
        self, opened_session_factory: Callable[..., SandboxTenkiSession], mock_sandbox: MagicMock
    ) -> None:
        """Test SIGKILL with exit code 0 is reported as failure with detail on stderr."""
        session = opened_session_factory()
        mock_sandbox.shell.return_value = make_command_result(
            exit_code=0, stderr=b"Traceback: boom", signal="SIGKILL", reason="oom", errno=137
        )

        result = session.execute_command("stress")

        assert result.exit_code != 0
        assert "Traceback: boom" in result.stderr
        assert "SIGKILL" in result.stderr
        assert "oom" in result.stderr


class TestSandboxTenkiSessionOutputProcessing:
    """Test output decoding and streaming callbacks."""

    def test_execute_command_with_callbacks(
        self, opened_session_factory: Callable[..., SandboxTenkiSession], mock_sandbox: MagicMock
    ) -> None:
        """Test callbacks receive the full output once."""
        session = opened_session_factory()
        mock_sandbox.shell.return_value = make_command_result(stdout=b"streamed")
        chunks: list[str] = []

        result = session.execute_command("echo streamed", on_stdout=chunks.append)

        assert chunks == ["streamed"]
        assert result.stdout == "streamed"

    def test_decode_honours_encoding_errors(self, tenki_session_factory: Callable[..., SandboxTenkiSession]) -> None:
        """Test encoding_errors=replace vs strict."""
        session = tenki_session_factory(encoding_errors="replace")
        stdout, _ = session._process_non_stream_output(make_command_result(stdout=b"\xff"))
        assert stdout == "�"

        session = tenki_session_factory()
        with pytest.raises(UnicodeDecodeError):
            session._process_non_stream_output(make_command_result(stdout=b"\xff"))


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

        mock_sandbox.shell.assert_any_call(f"mkdir -p {WORKDIR}", cwd=None)
        mock_sandbox.fs.upload.assert_called_once_with(str(src), f"{WORKDIR}/code.py")

    def test_copy_to_runtime_uploads_directory(
        self,
        opened_session_factory: Callable[..., SandboxTenkiSession],
        mock_sandbox: MagicMock,
        tmp_path: Path,
    ) -> None:
        """Test directory copies walk the tree and upload each file under dest."""
        session = opened_session_factory()
        src_dir = tmp_path / "input_data"
        src_dir.mkdir()
        (src_dir / "data.json").write_text("{}")
        nested = src_dir / "nested"
        nested.mkdir()
        (nested / "extra.txt").write_text("x")

        session.copy_to_runtime(str(src_dir), f"{WORKDIR}/input")

        mock_sandbox.fs.upload.assert_any_call(str(src_dir / "data.json"), f"{WORKDIR}/input/data.json")
        mock_sandbox.fs.upload.assert_any_call(str(nested / "extra.txt"), f"{WORKDIR}/input/nested/extra.txt")

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


class TestSandboxTenkiSessionOwnership:
    """Test ownership is a no-op because Client.create has no user parameter."""

    def test_ensure_ownership_does_nothing(
        self, opened_session_factory: Callable[..., SandboxTenkiSession], mock_sandbox: MagicMock
    ) -> None:
        """Test no chown is issued."""
        session = opened_session_factory()
        mock_sandbox.shell.reset_mock()

        session._ensure_ownership([f"{WORKDIR}/venv"])

        mock_sandbox.shell.assert_not_called()

    def test_ensure_ownership_with_non_root_user(
        self, opened_session_factory: Callable[..., SandboxTenkiSession], mock_sandbox: MagicMock
    ) -> None:
        """Test no chown even when a non-root user is configured.

        Unlike Docker/Podman, Tenki cannot run as a second identity, so a ``user``
        runtime config must not trigger a privileged chown that would only confuse.
        """
        session = opened_session_factory(runtime_configs={"user": "sandbox"})
        mock_sandbox.shell.reset_mock()

        session._ensure_ownership([f"{WORKDIR}/venv"])

        mock_sandbox.shell.assert_not_called()

    def test_the_sdk_has_no_user_parameter_to_honour(self) -> None:
        """Test Client.create cannot express a non-root user."""
        import inspect

        with pytest.raises(TypeError, match="user"):
            inspect.signature(Client.create).bind(None, user="sandbox")

    def test_a_user_runtime_config_is_rejected_not_silently_dropped(
        self, tenki_session_factory: Callable[..., SandboxTenkiSession], mock_client: MagicMock
    ) -> None:
        """Test runtime_configs is forwarded as-is, so a 'user' key fails loudly.

        Without this, silently filtering the key would look identical to the tests
        above while letting a caller believe a setting took effect. The mock client is
        made to enforce the real signature, since it would otherwise accept anything.
        """
        import inspect

        def enforce_signature(**kwargs: Any) -> MagicMock:
            inspect.signature(Client.create).bind(None, **kwargs)
            return mock_client.create.return_value

        mock_client.create.side_effect = enforce_signature
        session = tenki_session_factory(runtime_configs={"user": "sandbox"})

        with pytest.raises(ContainerError, match="Failed to create Tenki sandbox"):
            session.open()


class TestSandboxTenkiExistingSandbox:
    """Test attaching to an existing Tenki sandbox."""

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
        """Test a timeout tears the sandbox down."""
        session = opened_session_factory()

        session._handle_timeout()

        mock_sandbox.terminate.assert_called_once()
        assert session.container is None


class TestTenkiContainerAPI:
    """Test the TenkiContainerAPI adapter directly."""

    def test_copy_from_container_builds_tar(self, mock_client: MagicMock, mock_sandbox: MagicMock) -> None:
        """Test copy_from_container wraps guest bytes in a Docker-shaped tar archive."""
        data, stat = TenkiContainerAPI(mock_client).copy_from_container(mock_sandbox, f"{WORKDIR}/file.txt")

        assert stat["size"] == 5
        with tarfile.open(fileobj=io.BytesIO(data), mode="r") as tar:
            assert tar.extractfile("file.txt").read() == b"hello"  # type: ignore[union-attr]

    def test_copy_from_container_directory(self, mock_client: MagicMock, mock_sandbox: MagicMock) -> None:
        """Test directories are walked and archived under the root name."""
        mock_sandbox.fs.stat.return_value = FileInfo(
            path=f"{WORKDIR}/dir", size=0, mode=0o755, is_dir=True, modified_unix_ns=MTIME_NS
        )
        mock_sandbox.fs.list.side_effect = [
            [
                FileInfo(path="a.txt", size=3, mode=0o644, is_dir=False, modified_unix_ns=MTIME_NS),
                FileInfo(path="nested", size=0, mode=0o755, is_dir=True, modified_unix_ns=MTIME_NS),
            ],
            [FileInfo(path="b.txt", size=1, mode=0o644, is_dir=False, modified_unix_ns=MTIME_NS)],
        ]
        mock_sandbox.fs.read_bytes.side_effect = [b"aaa", b"b"]

        data, stat = TenkiContainerAPI(mock_client).copy_from_container(mock_sandbox, f"{WORKDIR}/dir")

        assert stat["size"] == 4
        with tarfile.open(fileobj=io.BytesIO(data), mode="r") as tar:
            members = {m.name: m.isdir() for m in tar.getmembers()}

        # Directory entries are carried too, root first. Without them the shared extractor
        # mistakes a single-file directory for a file copy and overwrites the destination.
        assert members == {"dir": True, "dir/a.txt": False, "dir/nested": True, "dir/nested/b.txt": False}

    def test_copy_from_container_single_file_directory_is_still_a_directory(
        self, mock_client: MagicMock, mock_sandbox: MagicMock
    ) -> None:
        """Test a directory holding exactly one file archives as two members, not one.

        _determine_extract_path treats a one-member tar whose member is a file as a
        single-file copy, so without the root entry the destination directory is replaced.
        """
        mock_sandbox.fs.stat.return_value = FileInfo(
            path=f"{WORKDIR}/dir", size=0, mode=0o755, is_dir=True, modified_unix_ns=MTIME_NS
        )
        mock_sandbox.fs.list.return_value = [
            FileInfo(path="only.txt", size=4, mode=0o644, is_dir=False, modified_unix_ns=MTIME_NS)
        ]
        mock_sandbox.fs.read_bytes.return_value = b"data"

        data, _ = TenkiContainerAPI(mock_client).copy_from_container(mock_sandbox, f"{WORKDIR}/dir")

        with tarfile.open(fileobj=io.BytesIO(data), mode="r") as tar:
            members = tar.getmembers()

        assert [m.name for m in members] == ["dir", "dir/only.txt"]
        assert members[0].isdir()

    def test_copy_from_container_empty_directory_is_not_reported_as_missing(
        self, mock_client: MagicMock, mock_sandbox: MagicMock
    ) -> None:
        """Test an empty directory yields a member and a non-zero size.

        Callers read size 0 as "not found", so an empty directory has to report something:
        it was found, it simply holds no bytes.
        """
        mock_sandbox.fs.stat.return_value = FileInfo(
            path=f"{WORKDIR}/empty", size=0, mode=0o755, is_dir=True, modified_unix_ns=MTIME_NS
        )
        mock_sandbox.fs.list.return_value = []

        data, stat = TenkiContainerAPI(mock_client).copy_from_container(mock_sandbox, f"{WORKDIR}/empty")

        with tarfile.open(fileobj=io.BytesIO(data), mode="r") as tar:
            assert [m.name for m in tar.getmembers()] == ["empty"]
        assert stat["size"] > 0

    def test_copy_from_container_falls_back_to_shell_outside_workdir(
        self, mock_client: MagicMock, mock_sandbox: MagicMock
    ) -> None:
        """Test paths the FS API rejects are archived via shell (plot capture)."""
        import base64

        mock_sandbox.fs.stat.side_effect = SandboxError("path outside workdir")

        tar_buf = io.BytesIO()
        with tarfile.open(fileobj=tar_buf, mode="w") as tar:
            member = tarfile.TarInfo(name="000001.png")
            member.size = 4
            tar.addfile(member, io.BytesIO(b"PNG!"))
        encoded = base64.b64encode(tar_buf.getvalue())

        mock_sandbox.shell.side_effect = [
            make_command_result(),
            make_command_result(stdout=encoded),
        ]

        data, _ = TenkiContainerAPI(mock_client).copy_from_container(mock_sandbox, "/tmp/sandbox_plots/000001.png")

        with tarfile.open(fileobj=io.BytesIO(data), mode="r") as tar:
            assert tar.extractfile("000001.png").read() == b"PNG!"  # type: ignore[union-attr]
