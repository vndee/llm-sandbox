"""Tests for Tenki backend implementation."""

import io
import tarfile
from collections.abc import Callable
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, Mock, patch

import pytest
from tenki_sandbox import CommandResult, FileInfo, MissingAuthTokenError, SandboxError

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
        """Test open creates a sandbox with wait=False and waits until it is ready."""
        session = tenki_session_factory()

        session.open()

        mock_client.create.assert_called_once()
        assert mock_client.create.call_args.kwargs["wait"] is False
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

    def test_open_attaches_to_existing_sandbox(
        self, tenki_session_factory: Callable[..., SandboxTenkiSession], mock_client: MagicMock
    ) -> None:
        """Test open attaches to an existing sandbox instead of creating one."""
        session = tenki_session_factory(container_id="sbx-999")

        session.open()

        mock_client.get.assert_called_once_with("sbx-999")
        mock_client.create.assert_not_called()


class TestSandboxTenkiSessionPythonBootstrap:
    """Test the guest Python interpreter alias and venv verification are gated apart.

    Managed Tenki images ship ``python3`` without a bare ``python``. The alias is needed
    whenever the session invokes ``python`` — to build the venv, or directly when
    skip_environment_setup drops the venv paths — but verifying the venv only makes
    sense when environment_setup actually built one.
    """

    def _bootstrap_calls(self, mock_sandbox: MagicMock) -> list[str]:
        return [c.args[0] for c in mock_sandbox.shell.call_args_list if "command -v python3" in str(c.args[0])]

    def test_interpreter_is_aliased_when_setting_up_the_environment(
        self, tenki_session_factory: Callable[..., SandboxTenkiSession], mock_sandbox: MagicMock
    ) -> None:
        """Test the python3 alias runs so `python -m venv` can find an interpreter."""
        session = tenki_session_factory(lang="python")

        session.open()

        assert self._bootstrap_calls(mock_sandbox)

    def test_interpreter_is_aliased_when_skipping_environment_setup(
        self, tenki_session_factory: Callable[..., SandboxTenkiSession], mock_sandbox: MagicMock
    ) -> None:
        """Test the alias still runs when there is no venv, since code runs on bare `python`."""
        session = tenki_session_factory(lang="python", skip_environment_setup=True)

        session.open()

        assert self._bootstrap_calls(mock_sandbox), "skip_environment_setup still executes a bare `python`"

    def test_interpreter_is_not_touched_on_an_attached_sandbox(
        self, tenki_session_factory: Callable[..., SandboxTenkiSession], mock_sandbox: MagicMock
    ) -> None:
        """Test attaching to someone else's sandbox modifies nothing in the guest."""
        session = tenki_session_factory(lang="python", container_id="sbx-999")

        session.open()

        assert not self._bootstrap_calls(mock_sandbox)

    def test_venv_is_not_verified_when_environment_setup_is_skipped(
        self, tenki_session_factory: Callable[..., SandboxTenkiSession]
    ) -> None:
        """Test no venv check runs for a session that never builds one."""
        session = tenki_session_factory(lang="python", skip_environment_setup=True)

        with patch.object(session, "_verify_python_environment") as mock_verify:
            session.open()

        mock_verify.assert_not_called()

    def test_venv_is_verified_when_environment_setup_runs(
        self, tenki_session_factory: Callable[..., SandboxTenkiSession]
    ) -> None:
        """Test a built venv is still verified, so a failed build fails loudly."""
        session = tenki_session_factory(lang="python")

        with patch.object(session, "_verify_python_environment") as mock_verify:
            session.open()

        mock_verify.assert_called_once()


class TestSandboxTenkiSessionCloseIsRetryable:
    """Test that a failed teardown is reported and stays retryable.

    A microVM that failed to terminate is still running and still billed. Returning
    normally claims the opposite, and dropping the handle leaves nobody able to try
    again, so the sandbox leaks silently.
    """

    def test_failed_termination_raises_and_keeps_the_handle(
        self, opened_session_factory: Callable[..., SandboxTenkiSession], mock_sandbox: MagicMock
    ) -> None:
        """Test close() surfaces a termination failure and retains the handle for retry."""
        session = opened_session_factory()
        mock_sandbox.terminate.side_effect = SandboxError("already gone")

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
        """Test a second close() after a transient failure actually frees the microVM."""
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
        """Test the attached-sandbox path reports failures on the same terms."""
        session = opened_session_factory(container_id="sbx-999")
        mock_sandbox.detach.side_effect = SandboxError("connection reset")

        with pytest.raises(ContainerError, match="Failed to release Tenki sandbox"):
            session.close()

        assert session.container is mock_sandbox

    def test_exit_does_not_let_cleanup_failure_mask_the_block_error(
        self, tenki_session_factory: Callable[..., SandboxTenkiSession], mock_sandbox: MagicMock
    ) -> None:
        """Test the caller's exception wins over a teardown failure on the way out."""
        mock_sandbox.terminate.side_effect = SandboxError("already gone")
        session = tenki_session_factory()

        body_error = ValueError("user code failed")

        with pytest.raises(ValueError, match="user code failed"), session:
            raise body_error

        assert session.container is mock_sandbox

    def test_exit_raises_cleanup_failure_when_the_block_succeeded(
        self, tenki_session_factory: Callable[..., SandboxTenkiSession], mock_sandbox: MagicMock
    ) -> None:
        """Test a clean block still surfaces a leaked sandbox instead of hiding it."""
        mock_sandbox.terminate.side_effect = SandboxError("already gone")
        session = tenki_session_factory()

        with pytest.raises(ContainerError, match="Failed to release Tenki sandbox"), session:
            pass


class TestSandboxTenkiSessionOpenIsFailureAtomic:
    """Test that a failed open() never leaves a microVM running.

    __exit__ does not run when __enter__ raises, so open() is the only place that can
    release a sandbox it already allocated. Every failure mode below is a billed leak
    if it is not cleaned up.
    """

    def test_open_terminates_sandbox_when_readiness_fails(
        self, tenki_session_factory: Callable[..., SandboxTenkiSession], mock_sandbox: MagicMock
    ) -> None:
        """Test a sandbox that never becomes ready is terminated, not abandoned."""
        mock_sandbox.wait_ready.side_effect = SandboxError("boot timed out")
        session = tenki_session_factory()

        with pytest.raises(ContainerError, match="Failed to create Tenki sandbox"):
            session.open()

        mock_sandbox.terminate.assert_called_once()
        assert session.container is None

    def test_open_terminates_sandbox_when_preparation_fails(
        self, tenki_session_factory: Callable[..., SandboxTenkiSession], mock_sandbox: MagicMock
    ) -> None:
        """Test a failure after a successful create still releases the microVM."""
        session = tenki_session_factory()

        with (
            patch.object(session, "environment_setup", side_effect=SandboxError("setup blew up")),
            pytest.raises(SandboxError, match="setup blew up"),
        ):
            session.open()

        mock_sandbox.terminate.assert_called_once()
        assert session.container is None

    def test_open_terminates_sandbox_on_cancellation(
        self, tenki_session_factory: Callable[..., SandboxTenkiSession], mock_sandbox: MagicMock
    ) -> None:
        """Test Ctrl-C mid-provision terminates the sandbox and propagates unchanged."""
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

    def test_open_leaves_original_error_intact_when_cleanup_also_fails(
        self, tenki_session_factory: Callable[..., SandboxTenkiSession], mock_sandbox: MagicMock
    ) -> None:
        """Test a failing teardown does not mask why open() failed."""
        mock_sandbox.wait_ready.side_effect = SandboxError("boot timed out")
        mock_sandbox.terminate.side_effect = SandboxError("already gone")
        session = tenki_session_factory()

        with pytest.raises(ContainerError, match="Failed to create Tenki sandbox"):
            session.open()

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

    def test_execute_command_unknown_exit_code_is_not_reported_as_success(
        self, opened_session_factory: Callable[..., SandboxTenkiSession], mock_sandbox: MagicMock
    ) -> None:
        """Test an absent exit code fails closed rather than passing as 0.

        The SDK types exit_code as int, so None means the outcome is unknown, and
        `CommandResult.ok` treats it as a failure. Normalizing it to 0 would hand
        callers a success they never got.
        """
        session = opened_session_factory()
        mock_sandbox.shell.return_value = make_command_result(exit_code=None)

        assert session.execute_command("true").exit_code != 0


class TestSandboxTenkiSessionSignalledCommands:
    """Test that a command killed by a signal is not mistaken for a successful one.

    The SDK's CommandResult carries signal/reason/errno alongside exit_code, and its
    own `ok` property is `exit_code == 0 and not signal`. Reading exit_code alone lets
    an OOM kill — which can arrive as exit_code 0 — pass as a clean run.
    """

    def test_sigkill_with_zero_exit_code_is_a_failure(
        self, opened_session_factory: Callable[..., SandboxTenkiSession], mock_sandbox: MagicMock
    ) -> None:
        """Test the canonical case: SIGKILL, exit code 0, no stderr."""
        session = opened_session_factory()
        mock_sandbox.shell.return_value = make_command_result(exit_code=0, signal="SIGKILL")

        assert session.execute_command("stress").exit_code != 0

    def test_signal_reason_and_errno_reach_stderr(
        self, opened_session_factory: Callable[..., SandboxTenkiSession], mock_sandbox: MagicMock
    ) -> None:
        """Test the SDK's failure detail is surfaced, not dropped."""
        session = opened_session_factory()
        mock_sandbox.shell.return_value = make_command_result(exit_code=0, signal="SIGKILL", reason="oom", errno=137)

        stderr = session.execute_command("stress").stderr

        assert "SIGKILL" in stderr
        assert "oom" in stderr
        assert "137" in stderr

    def test_detail_is_appended_without_losing_real_stderr(
        self, opened_session_factory: Callable[..., SandboxTenkiSession], mock_sandbox: MagicMock
    ) -> None:
        """Test the guest's own error output survives alongside the added detail."""
        session = opened_session_factory()
        mock_sandbox.shell.return_value = make_command_result(exit_code=0, stderr=b"Traceback: boom", signal="SIGKILL")

        stderr = session.execute_command("stress").stderr

        assert "Traceback: boom" in stderr
        assert "SIGKILL" in stderr


class TestSandboxTenkiSessionOutputProcessing:
    """Test SandboxTenkiSession output decoding and streaming callbacks."""

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
        mock_sandbox.shell.assert_any_call(f"mkdir -p {WORKDIR}/input")
        mock_sandbox.shell.assert_any_call(f"mkdir -p {WORKDIR}/input/nested")

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
        """Test a timeout tears the sandbox down so no hung work keeps billing."""
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
        assert stat["mode"] == 0o644
        assert stat["mtime"] == MTIME_NS // 1_000_000_000

        with tarfile.open(fileobj=io.BytesIO(data), mode="r") as tar:
            members = tar.getmembers()
            assert [m.name for m in members] == ["file.txt"]
            extracted = tar.extractfile(members[0])
            assert extracted is not None
            assert extracted.read() == b"hello"

    def test_copy_from_container_directory(self, mock_client: MagicMock, mock_sandbox: MagicMock) -> None:
        """Test directories are walked and archived with paths relative to the root name."""
        mock_sandbox.fs.stat.return_value = FileInfo(
            path=f"{WORKDIR}/dir", size=0, mode=0o755, is_dir=True, modified_unix_ns=MTIME_NS
        )
        mock_sandbox.fs.list.side_effect = [
            [
                FileInfo(path="a.txt", size=3, mode=0o644, is_dir=False, modified_unix_ns=MTIME_NS),
                FileInfo(path="nested", size=0, mode=0o755, is_dir=True, modified_unix_ns=MTIME_NS),
            ],
            [
                FileInfo(path="b.txt", size=1, mode=0o644, is_dir=False, modified_unix_ns=MTIME_NS),
            ],
        ]
        mock_sandbox.fs.read_bytes.side_effect = [b"aaa", b"b"]

        data, stat = TenkiContainerAPI(mock_client).copy_from_container(mock_sandbox, f"{WORKDIR}/dir")

        assert stat["size"] == 4
        with tarfile.open(fileobj=io.BytesIO(data), mode="r") as tar:
            assert sorted(m.name for m in tar.getmembers()) == ["dir/a.txt", "dir/nested/b.txt"]
            assert tar.extractfile("dir/a.txt").read() == b"aaa"  # type: ignore[union-attr]
            assert tar.extractfile("dir/nested/b.txt").read() == b"b"  # type: ignore[union-attr]
