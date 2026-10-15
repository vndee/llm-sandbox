"""Linux worker protocol tests with trusted commands; these do not exercise WSL isolation."""

from __future__ import annotations

import base64
import io
import os
import queue
import sys
import tempfile
import threading
import time
from pathlib import Path
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any

import pytest

from llm_sandbox.bubblewrap import BubblewrapSession
from llm_sandbox.exceptions import SandboxError
from llm_sandbox.local import wsl_worker
from llm_sandbox.local.config import LocalSandboxConfig
from llm_sandbox.local.session import LocalProcessSession
from llm_sandbox.local.wsl_protocol import MAX_CONTENT_BYTES, encode_frame, read_frame, validate_policy
from llm_sandbox.local.wsl_worker import Worker, _GuestSession
from llm_sandbox.security import SecurityPolicy
from tests.test_local_session import TrustedTestSession

if TYPE_CHECKING:
    from collections.abc import Iterator

pytestmark = pytest.mark.skipif(sys.platform != "linux", reason="Trusted fixture uses Linux process and file APIs")


class WorkerPeer:
    """Drive the actual worker with bounded pipe reads and fixed trusted test requests."""

    def __init__(self) -> None:
        """Start a worker and a bounded-response reader on independent pipes."""
        input_read, input_write = os.pipe()
        output_read, output_write = os.pipe()
        self.input = os.fdopen(input_write, "wb", buffering=0)
        self.output = os.fdopen(output_read, "rb", buffering=0)
        self.worker = Worker(os.fdopen(input_read, "rb", buffering=0), os.fdopen(output_write, "wb", buffering=0))
        self.messages: queue.Queue[dict[str, Any] | BaseException | None] = queue.Queue()
        self.failures: list[BaseException] = []
        self.serving = threading.Thread(target=self._serve, daemon=True)
        self.reading = threading.Thread(target=self._read, daemon=True)
        self.serving.start()
        self.reading.start()
        assert self.receive()["type"] == "hello"

    def _serve(self) -> None:
        try:
            self.worker.serve()
        except SandboxError as exc:
            # Cancelling a live request intentionally prevents its final response.
            if not self.worker.stopped.is_set():
                self.failures.append(exc)
        except BaseException as exc:  # noqa: BLE001 # Surface background failures in the test thread.
            self.failures.append(exc)
        finally:
            self.worker.outgoing.close()

    def _read(self) -> None:
        try:
            while (message := read_frame(self.output)) is not None:
                self.messages.put(message)
            self.messages.put(None)
        except BaseException as exc:  # noqa: BLE001 # Surface background failures in the test thread.
            self.messages.put(exc)

    def send(self, message: dict[str, Any]) -> None:
        """Send one bounded fixture record, including explicit control messages."""
        self.input.write(encode_frame(message))

    def receive(self) -> dict[str, Any]:
        """Fail within three seconds instead of hanging pytest on a missing response."""
        message = self.messages.get(timeout=3)
        if isinstance(message, BaseException):
            raise message
        assert message is not None, "Worker exited before returning the expected response"
        return message

    def expect_stdout(self, expected: str) -> None:
        """Accumulate decoded events without assuming pipe reads align with print calls."""
        received = ""
        while len(received) < len(expected):
            event = self.receive()
            assert event["type"] == "event"
            assert event["stream"] == "stdout"
            received += event["data"]
        assert received == expected

    def request(self, operation: str, params: Any, request_id: int = 1) -> list[dict[str, Any]]:
        """Collect streamed events and the matching terminal response."""
        self.send({"id": request_id, "op": operation, "params": params})
        messages = []
        while True:
            message = self.receive()
            assert message["id"] == request_id
            messages.append(message)
            if message["type"] != "event":
                return messages

    def open(self) -> Path:
        """Open a small private session using the real guest session lifecycle."""
        result = self.request(
            "open",
            {
                "sandbox_config": LocalSandboxConfig(max_artifact_bytes=16).model_dump(mode="json"),
                "skip_environment_setup": True,
                "security_policy": SecurityPolicy().model_dump(mode="json"),
            },
        )[-1]
        assert result["type"] == "result", result
        assert result["value"]["pid"] == os.getpid()
        return Path(result["value"]["workspace"])

    def close(self) -> None:
        """Deliver EOF and assert that all fixture-owned resources terminate."""
        self.input.close()
        self.serving.join(3)
        if self.serving.is_alive():
            cleanup = threading.Thread(target=self.worker.stop, daemon=True)
            cleanup.start()
            cleanup.join(3)
            self.serving.join(3)
        self.reading.join(3)
        self.output.close()
        self.worker.incoming.close()
        assert not self.serving.is_alive()
        assert not self.reading.is_alive()
        assert not self.failures


@pytest.fixture
def trusted_worker(monkeypatch: pytest.MonkeyPatch) -> Iterator[WorkerPeer]:
    """Replace only OS preflight/launcher; retain worker framing, files, limits and cleanup."""
    monkeypatch.setattr(wsl_worker, "_verify_wsl2", lambda: None)
    monkeypatch.setattr(wsl_worker, "_validate_mounts", lambda *_: None)
    monkeypatch.setattr(BubblewrapSession, "__init__", LocalProcessSession.__init__)
    monkeypatch.setattr(_GuestSession, "_launch_command", TrustedTestSession._launch_command)
    monkeypatch.setattr(tempfile, "tempdir", tempfile.tempdir)
    monkeypatch.setenv("WSL_INTEROP", "must-be-removed")
    monkeypatch.setenv("WSLENV", "must-be-removed")
    peer = WorkerPeer()
    try:
        yield peer
    finally:
        peer.close()


def test_worker_roundtrip_keeps_payload_json_out_of_control_frames(trusted_worker: WorkerPeer) -> None:
    """Real dispatch preserves request IDs, stdout/stderr, workspace state and binary transfers."""
    peer = trusted_worker
    workspace = peer.open()
    assert "WSL_INTEROP" not in os.environ
    assert "WSLENV" not in os.environ
    code = 'import sys;print(\'{"type":"result","id":999}\');sys.stderr.write(\'diagnostic\')'
    messages = peer.request("run", {"code": code}, request_id=41)
    result = messages[-1]
    assert result["type"] == "result"
    assert result["value"] == {"exit_code": 0, "stdout": '{"type":"result","id":999}\n', "stderr": "diagnostic"}
    for stream in ("stdout", "stderr"):
        assert "".join(item["data"] for item in messages[:-1] if item["stream"] == stream) == result["value"][stream]
    assert peer.request("execute", {"command": "printf saved > state"})[-1]["value"]["exit_code"] == 0
    assert (workspace / "state").read_bytes() == b"saved"
    assert peer.request("mkdir", {"path": "nested"})[-1]["type"] == "result"
    binary = b"\x00\xff\nfixture"
    encoded = base64.b64encode(binary).decode("ascii")
    assert peer.request("put", {"path": "nested/data", "data": encoded})[-1]["type"] == "result"
    assert peer.request("get", {"path": "nested/data"})[-1]["value"] == encoded
    assert peer.request("install", {"libraries": []})[-1] == {"type": "result", "id": 1, "value": None}
    peer.close()
    assert not workspace.parent.exists()


@pytest.mark.parametrize(
    ("operation", "params", "error"),
    [
        ("run", [], "SandboxError"),
        ("__getattribute__", {"name": "session"}, "SandboxError"),
        ("put", {"path": "data", "data": "not base64!"}, "Error"),
        ("put", {"path": "data", "data": base64.b64encode(b"x" * 17).decode("ascii")}, "SandboxError"),
        ("put", {"path": "../outside", "data": "YQ=="}, "SecurityError"),
        ("get", {"path": "/etc/passwd"}, "SecurityError"),
    ],
)
def test_bad_requests_return_matching_errors_without_writing_files(
    trusted_worker: WorkerPeer, operation: str, params: Any, error: str
) -> None:
    """Malformed operations, payloads and paths cannot trigger arbitrary dispatch or uploads."""
    workspace = trusted_worker.open()
    before = set(workspace.rglob("*"))
    result = trusted_worker.request(operation, params, request_id=73)[-1]
    assert result["type"] == "error"
    assert result["error"] == error
    assert result["message"]
    assert set(workspace.rglob("*")) == before


def test_download_refuses_symlinks_and_oversized_files(trusted_worker: WorkerPeer, tmp_path: Path) -> None:
    """Worker exports keep the nofollow and artifact-size boundary enforced by local file APIs."""
    workspace = trusted_worker.open()
    outside = tmp_path / "secret"
    outside.write_bytes(b"private")
    (workspace / "link").symlink_to(outside)
    (workspace / "large").write_bytes(b"x" * 17)
    for source, error in (("link", "SecurityError"), ("large", "SandboxError")):
        response = trusted_worker.request("get", {"path": source})[-1]
        assert response["type"] == "error"
        assert response["error"] == error
        assert "private" not in str(response)


@pytest.mark.parametrize("termination", ["eof", "malformed", "cancel"])
def test_transport_failure_cleans_an_active_command(trusted_worker: WorkerPeer, termination: str) -> None:
    """Control-channel failure reaps the actual trusted process and removes its private workspace."""
    peer = trusted_worker
    workspace = peer.open()
    peer.send({"id": 2, "op": "run", "params": {"code": "import time;print('ready',flush=True);time.sleep(30)"}})
    peer.expect_stdout("ready\n")
    if termination == "eof":
        peer.input.close()
    elif termination == "malformed":
        peer.input.write(b"not a JSON control message\n")
    else:
        peer.send({"op": "cancel"})
    peer.serving.join(3)
    assert not peer.serving.is_alive()
    assert peer.worker.stopped.is_set()
    assert peer.worker.session is not None
    assert peer.worker.session._active_process is None
    assert not workspace.parent.exists()


def test_heartbeats_preserve_active_work_then_missing_lease_cleans_it(
    trusted_worker: WorkerPeer, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Lease renewal runs independently of a live command; removing renewal cancels and cleans it."""
    peer = trusted_worker
    workspace = peer.open()
    peer.send({"id": 2, "op": "run", "params": {"code": "import time;print('ready',flush=True);time.sleep(30)"}})
    peer.expect_stdout("ready\n")
    monkeypatch.setattr(wsl_worker, "LEASE_SECONDS", 0.2)
    for _ in range(12):
        peer.send({"op": "heartbeat"})
        time.sleep(0.05)
    assert not peer.worker.stopped.is_set()
    assert workspace.exists()
    peer.serving.join(3)
    assert not peer.serving.is_alive()
    assert not workspace.parent.exists()


@pytest.mark.parametrize("during_open", [False, True])
@pytest.mark.usefixtures("trusted_worker")
def test_startup_cancellation_never_leaves_a_private_workspace(
    monkeypatch: pytest.MonkeyPatch, during_open: bool
) -> None:
    """Cancellation both before publication and after session creation leaves no executable lifecycle."""
    worker = Worker(io.BytesIO(), io.BytesIO())
    created: list[_GuestSession] = []
    original_open = _GuestSession.open

    def create(stopped: threading.Event, **kwargs: Any) -> _GuestSession:
        session = _GuestSession(stopped, **kwargs)
        created.append(session)
        if not during_open:
            worker.stop()
        return session

    def open_then_cancel(session: _GuestSession) -> None:
        original_open(session)
        worker.stop()

    monkeypatch.setattr(wsl_worker, "_GuestSession", create)
    if during_open:
        monkeypatch.setattr(_GuestSession, "open", open_then_cancel)
    request = {"op": "open", "params": {"sandbox_config": {}, "skip_environment_setup": True}}
    with pytest.raises(SandboxError, match="closed during startup"):
        worker.dispatch(request)
    assert worker.stopped.is_set()
    assert not created[0].is_open
    assert created[0]._workspace_temp is None


def test_commands_require_an_open_session_and_cannot_replace_it(trusted_worker: WorkerPeer) -> None:
    """Pre-open commands and a second open are protocol errors, preserving the existing lifecycle."""
    error = trusted_worker.request("run", {"code": "print('must not run')"})[-1]
    assert error["type"] == "error"
    assert "not open" in error["message"]
    workspace = trusted_worker.open()
    duplicate = trusted_worker.request("open", {"sandbox_config": {}})[-1]
    assert duplicate["type"] == "error"
    assert "second session" in duplicate["message"]
    assert trusted_worker.worker.session is not None
    assert trusted_worker.worker.session.workspace == workspace


def test_worker_output_ends_cleanly_when_the_peer_rejects_writes() -> None:
    """Zero-byte writes must fail immediately instead of retrying a closed transport forever."""

    class ClosedOutput(io.BytesIO):
        def write(self, _data: Any) -> int:
            return 0

    worker = Worker(io.BytesIO(), ClosedOutput())
    with pytest.raises(SandboxError, match="output pipe closed"):
        worker.send({"type": "result", "id": 1})


def test_output_frames_flush_and_late_callbacks_are_rejected() -> None:
    """A successful frame remains intact, and cancellation prevents any subsequent payload event."""
    outgoing = io.BytesIO()
    worker = Worker(io.BytesIO(), outgoing)
    message = {"type": "event", "data": "first\nsecond"}
    worker.send(message)
    assert outgoing.getvalue() == encode_frame(message)
    emit = worker._callback(3, "stdout")
    worker.stop()
    with pytest.raises(SandboxError, match="closed during execution"):
        emit("late payload")
    assert outgoing.getvalue() == encode_frame(message)


@pytest.mark.parametrize("setting", ["max_artifact_bytes", "max_output_bytes"])
def test_worker_policy_rejects_limits_larger_than_its_protocol(setting: str) -> None:
    """A valid local policy cannot promise output or artifacts that the remote protocol cannot carry."""
    config = LocalSandboxConfig(**{setting: MAX_CONTENT_BYTES + 1})
    with pytest.raises(ValueError, match="cannot exceed"):
        validate_policy(config, {})


def test_cli_startup_failure_uses_stderr_without_corrupting_control_stdout(monkeypatch: pytest.MonkeyPatch) -> None:
    """Unsupported guest hosts exit with one diagnostic and no protocol-shaped success response."""
    output = io.BytesIO()
    diagnostic = io.StringIO()
    monkeypatch.setattr(wsl_worker.platform, "release", lambda: "6.0-generic")
    monkeypatch.setattr(
        wsl_worker,
        "sys",
        SimpleNamespace(
            platform="linux",
            stdin=SimpleNamespace(buffer=io.BytesIO()),
            stdout=SimpleNamespace(buffer=output),
            stderr=diagnostic,
        ),
    )
    with pytest.raises(SystemExit) as failure:
        wsl_worker.main()
    assert failure.value.code == 1
    assert "requires the Microsoft WSL2" in diagnostic.getvalue()
    assert output.getvalue() == b""
