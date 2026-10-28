"""Real trusted pipes verify host deadlines without needing a WSL installation."""

from __future__ import annotations

import io
import os
import queue
import sys
import threading
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any, BinaryIO
from unittest.mock import Mock

import pytest

from llm_sandbox.exceptions import NotOpenSessionError, SandboxError, SandboxTimeoutError
from llm_sandbox.wsl2_bubblewrap import _DIAGNOSTIC_LIMIT, _OutboundFrame, _Transport


@pytest.fixture
def trusted_transport(monkeypatch: pytest.MonkeyPatch) -> Any:
    """Substitute a trusted Python peer, preserving the real process and pipe lifecycle."""
    monkeypatch.setattr("llm_sandbox.wsl2_bubblewrap._launcher_environment", lambda _: dict(os.environ))
    monkeypatch.setattr("llm_sandbox.wsl2_bubblewrap.HEARTBEAT_INTERVAL", 60)
    monkeypatch.setattr("llm_sandbox.wsl2_bubblewrap.LEASE_SECONDS", 0.01)
    transports: list[_Transport] = []

    def create(code: str) -> _Transport:
        transport = _Transport([str(Path(sys.executable).resolve()), "-I", "-u", "-c", code])
        transports.append(transport)
        return transport

    yield create
    for transport in transports:
        transport.close()
        assert transport.process.poll() is not None
        assert not any(thread.is_alive() for thread in transport._threads)


@pytest.fixture
def dormant_transport(monkeypatch: pytest.MonkeyPatch) -> Any:
    """Keep transport threads paused while injecting deterministic pipe boundary faults."""
    process = SimpleNamespace(stdin=io.BytesIO(), stdout=io.BytesIO(), stderr=io.BytesIO(), wait=Mock())
    managed = SimpleNamespace(process=process, close=Mock())
    monkeypatch.setattr("llm_sandbox.windows.process.ManagedProcess", Mock(return_value=managed))
    monkeypatch.setattr(
        "llm_sandbox.wsl2_bubblewrap.threading.Thread",
        lambda **_kwargs: SimpleNamespace(start=Mock(), ident=None, join=Mock()),
    )
    transport = _Transport(["C:/Windows/System32/wsl.exe"])
    yield transport
    transport.close()


def test_request_deadline_expires_while_peer_never_reads_input(trusted_transport: Any) -> None:
    """A full host pipe cannot trap the caller before its receive deadline is enforced."""
    transport = trusted_transport("import time; time.sleep(30)")
    started = time.monotonic()
    with pytest.raises(SandboxTimeoutError, match="sending a request"):
        transport.send({"op": "put", "data": "x" * (1024 * 1024)}, deadline=started + 0.15)
    assert time.monotonic() - started < 1
    assert transport._writing.is_set()


def test_close_unblocks_both_active_and_queued_request_senders(trusted_transport: Any) -> None:
    """Cancellation covers waiting for queue capacity and waiting for a blocked pipe writer."""
    transport = trusted_transport("import time; time.sleep(30)")
    errors: list[BaseException] = []

    def send() -> None:
        try:
            transport.send({"op": "put", "data": "x" * (1024 * 1024)})
        except BaseException as exc:  # noqa: BLE001 # Assert every sender's cross-thread outcome below.
            errors.append(exc)

    senders = [threading.Thread(target=send, daemon=True) for _ in range(3)]
    for sender in senders:
        sender.start()
    assert transport._writing.wait(2)
    deadline = time.monotonic() + 2
    while transport._outbound.qsize() != 1 and time.monotonic() < deadline:
        time.sleep(0.01)
    assert transport._outbound.qsize() == 1
    transport.close()
    for sender in senders:
        sender.join(1)
    assert not any(sender.is_alive() for sender in senders)
    assert len(errors) == len(senders)
    assert all(isinstance(error, NotOpenSessionError) for error in errors)


def test_partial_writes_cannot_interleave_request_and_heartbeat_frames(trusted_transport: Any) -> None:
    """The trusted peer receives complete JSON records even when every write is partial."""
    transport = trusted_transport(
        "import json,sys\nfor line in sys.stdin:\n print(json.dumps(json.loads(line)), flush=True)"
    )
    pipe = transport.process.stdin
    assert pipe is not None

    class PartialWriter:
        def __init__(self, underlying: BinaryIO) -> None:
            self.underlying = underlying

        def write(self, data: bytes) -> int | None:
            return self.underlying.write(data[:3])

        def flush(self) -> None:
            self.underlying.flush()

        def close(self) -> None:
            self.underlying.close()

    transport.process.stdin = PartialWriter(pipe)  # type: ignore[assignment]
    records = [{"op": "request", "data": "value" * 1000}, {"op": "heartbeat"}]
    errors: list[BaseException] = []
    deadline = time.monotonic() + 5

    def send(record: dict[str, str]) -> None:
        try:
            transport.send(record, deadline=deadline)
        except BaseException as exc:  # noqa: BLE001 # Surface sender failures in the test thread.
            errors.append(exc)

    senders = [threading.Thread(target=send, args=(record,), daemon=True) for record in records]
    for sender in senders:
        sender.start()
    received = [transport.receive(deadline) for _ in records]
    for sender in senders:
        sender.join(1)
    assert not errors
    assert not any(sender.is_alive() for sender in senders)
    assert sorted(received, key=lambda item: item["op"]) == sorted(records, key=lambda item: item["op"])


def test_heartbeat_reaches_idle_peer_as_a_framed_request(
    trusted_transport: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An idle session renews the worker lease without an application request."""
    monkeypatch.setattr("llm_sandbox.wsl2_bubblewrap.HEARTBEAT_INTERVAL", 0.01)
    monkeypatch.setattr("llm_sandbox.wsl2_bubblewrap.LEASE_SECONDS", 1)
    transport = trusted_transport(
        "import json,sys\nfor line in sys.stdin:\n print(json.dumps(json.loads(line)), flush=True)"
    )
    assert transport.receive(time.monotonic() + 3) == {"op": "heartbeat"}


def test_peer_diagnostics_are_bounded_without_stopping_stderr_drain(trusted_transport: Any) -> None:
    """A verbose worker can exit normally while only its diagnostic prefix is retained."""
    transport = trusted_transport(
        f"import sys\nsys.stderr.buffer.write(b'x'*{_DIAGNOSTIC_LIMIT * 32}+b'PRIVATE-TAIL')\n"
        'sys.stderr.flush()\nprint(\'{"status":"finished"}\',flush=True)'
    )
    assert transport.receive(time.monotonic() + 5) == {"status": "finished"}
    transport._threads[1].join(2)
    assert not transport._threads[1].is_alive()
    assert bytes(transport.diagnostics) == b"x" * _DIAGNOSTIC_LIMIT
    deadline = time.monotonic() + 2
    with pytest.raises(SandboxError, match="Install the matching llm-sandbox package") as raised:
        transport.receive(deadline)
    assert "PRIVATE-TAIL" not in str(raised.value)
    assert str(raised.value).endswith("x" * _DIAGNOSTIC_LIMIT)


@pytest.mark.parametrize("output", [b"worker banner\n", b'{"status":"truncated"}'])
def test_unframed_peer_output_fails_the_waiting_receiver(trusted_transport: Any, output: bytes) -> None:
    """Corrupt control output reaches the caller as a protocol error, never a result."""
    transport = trusted_transport(f"import sys\nsys.stdout.buffer.write({output!r})\nsys.stdout.flush()")
    deadline = time.monotonic() + 3
    with pytest.raises(SandboxError, match="control message"):
        transport.receive(deadline)


def test_receive_deadline_expires_for_a_silent_peer(trusted_transport: Any) -> None:
    """A healthy host pipe does not disable the caller's response deadline."""
    transport = trusted_transport("import sys\nsys.stdin.read()")
    started = time.monotonic()
    with pytest.raises(SandboxTimeoutError, match="operation exceeded its deadline"):
        transport.receive(started + 0.15)
    assert time.monotonic() - started < 1


def test_close_cancels_receiver_without_an_execution_deadline(trusted_transport: Any) -> None:
    """An indefinite response wait still observes explicit session cancellation."""
    transport = trusted_transport("import sys\nsys.stdin.read()")
    received: list[BaseException] = []
    started = threading.Event()

    def receive() -> None:
        started.set()
        try:
            transport.receive(None)
        except BaseException as exc:  # noqa: BLE001 # Assert the exact cross-thread failure below.
            received.append(exc)

    receiver = threading.Thread(target=receive, daemon=True)
    receiver.start()
    assert started.wait(2)
    transport.close()
    receiver.join(1)
    assert not receiver.is_alive()
    assert len(received) == 1
    assert isinstance(received[0], NotOpenSessionError)


def test_close_drains_backpressured_output_so_guest_cleanup_can_finish(trusted_transport: Any) -> None:
    """Cancellation releases a full response queue before the guest handles stdin EOF."""
    transport = trusted_transport(
        "import json,sys\n"
        "for _ in range(128):\n print(json.dumps({'event':'x'*4096}),flush=True)\n"
        "sys.stdin.read()\nsys.stderr.write('guest-cleanup-complete')\nsys.stderr.flush()"
    )
    deadline = time.monotonic() + 3
    while not transport.messages.full() and time.monotonic() < deadline:
        time.sleep(0.01)
    assert transport.messages.full()
    transport.close()
    assert transport.process.returncode == 0
    assert transport.diagnostics == b"guest-cleanup-complete"
    assert not any(thread.is_alive() for thread in transport._threads)


@pytest.mark.parametrize("error", [OSError("pipe failed"), ValueError("pipe closed")])
def test_read_pipe_failures_reach_the_receiver(dormant_transport: Any, error: Exception) -> None:
    """Read errors become control-channel failures with their original cause preserved."""
    dormant_transport.process.stdout = SimpleNamespace(readline=Mock(side_effect=error))
    dormant_transport._read()
    with pytest.raises(type(error)) as raised:
        dormant_transport.receive(None)
    assert raised.value is error


@pytest.mark.parametrize("error", [OSError("pipe failed"), ValueError("pipe closed")])
def test_stderr_read_failure_does_not_destroy_retained_diagnostics(dormant_transport: Any, error: Exception) -> None:
    """A closing diagnostic pipe must not erase already captured worker errors."""
    dormant_transport.process.stderr = SimpleNamespace(read=Mock(side_effect=[b"startup failed", error]))
    dormant_transport._drain_stderr()
    assert dormant_transport.diagnostics == b"startup failed"


def test_missing_pipes_fail_closed_without_breaking_cleanup(dormant_transport: Any) -> None:
    """A partially initialized launcher remains closable and cannot accept requests."""
    dormant_transport.process.stdin = None
    dormant_transport.process.stdout = None
    dormant_transport.process.stderr = None
    dormant_transport._read()
    dormant_transport._drain_stderr()
    assert dormant_transport.messages.empty()
    frame = _OutboundFrame(b"{}\n")
    with pytest.raises(NotOpenSessionError):
        dormant_transport._write_frame(frame)
    dormant_transport.close()
    dormant_transport.managed.close.assert_called_once()


def test_full_response_queue_retries_until_space_is_available(dormant_transport: Any) -> None:
    """Transient response backpressure preserves the complete original message."""
    message = {"event": "stdout", "data": "retained"}
    dormant_transport.messages = SimpleNamespace(put=Mock(side_effect=[queue.Full, None]))
    dormant_transport._enqueue(message)
    assert dormant_transport.messages.put.call_count == 2
    for call in dormant_transport.messages.put.call_args_list:
        assert call.args == (message,)
    dormant_transport.stopped.set()
    dormant_transport._enqueue({"event": "discarded after close"})
    assert dormant_transport.messages.put.call_count == 2


@pytest.mark.parametrize("error", [OSError("disconnected"), ValueError("closed"), SandboxError("timed out")])
def test_heartbeat_failure_wakes_the_response_waiter(
    dormant_transport: Any, monkeypatch: pytest.MonkeyPatch, error: Exception
) -> None:
    """Losing lease renewals must surface as a session failure even without guest output."""
    monkeypatch.setattr("llm_sandbox.wsl2_bubblewrap.HEARTBEAT_INTERVAL", 0)
    dormant_transport.send = Mock(side_effect=error)
    dormant_transport._heartbeat()
    dormant_transport.send.assert_called_once()
    assert dormant_transport.send.call_args.args == ({"op": "heartbeat"},)
    assert dormant_transport.send.call_args.kwargs["deadline"] > time.monotonic()
    with pytest.raises(SandboxError, match="heartbeat transport failed"):
        dormant_transport.receive(None)


@pytest.mark.parametrize("written", [None, 0])
def test_zero_progress_write_fails_current_and_subsequent_requests(dormant_transport: Any, written: int | None) -> None:
    """A closed input pipe cannot spin forever or silently acknowledge a partial request."""
    frame = _OutboundFrame(b'{"op":"run"}\n')
    dormant_transport.process.stdin = SimpleNamespace(write=Mock(return_value=written), flush=Mock(), close=Mock())
    dormant_transport._outbound.put(frame)
    dormant_transport._write()
    assert frame.completed.is_set()
    assert not dormant_transport._writing.is_set()
    assert frame.error is dormant_transport._write_error
    assert isinstance(frame.error, SandboxError)
    dormant_transport.process.stdin.flush.assert_not_called()
    with pytest.raises(SandboxError, match="worker input closed") as raised:
        dormant_transport.send({"op": "next"})
    assert raised.value is frame.error


def test_cancel_between_partial_writes_stops_before_next_write(dormant_transport: Any) -> None:
    """Cancellation is rechecked inside a frame, including after a successful short write."""

    def cancel_after_write(_data: bytes) -> int:
        dormant_transport.stopped.set()
        return 1

    dormant_transport.process.stdin = SimpleNamespace(
        write=Mock(side_effect=cancel_after_write), flush=Mock(), close=Mock()
    )
    frame = _OutboundFrame(b"{}\n")
    with pytest.raises(NotOpenSessionError):
        dormant_transport._write_frame(frame)
    dormant_transport.process.stdin.write.assert_called_once()
    dormant_transport.process.stdin.flush.assert_not_called()


def test_writer_recovers_from_an_empty_queue_then_publishes_io_failure(dormant_transport: Any) -> None:
    """An idle writer remains available and signals completion even when the next write fails."""
    frame = _OutboundFrame(b"{}\n")
    failure = OSError("pipe disconnected")
    dormant_transport._outbound = SimpleNamespace(get=Mock(side_effect=[queue.Empty, frame]))
    dormant_transport.process.stdin = SimpleNamespace(write=Mock(side_effect=failure), flush=Mock(), close=Mock())
    dormant_transport._write()
    assert dormant_transport._outbound.get.call_count == 2
    assert frame.completed.is_set()
    assert frame.error is failure
    assert dormant_transport._write_error is failure
    assert not dormant_transport._writing.is_set()


def test_constructor_thread_start_failure_releases_the_managed_process(monkeypatch: pytest.MonkeyPatch) -> None:
    """A partially started transport never leaks the trusted launcher or joins an unstarted thread."""
    process = SimpleNamespace(stdin=io.BytesIO(), wait=Mock())
    managed = SimpleNamespace(process=process, close=Mock())
    monkeypatch.setattr("llm_sandbox.windows.process.ManagedProcess", Mock(return_value=managed))
    started = SimpleNamespace(start=Mock(), ident=123, join=Mock())
    failure = RuntimeError("thread allocation failed")
    unstarted = SimpleNamespace(start=Mock(side_effect=failure), ident=None, join=Mock())
    later = SimpleNamespace(start=Mock(), ident=None, join=Mock())
    monkeypatch.setattr(
        "llm_sandbox.wsl2_bubblewrap.threading.Thread", Mock(side_effect=[started, unstarted, later, later])
    )
    with pytest.raises(RuntimeError, match="thread allocation failed") as raised:
        _Transport(["C:/Windows/System32/wsl.exe"])
    assert raised.value is failure
    assert process.stdin.closed
    managed.close.assert_called_once()
    started.join.assert_called_once_with(timeout=0.5)
    unstarted.join.assert_not_called()
    later.start.assert_not_called()
