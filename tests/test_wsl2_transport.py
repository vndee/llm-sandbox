"""Real trusted pipes verify host deadlines without needing a WSL installation."""

from __future__ import annotations

import os
import sys
import threading
import time
from pathlib import Path
from typing import Any, BinaryIO

import pytest

from llm_sandbox.exceptions import NotOpenSessionError, SandboxTimeoutError
from llm_sandbox.wsl2_bubblewrap import _Transport


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
