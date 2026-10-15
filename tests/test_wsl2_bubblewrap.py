"""WSL transport boundaries and opt-in Windows/WSL2 enforcement checks."""

from __future__ import annotations

import base64
import io
import json
import os
import sys
import threading
import time
import uuid
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any, ClassVar
from unittest.mock import Mock

import pytest

from llm_sandbox.data import ConsoleOutput
from llm_sandbox.exceptions import (
    CommandEmptyError,
    MissingDependencyError,
    NotOpenSessionError,
    SandboxError,
    SandboxTimeoutError,
    SecurityError,
    ValidationError,
)
from llm_sandbox.local.config import LocalSandboxConfig
from llm_sandbox.local.wsl_protocol import encode_frame, identity, read_frame, validate_policy
from llm_sandbox.local.wsl_worker import Worker, _GuestSession, _validate_mounts, _verify_wsl2
from llm_sandbox.security.seccomp import SeccompPolicy
from llm_sandbox.wsl2_bubblewrap import (
    Wsl2BubblewrapSession,
    _distribution_version,
    _launcher_environment,
    _runtime_path,
    _wsl_executable,
)

if TYPE_CHECKING:
    from collections.abc import Iterator


@pytest.fixture
def windows_launcher(monkeypatch: pytest.MonkeyPatch) -> None:
    """Allow construction on Linux without pretending to exercise Windows isolation."""
    monkeypatch.setattr("llm_sandbox.wsl2_bubblewrap._wsl_executable", lambda: Path("C:/Windows/System32/wsl.exe"))


class FakeTransport:
    """Capture requests separately from shell argv and return framed stream records."""

    instances: ClassVar[list[FakeTransport]] = []

    def __init__(self, argv: list[str]) -> None:
        """Initialize the trusted fake peer."""
        self.argv = argv
        self.sent: list[dict[str, Any]] = []
        self.responses: list[dict[str, Any]] = [{"type": "hello", **identity()}]
        self.closed = False
        self.instances.append(self)

    def send(self, message: dict[str, Any], *, deadline: float | None = None) -> None:
        """Queue a response for the requested operation."""
        self.sent.append(message)
        if message["op"] in {"execute", "run"}:
            self.responses.append({
                "type": "event",
                "id": message["id"],
                "stream": "stdout",
                "data": '{"type":"hello"}\n',
            })
            value = {"exit_code": 0, "stdout": '{"type":"hello"}\n', "stderr": ""}
        else:
            value = None
        self.responses.append({"type": "result", "id": message["id"], "value": value})

    def receive(self, deadline: float | None) -> dict[str, Any]:
        """Return the next already constructed frame."""
        return self.responses.pop(0)

    def close(self) -> None:
        """Record cancellation."""
        self.closed = True


@pytest.fixture
def fake_transport(windows_launcher: None, monkeypatch: pytest.MonkeyPatch) -> type[FakeTransport]:  # noqa: ARG001
    """Replace only the Windows boundary, leaving client protocol validation active."""
    FakeTransport.instances.clear()
    monkeypatch.setattr("llm_sandbox.wsl2_bubblewrap._Transport", FakeTransport)
    monkeypatch.setattr(Wsl2BubblewrapSession, "_verify_distribution", lambda _: None)
    return FakeTransport


def test_non_windows_host_rejected() -> None:
    """There is no Linux-host or direct-process fallback for this backend."""
    if sys.platform == "win32":
        pytest.skip("Requires a non-Windows host")
    with pytest.raises(ValidationError, match="Windows Python"):
        Wsl2BubblewrapSession(distribution="Ubuntu")


@pytest.mark.parametrize("name", ["", "--exec", "Ubuntu\n--exec", "Ubuntu\0", "../Ubuntu"])
def test_distribution_cannot_inject_arguments(windows_launcher: None, name: str) -> None:  # noqa: ARG001
    """Distro selection remains one validated, explicit argument."""
    with pytest.raises(ValueError, match="distribution"):
        Wsl2BubblewrapSession(distribution=name)


@pytest.mark.parametrize("python", ["python3", "/mnt/c/python.exe", "/run/python", "/usr/../bin/python", "/"])
def test_guest_interpreter_is_an_absolute_linux_path(windows_launcher: None, python: str) -> None:  # noqa: ARG001
    """Shared Windows paths and ambiguous interpreter paths fail before spawning WSL."""
    with pytest.raises(ValueError, match="guest_python"):
        Wsl2BubblewrapSession(distribution="Ubuntu", guest_python=python)


def test_distribution_version_is_specific_and_localization_independent() -> None:
    """Default distro and translated status/header text do not affect explicit matching."""
    output = "  NAME STATE VERSION\n* Ubuntu-Old Arrêté 1\n  Ubuntu Déjà démarré 2\n"
    assert _distribution_version(output, "Ubuntu") == 2
    assert _distribution_version(output, "Ubuntu-Old") == 1
    with pytest.raises(MissingDependencyError):
        _distribution_version(output, "Missing")


@pytest.mark.parametrize("release", ["5.15.0-generic", "4.4.0-Microsoft", "6.1.0"])
def test_guest_rejects_non_wsl2_kernel(monkeypatch: pytest.MonkeyPatch, release: str) -> None:
    """WSL availability alone does not authorize execution in an incompatible kernel."""
    monkeypatch.setattr("llm_sandbox.local.wsl_worker.platform.release", lambda: release)
    with pytest.raises(SandboxError, match="WSL2"):
        _verify_wsl2()


@pytest.mark.parametrize(
    "config",
    [
        LocalSandboxConfig(seccomp=None),
        LocalSandboxConfig(seccomp=SeccompPolicy(blocked_syscalls=("ptrace",))),
        LocalSandboxConfig(read_only_paths=("/run/WSL",)),
        LocalSandboxConfig(env={"WSL_INTEROP": "/run/WSL/123_interop"}),
    ],
)
def test_policy_cannot_disable_interop_protection(config: LocalSandboxConfig) -> None:
    """Mandatory syscall hardening, narrow mounts and environment filtering are enforced twice."""
    with pytest.raises(SecurityError):
        validate_policy(config, {})


def test_nested_windows_mount_is_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    """A default /usr grant cannot recursively expose an administrator's Windows mount."""
    monkeypatch.setattr("llm_sandbox.local.wsl_worker.read_only_mounts", lambda *_: (("/usr", "/usr"),))
    monkeypatch.setattr(Path, "read_text", lambda *_args, **_kwargs: "1 0 0:1 / /usr/shared rw - 9p C: rw\n")
    monkeypatch.setattr(Path, "resolve", lambda path, **_kwargs: path)
    config = LocalSandboxConfig()
    with pytest.raises(SecurityError, match="shared filesystem"):
        _validate_mounts(config, "/sandbox")


def test_windows_launcher_environment_contains_no_host_secrets(monkeypatch: pytest.MonkeyPatch) -> None:
    """Only system loader variables are derived from the trusted launcher path."""
    monkeypatch.setenv("WSLENV", "SECRET/u")
    monkeypatch.setenv("SECRET", "sensitive")
    assert _launcher_environment(Path("C:/Windows/System32/wsl.exe")) == {
        "SystemRoot": "C:\\Windows",
        "WINDIR": "C:\\Windows",
        "SystemDrive": "C:",
    }


def test_windows_tmp_mount_is_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    """Private mkdtemp state must never be backed by a Windows bind mount."""
    monkeypatch.setattr("llm_sandbox.local.wsl_worker.read_only_mounts", lambda *_: ())
    monkeypatch.setattr(Path, "resolve", lambda path, **_kwargs: path)
    monkeypatch.setattr(Path, "read_text", lambda *_args, **_kwargs: "1 0 0:1 / /tmp rw - 9p C: rw\n")
    config = LocalSandboxConfig()
    with pytest.raises(SecurityError, match="shared filesystem"):
        _validate_mounts(config, "/sandbox")


@pytest.mark.parametrize("filesystem", ["tmpfs", "9p"])
def test_exact_wsl_resolver_file_grant(monkeypatch: pytest.MonkeyPatch, filesystem: str) -> None:
    """Host networking may expose the generated Linux resolver file, never a Windows-backed file."""
    monkeypatch.setattr(
        "llm_sandbox.local.wsl_worker.read_only_mounts", lambda *_: (("/mnt/wsl/resolv.conf", "/etc/resolv.conf"),)
    )
    monkeypatch.setattr(Path, "resolve", lambda path, **_kwargs: path)
    monkeypatch.setattr(Path, "is_file", lambda _: True)
    monkeypatch.setattr(Path, "read_text", lambda *_args, **_kwargs: f"1 0 0:1 / /mnt/wsl rw - {filesystem} none rw\n")
    config = LocalSandboxConfig(network="host")
    if filesystem == "9p":
        with pytest.raises(SecurityError, match="shared filesystem"):
            _validate_mounts(config, "/sandbox")
    else:
        _validate_mounts(config, "/sandbox")


def test_tmp_symlink_to_root_is_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    """A pathological temporary root fails explicitly before allocating guest state."""
    monkeypatch.setattr("llm_sandbox.local.wsl_worker.read_only_mounts", lambda *_: ())
    monkeypatch.setattr(Path, "read_text", lambda *_args, **_kwargs: "")
    monkeypatch.setattr(Path, "resolve", lambda _path, **_kwargs: Path("/"))
    config = LocalSandboxConfig()
    with pytest.raises(SecurityError, match="integration path"):
        _validate_mounts(config, "/sandbox")


def test_interop_bind_mount_beneath_runtime_is_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    """The nested-mount check covers tmpfs socket bindings as well as Windows drives."""
    monkeypatch.setattr("llm_sandbox.local.wsl_worker.read_only_mounts", lambda *_: (("/usr", "/usr"),))
    monkeypatch.setattr(Path, "resolve", lambda path, **_kwargs: path)
    monkeypatch.setattr(Path, "read_text", lambda *_args, **_kwargs: "1 0 0:1 / /usr/interop rw - tmpfs tmpfs rw\n")
    config = LocalSandboxConfig()
    with pytest.raises(SecurityError, match="Nested runtime mounts"):
        _validate_mounts(config, "/sandbox")


def test_wsl_driver_mount_is_masked_readonly(monkeypatch: pytest.MonkeyPatch) -> None:
    """Standard WSL driver mounts stay hidden while the rest of /usr remains usable."""
    monkeypatch.setattr("llm_sandbox.local.wsl_worker._validate_mounts", lambda *_: None)
    monkeypatch.setattr("llm_sandbox.local.wsl_worker._wsl_driver_directory", lambda: Path("/usr/lib/wsl"))

    @contextmanager
    def launcher(*_args: Any) -> Any:
        yield ["bwrap", "--ro-bind", "/usr", "/usr", "--ro-bind", "/usr/lib", "/lib", "--", "/bin/sh", "-c", "true"], ()

    monkeypatch.setattr("llm_sandbox.bubblewrap.BubblewrapSession._launch_command", launcher)
    session = object.__new__(_GuestSession)
    session.sandbox_config = LocalSandboxConfig()
    session.config = Mock(workdir="/sandbox")
    with session._launch_command("true", "/sandbox") as (argv, _):
        assert argv[7:16] == [
            "--tmpfs",
            str(Path("/lib/wsl")),
            "--remount-ro",
            str(Path("/lib/wsl")),
            "--tmpfs",
            str(Path("/usr/lib/wsl")),
            "--remount-ro",
            str(Path("/usr/lib/wsl")),
            "--",
        ]


def test_wsl_driver_mask_rejects_symlink(monkeypatch: pytest.MonkeyPatch) -> None:
    """Unexpected driver layout cannot redirect a mask onto another mounted runtime path."""
    from llm_sandbox.local.wsl_worker import _wsl_driver_directory

    monkeypatch.setattr(Path, "is_symlink", lambda _: True)
    with pytest.raises(SecurityError, match="canonical real directory"):
        _wsl_driver_directory()


@pytest.mark.parametrize("mountpoint", ["/usr/lib/wsl/drivers", "/usr/unrelated"])
def test_only_standard_wsl_driver_submount_is_exempted(monkeypatch: pytest.MonkeyPatch, mountpoint: str) -> None:
    """The mask exemption never broadens to unrelated Windows mounts beneath the runtime."""
    monkeypatch.setattr("llm_sandbox.local.wsl_worker._wsl_driver_directory", lambda: Path("/usr/lib/wsl"))
    monkeypatch.setattr("llm_sandbox.local.wsl_worker.read_only_mounts", lambda *_: (("/usr", "/usr"),))
    monkeypatch.setattr(Path, "resolve", lambda path, **_kwargs: path)
    monkeypatch.setattr(Path, "read_text", lambda *_args, **_kwargs: f"1 0 0:1 / {mountpoint} rw - 9p none rw\n")
    config = LocalSandboxConfig()
    if mountpoint == "/usr/unrelated":
        with pytest.raises(SecurityError, match="shared filesystem"):
            _validate_mounts(config, "/sandbox")
    else:
        _validate_mounts(config, "/sandbox")


def test_wsl1_preflight_fails_before_worker_launch(windows_launcher: None, monkeypatch: pytest.MonkeyPatch) -> None:  # noqa: ARG001
    """A real distro-version probe is required even when wsl.exe exists."""
    monkeypatch.setattr(
        "llm_sandbox.windows.process.ProcessSupervisor.run",
        lambda *_args, **_kwargs: ConsoleOutput(stdout="Ubuntu Running 1\n"),
    )
    session = Wsl2BubblewrapSession(distribution="Ubuntu")
    with pytest.raises(MissingDependencyError, match="must use WSL2"):
        session.open()


@pytest.mark.parametrize("path", ["/mnt/c/secret", "../file", "/tmp/../etc/passwd", r"C:\secret", "//server/share"])
def test_runtime_file_mapping_rejects_host_paths(path: str) -> None:
    """Both transport endpoints restrict files to the private guest storage."""
    with pytest.raises(SecurityError):
        _runtime_path(path, "/sandbox")


def test_payload_json_is_not_control_protocol(fake_transport: type[FakeTransport]) -> None:
    """Payload output resembling a control frame remains plain stdout data."""
    callback: list[str] = []
    with Wsl2BubblewrapSession(distribution="Ubuntu", skip_environment_setup=True) as sandbox:
        command = "printf '%s' '; powershell.exe -Command malicious'"
        result = sandbox.execute_command(command, on_stdout=callback.append)
        transport = fake_transport.instances[-1]
        assert command not in transport.argv
        assert transport.argv[-4:] == ["-I", "-u", "-m", "llm_sandbox.local.wsl_worker"]
        assert transport.sent[-1]["params"]["command"] == command
        assert result.stdout == '{"type":"hello"}\n'
        assert callback == [result.stdout]
    assert transport.closed


def test_handshake_rejects_different_security_implementation(
    fake_transport: type[FakeTransport], monkeypatch: pytest.MonkeyPatch
) -> None:
    """Equal package versions alone do not accept an older local security implementation."""
    original = fake_transport.receive

    def changed(self: FakeTransport, deadline: float | None) -> dict[str, Any]:
        result = original(self, deadline)
        if result["type"] == "hello":
            result["implementation"] = "old"
        return result

    monkeypatch.setattr(fake_transport, "receive", changed)
    sandbox = Wsl2BubblewrapSession(distribution="Ubuntu")
    with pytest.raises(MissingDependencyError, match="version mismatch"):
        sandbox.open()
    assert fake_transport.instances[-1].closed
    assert not sandbox.is_open


@pytest.mark.parametrize("error", [SecurityError, FileNotFoundError])
def test_rejected_artifact_read_keeps_worker_available(
    fake_transport: type[FakeTransport], monkeypatch: pytest.MonkeyPatch, error: type[Exception]
) -> None:
    """A validated completed read rejection must not discard later safe artifacts."""
    original = fake_transport.send

    def respond(self: FakeTransport, message: dict[str, Any], *, deadline: float | None = None) -> None:
        if message["op"] != "get":
            original(self, message, deadline=deadline)
            return
        if message["params"]["path"] == "bad.png":
            response = {"type": "error", "error": error.__name__, "message": "Artifact is unavailable"}
        else:
            response = {"type": "result", "value": base64.b64encode(b"safe plot").decode("ascii")}
        self.responses.append({"id": message["id"], **response})

    monkeypatch.setattr(fake_transport, "send", respond)
    with Wsl2BubblewrapSession(distribution="Ubuntu", skip_environment_setup=True) as session:
        with pytest.raises(error, match="unavailable"):
            session.get_archive("bad.png")
        assert session.is_open
        archive, metadata = session.get_archive("good.png")
        assert metadata["size"] == len(b"safe plot")
        assert b"safe plot" in archive
        assert not fake_transport.instances[-1].closed


@pytest.mark.parametrize(
    "malformation", ["missing_message", "wrong_message", "extra_field", "wrong_id", "unknown_error"]
)
def test_malformed_artifact_error_closes_worker(
    fake_transport: type[FakeTransport], monkeypatch: pytest.MonkeyPatch, malformation: str
) -> None:
    """Only the exact completed and explicitly recoverable errors permit another get request."""
    original = fake_transport.send

    def respond(self: FakeTransport, message: dict[str, Any], *, deadline: float | None = None) -> None:
        if message["op"] != "get":
            original(self, message, deadline=deadline)
            return
        response = {"id": message["id"], "type": "error", "error": "SecurityError", "message": "rejected"}
        if malformation == "missing_message":
            response.pop("message")
        elif malformation == "wrong_message":
            response["message"] = None
        elif malformation == "extra_field":
            response["unexpected"] = True
        elif malformation == "wrong_id":
            response["id"] = message["id"] + 1
        else:
            response["error"] = "UnknownError"
        self.responses.append(response)

    monkeypatch.setattr(fake_transport, "send", respond)
    with Wsl2BubblewrapSession(distribution="Ubuntu", skip_environment_setup=True) as session:
        with pytest.raises(SandboxError):
            session.get_archive("bad.png")
        assert not session.is_open
        assert fake_transport.instances[-1].closed


def test_fingerprint_covers_base_session(monkeypatch: pytest.MonkeyPatch) -> None:
    """Equal distribution versions cannot hide differing transitive execution semantics."""
    expected = identity()
    original = Path.read_bytes

    def modified(path: Path) -> bytes:
        content = original(path)
        return content + b"\n# changed execution semantics\n" if path.name == "session_base.py" else content

    monkeypatch.setattr(Path, "read_bytes", modified)
    actual = identity()
    assert actual["package"] == expected["package"]
    assert actual["implementation"] != expected["implementation"]


def test_close_during_probe_prevents_late_worker(
    fake_transport: type[FakeTransport], monkeypatch: pytest.MonkeyPatch
) -> None:
    """A close before worker registration cancels the whole pending open generation."""
    entered, release = threading.Event(), threading.Event()
    failures: list[Exception] = []

    def probe(_: Wsl2BubblewrapSession) -> None:
        entered.set()
        assert release.wait(5)

    monkeypatch.setattr(Wsl2BubblewrapSession, "_verify_distribution", probe)
    session = Wsl2BubblewrapSession(distribution="Ubuntu")

    def open_session() -> None:
        try:
            session.open()
        except SandboxError as exc:
            failures.append(exc)

    thread = threading.Thread(target=open_session, daemon=True)
    thread.start()
    try:
        assert entered.wait(5)
        session.close()
    finally:
        release.set()
        thread.join(5)
    assert not thread.is_alive()
    assert failures
    assert not fake_transport.instances
    assert not session.is_open


@pytest.mark.parametrize("error", [RuntimeError, SecurityError])
def test_callback_exception_closes_guest(fake_transport: type[FakeTransport], error: type[Exception]) -> None:
    """An abandoned streamed result never leaves the remote command/session running."""

    def fail(_: str) -> None:
        msg = "callback failed"
        raise error(msg)

    sandbox = Wsl2BubblewrapSession(distribution="Ubuntu", skip_environment_setup=True)
    sandbox.open()
    with pytest.raises(error, match="callback failed"):
        sandbox.execute_command("echo", on_stdout=fail)
    assert fake_transport.instances[-1].closed
    assert not sandbox.is_open


@pytest.mark.parametrize(("pointer_bytes", "native_present"), [(8, True), (4, True), (4, False)])
def test_wsl_launcher_uses_system_api_not_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, pointer_bytes: int, native_present: bool
) -> None:
    """Both native and WOW64 clients resolve only the trusted Windows system launcher."""
    system = tmp_path / "System32"
    system.mkdir()
    (system / "wsl.exe").touch()
    native = tmp_path / "Sysnative" / "wsl.exe"
    if native_present:
        native.parent.mkdir()
        native.touch()

    def system_directory(buffer: Any, _length: int) -> int:
        buffer.value = str(system)
        return len(str(system))

    kernel = Mock(GetSystemDirectoryW=Mock(side_effect=system_directory))
    monkeypatch.setattr("llm_sandbox.wsl2_bubblewrap.sys", SimpleNamespace(platform="win32"))
    monkeypatch.setattr("llm_sandbox.wsl2_bubblewrap.ctypes.WinDLL", lambda *_args, **_kwargs: kernel, raising=False)
    monkeypatch.setattr("llm_sandbox.wsl2_bubblewrap.ctypes.sizeof", lambda _: pointer_bytes)
    monkeypatch.setenv("PATH", str(tmp_path / "untrusted"))
    expected = native if pointer_bytes == 4 and native_present else system / "wsl.exe"
    assert _wsl_executable() == expected
    kernel.GetSystemDirectoryW.assert_called_once()


@pytest.mark.parametrize("failure", ["api_error", "truncated", "missing_launcher"])
def test_wsl_launcher_resolution_fails_closed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, failure: str) -> None:
    """A broken system API or absent wsl.exe never falls back to a caller-provided PATH."""

    def system_directory(buffer: Any, length: int) -> int:
        buffer.value = str(tmp_path / "System32")
        return {"api_error": 0, "truncated": length, "missing_launcher": len(buffer.value)}[failure]

    kernel = Mock(GetSystemDirectoryW=Mock(side_effect=system_directory))
    monkeypatch.setattr("llm_sandbox.wsl2_bubblewrap.sys", SimpleNamespace(platform="win32"))
    monkeypatch.setattr("llm_sandbox.wsl2_bubblewrap.ctypes.WinDLL", lambda *_args, **_kwargs: kernel, raising=False)
    with pytest.raises(MissingDependencyError):
        _wsl_executable()


@pytest.mark.parametrize(
    ("options", "exception"),
    [
        ({"workdir": "/"}, ValueError),
        ({"workdir": "/mnt/private"}, ValueError),
        ({"workdir": "/sandbox/../escape"}, ValueError),
        ({"default_timeout": 0}, ValueError),
        ({"execution_timeout": float("inf")}, ValueError),
        ({"session_timeout": float("nan")}, ValueError),
        ({"env_vars": {"": "value"}}, ValueError),
        ({"env_vars": {"KEY": "value\0"}}, ValueError),
        ({"env_vars": {"KEY": 1}}, ValueError),
        ({"image": "container:latest"}, ValueError),
        ({"unrecognized_option": True}, TypeError),
    ],
)
def test_invalid_client_policy_fails_before_worker_creation(
    fake_transport: type[FakeTransport], options: dict[str, Any], exception: type[Exception]
) -> None:
    """Path, deadline, environment and backend-option mistakes cannot reach guest execution."""
    with pytest.raises(exception):
        Wsl2BubblewrapSession(distribution="Ubuntu", **options)
    assert not fake_transport.instances


def test_client_api_preserves_operation_data_and_file_limits(
    fake_transport: type[FakeTransport], monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The Windows boundary receives bounded bytes while commands/install stay in guest RPC."""
    payload = b"input\x00bytes"
    read = Mock(return_value=payload)
    write = Mock()
    monkeypatch.setattr("llm_sandbox.windows.files.safe_read_file", read)
    monkeypatch.setattr("llm_sandbox.windows.files.safe_write_file", write)
    original = fake_transport.send

    def respond(self: FakeTransport, message: dict[str, Any], *, deadline: float | None = None) -> None:
        original(self, message, deadline=deadline)
        if message["op"] == "get":
            self.responses[-1]["value"] = base64.b64encode(payload).decode("ascii")

    monkeypatch.setattr(fake_transport, "send", respond)
    config = LocalSandboxConfig(max_artifact_bytes=32, env={"ALLOWED": "yes"})
    with Wsl2BubblewrapSession(
        distribution="Ubuntu",
        sandbox_config=config,
        workdir="/workspace",
        env_vars={"EXPLICIT": "value"},
        default_timeout=None,
    ) as session:
        output = session.run("print('guest')", libraries=["example"], timeout=5)
        assert output.exit_code == 0
        session.execute_command("pwd", workdir="/workspace/subdir")
        session.install(["example"])
        session.copy_to_runtime(str(tmp_path / "input"), "/tmp/input")
        session.copy_from_runtime("output", str(tmp_path / "output"))
        session._ensure_directory_exists("subdir")
        session._ensure_ownership(["subdir", "/tmp/input"])
        requests = fake_transport.instances[-1].sent
        assert [request["op"] for request in requests] == ["open", "run", "execute", "install", "put", "get", "mkdir"]
        assert requests[1]["params"] == {"code": "print('guest')", "libraries": ["example"], "timeout": 5}
        assert requests[4]["params"] == {"path": "/tmp/input", "data": base64.b64encode(payload).decode("ascii")}
        assert requests[0]["params"]["env_vars"] == {"EXPLICIT": "value"}
        read.assert_called_once_with((tmp_path / "input").absolute(), 32)
        write.assert_called_once_with((tmp_path / "output").absolute(), payload)


@pytest.mark.parametrize("timeout", [0, -1, float("inf"), float("nan")])
def test_invalid_per_run_timeout_does_not_reach_worker(fake_transport: type[FakeTransport], timeout: float) -> None:
    """Validation errors before dispatch leave an otherwise usable session open."""
    with Wsl2BubblewrapSession(distribution="Ubuntu") as session:
        with pytest.raises(ValueError, match="timeout"):
            session.run("pass", timeout=timeout)
        assert [message["op"] for message in fake_transport.instances[-1].sent] == ["open"]
        assert session.is_open


def test_closed_session_and_unsupported_paths_cannot_execute(fake_transport: type[FakeTransport]) -> None:
    """Closed sessions and container-only adapter hooks never start a worker or host fallback."""
    session = Wsl2BubblewrapSession(distribution="Ubuntu")
    with pytest.raises(NotOpenSessionError):
        session.execute_command("true")
    with pytest.raises(CommandEmptyError):
        session.execute_command("")
    with pytest.raises(NotOpenSessionError):
        session._rpc("get", {"path": "input"})
    with pytest.raises(NotImplementedError):
        session._connect_to_existing_container("old-container")
    with pytest.raises(NotImplementedError):
        session._process_stream_output(b"unframed output")
    assert not fake_transport.instances


def test_timer_closes_only_its_own_session_generation(
    fake_transport: type[FakeTransport], monkeypatch: pytest.MonkeyPatch
) -> None:
    """A late old timer must not cancel a freshly reopened worker."""
    timers: list[Mock] = []

    def make_timer(interval: float, function: Any, args: tuple[Any, ...]) -> Mock:
        timer = Mock(interval=interval, function=function, args=args)
        timers.append(timer)
        return timer

    monkeypatch.setattr("llm_sandbox.wsl2_bubblewrap.threading.Timer", make_timer)
    session = Wsl2BubblewrapSession(distribution="Ubuntu", session_timeout=5)
    session.open()
    session.open()
    assert len(fake_transport.instances) == 1
    first = fake_transport.instances[0]
    timers[0].function(*timers[0].args)
    assert first.closed
    assert not session.is_open
    timers[0].cancel.assert_called_once()
    session.open()
    timers[0].function(*timers[0].args)
    assert session.is_open
    assert not fake_transport.instances[1].closed
    session._handle_timeout()
    assert not session.is_open
    assert fake_transport.instances[1].closed


def test_close_during_completed_open_cannot_publish_live_session(
    fake_transport: type[FakeTransport], monkeypatch: pytest.MonkeyPatch
) -> None:
    """Cancellation after the guest open reply still wins over host session publication."""
    session = Wsl2BubblewrapSession(distribution="Ubuntu")
    original = fake_transport.send

    def cancel(self: FakeTransport, message: dict[str, Any], *, deadline: float | None = None) -> None:
        original(self, message, deadline=deadline)
        if message["op"] == "open":
            session.close()

    monkeypatch.setattr(fake_transport, "send", cancel)
    with pytest.raises(NotOpenSessionError):
        session.open()
    assert fake_transport.instances[-1].closed
    assert not session.is_open


def test_stream_callback_reentry_closes_worker(fake_transport: type[FakeTransport]) -> None:
    """Reentrant callbacks cannot interleave a second operation with an active command."""
    with Wsl2BubblewrapSession(distribution="Ubuntu") as session:

        def reenter(_: str) -> None:
            session.execute_command("nested")

        with pytest.raises(SandboxError, match="reentered"):
            session.execute_command("echo", on_stdout=reenter)
        assert not session.is_open
        assert fake_transport.instances[-1].closed


@pytest.mark.parametrize("exit_code", [0, 1])
@pytest.mark.usefixtures("windows_launcher")
def test_distribution_probe_validates_result_and_releases_supervisor(
    monkeypatch: pytest.MonkeyPatch, exit_code: int
) -> None:
    """Successful and failing WSL queries both release their owned process supervisor."""
    probe = Mock()
    probe.run.return_value = ConsoleOutput(exit_code=exit_code, stdout="Ubuntu Running 2\n")
    monkeypatch.setattr("llm_sandbox.windows.process.ProcessSupervisor", lambda **_kwargs: probe)
    session = Wsl2BubblewrapSession(distribution="Ubuntu")
    if exit_code:
        with pytest.raises(MissingDependencyError, match="must use WSL2"):
            session._verify_distribution()
    else:
        session._verify_distribution()
    assert probe.run.call_args.args[0][-2:] == ["--list", "--verbose"]
    probe.close.assert_called_once()
    assert session._startup_probe is None


@pytest.mark.usefixtures("windows_launcher")
def test_cancellation_reaches_active_distribution_probe(monkeypatch: pytest.MonkeyPatch) -> None:
    """A close during the real preflight path cancels the probe and prevents worker construction."""
    probe = Mock()
    worker = Mock()
    session = Wsl2BubblewrapSession(distribution="Ubuntu")

    def query(*_args: Any, **_kwargs: Any) -> ConsoleOutput:
        session.close()
        return ConsoleOutput(stdout="Ubuntu Running 2\n")

    probe.run.side_effect = query
    monkeypatch.setattr("llm_sandbox.windows.process.ProcessSupervisor", lambda **_kwargs: probe)
    monkeypatch.setattr("llm_sandbox.wsl2_bubblewrap._Transport", worker)
    with pytest.raises(NotOpenSessionError):
        session.open()
    assert probe.close.call_count == 2
    worker.assert_not_called()
    assert not session.is_open
    assert session._startup_probe is None


@pytest.mark.usefixtures("windows_launcher")
def test_already_cancelled_startup_never_runs_distribution_query(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Preflight respects cancellation even before a native process is created."""
    probe = Mock()
    monkeypatch.setattr("llm_sandbox.windows.process.ProcessSupervisor", lambda **_kwargs: probe)
    session = Wsl2BubblewrapSession(distribution="Ubuntu")
    session._opening_cancelled = threading.Event()
    session._opening_cancelled.set()
    with pytest.raises(NotOpenSessionError):
        session._verify_distribution()
    probe.run.assert_not_called()


@pytest.mark.parametrize(
    "event",
    [
        {"type": "unknown", "stream": "stdout", "data": "text"},
        {"type": "event", "stream": "unknown", "data": "text"},
        {"type": "event", "stream": "stdout", "data": 42},
    ],
)
def test_invalid_output_event_closes_session(
    fake_transport: type[FakeTransport], monkeypatch: pytest.MonkeyPatch, event: dict[str, Any]
) -> None:
    """Invalid event envelopes cannot become user-visible output or keep a worker alive."""
    original = fake_transport.send

    def respond(self: FakeTransport, message: dict[str, Any], *, deadline: float | None = None) -> None:
        original(self, message, deadline=deadline)
        if message["op"] == "run":
            self.responses[0] = {"id": message["id"], **event}

    monkeypatch.setattr(fake_transport, "send", respond)
    with Wsl2BubblewrapSession(distribution="Ubuntu") as session:
        with pytest.raises(SandboxError, match="Unexpected WSL worker response"):
            session.run("pass")
        assert not session.is_open
        assert fake_transport.instances[-1].closed


def test_excessive_output_events_cancel_worker(fake_transport: type[FakeTransport]) -> None:
    """The host independently caps events even if a mismatching peer ignores its own limit."""
    delivered: list[str] = []
    with Wsl2BubblewrapSession(distribution="Ubuntu", sandbox_config=LocalSandboxConfig(max_output_bytes=1)) as session:
        with pytest.raises(SandboxError, match="transport limit"):
            session.run("pass", on_stdout=delivered.append)
        assert delivered == []
        assert not session.is_open
        assert fake_transport.instances[-1].closed


def test_streaming_routes_stderr_and_logs_without_callback(
    fake_transport: type[FakeTransport], monkeypatch: pytest.MonkeyPatch
) -> None:
    """Verbose streaming and stderr callbacks receive only their decoded event data."""
    original = fake_transport.send

    def respond(self: FakeTransport, message: dict[str, Any], *, deadline: float | None = None) -> None:
        original(self, message, deadline=deadline)
        if message["op"] == "execute" and message["params"]["command"] == "stderr":
            self.responses[0]["stream"] = "stderr"

    monkeypatch.setattr(fake_transport, "send", respond)
    logs: list[str] = []
    errors: list[str] = []
    with Wsl2BubblewrapSession(distribution="Ubuntu", stream=True, verbose=True) as session:
        monkeypatch.setattr(session, "_log", logs.append)
        session.execute_command("stdout")
        session.execute_command("stderr", on_stderr=errors.append)
    assert logs == ['{"type":"hello"}\n']
    assert errors == logs


def test_oversized_download_never_reaches_windows_write_boundary(
    fake_transport: type[FakeTransport], monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A peer returning too much file data cannot bypass the host artifact limit."""
    write = Mock()
    monkeypatch.setattr("llm_sandbox.windows.files.safe_write_file", write)
    original = fake_transport.send

    def respond(self: FakeTransport, message: dict[str, Any], *, deadline: float | None = None) -> None:
        original(self, message, deadline=deadline)
        if message["op"] == "get":
            self.responses[-1]["value"] = base64.b64encode(b"oversized").decode("ascii")

    monkeypatch.setattr(fake_transport, "send", respond)
    destination = str(tmp_path / "never-published")
    session = Wsl2BubblewrapSession(distribution="Ubuntu", sandbox_config=LocalSandboxConfig(max_artifact_bytes=4))
    with session, pytest.raises(SandboxError, match="artifact limit"):
        session.copy_from_runtime("result", destination)
    write.assert_not_called()


def test_protocol_rejects_unstructured_and_oversized_records(monkeypatch: pytest.MonkeyPatch) -> None:
    """A malformed worker cannot allocate unbounded input or impersonate a response."""
    monkeypatch.setattr("llm_sandbox.local.wsl_protocol.MAX_FRAME_BYTES", 32)
    for data in (b"not-json\n", b"[]\n", b"x" * 33, b"{}"):
        pipe = io.BytesIO(data)
        with pytest.raises(SandboxError):
            read_frame(pipe)
    with pytest.raises(SandboxError):
        encode_frame({"value": "x" * 50})


def test_eof_closes_guest_while_request_is_active() -> None:
    """The input reader can cancel independently of a blocked execution dispatch."""
    worker = Worker(io.BytesIO(b""), io.BytesIO())
    session = Mock()
    worker.session = session
    worker._read_requests()
    assert worker.stopped.is_set()
    session.close.assert_called_once()


def test_missed_lease_closes_guest_even_without_eof(monkeypatch: pytest.MonkeyPatch) -> None:
    """Broken WSL transport implementations cannot leave a payload running indefinitely."""
    monkeypatch.setattr("llm_sandbox.local.wsl_worker.LEASE_SECONDS", 0.01)
    worker = Worker(io.BytesIO(), io.BytesIO())
    worker.session = Mock()
    worker._last_heartbeat = time.monotonic() - 1
    worker._watch_lease()
    assert worker.stopped.is_set()
    worker.session.close.assert_called_once()


def test_guest_output_backpressure_is_cancelled(monkeypatch: pytest.MonkeyPatch) -> None:
    """A full host output pipe cannot trap a callback while lease cleanup waits for it."""
    worker = Worker(io.BytesIO(), io.BytesIO())
    worker._output_fd = 42
    blocked = threading.Event()
    failures: list[Exception] = []

    def write(_fd: int, _data: bytes) -> int:
        blocked.set()
        raise BlockingIOError

    monkeypatch.setattr("llm_sandbox.local.wsl_worker.os.write", write)

    def send() -> None:
        try:
            worker.send({"type": "event", "data": "payload"})
        except SandboxError as exc:
            failures.append(exc)

    thread = threading.Thread(target=send, daemon=True)
    thread.start()
    try:
        assert blocked.wait(5)
    finally:
        worker.stop()
        thread.join(5)
    assert not thread.is_alive()
    assert failures


def test_guest_output_writes_complete_frames(monkeypatch: pytest.MonkeyPatch) -> None:
    """Raw unbuffered WSL pipes can accept only part of a control record per write."""
    worker = Worker(io.BytesIO(), io.BytesIO())
    worker._output_fd = 42
    captured = bytearray()

    def write(_fd: int, data: bytes) -> int:
        captured.extend(data[:3])
        return len(data[:3])

    monkeypatch.setattr("llm_sandbox.local.wsl_worker.os.write", write)
    worker.send({"type": "event", "data": "payload"})
    assert read_frame(io.BytesIO(captured)) == {"type": "event", "data": "payload"}


def test_guest_late_start_rejected_after_eof() -> None:
    """An EOF that races with open prevents the next trusted launcher spawn."""
    session = object.__new__(_GuestSession)
    session._transport_stopped = threading.Event()
    session._transport_stopped.set()
    with pytest.raises(SandboxError, match="transport has closed"):
        session._require_open()


@pytest.fixture
def real_wsl2(request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch) -> Iterator[Wsl2BubblewrapSession]:
    """Require explicit opt-in; a configured runner must fail rather than skip on misconfiguration."""
    distro = os.environ.get("LLM_SANDBOX_WSL2_DISTRO")
    if not distro:
        pytest.skip("Set LLM_SANDBOX_WSL2_DISTRO on a Windows host for real enforcement tests")
    monkeypatch.setenv("LLM_SANDBOX_WSL_HOST_SECRET", "must-not-enter-the-guest")
    monkeypatch.setenv("WSLENV", "LLM_SANDBOX_WSL_HOST_SECRET")
    session = Wsl2BubblewrapSession(
        distribution=distro,
        guest_python=os.environ.get("LLM_SANDBOX_WSL2_PYTHON", "/usr/bin/python3"),
        skip_environment_setup=True,
        **getattr(request, "param", {}),
    )
    with session:
        yield session


@pytest.mark.integration
def test_real_wsl2_isolation_persistence_and_copies(real_wsl2: Wsl2BubblewrapSession, tmp_path: Path) -> None:
    """Exercise the actual Windows launcher, WSL guest, namespaces and file boundary."""
    with real_wsl2 as session:
        assert session.run("print('guest')").stdout == "guest\n"
        assert session.execute_command("printf retained > state").exit_code == 0
        assert session.execute_command("cat state").stdout == "retained"
        assert (
            session.execute_command(
                "test ! -e /mnt && test ! -e /run/WSL && test ! -e /init && test ! -e /home && test ! -e /root"
            ).exit_code
            == 0
        )
        assert session.execute_command('test -z "$WSL_INTEROP$WSLENV$LLM_SANDBOX_WSL_HOST_SECRET"').exit_code == 0
        result = session.run("import socket\nsocket.create_connection(('1.1.1.1', 443), timeout=1)")
        assert result.exit_code != 0
        source, destination = tmp_path / "upload.txt", tmp_path / "download.txt"
        source.write_bytes(b"roundtrip")
        session.copy_to_runtime(str(source), "upload.txt")
        session.copy_from_runtime("upload.txt", str(destination))
        assert destination.read_bytes() == b"roundtrip"
        assert (
            session.execute_command(
                "mkdir -p /tmp/sandbox_plots; "
                "printf first > /tmp/sandbox_plots/01_safe.png; "
                "ln -s /etc/passwd /tmp/sandbox_plots/02_link.png; "
                "mkdir /tmp/sandbox_plots/03_directory.png; "
                "printf last > /tmp/sandbox_plots/04_safe.png"
            ).exit_code
            == 0
        )
        plots = session.language_handler.extract_plots(session, "/tmp/sandbox_plots")
        assert [base64.b64decode(plot.content_base64) for plot in plots] == [b"first", b"last"]


@pytest.mark.integration
def test_real_wsl2_command_timeout(real_wsl2: Wsl2BubblewrapSession) -> None:
    """A live guest command deadline cancels the remote session and its process tree."""
    with pytest.raises(SandboxTimeoutError):
        real_wsl2.run("import time; time.sleep(60)", timeout=0.5)
    assert not real_wsl2.is_open


@pytest.mark.integration
@pytest.mark.parametrize("real_wsl2", [{"sandbox_config": LocalSandboxConfig(max_output_bytes=1024)}], indirect=True)
def test_real_wsl2_output_limit(real_wsl2: Wsl2BubblewrapSession) -> None:
    """Excess payload output is rejected before it can fill the host transport indefinitely."""
    with pytest.raises(SandboxError, match="output"):
        real_wsl2.run("print('x' * 8192)")
    assert not real_wsl2.is_open


@pytest.mark.integration
@pytest.mark.parametrize("kind", ["symlink", "hardlink", "directory"])
def test_real_wsl2_download_rejects_nonregular_or_linked_files(
    real_wsl2: Wsl2BubblewrapSession, tmp_path: Path, kind: str
) -> None:
    """Guest file transfer does not follow payload-created links or recursively copy directories."""
    commands = {"symlink": "ln -s original target", "hardlink": "ln original target", "directory": "mkdir target"}
    assert real_wsl2.execute_command("printf private > original; " + commands[kind]).exit_code == 0
    destination = tmp_path / "download"
    destination_path = str(destination)
    with pytest.raises(SecurityError):
        real_wsl2.copy_from_runtime("target", destination_path)
    assert not destination.exists()
    assert real_wsl2.is_open
    assert real_wsl2.execute_command("printf safe > good").exit_code == 0
    real_wsl2.copy_from_runtime("good", destination_path)
    assert destination.read_bytes() == b"safe"


@pytest.mark.integration
@pytest.mark.parametrize("real_wsl2", [{"sandbox_config": LocalSandboxConfig(network="host")}], indirect=True)
def test_real_wsl2_host_network_has_only_resolver_file(real_wsl2: Wsl2BubblewrapSession) -> None:
    """Standard WSL generated resolver configuration is readable without exposing its parent mount."""
    result = real_wsl2.execute_command("test -r /etc/resolv.conf && test ! -e /mnt/wsl && test ! -e /run/WSL")
    assert result.exit_code == 0


_INSPECT_GUEST = """
import json, os, pathlib, sys
pid, workspace, marker = sys.argv[1:]
running = []
for item in pathlib.Path('/proc').iterdir():
    if item.name.isdigit() and int(item.name) != os.getpid():
        try:
            if marker.encode() in (item / 'cmdline').read_bytes():
                running.append(item.name)
        except (FileNotFoundError, PermissionError, ProcessLookupError):
            pass
print(json.dumps({'worker': pathlib.Path('/proc', pid).exists(),
                  'workspace': pathlib.Path(workspace).exists(), 'running': running}))
"""


@pytest.mark.integration
@pytest.mark.parametrize("kill_transport", [False, True])
def test_real_wsl2_transport_loss_cleans_payload_and_workspace(
    real_wsl2: Wsl2BubblewrapSession,
    kill_transport: bool,
) -> None:
    """Check both orderly EOF and a killed wsl.exe against live guest process/storage state."""
    from llm_sandbox.windows.process import ProcessSupervisor

    session = real_wsl2
    transport = session._transport
    assert transport is not None
    metadata = session._guest_state
    marker = uuid.uuid4().hex
    started = threading.Event()
    failures: list[Exception] = []

    def execute() -> None:
        try:
            session.execute_command(
                f"printf started; /usr/bin/python3 -c 'import time; time.sleep(60)' # {marker}",
                on_stdout=lambda _: started.set(),
            )
        except SandboxError as exc:
            failures.append(exc)

    thread = threading.Thread(target=execute, daemon=True)
    thread.start()
    try:
        assert started.wait(15), "Payload never reached its sandboxed execution point"
        if kill_transport:
            transport.process.kill()
        else:
            session.close()
        thread.join(15)
        assert not thread.is_alive()
        assert failures
        deadline = time.monotonic() + 15
        while True:
            inspector = ProcessSupervisor(max_output_bytes=65536)
            output = inspector.run(
                [
                    str(session.wsl_executable),
                    "--distribution",
                    session.distribution,
                    "--cd",
                    "/",
                    "--exec",
                    session.guest_python,
                    "-I",
                    "-c",
                    _INSPECT_GUEST,
                    str(metadata["pid"]),
                    metadata["workspace"],
                    marker,
                ],
                env=_launcher_environment(session.wsl_executable),
                timeout=10,
            )
            state = json.loads(output.stdout)
            if not any(state.values()):
                break
            assert time.monotonic() < deadline, state
            time.sleep(0.2)
    finally:
        session.close()
        thread.join(15)
