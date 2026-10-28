"""Trusted subprocess tests for Windows-compatible process supervision."""

from __future__ import annotations

import ctypes
import json
import os
import queue
import subprocess
import sys
import threading
import time
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any
from unittest.mock import Mock

import pytest

from llm_sandbox.exceptions import MissingDependencyError, SandboxError, SandboxTimeoutError
from llm_sandbox.windows import process as process_module
from llm_sandbox.windows.process import ManagedProcess, ProcessSupervisor, resolve_executable

if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path


@pytest.fixture
def environment() -> dict[str, str]:
    """Provide only Python's required loader settings, never the caller's general environment."""
    return {name: os.environ[name] for name in ("SystemRoot", "WINDIR", "LD_LIBRARY_PATH") if name in os.environ}


def python_command(code: str) -> list[str]:
    """Use the current test interpreter with fixed trusted code."""
    return [sys.executable, "-u", "-c", code]


def test_exact_argv_clean_environment_and_no_inherited_stdin(
    environment: dict[str, str], monkeypatch: pytest.MonkeyPatch
) -> None:
    """Shell metacharacters remain literal and no host secrets/stdin enter the launcher."""
    monkeypatch.setenv("SUPERVISOR_PRIVATE_SECRET", "must-not-inherit")
    arguments = ["spaces 'quotes' ; $(echo nope) & %TEMP%", 'back\\slash"', ""]
    code = (
        "import json,os,sys; print(json.dumps([sys.argv[1:],os.getenv('SUPERVISOR_PRIVATE_SECRET'),sys.stdin.read()]))"
    )
    output = ProcessSupervisor().run([*python_command(code), *arguments], env=environment)
    assert output.exit_code == 0
    assert json.loads(output.stdout) == [arguments, None, ""]


def test_full_duplex_input_cannot_deadlock_on_full_output_pipes(environment: dict[str, str]) -> None:
    """Read output concurrently with writing a request larger than both OS pipe buffers."""
    payload = b"request" * 200_000
    code = "import sys; sys.stdout.write('x'*200000); sys.stdout.flush(); print(len(sys.stdin.buffer.read()))"
    output = ProcessSupervisor().run(python_command(code), env=environment, input=payload, timeout=5)
    assert output.exit_code == 0
    assert output.stdout == "x" * 200000 + str(len(payload)) + os.linesep


def test_incremental_utf8_callbacks_on_both_pipes(environment: dict[str, str]) -> None:
    """Split multibyte output is decoded once and callbacks preserve both streams."""
    stdout: list[str] = []
    stderr: list[str] = []
    code = "import os,time;os.write(1,b'\\xe2');time.sleep(.03);os.write(1,b'\\x82\\xac');os.write(2,b'error')"
    output = ProcessSupervisor().run(
        python_command(code), env=environment, on_stdout=stdout.append, on_stderr=stderr.append
    )
    assert "".join(stdout) == output.stdout == "€"
    assert "".join(stderr) == output.stderr == "error"


def test_explicit_utf16_decoder_for_windows_tool_output(environment: dict[str, str]) -> None:
    """Windows tools that emit UTF-16 can use the same bounded byte reader."""
    output = ProcessSupervisor(output_encoding="utf-16-le").run(
        python_command("import sys;sys.stdout.buffer.write('Windows ✓'.encode('utf-16-le'))"), env=environment
    )
    assert output.stdout == "Windows ✓"


def test_combined_output_limit_and_reuse_after_failure(environment: dict[str, str]) -> None:
    """The byte limit covers stdout and stderr together and cleanup allows another command."""
    supervisor = ProcessSupervisor(max_output_bytes=1024)
    output = supervisor.run(python_command("import os;os.write(1,b'a'*512);os.write(2,b'b'*512)"), env=environment)
    assert len(output.stdout) + len(output.stderr) == 1024
    overflow_command = python_command("import os;os.write(1,b'a'*512);os.write(2,b'b'*513)")
    with pytest.raises(SandboxError, match="output exceeded"):
        supervisor.run(overflow_command, env=environment)
    assert supervisor.run(python_command("print('recovered')"), env=environment).stdout == "recovered" + os.linesep


@pytest.mark.parametrize("closed_pipes", [False, True])
def test_deadline_terminates_process_and_pipe_threads(environment: dict[str, str], closed_pipes: bool) -> None:
    """Timeouts remain effective even after the process closes both output streams."""
    before = {thread.ident for thread in threading.enumerate()}
    code = "import os,time;" + ("os.close(1);os.close(2);" if closed_pipes else "") + "time.sleep(20)"
    command = python_command(code)
    supervisor = ProcessSupervisor()
    with pytest.raises(SandboxTimeoutError):
        supervisor.run(command, env=environment, timeout=0.3)
    assert {thread.ident for thread in threading.enumerate()} == before


def test_deadline_cancels_a_writer_whose_child_never_reads_input(environment: dict[str, str]) -> None:
    """Closing a synchronous Windows input pipe cannot deadlock ahead of job termination."""
    before = {thread.ident for thread in threading.enumerate()}
    command = python_command("import time;time.sleep(20)")
    supervisor = ProcessSupervisor()
    started = time.monotonic()
    with pytest.raises(SandboxTimeoutError):
        supervisor.run(command, env=environment, input=b"x" * (1024 * 1024), timeout=0.15)
    assert time.monotonic() - started < 3
    assert {thread.ident for thread in threading.enumerate()} == before


def test_callback_failure_terminates_process_without_leaking_workers(environment: dict[str, str]) -> None:
    """A callback exception cannot strand a launcher blocked on output or sleeping."""
    before = {thread.ident for thread in threading.enumerate()}

    def fail(_: str) -> None:
        msg = "callback rejected output"
        raise ValueError(msg)

    supervisor = ProcessSupervisor()
    command = python_command("import time;print('ready');time.sleep(20)")
    with pytest.raises(ValueError, match="callback rejected"):
        supervisor.run(command, env=environment, on_stdout=fail)
    assert {thread.ident for thread in threading.enumerate()} == before
    assert supervisor.run(python_command("print('alive')"), env=environment).stdout == "alive" + os.linesep


def test_decode_failure_terminates_and_reaps_launcher(environment: dict[str, str]) -> None:
    """Invalid output takes the same cleanup path as callback and deadline failures."""
    supervisor = ProcessSupervisor()
    command = python_command("import os,time;os.write(1,b'\\xff');time.sleep(20)")
    with pytest.raises(UnicodeDecodeError):
        supervisor.run(command, env=environment)


@pytest.mark.parametrize("permanent", [False, True])
def test_cancel_from_another_thread_and_sticky_close(environment: dict[str, str], permanent: bool) -> None:
    """Close prevents future launches while ordinary cancellation leaves the supervisor reusable."""
    supervisor = ProcessSupervisor()
    ready = threading.Event()
    errors: list[BaseException] = []

    def run() -> None:
        try:
            supervisor.run(
                python_command("import time;print('ready');time.sleep(20)"),
                env=environment,
                on_stdout=lambda _: ready.set(),
            )
        except BaseException as exc:  # noqa: BLE001
            errors.append(exc)

    worker = threading.Thread(target=run)
    worker.start()
    try:
        assert ready.wait(5)
        if permanent:
            supervisor.close()
        else:
            supervisor.cancel()
        worker.join(5)
        assert not worker.is_alive()
        assert len(errors) == 1
        assert isinstance(errors[0], SandboxError)
        if permanent:
            command = python_command("print('must not run')")
            with pytest.raises(SandboxError, match="closed"):
                supervisor.run(command, env=environment)
        else:
            assert supervisor.run(python_command("print('reused')"), env=environment).stdout == "reused" + os.linesep
    finally:
        supervisor.close()
        worker.join(5)


def test_close_waits_for_inflight_creation_and_kills_late_launcher(
    environment: dict[str, str], monkeypatch: pytest.MonkeyPatch
) -> None:
    """Close cannot return successfully while an already-registered launch can still start later."""
    entered = threading.Event()
    release = threading.Event()
    closed = threading.Event()
    instances: list[ManagedProcess] = []
    errors: list[BaseException] = []

    class DelayedProcess(ManagedProcess):
        def __init__(self, argv: Sequence[str], **kwargs: Any) -> None:
            entered.set()
            assert release.wait(5)
            super().__init__(argv, **kwargs)
            instances.append(self)

    monkeypatch.setattr(process_module, "ManagedProcess", DelayedProcess)
    supervisor = ProcessSupervisor()

    def run() -> None:
        try:
            supervisor.run(python_command("import time;time.sleep(20)"), env=environment)
        except BaseException as exc:  # noqa: BLE001
            errors.append(exc)

    def close() -> None:
        supervisor.close()
        closed.set()

    worker = threading.Thread(target=run)
    closer = threading.Thread(target=close)
    worker.start()
    try:
        assert entered.wait(5)
        closer.start()
        assert not closed.wait(0.05)
        release.set()
        worker.join(5)
        closer.join(5)
        assert closed.is_set()
        assert not worker.is_alive()
        assert len(errors) == 1
        assert isinstance(errors[0], SandboxError)
        assert instances[0].process.poll() is not None
    finally:
        release.set()
        worker.join(5)
        if closer.ident is not None:
            closer.join(5)
        supervisor.close()


def test_callback_cannot_reenter_supervisor(environment: dict[str, str]) -> None:
    """Reentry cannot replace the process handle used by cancellation."""
    supervisor = ProcessSupervisor()
    command = python_command("import time;print('ready');time.sleep(20)")
    nested_command = python_command("pass")

    def reenter(_: str) -> None:
        supervisor.run(nested_command, env=environment)

    with pytest.raises(SandboxError, match="already running"):
        supervisor.run(command, env=environment, on_stdout=reenter)


def test_managed_process_supports_persistent_stdin_protocol(environment: dict[str, str]) -> None:
    """Transport users can keep stdin open for multiple messages before explicit EOF."""
    command = python_command(
        "import sys\nfor line in sys.stdin.buffer:\n sys.stdout.buffer.write(line.upper());sys.stdout.buffer.flush()"
    )
    with ManagedProcess(command, env=environment, stdin_pipe=True) as managed:
        assert managed.process.stdin is not None
        assert managed.process.stdout is not None
        for request in (b"one\n", b"two\n"):
            managed.process.stdin.write(request)
            assert managed.process.stdout.readline() == request.upper()
        managed.process.stdin.close()
        assert managed.process.wait(timeout=5) == 0


def test_resolver_ignores_relative_search_directories(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A launcher name cannot resolve through an implicit or explicit current-directory PATH entry."""
    name = "untrusted.exe" if sys.platform == "win32" else "untrusted"
    candidate = tmp_path / name
    candidate.write_text("untrusted", encoding="utf-8")
    candidate.chmod(0o700)
    monkeypatch.chdir(tmp_path)
    search_path = os.pathsep.join(["", ".", "relative"])
    with pytest.raises(MissingDependencyError):
        resolve_executable(name, search_path=search_path)
    assert resolve_executable(sys.executable).is_absolute()


@pytest.mark.parametrize("extension", [".bat", ".cmd"])
def test_resolver_rejects_batch_files(tmp_path: Path, extension: str) -> None:
    """shell=False must not be bypassed by Windows' implicit batch-file handling."""
    candidate = tmp_path / ("launcher" + extension)
    candidate.write_text("echo must-not-run", encoding="utf-8")
    candidate.chmod(0o700)
    with pytest.raises(ValueError, match="Batch files"):
        resolve_executable(candidate)


@pytest.mark.parametrize("timeout", [0, -1, float("nan"), float("inf")])
def test_invalid_timeout_rejected_before_process_creation(environment: dict[str, str], timeout: float) -> None:
    """Invalid deadlines never implicitly become unlimited execution."""
    supervisor = ProcessSupervisor()
    command = python_command("pass")
    with pytest.raises(ValueError, match="timeout"):
        supervisor.run(command, env=environment, timeout=timeout)


def test_job_assignment_failure_kills_launcher_before_return(
    environment: dict[str, str], monkeypatch: pytest.MonkeyPatch
) -> None:
    """A Windows job assignment error must fail closed rather than run without descendant cleanup."""
    calls: list[str] = []
    processes: list[subprocess.Popen[bytes]] = []
    real_popen = subprocess.Popen

    class RejectedJob:
        def assign_and_resume(self, process: subprocess.Popen[bytes]) -> None:
            calls.append("assign")
            msg = "job assignment rejected"
            raise OSError(msg)

        def close(self) -> None:
            calls.append("job-close")

    def spawn(argv: Sequence[str], **kwargs: Any) -> subprocess.Popen[bytes]:
        assert kwargs["creationflags"] & process_module._CREATE_SUSPENDED
        # Linux exercises only failure cleanup with a fixed sleeping command.
        kwargs["creationflags"] = 0
        kwargs["start_new_session"] = sys.platform != "win32"
        result = real_popen(argv, **kwargs)
        processes.append(result)
        return result

    monkeypatch.setattr(process_module, "_WINDOWS", True)
    monkeypatch.setattr(process_module, "_WindowsJob", RejectedJob)
    monkeypatch.setattr(process_module.subprocess, "Popen", spawn)
    command = python_command("import time;time.sleep(20)")
    with pytest.raises(OSError, match="job assignment rejected"):
        ManagedProcess(command, env=environment)
    assert calls == ["assign", "job-close"]
    assert processes[0].poll() is not None


def test_job_limits_are_set_before_assignment_and_resume(monkeypatch: pytest.MonkeyPatch) -> None:
    """The Windows API sequence never resumes a launcher before kill-on-close is active."""
    calls: list[str] = []

    def limits(handle: int, kind: int, info: Any, size: int) -> bool:
        assert handle == 123
        assert kind == 9
        assert info._obj.basic.limit_flags == 0x2000
        assert size > 0
        calls.append("limits")
        return True

    def assign(job: int, process: int) -> bool:
        assert job == 123
        assert process == 456
        calls.append("assign")
        return True

    kernel = SimpleNamespace(
        CreateJobObjectW=Mock(return_value=123),
        SetInformationJobObject=Mock(side_effect=limits),
        AssignProcessToJobObject=Mock(side_effect=assign),
        CloseHandle=Mock(side_effect=lambda _: calls.append("close") or True),
    )
    ntdll = SimpleNamespace(NtResumeProcess=Mock(side_effect=lambda _: calls.append("resume") or 0))
    monkeypatch.setattr(process_module, "_windows_library", lambda name: kernel if name == "kernel32" else ntdll)
    job = process_module._WindowsJob()
    job.assign_and_resume(SimpleNamespace(_handle=456))
    job.close()
    job.close()
    assert calls == ["limits", "assign", "resume", "close"]


@pytest.mark.integration
@pytest.mark.skipif(sys.platform != "win32", reason="Requires native Windows job objects")
def test_native_job_kills_descendant_in_new_process_group(environment: dict[str, str], tmp_path: Path) -> None:
    """A detached process group remains in the assigned job and cannot outlive command cleanup."""
    marker = tmp_path / "escaped"
    child = f"import pathlib,time;time.sleep(1);pathlib.Path({str(marker)!r}).write_text('escaped')"
    parent = (
        f"import subprocess,sys;subprocess.Popen([sys.executable,'-c',{child!r}],"
        "stdin=subprocess.DEVNULL,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL,"
        "creationflags=subprocess.CREATE_NEW_PROCESS_GROUP);print('spawned')"
    )
    assert ProcessSupervisor().run(python_command(parent), env=environment).stdout == "spawned" + os.linesep
    time.sleep(1.2)
    assert not marker.exists()


@pytest.mark.parametrize("value", ["", "bad\0name", "relative/launcher"])
def test_resolver_rejects_invalid_or_relative_explicit_paths(value: str) -> None:
    """Malformed explicit paths cannot become a current-directory launcher lookup."""
    with pytest.raises(ValueError, match="must be"):
        resolve_executable(value)


@pytest.mark.parametrize("first_is_directory", [False, True])
def test_windows_bare_name_search_skips_missing_and_nonfile_candidates(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, first_is_directory: bool
) -> None:
    """Only regular executable candidates are accepted from absolute trusted PATH entries."""
    monkeypatch.setattr(process_module, "_WINDOWS", True)
    if first_is_directory:
        (tmp_path / "launcher.exe").mkdir()
    executable = tmp_path / "launcher.com"
    executable.write_bytes(b"trusted resolver fixture; never executed")
    executable.chmod(0o700)
    assert resolve_executable("launcher", search_path=str(tmp_path)) == executable
    with pytest.raises(MissingDependencyError):
        resolve_executable(tmp_path)


@pytest.mark.parametrize(
    ("arguments", "environment"),
    [([], {}), (["bad\0argument"], {}), ([sys.executable], {"": "value"}), ([sys.executable], {"KEY": "bad\0value"})],
)
def test_invalid_launch_contract_never_creates_a_process(
    monkeypatch: pytest.MonkeyPatch, arguments: list[str], environment: dict[str, str]
) -> None:
    """Invalid argv or environment entries fail before native creation and job registration."""
    launch = Mock()
    monkeypatch.setattr(process_module.subprocess, "Popen", launch)
    with pytest.raises(ValueError, match="Launcher"):
        ManagedProcess(arguments, env=environment)
    launch.assert_not_called()


def test_invalid_supervisor_limits_and_input_fail_before_launch(environment: dict[str, str]) -> None:
    """Zero capture budgets and text input do not silently weaken the transport contract."""
    with pytest.raises(ValueError, match="positive"):
        ProcessSupervisor(max_output_bytes=0)
    supervisor = ProcessSupervisor()
    command = python_command("pass")
    with pytest.raises(TypeError, match="input must be bytes"):
        supervisor.run(command, env=environment, input="text")  # type: ignore[arg-type]
    assert supervisor._state is None


@pytest.fixture
def native_job_api(monkeypatch: pytest.MonkeyPatch) -> tuple[Any, Any]:
    """Inject native API outcomes while retaining the production job ownership logic."""
    kernel = SimpleNamespace(
        CreateJobObjectW=Mock(return_value=123),
        SetInformationJobObject=Mock(return_value=True),
        AssignProcessToJobObject=Mock(return_value=True),
        CloseHandle=Mock(return_value=True),
    )
    ntdll = SimpleNamespace(NtResumeProcess=Mock(return_value=0))
    monkeypatch.setattr(process_module, "_windows_library", lambda name: kernel if name == "kernel32" else ntdll)
    monkeypatch.setattr(process_module, "_windows_error", lambda: OSError("native job operation failed"))
    return kernel, ntdll


@pytest.mark.parametrize("stage", ["allocate", "limits"])
def test_native_job_creation_failure_releases_any_allocated_handle(native_job_api: tuple[Any, Any], stage: str) -> None:
    """A failed containment setup cannot leak the job or leave an uncontained launcher running."""
    kernel, _ = native_job_api
    if stage == "allocate":
        kernel.CreateJobObjectW.return_value = None
    else:
        kernel.SetInformationJobObject.return_value = False
    with pytest.raises(OSError, match="native job operation failed"):
        process_module._WindowsJob()
    assert kernel.CloseHandle.call_count == int(stage == "limits")
    kernel.AssignProcessToJobObject.assert_not_called()


@pytest.mark.parametrize("stage", ["assign", "resume"])
def test_native_job_never_ignores_assignment_or_resume_failures(native_job_api: tuple[Any, Any], stage: str) -> None:
    """Both native operations must succeed before the caller can consider its child running safely."""
    kernel, ntdll = native_job_api
    job = process_module._WindowsJob()
    process = SimpleNamespace(_handle=456)
    if stage == "assign":
        kernel.AssignProcessToJobObject.return_value = False
    else:
        ntdll.NtResumeProcess.return_value = -1
    try:
        with pytest.raises(OSError, match="native job operation failed|Could not resume"):
            job.assign_and_resume(process)
        assert ntdll.NtResumeProcess.call_count == int(stage == "resume")
    finally:
        job.close()
    kernel.CloseHandle.assert_called_once_with(123)


def test_native_job_close_failure_retains_handle_for_retry(native_job_api: tuple[Any, Any]) -> None:
    """Failed cleanup remains observable and does not discard the only handle able to kill descendants."""
    kernel, _ = native_job_api
    job = process_module._WindowsJob()
    kernel.CloseHandle.side_effect = [False, True]
    with pytest.raises(OSError, match="native job operation failed"):
        job.close()
    assert job._handle == 123
    job.close()
    assert job._handle is None


def test_native_library_is_not_loaded_on_non_windows_hosts() -> None:
    """Importability on Linux must not become a pretend Windows API implementation."""
    if sys.platform == "win32":
        pytest.skip("Windows native loading is covered by the actual job integration test")
    with pytest.raises(OSError, match="require Windows"):
        process_module._windows_library("kernel32")
    assert "require Windows" in str(process_module._windows_error())


@pytest.mark.skipif(sys.platform != "win32", reason="Requires the real Windows thread-local error API")
def test_native_error_preserves_the_failed_api_error_code() -> None:
    """Failure diagnostics must preserve the original native error rather than report a later cleanup call."""
    previous = ctypes.get_last_error()
    try:
        ctypes.set_last_error(5)  # ERROR_ACCESS_DENIED
        error = process_module._windows_error()
    finally:
        ctypes.set_last_error(previous)
    assert error.winerror == 5


@pytest.mark.parametrize("native_job", [False, True])
def test_failed_process_creation_releases_job_and_allows_reuse(
    environment: dict[str, str], monkeypatch: pytest.MonkeyPatch, native_job: bool
) -> None:
    """Failure before Popen returns has no child to reap but still owns the allocated job."""
    supervisor = ProcessSupervisor()
    command = python_command("pass")
    job = Mock()
    with monkeypatch.context() as patch:
        patch.setattr(process_module, "_WINDOWS", native_job)
        patch.setattr(process_module, "_WindowsJob", lambda: job)
        patch.setattr(process_module.subprocess, "Popen", Mock(side_effect=OSError("creation rejected")))
        with pytest.raises(OSError, match="creation rejected"):
            supervisor.run(command, env=environment)
    assert job.close.call_count == int(native_job)
    assert supervisor.run(command, env=environment).exit_code == 0


def test_termination_retries_reaping_without_retaining_input_handles(
    environment: dict[str, str], monkeypatch: pytest.MonkeyPatch
) -> None:
    """A delayed initial wait still kills/reaps the leader before closing a potentially busy stdin."""
    command = python_command("import time;time.sleep(20)")
    managed = ManagedProcess(command, env=environment, stdin_pipe=True)
    original_wait = managed.process.wait
    wait = Mock(side_effect=[subprocess.TimeoutExpired(command, 2), 0])
    try:
        monkeypatch.setattr(managed.process, "wait", wait)
        managed.terminate()
        assert wait.call_count == 2
        assert managed.process.stdin is not None
        assert managed.process.stdin.closed
    finally:
        monkeypatch.setattr(managed.process, "wait", original_wait)
        original_wait(timeout=3)
        managed.close()


@pytest.mark.parametrize(("stream", "cancelled"), [("stdout", False), ("stdin", False), ("stdin", True)])
def test_pipe_io_failures_cancel_and_reap_the_launcher(
    environment: dict[str, str], monkeypatch: pytest.MonkeyPatch, stream: str, cancelled: bool
) -> None:
    """Reader and writer failures reach the caller without leaving processes or pipe workers alive."""
    created: list[ManagedProcess] = []
    descriptors: set[int] = set()
    before = {thread.ident for thread in threading.enumerate()}
    original_io = os.read if stream == "stdout" else os.write

    class ObservedProcess(ManagedProcess):
        def __init__(self, argv: Sequence[str], **kwargs: Any) -> None:
            super().__init__(argv, **kwargs)
            created.append(self)
            descriptors.add(getattr(self.process, stream).fileno())

    def fail_io(descriptor: int, data: Any) -> Any:
        if descriptor in descriptors:
            if cancelled:
                assert supervisor._state is not None
                supervisor._state.cancelled.set()
            msg = "pipe I/O failed"
            raise OSError(msg)
        return original_io(descriptor, data)

    supervisor = ProcessSupervisor()
    command = python_command("import time;time.sleep(20)")
    monkeypatch.setattr(process_module, "ManagedProcess", ObservedProcess)
    monkeypatch.setattr(process_module.os, "read" if stream == "stdout" else "write", fail_io)
    expected_error = SandboxError if cancelled else OSError
    expected_message = "cancelled" if cancelled else "pipe I/O failed"
    with pytest.raises(expected_error, match=expected_message):
        supervisor.run(command, env=environment, input=b"request")
    assert created[0].process.poll() is not None
    assert {thread.ident for thread in threading.enumerate()} == before


def test_close_reports_stalled_creation_and_still_cancels_the_late_process(
    environment: dict[str, str], monkeypatch: pytest.MonkeyPatch
) -> None:
    """A creation-wait timeout cannot falsely report cleanup success or permit the delayed launch to survive."""
    entered = threading.Event()
    release = threading.Event()
    created: list[ManagedProcess] = []
    errors: list[BaseException] = []

    class DelayedProcess(ManagedProcess):
        def __init__(self, argv: Sequence[str], **kwargs: Any) -> None:
            entered.set()
            assert release.wait(3)
            super().__init__(argv, **kwargs)
            created.append(self)

    monkeypatch.setattr(process_module, "ManagedProcess", DelayedProcess)
    supervisor = ProcessSupervisor()
    command = python_command("import time;time.sleep(20)")

    def run() -> None:
        try:
            supervisor.run(command, env=environment)
        except BaseException as exc:  # noqa: BLE001 # Assert the asynchronous cancellation outcome below.
            errors.append(exc)

    thread = threading.Thread(target=run, daemon=True)
    thread.start()
    try:
        assert entered.wait(3)
        assert supervisor._state is not None
        # Simulate expiry of the creation wait without sleeping for five seconds.
        with monkeypatch.context() as patch:
            creation_wait = Mock(return_value=False)
            patch.setattr(supervisor._state.created, "wait", creation_wait)
            with pytest.raises(SandboxError, match="creation did not finish"):
                supervisor.close()
            creation_wait.assert_called_once_with(timeout=5)
    finally:
        release.set()
        thread.join(3)
        supervisor.close()
    assert not thread.is_alive()
    assert created[0].process.poll() is not None
    assert len(errors) == 1
    assert isinstance(errors[0], SandboxError)
    with pytest.raises(SandboxError, match="closed"):
        supervisor.run(command, env=environment)


def test_output_backpressure_remains_cancellable(environment: dict[str, str], monkeypatch: pytest.MonkeyPatch) -> None:
    """A full bounded output queue never traps the publisher during cancellation."""
    full = threading.Event()
    callback_entered = threading.Event()
    release = threading.Event()
    errors: list[BaseException] = []

    class ObservedQueue(queue.Queue[Any]):
        def put(self, *args: Any, **kwargs: Any) -> None:
            try:
                super().put(*args, **kwargs)
            except queue.Full:
                full.set()
                raise

    monkeypatch.setattr(process_module.queue, "Queue", ObservedQueue)
    supervisor = ProcessSupervisor()
    command = python_command("import os,time;os.write(1,b'x'*2000000);time.sleep(20)")

    def callback(_: str) -> None:
        callback_entered.set()
        assert release.wait(5)

    def run() -> None:
        try:
            supervisor.run(command, env=environment, on_stdout=callback)
        except BaseException as exc:  # noqa: BLE001 # Assert the asynchronous cancellation outcome below.
            errors.append(exc)

    thread = threading.Thread(target=run, daemon=True)
    thread.start()
    try:
        assert callback_entered.wait(3)
        assert full.wait(3)
        supervisor.cancel()
    finally:
        release.set()
        supervisor.close()
        thread.join(3)
    assert not thread.is_alive()
    assert len(errors) == 1
    assert isinstance(errors[0], SandboxError)


def test_partial_thread_startup_failure_cleans_already_started_readers(
    environment: dict[str, str], monkeypatch: pytest.MonkeyPatch
) -> None:
    """Thread resource exhaustion after one pipe reader starts must still reap the process and reader."""
    original_start = threading.Thread.start
    before = {thread.ident for thread in threading.enumerate()}
    starts = 0

    def start(thread: threading.Thread) -> None:
        nonlocal starts
        starts += 1
        if starts == 2:
            msg = "no reader threads available"
            raise RuntimeError(msg)
        original_start(thread)

    supervisor = ProcessSupervisor()
    command = python_command("import time;time.sleep(20)")
    monkeypatch.setattr(threading.Thread, "start", start)
    with pytest.raises(RuntimeError, match="no reader threads"):
        supervisor.run(command, env=environment)
    assert {thread.ident for thread in threading.enumerate()} == before
