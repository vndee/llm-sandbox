"""Supervisor unit tests with trusted commands; the test driver provides NO isolation."""

from __future__ import annotations

import base64
import gc
import io
import os
import re
import shlex
import shutil
import subprocess
import sys
import tarfile
import threading
import time
from contextlib import contextmanager
from dataclasses import replace
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pytest

from llm_sandbox.exceptions import (
    CommandEmptyError,
    CommandFailedError,
    LibraryInstallationNotSupportedError,
    NotOpenSessionError,
    SandboxError,
    SandboxTimeoutError,
    SecurityError,
    SecurityViolationError,
)
from llm_sandbox.language_handlers.python_handler import PythonHandler
from llm_sandbox.local.config import LocalSandboxConfig
from llm_sandbox.local.files import safe_remove_tree
from llm_sandbox.local.session import LocalProcessSession
from llm_sandbox.security import RestrictedModule, SecurityIssueSeverity, SecurityPolicy
from llm_sandbox.session import ArtifactSandboxSession

if TYPE_CHECKING:
    from collections.abc import Iterator

    from llm_sandbox.data import ConsoleOutput


class TrustedTestSession(LocalProcessSession):
    """Exercise only process supervision, using fixed trusted test inputs on the host."""

    @contextmanager
    def _launch_command(self, command: str, workdir: str) -> Iterator[tuple[list[str], tuple[int, ...]]]:
        """Translate fixture paths; this intentionally is not a production sandbox."""
        pattern = rf"(?<![A-Za-z0-9_./-]){re.escape(self.config.workdir)}(?=/|[^A-Za-z0-9_.-]|$)"
        command = re.sub(pattern, str(self.workspace), command)
        command = command.replace("/tmp/sandbox_plots", str(self.tempdir / "sandbox_plots"))
        environment = [f"{key}={value}" for key, value in self._environment().items()]
        yield ["/usr/bin/env", "-i", *environment, "/bin/sh", "-c", command], ()


@pytest.fixture
def session() -> Iterator[TrustedTestSession]:
    """Open the trusted process test driver without language environment setup."""
    with TrustedTestSession(skip_environment_setup=True) as sandbox:
        yield sandbox


def test_workspace_lifetime_and_regular_file_copy(tmp_path: Path) -> None:
    """Files persist across commands and the private workspace disappears on close."""
    source = tmp_path / "input"
    source.write_text("persistent", encoding="utf-8")
    with TrustedTestSession(skip_environment_setup=True) as sandbox:
        workspace = sandbox.workspace
        tempdir = sandbox.tempdir
        assert workspace.stat().st_mode & 0o777 == 0o700
        assert tempdir.stat().st_mode & 0o777 == 0o700
        assert workspace.parent == tempdir.parent
        assert workspace != tempdir
        sandbox.copy_to_runtime(str(source), "/sandbox/nested/data")
        sandbox.open()
        assert sandbox.workspace == workspace
        assert sandbox.execute_command("cat /sandbox/nested/data").stdout == "persistent"
        destination = tmp_path / "copied"
        sandbox.copy_from_runtime("nested/data", str(destination))
        assert destination.read_text(encoding="utf-8") == "persistent"
    assert not workspace.exists()
    assert not tempdir.exists()
    assert not workspace.parent.exists()
    sandbox.close()


@pytest.mark.parametrize("cleanup", ["close", "finalizer"])
def test_symlinked_temp_root_supports_session_lifecycle(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, cleanup: str
) -> None:
    """Trusted temp ancestors are resolved once for setup, execution, and both cleanup paths."""
    real_temp = tmp_path / "real-temp"
    real_temp.mkdir()
    temp_alias = tmp_path / "temp-alias"
    temp_alias.symlink_to(real_temp, target_is_directory=True)
    monkeypatch.setenv("TMPDIR", str(temp_alias))
    monkeypatch.setattr("tempfile.tempdir", None)

    sandbox = TrustedTestSession(skip_environment_setup=True)
    sandbox.open()
    private_root = sandbox.workspace.parent
    assert private_root.parent == real_temp.resolve()
    assert sandbox._workspace_temp is not None
    assert sandbox._workspace_temp.name == str(private_root)
    assert sandbox.tempdir.parent == private_root
    assert sandbox.run("print('canonical root')").stdout == "canonical root\n"
    # Cleanup must retain the canonical root even if the original alias disappears.
    temp_alias.unlink()

    if cleanup == "finalizer":
        sandbox = None
        gc.collect()
    else:
        sandbox.close()

    assert not private_root.exists()
    assert list(real_temp.iterdir()) == []


def _make_deep_workspace_tree(workspace: Path, outside: Path) -> None:
    """Build beyond PATH_MAX without retaining one descriptor per directory level."""
    directory = os.open(workspace, os.O_RDONLY | os.O_DIRECTORY)
    try:
        for _ in range(160):
            name = "nested_workspace_directory_000000"
            os.mkdir(name, dir_fd=directory)
            child = os.open(name, os.O_RDONLY | os.O_DIRECTORY, dir_fd=directory)
            os.close(directory)
            directory = child
        os.symlink(outside, "outside-link", dir_fd=directory)
        os.mkdir("leaf", dir_fd=directory)
    finally:
        os.close(directory)
    (workspace / "outside-link").symlink_to(outside, target_is_directory=True)


def _exercise_deep_workspace_cleanup(destination: str, mode: str) -> None:
    """Run constrained cleanup in an expendable process, never changing pytest's limits."""
    import resource  # This worker is Linux-only; the test module remains importable elsewhere.

    outside = Path(destination) / "outside"
    outside.mkdir()
    marker = outside / "keep"
    marker.write_text("outside data", encoding="utf-8")
    sandbox = TrustedTestSession(skip_environment_setup=True, session_timeout=60 if mode == "timer" else None)
    sandbox.open()
    private_root = sandbox.workspace.parent
    descriptor_limit = resource.getrlimit(resource.RLIMIT_NOFILE)
    recursion_limit = sys.getrecursionlimit()
    try:
        _make_deep_workspace_tree(sandbox.workspace, outside)
        resource.setrlimit(resource.RLIMIT_NOFILE, (64, descriptor_limit[1]))
        sys.setrecursionlimit(100)
        if mode == "execution-timeout":
            with pytest.raises(SandboxTimeoutError):
                sandbox.run("import time; time.sleep(20)", timeout=0.1)
        elif mode == "timer":
            timer = sandbox._session_timer
            assert timer is not None
            timer.function()
        elif mode == "finalizer":
            sandbox = None
            gc.collect()
        else:
            sandbox.close()
        assert not private_root.exists(), "Cleanup retained the session's private directory"
        assert marker.read_text(encoding="utf-8") == "outside data"
        if sandbox is not None:
            assert not sandbox.is_open
            assert sandbox._active_process is None
            sandbox.open()
            assert sandbox.run("print('reopened')").stdout == "reopened\n"
            sandbox.close()
    finally:
        # Independently clean up failed regressions so pytest never inherits a
        # deep fixture. Restored limits make the standard cleanup sufficient.
        resource.setrlimit(resource.RLIMIT_NOFILE, descriptor_limit)
        sys.setrecursionlimit(recursion_limit)
        if private_root.exists():
            shutil.rmtree(private_root)
        if sandbox is not None:
            sandbox.close()


@pytest.mark.skipif(sys.platform != "linux", reason="Requires Linux directory descriptors and resource limits")
@pytest.mark.parametrize("mode", ["close", "execution-timeout", "timer", "finalizer"])
def test_deep_workspace_cleanup_under_low_descriptor_and_recursion_limits(tmp_path: Path, mode: str) -> None:
    """All cleanup paths remove deep state, preserve external targets, and allow reuse."""
    worker = (
        "import sys; sys.path.insert(0, sys.argv[1]); "
        "from test_local_session import _exercise_deep_workspace_cleanup; "
        "_exercise_deep_workspace_cleanup(sys.argv[2], sys.argv[3])"
    )
    result = subprocess.run(  # noqa: S603 # Only the trusted test helper above runs in this subprocess.
        [sys.executable, "-c", worker, str(Path(__file__).parent), str(tmp_path), mode],
        capture_output=True,
        text=True,
        check=False,
        timeout=30,
    )
    assert result.returncode == 0, result.stderr
    assert (tmp_path / "outside/keep").read_text(encoding="utf-8") == "outside data"


def test_failed_workspace_cleanup_retains_owner_for_retry_and_reopen(monkeypatch: pytest.MonkeyPatch) -> None:
    """A transient cleanup error preserves ownership until reopening successfully retries removal."""
    with TrustedTestSession(skip_environment_setup=True) as sandbox:
        owner = sandbox._workspace_temp
        assert owner is not None
        previous_root = sandbox.workspace.parent
        failed = False

        def fail_once(path: Path) -> None:
            nonlocal failed
            if not failed:
                failed = True
                msg = "temporary cleanup failure"
                raise OSError(msg)
            safe_remove_tree(path)

        monkeypatch.setattr("llm_sandbox.local.session.safe_remove_tree", fail_once)
        with pytest.raises(OSError, match="temporary cleanup failure"):
            sandbox.close()
        assert sandbox._workspace_temp is owner
        assert owner._finalizer.alive
        assert previous_root.exists()
        sandbox.open()
        assert not previous_root.exists()
        assert not owner._finalizer.alive
        assert sandbox._workspace_temp is not owner
        assert sandbox.run("print('reopened after retry')").stdout == "reopened after retry\n"


def test_explicit_environment_does_not_inherit_secrets(monkeypatch: pytest.MonkeyPatch) -> None:
    """Only fixed defaults and explicitly supplied values reach the child."""
    monkeypatch.setenv("LOCAL_SANDBOX_TEST_SECRET", "hidden")
    with TrustedTestSession(
        sandbox_config=LocalSandboxConfig(env={"ALLOWED": "yes"}),
        env_vars={"EXTRA": "also"},
        skip_environment_setup=True,
    ) as sandbox:
        output = sandbox.execute_command('printf "%s|%s|%s" "$LOCAL_SANDBOX_TEST_SECRET" "$ALLOWED" "$EXTRA"')
        assert output.stdout == "|yes|also"
        assert sandbox._environment()["HOME"] == "/sandbox/.home"


def test_replacing_config_revokes_environment_grants_and_preserves_explicit_overrides() -> None:
    """Each command uses the current config, while env_vars remains a caller-input snapshot."""
    explicit = {"OVERRIDE": "explicit"}
    with TrustedTestSession(
        sandbox_config=LocalSandboxConfig(env={"REMOVED_SECRET": "private", "UPDATED": "old", "OVERRIDE": "old"}),
        env_vars=explicit,
        skip_environment_setup=True,
    ) as sandbox:
        assert sandbox.execute_command('printf "%s" "$REMOVED_SECRET"').stdout == "private"
        explicit["OVERRIDE"] = "changed-after-construction"
        sandbox.sandbox_config = LocalSandboxConfig(env={"UPDATED": "new", "ADDED": "added", "OVERRIDE": "config"})
        output = sandbox.execute_command(
            'printf "%s|%s|%s|%s" "${REMOVED_SECRET-unset}" "$UPDATED" "$ADDED" "$OVERRIDE"'
        )
        assert output.stdout == "unset|new|added|explicit"


def test_payload_loader_environment_never_reaches_host_launcher(monkeypatch: pytest.MonkeyPatch) -> None:
    """Do not apply payload loader hooks until a real driver has established isolation."""

    class NoopDriver(TrustedTestSession):
        @contextmanager
        def _launch_command(self, command: str, workdir: str) -> Iterator[tuple[list[str], tuple[int, ...]]]:
            # Deliberately ignore payload settings so the regression never loads a host library.
            yield ["/bin/true"], ()

    received: list[dict[str, str]] = []
    original_popen = subprocess.Popen

    def inspect_popen(argv: list[str], **kwargs: Any) -> subprocess.Popen[bytes]:
        received.append(kwargs["env"])
        return original_popen(argv, **kwargs)  # Fixed no-op driver above.

    monkeypatch.setattr("llm_sandbox.local.session.subprocess.Popen", inspect_popen)
    payload = {
        "LD_PRELOAD": "/never-load-this.so",
        "LD_LIBRARY_PATH": "/never-search-this",
        "PYTHONPATH": "/payload-python",
        "PATH": "/payload-bin",
        "HOME": "/payload-home",
    }
    with NoopDriver(skip_environment_setup=True, env_vars=payload) as sandbox:
        assert all(sandbox._environment()[key] == value for key, value in payload.items())
        sandbox.execute_command("ignored")
    assert len(received) == 2
    assert all(environment == {"PATH": "/usr/local/bin:/usr/bin:/bin", "LANG": "C.UTF-8"} for environment in received)


def test_run_uses_python3_when_setup_is_skipped(session: TrustedTestSession) -> None:
    """The shared run path invokes the configured interpreter instead of a python alias."""
    output = session.run("print('native supervisor')")
    assert output.exit_code == 0
    assert output.stdout == "native supervisor\n"
    assert session.python_executable_path == "/usr/bin/python3"


def test_replacing_config_before_open_selects_updated_python() -> None:
    """The constructor's previous interpreter selection cannot survive a config replacement."""
    sandbox = TrustedTestSession(
        sandbox_config=LocalSandboxConfig(python_executable="/nonexistent/old-python"),
        skip_environment_setup=True,
    )
    sandbox.sandbox_config = LocalSandboxConfig(python_executable=sys.executable)
    with sandbox:
        output = sandbox.run("print('updated interpreter')")
        assert output.exit_code == 0
        assert output.stdout == "updated interpreter\n"


def test_python_config_replacement_preserves_non_python_handler() -> None:
    """Changing the Python option must not replace another language's execution commands."""
    sandbox = TrustedTestSession(lang="javascript", skip_environment_setup=True)
    sandbox.language_handler.config.execution_commands = ["/bin/sh {file}"]
    sandbox.sandbox_config = LocalSandboxConfig(python_executable="/nonexistent/python")
    with sandbox:
        assert sandbox.run("printf unchanged-handler").stdout == "unchanged-handler"


def test_python_environment_uses_private_interpreter_and_cache() -> None:
    """Offline setup creates an executable venv and writable cache inside the session."""
    with TrustedTestSession(sandbox_config=LocalSandboxConfig(python_executable=sys.executable)) as sandbox:
        output = sandbox.run("import sys; print(sys.prefix)")
        assert output.exit_code == 0
        assert output.stdout.strip() == str(sandbox.workspace / ".sandbox-venv")
        assert (sandbox.workspace / ".sandbox-pip-cache").is_dir()
        assert sandbox.python_executable_path.startswith(sandbox.config.workdir + "/")
        sandbox._ensure_directory_exists("nested/directory")
        assert (sandbox.workspace / "nested/directory").is_dir()


def test_stream_callbacks_receive_correct_utf8_and_both_channels(session: TrustedTestSession) -> None:
    """Incremental decoders handle a multibyte character split across pipe writes."""
    script = "import os,time;os.write(1,b'\\xe2');time.sleep(.02);os.write(1,b'\\x82\\xac');os.write(2,b'err')"
    stdout: list[str] = []
    stderr: list[str] = []
    output = session.execute_command(
        f"/usr/bin/python3 -c {shlex.quote(script)}", on_stdout=stdout.append, on_stderr=stderr.append
    )
    assert output.stdout == "€"
    assert output.stderr == "err"
    assert "".join(stdout) == "€"
    assert "".join(stderr) == "err"


def test_timeout_kills_process_group_and_reaps_parent(session: TrustedTestSession) -> None:
    """Deadline handling leaves no active child and does not use worker threads."""
    before = {thread.ident for thread in threading.enumerate()}
    with pytest.raises(SandboxTimeoutError):
        session.run("import time; time.sleep(20)", timeout=0.1)
    assert session._active_process is None
    assert not session.is_open
    assert {thread.ident for thread in threading.enumerate()} == before


def test_explicit_run_timeout_overrides_shorter_default() -> None:
    """The command supervisor preserves an explicit run deadline instead of shortening it."""
    with TrustedTestSession(default_timeout=0.1, skip_environment_setup=True) as sandbox:
        output = sandbox.run("import time; time.sleep(.15); print('done')", timeout=2)
        assert output.stdout == "done\n"


def test_timeout_still_kills_child_after_both_output_pipes_close() -> None:
    """Closing stdout and stderr must not bypass the supervisor's final wait deadline."""
    script = "import os,time;os.close(1);os.close(2);time.sleep(20)"
    command = f"exec /usr/bin/python3 -c {shlex.quote(script)}"
    with TrustedTestSession(skip_environment_setup=True, execution_timeout=0.2) as sandbox:
        with pytest.raises(SandboxTimeoutError):
            sandbox.execute_command(command)
        assert sandbox._active_process is None
        assert sandbox.execute_command("printf recovered").stdout == "recovered"


def test_explicitly_disabled_deadline_runs_and_cleans_up() -> None:
    """An unlimited session still captures output and reaps each completed command."""
    with TrustedTestSession(default_timeout=None, skip_environment_setup=True) as sandbox:
        assert sandbox.run("print('unlimited')").stdout == "unlimited\n"
        assert sandbox._active_process is None


@pytest.mark.parametrize("timeout", [0, -1, float("inf"), float("nan")])
def test_internal_deadline_rejects_invalid_value_before_work(session: TrustedTestSession, timeout: float) -> None:
    """The shared deadline boundary cannot silently turn invalid timeouts into work."""
    invoked: list[bool] = []
    with pytest.raises(ValueError, match="timeout must be finite and positive"):
        session._execute_with_timeout(lambda: invoked.append(True), timeout=timeout)
    assert not invoked
    assert session.execute_command("printf valid").stdout == "valid"


def test_nested_deadline_cannot_extend_outer_timeout(session: TrustedTestSession) -> None:
    """An inner operation requesting more time must still honor its enclosing budget."""
    with pytest.raises(SandboxTimeoutError):
        session._execute_with_timeout(
            lambda: session._execute_with_timeout(lambda: session.execute_command("sleep 20"), timeout=20),
            timeout=0.1,
        )
    assert session._active_process is None
    assert session.execute_command("printf recovered").stdout == "recovered"


def test_callback_exception_cleans_up_process(session: TrustedTestSession) -> None:
    """User callback failures cannot leave command processes running."""

    def broken_callback(chunk: str) -> None:
        del chunk
        msg = "callback failed"
        raise RuntimeError(msg)

    with pytest.raises(RuntimeError, match="callback failed"):
        session.execute_command("printf ready; sleep 20", on_stdout=broken_callback)
    assert session._active_process is None
    assert session.execute_command("printf next").stdout == "next"


def test_invalid_output_encoding_kills_child_and_preserves_session(session: TrustedTestSession) -> None:
    """A decoder error must take the same cleanup path as process and callback failures."""
    script = "import os,time;os.write(1,b'\\xff');time.sleep(20)"
    command = f"/usr/bin/python3 -c {shlex.quote(script)}"
    with pytest.raises(UnicodeDecodeError):
        session.execute_command(command)
    assert session._active_process is None
    assert session.execute_command("printf recovered").stdout == "recovered"


def test_verbose_stream_logs_output_without_callbacks(monkeypatch: pytest.MonkeyPatch) -> None:
    """Streaming to the logger must preserve the complete returned output."""
    with TrustedTestSession(skip_environment_setup=True, verbose=True, stream=True) as sandbox:
        messages: list[str] = []
        monkeypatch.setattr(sandbox, "_log", messages.append)
        output = sandbox.execute_command("printf streamed")
        assert "".join(messages) == output.stdout == "streamed"


def test_output_limit_terminates_child() -> None:
    """A runaway output stream is stopped before unbounded accumulation."""
    with TrustedTestSession(
        sandbox_config=LocalSandboxConfig(max_output_bytes=1024), skip_environment_setup=True
    ) as sandbox:
        with pytest.raises(SandboxError, match="output exceeded"):
            sandbox.execute_command("/usr/bin/python3 -c 'print(\"x\" * 10000)' ")
        assert sandbox._active_process is None


def test_callback_cannot_copy_while_sandbox_is_running(session: TrustedTestSession, tmp_path: Path) -> None:
    """The reentrant operation lock does not allow callbacks to bypass copy serialization."""
    source = tmp_path / "source"
    source.write_text("data", encoding="utf-8")
    with pytest.raises(SandboxError, match="cannot be reentered"):
        session.execute_command("printf ready; sleep 20", on_stdout=lambda _: session.copy_to_runtime(str(source), "x"))
    assert session._active_process is None


def test_callback_cannot_start_another_process(session: TrustedTestSession) -> None:
    """Nested callbacks cannot replace the process handle needed to cancel the outer command."""
    with pytest.raises(SandboxError, match="cannot be reentered"):
        session.execute_command("printf ready; sleep 20", on_stdout=lambda _: session.execute_command("true"))
    assert session._active_process is None


def test_close_interrupts_execution_before_acquiring_operation_lock(session: TrustedTestSession) -> None:
    """Concurrent close can stop a command instead of waiting for its default timeout."""
    ready = threading.Event()
    outcomes: list[object] = []

    def execute() -> None:
        try:
            outcomes.append(session.execute_command("printf ready; sleep 20", on_stdout=lambda _: ready.set()))
        except BaseException as exc:  # noqa: BLE001
            outcomes.append(exc)

    worker = threading.Thread(target=execute)
    worker.start()
    assert ready.wait(3)
    start = time.monotonic()
    session.close()
    worker.join(3)
    assert time.monotonic() - start < 3
    assert not worker.is_alive()
    assert len(outcomes) == 1
    assert session._active_process is None


def test_old_timer_cleanup_does_not_delete_reopened_workspace(monkeypatch: pytest.MonkeyPatch) -> None:
    """A timer paused after cancellation cannot clean up a newer session generation."""
    sandbox = TrustedTestSession(skip_environment_setup=True, session_timeout=60)
    sandbox.open()
    timer = sandbox._session_timer
    assert timer is not None
    original_lock = sandbox._operation_lock
    paused = threading.Event()
    resume = threading.Event()
    ready = threading.Event()
    outcomes: list[object] = []

    class PausingOperationLock:
        def __enter__(self) -> None:
            if threading.current_thread() is cleanup:
                paused.set()
                assert resume.wait(3)
            original_lock.acquire()

        def __exit__(self, *args: object) -> None:
            original_lock.release()

    def execute() -> None:
        try:
            outcomes.append(sandbox.execute_command("printf ready; sleep 20", on_stdout=lambda _: ready.set()))
        except BaseException as exc:  # noqa: BLE001
            outcomes.append(exc)

    monkeypatch.setattr(sandbox, "_operation_lock", PausingOperationLock())
    worker = threading.Thread(target=execute)
    cleanup = threading.Thread(target=timer.function)
    try:
        worker.start()
        assert ready.wait(3)
        cleanup.start()
        assert paused.wait(3)
        worker.join(3)
        assert not worker.is_alive()
        assert len(outcomes) == 1
        previous_workspace = sandbox.workspace
        sandbox.open()
        assert not previous_workspace.exists()
        reopened_workspace = sandbox.workspace
        new_timer = sandbox._session_timer
        resume.set()
        cleanup.join(3)
        assert not cleanup.is_alive()
        assert sandbox.is_open
        assert reopened_workspace.exists()
        assert sandbox._session_timer is new_timer
        assert sandbox.execute_command("printf reopened").stdout == "reopened"
    finally:
        resume.set()
        if cleanup.ident is not None:
            cleanup.join(3)
        sandbox.close()
        worker.join(3)


def test_old_timer_delayed_before_close_cannot_cancel_new_command(monkeypatch: pytest.MonkeyPatch) -> None:
    """The timer captures its generation when scheduled, before its callback can be delayed."""
    sandbox = TrustedTestSession(skip_environment_setup=True, session_timeout=60)
    sandbox.open()
    timer = sandbox._session_timer
    assert timer is not None
    original_close = sandbox._close_generation
    paused = threading.Event()
    resume = threading.Event()
    command_ready = threading.Event()
    command_release = threading.Event()
    outcomes: list[object] = []

    def delayed_close(generation: int) -> None:
        if threading.current_thread() is cleanup:
            paused.set()
            assert resume.wait(3)
        original_close(generation)

    def callback(_: str) -> None:
        command_ready.set()
        assert command_release.wait(3)

    def execute() -> None:
        try:
            outcomes.append(sandbox.execute_command("printf ready", on_stdout=callback))
        except BaseException as exc:  # noqa: BLE001
            outcomes.append(exc)

    monkeypatch.setattr(sandbox, "_close_generation", delayed_close)
    cleanup = threading.Thread(target=timer.function)
    worker = threading.Thread(target=execute)
    try:
        cleanup.start()
        assert paused.wait(3)
        sandbox.close()
        sandbox.open()
        worker.start()
        assert command_ready.wait(3)
        resume.set()
        cleanup.join(3)
        assert not cleanup.is_alive()
        assert not sandbox._closing
        command_release.set()
        worker.join(3)
        assert not worker.is_alive()
        assert not isinstance(outcomes[0], BaseException)
    finally:
        resume.set()
        command_release.set()
        cleanup.join(3)
        sandbox.close()
        if worker.ident is not None:
            worker.join(3)


def test_concurrent_closers_cancel_execution_without_deadlock() -> None:
    """Multiple closers can reap one command and leave the session ready to reopen."""
    sandbox = TrustedTestSession(skip_environment_setup=True)
    sandbox.open()
    ready = threading.Event()
    start = threading.Barrier(3)
    errors: list[BaseException] = []

    def execute() -> None:
        try:
            sandbox.execute_command("printf ready; sleep 20", on_stdout=lambda _: ready.set())
        except BaseException as exc:  # noqa: BLE001
            errors.append(exc)

    def close() -> None:
        try:
            start.wait(3)
            sandbox.close()
        except BaseException as exc:  # noqa: BLE001
            errors.append(exc)

    worker = threading.Thread(target=execute)
    closers = [threading.Thread(target=close) for _ in range(2)]
    try:
        worker.start()
        assert ready.wait(3)
        for closer in closers:
            closer.start()
        start.wait(3)
        for closer in closers:
            closer.join(3)
            assert not closer.is_alive()
        worker.join(3)
        assert not worker.is_alive()
        assert not errors
        assert not sandbox.is_open
        assert sandbox._active_process is None
        sandbox.open()
        assert sandbox.execute_command("printf reopened").stdout == "reopened"
    finally:
        sandbox.close()
        worker.join(3)


def test_timer_cleanup_failure_is_reported_without_escaping_callback(monkeypatch: pytest.MonkeyPatch) -> None:
    """Background cleanup failures remain visible without an unhandled timer-thread error."""
    with TrustedTestSession(skip_environment_setup=True, session_timeout=60) as sandbox:
        timer = sandbox._session_timer
        assert timer is not None
        messages: list[tuple[str, str]] = []

        def fail_cleanup(_generation: int) -> None:
            msg = "cleanup failed"
            raise OSError(msg)

        with monkeypatch.context() as patch:
            patch.setattr(sandbox, "_close_generation", fail_cleanup)
            patch.setattr(sandbox, "_log", lambda message, level: messages.append((message, level)))
            timer.function()
        assert messages == [("Error during timeout cleanup: cleanup failed", "error")]


def test_copy_waits_until_command_has_finished(session: TrustedTestSession, tmp_path: Path) -> None:
    """A copy never races an active sandbox process touching the workspace."""
    source = tmp_path / "source"
    source.write_text("safe", encoding="utf-8")
    ready = threading.Event()
    release = threading.Event()
    copied = threading.Event()

    def block_callback(_: str) -> None:
        ready.set()
        assert release.wait(3)

    def copy() -> None:
        session.copy_to_runtime(str(source), "target")
        copied.set()

    command = threading.Thread(target=lambda: session.execute_command("printf ready", on_stdout=block_callback))
    command.start()
    assert ready.wait(3)
    copying = threading.Thread(target=copy)
    copying.start()
    assert not copied.wait(0.05)
    release.set()
    command.join(3)
    copying.join(3)
    assert copied.is_set()


def test_probe_failure_cleans_workspace() -> None:
    """A failed launch probe fails closed and removes temporary state."""

    class FailingProbe(TrustedTestSession):
        @contextmanager
        def _launch_command(self, command: str, workdir: str) -> Iterator[tuple[list[str], tuple[int, ...]]]:
            yield ["/bin/false"], ()

    sandbox = FailingProbe(skip_environment_setup=True)
    with pytest.raises(CommandFailedError):
        sandbox.open()
    assert not sandbox.is_open
    assert not sandbox.workspace.exists()


def test_setup_does_not_upgrade_pip_or_enable_network(monkeypatch: pytest.MonkeyPatch) -> None:
    """Python setup only creates a local venv and cache; it never downloads packages."""
    sandbox = TrustedTestSession()
    commands: list[str] = []
    monkeypatch.setattr(sandbox, "_require_success", commands.append)
    monkeypatch.setattr(sandbox, "_ensure_directory_exists", lambda _: None)
    sandbox.environment_setup()
    assert len(commands) == 1
    assert "-m venv --system-site-packages" in commands[0]
    assert sandbox.sandbox_config.network == "none"


def test_initial_libraries_are_not_silently_ignored() -> None:
    """Requests that cannot be satisfied during initialization fail explicitly."""
    sandbox = TrustedTestSession(skip_environment_setup=True, libraries=["requests"])
    with pytest.raises(LibraryInstallationNotSupportedError):
        sandbox.open()
    assert not sandbox.is_open


@pytest.mark.parametrize("options", [{"skip_environment_setup": True}, {"lang": "java"}])
def test_install_rejects_sessions_without_installation_support(options: dict) -> None:
    """Unsupported installations fail before attempting a package manager."""
    with TrustedTestSession(**options) as sandbox, pytest.raises(LibraryInstallationNotSupportedError):
        sandbox.install(["dependency"])


def test_install_requires_explicit_network_permission() -> None:
    """A package request cannot implicitly upgrade the default network policy."""
    with TrustedTestSession(lang="javascript") as sandbox:
        with pytest.raises(ValueError, match="needs network='host'"):
            sandbox.install(["dependency"])
        assert sandbox.sandbox_config.network == "none"


def test_install_checks_entire_restricted_list_before_running_any_package(monkeypatch: pytest.MonkeyPatch) -> None:
    """A forbidden package later in the request must stop earlier package side effects too."""
    policy = SecurityPolicy(
        restricted_modules=[
            RestrictedModule(name="forbidden", description="disallowed dependency", severity=SecurityIssueSeverity.HIGH)
        ]
    )
    with TrustedTestSession(
        lang="javascript", sandbox_config=LocalSandboxConfig(network="host"), security_policy=policy
    ) as sandbox:
        commands: list[str] = []
        monkeypatch.setattr(sandbox, "_require_success", commands.append)
        with pytest.raises(SecurityViolationError, match="forbidden"):
            sandbox.install(["allowed", "forbidden"])
        assert not commands


@pytest.mark.parametrize("policy", [None, SecurityPolicy()])
def test_install_stops_on_failure_and_reports_package_output(
    monkeypatch: pytest.MonkeyPatch, policy: SecurityPolicy | None
) -> None:
    """The first package error is surfaced; later package commands must not run."""
    with TrustedTestSession(
        lang="javascript", sandbox_config=LocalSandboxConfig(network="host"), security_policy=policy
    ) as sandbox:

        def installer(library: str, context: Any) -> str:
            assert context.workdir == "/sandbox"
            if library == "broken":
                return "printf package-failure >&2; exit 7"
            return f"printf '%s\\n' {shlex.quote(library)} >> /sandbox/installed"

        monkeypatch.setattr(sandbox.language_handler, "get_library_installation_command", installer)
        sandbox.install(["first", "second"])
        with pytest.raises(CommandFailedError, match="package-failure"):
            sandbox.install(["broken", "never-reached"])
        assert (sandbox.workspace / "installed").read_text(encoding="utf-8") == "first\nsecond\n"
        assert sandbox.execute_command("printf healthy").stdout == "healthy"


def test_failed_environment_setup_releases_workspace_and_timer(monkeypatch: pytest.MonkeyPatch) -> None:
    """Initialization failures after the probe cannot leave a live session or its files."""
    sandbox = TrustedTestSession(session_timeout=60)

    def fail_setup() -> None:
        sandbox.execute_command("printf setup-state > /sandbox/setup-state")
        msg = "setup failed"
        raise RuntimeError(msg)

    monkeypatch.setattr(sandbox, "environment_setup", fail_setup)
    with pytest.raises(RuntimeError, match="setup failed"):
        sandbox.open()
    assert not sandbox.is_open
    assert sandbox._session_timer is None
    assert not sandbox.workspace.exists()
    assert sandbox._active_process is None


@pytest.mark.parametrize(
    "options",
    [
        {"image": "python"},
        {"dockerfile": "Dockerfile"},
        {"runtime_configs": {"privileged": True}},
        {"keep_template": True},
        {"commit_container": True},
        {"container_id": "existing"},
        {"workdir": "/sandbox;echo"},
        {"workdir": "/sandbox/../etc"},
        {"workdir": "/"},
        {"workdir": "/./"},
        {"default_timeout": 0},
        {"execution_timeout": -1},
        {"session_timeout": float("nan")},
        {"env_vars": {"bad=name": "x"}},
        {"env_vars": {"VALUE": "has\0nul"}},
        {"env_vars": {"VALUE": 123}},
    ],
)
def test_unsupported_or_unsafe_options_fail_early(options: dict) -> None:
    """Reject unsupported settings instead of silently weakening isolation."""
    with pytest.raises(ValueError, match=".+"):
        TrustedTestSession(**options)


def test_unknown_options_fail_early() -> None:
    """Typos and container options cannot silently disappear."""
    with pytest.raises(TypeError, match="Unknown native sandbox options"):
        TrustedTestSession(privileged=True)


def test_closed_session_rejects_commands() -> None:
    """No path executes host commands if the native session has not opened."""
    sandbox = TrustedTestSession()
    with pytest.raises(NotOpenSessionError):
        sandbox.execute_command("true")


def test_empty_command_does_not_change_session_state(session: TrustedTestSession) -> None:
    """Reject empty input before launching a process or changing the workspace."""
    with pytest.raises(CommandEmptyError):
        session.execute_command("")
    assert session._active_process is None
    assert session.execute_command("printf valid").stdout == "valid"


def test_zero_run_timeout_is_not_treated_as_unlimited(session: TrustedTestSession) -> None:
    """Explicit zero is rejected before BaseSession's truthy fallback can discard it."""
    with pytest.raises(ValueError, match="timeout"):
        session.run("pass", timeout=0)


def test_subprocess_stdin_is_devnull(session: TrustedTestSession) -> None:
    """Child code cannot wait on or consume the parent application's stdin."""
    assert session.execute_command("/usr/bin/python3 -c 'import sys; print(repr(sys.stdin.read()))'").stdout == "''\n"


def test_host_process_descriptors_are_closed(session: TrustedTestSession, tmp_path: Path) -> None:
    """Inheritable descriptors not explicitly required by the driver stay private."""
    with (tmp_path / "secret").open("w") as secret:
        os.set_inheritable(secret.fileno(), True)  # noqa: FBT003
        command = f"/usr/bin/python3 -c 'import os; os.fstat({secret.fileno()})'"
        assert session.execute_command(command).exit_code != 0


def test_artifacts_persist_between_commands(session: TrustedTestSession, tmp_path: Path) -> None:
    """Private /tmp holds files until the session closes, including artifact transfers."""
    session.execute_command("mkdir -p /tmp/sandbox_plots; printf image > /tmp/sandbox_plots/plot.png")
    assert session.execute_command("cat /tmp/sandbox_plots/plot.png").stdout == "image"
    destination = tmp_path / "download.png"
    session.copy_from_runtime("/tmp/sandbox_plots/plot.png", str(destination))
    assert destination.read_bytes() == b"image"
    session.copy_to_runtime(str(destination), "/tmp/roundtrip.png")
    assert (session.tempdir / "roundtrip.png").read_bytes() == b"image"


def test_get_archive_preserves_handler_contract(session: TrustedTestSession) -> None:
    """The file-only adapter provides a bounded tar with the requested basename and bytes."""
    target = session.tempdir / "image.png"
    target.write_bytes(b"fake image bytes")
    data, metadata = session.get_archive("/tmp/image.png")
    assert metadata["name"] == "image.png"
    assert metadata["size"] == len(b"fake image bytes")
    with tarfile.open(fileobj=io.BytesIO(data)) as archive:
        assert archive.getnames() == ["image.png"]
        extracted = archive.extractfile("image.png")
        assert extracted is not None
        assert extracted.read() == b"fake image bytes"


def test_python_handler_extracts_persistent_artifacts(
    session: TrustedTestSession, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Exercise the existing handler without depending on matplotlib or a real sandbox."""
    session.execute_command("mkdir -p /tmp/sandbox_plots; printf image > /tmp/sandbox_plots/plot.png")
    original_execute = session.execute_command

    def virtual_output(command: str) -> ConsoleOutput:
        result = original_execute(command)
        if command.startswith("find "):
            # The trusted driver must translate its host-path output to the runtime's view.
            return replace(result, stdout=result.stdout.replace(str(session.tempdir), "/tmp"))
        return result

    monkeypatch.setattr(session, "execute_command", virtual_output)
    plots = PythonHandler().extract_plots(session, "/tmp/sandbox_plots")
    assert len(plots) == 1
    assert base64.b64decode(plots[0].content_base64) == b"image"


def test_artifact_run_skips_rejected_files_and_keeps_safe_plots(
    session: TrustedTestSession, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Unsafe matches cannot discard completed output or valid plots later in collection."""
    plots_dir = session.tempdir / "sandbox_plots"
    plots_dir.mkdir()
    (plots_dir / "01_safe.png").write_bytes(b"first safe plot")
    secret = tmp_path / "secret"
    secret.write_bytes(b"host secret")
    (plots_dir / "02_link.png").symlink_to(secret)
    (plots_dir / "03_directory.png").mkdir()
    (plots_dir / "04_safe.svg").write_bytes(b"last safe plot")
    original_execute = session.execute_command

    def virtual_output(command: str, *args: Any, **kwargs: Any) -> ConsoleOutput:
        result = original_execute(command, *args, **kwargs)
        if command.startswith("find "):
            return replace(result, stdout=result.stdout.replace(str(session.tempdir), "/tmp"))
        return result

    monkeypatch.setattr(session, "execute_command", virtual_output)
    monkeypatch.setattr(session.language_handler, "inject_plot_detection_code", lambda code: code)
    monkeypatch.setattr("llm_sandbox.session.create_session", lambda **_kwargs: session)
    wrapper = ArtifactSandboxSession(backend="bubblewrap")
    result = wrapper.run("print('completed')")
    assert result.exit_code == 0
    assert result.stdout == "completed\n"
    assert [base64.b64decode(plot.content_base64) for plot in result.plots] == [b"first safe plot", b"last safe plot"]
    with pytest.raises(SecurityError):
        session.get_archive("/tmp/sandbox_plots/02_link.png")
    with pytest.raises(SecurityError):
        session.get_archive("/tmp/sandbox_plots/03_directory.png")


def test_get_archive_rejects_oversized_file() -> None:
    """Artifact content is capped before an in-memory tar is allocated."""
    with TrustedTestSession(
        sandbox_config=LocalSandboxConfig(max_artifact_bytes=4), skip_environment_setup=True
    ) as sandbox:
        (sandbox.tempdir / "large").write_bytes(b"12345")
        with pytest.raises(SandboxError, match="exceed"):
            sandbox.get_archive("/tmp/large")


@pytest.mark.parametrize("path", ["/etc/passwd", "/proc/self/environ", "/tmp/../etc/passwd", "../tmp/file"])
def test_runtime_copy_mapping_rejects_outside_paths(session: TrustedTestSession, path: str) -> None:
    """The temporary-directory mapping never grants access to host or other runtime paths."""
    with pytest.raises(SecurityError):
        session.get_archive(path)


def test_get_archive_rejects_symlinks_and_directories(session: TrustedTestSession) -> None:
    """Artifact extraction shares the same nofollow boundary as regular copies."""
    (session.tempdir / "link").symlink_to("/etc/passwd")
    with pytest.raises(SecurityError):
        session.get_archive("/tmp/link")
    with pytest.raises(SecurityError):
        session.get_archive("/tmp")
