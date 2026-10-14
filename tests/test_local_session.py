"""Supervisor unit tests with trusted commands; the test driver provides NO isolation."""

from __future__ import annotations

import base64
import io
import os
import re
import shlex
import subprocess
import tarfile
import threading
import time
from contextlib import contextmanager
from dataclasses import replace
from typing import TYPE_CHECKING, Any

import pytest

from llm_sandbox.exceptions import (
    CommandFailedError,
    LibraryInstallationNotSupportedError,
    NotOpenSessionError,
    SandboxError,
    SandboxTimeoutError,
    SecurityError,
)
from llm_sandbox.language_handlers.python_handler import PythonHandler
from llm_sandbox.local.config import LocalSandboxConfig
from llm_sandbox.local.session import LocalProcessSession

if TYPE_CHECKING:
    from collections.abc import Iterator
    from pathlib import Path

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
        assert sandbox.execute_command("cat /sandbox/nested/data").stdout == "persistent"
        destination = tmp_path / "copied"
        sandbox.copy_from_runtime("nested/data", str(destination))
        assert destination.read_text(encoding="utf-8") == "persistent"
    assert not workspace.exists()
    assert not tempdir.exists()
    assert not workspace.parent.exists()
    sandbox.close()


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
        sandbox.close()
        sandbox.open()
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
        {"default_timeout": 0},
        {"execution_timeout": -1},
        {"session_timeout": float("nan")},
        {"env_vars": {"bad=name": "x"}},
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
    with pytest.raises(NotOpenSessionError):
        TrustedTestSession().execute_command("true")


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
