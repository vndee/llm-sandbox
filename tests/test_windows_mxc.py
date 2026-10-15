"""Native MXC policy contracts and opt-in tests of real PSEC enforcement."""

from __future__ import annotations

import io
import json
import os
import socket
import sys
import tarfile
import threading
import time
from pathlib import Path, PureWindowsPath
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any
from unittest.mock import Mock

import pytest
from pydantic import ValidationError

from llm_sandbox import windows_mxc as mxc
from llm_sandbox.data import ConsoleOutput
from llm_sandbox.exceptions import (
    CommandEmptyError,
    CommandFailedError,
    LanguageNotSupportedError,
    LibraryInstallationNotSupportedError,
    MissingDependencyError,
    NotOpenSessionError,
    SandboxError,
    SandboxTimeoutError,
    SecurityError,
)
from llm_sandbox.local.config import LocalSandboxConfig
from llm_sandbox.security.policy import SecurityIssueSeverity, SecurityPattern, SecurityPolicy
from llm_sandbox.windows.config import WindowsMxcConfig, explicit_environment

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator


@pytest.fixture
def adapter(monkeypatch: pytest.MonkeyPatch) -> Iterator[SimpleNamespace]:
    """Mock native boundaries while preserving Windows policy and request construction."""
    resolve_read_roots = mxc.WindowsMxcSession._resolve_read_roots
    monkeypatch.setattr(mxc, "_require_windows", lambda: None)
    monkeypatch.setattr(mxc, "Path", PureWindowsPath)
    bootstrap = {
        "SystemRoot": r"C:\Windows",
        "WINDIR": r"C:\Windows",
        "SystemDrive": "C:",
        "LOCALAPPDATA": r"C:\Users\caller\AppData\Local",
        "PATH": r"C:\Windows\System32",
    }
    monkeypatch.setattr(mxc, "_host_environment", lambda: dict(bootstrap))
    monkeypatch.setattr(mxc, "_resolve_executable", lambda value, _label: PureWindowsPath(value))
    monkeypatch.setattr(
        mxc.WindowsMxcSession, "_resolve_read_roots", lambda _self, _language: (r"C:\Windows", r"C:\Python")
    )
    workspace = Mock(name=r"C:\Private\workspace")
    workspace.name = r"C:\Private\workspace"
    owner_factory = Mock(return_value=workspace)
    supervisor = Mock()
    probe = ConsoleOutput(stdout=json.dumps({"protocol": 1, "available": True, "deny_paths": True}))
    supervisor.run.side_effect = [probe, ConsoleOutput(stdout="Python 3.12")]
    supervisor_factory = Mock(return_value=supervisor)
    writes = Mock()
    validates = Mock()
    monkeypatch.setattr(mxc, "PrivateWorkspace", owner_factory)
    monkeypatch.setattr(mxc, "ProcessSupervisor", supervisor_factory)
    monkeypatch.setattr(mxc, "safe_mkdir", Mock())
    monkeypatch.setattr(mxc, "safe_write_file", writes)
    monkeypatch.setattr(mxc, "validate_existing", validates)
    sessions: list[mxc.WindowsMxcSession] = []

    def create(**kwargs: Any) -> mxc.WindowsMxcSession:
        kwargs.setdefault("sandbox_config", WindowsMxcConfig(python_executable=r"C:\Python\python.exe"))
        result = mxc.WindowsMxcSession(mxc_path=r"C:\Tools\llm-sandbox-mxc.exe", **kwargs)
        sessions.append(result)
        return result

    yield SimpleNamespace(
        create=create,
        supervisor=supervisor,
        supervisor_factory=supervisor_factory,
        workspace=workspace,
        owner_factory=owner_factory,
        writes=writes,
        validates=validates,
        probe=probe,
        resolve_read_roots=resolve_read_roots,
    )
    for session in sessions:
        session.close()


def open_adapter(adapter: SimpleNamespace, **kwargs: Any) -> mxc.WindowsMxcSession:
    """Open through the capability and runtime checks, then isolate command assertions."""
    session = adapter.create(**kwargs)
    session.open()
    adapter.supervisor.run.reset_mock(side_effect=True)
    adapter.supervisor.run.return_value = ConsoleOutput(stdout="result")
    return session


@pytest.mark.parametrize(
    "path",
    [
        r"relative\file",
        r"C:relative",
        r"\Windows",
        r"\\server\share\file",
        r"\\?\C:\file",
        r"C:\data\..\secret",
        r"C:\data\.\file",
        r"C:\data\file:stream",
        "C:\\data\\trailing.",
        "C:\\data\\trailing ",
        r"C:\data\CON.txt",
        r"C:\data\LPT1",
        r"C:\data\CONIN$",
        r"C:\data\CON .txt",
        "C:\\data\\bad\0name",
        "C:\\",
    ],
)
def test_policy_rejects_ambiguous_windows_paths(path: str) -> None:
    """Grants cannot silently expand through alternate Windows path interpretations."""
    with pytest.raises(ValidationError):
        WindowsMxcConfig(read_only_paths=(path,))


def test_configuration_defaults_and_case_insensitive_deduplication() -> None:
    """Networking is closed by default and duplicated grants have one canonical form."""
    config = WindowsMxcConfig(read_only_paths=("C:/Data", r"c:\data"))
    assert config.network == "none"
    assert config.read_only_paths == (r"C:\Data",)
    assert config.max_output_bytes == 8 * 1024 * 1024


@pytest.mark.parametrize(
    "values", [{"SystemRoot": r"C:\Fake"}, {"LOCALAPPDATA": r"C:\Fake"}, {"Path": "a", "PATH": "b"}]
)
def test_payload_cannot_replace_host_bootstrap(values: dict[str, str]) -> None:
    """SDK initialization uses the launcher's trusted Windows locations."""
    with pytest.raises(ValueError, match="reserved|duplicated"):
        explicit_environment(values)


@pytest.mark.parametrize("options", [{"network": "filtered"}, {"max_output_bytes": 0}, {"max_artifact_bytes": True}])
def test_unsupported_policy_is_rejected(options: dict[str, Any]) -> None:
    """Unsupported network modes and ineffective limits never become permissive defaults."""
    with pytest.raises(ValidationError):
        WindowsMxcConfig(**options)


@pytest.mark.skipif(sys.platform == "win32", reason="Checks the non-Windows platform guard")
def test_non_windows_host_refuses_backend() -> None:
    """There is no compatibility or unsandboxed execution fallback."""
    with pytest.raises(MissingDependencyError, match="native Windows"):
        mxc.WindowsMxcSession()


def test_trusted_executable_allows_hardlinks_but_requires_guarded_validation(monkeypatch: pytest.MonkeyPatch) -> None:
    """Windows system binaries may be WinSxS hardlinks; reparse-point validation remains mandatory."""
    validator = Mock()
    monkeypatch.setattr(mxc, "Path", PureWindowsPath)
    monkeypatch.setattr(mxc, "validate_existing", validator)
    path = mxc._resolve_executable(r"C:\Windows\System32\cmd.exe", "command shell")
    validator.assert_called_once_with(path, directory=False, allow_hardlinks=True)


def test_incompatible_backend_configuration_rejected(adapter: SimpleNamespace) -> None:
    """A Linux policy cannot be mistaken for the native Windows policy model."""
    config = LocalSandboxConfig()
    with pytest.raises(TypeError, match="WindowsMxcConfig"):
        adapter.create(sandbox_config=config)
    adapter.supervisor.run.assert_not_called()


@pytest.mark.parametrize("language", ["r", "java", "go"])
def test_unsupported_language_refused_before_any_launch(adapter: SimpleNamespace, language: str) -> None:
    """Only native language handlers that are actually implemented are selectable."""
    with pytest.raises(LanguageNotSupportedError):
        adapter.create(lang=language)
    adapter.supervisor.run.assert_not_called()


@pytest.mark.parametrize(
    "options", [{"image": "python:3"}, {"runtime_configs": {"privileged": True}}, {"workdir": "C:\\"}]
)
def test_container_options_cannot_be_silently_ignored(adapter: SimpleNamespace, options: dict[str, Any]) -> None:
    """Meaningful unsupported settings fail before any native process starts."""
    with pytest.raises(ValueError, match="Windows MXC"):
        adapter.create(**options)
    adapter.supervisor.run.assert_not_called()


def test_open_probes_and_starts_runtime_only_through_helper(adapter: SimpleNamespace) -> None:
    """Construction performs no launches, then open verifies native capability and runtime startup."""
    session = adapter.create(image=None, runtime_configs={}, keep_template=False, commit_container=False)
    adapter.supervisor.run.assert_not_called()
    session.open()
    calls = adapter.supervisor.run.call_args_list
    assert calls[0].args[0] == [r"C:\Tools\llm-sandbox-mxc.exe", "--probe"]
    assert calls[1].args[0] == [r"C:\Tools\llm-sandbox-mxc.exe", "--run"]
    request = json.loads(calls[1].kwargs["input"])
    assert request["argv"] == [r"C:\Python\python.exe", "--version"]
    assert request["network"] == "none"
    assert request["writable_paths"] == [str(session.workspace)]


@pytest.mark.parametrize(
    "response",
    [
        "not-json",
        "[]",
        '{"protocol":true,"available":true}',
        '{"protocol":2,"available":true}',
        '{"protocol":1,"available":false}',
    ],
)
def test_incompatible_or_unavailable_probe_fails_closed(adapter: SimpleNamespace, response: str) -> None:
    """An unsuccessful probe cleans its workspace and never attempts a runtime launch."""
    adapter.supervisor.run.side_effect = [ConsoleOutput(stdout=response)]
    session = adapter.create()
    with pytest.raises(SandboxError):
        session.open()
    assert adapter.supervisor.run.call_count == 1
    adapter.supervisor.close.assert_called_once()
    adapter.workspace.cleanup.assert_called_once()
    assert not session.is_open


def test_deny_policy_requires_native_support(adapter: SimpleNamespace) -> None:
    """A host without native deny-path enforcement cannot accept that policy."""
    adapter.supervisor.run.side_effect = [ConsoleOutput(stdout='{"protocol":1,"available":true,"deny_paths":false}')]
    config = WindowsMxcConfig(python_executable=r"C:\Python\python.exe", deny_paths=(r"C:\Python\secret",))
    session = adapter.create(sandbox_config=config)
    with pytest.raises(SandboxError, match="deny paths"):
        session.open()
    assert adapter.supervisor.run.call_count == 1


@pytest.mark.parametrize(
    ("language", "executable", "extension"),
    [("python", r"C:\Python\python.exe", ".py"), ("javascript", r"C:\Node\node.exe", ".js")],
)
def test_source_is_staged_and_passed_as_exact_arguments(
    adapter: SimpleNamespace, language: str, executable: str, extension: str
) -> None:
    """Code is a file inside the writable workspace, never part of a host shell command."""
    config = WindowsMxcConfig(python_executable=r"C:\Python\python.exe", javascript_executable=r"C:\Node\node.exe")
    session = open_adapter(adapter, sandbox_config=config, lang=language)
    code = 'print("quotes & %VARIABLE% $() ✓")'
    stdout = Mock()
    result = session.run(code, timeout=4, on_stdout=stdout)
    script, content = adapter.writes.call_args.args
    assert script.parent == session.workspace
    assert script.suffix == extension
    assert content == code.encode("utf-8")
    call = adapter.supervisor.run.call_args
    assert call.args[0] == [r"C:\Tools\llm-sandbox-mxc.exe", "--run"]
    assert call.kwargs["timeout"] == 4
    assert call.kwargs["on_stdout"] is stdout
    assert json.loads(call.kwargs["input"])["argv"] == [executable, str(script)]
    assert result.stdout == "result"


def test_command_uses_staged_batch_script_inside_psec(adapter: SimpleNamespace) -> None:
    """cmd.exe only receives a staged script as an isolated payload argument."""
    adapter.workspace.name = r"C:\Users\Name & %TEMP%\private"
    session = open_adapter(adapter)
    session.execute_command('echo "hello & world"')
    script, content = adapter.writes.call_args.args
    assert script.suffix == ".cmd"
    assert content == b'@echo off\r\nchcp 65001 >nul\r\necho "hello & world"\r\n'
    call = adapter.supervisor.run.call_args
    assert call.args[0][0].endswith("llm-sandbox-mxc.exe")
    request = json.loads(call.kwargs["input"])
    assert request["argv"] == [r"C:\Windows\System32\cmd.exe", "/d", "/c", script.name]
    assert request["cwd"] == str(session.workspace)


def test_environment_overrides_only_reach_payload(adapter: SimpleNamespace, monkeypatch: pytest.MonkeyPatch) -> None:
    """Host secrets and payload-specified settings cannot alter the helper's environment."""
    monkeypatch.setenv("HOST_SECRET", "not-for-payload")
    config = WindowsMxcConfig(python_executable=r"C:\Python\python.exe", env={"APP_MODE": "explicit", "Path": "custom"})
    session = open_adapter(adapter, sandbox_config=config, env_vars={"app_mode": "override"})
    session.run("pass")
    call = adapter.supervisor.run.call_args
    payload = json.loads(call.kwargs["input"])["env"]
    assert payload["app_mode"] == "override"
    assert "APP_MODE" not in payload
    assert payload["Path"] == "custom"
    assert payload["PYTHONUTF8"] == "1"
    assert payload["PYTHONIOENCODING"] == "utf-8"
    assert "HOST_SECRET" not in payload
    assert "APP_MODE" not in call.kwargs["env"]
    assert "app_mode" not in call.kwargs["env"]
    assert call.kwargs["env"]["PATH"] == r"C:\Windows\System32"
    config.env["SystemRoot"] = r"C:\malicious"
    with pytest.raises(ValueError, match="reserved"):
        session.run("pass")
    assert adapter.supervisor.run.call_count == 1


@pytest.mark.parametrize("path", [r"..\escape", r"C:\elsewhere", r"file:stream", r"\\server\share"])
def test_command_workdir_cannot_escape_workspace(adapter: SimpleNamespace, path: str) -> None:
    """All execution working directories stay within the session's single writable root."""
    session = open_adapter(adapter)
    with pytest.raises(SecurityError):
        session.execute_argv([r"C:\Python\python.exe", "--version"], workdir=path)
    adapter.supervisor.run.assert_not_called()


def test_request_limit_refuses_launch(adapter: SimpleNamespace, monkeypatch: pytest.MonkeyPatch) -> None:
    """An oversized request fails before creating the helper process."""
    session = open_adapter(adapter)
    monkeypatch.setattr(mxc, "_MAX_REQUEST_BYTES", 1)
    with pytest.raises(ValueError, match="protocol limit"):
        session.execute_argv([r"C:\Python\python.exe", "--version"])
    adapter.supervisor.run.assert_not_called()


def test_failure_stops_processes_before_workspace_cleanup(adapter: SimpleNamespace) -> None:
    """Timeouts invalidate the session and keep deletion ordered after process termination."""
    session = open_adapter(adapter)
    events: list[str] = []
    adapter.supervisor.close.side_effect = lambda: events.append("stopped")
    adapter.workspace.cleanup.side_effect = lambda: events.append("deleted")
    adapter.supervisor.run.side_effect = SandboxTimeoutError("expired")
    with pytest.raises(SandboxTimeoutError):
        session.run("while True: pass")
    assert events == ["stopped", "deleted"]
    assert not session.is_open
    with pytest.raises(NotOpenSessionError):
        session.run("pass")


@pytest.mark.parametrize("operation", ["run", "read", "copy"])
def test_callback_cannot_reenter_workspace_operations(
    adapter: SimpleNamespace, monkeypatch: pytest.MonkeyPatch, operation: str
) -> None:
    """Callbacks cannot stage code or touch files while an untrusted payload is active."""
    session = open_adapter(adapter)
    read = Mock()
    copy = Mock()
    monkeypatch.setattr(mxc, "safe_read_file", read)
    monkeypatch.setattr(mxc, "safe_copy", copy)

    def reenter() -> None:
        if operation == "run":
            session.run("nested")
        elif operation == "read":
            session.get_archive("artifact")
        else:
            session.copy_from_runtime("artifact", r"C:\Output\artifact")

    def execute(_argv: list[str], **kwargs: Any) -> ConsoleOutput:
        kwargs["on_stdout"]("chunk")
        return ConsoleOutput()

    adapter.supervisor.run.side_effect = execute

    def callback(_chunk: str) -> None:
        reenter()

    with pytest.raises(SandboxError, match="reentered"):
        session.run("outer", on_stdout=callback)
    assert adapter.writes.call_count == 1
    read.assert_not_called()
    copy.assert_not_called()
    assert not session.is_open


def test_stream_logging_preserves_explicit_callbacks(adapter: SimpleNamespace, monkeypatch: pytest.MonkeyPatch) -> None:
    """stream/verbose retains the existing logging behavior without duplicating callback output."""
    session = open_adapter(adapter, stream=True, verbose=True)
    logger = Mock()
    stdout = Mock()
    monkeypatch.setattr(session, "_log", logger)
    session.run("pass", on_stdout=stdout)
    kwargs = adapter.supervisor.run.call_args.kwargs
    assert kwargs["on_stdout"] is stdout
    assert kwargs["on_stderr"] is logger


def test_reopen_uses_fresh_supervisor_and_ignores_old_timer(adapter: SimpleNamespace) -> None:
    """A timer from an earlier lifetime cannot cancel newly opened work."""
    session = open_adapter(adapter)
    old_generation = session._generation
    session.close()
    adapter.supervisor.run.side_effect = [adapter.probe, ConsoleOutput()]
    session.open()
    session._close_generation(old_generation)
    assert session.is_open
    assert adapter.supervisor_factory.call_count == 2
    assert adapter.workspace.cleanup.call_count == 1


def test_close_during_workspace_creation_prevents_late_launch(adapter: SimpleNamespace) -> None:
    """Closing an opening session cannot be undone by delayed workspace allocation."""
    entered = threading.Event()
    release = threading.Event()
    errors: list[BaseException] = []
    session = adapter.create()

    def allocate() -> Mock:
        entered.set()
        assert release.wait(5)
        return adapter.workspace

    def open_session() -> None:
        try:
            session.open()
        except BaseException as exc:  # noqa: BLE001
            errors.append(exc)

    adapter.owner_factory.side_effect = allocate
    worker = threading.Thread(target=open_session)
    closer = threading.Thread(target=session.close)
    worker.start()
    try:
        assert entered.wait(5)
        closer.start()
        deadline = time.monotonic() + 5
        while not session._closing and time.monotonic() < deadline:
            time.sleep(0.001)
        assert session._closing
        release.set()
        worker.join(5)
        closer.join(5)
        assert not worker.is_alive()
        assert not closer.is_alive()
        assert len(errors) == 1
        assert isinstance(errors[0], NotOpenSessionError)
        adapter.supervisor.run.assert_not_called()
        adapter.workspace.cleanup.assert_called_once()
    finally:
        release.set()
        worker.join(5)
        if closer.ident is not None:
            closer.join(5)


def test_archive_is_bounded_regular_file(adapter: SimpleNamespace, monkeypatch: pytest.MonkeyPatch) -> None:
    """Artifact bytes use the guarded reader and the configured export limit."""
    reader = Mock(return_value=b"artifact")
    monkeypatch.setattr(mxc, "safe_read_file", reader)
    config = WindowsMxcConfig(python_executable=r"C:\Python\python.exe", max_artifact_bytes=100)
    session = open_adapter(adapter, sandbox_config=config)
    raw, info = session.get_archive("output.txt")
    reader.assert_called_once_with(session.workspace / "output.txt", 100)
    assert info["size"] == 8
    with tarfile.open(fileobj=io.BytesIO(raw)) as archive:
        assert archive.getnames() == ["output.txt"]
        member = archive.extractfile("output.txt")
        assert member is not None
        assert member.read() == b"artifact"


def test_package_installation_is_explicitly_unsupported(adapter: SimpleNamespace) -> None:
    """Missing packaging support cannot mutate the host interpreter."""
    session = open_adapter(adapter)
    with pytest.raises(LibraryInstallationNotSupportedError):
        session.run("pass", libraries=["numpy"])
    adapter.supervisor.run.assert_not_called()
    adapter.writes.assert_not_called()


def test_source_security_policy_remains_explicit_advisory_check(adapter: SimpleNamespace) -> None:
    """Preserve BaseSession's caller-controlled regex check rather than changing run semantics."""
    policy = SecurityPolicy(
        severity_threshold=SecurityIssueSeverity.HIGH,
        patterns=[
            SecurityPattern(pattern="forbidden", description="test pattern", severity=SecurityIssueSeverity.HIGH)
        ],
    )
    session = open_adapter(adapter, security_policy=policy)
    code = "print('forbidden')"
    safe, issues = session.is_safe(code)
    assert not safe
    assert len(issues) == 1
    assert session.run(code).stdout == "result"


@pytest.mark.parametrize(
    ("environment", "message"),
    [
        ({"": "value"}, "keys"),
        ({"name=value": "value"}, "keys"),
        ({"nul\0name": "value"}, "keys"),
        ({42: "value"}, "keys"),
        ({"NAME": "nul\0value"}, "values"),
        ({"NAME": 42}, "values"),
    ],
)
def test_session_environment_rejects_invalid_native_entries(
    adapter: SimpleNamespace, environment: dict[Any, Any], message: str
) -> None:
    """Malformed Windows environment blocks are rejected before helper startup."""
    with pytest.raises(ValueError, match=message):
        adapter.create(env_vars=environment)
    adapter.supervisor.run.assert_not_called()


@pytest.mark.parametrize("field", ["python_executable", "javascript_executable"])
def test_runtime_configuration_accepts_discovery_but_rejects_batch_launchers(field: str) -> None:
    """Explicit None selects runtime discovery; batch files cannot masquerade as native executables."""
    config = WindowsMxcConfig(**{field: None})
    assert getattr(config, field) is None
    options = {field: r"C:\Tools\launcher.cmd"}
    with pytest.raises(ValidationError, match=".exe executable"):
        WindowsMxcConfig(**options)


def test_host_bootstrap_is_minimal_case_insensitive_and_ignores_path(monkeypatch: pytest.MonkeyPatch) -> None:
    """Only required Windows locations enter the launcher, with a fixed System32 search path."""
    environment = {
        "systemroot": r"C:\Windows",
        "SYSTEMDRIVE": "C:",
        "localappdata": r"C:\Users\caller\AppData\Local",
        "PATH": r"C:\Untrusted",
        "HOST_SECRET": "must-not-inherit",
    }
    monkeypatch.setattr(mxc, "os", SimpleNamespace(environ=environment))
    monkeypatch.setattr(mxc, "Path", PureWindowsPath)
    assert mxc._host_environment() == {
        "SystemRoot": r"C:\Windows",
        "SystemDrive": "C:",
        "LOCALAPPDATA": r"C:\Users\caller\AppData\Local",
        "WINDIR": r"C:\Windows",
        "PATH": r"C:\Windows\System32",
    }


@pytest.mark.parametrize(("key", "value"), [("SystemRoot", None), ("LOCALAPPDATA", ""), ("SystemRoot", "C:\\bad\0")])
def test_missing_or_corrupt_host_bootstrap_fails_closed(
    monkeypatch: pytest.MonkeyPatch, key: str, value: str | None
) -> None:
    """Absent bootstrap state must not make the helper discover paths from the payload's environment."""
    environment = {"SystemRoot": r"C:\Windows", "SystemDrive": "C:", "LOCALAPPDATA": r"C:\Users\caller\AppData\Local"}
    if value is None:
        environment.pop(key)
    else:
        environment[key] = value
    monkeypatch.setattr(mxc, "os", SimpleNamespace(environ=environment))
    with pytest.raises(MissingDependencyError, match=key):
        mxc._host_environment()


def test_host_system_drive_must_be_a_drive_designator(monkeypatch: pytest.MonkeyPatch) -> None:
    """A path or command string cannot replace Windows' drive bootstrap value."""
    environment = {"SystemRoot": r"C:\Windows", "SystemDrive": r"C:\bad", "LOCALAPPDATA": r"C:\Users\caller"}
    monkeypatch.setattr(mxc, "os", SimpleNamespace(environ=environment))
    with pytest.raises(ValueError, match="local Windows drive"):
        mxc._host_environment()


def test_platform_guard_accepts_native_windows_without_selecting_a_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    """A native host passes the platform check; actual PSEC support is probed separately."""
    monkeypatch.setattr(mxc, "sys", SimpleNamespace(platform="win32"))
    mxc._require_windows()


def test_executable_resolution_rejects_batch_files_and_reports_missing_binary(monkeypatch: pytest.MonkeyPatch) -> None:
    """Trusted launchers need real native binaries; resolver errors retain useful installation context."""
    validator = Mock(side_effect=FileNotFoundError("absent"))
    monkeypatch.setattr(mxc, "Path", PureWindowsPath)
    monkeypatch.setattr(mxc, "validate_existing", validator)
    with pytest.raises(ValueError, match="absolute .exe"):
        mxc._resolve_executable(r"C:\Tools\launcher.bat", "helper")
    validator.assert_not_called()
    with pytest.raises(MissingDependencyError, match="helper was not found") as failure:
        mxc._resolve_executable(r"C:\Tools\launcher.exe", "helper")
    assert isinstance(failure.value.__cause__, FileNotFoundError)


def test_missing_helper_explains_manual_installation_without_launching(
    adapter: SimpleNamespace, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The backend gives an actionable error without attempting a runtime download or fallback."""
    resolver = Mock(side_effect=MissingDependencyError("missing helper"))
    monkeypatch.setattr(mxc, "_resolve_executable", resolver)
    with pytest.raises(MissingDependencyError, match="pass mxc_path.*never downloaded"):
        adapter.create()
    adapter.supervisor.run.assert_not_called()
    adapter.owner_factory.assert_not_called()


@pytest.mark.parametrize("node_path", [None, r"C:\Node\node.exe"])
def test_javascript_discovery_requires_installed_node(
    adapter: SimpleNamespace, monkeypatch: pytest.MonkeyPatch, node_path: str | None
) -> None:
    """Missing Node.js fails early; discovered Node is launched only as an isolated payload."""
    discover = Mock(return_value=node_path)
    monkeypatch.setattr(mxc, "shutil", SimpleNamespace(which=discover))
    if node_path is None:
        with pytest.raises(MissingDependencyError, match="Install Node.js"):
            adapter.create(lang="javascript")
        adapter.supervisor.run.assert_not_called()
    else:
        session = open_adapter(adapter, lang="javascript")
        session.run("console.log('hello')")
        request = json.loads(adapter.supervisor.run.call_args.kwargs["input"])
        assert request["argv"][0] == node_path
    discover.assert_called_once_with("node.exe")


@pytest.mark.parametrize("explicit_interpreter", [False, True])
def test_runtime_read_grants_include_only_required_and_explicit_roots(
    adapter: SimpleNamespace, monkeypatch: pytest.MonkeyPatch, explicit_interpreter: bool
) -> None:
    """Current Python includes its base install; external interpreters receive no guessed host grants."""
    monkeypatch.setattr(mxc.WindowsMxcSession, "_resolve_read_roots", adapter.resolve_read_roots)
    monkeypatch.setattr(
        mxc,
        "sys",
        SimpleNamespace(executable=r"C:\Venv\Scripts\python.exe", prefix=r"C:\Venv", base_prefix=r"C:\Python"),
    )
    executable = r"C:\External\python.exe" if explicit_interpreter else None
    config = WindowsMxcConfig(python_executable=executable, read_only_paths=(r"c:\WINDOWS", r"C:\Project"))
    session = open_adapter(adapter, sandbox_config=config)
    session.run("pass")
    roots = json.loads(adapter.supervisor.run.call_args.kwargs["input"])["read_only_paths"]
    expected = (
        [r"C:\Windows", r"C:\External", r"C:\Project"]
        if explicit_interpreter
        else [r"C:\Windows", r"C:\Venv\Scripts", r"C:\Venv", r"C:\Python", r"C:\Project"]
    )
    assert roots == expected
    assert session.python_executable_path == (executable or r"C:\Venv\Scripts\python.exe")
    adapter.validates.assert_any_call(PureWindowsPath(r"C:\Project"))


@pytest.mark.parametrize(
    ("options", "error", "message"),
    [
        ({"libraries": ["numpy"]}, LibraryInstallationNotSupportedError, "Library installation"),
        ({"stream": "yes"}, TypeError, "stream must be a boolean"),
        ({"unsupported_option": True}, TypeError, "Unknown Windows MXC options: unsupported_option"),
    ],
)
def test_unsupported_session_options_fail_before_allocating_resources(
    adapter: SimpleNamespace, options: dict[str, Any], error: type[Exception], message: str
) -> None:
    """Unsupported behavior must not become silently ignored session configuration."""
    with pytest.raises(error, match=message):
        adapter.create(**options)
    adapter.owner_factory.assert_not_called()
    adapter.supervisor.run.assert_not_called()


@pytest.mark.parametrize("field", ["default_timeout", "execution_timeout", "session_timeout"])
@pytest.mark.parametrize("value", [0, -1, float("nan"), float("inf")])
def test_invalid_configured_deadlines_are_rejected(adapter: SimpleNamespace, field: str, value: float) -> None:
    """Invalid deadlines cannot silently disable resource limits."""
    with pytest.raises(ValueError, match="finite and positive"):
        adapter.create(**{field: value})
    adapter.owner_factory.assert_not_called()


def test_open_is_idempotent_and_failed_runtime_startup_cleans_up(adapter: SimpleNamespace) -> None:
    """A usable session has one startup; a runtime missing its dependencies never stays open."""
    session = adapter.create()
    session.open()
    session.open()
    assert adapter.owner_factory.call_count == 1
    assert adapter.supervisor.run.call_count == 2
    session.close()
    adapter.supervisor.run.side_effect = [adapter.probe, ConsoleOutput(exit_code=1, stderr="missing base Python")]
    with pytest.raises(CommandFailedError, match="missing base Python"):
        session.open()
    assert not session.is_open
    assert adapter.workspace.cleanup.call_count == 2


@pytest.mark.parametrize("cleanup_fails", [False, True])
def test_workspace_setup_failure_retains_cleanup_ownership(
    adapter: SimpleNamespace, monkeypatch: pytest.MonkeyPatch, cleanup_fails: bool
) -> None:
    """Allocation failures preserve their error and a failed deletion remains retryable on close."""
    setup_error = OSError("workspace setup failed")
    monkeypatch.setattr(mxc, "safe_mkdir", Mock(side_effect=setup_error))
    if cleanup_fails:
        adapter.workspace.cleanup.side_effect = [PermissionError("cleanup denied"), None]
    session = adapter.create()
    with pytest.raises(OSError, match="workspace setup failed") as failure:
        session.open()
    assert failure.value is setup_error
    assert not session.is_open
    adapter.supervisor.run.assert_not_called()
    adapter.workspace.cleanup.assert_called_once()
    if cleanup_fails:
        assert isinstance(failure.value.__cause__, PermissionError)
        session.close()
        assert adapter.workspace.cleanup.call_count == 2


@pytest.mark.parametrize("stage", ["supervisor", "timer"])
@pytest.mark.parametrize("cleanup_fails", [False, True])
def test_failed_initialization_retries_checks_and_cancels_timers(
    adapter: SimpleNamespace, monkeypatch: pytest.MonkeyPatch, stage: str, cleanup_fails: bool
) -> None:
    """Partially initialized sessions stay closed and must pass all checks on their next open."""
    session = adapter.create(session_timeout=30)
    setup_error = RuntimeError(f"{stage} startup failed")
    first_timer, second_timer = Mock(), Mock()
    timer_factory = Mock(side_effect=[first_timer, second_timer])
    monkeypatch.setattr(mxc, "threading", SimpleNamespace(Timer=timer_factory))
    if stage == "supervisor":
        adapter.supervisor_factory.side_effect = [setup_error, adapter.supervisor]
    else:
        first_timer.start.side_effect = setup_error
    if cleanup_fails:
        adapter.workspace.cleanup.side_effect = [PermissionError("cleanup denied"), None, None]

    with pytest.raises(RuntimeError, match="startup failed") as failure:
        session.open()

    assert failure.value is setup_error
    assert not session.is_open
    assert session._supervisor is None
    assert session._session_timer is None
    adapter.workspace.cleanup.assert_called_once()
    adapter.supervisor.run.assert_not_called()
    if stage == "timer":
        first_timer.cancel.assert_called_once()
        adapter.supervisor.close.assert_called_once()
    else:
        timer_factory.assert_not_called()
    if cleanup_fails:
        assert isinstance(failure.value.__cause__, PermissionError)
        assert session._workspace_owner is adapter.workspace
    else:
        assert session._workspace_owner is None

    session.open()

    assert session.is_open
    assert adapter.owner_factory.call_count == 2
    assert adapter.supervisor_factory.call_count == 2
    assert adapter.supervisor.run.call_count == 2
    assert adapter.supervisor.run.call_args_list[0].args[0][-1] == "--probe"
    assert adapter.supervisor.run.call_args_list[1].args[0][-1] == "--run"
    live_timer = second_timer if stage == "timer" else first_timer
    live_timer.start.assert_called_once()
    live_timer.cancel.assert_not_called()
    session.close()
    live_timer.cancel.assert_called_once()
    assert session._session_timer is None
    assert session._workspace_owner is None


def test_reopen_retries_retained_cleanup_before_allocating_another_workspace(adapter: SimpleNamespace) -> None:
    """A failed cleanup cannot be abandoned when the caller reopens the same session."""
    session = open_adapter(adapter)
    adapter.workspace.cleanup.side_effect = [PermissionError("cleanup denied"), None, None]
    with pytest.raises(PermissionError, match="cleanup denied"):
        session.close()
    assert not session.is_open
    adapter.supervisor.run.side_effect = [adapter.probe, ConsoleOutput()]
    session.open()
    assert session.is_open
    assert adapter.workspace.cleanup.call_count == 2
    assert adapter.owner_factory.call_count == 2


@pytest.mark.parametrize("default_timeout", [None, 60])
def test_session_deadline_caps_commands_and_timer_cancels_lifetime(
    adapter: SimpleNamespace, monkeypatch: pytest.MonkeyPatch, default_timeout: float | None
) -> None:
    """Both bounded and unlimited commands remain bounded by the session's remaining lifetime."""
    session = adapter.create(default_timeout=default_timeout, session_timeout=30)
    timer = Mock()
    timer_factory = Mock(return_value=timer)
    clock = Mock(return_value=100.0)
    monkeypatch.setattr(mxc, "threading", SimpleNamespace(Timer=timer_factory))
    monkeypatch.setattr(mxc, "time", SimpleNamespace(time=clock, monotonic=clock))
    session.open()
    timer.start.assert_called_once()
    assert timer.daemon is True
    assert adapter.supervisor.run.call_args.kwargs["timeout"] == 30
    adapter.supervisor.run.reset_mock(side_effect=True)
    adapter.supervisor.run.return_value = ConsoleOutput()
    clock.return_value = 125.0
    session.run("pass")
    assert adapter.supervisor.run.call_args.kwargs["timeout"] == 5
    _, callback, args = timer_factory.call_args.args
    callback(*args)
    assert not session.is_open
    timer.cancel.assert_called_once()
    adapter.workspace.cleanup.assert_called_once()


def test_expired_session_deadline_rejects_launch_even_before_timer_callback(
    adapter: SimpleNamespace, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A delayed timer thread cannot allow execution beyond the monotonic deadline."""
    session = adapter.create(session_timeout=30)
    timer_factory = Mock(return_value=Mock())
    clock = Mock(return_value=100.0)
    monkeypatch.setattr(mxc, "threading", SimpleNamespace(Timer=timer_factory))
    monkeypatch.setattr(mxc, "time", SimpleNamespace(time=clock, monotonic=clock))
    session.open()
    adapter.supervisor.run.reset_mock(side_effect=True)
    clock.return_value = 130.0
    with pytest.raises(SandboxTimeoutError, match="session exceeded its deadline"):
        session.run("must not execute")
    adapter.supervisor.run.assert_not_called()
    assert not session.is_open


@pytest.mark.parametrize("timeout", [0, -1, float("nan"), float("inf")])
def test_invalid_per_command_deadline_never_launches(adapter: SimpleNamespace, timeout: float) -> None:
    """Per-command overrides cannot bypass finite positive timeout validation."""
    session = open_adapter(adapter)
    with pytest.raises(ValueError, match="Execution timeout must be finite and positive"):
        session.run("must not execute", timeout=timeout)
    adapter.supervisor.run.assert_not_called()


@pytest.mark.parametrize(
    ("argv", "error", "message"),
    [
        ([], CommandEmptyError, "Command cannot be empty"),
        ([r"C:\Python\python.exe", "nul\0argument"], ValueError, "without NUL"),
        ([r"C:\Python\python.exe", 42], ValueError, "must be strings"),
        ([r"C:\Tools\script.cmd"], ValueError, "absolute .exe"),
    ],
)
def test_invalid_command_arguments_never_reach_helper(
    adapter: SimpleNamespace, argv: list[Any], error: type[Exception], message: str
) -> None:
    """Reject arguments that cannot be represented faithfully without shell or string truncation."""
    session = open_adapter(adapter)
    with pytest.raises(error, match=message):
        session.execute_argv(argv)
    adapter.supervisor.run.assert_not_called()


@pytest.mark.parametrize(
    ("command", "error", "message"), [("", CommandEmptyError, "empty"), ("echo nul\0", ValueError, "NUL")]
)
def test_invalid_batch_commands_are_rejected_before_staging(
    adapter: SimpleNamespace, command: str, error: type[Exception], message: str
) -> None:
    """An empty or truncated command does not create a runnable workspace file."""
    session = open_adapter(adapter)
    with pytest.raises(error, match=message):
        session.execute_command(command)
    adapter.writes.assert_not_called()
    adapter.supervisor.run.assert_not_called()


def test_batch_command_in_subdirectory_keeps_argv_and_cwd_consistent(adapter: SimpleNamespace) -> None:
    """A relative script basename resolves in the requested guarded workspace subdirectory."""
    session = open_adapter(adapter)
    session.execute_command("echo subdirectory", workdir=r"sub dir\child")
    script = adapter.writes.call_args.args[0]
    request = json.loads(adapter.supervisor.run.call_args.kwargs["input"])
    assert script.parent == session.workspace / "sub dir" / "child"
    assert request["cwd"] == str(script.parent)
    assert request["argv"][-1] == script.name
    adapter.validates.assert_any_call(script.parent, directory=True)


def test_file_transfers_use_guarded_copy_with_host_absolute_and_workspace_relative_paths(
    adapter: SimpleNamespace, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Transfer endpoints never accidentally resolve runtime paths against the host working directory."""
    session = open_adapter(adapter)
    copy = Mock()
    monkeypatch.setattr(mxc, "safe_copy", copy)
    monkeypatch.setattr(mxc, "Path", Path)
    monkeypatch.chdir(tmp_path)
    session.copy_to_runtime("input.txt", r"nested\input.txt")
    copy.assert_called_with(tmp_path / "input.txt", session.workspace / "nested" / "input.txt")
    session.copy_from_runtime(r"nested\output.txt", "output.txt")
    copy.assert_called_with(session.workspace / "nested" / "output.txt", tmp_path / "output.txt")
    assert copy.call_count == 2
    adapter.supervisor.run.assert_not_called()


def test_base_file_hooks_create_only_workspace_paths_without_posix_commands(
    adapter: SimpleNamespace, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Inherited file helpers retain workspace containment and never emit mkdir/chown shell commands."""
    session = open_adapter(adapter)
    mkdir = Mock()
    monkeypatch.setattr(mxc, "safe_mkdir", mkdir)
    session._ensure_directory_exists(r"nested\directory")
    session._ensure_ownership(["nested", str(session.workspace / "nested" / "directory")])
    mkdir.assert_called_once_with(session.workspace / "nested" / "directory")
    with pytest.raises(SecurityError, match="inside"):
        session._ensure_ownership([r"C:\outside"])
    adapter.supervisor.run.assert_not_called()


def test_callback_cannot_close_then_reopen_the_same_active_session(adapter: SimpleNamespace) -> None:
    """Closing inside a callback cannot let a new lifetime replace the active supervisor."""
    session = open_adapter(adapter)

    def callback(_chunk: str) -> None:
        session.close()
        session.open()

    def execute(_argv: list[str], **kwargs: Any) -> ConsoleOutput:
        kwargs["on_stdout"]("ready")
        return ConsoleOutput()

    adapter.supervisor.run.side_effect = execute
    with pytest.raises(SandboxError, match="cannot reopen"):
        session.run("outer", on_stdout=callback)
    assert adapter.owner_factory.call_count == 1
    assert not session.is_open


def test_stale_close_waiting_for_operation_lock_preserves_reopened_session(adapter: SimpleNamespace) -> None:
    """A close that already captured an old supervisor must recheck generation after waiting."""
    session = open_adapter(adapter)
    old_supervisor_closed = threading.Event()
    adapter.supervisor.close.side_effect = old_supervisor_closed.set
    new_supervisor = Mock()
    new_supervisor.run.side_effect = [adapter.probe, ConsoleOutput()]
    new_workspace = Mock()
    new_workspace.name = r"C:\Private\new-workspace"
    adapter.supervisor_factory.return_value = new_supervisor
    adapter.owner_factory.return_value = new_workspace
    closer = threading.Thread(target=session.close)
    try:
        with session._operation_lock:
            closer.start()
            assert old_supervisor_closed.wait(5)
            session.open()
        closer.join(5)
        assert not closer.is_alive()
        assert session.is_open
        assert str(session.workspace) == new_workspace.name
        new_workspace.cleanup.assert_not_called()
        new_supervisor.close.assert_not_called()
    finally:
        closer.join(5)
        session.close()


@pytest.fixture
def native_mxc() -> Callable[..., mxc.WindowsMxcSession]:
    """Require an explicit native run; an opted-in unavailable host is a failure."""
    if os.environ.get("LLM_SANDBOX_RUN_MXC_INTEGRATION") != "1":
        pytest.skip("Set LLM_SANDBOX_RUN_MXC_INTEGRATION=1 on a PSEC-capable Windows host")
    assert sys.platform == "win32", "Native MXC integration requires Windows"
    helper = os.environ.get("LLM_SANDBOX_MXC_HELPER")
    assert helper, "Set LLM_SANDBOX_MXC_HELPER to the built native helper"

    def create(**kwargs: Any) -> mxc.WindowsMxcSession:
        return mxc.WindowsMxcSession(mxc_path=helper, **kwargs)

    return create


@pytest.mark.integration
def test_native_python_commands_files_and_exact_arguments(
    native_mxc: Callable[..., mxc.WindowsMxcSession], tmp_path: Path
) -> None:
    """Real PSEC starts Python/cmd and preserves workspace files and argument boundaries."""
    source = tmp_path / "input.txt"
    source.write_text("input ✓", encoding="utf-8")
    destination = tmp_path / "output.txt"
    with native_mxc() as session:
        session.copy_to_runtime(str(source), "input.txt")
        output = session.run("from pathlib import Path; print(Path('input.txt').read_text(encoding='utf-8'))")
        assert output.exit_code == 0, output.stderr
        assert output.stdout.strip() == "input ✓"
        session.copy_from_runtime("input.txt", str(destination))
        assert destination.read_text(encoding="utf-8") == "input ✓"
        arguments = ["space and & %TEMP%", 'quote"backslash\\', "", "Unicode ✓"]
        command = [sys.executable, "-c", "import json,sys;print(json.dumps(sys.argv[1:]))", *arguments]
        output = session.execute_argv(command)
        assert output.exit_code == 0, output.stderr
        assert json.loads(output.stdout) == arguments
        output = session.execute_command("echo native-command")
        assert output.exit_code == 0, output.stderr
        assert output.stdout.strip() == "native-command"


@pytest.mark.integration
def test_native_default_filesystem_rejects_external_reads_and_writes(
    native_mxc: Callable[..., mxc.WindowsMxcSession], tmp_path: Path
) -> None:
    """A real host secret and a writable host directory remain outside default grants."""
    secret = tmp_path / "secret.txt"
    secret.write_text("host secret", encoding="utf-8")
    target = tmp_path / "must-not-write.txt"
    code = (
        "from pathlib import Path\n"
        f"for operation in [lambda:Path({str(secret)!r}).read_bytes(),"
        f"lambda:Path({str(target)!r}).write_text('escaped')]:\n"
        " try:\n  operation()\n  print('ESCAPED')\n"
        " except OSError:\n  print('denied')\n"
    )
    with native_mxc() as session:
        output = session.run(code)
        assert output.exit_code == 0, output.stderr
        assert output.stdout.splitlines() == ["denied", "denied"]
    assert not target.exists()
    assert secret.read_text(encoding="utf-8") == "host secret"


@pytest.mark.integration
def test_native_read_grant_with_explicit_deny(native_mxc: Callable[..., mxc.WindowsMxcSession], tmp_path: Path) -> None:
    """Native deny paths override a read grant, or unsupported policy fails before execution."""
    allowed = tmp_path / "allowed.txt"
    denied = tmp_path / "denied.txt"
    allowed.write_text("allowed", encoding="utf-8")
    denied.write_text("secret", encoding="utf-8")
    config = WindowsMxcConfig(read_only_paths=(str(tmp_path),), deny_paths=(str(denied),))
    with native_mxc() as probe_session:
        output = probe_session._require_open().run(
            [str(probe_session._helper), "--probe"],
            env=probe_session._launcher_environment(),
            timeout=10,
        )
        assert output.exit_code == 0, output.stderr
        capabilities = json.loads(output.stdout)
        assert capabilities["available"] is True
        assert isinstance(capabilities["deny_paths"], bool)
    if not capabilities["deny_paths"]:
        session = native_mxc(sandbox_config=config)
        with pytest.raises(SandboxError, match="cannot enforce.*deny paths"):
            session.open()
        assert not session.is_open
        return
    code = (
        f"from pathlib import Path; print(Path({str(allowed)!r}).read_text())\n"
        f"try:\n Path({str(denied)!r}).read_text()\n print('ESCAPED')\n"
        "except OSError:\n print('denied')\n"
    )
    with native_mxc(sandbox_config=config) as session:
        output = session.run(code)
        assert output.exit_code == 0, output.stderr
        assert output.stdout.splitlines() == ["allowed", "denied"]


@pytest.mark.integration
def test_native_network_none_blocks_listening_host_service(native_mxc: Callable[..., mxc.WindowsMxcSession]) -> None:
    """Native default denial is paired with a working explicit host-network positive control."""
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        listener.listen()
        port = listener.getsockname()[1]
        code = f"import socket;s=socket.socket();s.settimeout(1);print(s.connect_ex(('127.0.0.1',{port})))"
        with native_mxc() as session:
            output = session.run(code)
            assert output.exit_code == 0, output.stderr
            assert int(output.stdout.strip()) != 0
        config = WindowsMxcConfig(network="host")
        with native_mxc(sandbox_config=config) as session:
            output = session.run(code)
            assert output.exit_code == 0, output.stderr
            assert int(output.stdout.strip()) == 0


@pytest.mark.integration
def test_native_descendants_cannot_outlive_foreground(native_mxc: Callable[..., mxc.WindowsMxcSession]) -> None:
    """Job cleanup stops detached child process groups when the command returns."""
    child = "import pathlib,time;time.sleep(1);pathlib.Path('escaped.txt').write_text('escaped')"
    parent = (
        f"import subprocess,sys;subprocess.Popen([sys.executable,'-c',{child!r}],"
        "stdin=subprocess.DEVNULL,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL,"
        "creationflags=subprocess.CREATE_NEW_PROCESS_GROUP);print('spawned')"
    )
    with native_mxc() as session:
        output = session.run(parent)
        assert output.exit_code == 0, output.stderr
        assert output.stdout.strip() == "spawned"
        time.sleep(1.2)
        assert not (session.workspace / "escaped.txt").exists()


@pytest.mark.integration
def test_native_deadline_and_output_cap_close_session(native_mxc: Callable[..., mxc.WindowsMxcSession]) -> None:
    """Resource guard failures reap the helper and remove its private workspace."""
    with native_mxc() as session:
        workspace = session.workspace
        with pytest.raises(SandboxTimeoutError):
            session.run("import time;time.sleep(60)", timeout=0.5)
        assert not session.is_open
        assert not workspace.exists()
    config = WindowsMxcConfig(max_output_bytes=4096)
    with native_mxc(sandbox_config=config) as session:
        workspace = session.workspace
        with pytest.raises(SandboxError, match="output exceeded"):
            session.run("import sys;sys.stdout.write('x'*100000)")
        assert not workspace.exists()


@pytest.mark.integration
def test_native_close_cancels_active_command(native_mxc: Callable[..., mxc.WindowsMxcSession]) -> None:
    """Concurrent close waits for the supervised native process before deleting files."""
    ready = threading.Event()
    errors: list[BaseException] = []
    with native_mxc() as session:
        workspace = session.workspace

        def run() -> None:
            try:
                session.run("import time;print('ready',flush=True);time.sleep(60)", on_stdout=lambda _: ready.set())
            except BaseException as exc:  # noqa: BLE001
                errors.append(exc)

        worker = threading.Thread(target=run)
        worker.start()
        try:
            assert ready.wait(10)
            session.close()
            worker.join(10)
            assert not worker.is_alive()
            assert len(errors) == 1
            assert isinstance(errors[0], SandboxError)
            assert not workspace.exists()
        finally:
            session.close()
            worker.join(10)
