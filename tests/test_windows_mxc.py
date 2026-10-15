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
from pathlib import PureWindowsPath
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any
from unittest.mock import Mock

import pytest
from pydantic import ValidationError

from llm_sandbox import windows_mxc as mxc
from llm_sandbox.data import ConsoleOutput
from llm_sandbox.exceptions import (
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
    from pathlib import Path


@pytest.fixture
def adapter(monkeypatch: pytest.MonkeyPatch) -> Iterator[SimpleNamespace]:
    """Mock native boundaries while preserving Windows policy and request construction."""
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
