# ruff: noqa: SLF001
"""Tests for the InteractiveSandboxSession."""

import json
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from llm_sandbox.const import SandboxBackend
from llm_sandbox.core.config import SessionConfig
from llm_sandbox.data import ConsoleOutput
from llm_sandbox.exceptions import LanguageNotSupportedError, SandboxTimeoutError, UnsupportedBackendError
from llm_sandbox.interactive import InteractiveSandboxSession, KernelType


def _stub_docker_session(self: InteractiveSandboxSession, *args: Any, **kwargs: Any) -> None:
    """Stub for SandboxDockerSession.__init__ to avoid Docker dependency."""
    del args, kwargs
    self.config = SessionConfig()
    self.container = MagicMock()
    self.is_open = True
    self.using_existing_container = False
    self.verbose = False


def test_interactive_session_requires_docker_backend() -> None:
    """Interactive session only supports Docker backend."""
    with pytest.raises(UnsupportedBackendError):
        InteractiveSandboxSession(backend=SandboxBackend.KUBERNETES)


def test_interactive_session_requires_python_language() -> None:
    """Interactive session currently only supports Python."""
    with pytest.raises(LanguageNotSupportedError):
        InteractiveSandboxSession(lang="javascript")


@patch("llm_sandbox.interactive.SandboxDockerSession.__init__", new=_stub_docker_session)
def test_run_executes_code_and_returns_output() -> None:
    """Run should deliver stdout/stderr from the persistent interpreter."""
    session = InteractiveSandboxSession(kernel_type=KernelType.IPYTHON)
    session._commands_dir = "/sandbox/.interactive/commands"
    session._results_dir = "/sandbox/.interactive/results"
    session._runner_ready = True
    session.settings.timeout = 2
    session.settings.poll_interval = 0.01

    session.install = MagicMock()
    session._check_session_timeout = MagicMock()

    def fake_copy_from_runtime(src: str, dest: str) -> None:
        req_id = Path(src).stem.split("-")[-1]
        payload = {"id": req_id, "success": True, "stdout": "ok", "stderr": ""}
        Path(dest).write_text(json.dumps(payload), encoding="utf-8")

    session.copy_to_runtime = MagicMock()
    session.copy_from_runtime = MagicMock(side_effect=fake_copy_from_runtime)

    def fake_execute_command(command: str, **_: Any) -> ConsoleOutput:
        if command.startswith("test -f"):
            return ConsoleOutput(exit_code=0, stdout="", stderr="")
        return ConsoleOutput(exit_code=0, stdout="", stderr="")

    session.execute_command = MagicMock(side_effect=fake_execute_command)

    result = session.run("value = 42")

    assert result.stdout == "ok"
    assert result.exit_code == 0
    session.copy_to_runtime.assert_called_once()
    assert session.copy_from_runtime.call_count == 1


@patch("llm_sandbox.interactive.SandboxDockerSession.__init__", new=_stub_docker_session)
def test_run_times_out_when_result_missing() -> None:
    """Run should raise SandboxTimeoutError when the result never arrives."""
    session = InteractiveSandboxSession(kernel_type=KernelType.IPYTHON, timeout=0.2)
    session._commands_dir = "/sandbox/.interactive/commands"
    session._results_dir = "/sandbox/.interactive/results"
    session._runner_ready = True
    session.settings.poll_interval = 0.01

    session.install = MagicMock()
    session._check_session_timeout = MagicMock()
    session.copy_to_runtime = MagicMock()
    session.copy_from_runtime = MagicMock()
    session._interrupt_runner = MagicMock()

    def fake_execute_command(command: str, **_: Any) -> ConsoleOutput:
        if command.startswith("test -f"):
            return ConsoleOutput(exit_code=1, stdout="", stderr="")
        return ConsoleOutput(exit_code=0, stdout="", stderr="")

    session.execute_command = MagicMock(side_effect=fake_execute_command)

    with pytest.raises(SandboxTimeoutError):
        session.run("value = 1")

    session._interrupt_runner.assert_called_once()
