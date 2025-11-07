"""Tests for the InteractiveSandboxSession."""

from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from llm_sandbox.const import SandboxBackend
from llm_sandbox.core.config import SessionConfig
from llm_sandbox.data import ConsoleOutput
from llm_sandbox.exceptions import LanguageNotSupportedError, UnsupportedBackendError
from llm_sandbox.interactive import InteractiveSandboxSession, KernelType


def _fake_docker_session_init(self: InteractiveSandboxSession, *args: Any, **kwargs: Any) -> None:
    """Stub for SandboxDockerSession.__init__ that avoids Docker dependency in unit tests."""
    config = SessionConfig()
    self.config = config
    self.container = MagicMock()
    self.is_open = True
    self.using_existing_container = False
    self.verbose = False


def test_interactive_session_rejects_non_docker_backend() -> None:
    """Interactive session only supports Docker backend."""
    with pytest.raises(UnsupportedBackendError):
        InteractiveSandboxSession(backend=SandboxBackend.KUBERNETES)


def test_interactive_session_requires_python_language() -> None:
    """Interactive session currently only supports Python."""
    with pytest.raises(LanguageNotSupportedError):
        InteractiveSandboxSession(lang="javascript")


@patch("llm_sandbox.interactive.SandboxDockerSession.__init__", side_effect=_fake_docker_session_init)
def test_run_uses_persistent_executor(mock_super: MagicMock) -> None:
    """Interactive run should invoke runner script with context persistence."""
    session = InteractiveSandboxSession(kernel_type=KernelType.STANDARD)

    session.settings.timeout = None
    session.copy_to_runtime = MagicMock()
    session.install = MagicMock()
    session._check_session_timeout = MagicMock()
    session._handle_timeout = MagicMock()
    session._runtime_ready = True

    runner_output = ConsoleOutput(exit_code=0, stdout="ok", stderr="")
    cleanup_output = ConsoleOutput(exit_code=0, stdout="", stderr="")

    session.execute_command = MagicMock(side_effect=[runner_output, cleanup_output])
    session._execute_with_timeout = lambda func, timeout=None: func()

    result = session.run("x = 1")

    assert result == runner_output

    invoked_command = session.execute_command.call_args_list[0].args[0]
    assert "--context-file" in invoked_command
    assert "--history-file" in invoked_command
