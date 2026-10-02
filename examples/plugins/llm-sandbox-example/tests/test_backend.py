"""Tests specific to this backend, beyond what the compliance kit covers.

The compliance kit checks that you satisfy the interface. These check that your backend does
its own job correctly -- which only you can write.
"""

from pathlib import Path

import pytest
from llm_sandbox_example import ExampleBackend

from llm_sandbox import SandboxSession, list_backends
from llm_sandbox.backends import BackendCapability
from llm_sandbox.exceptions import BackendCapabilityError
from llm_sandbox.registry import get_backend


class TestRegistration:
    """The plugin is wired up correctly as an installed distribution."""

    def test_listed_as_an_installed_backend(self) -> None:
        """Listed as an installed backend."""
        info = next(item for item in list_backends() if item.name == "example")

        assert info.is_builtin is False
        assert info.distribution == "llm-sandbox-example"

    def test_resolves_to_this_class(self) -> None:
        """Resolves to this class."""
        assert get_backend("example") is ExampleBackend

    def test_declares_only_what_it_implements(self) -> None:
        """Declares only what it implements."""
        assert ExampleBackend.capabilities == frozenset({BackendCapability.ARTIFACTS})

        with pytest.raises(BackendCapabilityError):
            ExampleBackend.create_pool_manager()


class TestSessionBehaviour:
    """Behaviour particular to running code in a local directory."""

    def test_runs_code_and_captures_stdout(self) -> None:
        """Runs code and captures stdout."""
        with SandboxSession(backend="example", lang="python") as session:
            result = session.run("print(6 * 7)")

        assert result.exit_code == 0
        assert result.stdout.strip() == "42"

    def test_stderr_is_captured_separately(self) -> None:
        """Stderr is captured separately."""
        with SandboxSession(backend="example", lang="python") as session:
            result = session.run("import sys; sys.stderr.write('to-stderr')")

        assert "to-stderr" in result.stderr
        assert "to-stderr" not in result.stdout

    def test_streaming_callbacks_fire(self) -> None:
        """Streaming callbacks fire."""
        chunks: list[str] = []

        with SandboxSession(backend="example", lang="python") as session:
            session.run("print('streamed')", on_stdout=chunks.append)

        assert any("streamed" in chunk for chunk in chunks)

    def test_temporary_workdir_is_removed_on_close(self) -> None:
        """The backend owns the directory it created, so it must clean it up."""
        session = SandboxSession(backend="example", lang="python")
        workdir = Path(session.config.workdir)

        with session:
            assert workdir.exists()

        assert not workdir.exists(), "Session leaked its temporary working directory"

    def test_caller_supplied_workdir_is_left_alone(self, tmp_path: Path) -> None:
        """Caller supplied workdir is left alone."""
        workdir = tmp_path / "mine"
        workdir.mkdir()

        with SandboxSession(backend="example", lang="python", workdir=str(workdir)) as session:
            session.run("print('hello')")

        assert workdir.exists(), "Backend deleted a directory it did not create"
