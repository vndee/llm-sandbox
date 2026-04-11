"""Tests for ContainerAPI protocol runtime checks."""
from typing import Any

from llm_sandbox.core.mixins import ContainerAPI


class MockRuntime:
    """Mock runtime that implements ContainerAPI."""

    def create_container(self, config: Any) -> Any:
        return {"config": config}

    def start_container(self, container: Any) -> None:
        pass

    def stop_container(self, container: Any) -> None:
        pass

    def execute_command(self, container: Any, command: str, **kwargs: Any) -> tuple[int, Any]:
        return 0, ("stdout", "stderr")

    def copy_to_container(self, container: Any, src: str, dest: str, **kwargs: Any) -> None:
        pass

    def copy_from_container(self, container: Any, src: str, **kwargs: Any) -> tuple[bytes, dict]:
        return b"data", {"size": 4}


class IncompleteRuntime:
    """Mock runtime missing methods."""

    def create_container(self, config: Any) -> Any:
        return {"config": config}


def test_mock_runtime_satisfies_protocol():
    """MockRuntime satisfies ContainerAPI protocol."""
    runtime: ContainerAPI = MockRuntime()
    assert isinstance(runtime, ContainerAPI)


def test_incomplete_runtime_does_not_satisfy_protocol():
    """IncompleteRuntime does not satisfy ContainerAPI."""
    runtime = IncompleteRuntime()
    assert not isinstance(runtime, ContainerAPI)
