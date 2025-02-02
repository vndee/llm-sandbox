"""Tests for main session functionality."""


import pytest
from unittest.mock import patch, MagicMock

from llm_sandbox import SandboxSession
from llm_sandbox.exceptions import SecurityError, ResourceError, ValidationError


@pytest.fixture
def mock_docker_client():
    with patch("docker.from_env") as mock_docker_from_env:
        mock_client = MagicMock()
        mock_docker_from_env.return_value = mock_client
        yield mock_client


@pytest.fixture
def session(mock_docker_client):
    return SandboxSession(
        backend="docker",
        image="python:3.9.19-bullseye",
        lang="python",
        keep_template=False,
        verbose=False,
        strict_security=True,
    )


def test_init_with_invalid_lang():
    with pytest.raises(ValueError):
        SandboxSession(lang="invalid_language")


def test_init_with_both_image_and_dockerfile():
    with pytest.raises(ValueError):
        SandboxSession(image="some_image", dockerfile="some_dockerfile")


def test_security_scanning_safe_code(session):
    safe_code = """
def calculate_sum(a, b):
    return a + b

print(calculate_sum(1, 2))
"""
    session.container = MagicMock()
    session.execute_command = MagicMock(return_value=MagicMock(text="3", exit_code=0))

    result = session.run(safe_code)
    assert result.output == "3"
    assert result.exit_code == 0


def test_security_scanning_unsafe_code(session):
    unsafe_code = """
import os
os.system('rm -rf /')
"""
    session.container = MagicMock()

    with pytest.raises(SecurityError) as exc_info:
        session.run(unsafe_code)

    assert "High severity security issues found" in str(exc_info.value)


def test_resource_monitoring(session):
    session.container = MagicMock()

    # Mock container stats
    mock_stats = {
        "cpu_stats": {
            "cpu_usage": {"total_usage": 100000},
            "system_cpu_usage": 1000000,
        },
        "precpu_stats": {
            "cpu_usage": {"total_usage": 90000},
            "system_cpu_usage": 900000,
        },
        "memory_stats": {
            "usage": 100 * 1024 * 1024,  # 100MB
            "limit": 512 * 1024 * 1024,  # 512MB
        },
        "networks": {"eth0": {"rx_bytes": 1000000, "tx_bytes": 500000}},
    }

    session.container.stats.return_value = mock_stats
    session._setup_monitoring()

    code = "print('Hello, World!')"
    session.execute_command = MagicMock(
        return_value=MagicMock(text="Hello, World!", exit_code=0)
    )

    result = session.run(code)

    assert "cpu_percent" in result.resource_usage
    assert "memory_mb" in result.resource_usage
    assert "duration_seconds" in result.resource_usage


def test_resource_limits_exceeded(session):
    session.container = MagicMock()

    # Mock container stats that exceed limits
    mock_stats = {
        "cpu_stats": {
            "cpu_usage": {"total_usage": 1000000},
            "system_cpu_usage": 1000000,
        },
        "precpu_stats": {"cpu_usage": {"total_usage": 0}, "system_cpu_usage": 0},
        "memory_stats": {
            "usage": 1024 * 1024 * 1024,  # 1GB (exceeds limit)
            "limit": 2 * 1024 * 1024 * 1024,
        },
        "networks": {"eth0": {"rx_bytes": 0, "tx_bytes": 0}},
    }

    session.container.stats.return_value = mock_stats
    session._setup_monitoring()

    code = "print('Hello, World!')"
    with pytest.raises(ResourceError) as exc_info:
        session.run(code)

    assert "Memory usage exceeded" in str(exc_info.value)


@pytest.mark.parametrize(
    "backend,use_k8s,use_podman",
    [("docker", False, False), ("kubernetes", True, False), ("podman", False, True)],
)
def test_factory_creation(backend, use_k8s, use_podman):
    if backend == "kubernetes":
        with patch("kubernetes.client.CoreV1Api"):
            session = SandboxSession(
                lang="python", use_kubernetes=use_k8s, use_podman=use_podman
            )
    else:
        session = SandboxSession(
            lang="python", use_kubernetes=use_k8s, use_podman=use_podman
        )
    assert session is not None

    def test_execute_failing_command(self):
        mock_container = MagicMock()
        self.session.container = mock_container

        command = "exit 1"
        mock_container.exec_run.return_value = (1, iter([]))

        output = self.session.execute_command(command)
        mock_container.exec_run.assert_called_with(command, stream=True, tty=True)
        self.assertEqual(output.exit_code, 1)
        self.assertEqual(output.text, "")


def test_factory_invalid_backend():
    with patch.object(SandboxSession, "__new__") as mock_new:
        mock_new.side_effect = ValidationError("Invalid backend")
        with pytest.raises(ValidationError):
            SandboxSession(
                lang="python", use_kubernetes=True, use_podman=True  # Can't use both
            )


def test_open_with_image(session, mock_docker_client):
    mock_docker_client.images.get.return_value = MagicMock(
        tags=["python:3.9.19-bullseye"]
    )
    mock_docker_client.containers.run.return_value = MagicMock()

    session.open()
    mock_docker_client.containers.run.assert_called_once()
    assert session.container is not None


def test_close(session):
    mock_container = MagicMock()
    session.container = mock_container
    mock_container.commit.return_values = MagicMock(tags=["python:3.9.19-bullseye"])

    session.close()
    mock_container.remove.assert_called_once()
    assert session.container is None


def test_run_without_open(session):
    with pytest.raises(RuntimeError):
        session.run("print('Hello')")


def test_copy_to_runtime(session, tmp_path):
    session.container = MagicMock()
    src = tmp_path / "test.txt"
    dest = "/tmp/test.txt"

    src.write_text("test content")
    session.copy_to_runtime(str(src), dest)
    session.container.put_archive.assert_called()


@pytest.mark.parametrize(
    "command,expected_output",
    [
        ("echo 'Hello'", "Hello\n"),
        ("python --version", "Python 3.9.19\n"),
    ],
)
def test_execute_command(session, command, expected_output):
    mock_container = MagicMock()
    session.container = mock_container

    mock_container.exec_run.return_value = (0, iter([expected_output.encode()]))

    output = session.execute_command(command)
    mock_container.exec_run.assert_called_with(command, stream=True, tty=True)
    assert output.text == expected_output


def test_execute_empty_command(session):
    with pytest.raises(ValueError):
        session.execute_command("")
