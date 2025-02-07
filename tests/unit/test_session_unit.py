"""Unit tests for the SandboxSession class."""

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

class TestSessionInitialization:
    def test_init_with_valid_params(self):
        session = SandboxSession(lang="python")
        assert session.lang == "python"
        assert session.backend == "docker"  # default backend

    def test_init_with_invalid_lang(self):
        with pytest.raises(ValueError, match="Unsupported language"):
            SandboxSession(lang="invalid_language")

    def test_init_with_both_image_and_dockerfile(self):
        with pytest.raises(ValueError, match="Cannot specify both"):
            SandboxSession(image="some_image", dockerfile="some_dockerfile")

    def test_init_with_resource_limits(self):
        session = SandboxSession(
            lang="python",
            max_cpu_percent=50.0,
            max_memory_bytes=1024 * 1024 * 1024,
            max_disk_bytes=5 * 1024 * 1024 * 1024
        )
        assert session.max_cpu_percent == 50.0
        assert session.max_memory_bytes == 1024 * 1024 * 1024
        assert session.max_disk_bytes == 5 * 1024 * 1024 * 1024

class TestSessionSecurity:
    def test_security_scanning_safe_code(self, session):
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

    def test_security_scanning_unsafe_system_calls(self, session):
        unsafe_code = """
        import os
        os.system('rm -rf /')
        """
        session.container = MagicMock()
        with pytest.raises(SecurityError, match="High severity security issues found"):
            session.run(unsafe_code)

    def test_security_scanning_unsafe_imports(self, session):
        unsafe_code = """
        import socket
        s = socket.socket()
        """
        session.container = MagicMock()
        with pytest.raises(SecurityError, match="Unauthorized module"):
            session.run(unsafe_code)

    def test_security_scanning_file_operations(self, session):
        unsafe_code = """
        with open('/etc/passwd', 'r') as f:
            print(f.read())
        """
        session.container = MagicMock()
        with pytest.raises(SecurityError, match="Unauthorized file access"):
            session.run(unsafe_code)

class TestSessionResourceMonitoring:
    def test_resource_monitoring_normal_usage(self, session):
        session.container = MagicMock()
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

    def test_resource_limits_cpu_exceeded(self, session):
        session.container = MagicMock()
        mock_stats = {
            "cpu_stats": {
                "cpu_usage": {"total_usage": 1000000},
                "system_cpu_usage": 1000000,
            },
            "precpu_stats": {"cpu_usage": {"total_usage": 0}, "system_cpu_usage": 0},
            "memory_stats": {
                "usage": 100 * 1024 * 1024,
                "limit": 512 * 1024 * 1024,
            },
        }
        session.container.stats.return_value = mock_stats
        session._setup_monitoring()
        session.max_cpu_percent = 50.0

        with pytest.raises(ResourceError, match="CPU usage exceeded"):
            session.run("print('Hello')")

    def test_resource_limits_memory_exceeded(self, session):
        session.container = MagicMock()
        mock_stats = {
            "cpu_stats": {
                "cpu_usage": {"total_usage": 100000},
                "system_cpu_usage": 1000000,
            },
            "memory_stats": {
                "usage": 1024 * 1024 * 1024,  # 1GB
                "limit": 2 * 1024 * 1024 * 1024,
            },
        }
        session.container.stats.return_value = mock_stats
        session._setup_monitoring()
        session.max_memory_bytes = 512 * 1024 * 1024  # 512MB

        with pytest.raises(ResourceError, match="Memory usage exceeded"):
            session.run("print('Hello')")

class TestSessionFileOperations:
    def test_copy_to_runtime(self, session, tmp_path):
        session.container = MagicMock()
        src = tmp_path / "test.txt"
        dest = "/tmp/test.txt"

        src.write_text("test content")
        session.copy_to_runtime(str(src), dest)
        session.container.put_archive.assert_called_once()

    def test_copy_from_runtime(self, session, tmp_path):
        session.container = MagicMock()
        src = "/tmp/test.txt"
        dest = tmp_path / "test.txt"

        session.copy_from_runtime(src, str(dest))
        session.container.get_archive.assert_called_once()

    def test_copy_invalid_path(self, session):
        session.container = MagicMock()
        with pytest.raises(ValueError):
            session.copy_to_runtime("", "/tmp/test.txt")

class TestSessionCommands:
    @pytest.mark.parametrize(
        "command,expected_output",
        [
            ("echo 'Hello'", "Hello\n"),
            ("python --version", "Python 3.9.19\n"),
            ("ls -la", "total 0\n"),
        ],
    )
    def test_execute_command(self, session, command, expected_output):
        mock_container = MagicMock()
        session.container = mock_container
        mock_container.exec_run.return_value = (0, iter([expected_output.encode()]))

        output = session.execute_command(command)
        mock_container.exec_run.assert_called_with(command, stream=True, tty=True)
        assert output.text == expected_output
        assert output.exit_code == 0

    def test_execute_failing_command(self, session):
        mock_container = MagicMock()
        session.container = mock_container
        mock_container.exec_run.return_value = (1, iter([b"error message"]))

        output = session.execute_command("invalid_command")
        assert output.exit_code == 1
        assert "error message" in output.text

    def test_execute_empty_command(self, session):
        with pytest.raises(ValueError, match="Command cannot be empty"):
            session.execute_command("") 