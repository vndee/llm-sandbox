"""Tests for Firecracker backend implementation."""

import os
import subprocess
import tempfile
import uuid
from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

import pytest

from llm_sandbox.const import SandboxBackend, SupportedLanguage
from llm_sandbox.exceptions import ContainerError, MissingDependencyError
from llm_sandbox.firecracker import FirecrackerContainerAPI, SandboxFirecrackerSession
from llm_sandbox.session import _check_dependency, create_session


class TestFirecrackerContainerAPI:
    """Test cases for FirecrackerContainerAPI."""

    def setup_method(self) -> None:
        """Set up test fixtures."""
        self.socket_path = "/tmp/test-firecracker.sock"
        self.api = FirecrackerContainerAPI(self.socket_path)

    @patch("llm_sandbox.firecracker.requests_unixsocket.Session")
    def test_make_request_get(self, mock_session_class: Mock) -> None:
        """Test making GET request to Firecracker API."""
        mock_session = Mock()
        mock_response = Mock()
        mock_response.status_code = 200
        mock_session.get.return_value = mock_response
        mock_session_class.return_value = mock_session

        response = self.api._make_request("GET", "/machine-config")

        assert response == mock_response
        mock_session.get.assert_called_once()

    @patch("llm_sandbox.firecracker.requests_unixsocket.Session")
    def test_make_request_put(self, mock_session_class: Mock) -> None:
        """Test making PUT request to Firecracker API."""
        mock_session = Mock()
        mock_response = Mock()
        mock_response.status_code = 204
        mock_session.put.return_value = mock_response
        mock_session_class.return_value = mock_session

        test_data = {"vcpu_count": 2}
        response = self.api._make_request("PUT", "/machine-config", test_data)

        assert response == mock_response
        mock_session.put.assert_called_once_with(
            f"http+unix://{self.socket_path.replace('/', '%2F')}/machine-config",
            json=test_data
        )

    def test_make_request_unsupported_method(self) -> None:
        """Test unsupported HTTP method raises error."""
        with pytest.raises(ValueError, match="Unsupported HTTP method: DELETE"):
            self.api._make_request("DELETE", "/test")

    @patch("llm_sandbox.firecracker.subprocess.run")
    def test_execute_command(self, mock_run: Mock) -> None:
        """Test command execution via SSH."""
        mock_result = Mock()
        mock_result.returncode = 0
        mock_result.stdout = "test output"
        mock_result.stderr = ""
        mock_run.return_value = mock_result

        exit_code, output = self.api.execute_command("test-vm", "echo hello")

        assert exit_code == 0
        assert output == ("test output", "")
        mock_run.assert_called_once()

    @patch("llm_sandbox.firecracker.subprocess.run")
    def test_execute_command_timeout(self, mock_run: Mock) -> None:
        """Test command execution timeout."""
        mock_run.side_effect = subprocess.TimeoutExpired("ssh", 30)

        exit_code, output = self.api.execute_command("test-vm", "sleep 60")

        assert exit_code == 1
        assert output == ("", "Command timed out")

    @patch("llm_sandbox.firecracker.subprocess.run")
    def test_copy_to_container(self, mock_run: Mock) -> None:
        """Test copying file to container via SCP."""
        mock_result = Mock()
        mock_result.returncode = 0
        mock_run.return_value = mock_result

        self.api.copy_to_container("test-vm", "/local/file", "/remote/file")

        mock_run.assert_called_once()
        args = mock_run.call_args[0][0]
        assert "scp" in args
        assert "/local/file" in args
        assert "root@test-vm:/remote/file" in args

    @patch("llm_sandbox.firecracker.subprocess.run")
    def test_copy_to_container_failure(self, mock_run: Mock) -> None:
        """Test copy failure raises ContainerError."""
        mock_result = Mock()
        mock_result.returncode = 1
        mock_result.stderr = b"Permission denied"
        mock_run.return_value = mock_result

        with pytest.raises(ContainerError, match="Failed to copy file"):
            self.api.copy_to_container("test-vm", "/local/file", "/remote/file")


class TestSandboxFirecrackerSession:
    """Test cases for SandboxFirecrackerSession."""

    def setup_method(self) -> None:
        """Set up test fixtures."""
        self.session = SandboxFirecrackerSession(
            lang="python",
            verbose=False,
            vcpu_count=1,
            mem_size_mib=128
        )

    def test_initialization(self) -> None:
        """Test session initialization."""
        assert self.session.config.lang == SupportedLanguage.PYTHON
        assert self.session.vcpu_count == 1
        assert self.session.mem_size_mib == 128
        assert self.session.kernel_image_path == "/usr/share/firecracker/vmlinux"
        assert "sandbox-python-" in self.session.vm_id
        assert self.session.socket_path.startswith("/tmp/firecracker-")

    def test_get_default_rootfs_path(self) -> None:
        """Test default rootfs path generation."""
        path = self.session._get_default_rootfs_path()
        assert path == "/usr/share/firecracker/rootfs-python.ext4"

    @patch("llm_sandbox.firecracker.os.path.exists")
    @patch("llm_sandbox.firecracker.os.unlink")
    @patch("llm_sandbox.firecracker.subprocess.Popen")
    @patch.object(SandboxFirecrackerSession, "_wait_for_socket")
    def test_start_firecracker_process(
        self,
        mock_wait_socket: Mock,
        mock_popen: Mock,
        mock_unlink: Mock,
        mock_exists: Mock
    ) -> None:
        """Test starting Firecracker process."""
        mock_exists.return_value = True
        mock_process = Mock()
        mock_process.stdin = Mock()
        mock_popen.return_value = mock_process

        self.session._start_firecracker_process()

        mock_unlink.assert_called_once_with(self.session.socket_path)
        mock_popen.assert_called_once()
        mock_wait_socket.assert_called_once()

    @patch("llm_sandbox.firecracker.os.path.exists")
    def test_connect_to_existing_container_not_found(self, mock_exists: Mock) -> None:
        """Test connecting to non-existent container."""
        mock_exists.return_value = False

        with pytest.raises(ContainerError, match="Firecracker socket for .* not found"):
            self.session._connect_to_existing_container("non-existent-vm")

    @patch("llm_sandbox.firecracker.os.path.exists")
    def test_connect_to_existing_container_success(self, mock_exists: Mock) -> None:
        """Test successful connection to existing container."""
        mock_exists.return_value = True
        container_id = "existing-vm"

        self.session._connect_to_existing_container(container_id)

        assert self.session.vm_id == container_id
        assert self.session.container == container_id
        assert self.session.socket_path == f"/tmp/firecracker-{container_id}.sock"

    def test_setup_networking(self) -> None:
        """Test basic networking setup."""
        self.session._setup_networking()
        assert self.session.vm_ip == "172.16.0.2"

    @patch.object(SandboxFirecrackerSession, "_start_firecracker_process")
    @patch.object(SandboxFirecrackerSession, "_setup_networking")
    @patch.object(SandboxFirecrackerSession, "environment_setup")
    def test_open_new_session(
        self,
        mock_env_setup: Mock,
        mock_setup_networking: Mock,
        mock_start_process: Mock
    ) -> None:
        """Test opening new Firecracker session."""
        with patch.object(self.session.container_api, "create_container") as mock_create, \
             patch.object(self.session.container_api, "start_container") as mock_start:
            
            mock_create.return_value = "test-vm"
            
            self.session.open()

            mock_start_process.assert_called_once()
            mock_setup_networking.assert_called_once()
            mock_create.assert_called_once()
            mock_start.assert_called_once_with("test-vm")
            mock_env_setup.assert_called_once()
            assert self.session.is_open is True

    @patch.object(SandboxFirecrackerSession, "_connect_to_existing_container")
    def test_open_existing_session(self, mock_connect: Mock) -> None:
        """Test opening existing Firecracker session."""
        self.session.config.container_id = "existing-vm"
        self.session.using_existing_container = True

        self.session.open()

        mock_connect.assert_called_once_with("existing-vm")
        assert self.session.is_open is True

    @patch("llm_sandbox.firecracker.os.path.exists")
    @patch("llm_sandbox.firecracker.os.unlink")
    def test_close_session(self, mock_unlink: Mock, mock_exists: Mock) -> None:
        """Test closing Firecracker session."""
        mock_exists.return_value = True
        self.session.container = "test-vm"
        self.session.firecracker_process = Mock()
        self.session.is_open = True

        with patch.object(self.session.container_api, "stop_container") as mock_stop:
            self.session.close()

            mock_stop.assert_called_once_with("test-vm")
            self.session.firecracker_process.terminate.assert_called_once()
            mock_unlink.assert_called_once_with(self.session.socket_path)
            assert self.session.is_open is False
            assert self.session.container is None


class TestFirecrackerIntegration:
    """Test Firecracker integration with main session factory."""

    def test_check_dependency_missing(self) -> None:
        """Test dependency check fails when requests-unixsocket not available."""
        with patch("llm_sandbox.session.find_spec", return_value=None):
            with pytest.raises(MissingDependencyError, match="Firecracker backend requires"):
                _check_dependency(SandboxBackend.FIRECRACKER)

    def test_check_dependency_available(self) -> None:
        """Test dependency check passes when requests-unixsocket is available."""
        with patch("llm_sandbox.session.find_spec", return_value=Mock()):
            # Should not raise any exception
            _check_dependency(SandboxBackend.FIRECRACKER)

    @patch("llm_sandbox.session.find_spec")
    def test_create_firecracker_session(self, mock_find_spec: Mock) -> None:
        """Test creating Firecracker session via factory."""
        mock_find_spec.return_value = Mock()  # Mock requests_unixsocket availability

        session = create_session(
            backend=SandboxBackend.FIRECRACKER,
            lang="python",
            verbose=True
        )

        assert isinstance(session, SandboxFirecrackerSession)
        assert session.config.lang == SupportedLanguage.PYTHON
        assert session.verbose is True


@pytest.fixture
def temp_socket() -> Path:
    """Create temporary socket path for testing."""
    with tempfile.NamedTemporaryFile(delete=False) as f:
        socket_path = Path(f.name)
    
    yield socket_path
    
    # Cleanup
    if socket_path.exists():
        socket_path.unlink()


@pytest.mark.integration
class TestFirecrackerEndToEnd:
    """End-to-end integration tests for Firecracker backend."""

    @pytest.mark.skipif(
        not Path("/usr/bin/firecracker").exists(),
        reason="Firecracker binary not available"
    )
    def test_firecracker_session_lifecycle(self) -> None:
        """Test complete Firecracker session lifecycle."""
        # This test would require actual Firecracker installation
        # and proper kernel/rootfs images to run
        pytest.skip("Requires Firecracker installation and rootfs images")

    def test_firecracker_config_generation(self) -> None:
        """Test Firecracker configuration generation."""
        session = SandboxFirecrackerSession(
            lang="python",
            vcpu_count=2,
            mem_size_mib=256,
            kernel_image_path="/custom/kernel",
            rootfs_path="/custom/rootfs.ext4"
        )

        assert session.vcpu_count == 2
        assert session.mem_size_mib == 256
        assert session.kernel_image_path == "/custom/kernel"
        assert session.rootfs_path == "/custom/rootfs.ext4"