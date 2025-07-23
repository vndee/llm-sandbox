import json
import os
import socket
import subprocess
import tempfile
import time
import uuid
from pathlib import Path
from typing import Any

import requests

from llm_sandbox.const import SupportedLanguage
from llm_sandbox.core.config import SessionConfig
from llm_sandbox.core.session_base import BaseSession
from llm_sandbox.exceptions import ContainerError, NotOpenSessionError
from llm_sandbox.security import SecurityPolicy

FIRECRACKER_BINARY = "firecracker"
FIRECRACKER_STARTUP_TIMEOUT = 60  # 1 minute
FIRECRACKER_API_POLL_INTERVAL = 1


class FirecrackerContainerAPI:
    """Firecracker implementation of the ContainerAPI protocol."""

    def __init__(self, socket_path: str, verbose: bool = False) -> None:
        """Initialize Firecracker container API."""
        self.socket_path = socket_path
        self.verbose = verbose
        self.base_url = f"http+unix://{socket_path.replace('/', '%2F')}"

    def _make_request(self, method: str, endpoint: str, data: dict | None = None) -> requests.Response:
        """Make HTTP request to Firecracker API."""
        import requests_unixsocket

        session = requests_unixsocket.Session()
        url = f"http+unix://{self.socket_path.replace('/', '%2F')}{endpoint}"

        if method.upper() == "GET":
            return session.get(url)
        if method.upper() == "PUT":
            return session.put(url, json=data)
        if method.upper() == "PATCH":
            return session.patch(url, json=data)
        msg = f"Unsupported HTTP method: {method}"
        raise ValueError(msg)

    def create_container(self, config: Any) -> Any:
        """Create Firecracker microVM."""
        vm_config = config["vm_config"]

        # Configure boot source
        response = self._make_request("PUT", "/boot-source", vm_config["boot_source"])
        if response.status_code not in (200, 204):
            msg = f"Failed to configure boot source: {response.text}"
            raise ContainerError(msg)

        # Configure drives
        for drive in vm_config.get("drives", []):
            response = self._make_request("PUT", f"/drives/{drive['drive_id']}", drive)
            if response.status_code not in (200, 204):
                msg = f"Failed to configure drive: {response.text}"
                raise ContainerError(msg)

        # Configure machine config
        response = self._make_request("PUT", "/machine-config", vm_config["machine_config"])
        if response.status_code not in (200, 204):
            msg = f"Failed to configure machine: {response.text}"
            raise ContainerError(msg)

        # Configure network interfaces
        for network in vm_config.get("network_interfaces", []):
            response = self._make_request("PUT", f"/network-interfaces/{network['iface_id']}", network)
            if response.status_code not in (200, 204):
                msg = f"Failed to configure network: {response.text}"
                raise ContainerError(msg)

        return vm_config["vm_id"]

    def start_container(self, container: Any) -> None:
        """Start Firecracker microVM."""
        response = self._make_request("PUT", "/actions", {"action_type": "InstanceStart"})
        if response.status_code not in (200, 204):
            msg = f"Failed to start microVM: {response.text}"
            raise ContainerError(msg)

    def stop_container(self, container: Any) -> None:
        """Stop Firecracker microVM."""
        # Send shutdown signal
        try:
            response = self._make_request("PUT", "/actions", {"action_type": "SendCtrlAltDel"})
            if response.status_code not in (200, 204):
                # Force shutdown if graceful shutdown fails
                self._make_request("PUT", "/actions", {"action_type": "InstanceShutdown"})
        except Exception:
            # Ignore errors during shutdown
            pass

    def execute_command(self, container: Any, command: str, **kwargs: Any) -> tuple[int, Any]:
        """Execute command in Firecracker microVM via SSH."""
        workdir = kwargs.get("workdir", "/")

        # For now, we'll use a simple approach with SSH
        # In a real implementation, you might want to use a more sophisticated method
        ssh_command = [
            "ssh",
            "-i",
            "/tmp/firecracker_key",
            "-o",
            "StrictHostKeyChecking=no",
            "-o",
            "UserKnownHostsFile=/dev/null",
            f"root@{container}",
            f"cd {workdir} && {command}",
        ]

        try:
            result = subprocess.run(
                ssh_command, capture_output=True, text=True, timeout=kwargs.get("timeout", 30), check=False
            )
            return result.returncode, (result.stdout, result.stderr)
        except subprocess.TimeoutExpired:
            return 1, ("", "Command timed out")

    def copy_to_container(self, container: Any, src: str, dest: str) -> None:
        """Copy file to Firecracker microVM via SCP."""
        scp_command = [
            "scp",
            "-i",
            "/tmp/firecracker_key",
            "-o",
            "StrictHostKeyChecking=no",
            "-o",
            "UserKnownHostsFile=/dev/null",
            src,
            f"root@{container}:{dest}",
        ]

        result = subprocess.run(scp_command, capture_output=True, check=False)
        if result.returncode != 0:
            msg = f"Failed to copy file: {result.stderr.decode()}"
            raise ContainerError(msg)

    def copy_from_container(self, container: Any, src: str) -> tuple[bytes, dict]:
        """Copy file from Firecracker microVM via SCP."""
        # Create temporary file for download
        with tempfile.NamedTemporaryFile(delete=False) as temp_file:
            temp_path = temp_file.name

        try:
            scp_command = [
                "scp",
                "-i",
                "/tmp/firecracker_key",
                "-o",
                "StrictHostKeyChecking=no",
                "-o",
                "UserKnownHostsFile=/dev/null",
                f"root@{container}:{src}",
                temp_path,
            ]

            result = subprocess.run(scp_command, capture_output=True, check=False)
            if result.returncode != 0:
                return b"", {"size": 0}

            # Read the downloaded file
            with open(temp_path, "rb") as f:
                data = f.read()

            # Get file stats
            stat_info = os.stat(temp_path)
            stat_dict = {
                "name": Path(src).name,
                "size": stat_info.st_size,
                "mtime": int(stat_info.st_mtime),
                "mode": stat_info.st_mode,
                "linkTarget": "",
            }

            return data, stat_dict
        finally:
            # Clean up temp file
            if os.path.exists(temp_path):
                os.unlink(temp_path)


class SandboxFirecrackerSession(BaseSession):
    """Sandbox session implemented using Firecracker microVMs.

    This class provides a sandboxed environment for code execution by leveraging Firecracker.
    It handles microVM creation and lifecycle, code execution, library installation, and
    file operations within the Firecracker microVM.
    """

    def __init__(
        self,
        image: str | None = None,
        kernel_image_path: str | None = None,
        rootfs_path: str | None = None,
        lang: str = SupportedLanguage.PYTHON,
        verbose: bool = False,
        vcpu_count: int = 1,
        mem_size_mib: int = 128,
        workdir: str = "/sandbox",
        security_policy: SecurityPolicy | None = None,
        default_timeout: float | None = None,
        execution_timeout: float | None = None,
        session_timeout: float | None = None,
        container_id: str | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialize Firecracker session.

        Args:
            image (str | None): The rootfs image to use.
            kernel_image_path (str | None): Path to the kernel image.
            rootfs_path (str | None): Path to the root filesystem image.
            lang (str): The language to use.
            verbose (bool): Whether to enable verbose output.
            vcpu_count (int): Number of virtual CPUs.
            mem_size_mib (int): Memory size in MiB.
            workdir (str): The working directory to use.
            security_policy (SecurityPolicy | None): The security policy to use.
            default_timeout (float | None): The default timeout to use.
            execution_timeout (float | None): The execution timeout to use.
            session_timeout (float | None): The session timeout to use.
            container_id (str | None): ID of existing microVM to connect to.
            **kwargs: Additional keyword arguments.

        """
        config = SessionConfig(
            image=image,
            lang=SupportedLanguage(lang.upper()),
            verbose=verbose,
            workdir=workdir,
            security_policy=security_policy,
            default_timeout=default_timeout,
            execution_timeout=execution_timeout,
            session_timeout=session_timeout,
            container_id=container_id,
        )

        super().__init__(config=config, **kwargs)

        # Firecracker-specific attributes
        self.kernel_image_path = kernel_image_path or "/usr/share/firecracker/vmlinux"
        self.rootfs_path = rootfs_path or self._get_default_rootfs_path()
        self.vcpu_count = vcpu_count
        self.mem_size_mib = mem_size_mib

        # Generate unique identifiers
        self.vm_id = f"sandbox-{lang.lower()}-{uuid.uuid4().hex[:8]}"
        self.socket_path = f"/tmp/firecracker-{self.vm_id}.sock"

        # Firecracker process and networking
        self.firecracker_process: subprocess.Popen | None = None
        self.vm_ip: str | None = None

        # Initialize container API
        self.container_api = FirecrackerContainerAPI(self.socket_path, verbose)

    def _get_default_rootfs_path(self) -> str:
        """Get default rootfs path based on language."""
        # In a real implementation, you would have pre-built rootfs images
        # For now, we'll use a generic path
        return f"/usr/share/firecracker/rootfs-{self.config.lang.lower()}.ext4"

    def _start_firecracker_process(self) -> None:
        """Start the Firecracker process."""
        # Remove socket if it exists
        if os.path.exists(self.socket_path):
            os.unlink(self.socket_path)

        cmd = [FIRECRACKER_BINARY, "--api-sock", self.socket_path, "--config-file", "/dev/stdin"]

        # Basic Firecracker configuration
        config = {
            "boot-source": {
                "kernel_image_path": self.kernel_image_path,
                "boot_args": "console=ttyS0 reboot=k panic=1 pci=off",
            },
            "drives": [
                {"drive_id": "rootfs", "path_on_host": self.rootfs_path, "is_root_device": True, "is_read_only": False}
            ],
            "machine-config": {"vcpu_count": self.vcpu_count, "mem_size_mib": self.mem_size_mib},
        }

        try:
            self.firecracker_process = subprocess.Popen(
                cmd,
                stdin=subprocess.PIPE,
                stdout=subprocess.DEVNULL if not self.verbose else None,
                stderr=subprocess.PIPE,
                text=True,
            )

            # Send configuration
            if self.firecracker_process.stdin:
                json.dump(config, self.firecracker_process.stdin)
                self.firecracker_process.stdin.close()

            # Wait for socket to be available
            self._wait_for_socket()

        except Exception as e:
            msg = f"Failed to start Firecracker process: {e}"
            self._log(msg, "error")
            raise ContainerError(msg) from e

    def _wait_for_socket(self) -> None:
        """Wait for Firecracker socket to become available."""
        start_time = time.time()
        while time.time() - start_time < FIRECRACKER_STARTUP_TIMEOUT:
            if os.path.exists(self.socket_path):
                # Try to connect to verify it's ready
                try:
                    sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
                    sock.connect(self.socket_path)
                    sock.close()
                    return
                except (OSError, ConnectionRefusedError):
                    pass
            time.sleep(FIRECRACKER_API_POLL_INTERVAL)

        msg = f"Firecracker socket not available after {FIRECRACKER_STARTUP_TIMEOUT} seconds"
        raise ContainerError(msg)

    def _setup_networking(self) -> None:
        """Set up basic networking for the microVM."""
        # This is a simplified networking setup
        # In a real implementation, you would set up proper TAP interfaces
        self.vm_ip = "172.16.0.2"  # Static IP for simplicity

    def _connect_to_existing_container(self, container_id: str) -> None:
        """Connect to an existing Firecracker microVM.

        Args:
            container_id (str): The ID of the existing microVM to connect to.

        Raises:
            ContainerError: If the microVM cannot be found or accessed.

        """
        # For existing microVMs, we would need to discover the socket path
        # and connect to it. This is a simplified implementation.
        self.socket_path = f"/tmp/firecracker-{container_id}.sock"
        self.vm_id = container_id

        if not os.path.exists(self.socket_path):
            msg = f"Firecracker socket for {container_id} not found"
            raise ContainerError(msg)

        # Update container API with new socket path
        self.container_api = FirecrackerContainerAPI(self.socket_path, self.verbose)
        self.container = container_id
        self._log(f"Connected to existing microVM {container_id}")

    def _ensure_directory_exists(self, path: str) -> None:
        """Ensure directory exists in microVM."""
        mkdir_result = self.container_api.execute_command(self.container, f"mkdir -p '{path}'")
        if mkdir_result[0] != 0:
            stdout_output, stderr_output = mkdir_result[1]
            error_msg = stderr_output if stderr_output else stdout_output
            self._log(f"Failed to create directory {path}: {error_msg}", "error")

    def _ensure_ownership(self, paths: list[str]) -> None:
        """Ensure ownership of paths in microVM."""
        # Since we're typically running as root in microVMs, this is simplified
        for path in paths:
            self.container_api.execute_command(self.container, f"chown -R root:root {path}")

    def _process_non_stream_output(self, output: Any) -> tuple[str, str]:
        """Process non-streaming output."""
        if isinstance(output, tuple) and len(output) == 2:
            stdout_data, stderr_data = output
            return str(stdout_data), str(stderr_data)
        return "", ""

    def _process_stream_output(self, output: Any) -> tuple[str, str]:
        """Process streaming output (not used but required by mixin)."""
        return self._process_non_stream_output(output)

    def _handle_timeout(self) -> None:
        """Handle timeout cleanup."""
        if self.using_existing_container:
            try:
                self.close()
            except Exception as e:
                self._log(f"Error during timeout cleanup: {e}", "error")

    def open(self) -> None:
        """Open Firecracker session."""
        super().open()

        if self.using_existing_container and self.config.container_id:
            # Connect to existing microVM
            self._connect_to_existing_container(self.config.container_id)
        else:
            # Create new microVM
            self._start_firecracker_process()
            self._setup_networking()

            # Configure and start the microVM
            vm_config = {
                "vm_id": self.vm_id,
                "boot_source": {
                    "kernel_image_path": self.kernel_image_path,
                    "boot_args": "console=ttyS0 reboot=k panic=1 pci=off",
                },
                "drives": [
                    {
                        "drive_id": "rootfs",
                        "path_on_host": self.rootfs_path,
                        "is_root_device": True,
                        "is_read_only": False,
                    }
                ],
                "machine_config": {"vcpu_count": self.vcpu_count, "mem_size_mib": self.mem_size_mib},
            }

            container_config = {"vm_config": vm_config}
            self.container = self.container_api.create_container(container_config)
            self.container_api.start_container(self.container)

        # Setup environment (skipped for existing microVMs)
        if not self.using_existing_container:
            self.environment_setup()

    def close(self) -> None:
        """Close Firecracker session."""
        super().close()

        if self.container:
            # Only stop microVM if we created it
            if not self.using_existing_container:
                try:
                    self.container_api.stop_container(self.container)
                    self._log("Stopped microVM")
                except Exception as e:
                    self._log(f"Error stopping microVM: {e}", "error")
            else:
                self._log("Disconnected from existing microVM")

            self.container = None

        # Clean up Firecracker process
        if self.firecracker_process and not self.using_existing_container:
            try:
                self.firecracker_process.terminate()
                self.firecracker_process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.firecracker_process.kill()
            except Exception as e:
                self._log(f"Error cleaning up Firecracker process: {e}", "error")
            finally:
                self.firecracker_process = None

        # Clean up socket
        if os.path.exists(self.socket_path) and not self.using_existing_container:
            try:
                os.unlink(self.socket_path)
            except Exception as e:
                self._log(f"Error removing socket: {e}", "error")

    def get_archive(self, path: str) -> tuple[bytes, dict]:
        """Get archive from microVM."""
        if not self.container:
            raise NotOpenSessionError

        return self.container_api.copy_from_container(self.container, path)
