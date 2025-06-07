# ruff: noqa: SLF001, PLR2004, ARG002, PT011

"""Tests for Kubernetes backend implementation."""

import io
import tarfile
import tempfile
from unittest.mock import MagicMock, Mock, patch

import pytest

from llm_sandbox.const import DefaultImage, SupportedLanguage
from llm_sandbox.data import ConsoleOutput
from llm_sandbox.exceptions import NotOpenSessionError
from llm_sandbox.kubernetes import SandboxKubernetesSession
from llm_sandbox.security import SecurityPolicy


class TestSandboxKubernetesSessionInit:
    """Test SandboxKubernetesSession initialization."""

    @patch("kubernetes.config.load_kube_config")
    @patch("llm_sandbox.kubernetes.CoreV1Api")
    @patch("llm_sandbox.language_handlers.factory.LanguageHandlerFactory.create_handler")
    def test_init_with_defaults(
        self, mock_create_handler: MagicMock, mock_core_v1_api: MagicMock, mock_load_config: MagicMock
    ) -> None:
        """Test initialization with default parameters."""
        mock_handler = MagicMock()
        mock_create_handler.return_value = mock_handler
        mock_client = MagicMock()
        mock_core_v1_api.return_value = mock_client

        session = SandboxKubernetesSession()

        assert session.config.lang == SupportedLanguage.PYTHON
        assert session.config.verbose is False
        assert session.config.image is None  # Image is set during pod manifest creation
        assert session.config.workdir == "/sandbox"
        assert session.kube_namespace == "default"
        assert session.client == mock_client
        assert session.env_vars is None
        mock_load_config.assert_called_once()

    @patch("llm_sandbox.language_handlers.factory.LanguageHandlerFactory.create_handler")
    def test_init_with_custom_client(self, mock_create_handler: MagicMock) -> None:
        """Test initialization with custom Kubernetes client."""
        mock_handler = MagicMock()
        mock_create_handler.return_value = mock_handler
        custom_client = MagicMock()

        session = SandboxKubernetesSession(client=custom_client)

        assert session.client == custom_client

    @patch("kubernetes.config.load_kube_config")
    @patch("llm_sandbox.kubernetes.CoreV1Api")
    @patch("llm_sandbox.language_handlers.factory.LanguageHandlerFactory.create_handler")
    def test_init_with_custom_params(
        self, mock_create_handler: MagicMock, mock_core_v1_api: MagicMock, mock_load_config: MagicMock
    ) -> None:
        """Test initialization with custom parameters."""
        mock_handler = MagicMock()
        mock_create_handler.return_value = mock_handler
        mock_client = MagicMock()
        mock_core_v1_api.return_value = mock_client
        security_policy = SecurityPolicy(patterns=[], restricted_modules=[])
        env_vars = {"MY_VAR": "value"}

        session = SandboxKubernetesSession(
            image="custom:latest",
            lang="java",
            verbose=True,
            kube_namespace="custom-ns",
            env_vars=env_vars,
            workdir="/custom",
            security_policy=security_policy,
        )

        assert session.config.image == "custom:latest"
        assert session.config.lang == SupportedLanguage.JAVA
        assert session.config.verbose is True
        assert session.kube_namespace == "custom-ns"
        assert session.env_vars == env_vars
        assert session.config.workdir == "/custom"
        assert session.config.security_policy == security_policy

    @patch("kubernetes.config.load_kube_config")
    @patch("llm_sandbox.kubernetes.CoreV1Api")
    @patch("llm_sandbox.language_handlers.factory.LanguageHandlerFactory.create_handler")
    def test_init_with_custom_pod_manifest(
        self, mock_create_handler: MagicMock, mock_core_v1_api: MagicMock, mock_load_config: MagicMock
    ) -> None:
        """Test initialization with custom pod manifest."""
        mock_handler = MagicMock()
        mock_create_handler.return_value = mock_handler

        custom_manifest = {
            "apiVersion": "v1",
            "kind": "Pod",
            "metadata": {"name": "custom-pod", "namespace": "custom-ns"},
            "spec": {"containers": [{"name": "test", "image": "test:latest"}]},
        }

        session = SandboxKubernetesSession(pod_manifest=custom_manifest)

        # Pod name is now always made unique, so we just check it was updated
        assert session.pod_name.startswith("sandbox-python-")
        assert session.kube_namespace == "custom-ns"
        # The manifest name should be updated to match the unique pod name
        assert session.pod_manifest["metadata"]["name"] == session.pod_name

    @patch("kubernetes.config.load_kube_config")
    @patch("llm_sandbox.kubernetes.CoreV1Api")
    @patch("llm_sandbox.language_handlers.factory.LanguageHandlerFactory.create_handler")
    def test_default_pod_manifest_generation(
        self, mock_create_handler: MagicMock, mock_core_v1_api: MagicMock, mock_load_config: MagicMock
    ) -> None:
        """Test default pod manifest generation."""
        mock_handler = MagicMock()
        mock_create_handler.return_value = mock_handler

        env_vars = {"TEST_VAR": "test_value"}
        session = SandboxKubernetesSession(env_vars=env_vars)

        manifest = session.pod_manifest
        assert manifest["apiVersion"] == "v1"
        assert manifest["kind"] == "Pod"
        assert manifest["metadata"]["namespace"] == "default"
        assert manifest["spec"]["containers"][0]["image"] == DefaultImage.PYTHON
        assert manifest["spec"]["containers"][0]["env"] == [{"name": "TEST_VAR", "value": "test_value"}]


class TestSandboxKubernetesSessionOpen:
    """Test SandboxKubernetesSession open functionality."""

    @patch("kubernetes.config.load_kube_config")
    @patch("llm_sandbox.kubernetes.CoreV1Api")
    @patch("llm_sandbox.language_handlers.factory.LanguageHandlerFactory.create_handler")
    @patch("time.sleep")  # Speed up the test
    def test_open_success(
        self,
        mock_sleep: MagicMock,
        mock_create_handler: MagicMock,
        mock_core_v1_api: MagicMock,
        mock_load_config: MagicMock,
    ) -> None:
        """Test successful pod creation and waiting."""
        mock_handler = MagicMock()
        mock_create_handler.return_value = mock_handler
        mock_client = MagicMock()
        mock_core_v1_api.return_value = mock_client

        # Mock pod status progression
        mock_pod_running = MagicMock()
        mock_pod_running.status.phase = "Running"
        mock_client.read_namespaced_pod.return_value = mock_pod_running

        session = SandboxKubernetesSession()

        with patch.object(session, "environment_setup") as mock_env_setup:
            session.open()

        mock_client.create_namespaced_pod.assert_called_once()
        mock_client.read_namespaced_pod.assert_called()
        mock_env_setup.assert_called_once()
        assert session.container == session.pod_name

    @patch("kubernetes.config.load_kube_config")
    @patch("llm_sandbox.kubernetes.CoreV1Api")
    @patch("llm_sandbox.language_handlers.factory.LanguageHandlerFactory.create_handler")
    @patch("time.sleep")
    def test_open_waiting_for_pod(
        self,
        mock_sleep: MagicMock,
        mock_create_handler: MagicMock,
        mock_core_v1_api: MagicMock,
        mock_load_config: MagicMock,
    ) -> None:
        """Test waiting for pod to become ready."""
        mock_handler = MagicMock()
        mock_create_handler.return_value = mock_handler
        mock_client = MagicMock()
        mock_core_v1_api.return_value = mock_client

        # Mock pod status progression: Pending -> Running
        mock_pod_pending = MagicMock()
        mock_pod_pending.status.phase = "Pending"
        mock_pod_running = MagicMock()
        mock_pod_running.status.phase = "Running"

        mock_client.read_namespaced_pod.side_effect = [
            mock_pod_pending,
            mock_pod_pending,
            mock_pod_running,
        ]

        session = SandboxKubernetesSession()

        with patch.object(session, "environment_setup"):
            session.open()

        assert mock_client.read_namespaced_pod.call_count == 3
        assert mock_sleep.call_count == 2


class TestSandboxKubernetesSessionClose:
    """Test SandboxKubernetesSession close functionality."""

    @patch("kubernetes.config.load_kube_config")
    @patch("llm_sandbox.kubernetes.CoreV1Api")
    @patch("llm_sandbox.kubernetes.k8s_client.V1DeleteOptions")
    @patch("llm_sandbox.language_handlers.factory.LanguageHandlerFactory.create_handler")
    def test_close_success(
        self,
        mock_create_handler: MagicMock,
        mock_delete_options: MagicMock,
        mock_core_v1_api: MagicMock,
        mock_load_config: MagicMock,
    ) -> None:
        """Test successful pod deletion."""
        mock_handler = MagicMock()
        mock_create_handler.return_value = mock_handler
        mock_client = MagicMock()
        mock_core_v1_api.return_value = mock_client
        mock_delete_opts = MagicMock()
        mock_delete_options.return_value = mock_delete_opts

        session = SandboxKubernetesSession()
        session.container = "test-pod"

        session.close()

        mock_client.delete_namespaced_pod.assert_called_once_with(
            name="test-pod",
            namespace=session.kube_namespace,
            body=mock_delete_opts,
        )


class TestSandboxKubernetesSessionRun:
    """Test SandboxKubernetesSession run functionality."""

    @patch("kubernetes.config.load_kube_config")
    @patch("llm_sandbox.kubernetes.CoreV1Api")
    @patch("llm_sandbox.language_handlers.factory.LanguageHandlerFactory.create_handler")
    def test_run_success(
        self,
        mock_create_handler: MagicMock,
        mock_core_v1_api: MagicMock,
        mock_load_config: MagicMock,
    ) -> None:
        """Test successful code execution."""
        mock_handler = MagicMock()
        mock_handler.file_extension = "py"
        mock_handler.get_execution_commands.return_value = ["python /sandbox/code.py"]
        mock_create_handler.return_value = mock_handler

        session = SandboxKubernetesSession()
        session.container = "test-pod"
        session.is_open = True  # Set session as open

        with (
            patch.object(session, "install") as mock_install,
            patch.object(session, "copy_to_runtime") as mock_copy,
            patch.object(session, "execute_commands") as mock_execute,
            patch("tempfile.NamedTemporaryFile") as mock_temp_file,
        ):
            mock_file_instance = mock_temp_file.return_value
            mock_file_instance.name = "/tmp/code.py"
            mock_file_instance.write = Mock()
            mock_file_instance.seek = Mock()

            mock_file_instance.__enter__.return_value = mock_file_instance
            mock_file_instance.__exit__ = Mock()

            expected_result = ConsoleOutput(exit_code=0, stdout="output")
            mock_execute.return_value = expected_result

            result = session.run("print('hello')", ["numpy"])

            assert result == expected_result
            mock_install.assert_called_once_with(["numpy"])
            mock_copy.assert_called_once_with("/tmp/code.py", "/sandbox/code.py")
            mock_execute.assert_called_once_with(["python /sandbox/code.py"], workdir="/sandbox")

    @patch("kubernetes.config.load_kube_config")
    @patch("llm_sandbox.kubernetes.CoreV1Api")
    @patch("llm_sandbox.language_handlers.factory.LanguageHandlerFactory.create_handler")
    def test_run_without_open_session(
        self,
        mock_create_handler: MagicMock,
        mock_core_v1_api: MagicMock,
        mock_load_config: MagicMock,
    ) -> None:
        """Test run fails when session is not open."""
        mock_handler = MagicMock()
        mock_create_handler.return_value = mock_handler

        session = SandboxKubernetesSession()
        session.container = None
        session.is_open = False

        with pytest.raises(NotOpenSessionError):
            session.run("print('hello')")


class TestSandboxKubernetesSessionFileOperations:
    """Test SandboxKubernetesSession file operations."""

    @patch("kubernetes.config.load_kube_config")
    @patch("llm_sandbox.kubernetes.CoreV1Api")
    @patch("llm_sandbox.kubernetes.stream")
    @patch("llm_sandbox.language_handlers.factory.LanguageHandlerFactory.create_handler")
    def test_copy_to_runtime(
        self,
        mock_create_handler: MagicMock,
        mock_stream: MagicMock,
        mock_core_v1_api: MagicMock,
        mock_load_config: MagicMock,
    ) -> None:
        """Test copying file to pod."""
        mock_handler = MagicMock()
        mock_create_handler.return_value = mock_handler
        mock_client = MagicMock()
        mock_core_v1_api.return_value = mock_client

        session = SandboxKubernetesSession()
        session.container = "test-pod"

        # Mock streaming response
        mock_resp = MagicMock()
        mock_resp.is_open.side_effect = [True, False]  # Loop once then exit
        mock_resp.peek_stdout.return_value = True
        mock_resp.read_stdout.return_value = "output"
        mock_resp.peek_stderr.return_value = False
        mock_resp.write_stdin = MagicMock()
        mock_resp.close = MagicMock()
        mock_stream.return_value = mock_resp

        with (
            patch.object(session.container_api, "copy_to_container") as mock_copy_to_container,
            patch.object(session, "_ensure_ownership") as mock_ownership,
            tempfile.NamedTemporaryFile() as temp_file,
        ):
            # Write some content to the temporary file
            temp_file.write(b"test content")
            temp_file.flush()

            session.copy_to_runtime(temp_file.name, "/pod/file.txt")

            mock_copy_to_container.assert_called_once_with("test-pod", temp_file.name, "/pod/file.txt")
            mock_ownership.assert_called_once_with(["/pod/file.txt"])

    @patch("kubernetes.config.load_kube_config")
    @patch("llm_sandbox.kubernetes.CoreV1Api")
    @patch("llm_sandbox.kubernetes.stream")
    @patch("llm_sandbox.language_handlers.factory.LanguageHandlerFactory.create_handler")
    def test_copy_from_runtime(
        self,
        mock_create_handler: MagicMock,
        mock_stream: MagicMock,
        mock_core_v1_api: MagicMock,
        mock_load_config: MagicMock,
    ) -> None:
        """Test copying file from pod."""
        mock_handler = MagicMock()
        mock_create_handler.return_value = mock_handler

        session = SandboxKubernetesSession()
        session.container = "test-pod"

        # Create mock tar data with file content
        file_content = b"test content"
        tar_buffer = io.BytesIO()
        with tarfile.open(fileobj=tar_buffer, mode="w") as tar:
            info = tarfile.TarInfo("file.txt")
            info.size = len(file_content)
            tar.addfile(info, io.BytesIO(file_content))
        tar_data = tar_buffer.getvalue()

        with (
            patch.object(session.container_api, "copy_from_container") as mock_copy_from_container,
            tempfile.TemporaryDirectory() as temp_dir,
        ):
            mock_copy_from_container.return_value = (tar_data, {"size": len(tar_data)})

            dest_file = f"{temp_dir}/file.txt"
            session.copy_from_runtime("/pod/file.txt", dest_file)

            mock_copy_from_container.assert_called_once_with("test-pod", "/pod/file.txt")

    @patch("kubernetes.config.load_kube_config")
    @patch("llm_sandbox.kubernetes.CoreV1Api")
    @patch("llm_sandbox.language_handlers.factory.LanguageHandlerFactory.create_handler")
    def test_copy_to_runtime_no_container(
        self,
        mock_create_handler: MagicMock,
        mock_core_v1_api: MagicMock,
        mock_load_config: MagicMock,
    ) -> None:
        """Test copy_to_runtime fails when no container."""
        mock_handler = MagicMock()
        mock_create_handler.return_value = mock_handler

        session = SandboxKubernetesSession()
        session.container = None

        with pytest.raises(NotOpenSessionError):
            session.copy_to_runtime("/host/file.txt", "/pod/file.txt")

    @patch("kubernetes.config.load_kube_config")
    @patch("llm_sandbox.kubernetes.CoreV1Api")
    @patch("llm_sandbox.kubernetes.stream")
    @patch("llm_sandbox.language_handlers.factory.LanguageHandlerFactory.create_handler")
    def test_copy_from_runtime_file_not_found(
        self,
        mock_create_handler: MagicMock,
        mock_stream: MagicMock,
        mock_core_v1_api: MagicMock,
        mock_load_config: MagicMock,
    ) -> None:
        """Test copy_from_runtime when file not found in tar."""
        mock_handler = MagicMock()
        mock_create_handler.return_value = mock_handler

        session = SandboxKubernetesSession()
        session.container = "test-pod"

        with (
            patch.object(session.container_api, "copy_from_container") as mock_copy_from_container,
        ):
            # Simulate file not found by having container_api.copy_from_container return empty data
            mock_copy_from_container.return_value = (b"", {"size": 0})

            # The mixin should raise FileNotFoundError when no tar data is found
            # But since we're mocking the container_api, we need to mock the behavior
            with patch.object(session, "container_api") as mock_container_api:
                mock_container_api.copy_from_container.side_effect = FileNotFoundError("File not found")

                with pytest.raises(FileNotFoundError):
                    session.copy_from_runtime("/pod/missing.txt", "/host/file.txt")


class TestSandboxKubernetesSessionCommands:
    """Test SandboxKubernetesSession command execution."""

    @patch("kubernetes.config.load_kube_config")
    @patch("llm_sandbox.kubernetes.CoreV1Api")
    @patch("llm_sandbox.kubernetes.stream")
    @patch("llm_sandbox.language_handlers.factory.LanguageHandlerFactory.create_handler")
    def test_execute_command_success(
        self,
        mock_create_handler: MagicMock,
        mock_stream: MagicMock,
        mock_core_v1_api: MagicMock,
        mock_load_config: MagicMock,
    ) -> None:
        """Test successful command execution."""
        mock_handler = MagicMock()
        mock_create_handler.return_value = mock_handler

        session = SandboxKubernetesSession()
        session.container = "test-pod"

        # Mock streaming response
        mock_resp = MagicMock()
        mock_resp.is_open.side_effect = [True, True, False]  # Loop twice then exit
        mock_resp.update = MagicMock()
        mock_resp.peek_stdout.side_effect = [True, False, False]
        mock_resp.read_stdout.side_effect = ["stdout output", "", ""]
        mock_resp.peek_stderr.side_effect = [False, True, False]
        mock_resp.read_stderr.side_effect = ["stderr output"]
        mock_resp.returncode = 0
        mock_stream.return_value = mock_resp

        result = session.execute_command("ls -l", workdir="/tmp")

        assert result.exit_code == 0
        assert result.stdout == "stdout output"
        assert result.stderr == "stderr output"
        mock_stream.assert_called_once()

    @patch("kubernetes.config.load_kube_config")
    @patch("llm_sandbox.kubernetes.CoreV1Api")
    @patch("llm_sandbox.language_handlers.factory.LanguageHandlerFactory.create_handler")
    def test_execute_command_no_container(
        self,
        mock_create_handler: MagicMock,
        mock_core_v1_api: MagicMock,
        mock_load_config: MagicMock,
    ) -> None:
        """Test execute_command fails when no container."""
        mock_handler = MagicMock()
        mock_create_handler.return_value = mock_handler

        session = SandboxKubernetesSession()
        session.container = None

        with pytest.raises(NotOpenSessionError):
            session.execute_command("ls")

    @patch("kubernetes.config.load_kube_config")
    @patch("llm_sandbox.kubernetes.CoreV1Api")
    @patch("llm_sandbox.kubernetes.stream")
    @patch("llm_sandbox.language_handlers.factory.LanguageHandlerFactory.create_handler")
    def test_execute_command_with_workdir(
        self,
        mock_create_handler: MagicMock,
        mock_stream: MagicMock,
        mock_core_v1_api: MagicMock,
        mock_load_config: MagicMock,
    ) -> None:
        """Test command execution with working directory."""
        mock_handler = MagicMock()
        mock_create_handler.return_value = mock_handler

        session = SandboxKubernetesSession()
        session.container = "test-pod"

        mock_resp = MagicMock()
        mock_resp.is_open.return_value = False
        mock_resp.returncode = 0
        mock_stream.return_value = mock_resp

        session.execute_command("ls", workdir="/custom")

        # Verify the command was wrapped with cd
        call_args = mock_stream.call_args
        executed_command = call_args[1]["command"]
        assert "cd /custom && ls" in " ".join(executed_command)


class TestSandboxKubernetesSessionArchive:
    """Test SandboxKubernetesSession archive operations."""

    @patch("kubernetes.config.load_kube_config")
    @patch("llm_sandbox.kubernetes.CoreV1Api")
    @patch("llm_sandbox.language_handlers.factory.LanguageHandlerFactory.create_handler")
    def test_get_archive_success(
        self,
        mock_create_handler: MagicMock,
        mock_core_v1_api: MagicMock,
        mock_load_config: MagicMock,
    ) -> None:
        """Test successful archive retrieval."""
        mock_handler = MagicMock()
        mock_create_handler.return_value = mock_handler

        session = SandboxKubernetesSession()
        session.container = "test-pod"

        # Mock file stats
        file_content = b"test content"

        with patch.object(session.container_api, "copy_from_container") as mock_copy_from:
            mock_copy_from.return_value = (
                file_content,
                {
                    "name": "/pod/file.txt",
                    "size": 100,
                    "mtime": 1234567890,
                    "mode": 0o644,
                    "linkTarget": "",
                },
            )

            data, stat = session.get_archive("/pod/file.txt")

            assert data == file_content
            assert stat["name"] == "/pod/file.txt"
            assert stat["size"] == 100
            assert stat["mtime"] == 1234567890
            mock_copy_from.assert_called_once_with("test-pod", "/pod/file.txt")

    @patch("kubernetes.config.load_kube_config")
    @patch("llm_sandbox.kubernetes.CoreV1Api")
    @patch("llm_sandbox.language_handlers.factory.LanguageHandlerFactory.create_handler")
    def test_get_archive_file_not_found(
        self,
        mock_create_handler: MagicMock,
        mock_core_v1_api: MagicMock,
        mock_load_config: MagicMock,
    ) -> None:
        """Test archive retrieval when file not found."""
        mock_handler = MagicMock()
        mock_create_handler.return_value = mock_handler

        session = SandboxKubernetesSession()
        session.container = "test-pod"

        with patch.object(session.container_api, "copy_from_container") as mock_copy_from:
            mock_copy_from.return_value = (b"", {"size": 0})

            data, stat = session.get_archive("/pod/missing.txt")

            assert data == b""
            assert stat == {"size": 0}

    @patch("kubernetes.config.load_kube_config")
    @patch("llm_sandbox.kubernetes.CoreV1Api")
    @patch("llm_sandbox.language_handlers.factory.LanguageHandlerFactory.create_handler")
    def test_get_archive_base64_decode_error(
        self,
        mock_create_handler: MagicMock,
        mock_core_v1_api: MagicMock,
        mock_load_config: MagicMock,
    ) -> None:
        """Test archive retrieval with base64 decode error."""
        mock_handler = MagicMock()
        mock_create_handler.return_value = mock_handler

        session = SandboxKubernetesSession()
        session.container = "test-pod"

        with patch.object(session.container_api, "copy_from_container") as mock_copy_from:
            mock_copy_from.return_value = (b"", {"size": 0})  # Simulate decode error result

            data, stat = session.get_archive("/pod/file.txt")

            assert data == b""
            assert stat == {"size": 0}

    @patch("kubernetes.config.load_kube_config")
    @patch("llm_sandbox.kubernetes.CoreV1Api")
    @patch("llm_sandbox.language_handlers.factory.LanguageHandlerFactory.create_handler")
    def test_get_archive_no_container(
        self,
        mock_create_handler: MagicMock,
        mock_core_v1_api: MagicMock,
        mock_load_config: MagicMock,
    ) -> None:
        """Test get_archive fails when no container."""
        mock_handler = MagicMock()
        mock_create_handler.return_value = mock_handler

        session = SandboxKubernetesSession()
        session.container = None

        with pytest.raises(NotOpenSessionError):
            session.get_archive("/pod/path")


class TestSandboxKubernetesSessionOwnership:
    """Test SandboxKubernetesSession ownership management."""

    @patch("kubernetes.config.load_kube_config")
    @patch("llm_sandbox.kubernetes.CoreV1Api")
    @patch("llm_sandbox.language_handlers.factory.LanguageHandlerFactory.create_handler")
    def test_ensure_ownership_with_non_root_user(
        self,
        mock_create_handler: MagicMock,
        mock_core_v1_api: MagicMock,
        mock_load_config: MagicMock,
    ) -> None:
        """Test _ensure_ownership with non-root user."""
        mock_handler = MagicMock()
        mock_create_handler.return_value = mock_handler

        session = SandboxKubernetesSession()
        session.container = "test-pod"

        with (
            patch.object(session, "execute_command") as mock_execute_command,
        ):
            # Mock user check returning non-root
            mock_execute_command.side_effect = [
                ConsoleOutput(exit_code=0, stdout="1000"),  # id -u command
                ConsoleOutput(exit_code=0, stdout=""),  # chown command
            ]

            session._ensure_ownership(["/tmp/test", "/tmp/test2"])

            assert mock_execute_command.call_count == 2
            # Verify the chown command was called
            chown_call = mock_execute_command.call_args_list[1]
            assert "chown -R $(id -u):$(id -g)" in str(chown_call)

    @patch("kubernetes.config.load_kube_config")
    @patch("llm_sandbox.kubernetes.CoreV1Api")
    @patch("llm_sandbox.language_handlers.factory.LanguageHandlerFactory.create_handler")
    def test_ensure_ownership_with_root_user(
        self,
        mock_create_handler: MagicMock,
        mock_core_v1_api: MagicMock,
        mock_load_config: MagicMock,
    ) -> None:
        """Test _ensure_ownership with root user (no additional chown)."""
        mock_handler = MagicMock()
        mock_create_handler.return_value = mock_handler

        session = SandboxKubernetesSession()
        session.container = "test-pod"

        with (
            patch.object(session, "execute_command") as mock_execute_command,
        ):
            # Mock user check returning root
            mock_execute_command.return_value = ConsoleOutput(exit_code=0, stdout="0")

            session._ensure_ownership(["/tmp/test"])

            # Should only call id -u, not chown (since root doesn't need ownership change)
            assert mock_execute_command.call_count == 1


class TestSandboxKubernetesSessionContextManager:
    """Test SandboxKubernetesSession context manager functionality."""

    @patch("kubernetes.config.load_kube_config")
    @patch("llm_sandbox.kubernetes.CoreV1Api")
    @patch("llm_sandbox.language_handlers.factory.LanguageHandlerFactory.create_handler")
    def test_context_manager(
        self,
        mock_create_handler: MagicMock,
        mock_core_v1_api: MagicMock,
        mock_load_config: MagicMock,
    ) -> None:
        """Test using session as context manager."""
        mock_handler = MagicMock()
        mock_create_handler.return_value = mock_handler

        session = SandboxKubernetesSession()

        with (
            patch.object(session, "open") as mock_open,
            patch.object(session, "close") as mock_close,
        ):
            with session as s:
                assert s == session
                mock_open.assert_called_once()

            mock_close.assert_called_once()

    @patch("kubernetes.config.load_kube_config")
    @patch("llm_sandbox.kubernetes.CoreV1Api")
    @patch("llm_sandbox.language_handlers.factory.LanguageHandlerFactory.create_handler")
    def test_context_manager_with_exception(
        self,
        mock_create_handler: MagicMock,
        mock_core_v1_api: MagicMock,
        mock_load_config: MagicMock,
    ) -> None:
        """Test context manager ensures close is called even with exception."""
        mock_handler = MagicMock()
        mock_create_handler.return_value = mock_handler

        session = SandboxKubernetesSession()

        with (
            patch.object(session, "open") as mock_open,
            patch.object(session, "close") as mock_close,
        ):
            with pytest.raises(ValueError), session:
                raise ValueError

            mock_open.assert_called_once()
            mock_close.assert_called_once()
