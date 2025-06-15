# ruff: noqa: SLF001, PLR2004, ARG002, PT011, PT012

"""Tests for Kubernetes backend implementation."""

import io
import tarfile
import tempfile
from unittest.mock import MagicMock, Mock, patch

import pytest
from kubernetes.client.exceptions import ApiException

from llm_sandbox.const import DefaultImage, SupportedLanguage
from llm_sandbox.data import ConsoleOutput
from llm_sandbox.exceptions import ContainerError, NotOpenSessionError
from llm_sandbox.kubernetes import KubernetesContainerAPI, SandboxKubernetesSession
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
        mock_create_handler.return_value = mock_handler

        session = SandboxKubernetesSession()
        session.container = "test-pod"
        session.is_open = True  # Set session as open

        with (
            patch.object(session, "install") as mock_install,
            patch.object(session, "copy_to_runtime") as _,
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


class TestKubernetesContainerAPI:
    """Test KubernetesContainerAPI class."""

    @patch("llm_sandbox.kubernetes.CoreV1Api")
    def test_init_default_namespace(self, mock_core_v1_api: MagicMock) -> None:
        """Test initialization with default namespace."""
        mock_client = MagicMock()
        api = KubernetesContainerAPI(mock_client)

        assert api.client == mock_client
        assert api.namespace == "default"

    @patch("llm_sandbox.kubernetes.CoreV1Api")
    def test_init_custom_namespace(self, mock_core_v1_api: MagicMock) -> None:
        """Test initialization with custom namespace."""
        mock_client = MagicMock()
        api = KubernetesContainerAPI(mock_client, "custom-ns")

        assert api.client == mock_client
        assert api.namespace == "custom-ns"

    @patch("llm_sandbox.kubernetes.CoreV1Api")
    @patch("time.sleep")
    def test_create_container_success(self, mock_sleep: MagicMock, mock_core_v1_api: MagicMock) -> None:
        """Test successful container creation."""
        mock_client = MagicMock()
        api = KubernetesContainerAPI(mock_client)

        pod_manifest = {"metadata": {"name": "test-pod"}, "spec": {"containers": [{"name": "test"}]}}
        config = {"pod_manifest": pod_manifest}

        # Mock pod status becoming Running
        mock_pod = MagicMock()
        mock_pod.status.phase = "Running"
        mock_client.read_namespaced_pod.return_value = mock_pod

        result = api.create_container(config)

        assert result == "test-pod"
        mock_client.create_namespaced_pod.assert_called_once_with(namespace="default", body=pod_manifest)

    @patch("llm_sandbox.kubernetes.CoreV1Api")
    @patch("time.sleep")
    @patch("time.time")
    def test_create_container_timeout(
        self, mock_time: MagicMock, mock_sleep: MagicMock, mock_core_v1_api: MagicMock
    ) -> None:
        """Test container creation timeout."""
        mock_client = MagicMock()
        api = KubernetesContainerAPI(mock_client)

        pod_manifest = {"metadata": {"name": "test-pod"}, "spec": {"containers": [{"name": "test"}]}}
        config = {"pod_manifest": pod_manifest}

        # Mock time progression to simulate timeout
        mock_time.side_effect = [0, 400]  # Start time and timeout exceeded

        # Mock pod status staying Pending
        mock_pod = MagicMock()
        mock_pod.status.phase = "Pending"
        mock_client.read_namespaced_pod.return_value = mock_pod

        with pytest.raises(TimeoutError, match="did not start within"):
            api.create_container(config)

    @patch("llm_sandbox.kubernetes.CoreV1Api")
    def test_start_container_noop(self, mock_core_v1_api: MagicMock) -> None:
        """Test start_container is a no-op."""
        mock_client = MagicMock()
        api = KubernetesContainerAPI(mock_client)

        # Should not raise any exception
        api.start_container("test-container")

    @patch("llm_sandbox.kubernetes.CoreV1Api")
    @patch("llm_sandbox.kubernetes.k8s_client.V1DeleteOptions")
    def test_stop_container_success(self, mock_delete_options: MagicMock, mock_core_v1_api: MagicMock) -> None:
        """Test successful container stopping."""
        mock_client = MagicMock()
        api = KubernetesContainerAPI(mock_client)
        mock_delete_opts = MagicMock()
        mock_delete_options.return_value = mock_delete_opts

        api.stop_container("test-pod")

        mock_client.delete_namespaced_pod.assert_called_once_with(
            name="test-pod", namespace="default", body=mock_delete_opts
        )

    @patch("llm_sandbox.kubernetes.CoreV1Api")
    @patch("llm_sandbox.kubernetes.k8s_client.V1DeleteOptions")
    def test_stop_container_with_exception(self, mock_delete_options: MagicMock, mock_core_v1_api: MagicMock) -> None:
        """Test container stopping with exception (should not raise)."""
        mock_client = MagicMock()
        api = KubernetesContainerAPI(mock_client)
        mock_client.delete_namespaced_pod.side_effect = Exception("Pod not found")

        # Should not raise exception
        api.stop_container("test-pod")

    @patch("llm_sandbox.kubernetes.CoreV1Api")
    @patch("llm_sandbox.kubernetes.stream")
    def test_execute_command_simple(self, mock_stream: MagicMock, mock_core_v1_api: MagicMock) -> None:
        """Test simple command execution."""
        mock_client = MagicMock()
        api = KubernetesContainerAPI(mock_client)

        mock_resp = MagicMock()
        mock_resp.is_open.side_effect = [True, False]
        mock_resp.update.return_value = None
        mock_resp.peek_stdout.side_effect = [True, False]
        mock_resp.read_stdout.return_value = "output"
        mock_resp.peek_stderr.return_value = False
        mock_resp.returncode = 0
        mock_stream.return_value = mock_resp

        exit_code, output = api.execute_command("test-container", "ls")

        assert exit_code == 0
        assert output == ("output", "")

    @patch("llm_sandbox.kubernetes.CoreV1Api")
    @patch("llm_sandbox.kubernetes.stream")
    def test_execute_command_with_workdir(self, mock_stream: MagicMock, mock_core_v1_api: MagicMock) -> None:
        """Test command execution with working directory."""
        mock_client = MagicMock()
        api = KubernetesContainerAPI(mock_client)

        mock_resp = MagicMock()
        mock_resp.is_open.return_value = False
        mock_resp.returncode = 0
        mock_stream.return_value = mock_resp

        api.execute_command("test-container", "ls", workdir="/tmp")

        # Verify the command was wrapped with cd
        call_args = mock_stream.call_args
        executed_command = call_args[1]["command"]
        assert "cd /tmp && ls" in " ".join(executed_command)

    @patch("llm_sandbox.kubernetes.CoreV1Api")
    @patch("llm_sandbox.kubernetes.stream")
    def test_execute_command_with_stderr(self, mock_stream: MagicMock, mock_core_v1_api: MagicMock) -> None:
        """Test command execution with stderr output."""
        mock_client = MagicMock()
        api = KubernetesContainerAPI(mock_client)

        mock_resp = MagicMock()
        mock_resp.is_open.side_effect = [True, True, False]
        mock_resp.update.return_value = None
        mock_resp.peek_stdout.side_effect = [False, False, False]
        mock_resp.peek_stderr.side_effect = [True, False, False]
        mock_resp.read_stderr.return_value = "error output"
        mock_resp.returncode = 1
        mock_stream.return_value = mock_resp

        exit_code, output = api.execute_command("test-container", "invalid-command")

        assert exit_code == 1
        assert output == ("", "error output")

    @patch("llm_sandbox.kubernetes.CoreV1Api")
    @patch("llm_sandbox.kubernetes.stream")
    @patch("pathlib.Path.exists")
    @patch("pathlib.Path.is_file")
    @patch("tarfile.open")
    def test_copy_to_container_success(
        self,
        mock_tarfile: MagicMock,
        mock_is_file: MagicMock,
        mock_exists: MagicMock,
        mock_stream: MagicMock,
        mock_core_v1_api: MagicMock,
    ) -> None:
        """Test successful file copy to container."""
        mock_client = MagicMock()
        api = KubernetesContainerAPI(mock_client)

        # Mock file existence checks
        mock_exists.return_value = True
        mock_is_file.return_value = True

        # Mock successful mkdir response
        mock_mkdir_resp = MagicMock()
        mock_mkdir_resp.is_open.return_value = False
        mock_mkdir_resp.returncode = 0

        # Mock successful tar response
        mock_tar_resp = MagicMock()
        mock_tar_resp.is_open.side_effect = [True, False]
        mock_tar_resp.update.return_value = None
        mock_tar_resp.write_stdin = MagicMock()
        mock_tar_resp.close = MagicMock()

        mock_stream.side_effect = [mock_mkdir_resp, mock_tar_resp]

        # Mock tarfile operations
        mock_tar = MagicMock()
        mock_tarfile.return_value.__enter__.return_value = mock_tar

        with tempfile.NamedTemporaryFile() as temp_file:
            api.copy_to_container("test-container", temp_file.name, "/pod/dest.txt")

        # Verify mkdir was called
        assert mock_stream.call_count == 2  # mkdir + tar

    @patch("llm_sandbox.kubernetes.CoreV1Api")
    @patch("pathlib.Path.exists")
    def test_copy_to_container_file_not_found(self, mock_exists: MagicMock, mock_core_v1_api: MagicMock) -> None:
        """Test copy_to_container with non-existent file."""
        mock_client = MagicMock()
        api = KubernetesContainerAPI(mock_client)

        mock_exists.return_value = False

        with pytest.raises(FileNotFoundError, match="does not exist"):
            api.copy_to_container("test-container", "/nonexistent/file.txt", "/pod/dest.txt")

    @patch("llm_sandbox.kubernetes.CoreV1Api")
    @patch("llm_sandbox.kubernetes.stream")
    @patch("pathlib.Path.exists")
    @patch("pathlib.Path.is_file")
    def test_copy_to_container_mkdir_failure(
        self, mock_is_file: MagicMock, mock_exists: MagicMock, mock_stream: MagicMock, mock_core_v1_api: MagicMock
    ) -> None:
        """Test copy_to_container when mkdir fails."""
        mock_client = MagicMock()
        api = KubernetesContainerAPI(mock_client)

        mock_exists.return_value = True
        mock_is_file.return_value = True

        # Mock failed mkdir response
        mock_mkdir_resp = MagicMock()
        mock_mkdir_resp.is_open.side_effect = [True, False]
        mock_mkdir_resp.update.return_value = None
        mock_mkdir_resp.peek_stderr.side_effect = [True, False]
        mock_mkdir_resp.read_stderr.return_value = "Permission denied"
        mock_mkdir_resp.returncode = 1

        mock_stream.return_value = mock_mkdir_resp

        with (
            tempfile.NamedTemporaryFile() as temp_file,
            pytest.raises(RuntimeError, match="Failed to create directory"),
        ):
            api.copy_to_container("test-container", temp_file.name, "/pod/dest.txt")

    @patch("llm_sandbox.kubernetes.CoreV1Api")
    @patch("llm_sandbox.kubernetes.stream")
    def test_copy_from_container_success(self, mock_stream: MagicMock, mock_core_v1_api: MagicMock) -> None:
        """Test successful file copy from container."""
        mock_client = MagicMock()
        api = KubernetesContainerAPI(mock_client)

        # Mock stat command response
        mock_stat_resp = MagicMock()
        mock_stat_resp.is_open.side_effect = [True, False]
        mock_stat_resp.update.return_value = None
        mock_stat_resp.peek_stdout.side_effect = [True, False]
        mock_stat_resp.read_stdout.return_value = "1024 1609459200 /pod/file.txt"

        # Mock tar command response
        mock_tar_resp = MagicMock()
        mock_tar_resp.is_open.side_effect = [True, False]
        mock_tar_resp.update.return_value = None
        mock_tar_resp.peek_stdout.side_effect = [True, False]
        mock_tar_resp.read_stdout.return_value = "dGVzdCBjb250ZW50"  # base64 encoded "test content"

        mock_stream.side_effect = [mock_stat_resp, mock_tar_resp]

        data, stat = api.copy_from_container("test-container", "/pod/file.txt")

        assert len(data) > 0  # Should contain decoded tar data
        assert stat["name"] == "/pod/file.txt"
        assert stat["size"] == 1024
        assert stat["mtime"] == 1609459200

    @patch("llm_sandbox.kubernetes.CoreV1Api")
    @patch("llm_sandbox.kubernetes.stream")
    def test_copy_from_container_file_not_found(self, mock_stream: MagicMock, mock_core_v1_api: MagicMock) -> None:
        """Test copy_from_container when file not found."""
        mock_client = MagicMock()
        api = KubernetesContainerAPI(mock_client)

        # Mock stat command returning NOT_FOUND
        mock_stat_resp = MagicMock()
        mock_stat_resp.is_open.side_effect = [True, False]
        mock_stat_resp.update.return_value = None
        mock_stat_resp.peek_stdout.side_effect = [True, False]
        mock_stat_resp.read_stdout.return_value = "NOT_FOUND"

        mock_stream.return_value = mock_stat_resp

        data, stat = api.copy_from_container("test-container", "/pod/missing.txt")

        assert data == b""
        assert stat == {"size": 0}

    @patch("llm_sandbox.kubernetes.CoreV1Api")
    @patch("llm_sandbox.kubernetes.stream")
    def test_copy_from_container_base64_decode_error(self, mock_stream: MagicMock, mock_core_v1_api: MagicMock) -> None:
        """Test copy_from_container with base64 decode error."""
        mock_client = MagicMock()
        api = KubernetesContainerAPI(mock_client)

        # Mock stat command response
        mock_stat_resp = MagicMock()
        mock_stat_resp.is_open.side_effect = [True, False]
        mock_stat_resp.update.return_value = None
        mock_stat_resp.peek_stdout.side_effect = [True, False]
        mock_stat_resp.read_stdout.return_value = "1024 1609459200 /pod/file.txt"

        # Mock tar command with invalid base64
        mock_tar_resp = MagicMock()
        mock_tar_resp.is_open.side_effect = [True, False]
        mock_tar_resp.update.return_value = None
        mock_tar_resp.peek_stdout.side_effect = [True, False]
        mock_tar_resp.read_stdout.return_value = "invalid_base64!!!"

        mock_stream.side_effect = [mock_stat_resp, mock_tar_resp]

        data, stat = api.copy_from_container("test-container", "/pod/file.txt")

        assert data == b""
        assert stat == {"size": 0}

    @patch("llm_sandbox.kubernetes.CoreV1Api")
    @patch("llm_sandbox.kubernetes.stream")
    def test_copy_from_container_empty_tar_output(self, mock_stream: MagicMock, mock_core_v1_api: MagicMock) -> None:
        """Test copy_from_container with empty tar output."""
        mock_client = MagicMock()
        api = KubernetesContainerAPI(mock_client)

        # Mock stat command response
        mock_stat_resp = MagicMock()
        mock_stat_resp.is_open.side_effect = [True, False]
        mock_stat_resp.update.return_value = None
        mock_stat_resp.peek_stdout.side_effect = [True, False]
        mock_stat_resp.read_stdout.return_value = "1024 1609459200 /pod/file.txt"

        # Mock tar command with empty output
        mock_tar_resp = MagicMock()
        mock_tar_resp.is_open.side_effect = [True, False]
        mock_tar_resp.update.return_value = None
        mock_tar_resp.peek_stdout.side_effect = [True, False]
        mock_tar_resp.read_stdout.return_value = ""

        mock_stream.side_effect = [mock_stat_resp, mock_tar_resp]

        data, stat = api.copy_from_container("test-container", "/pod/file.txt")

        assert data == b""
        assert stat == {"size": 0}


class TestSandboxKubernetesSessionEdgeCases:
    """Test additional edge cases for SandboxKubernetesSession."""

    @patch("kubernetes.config.load_kube_config")
    @patch("llm_sandbox.kubernetes.CoreV1Api")
    @patch("llm_sandbox.language_handlers.factory.LanguageHandlerFactory.create_handler")
    def test_init_with_pod_manifest_no_namespace(
        self, mock_create_handler: MagicMock, mock_core_v1_api: MagicMock, mock_load_config: MagicMock
    ) -> None:
        """Test initialization with pod manifest missing namespace."""
        mock_handler = MagicMock()
        mock_create_handler.return_value = mock_handler

        custom_manifest = {
            "apiVersion": "v1",
            "kind": "Pod",
            "metadata": {"name": "custom-pod"},  # No namespace
            "spec": {"containers": [{"name": "test", "image": "test:latest"}]},
        }

        session = SandboxKubernetesSession(pod_manifest=custom_manifest)

        # Should use default namespace when not specified in manifest
        assert session.kube_namespace == "default"

    @patch("kubernetes.config.load_kube_config")
    @patch("llm_sandbox.kubernetes.CoreV1Api")
    @patch("llm_sandbox.language_handlers.factory.LanguageHandlerFactory.create_handler")
    def test_default_pod_manifest_no_env_vars(
        self, mock_create_handler: MagicMock, mock_core_v1_api: MagicMock, mock_load_config: MagicMock
    ) -> None:
        """Test default pod manifest generation without environment variables."""
        mock_handler = MagicMock()
        mock_create_handler.return_value = mock_handler

        session = SandboxKubernetesSession()

        manifest = session.pod_manifest
        containers = manifest["spec"]["containers"]

        # Should not have env section when no env_vars provided
        assert "env" not in containers[0]

    @patch("kubernetes.config.load_kube_config")
    @patch("llm_sandbox.kubernetes.CoreV1Api")
    @patch("llm_sandbox.language_handlers.factory.LanguageHandlerFactory.create_handler")
    def test_ensure_directory_exists_success(
        self, mock_create_handler: MagicMock, mock_core_v1_api: MagicMock, mock_load_config: MagicMock
    ) -> None:
        """Test successful directory creation."""
        mock_handler = MagicMock()
        mock_create_handler.return_value = mock_handler

        session = SandboxKubernetesSession()
        session.container = "test-pod"

        with patch.object(session.container_api, "execute_command") as mock_execute:
            mock_execute.return_value = (0, ("", ""))

            session._ensure_directory_exists("/test/path")

            mock_execute.assert_called_once_with("test-pod", "mkdir -p '/test/path'")

    @patch("kubernetes.config.load_kube_config")
    @patch("llm_sandbox.kubernetes.CoreV1Api")
    @patch("llm_sandbox.language_handlers.factory.LanguageHandlerFactory.create_handler")
    def test_ensure_directory_exists_failure(
        self, mock_create_handler: MagicMock, mock_core_v1_api: MagicMock, mock_load_config: MagicMock
    ) -> None:
        """Test directory creation failure (should log error but not raise)."""
        mock_handler = MagicMock()
        mock_create_handler.return_value = mock_handler

        session = SandboxKubernetesSession(verbose=True)
        session.container = "test-pod"

        with (
            patch.object(session.container_api, "execute_command") as mock_execute,
            patch.object(session, "_log") as mock_log,
        ):
            mock_execute.return_value = (1, ("", "Permission denied"))

            session._ensure_directory_exists("/test/path")

            mock_log.assert_called_with("Failed to create directory /test/path: Permission denied", "error")

    @patch("kubernetes.config.load_kube_config")
    @patch("llm_sandbox.kubernetes.CoreV1Api")
    @patch("llm_sandbox.language_handlers.factory.LanguageHandlerFactory.create_handler")
    def test_ensure_directory_exists_stderr_error(
        self, mock_create_handler: MagicMock, mock_core_v1_api: MagicMock, mock_load_config: MagicMock
    ) -> None:
        """Test directory creation with stderr output only."""
        mock_handler = MagicMock()
        mock_create_handler.return_value = mock_handler

        session = SandboxKubernetesSession(verbose=True)
        session.container = "test-pod"

        with (
            patch.object(session.container_api, "execute_command") as mock_execute,
            patch.object(session, "_log") as mock_log,
        ):
            mock_execute.return_value = (1, ("some output", ""))

            session._ensure_directory_exists("/test/path")

            mock_log.assert_called_with("Failed to create directory /test/path: some output", "error")

    @patch("kubernetes.config.load_kube_config")
    @patch("llm_sandbox.kubernetes.CoreV1Api")
    @patch("llm_sandbox.language_handlers.factory.LanguageHandlerFactory.create_handler")
    def test_handle_timeout_with_container(
        self, mock_create_handler: MagicMock, mock_core_v1_api: MagicMock, mock_load_config: MagicMock
    ) -> None:
        """Test timeout handling when container exists."""
        mock_handler = MagicMock()
        mock_create_handler.return_value = mock_handler

        session = SandboxKubernetesSession()
        session.container = "test-pod"

        with patch.object(session, "close") as mock_close:
            session._handle_timeout()

            mock_close.assert_called_once()

    @patch("kubernetes.config.load_kube_config")
    @patch("llm_sandbox.kubernetes.CoreV1Api")
    @patch("llm_sandbox.language_handlers.factory.LanguageHandlerFactory.create_handler")
    def test_handle_timeout_with_stop_exception(
        self, mock_create_handler: MagicMock, mock_core_v1_api: MagicMock, mock_load_config: MagicMock
    ) -> None:
        """Test timeout handling when close raises exception."""
        mock_handler = MagicMock()
        mock_create_handler.return_value = mock_handler

        session = SandboxKubernetesSession()
        session.container = "test-pod"

        with patch.object(session, "close", side_effect=Exception("Close failed")) as mock_close:
            # Should not raise exception, just call close
            session._handle_timeout()

            mock_close.assert_called_once()

    @patch("kubernetes.config.load_kube_config")
    @patch("llm_sandbox.kubernetes.CoreV1Api")
    @patch("llm_sandbox.language_handlers.factory.LanguageHandlerFactory.create_handler")
    def test_handle_timeout_no_container(
        self, mock_create_handler: MagicMock, mock_core_v1_api: MagicMock, mock_load_config: MagicMock
    ) -> None:
        """Test timeout handling when no container exists."""
        mock_handler = MagicMock()
        mock_create_handler.return_value = mock_handler

        session = SandboxKubernetesSession()
        session.container = None

        with patch.object(session, "close") as mock_close:
            session._handle_timeout()

            # Should not call close() when no container exists
            mock_close.assert_not_called()

    @patch("kubernetes.config.load_kube_config")
    @patch("llm_sandbox.kubernetes.CoreV1Api")
    @patch("llm_sandbox.language_handlers.factory.LanguageHandlerFactory.create_handler")
    def test_process_non_stream_output_tuple(
        self, mock_create_handler: MagicMock, mock_core_v1_api: MagicMock, mock_load_config: MagicMock
    ) -> None:
        """Test processing of non-stream tuple output."""
        mock_handler = MagicMock()
        mock_create_handler.return_value = mock_handler

        session = SandboxKubernetesSession()

        output = ("stdout_content", "stderr_content")
        stdout, stderr = session._process_non_stream_output(output)

        assert stdout == "stdout_content"
        assert stderr == "stderr_content"

    @patch("kubernetes.config.load_kube_config")
    @patch("llm_sandbox.kubernetes.CoreV1Api")
    @patch("llm_sandbox.language_handlers.factory.LanguageHandlerFactory.create_handler")
    def test_process_non_stream_output_invalid(
        self, mock_create_handler: MagicMock, mock_core_v1_api: MagicMock, mock_load_config: MagicMock
    ) -> None:
        """Test processing of invalid non-stream output."""
        mock_handler = MagicMock()
        mock_create_handler.return_value = mock_handler

        session = SandboxKubernetesSession()

        # Test with non-tuple input
        stdout, stderr = session._process_non_stream_output("invalid")
        assert stdout == ""
        assert stderr == ""

        # Test with wrong tuple length
        stdout, stderr = session._process_non_stream_output(("only_one",))
        assert stdout == ""
        assert stderr == ""

    @patch("kubernetes.config.load_kube_config")
    @patch("llm_sandbox.kubernetes.CoreV1Api")
    @patch("llm_sandbox.language_handlers.factory.LanguageHandlerFactory.create_handler")
    def test_process_stream_output_delegates(
        self, mock_create_handler: MagicMock, mock_core_v1_api: MagicMock, mock_load_config: MagicMock
    ) -> None:
        """Test that stream output processing delegates to non-stream processing."""
        mock_handler = MagicMock()
        mock_create_handler.return_value = mock_handler

        session = SandboxKubernetesSession()

        with patch.object(session, "_process_non_stream_output") as mock_process:
            mock_process.return_value = ("stdout", "stderr")

            result = session._process_stream_output("test_output")

            mock_process.assert_called_once_with("test_output")
            assert result == ("stdout", "stderr")


class TestSandboxKubernetesSessionTimeoutAndStream:
    """Test timeout and streaming functionality."""

    @patch("kubernetes.config.load_kube_config")
    @patch("llm_sandbox.kubernetes.CoreV1Api")
    @patch("llm_sandbox.language_handlers.factory.LanguageHandlerFactory.create_handler")
    def test_stream_property_default(
        self, mock_create_handler: MagicMock, mock_core_v1_api: MagicMock, mock_load_config: MagicMock
    ) -> None:
        """Test that stream property is set to False by default."""
        mock_handler = MagicMock()
        mock_create_handler.return_value = mock_handler

        session = SandboxKubernetesSession()

        assert session.stream is False

    @patch("kubernetes.config.load_kube_config")
    @patch("llm_sandbox.kubernetes.CoreV1Api")
    @patch("llm_sandbox.language_handlers.factory.LanguageHandlerFactory.create_handler")
    def test_close_with_stop_exception(
        self, mock_create_handler: MagicMock, mock_core_v1_api: MagicMock, mock_load_config: MagicMock
    ) -> None:
        """Test close method when stop_container raises exception."""
        mock_handler = MagicMock()
        mock_create_handler.return_value = mock_handler

        session = SandboxKubernetesSession(verbose=True)
        session.container = "test-pod"
        session.is_open = True

        with (
            patch.object(session.container_api, "stop_container", side_effect=Exception("Stop failed")),
            patch.object(session, "_log") as mock_log,
        ):
            session.close()

            mock_log.assert_called_with("Error cleaning up pod: Stop failed", "error")
            assert session.container is None
            assert session.is_open is False


class TestSandboxKubernetesSessionInitializationVariations:
    """Test various initialization scenarios."""

    @patch("kubernetes.config.load_kube_config")
    @patch("llm_sandbox.kubernetes.CoreV1Api")
    @patch("llm_sandbox.language_handlers.factory.LanguageHandlerFactory.create_handler")
    def test_init_with_all_timeout_parameters(
        self, mock_create_handler: MagicMock, mock_core_v1_api: MagicMock, mock_load_config: MagicMock
    ) -> None:
        """Test initialization with all timeout parameters."""
        mock_handler = MagicMock()
        mock_create_handler.return_value = mock_handler

        session = SandboxKubernetesSession(
            default_timeout=60.0,
            execution_timeout=120.0,
            session_timeout=3600.0,
        )
        assert session.config is not None
        assert session.config.default_timeout is not None
        assert session.config.execution_timeout is not None
        assert session.config.session_timeout is not None

        assert abs(session.config.default_timeout - 60.0) < 1e-6
        assert abs(session.config.execution_timeout - 120.0) < 1e-6
        assert abs(session.config.session_timeout - 3600.0) < 1e-6

    @patch("kubernetes.config.load_kube_config")
    @patch("llm_sandbox.kubernetes.CoreV1Api")
    @patch("llm_sandbox.language_handlers.factory.LanguageHandlerFactory.create_handler")
    def test_init_with_different_languages(
        self, mock_create_handler: MagicMock, mock_core_v1_api: MagicMock, mock_load_config: MagicMock
    ) -> None:
        """Test initialization with different supported languages."""
        mock_handler = MagicMock()
        mock_create_handler.return_value = mock_handler

        # Test with string language
        session_python = SandboxKubernetesSession(lang=SupportedLanguage.PYTHON)
        assert session_python.config.lang == SupportedLanguage.PYTHON

        # Test with different language
        session_go = SandboxKubernetesSession(lang="go")
        assert session_go.config.lang == SupportedLanguage.GO

    @patch("llm_sandbox.language_handlers.factory.LanguageHandlerFactory.create_handler")
    def test_init_with_client_no_kube_config_load(self, mock_create_handler: MagicMock) -> None:
        """Test that providing a client doesn't trigger kube config loading."""
        mock_handler = MagicMock()
        mock_create_handler.return_value = mock_handler
        custom_client = MagicMock()

        with patch("kubernetes.config.load_kube_config") as mock_load_config:
            session = SandboxKubernetesSession(client=custom_client)

            # Should not load kube config when client is provided
            mock_load_config.assert_not_called()
            assert session.client == custom_client


class TestSandboxKubernetesSessionInheritance:
    """Test inherited functionality and mixin behavior."""

    @patch("kubernetes.config.load_kube_config")
    @patch("llm_sandbox.kubernetes.CoreV1Api")
    @patch("llm_sandbox.language_handlers.factory.LanguageHandlerFactory.create_handler")
    def test_inherited_properties_from_base_session(
        self, mock_create_handler: MagicMock, mock_core_v1_api: MagicMock, mock_load_config: MagicMock
    ) -> None:
        """Test that inherited properties work correctly."""
        mock_handler = MagicMock()
        mock_create_handler.return_value = mock_handler

        session = SandboxKubernetesSession(verbose=True)

        # Test inherited properties
        assert hasattr(session, "config")
        assert hasattr(session, "verbose")
        assert hasattr(session, "logger")
        assert hasattr(session, "language_handler")
        assert session.verbose is True

    @patch("kubernetes.config.load_kube_config")
    @patch("llm_sandbox.kubernetes.CoreV1Api")
    @patch("llm_sandbox.language_handlers.factory.LanguageHandlerFactory.create_handler")
    def test_copy_to_runtime_inherited_validation(
        self, mock_create_handler: MagicMock, mock_core_v1_api: MagicMock, mock_load_config: MagicMock
    ) -> None:
        """Test that copy_to_runtime inherits proper validation from mixin."""
        mock_handler = MagicMock()
        mock_create_handler.return_value = mock_handler

        session = SandboxKubernetesSession()
        session.container = "test-pod"

        # Test that inherited validation catches non-existent files
        with pytest.raises(FileNotFoundError, match="does not exist"):
            session.copy_to_runtime("/nonexistent/file.txt", "/pod/dest.txt")

    @patch("kubernetes.config.load_kube_config")
    @patch("llm_sandbox.kubernetes.CoreV1Api")
    @patch("llm_sandbox.language_handlers.factory.LanguageHandlerFactory.create_handler")
    def test_copy_from_runtime_inherited_validation(
        self, mock_create_handler: MagicMock, mock_core_v1_api: MagicMock, mock_load_config: MagicMock
    ) -> None:
        """Test that copy_from_runtime inherits proper validation from mixin."""
        mock_handler = MagicMock()
        mock_create_handler.return_value = mock_handler

        session = SandboxKubernetesSession()
        session.container = "test-pod"

        with (
            patch.object(session.container_api, "copy_from_container") as mock_copy_from,
            pytest.raises(FileNotFoundError, match="not found in container"),
        ):
            mock_copy_from.return_value = (b"", {"size": 0})
            session.copy_from_runtime("/pod/missing.txt", "/host/dest.txt")

    @patch("kubernetes.config.load_kube_config")
    @patch("llm_sandbox.kubernetes.CoreV1Api")
    @patch("llm_sandbox.language_handlers.factory.LanguageHandlerFactory.create_handler")
    def test_execute_command_empty_command(
        self, mock_create_handler: MagicMock, mock_core_v1_api: MagicMock, mock_load_config: MagicMock
    ) -> None:
        """Test that execute_command handles empty commands properly."""
        mock_handler = MagicMock()
        mock_create_handler.return_value = mock_handler

        session = SandboxKubernetesSession()
        session.container = "test-pod"

        from llm_sandbox.exceptions import CommandEmptyError

        with pytest.raises(CommandEmptyError):
            session.execute_command("")

    @patch("kubernetes.config.load_kube_config")
    @patch("llm_sandbox.kubernetes.CoreV1Api")
    @patch("llm_sandbox.language_handlers.factory.LanguageHandlerFactory.create_handler")
    def test_verbose_logging_behavior(
        self, mock_create_handler: MagicMock, mock_core_v1_api: MagicMock, mock_load_config: MagicMock
    ) -> None:
        """Test verbose logging behavior."""
        mock_handler = MagicMock()
        mock_create_handler.return_value = mock_handler

        session = SandboxKubernetesSession(verbose=True)
        session.container = "test-pod"

        with (
            patch.object(session.container_api, "execute_command") as mock_execute,
            patch.object(session, "logger") as mock_logger,
        ):
            mock_execute.return_value = (0, ("stdout_output", "stderr_output"))

            session.execute_command("test command")

            # Verify verbose logging was called
            mock_logger.info.assert_any_call("Executing command: %s", "test command")
            mock_logger.info.assert_any_call("STDOUT: %s", "stdout_output")
            mock_logger.error.assert_any_call("STDERR: %s", "stderr_output")


class TestSandboxKubernetesSessionSecurity:
    """Test security-related functionality."""

    @patch("kubernetes.config.load_kube_config")
    @patch("llm_sandbox.kubernetes.CoreV1Api")
    @patch("llm_sandbox.language_handlers.factory.LanguageHandlerFactory.create_handler")
    def test_init_with_security_policy(
        self, mock_create_handler: MagicMock, mock_core_v1_api: MagicMock, mock_load_config: MagicMock
    ) -> None:
        """Test initialization with security policy."""
        mock_handler = MagicMock()
        mock_create_handler.return_value = mock_handler

        security_policy = SecurityPolicy(patterns=[], restricted_modules=[])
        session = SandboxKubernetesSession(security_policy=security_policy)

        assert session.config.security_policy == security_policy

    @patch("kubernetes.config.load_kube_config")
    @patch("llm_sandbox.kubernetes.CoreV1Api")
    @patch("llm_sandbox.language_handlers.factory.LanguageHandlerFactory.create_handler")
    def test_security_context_in_pod_manifest(
        self, mock_create_handler: MagicMock, mock_core_v1_api: MagicMock, mock_load_config: MagicMock
    ) -> None:
        """Test that pod manifest includes security context."""
        mock_handler = MagicMock()
        mock_create_handler.return_value = mock_handler

        session = SandboxKubernetesSession()

        manifest = session.pod_manifest

        # Check pod-level security context
        assert "securityContext" in manifest["spec"]
        assert manifest["spec"]["securityContext"]["runAsUser"] == 0
        assert manifest["spec"]["securityContext"]["runAsGroup"] == 0

        # Check container-level security context
        container = manifest["spec"]["containers"][0]
        assert "securityContext" in container
        assert container["securityContext"]["runAsUser"] == 0
        assert container["securityContext"]["runAsGroup"] == 0


class TestSandboxKubernetesSessionComplexScenarios:
    """Test complex scenarios and integration-like tests."""

    @patch("kubernetes.config.load_kube_config")
    @patch("llm_sandbox.kubernetes.CoreV1Api")
    @patch("llm_sandbox.language_handlers.factory.LanguageHandlerFactory.create_handler")
    @patch("time.sleep")
    def test_full_session_lifecycle(
        self,
        mock_sleep: MagicMock,
        mock_create_handler: MagicMock,
        mock_core_v1_api: MagicMock,
        mock_load_config: MagicMock,
    ) -> None:
        """Test complete session lifecycle: init -> open -> run -> close."""
        mock_handler = MagicMock()
        mock_handler.file_extension = "py"
        mock_create_handler.return_value = mock_handler

        mock_client = MagicMock()
        mock_core_v1_api.return_value = mock_client

        # Mock pod becoming running
        mock_pod = MagicMock()
        mock_pod.status.phase = "Running"
        mock_client.read_namespaced_pod.return_value = mock_pod

        session = SandboxKubernetesSession()

        # Test full lifecycle with context manager
        with (
            patch.object(session, "environment_setup") as mock_env_setup,
            patch.object(session, "install") as mock_install,
            patch.object(session, "copy_to_runtime") as mock_copy,
            patch.object(session, "execute_commands") as mock_execute_commands,
            patch("tempfile.NamedTemporaryFile") as mock_temp_file,
        ):
            # Setup mocks
            mock_file_instance = mock_temp_file.return_value.__enter__.return_value
            mock_file_instance.name = "/tmp/code.py"
            mock_execute_commands.return_value = ConsoleOutput(exit_code=0, stdout="Hello World")

            with session:
                result = session.run("print('Hello World')", ["requests"])

            # Verify lifecycle calls
            mock_client.create_namespaced_pod.assert_called_once()
            mock_env_setup.assert_called_once()
            mock_install.assert_called_once_with(["requests"])
            mock_copy.assert_called_once()
            mock_execute_commands.assert_called_once()
            mock_client.delete_namespaced_pod.assert_called_once()

            assert result.exit_code == 0
            assert result.stdout == "Hello World"

    @patch("kubernetes.config.load_kube_config")
    @patch("llm_sandbox.kubernetes.CoreV1Api")
    @patch("llm_sandbox.language_handlers.factory.LanguageHandlerFactory.create_handler")
    def test_pod_manifest_with_complex_configuration(
        self, mock_create_handler: MagicMock, mock_core_v1_api: MagicMock, mock_load_config: MagicMock
    ) -> None:
        """Test pod manifest generation with complex environment variables."""
        mock_handler = MagicMock()
        mock_create_handler.return_value = mock_handler

        complex_env_vars = {
            "API_KEY": "secret123",
            "DATABASE_URL": "postgresql://localhost:5432/db",
            "DEBUG": "true",
            "PYTHONPATH": "/app:/libs",
        }

        session = SandboxKubernetesSession(
            image="custom:latest",
            env_vars=complex_env_vars,
            workdir="/custom/workdir",
        )

        manifest = session.pod_manifest
        container = manifest["spec"]["containers"][0]

        # Verify custom image is used
        assert container["image"] == "custom:latest"

        # Verify all environment variables are present
        env_vars = {env["name"]: env["value"] for env in container["env"]}
        assert env_vars == complex_env_vars

    @patch("kubernetes.config.load_kube_config")
    @patch("llm_sandbox.kubernetes.CoreV1Api")
    @patch("llm_sandbox.language_handlers.factory.LanguageHandlerFactory.create_handler")
    def test_error_propagation_from_container_api(
        self, mock_create_handler: MagicMock, mock_core_v1_api: MagicMock, mock_load_config: MagicMock
    ) -> None:
        """Test that errors from container API are properly propagated."""
        mock_handler = MagicMock()
        mock_create_handler.return_value = mock_handler

        session = SandboxKubernetesSession()
        session.container = "test-pod"

        # Test FileNotFoundError propagation in copy operations
        with (
            tempfile.NamedTemporaryFile() as temp_file,
            patch.object(session, "_ensure_directory_exists"),
            patch.object(session.container_api, "copy_to_container", side_effect=FileNotFoundError("File not found")),
            pytest.raises(FileNotFoundError, match="File not found"),
        ):
            session.copy_to_runtime(temp_file.name, "/dest")

        # Test RuntimeError propagation
        with (
            tempfile.NamedTemporaryFile() as temp_file,
            patch.object(session, "_ensure_directory_exists"),
            patch.object(session.container_api, "copy_to_container", side_effect=RuntimeError("Runtime error")),
            pytest.raises(RuntimeError, match="Runtime error"),
        ):
            session.copy_to_runtime(temp_file.name, "/dest")


class TestSandboxKubernetesSessionDifferencesFromDocker:
    """Test differences in Kubernetes implementation compared to Docker."""

    @patch("kubernetes.config.load_kube_config")
    @patch("llm_sandbox.kubernetes.CoreV1Api")
    @patch("llm_sandbox.language_handlers.factory.LanguageHandlerFactory.create_handler")
    def test_unique_pod_naming_strategy(
        self, mock_create_handler: MagicMock, mock_core_v1_api: MagicMock, mock_load_config: MagicMock
    ) -> None:
        """Test that pod names are made unique to avoid conflicts."""
        mock_handler = MagicMock()
        mock_create_handler.return_value = mock_handler

        session1 = SandboxKubernetesSession(lang="python")
        session2 = SandboxKubernetesSession(lang="python")

        # Pod names should be unique even with same language
        assert session1.pod_name != session2.pod_name
        assert "sandbox-python" in session1.pod_name
        assert "sandbox-python" in session2.pod_name

    @patch("kubernetes.config.load_kube_config")
    @patch("llm_sandbox.kubernetes.CoreV1Api")
    @patch("llm_sandbox.language_handlers.factory.LanguageHandlerFactory.create_handler")
    def test_namespace_awareness(
        self, mock_create_handler: MagicMock, mock_core_v1_api: MagicMock, mock_load_config: MagicMock
    ) -> None:
        """Test that the session is namespace-aware unlike Docker containers."""
        mock_handler = MagicMock()
        mock_create_handler.return_value = mock_handler

        session = SandboxKubernetesSession(kube_namespace="test-namespace")

        assert session.kube_namespace == "test-namespace"
        assert session.container_api.namespace == "test-namespace"  # type: ignore[attr-defined]

    @patch("kubernetes.config.load_kube_config")
    @patch("llm_sandbox.kubernetes.CoreV1Api")
    @patch("llm_sandbox.language_handlers.factory.LanguageHandlerFactory.create_handler")
    def test_pod_manifest_customization(
        self, mock_create_handler: MagicMock, mock_core_v1_api: MagicMock, mock_load_config: MagicMock
    ) -> None:
        """Test that pod manifest can be fully customized unlike Docker containers."""
        mock_handler = MagicMock()
        mock_create_handler.return_value = mock_handler

        custom_manifest = {
            "apiVersion": "v1",
            "kind": "Pod",
            "metadata": {
                "name": "custom-pod",
                "namespace": "custom-ns",
                "labels": {"app": "test", "version": "1.0"},
            },
            "spec": {
                "containers": [
                    {
                        "name": "custom-container",
                        "image": "custom:latest",
                        "resources": {
                            "limits": {"memory": "256Mi", "cpu": "100m"},
                            "requests": {"memory": "128Mi", "cpu": "50m"},
                        },
                    }
                ],
                "restartPolicy": "Never",
            },
        }

        session = SandboxKubernetesSession(pod_manifest=custom_manifest)

        # Verify custom labels and resources are preserved
        manifest = session.pod_manifest
        assert manifest["metadata"]["labels"]["app"] == "test"
        assert manifest["metadata"]["labels"]["version"] == "1.0"
        assert "resources" in manifest["spec"]["containers"][0]


class TestSandboxKubernetesSessionExistingPod:
    """Test cases for existing pod functionality."""

    @patch("kubernetes.config.load_kube_config")
    @patch("llm_sandbox.kubernetes.CoreV1Api")
    @patch("llm_sandbox.language_handlers.factory.LanguageHandlerFactory.create_handler")
    def test_connect_to_existing_pod_not_found(
        self, mock_create_handler: MagicMock, mock_core_v1_api: MagicMock, mock_load_config: MagicMock
    ) -> None:
        """Test connecting to non-existent pod."""
        mock_handler = MagicMock()
        mock_create_handler.return_value = mock_handler

        mock_client = MagicMock()
        mock_core_v1_api.return_value = mock_client
        mock_client.read_namespaced_pod.side_effect = ApiException(status=404, reason="Not Found")

        session = SandboxKubernetesSession(container_id="non-existent-pod")

        with pytest.raises(ContainerError, match="Pod non-existent-pod not found"):
            session.open()

    @patch("kubernetes.config.load_kube_config")
    @patch("llm_sandbox.kubernetes.CoreV1Api")
    @patch("llm_sandbox.language_handlers.factory.LanguageHandlerFactory.create_handler")
    def test_connect_to_existing_pod_api_error(
        self, mock_create_handler: MagicMock, mock_core_v1_api: MagicMock, mock_load_config: MagicMock
    ) -> None:
        """Test connecting to existing pod with API error."""
        mock_handler = MagicMock()
        mock_create_handler.return_value = mock_handler

        mock_client = MagicMock()
        mock_core_v1_api.return_value = mock_client
        mock_client.read_namespaced_pod.side_effect = ApiException(status=500, reason="Server Error")

        session = SandboxKubernetesSession(container_id="test-pod")

        with pytest.raises(ContainerError, match="Failed to access pod test-pod"):
            session.open()

    @patch("kubernetes.config.load_kube_config")
    @patch("llm_sandbox.kubernetes.CoreV1Api")
    @patch("llm_sandbox.language_handlers.factory.LanguageHandlerFactory.create_handler")
    def test_connect_to_existing_pod_other_error(
        self, mock_create_handler: MagicMock, mock_core_v1_api: MagicMock, mock_load_config: MagicMock
    ) -> None:
        """Test connecting to existing pod with non-API error."""
        mock_handler = MagicMock()
        mock_create_handler.return_value = mock_handler

        mock_client = MagicMock()
        mock_core_v1_api.return_value = mock_client
        mock_client.read_namespaced_pod.side_effect = Exception("Connection failed")

        session = SandboxKubernetesSession(container_id="test-pod")

        with pytest.raises(ContainerError, match="Failed to connect to pod test-pod"):
            session.open()

    @patch("kubernetes.config.load_kube_config")
    @patch("llm_sandbox.kubernetes.CoreV1Api")
    @patch("llm_sandbox.language_handlers.factory.LanguageHandlerFactory.create_handler")
    def test_pod_pending_timeout(
        self, mock_create_handler: MagicMock, mock_core_v1_api: MagicMock, mock_load_config: MagicMock
    ) -> None:
        """Test pod timeout when waiting for pending pod to start."""
        mock_handler = MagicMock()
        mock_create_handler.return_value = mock_handler

        mock_client = MagicMock()
        mock_core_v1_api.return_value = mock_client

        # Mock pod that stays pending
        mock_pod = MagicMock()
        mock_pod.status.phase = "Pending"
        mock_client.read_namespaced_pod.return_value = mock_pod

        session = SandboxKubernetesSession(container_id="pending-pod")

        with (
            patch("llm_sandbox.kubernetes.time.sleep"),
            patch(
                "llm_sandbox.kubernetes.time.time",
                side_effect=[0] + [301] * 10,  # safely covers extra calls
            ),
            pytest.raises(ContainerError, match="Failed to connect to pod pending-pod"),
        ):
            session.open()

    @patch("kubernetes.config.load_kube_config")
    @patch("llm_sandbox.kubernetes.CoreV1Api")
    @patch("llm_sandbox.language_handlers.factory.LanguageHandlerFactory.create_handler")
    def test_pod_failed_status(
        self, mock_create_handler: MagicMock, mock_core_v1_api: MagicMock, mock_load_config: MagicMock
    ) -> None:
        """Test pod with failed status."""
        mock_handler = MagicMock()
        mock_create_handler.return_value = mock_handler

        mock_client = MagicMock()
        mock_core_v1_api.return_value = mock_client

        # Mock pod with failed status
        mock_pod = MagicMock()
        mock_pod.status.phase = "Failed"
        mock_client.read_namespaced_pod.return_value = mock_pod

        session = SandboxKubernetesSession(container_id="failed-pod")

        with pytest.raises(ContainerError, match="Pod failed-pod is not running"):
            session.open()

    @patch("kubernetes.config.load_kube_config")
    @patch("llm_sandbox.kubernetes.CoreV1Api")
    @patch("llm_sandbox.language_handlers.factory.LanguageHandlerFactory.create_handler")
    def test_mkdir_command_failure(
        self, mock_create_handler: MagicMock, mock_core_v1_api: MagicMock, mock_load_config: MagicMock
    ) -> None:
        """Test directory creation failure."""
        mock_handler = MagicMock()
        mock_create_handler.return_value = mock_handler

        mock_client = MagicMock()
        mock_core_v1_api.return_value = mock_client

        session = SandboxKubernetesSession()
        session.container = "test-pod"
        session.container_api = MagicMock()
        session.container_api.execute_command.return_value = (1, ("", "mkdir failed"))

        # Should not raise exception, just log error
        session._ensure_directory_exists("/test/path")

        session.container_api.execute_command.assert_called_once()

    @patch("kubernetes.config.load_kube_config")
    @patch("llm_sandbox.kubernetes.CoreV1Api")
    @patch("llm_sandbox.language_handlers.factory.LanguageHandlerFactory.create_handler")
    def test_close_cleanup_error(
        self, mock_create_handler: MagicMock, mock_core_v1_api: MagicMock, mock_load_config: MagicMock
    ) -> None:
        """Test error during pod cleanup in close()."""
        mock_handler = MagicMock()
        mock_create_handler.return_value = mock_handler

        session = SandboxKubernetesSession()
        session.container = "test-pod"
        session.using_existing_container = False
        session.container_api = MagicMock()
        session.container_api.stop_container.side_effect = Exception("Cleanup failed")

        # Should not raise exception, just log error
        session.close()

        session.container_api.stop_container.assert_called_once()
