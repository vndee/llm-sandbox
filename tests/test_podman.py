# ruff: noqa: SLF001, PLR2004, ARG002, PT011

"""Tests for Podman backend implementation."""

import io
import tarfile
from unittest.mock import MagicMock, Mock, patch

import pytest

from llm_sandbox.const import DefaultImage, SupportedLanguage
from llm_sandbox.data import ConsoleOutput
from llm_sandbox.exceptions import (
    CommandEmptyError,
    ExtraArgumentsError,
    ImageNotFoundError,
    ImagePullError,
    NotOpenSessionError,
)
from llm_sandbox.podman import SandboxPodmanSession
from llm_sandbox.security import SecurityPolicy


class TestSandboxPodmanSessionInit:
    """Test SandboxPodmanSession initialization."""

    @patch("llm_sandbox.podman.PodmanClient.from_env")
    @patch("llm_sandbox.language_handlers.factory.LanguageHandlerFactory.create_handler")
    def test_init_with_defaults(self, mock_create_handler: MagicMock, mock_podman_from_env: MagicMock) -> None:
        """
        Tests that SandboxPodmanSession initializes with default parameters and creates a Podman client from the environment.
        """
        mock_handler = MagicMock()
        mock_create_handler.return_value = mock_handler
        mock_client = MagicMock()
        mock_podman_from_env.return_value = mock_client

        session = SandboxPodmanSession()

        assert session.lang == SupportedLanguage.PYTHON
        assert session.verbose is False
        assert session.image == DefaultImage.PYTHON
        assert session.keep_template is False
        assert session.commit_container is False
        assert session.stream is True
        assert session.workdir == "/sandbox"
        assert session.client == mock_client
        mock_podman_from_env.assert_called_once()

    @patch("llm_sandbox.podman.PodmanClient.from_env")
    @patch("llm_sandbox.language_handlers.factory.LanguageHandlerFactory.create_handler")
    def test_init_with_custom_client(self, mock_create_handler: MagicMock, mock_podman_from_env: MagicMock) -> None:
        """Test initialization with custom Podman client."""
        mock_handler = MagicMock()
        mock_create_handler.return_value = mock_handler
        custom_client = MagicMock()

        session = SandboxPodmanSession(client=custom_client)

        assert session.client == custom_client
        mock_podman_from_env.assert_not_called()

    @patch("llm_sandbox.podman.PodmanClient.from_env")
    @patch("llm_sandbox.language_handlers.factory.LanguageHandlerFactory.create_handler")
    def test_init_with_custom_params(self, mock_create_handler: MagicMock, mock_podman_from_env: MagicMock) -> None:
        """Test initialization with custom parameters."""
        mock_handler = MagicMock()
        mock_create_handler.return_value = mock_handler
        mock_client = MagicMock()
        mock_podman_from_env.return_value = mock_client
        security_policy = SecurityPolicy(patterns=[], restricted_modules=[])

        session = SandboxPodmanSession(
            image="custom:latest",
            lang="java",
            keep_template=True,
            commit_container=True,
            verbose=False,
            stream=False,
            workdir="/custom",
            security_policy=security_policy,
        )

        assert session.image == "custom:latest"
        assert session.lang == "java"
        assert session.keep_template is True
        assert session.commit_container is True
        assert session.verbose is False
        assert session.stream is False
        assert session.workdir == "/custom"
        assert session.security_policy == security_policy

    @patch("llm_sandbox.podman.PodmanClient.from_env")
    @patch("llm_sandbox.language_handlers.factory.LanguageHandlerFactory.create_handler")
    def test_init_with_dockerfile_and_image_raises_error(
        self, mock_create_handler: MagicMock, mock_podman_from_env: MagicMock
    ) -> None:
        """Test initialization fails when both dockerfile and image are provided."""
        mock_handler = MagicMock()
        mock_create_handler.return_value = mock_handler

        with pytest.raises(ExtraArgumentsError, match="Only one of `image` or `dockerfile` can be provided"):
            SandboxPodmanSession(image="test:latest", dockerfile="/path/to/Containerfile")

    @patch("llm_sandbox.podman.PodmanClient.from_env")
    @patch("llm_sandbox.language_handlers.factory.LanguageHandlerFactory.create_handler")
    def test_init_with_dockerfile_only(self, mock_create_handler: MagicMock, mock_podman_from_env: MagicMock) -> None:
        """Test initialization with dockerfile only."""
        mock_handler = MagicMock()
        mock_create_handler.return_value = mock_handler

        session = SandboxPodmanSession(dockerfile="/path/to/Containerfile")

        assert session.dockerfile == "/path/to/Containerfile"
        assert session.image is None


class TestSandboxPodmanSessionOpen:
    """Test SandboxPodmanSession open functionality."""

    @patch("llm_sandbox.podman.PodmanClient.from_env")
    @patch("llm_sandbox.language_handlers.factory.LanguageHandlerFactory.create_handler")
    def test_open_with_existing_image(self, mock_create_handler: MagicMock, mock_podman_from_env: MagicMock) -> None:
        """
        Tests that opening a session with an existing image retrieves the image, creates and starts a container, sets up the environment, and assigns the container and image to the session.
        """
        mock_handler = MagicMock()
        mock_create_handler.return_value = mock_handler
        mock_client = MagicMock()
        mock_podman_from_env.return_value = mock_client

        mock_image = MagicMock()
        mock_image.tags = [DefaultImage.PYTHON]
        mock_client.images.get.return_value = mock_image

        mock_container = MagicMock()
        mock_client.containers.create.return_value = mock_container

        session = SandboxPodmanSession()

        with patch.object(session, "environment_setup") as mock_env_setup:
            session.open()

        mock_client.images.get.assert_called_once_with(DefaultImage.PYTHON)
        mock_client.containers.create.assert_called_once()
        mock_container.start.assert_called_once()
        mock_env_setup.assert_called_once()
        assert session.container == mock_container
        assert session.docker_image == mock_image

    @patch("llm_sandbox.podman.PodmanClient.from_env")
    @patch("llm_sandbox.language_handlers.factory.LanguageHandlerFactory.create_handler")
    def test_open_with_image_pull_single_image(
        self, mock_create_handler: MagicMock, mock_podman_from_env: MagicMock
    ) -> None:
        """
        Tests that opening a session pulls the image when not found locally and the pull returns a single image object.
        
        Verifies that the image is pulled, the session is marked as having created a template, and the pulled image is set as the session's Docker image.
        """
        mock_handler = MagicMock()
        mock_create_handler.return_value = mock_handler
        mock_client = MagicMock()
        mock_podman_from_env.return_value = mock_client

        # Image not found locally, but pull succeeds returning single image
        from podman.domain.images import Image as PodmanImage  # Import for spec
        from podman.errors import ImageNotFound

        mock_client.images.get.side_effect = ImageNotFound("Image not found")

        mock_image = Mock(spec=PodmanImage)  # Use Mock with spec
        mock_image.tags = [DefaultImage.PYTHON]
        mock_client.images.pull.return_value = mock_image

        mock_container = MagicMock()
        mock_client.containers.create.return_value = mock_container

        session = SandboxPodmanSession(verbose=True)

        with patch.object(session, "environment_setup"):
            session.open()

        mock_client.images.pull.assert_called_once_with(DefaultImage.PYTHON)
        assert session.is_create_template is True
        assert session.docker_image == mock_image

    @patch("llm_sandbox.podman.PodmanClient.from_env")
    @patch("llm_sandbox.language_handlers.factory.LanguageHandlerFactory.create_handler")
    def test_open_with_image_pull_list_of_images(
        self, mock_create_handler: MagicMock, mock_podman_from_env: MagicMock
    ) -> None:
        """
        Tests that opening a session pulls the image when not found locally and handles the case where the pull returns a list of images.
        
        Verifies that the session uses the correct image from the returned list.
        """
        mock_handler = MagicMock()
        mock_create_handler.return_value = mock_handler
        mock_client = MagicMock()
        mock_podman_from_env.return_value = mock_client

        from podman.errors import ImageNotFound

        mock_client.images.get.side_effect = ImageNotFound("Image not found")

        mock_image = MagicMock()
        mock_image.tags = [DefaultImage.PYTHON]
        mock_client.images.pull.return_value = [mock_image]  # Returns list

        mock_container = MagicMock()
        mock_client.containers.create.return_value = mock_container

        session = SandboxPodmanSession()

        with patch.object(session, "environment_setup"):
            session.open()

        assert session.docker_image == mock_image

    @patch("llm_sandbox.podman.PodmanClient.from_env")
    @patch("llm_sandbox.language_handlers.factory.LanguageHandlerFactory.create_handler")
    def test_open_with_pull_unexpected_return_type(
        self, mock_create_handler: MagicMock, mock_podman_from_env: MagicMock
    ) -> None:
        """Test opening session when image pull returns unexpected type."""
        mock_handler = MagicMock()
        mock_create_handler.return_value = mock_handler
        mock_client = MagicMock()
        mock_podman_from_env.return_value = mock_client

        from podman.errors import ImageNotFound

        mock_client.images.get.side_effect = ImageNotFound("Image not found")
        mock_client.images.pull.return_value = "unexpected_type"  # Unexpected return type

        session = SandboxPodmanSession()

        with pytest.raises(ImageNotFoundError):
            session.open()

    @patch("llm_sandbox.podman.PodmanClient.from_env")
    @patch("llm_sandbox.language_handlers.factory.LanguageHandlerFactory.create_handler")
    def test_open_with_pull_failure(self, mock_create_handler: MagicMock, mock_podman_from_env: MagicMock) -> None:
        """Test opening session when image pull fails."""
        mock_handler = MagicMock()
        mock_create_handler.return_value = mock_handler
        mock_client = MagicMock()
        mock_podman_from_env.return_value = mock_client

        from podman.errors import ImageNotFound

        mock_client.images.get.side_effect = ImageNotFound("Image not found")
        mock_client.images.pull.side_effect = Exception("Pull failed")

        session = SandboxPodmanSession()

        with pytest.raises(ImagePullError, match="Failed to pull image"):
            session.open()

    @patch("llm_sandbox.podman.PodmanClient.from_env")
    @patch("llm_sandbox.language_handlers.factory.LanguageHandlerFactory.create_handler")
    @patch("llm_sandbox.podman.Path")
    def test_open_with_dockerfile(
        self, mock_path: MagicMock, mock_create_handler: MagicMock, mock_podman_from_env: MagicMock
    ) -> None:
        """Test opening session with dockerfile."""
        mock_handler = MagicMock()
        mock_create_handler.return_value = mock_handler
        mock_client = MagicMock()
        mock_podman_from_env.return_value = mock_client

        # Mock Path behavior
        mock_dockerfile_path = MagicMock()
        mock_dockerfile_path.parent = "/path/to"
        mock_dockerfile_path.name = "Containerfile"
        mock_path.return_value = mock_dockerfile_path

        mock_image = MagicMock()
        mock_build_logs = ["Build log"]
        mock_client.images.build.return_value = (mock_image, mock_build_logs)

        mock_container = MagicMock()
        mock_client.containers.create.return_value = mock_container

        session = SandboxPodmanSession(dockerfile="/path/to/Containerfile")

        with patch.object(session, "environment_setup"):
            session.open()

        mock_client.images.build.assert_called_once()
        assert session.is_create_template is True


class TestSandboxPodmanSessionClose:
    """Test SandboxPodmanSession close functionality."""

    @patch("llm_sandbox.podman.PodmanClient.from_env")
    @patch("llm_sandbox.language_handlers.factory.LanguageHandlerFactory.create_handler")
    def test_close_simple(self, mock_create_handler: MagicMock, mock_podman_from_env: MagicMock) -> None:
        """Test simple close without commit."""
        mock_handler = MagicMock()
        mock_create_handler.return_value = mock_handler
        mock_client = MagicMock()
        mock_podman_from_env.return_value = mock_client

        session = SandboxPodmanSession()

        mock_container = MagicMock()
        session.container = mock_container

        session.close()

        mock_container.stop.assert_called_once()
        mock_container.wait.assert_called_once()
        mock_container.remove.assert_called_once_with(force=True)

    @patch("llm_sandbox.podman.PodmanClient.from_env")
    @patch("llm_sandbox.language_handlers.factory.LanguageHandlerFactory.create_handler")
    def test_close_with_commit(self, mock_create_handler: MagicMock, mock_podman_from_env: MagicMock) -> None:
        """Test close with container commit."""
        mock_handler = MagicMock()
        mock_create_handler.return_value = mock_handler
        mock_client = MagicMock()
        mock_podman_from_env.return_value = mock_client

        session = SandboxPodmanSession(commit_container=True)

        mock_container = MagicMock()
        # Mock Image object for commit scenario
        from llm_sandbox.podman import Image

        mock_image = Mock(spec=Image)
        mock_image.tags = ["test:latest"]
        session.container = mock_container
        session.image = mock_image

        session.close()

        mock_container.commit.assert_called_once_with(repository="test", tag="latest")

    @patch("llm_sandbox.podman.PodmanClient.from_env")
    @patch("llm_sandbox.language_handlers.factory.LanguageHandlerFactory.create_handler")
    def test_close_with_image_removal(self, mock_create_handler: MagicMock, mock_podman_from_env: MagicMock) -> None:
        """Test close with image removal."""
        mock_handler = MagicMock()
        mock_create_handler.return_value = mock_handler
        mock_client = MagicMock()
        mock_podman_from_env.return_value = mock_client

        session = SandboxPodmanSession(keep_template=False)
        session.is_create_template = True

        mock_container = MagicMock()
        mock_image = MagicMock()
        mock_image.id = "image_id_123"
        mock_client.containers.list.return_value = []  # No other containers using image
        session.container = mock_container
        session.docker_image = mock_image

        session.close()

        mock_image.remove.assert_called_once_with(force=True)


class TestSandboxPodmanSessionRun:
    """Test SandboxPodmanSession run functionality."""

    @patch("llm_sandbox.podman.PodmanClient.from_env")
    @patch("llm_sandbox.language_handlers.factory.LanguageHandlerFactory.create_handler")
    def test_run_success(self, mock_create_handler: MagicMock, mock_podman_from_env: MagicMock) -> None:
        """Test successful code execution."""
        mock_handler = MagicMock()
        mock_handler.file_extension = "py"
        mock_handler.get_execution_commands.return_value = ["python /sandbox/code.py"]
        mock_create_handler.return_value = mock_handler

        session = SandboxPodmanSession()

        mock_container = MagicMock()
        session.container = mock_container

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
            mock_copy.assert_called_once_with(mock_file_instance.name, "/sandbox/code.py")
            mock_execute.assert_called_once_with(["python /sandbox/code.py"], workdir="/sandbox")

    @patch("llm_sandbox.podman.PodmanClient.from_env")
    @patch("llm_sandbox.language_handlers.factory.LanguageHandlerFactory.create_handler")
    def test_run_without_open_session(self, mock_create_handler: MagicMock, mock_podman_from_env: MagicMock) -> None:
        """Test run fails when session is not open."""
        mock_handler = MagicMock()
        mock_create_handler.return_value = mock_handler

        session = SandboxPodmanSession()
        session.container = None

        with pytest.raises(NotOpenSessionError):
            session.run("print('hello')")


class TestSandboxPodmanSessionFileOperations:
    """Test SandboxPodmanSession file operations."""

    @patch("llm_sandbox.podman.PodmanClient.from_env")
    @patch("llm_sandbox.language_handlers.factory.LanguageHandlerFactory.create_handler")
    def test_copy_to_runtime(self, mock_create_handler: MagicMock, mock_podman_from_env: MagicMock) -> None:
        """Test copying file to container."""
        mock_handler = MagicMock()
        mock_create_handler.return_value = mock_handler

        session = SandboxPodmanSession(runtime_configs={"user": "1000:1000"})

        mock_container = MagicMock()
        mock_container.exec_run.return_value = (1, b"")  # Directory doesn't exist
        session.container = mock_container

        with (
            patch("tarfile.open") as mock_tar_open,
            patch("io.BytesIO") as mock_bytesio,
        ):
            mock_tar = MagicMock()
            mock_tar_open.return_value.__enter__.return_value = mock_tar
            mock_stream = MagicMock()
            mock_stream.getvalue.return_value = b"tar_content"
            mock_bytesio.return_value = mock_stream

            session.copy_to_runtime("/host/file.txt", "/container/file.txt")

            mock_container.exec_run.assert_any_call("mkdir -p /container")
            mock_container.put_archive.assert_called_once_with("/container", b"tar_content")
            # Should change ownership since user is non-root
            ownership_calls = [call for call in mock_container.exec_run.call_args_list if "chown" in str(call)]
            assert len(ownership_calls) >= 1

    @patch("llm_sandbox.podman.PodmanClient.from_env")
    @patch("llm_sandbox.language_handlers.factory.LanguageHandlerFactory.create_handler")
    def test_copy_to_runtime_no_container(
        self, mock_create_handler: MagicMock, mock_podman_from_env: MagicMock
    ) -> None:
        """Test copy_to_runtime fails when no container."""
        mock_handler = MagicMock()
        mock_create_handler.return_value = mock_handler

        session = SandboxPodmanSession()
        session.container = None

        with pytest.raises(NotOpenSessionError):
            session.copy_to_runtime("/host/file.txt", "/container/file.txt")

    @patch("llm_sandbox.podman.PodmanClient.from_env")
    @patch("llm_sandbox.language_handlers.factory.LanguageHandlerFactory.create_handler")
    def test_copy_from_runtime(self, mock_create_handler: MagicMock, mock_podman_from_env: MagicMock) -> None:
        """Test copying file from container."""
        mock_handler = MagicMock()
        mock_create_handler.return_value = mock_handler

        session = SandboxPodmanSession()

        mock_container = MagicMock()

        # Create mock tar data
        file_content = b"test content"
        tar_buffer = io.BytesIO()
        with tarfile.open(fileobj=tar_buffer, mode="w") as tar:
            info = tarfile.TarInfo("file.txt")
            info.size = len(file_content)
            tar.addfile(info, io.BytesIO(file_content))
        tar_data = tar_buffer.getvalue()

        mock_container.get_archive.return_value = ([tar_data], {"size": len(tar_data)})
        session.container = mock_container

        with patch("tarfile.open") as mock_tar_open:
            mock_tar = MagicMock()
            mock_member = MagicMock()
            mock_member.name = "file.txt"
            mock_member.isfile.return_value = True
            mock_tar.getmembers.return_value = [mock_member]
            mock_tar_open.return_value.__enter__.return_value = mock_tar

            session.copy_from_runtime("/container/file.txt", "/host/file.txt")

            mock_container.get_archive.assert_called_once_with("/container/file.txt")
            mock_tar.extractall.assert_called_once()

    @patch("llm_sandbox.podman.PodmanClient.from_env")
    @patch("llm_sandbox.language_handlers.factory.LanguageHandlerFactory.create_handler")
    def test_copy_from_runtime_file_not_found(
        self, mock_create_handler: MagicMock, mock_podman_from_env: MagicMock
    ) -> None:
        """Test copy_from_runtime when file not found."""
        mock_handler = MagicMock()
        mock_create_handler.return_value = mock_handler

        session = SandboxPodmanSession()

        mock_container = MagicMock()
        mock_container.get_archive.return_value = ([], {"size": 0})
        session.container = mock_container

        with pytest.raises(FileNotFoundError):
            session.copy_from_runtime("/container/missing.txt", "/host/file.txt")


class TestSandboxPodmanSessionCommands:
    """Test SandboxPodmanSession command execution."""

    @patch("llm_sandbox.podman.PodmanClient.from_env")
    @patch("llm_sandbox.language_handlers.factory.LanguageHandlerFactory.create_handler")
    def test_execute_command_success_no_stream(
        self, mock_create_handler: MagicMock, mock_podman_from_env: MagicMock
    ) -> None:
        """Test successful command execution without streaming."""
        mock_handler = MagicMock()
        mock_create_handler.return_value = mock_handler

        session = SandboxPodmanSession(stream=False)

        mock_container = MagicMock()
        mock_container.exec_run.return_value = (0, (b"stdout content", b"stderr content"))
        session.container = mock_container

        result = session.execute_command("ls -l", workdir="/tmp")

        assert result.exit_code == 0
        assert result.stdout == "stdout content"
        assert result.stderr == "stderr content"
        mock_container.exec_run.assert_called_once_with(
            "ls -l",
            stream=False,
            tty=False,
            workdir="/tmp",
            stderr=True,
            stdout=True,
            demux=True,
        )

    @patch("llm_sandbox.podman.PodmanClient.from_env")
    @patch("llm_sandbox.language_handlers.factory.LanguageHandlerFactory.create_handler")
    def test_execute_command_success_with_stream(
        self, mock_create_handler: MagicMock, mock_podman_from_env: MagicMock
    ) -> None:
        """Test successful command execution with streaming."""
        mock_handler = MagicMock()
        mock_create_handler.return_value = mock_handler

        session = SandboxPodmanSession(stream=True)

        mock_container = MagicMock()
        # Mock streaming output with various data types that Podman might return
        mock_output = [
            (b"stdout chunk 1", None),
            (None, b"stderr chunk 1"),
            ([b"stdout ", b"chunk 2"], [b"stderr ", b"chunk 2"]),  # List format
            (b"final stdout", b"final stderr"),
        ]
        mock_container.exec_run.return_value = (0, iter(mock_output))
        session.container = mock_container

        result = session.execute_command("ls -l")

        assert result.exit_code == 0
        assert result.stdout == "stdout chunk 1stdout chunk 2final stdout"
        assert result.stderr == "stderr chunk 1stderr chunk 2final stderr"

    @patch("llm_sandbox.podman.PodmanClient.from_env")
    @patch("llm_sandbox.language_handlers.factory.LanguageHandlerFactory.create_handler")
    def test_execute_command_with_string_chunks(
        self, mock_create_handler: MagicMock, mock_podman_from_env: MagicMock
    ) -> None:
        """Test command execution with string chunks in streaming."""
        mock_handler = MagicMock()
        mock_create_handler.return_value = mock_handler

        session = SandboxPodmanSession(stream=True)

        mock_container = MagicMock()
        # Mock streaming with string chunks instead of bytes
        mock_output = [
            ("string stdout", "string stderr"),
        ]
        mock_container.exec_run.return_value = (0, iter(mock_output))
        session.container = mock_container

        result = session.execute_command("echo test")

        assert result.exit_code == 0
        assert result.stdout == "string stdout"
        assert result.stderr == "string stderr"

    @patch("llm_sandbox.podman.PodmanClient.from_env")
    @patch("llm_sandbox.language_handlers.factory.LanguageHandlerFactory.create_handler")
    def test_execute_command_empty_command(
        self, mock_create_handler: MagicMock, mock_podman_from_env: MagicMock
    ) -> None:
        """Test execute_command with empty command."""
        mock_handler = MagicMock()
        mock_create_handler.return_value = mock_handler

        session = SandboxPodmanSession()

        with pytest.raises(CommandEmptyError):
            session.execute_command("")

    @patch("llm_sandbox.podman.PodmanClient.from_env")
    @patch("llm_sandbox.language_handlers.factory.LanguageHandlerFactory.create_handler")
    def test_execute_command_no_container(
        self, mock_create_handler: MagicMock, mock_podman_from_env: MagicMock
    ) -> None:
        """Test execute_command fails when no container."""
        mock_handler = MagicMock()
        mock_create_handler.return_value = mock_handler

        session = SandboxPodmanSession()
        session.container = None

        with pytest.raises(NotOpenSessionError):
            session.execute_command("ls")

    @patch("llm_sandbox.podman.PodmanClient.from_env")
    @patch("llm_sandbox.language_handlers.factory.LanguageHandlerFactory.create_handler")
    def test_execute_command_none_exit_code(
        self, mock_create_handler: MagicMock, mock_podman_from_env: MagicMock
    ) -> None:
        """Test execute_command when exit code is None."""
        mock_handler = MagicMock()
        mock_create_handler.return_value = mock_handler

        session = SandboxPodmanSession(stream=False)

        mock_container = MagicMock()
        mock_container.exec_run.return_value = (None, (b"stdout", b"stderr"))  # None exit code
        session.container = mock_container

        result = session.execute_command("test command")

        assert result.exit_code == 0  # Should default to 0


class TestSandboxPodmanSessionArchive:
    """Test SandboxPodmanSession archive operations."""

    @patch("llm_sandbox.podman.PodmanClient.from_env")
    @patch("llm_sandbox.language_handlers.factory.LanguageHandlerFactory.create_handler")
    def test_get_archive(self, mock_create_handler: MagicMock, mock_podman_from_env: MagicMock) -> None:
        """Test getting archive from container."""
        mock_handler = MagicMock()
        mock_create_handler.return_value = mock_handler

        session = SandboxPodmanSession()

        mock_container = MagicMock()
        mock_data = [b"archive", b"content"]
        mock_stat = {"size": 100, "name": "file.txt"}
        mock_container.get_archive.return_value = (mock_data, mock_stat)
        session.container = mock_container

        data, stat = session.get_archive("/container/path")

        assert data == b"archivecontent"
        assert stat == mock_stat
        mock_container.get_archive.assert_called_once_with("/container/path")

    @patch("llm_sandbox.podman.PodmanClient.from_env")
    @patch("llm_sandbox.language_handlers.factory.LanguageHandlerFactory.create_handler")
    def test_get_archive_no_container(self, mock_create_handler: MagicMock, mock_podman_from_env: MagicMock) -> None:
        """Test get_archive fails when no container."""
        mock_handler = MagicMock()
        mock_create_handler.return_value = mock_handler

        session = SandboxPodmanSession()
        session.container = None

        with pytest.raises(NotOpenSessionError):
            session.get_archive("/container/path")


class TestSandboxPodmanSessionContextManager:
    """Test SandboxPodmanSession context manager functionality."""

    @patch("llm_sandbox.podman.PodmanClient.from_env")
    @patch("llm_sandbox.language_handlers.factory.LanguageHandlerFactory.create_handler")
    def test_context_manager(self, mock_create_handler: MagicMock, mock_podman_from_env: MagicMock) -> None:
        """Test using session as context manager."""
        mock_handler = MagicMock()
        mock_create_handler.return_value = mock_handler

        session = SandboxPodmanSession()

        with (
            patch.object(session, "open") as mock_open,
            patch.object(session, "close") as mock_close,
        ):
            with session as s:
                assert s == session
                mock_open.assert_called_once()

            mock_close.assert_called_once()

    @patch("llm_sandbox.podman.PodmanClient.from_env")
    @patch("llm_sandbox.language_handlers.factory.LanguageHandlerFactory.create_handler")
    def test_context_manager_with_exception(
        self, mock_create_handler: MagicMock, mock_podman_from_env: MagicMock
    ) -> None:
        """Test context manager ensures close is called even with exception."""
        mock_handler = MagicMock()
        mock_create_handler.return_value = mock_handler

        session = SandboxPodmanSession()

        with (
            patch.object(session, "open") as mock_open,
            patch.object(session, "close") as mock_close,
        ):
            with pytest.raises(ValueError), session:
                raise ValueError

            mock_open.assert_called_once()
            mock_close.assert_called_once()


class TestSandboxPodmanSessionOwnership:
    """Test SandboxPodmanSession ownership management."""

    @patch("llm_sandbox.podman.PodmanClient.from_env")
    @patch("llm_sandbox.language_handlers.factory.LanguageHandlerFactory.create_handler")
    def test_ensure_ownership_with_non_root_user(
        self, mock_create_handler: MagicMock, mock_podman_from_env: MagicMock
    ) -> None:
        """Test _ensure_ownership with non-root user."""
        mock_handler = MagicMock()
        mock_create_handler.return_value = mock_handler

        session = SandboxPodmanSession(runtime_configs={"user": "1000:1000"})

        mock_container = MagicMock()
        session.container = mock_container

        session._ensure_ownership(["/tmp/test", "/tmp/test2"])

        mock_container.exec_run.assert_called_once_with("chown -R 1000:1000 /tmp/test /tmp/test2", user="root")

    @patch("llm_sandbox.podman.PodmanClient.from_env")
    @patch("llm_sandbox.language_handlers.factory.LanguageHandlerFactory.create_handler")
    def test_ensure_ownership_with_root_user(
        self, mock_create_handler: MagicMock, mock_podman_from_env: MagicMock
    ) -> None:
        """Test _ensure_ownership with root user (no-op)."""
        mock_handler = MagicMock()
        mock_create_handler.return_value = mock_handler

        session = SandboxPodmanSession(runtime_configs={"user": "root"})

        mock_container = MagicMock()
        session.container = mock_container

        session._ensure_ownership(["/tmp/test"])

        mock_container.exec_run.assert_not_called()

    @patch("llm_sandbox.podman.PodmanClient.from_env")
    @patch("llm_sandbox.language_handlers.factory.LanguageHandlerFactory.create_handler")
    def test_ensure_ownership_no_runtime_configs(
        self, mock_create_handler: MagicMock, mock_podman_from_env: MagicMock
    ) -> None:
        """Test _ensure_ownership with no runtime configs (default root)."""
        mock_handler = MagicMock()
        mock_create_handler.return_value = mock_handler

        session = SandboxPodmanSession()

        mock_container = MagicMock()
        session.container = mock_container

        session._ensure_ownership(["/tmp/test"])

        mock_container.exec_run.assert_not_called()
