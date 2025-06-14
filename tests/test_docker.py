# ruff: noqa: SLF001, PLR2004, ARG002, PT011

"""Tests for Docker backend implementation."""

import io
import tarfile
import tempfile
from collections.abc import Generator
from unittest.mock import MagicMock, Mock, patch

import pytest
from docker.errors import ImageNotFound, NotFound
from pydantic_core import ValidationError

from llm_sandbox.const import DefaultImage, SupportedLanguage
from llm_sandbox.data import ConsoleOutput
from llm_sandbox.docker import DockerContainerAPI, SandboxDockerSession
from llm_sandbox.exceptions import CommandEmptyError, ContainerError, ImagePullError, NotOpenSessionError
from llm_sandbox.security import SecurityPolicy


class TestSandboxDockerSessionInit:
    """Test SandboxDockerSession initialization."""

    @patch("llm_sandbox.docker.docker.from_env")
    @patch("llm_sandbox.language_handlers.factory.LanguageHandlerFactory.create_handler")
    def test_init_with_defaults(self, mock_create_handler: MagicMock, mock_docker_from_env: MagicMock) -> None:
        """Test initialization with default parameters."""
        mock_handler = MagicMock()
        mock_create_handler.return_value = mock_handler
        mock_client = MagicMock()
        mock_docker_from_env.return_value = mock_client

        session = SandboxDockerSession()

        assert session.config.lang == SupportedLanguage.PYTHON
        assert session.config.verbose is False
        assert session.config.image is None  # Image is set during _prepare_image()
        assert session.keep_template is False
        assert session.commit_container is False
        assert session.stream is False
        assert session.config.workdir == "/sandbox"
        assert session.client == mock_client
        mock_docker_from_env.assert_called_once()

    @patch("llm_sandbox.docker.docker.from_env")
    @patch("llm_sandbox.language_handlers.factory.LanguageHandlerFactory.create_handler")
    def test_init_with_custom_client(self, mock_create_handler: MagicMock, mock_docker_from_env: MagicMock) -> None:
        """Test initialization with custom Docker client."""
        mock_handler = MagicMock()
        mock_create_handler.return_value = mock_handler
        custom_client = MagicMock()

        session = SandboxDockerSession(client=custom_client)

        assert session.client == custom_client
        mock_docker_from_env.assert_not_called()

    @patch("llm_sandbox.docker.docker.from_env")
    @patch("llm_sandbox.language_handlers.factory.LanguageHandlerFactory.create_handler")
    def test_init_with_custom_params(self, mock_create_handler: MagicMock, mock_docker_from_env: MagicMock) -> None:
        """Test initialization with custom parameters."""
        mock_handler = MagicMock()
        mock_create_handler.return_value = mock_handler
        mock_client = MagicMock()
        mock_docker_from_env.return_value = mock_client
        security_policy = SecurityPolicy(patterns=[], restricted_modules=[])

        session = SandboxDockerSession(
            image="custom:latest",
            lang="java",
            keep_template=True,
            commit_container=True,
            verbose=False,
            stream=False,
            workdir="/custom",
            security_policy=security_policy,
        )

        assert session.config.image == "custom:latest"
        assert session.config.lang == SupportedLanguage.JAVA
        assert session.keep_template is True
        assert session.commit_container is True
        assert session.config.verbose is False
        assert session.stream is False
        assert session.config.workdir == "/custom"
        assert session.config.security_policy == security_policy

    @patch("llm_sandbox.docker.docker.from_env")
    @patch("llm_sandbox.language_handlers.factory.LanguageHandlerFactory.create_handler")
    def test_init_with_dockerfile_and_image_raises_error(
        self, mock_create_handler: MagicMock, mock_docker_from_env: MagicMock
    ) -> None:
        """Test initialization fails when both dockerfile and image are provided."""
        mock_handler = MagicMock()
        mock_create_handler.return_value = mock_handler

        with pytest.raises(ValidationError, match="Only one of"):
            SandboxDockerSession(image="test:latest", dockerfile="/path/to/Dockerfile")

    @patch("llm_sandbox.docker.docker.from_env")
    @patch("llm_sandbox.language_handlers.factory.LanguageHandlerFactory.create_handler")
    def test_init_with_dockerfile_only(self, mock_create_handler: MagicMock, mock_docker_from_env: MagicMock) -> None:
        """Test initialization with dockerfile only."""
        mock_handler = MagicMock()
        mock_create_handler.return_value = mock_handler

        session = SandboxDockerSession(dockerfile="/path/to/Dockerfile")

        assert session.config.dockerfile == "/path/to/Dockerfile"
        assert session.config.image is None


class TestSandboxDockerSessionOpen:
    """Test SandboxDockerSession open functionality."""

    @patch("llm_sandbox.docker.docker.from_env")
    @patch("llm_sandbox.language_handlers.factory.LanguageHandlerFactory.create_handler")
    def test_open_with_existing_image(self, mock_create_handler: MagicMock, mock_docker_from_env: MagicMock) -> None:
        """Test opening session with existing image."""
        mock_handler = MagicMock()
        mock_create_handler.return_value = mock_handler
        mock_client = MagicMock()
        mock_docker_from_env.return_value = mock_client

        mock_image = MagicMock()
        mock_image.tags = [DefaultImage.PYTHON]
        mock_client.images.get.return_value = mock_image

        mock_container = MagicMock()
        mock_client.containers.create.return_value = mock_container

        session = SandboxDockerSession()

        with patch.object(session, "environment_setup") as mock_env_setup:
            session.open()

        mock_client.images.get.assert_called_once_with(DefaultImage.PYTHON)
        mock_client.containers.create.assert_called_once()
        mock_env_setup.assert_called_once()
        assert session.container == mock_container
        assert session.docker_image == mock_image

    @patch("llm_sandbox.docker.docker.from_env")
    @patch("llm_sandbox.language_handlers.factory.LanguageHandlerFactory.create_handler")
    def test_open_with_image_pull(self, mock_create_handler: MagicMock, mock_docker_from_env: MagicMock) -> None:
        """Test opening session when image needs to be pulled."""
        mock_handler = MagicMock()
        mock_create_handler.return_value = mock_handler
        mock_client = MagicMock()
        mock_docker_from_env.return_value = mock_client

        # Image not found locally, but pull succeeds
        mock_client.images.get.side_effect = ImageNotFound("Image not found")

        mock_image = MagicMock()
        mock_image.tags = [DefaultImage.PYTHON]
        mock_client.images.pull.return_value = mock_image

        mock_container = MagicMock()
        mock_client.containers.create.return_value = mock_container

        session = SandboxDockerSession(verbose=True)

        with patch.object(session, "environment_setup"):
            session.open()

        mock_client.images.pull.assert_called_once_with(DefaultImage.PYTHON)
        assert session.is_create_template is True

    @patch("llm_sandbox.docker.docker.from_env")
    @patch("llm_sandbox.language_handlers.factory.LanguageHandlerFactory.create_handler")
    def test_open_with_pull_failure(self, mock_create_handler: MagicMock, mock_docker_from_env: MagicMock) -> None:
        """Test opening session when image pull fails."""
        mock_handler = MagicMock()
        mock_create_handler.return_value = mock_handler
        mock_client = MagicMock()
        mock_docker_from_env.return_value = mock_client

        mock_client.images.get.side_effect = ImageNotFound("Image not found")
        mock_client.images.pull.side_effect = Exception("Pull failed")

        session = SandboxDockerSession()

        with pytest.raises(ImagePullError, match="Failed to pull image"):
            session.open()

    @patch("llm_sandbox.docker.docker.from_env")
    @patch("llm_sandbox.language_handlers.factory.LanguageHandlerFactory.create_handler")
    @patch("llm_sandbox.docker.Path")
    def test_open_with_dockerfile(
        self, mock_path: MagicMock, mock_create_handler: MagicMock, mock_docker_from_env: MagicMock
    ) -> None:
        """Test opening session with dockerfile."""
        mock_handler = MagicMock()
        mock_create_handler.return_value = mock_handler
        mock_client = MagicMock()
        mock_docker_from_env.return_value = mock_client

        # Mock Path behavior
        mock_dockerfile_path = MagicMock()
        # Use a string to avoid Mock serialization issues
        mock_dockerfile_path.parent = "/path/to"
        mock_dockerfile_path.name = "Dockerfile"
        mock_path.return_value = mock_dockerfile_path

        mock_image = MagicMock()
        mock_build_logs = ["Build log"]
        mock_client.images.build.return_value = (mock_image, mock_build_logs)

        mock_container = MagicMock()
        mock_client.containers.create.return_value = mock_container

        session = SandboxDockerSession(dockerfile="/path/to/Dockerfile")

        with patch.object(session, "environment_setup"):
            session.open()

        mock_client.images.build.assert_called_once()
        assert session.is_create_template is True


class TestSandboxDockerSessionClose:
    """Test SandboxDockerSession close functionality."""

    @patch("llm_sandbox.docker.docker.from_env")
    @patch("llm_sandbox.language_handlers.factory.LanguageHandlerFactory.create_handler")
    def test_close_simple(self, mock_create_handler: MagicMock, mock_docker_from_env: MagicMock) -> None:
        """Test simple close without commit."""
        mock_handler = MagicMock()
        mock_create_handler.return_value = mock_handler
        mock_client = MagicMock()
        mock_docker_from_env.return_value = mock_client

        session = SandboxDockerSession()

        mock_container = MagicMock()
        session.container = mock_container

        session.close()

        mock_container.stop.assert_called_once()
        mock_container.wait.assert_called_once()
        mock_container.remove.assert_called_once_with(force=True)
        assert session.container is None

    @patch("llm_sandbox.docker.docker.from_env")
    @patch("llm_sandbox.language_handlers.factory.LanguageHandlerFactory.create_handler")
    def test_close_with_commit(self, mock_create_handler: MagicMock, mock_docker_from_env: MagicMock) -> None:
        """Test close with container commit."""
        mock_handler = MagicMock()
        mock_create_handler.return_value = mock_handler
        mock_client = MagicMock()
        mock_docker_from_env.return_value = mock_client

        session = SandboxDockerSession(commit_container=True)
        session.keep_template = True  # Need to set this for commit to happen

        mock_container = MagicMock()
        mock_image = MagicMock()
        mock_image.tags = ["test:latest"]
        session.container = mock_container
        session.docker_image = mock_image

        session.close()

        mock_container.commit.assert_called_once_with(repository="test", tag="latest")

    @patch("llm_sandbox.docker.docker.from_env")
    @patch("llm_sandbox.language_handlers.factory.LanguageHandlerFactory.create_handler")
    def test_close_with_image_removal(self, mock_create_handler: MagicMock, mock_docker_from_env: MagicMock) -> None:
        """Test close with image removal."""
        mock_handler = MagicMock()
        mock_create_handler.return_value = mock_handler
        mock_client = MagicMock()
        mock_docker_from_env.return_value = mock_client

        session = SandboxDockerSession(keep_template=False)
        session.is_create_template = True
        session.config.image = "test:latest"

        mock_container = MagicMock()
        mock_image = MagicMock()
        mock_client.containers.list.return_value = []  # No other containers using image
        session.container = mock_container
        session.docker_image = mock_image

        session.close()

        mock_image.remove.assert_called_once_with(force=True)


class TestSandboxDockerSessionRun:
    """Test SandboxDockerSession run functionality."""

    @patch("llm_sandbox.docker.docker.from_env")
    @patch("llm_sandbox.language_handlers.factory.LanguageHandlerFactory.create_handler")
    def test_run_success(self, mock_create_handler: MagicMock, mock_docker_from_env: MagicMock) -> None:
        """Test successful code execution."""
        mock_handler = MagicMock()
        mock_handler.file_extension = "py"
        mock_create_handler.return_value = mock_handler

        session = SandboxDockerSession()

        mock_container = MagicMock()
        session.container = mock_container
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

    @patch("llm_sandbox.docker.docker.from_env")
    @patch("llm_sandbox.language_handlers.factory.LanguageHandlerFactory.create_handler")
    def test_run_without_open_session(self, mock_create_handler: MagicMock, mock_docker_from_env: MagicMock) -> None:
        """Test run fails when session is not open."""
        mock_handler = MagicMock()
        mock_create_handler.return_value = mock_handler

        session = SandboxDockerSession()
        session.container = None
        session.is_open = False

        with pytest.raises(NotOpenSessionError):
            session.run("print('hello')")


class TestSandboxDockerSessionFileOperations:
    """Test SandboxDockerSession file operations."""

    @patch("llm_sandbox.docker.docker.from_env")
    @patch("llm_sandbox.language_handlers.factory.LanguageHandlerFactory.create_handler")
    def test_copy_to_runtime(self, mock_create_handler: MagicMock, mock_docker_from_env: MagicMock) -> None:
        """Test copying file to container."""
        mock_handler = MagicMock()
        mock_create_handler.return_value = mock_handler

        session = SandboxDockerSession()

        mock_container = MagicMock()
        session.container = mock_container

        with (
            patch("tarfile.open") as mock_tar_open,
            patch.object(session, "_ensure_ownership") as mock_ownership,
            patch("llm_sandbox.docker.Path") as mock_path,
            tempfile.NamedTemporaryFile() as temp_file,
        ):
            # Create a real temporary file for the test
            temp_file.write(b"test content")
            temp_file.flush()

            # Mock Path to return our temp file for source validation
            mock_src_path = MagicMock()
            mock_src_path.exists.return_value = True
            mock_src_path.is_file.return_value = True
            mock_src_path.is_dir.return_value = False

            mock_dest_path = MagicMock()
            # Use a string to avoid Mock serialization issues
            mock_dest_path.parent = "/container"
            mock_dest_path.name = "file.txt"

            def path_side_effect(arg: str) -> MagicMock:
                if arg == temp_file.name:
                    return mock_src_path
                if arg == "/container/file.txt":
                    return mock_dest_path
                return MagicMock()

            mock_path.side_effect = path_side_effect

            mock_tar = MagicMock()
            mock_tar_open.return_value.__enter__.return_value = mock_tar

            with patch.object(session, "_ensure_directory_exists") as mock_ensure_dir:
                session.copy_to_runtime(temp_file.name, "/container/file.txt")

            mock_ensure_dir.assert_called_once_with("/container")
            mock_container.put_archive.assert_called_once()
            mock_ownership.assert_called_once_with(["/container/file.txt"])

    @patch("llm_sandbox.docker.docker.from_env")
    @patch("llm_sandbox.language_handlers.factory.LanguageHandlerFactory.create_handler")
    def test_copy_to_runtime_no_container(
        self, mock_create_handler: MagicMock, mock_docker_from_env: MagicMock
    ) -> None:
        """Test copy_to_runtime fails when no container."""
        mock_handler = MagicMock()
        mock_create_handler.return_value = mock_handler

        session = SandboxDockerSession()
        session.container = None

        with pytest.raises(NotOpenSessionError):
            session.copy_to_runtime("/host/file.txt", "/container/file.txt")

    @patch("llm_sandbox.docker.docker.from_env")
    @patch("llm_sandbox.language_handlers.factory.LanguageHandlerFactory.create_handler")
    def test_copy_from_runtime(self, mock_create_handler: MagicMock, mock_docker_from_env: MagicMock) -> None:
        """Test copying file from container."""
        mock_handler = MagicMock()
        mock_create_handler.return_value = mock_handler

        session = SandboxDockerSession()

        mock_container = MagicMock()

        # Create mock tar data with proper member names (no leading slash)
        file_content = b"test content"
        tar_buffer = io.BytesIO()
        with tarfile.open(fileobj=tar_buffer, mode="w") as tar:
            info = tarfile.TarInfo("file.txt")  # Remove leading slash
            info.size = len(file_content)
            tar.addfile(info, io.BytesIO(file_content))
        tar_data = tar_buffer.getvalue()

        mock_container.get_archive.return_value = ([tar_data], {"size": len(tar_data)})
        session.container = mock_container

        with (
            patch("tarfile.open") as mock_tar_open,
            patch("llm_sandbox.docker.Path") as mock_path,
            tempfile.TemporaryDirectory() as temp_dir,
        ):
            mock_tar = MagicMock()
            mock_member = MagicMock()
            mock_member.name = "file.txt"  # No leading slash
            mock_member.isfile.return_value = True
            mock_member.issym.return_value = False
            mock_member.islnk.return_value = False
            mock_tar.getmembers.return_value = [mock_member]
            mock_tar_open.return_value.__enter__.return_value = mock_tar

            # Use temp directory instead of /host
            dest_file = f"{temp_dir}/file.txt"

            # Mock Path for destination directory creation
            mock_dest_path = MagicMock()

            # Create a simple class that won't cause Mock stringification issues
            class MockParent:
                def __init__(self) -> None:
                    self.mkdir = MagicMock()

                def __str__(self) -> str:
                    return "/temp/dest"

                def __fspath__(self) -> str:
                    return "/temp/dest"

            mock_dest_path.parent = MockParent()
            mock_path.return_value = mock_dest_path

            session.copy_from_runtime("/container/file.txt", dest_file)

            mock_container.get_archive.assert_called_once_with("/container/file.txt")
            mock_tar.extract.assert_called_once()

    @patch("llm_sandbox.docker.docker.from_env")
    @patch("llm_sandbox.language_handlers.factory.LanguageHandlerFactory.create_handler")
    def test_copy_from_runtime_file_not_found(
        self, mock_create_handler: MagicMock, mock_docker_from_env: MagicMock
    ) -> None:
        """Test copy_from_runtime when file not found."""
        mock_handler = MagicMock()
        mock_create_handler.return_value = mock_handler

        session = SandboxDockerSession()

        mock_container = MagicMock()
        mock_container.get_archive.return_value = ([], {"size": 0})
        session.container = mock_container

        with pytest.raises(FileNotFoundError):
            session.copy_from_runtime("/container/missing.txt", "/host/file.txt")


class TestSandboxDockerSessionCommands:
    """Test SandboxDockerSession command execution."""

    @patch("llm_sandbox.docker.docker.from_env")
    @patch("llm_sandbox.language_handlers.factory.LanguageHandlerFactory.create_handler")
    def test_execute_command_success_no_stream(
        self, mock_create_handler: MagicMock, mock_docker_from_env: MagicMock
    ) -> None:
        """Test successful command execution without streaming."""
        mock_handler = MagicMock()
        mock_create_handler.return_value = mock_handler

        session = SandboxDockerSession(stream=False)

        mock_container = MagicMock()
        mock_container.exec_run.return_value = Mock(exit_code=0, output=(b"stdout content", b"stderr content"))
        session.container = mock_container

        result = session.execute_command("ls -l", workdir="/tmp")

        assert result.exit_code == 0
        assert result.stdout == "stdout content"
        assert result.stderr == "stderr content"
        mock_container.exec_run.assert_called_once_with(
            cmd="ls -l",
            stream=False,
            tty=False,
            workdir="/tmp",
            stderr=True,
            stdout=True,
            demux=True,
        )

    @patch("llm_sandbox.docker.docker.from_env")
    @patch("llm_sandbox.language_handlers.factory.LanguageHandlerFactory.create_handler")
    def test_execute_command_success_with_stream(
        self, mock_create_handler: MagicMock, mock_docker_from_env: MagicMock
    ) -> None:
        """Test successful command execution with streaming."""
        mock_handler = MagicMock()
        mock_create_handler.return_value = mock_handler

        session = SandboxDockerSession(stream=True)

        mock_container = MagicMock()
        # Mock streaming output
        mock_output = [
            (b"stdout chunk 1", None),
            (None, b"stderr chunk 1"),
            (b"stdout chunk 2", b"stderr chunk 2"),
        ]
        mock_container.exec_run.return_value = Mock(exit_code=0, output=iter(mock_output))
        session.container = mock_container

        result = session.execute_command("ls -l")

        assert result.exit_code == 0
        assert result.stdout == "stdout chunk 1stdout chunk 2"
        assert result.stderr == "stderr chunk 1stderr chunk 2"

    @patch("llm_sandbox.docker.docker.from_env")
    @patch("llm_sandbox.language_handlers.factory.LanguageHandlerFactory.create_handler")
    def test_execute_command_empty_command(
        self, mock_create_handler: MagicMock, mock_docker_from_env: MagicMock
    ) -> None:
        """Test execute_command with empty command."""
        mock_handler = MagicMock()
        mock_create_handler.return_value = mock_handler

        session = SandboxDockerSession()

        with pytest.raises(CommandEmptyError):
            session.execute_command("")

    @patch("llm_sandbox.docker.docker.from_env")
    @patch("llm_sandbox.language_handlers.factory.LanguageHandlerFactory.create_handler")
    def test_execute_command_no_container(
        self, mock_create_handler: MagicMock, mock_docker_from_env: MagicMock
    ) -> None:
        """Test execute_command fails when no container."""
        mock_handler = MagicMock()
        mock_create_handler.return_value = mock_handler

        session = SandboxDockerSession()
        session.container = None

        with pytest.raises(NotOpenSessionError):
            session.execute_command("ls")


class TestSandboxDockerSessionArchive:
    """Test SandboxDockerSession archive operations."""

    @patch("llm_sandbox.docker.docker.from_env")
    @patch("llm_sandbox.language_handlers.factory.LanguageHandlerFactory.create_handler")
    def test_get_archive(self, mock_create_handler: MagicMock, mock_docker_from_env: MagicMock) -> None:
        """Test getting archive from container."""
        mock_handler = MagicMock()
        mock_create_handler.return_value = mock_handler

        session = SandboxDockerSession()

        mock_container = MagicMock()
        mock_data = [b"archive", b"content"]
        mock_stat = {"size": 100, "name": "file.txt"}
        mock_container.get_archive.return_value = (mock_data, mock_stat)
        session.container = mock_container

        data, stat = session.get_archive("/container/path")

        assert data == b"archivecontent"
        assert stat == mock_stat
        mock_container.get_archive.assert_called_once_with("/container/path")

    @patch("llm_sandbox.docker.docker.from_env")
    @patch("llm_sandbox.language_handlers.factory.LanguageHandlerFactory.create_handler")
    def test_get_archive_no_container(self, mock_create_handler: MagicMock, mock_docker_from_env: MagicMock) -> None:
        """Test get_archive fails when no container."""
        mock_handler = MagicMock()
        mock_create_handler.return_value = mock_handler

        session = SandboxDockerSession()
        session.container = None

        with pytest.raises(NotOpenSessionError):
            session.get_archive("/container/path")


class TestSandboxDockerSessionContextManager:
    """Test SandboxDockerSession context manager functionality."""

    @patch("llm_sandbox.docker.docker.from_env")
    @patch("llm_sandbox.language_handlers.factory.LanguageHandlerFactory.create_handler")
    def test_context_manager(self, mock_create_handler: MagicMock, mock_docker_from_env: MagicMock) -> None:
        """Test using session as context manager."""
        mock_handler = MagicMock()
        mock_create_handler.return_value = mock_handler

        session = SandboxDockerSession()

        with (
            patch.object(session, "open") as mock_open,
            patch.object(session, "close") as mock_close,
        ):
            with session as s:
                assert s == session
                mock_open.assert_called_once()

            mock_close.assert_called_once()

    @patch("llm_sandbox.docker.docker.from_env")
    @patch("llm_sandbox.language_handlers.factory.LanguageHandlerFactory.create_handler")
    def test_context_manager_with_exception(
        self, mock_create_handler: MagicMock, mock_docker_from_env: MagicMock
    ) -> None:
        """Test context manager ensures close is called even with exception."""
        mock_handler = MagicMock()
        mock_create_handler.return_value = mock_handler

        session = SandboxDockerSession()

        with (
            patch.object(session, "open") as mock_open,
            patch.object(session, "close") as mock_close,
        ):
            with pytest.raises(ValueError), session:
                raise ValueError

            mock_open.assert_called_once()
            mock_close.assert_called_once()


class TestSandboxDockerSessionOwnership:
    """Test SandboxDockerSession ownership management."""

    @patch("llm_sandbox.docker.docker.from_env")
    @patch("llm_sandbox.language_handlers.factory.LanguageHandlerFactory.create_handler")
    def test_ensure_ownership_with_non_root_user(
        self, mock_create_handler: MagicMock, mock_docker_from_env: MagicMock
    ) -> None:
        """Test _ensure_ownership with non-root user."""
        mock_handler = MagicMock()
        mock_create_handler.return_value = mock_handler

        session = SandboxDockerSession(runtime_configs={"user": "1000:1000"})

        mock_container = MagicMock()
        session.container = mock_container

        session._ensure_ownership(["/tmp/test", "/tmp/test2"])

        mock_container.exec_run.assert_called_once_with("chown -R 1000:1000 /tmp/test /tmp/test2", user="root")

    @patch("llm_sandbox.docker.docker.from_env")
    @patch("llm_sandbox.language_handlers.factory.LanguageHandlerFactory.create_handler")
    def test_ensure_ownership_with_root_user(
        self, mock_create_handler: MagicMock, mock_docker_from_env: MagicMock
    ) -> None:
        """Test _ensure_ownership with root user (no-op)."""
        mock_handler = MagicMock()
        mock_create_handler.return_value = mock_handler

        session = SandboxDockerSession(runtime_configs={"user": "root"})

        mock_container = MagicMock()
        session.container = mock_container

        session._ensure_ownership(["/tmp/test"])

        mock_container.exec_run.assert_not_called()

    @patch("llm_sandbox.docker.docker.from_env")
    @patch("llm_sandbox.language_handlers.factory.LanguageHandlerFactory.create_handler")
    def test_ensure_ownership_no_runtime_configs(
        self, mock_create_handler: MagicMock, mock_docker_from_env: MagicMock
    ) -> None:
        """Test _ensure_ownership with no runtime configs (default root)."""
        mock_handler = MagicMock()
        mock_create_handler.return_value = mock_handler

        session = SandboxDockerSession()

        mock_container = MagicMock()
        session.container = mock_container

        session._ensure_ownership(["/tmp/test"])

        mock_container.exec_run.assert_not_called()


class TestDockerContainerAPI:
    """Test DockerContainerAPI class methods."""

    def test_init(self) -> None:
        """Test DockerContainerAPI initialization."""
        mock_client = MagicMock()
        api = DockerContainerAPI(mock_client, stream=True)

        assert api.client == mock_client
        assert api.stream is True

    def test_init_default_stream(self) -> None:
        """Test DockerContainerAPI initialization with default stream."""
        mock_client = MagicMock()
        api = DockerContainerAPI(mock_client)

        assert api.client == mock_client
        assert api.stream is False

    def test_create_container(self) -> None:
        """Test container creation."""
        mock_client = MagicMock()
        mock_container = MagicMock()
        mock_client.containers.create.return_value = mock_container

        api = DockerContainerAPI(mock_client)
        config = {"image": "test:latest", "detach": True}

        result = api.create_container(config)

        assert result == mock_container
        mock_client.containers.create.assert_called_once_with(**config)

    def test_start_container(self) -> None:
        """Test starting container."""
        mock_client = MagicMock()
        mock_container = MagicMock()

        api = DockerContainerAPI(mock_client)
        api.start_container(mock_container)

        mock_container.start.assert_called_once()

    def test_stop_container(self) -> None:
        """Test stopping container."""
        mock_client = MagicMock()
        mock_container = MagicMock()

        api = DockerContainerAPI(mock_client)
        api.stop_container(mock_container)

        mock_container.stop.assert_called_once()

    def test_execute_command_with_workdir(self) -> None:
        """Test command execution with custom workdir."""
        mock_client = MagicMock()
        mock_container = MagicMock()
        mock_container.exec_run.return_value = Mock(exit_code=0, output=(b"output", None))

        api = DockerContainerAPI(mock_client, stream=False)

        exit_code, _ = api.execute_command(mock_container, "ls -la", workdir="/tmp")

        assert exit_code == 0
        mock_container.exec_run.assert_called_once_with(
            cmd="ls -la",
            stream=False,
            tty=False,
            workdir="/tmp",
            stderr=True,
            stdout=True,
            demux=True,
        )

    def test_execute_command_no_workdir(self) -> None:
        """Test command execution without workdir."""
        mock_client = MagicMock()
        mock_container = MagicMock()
        mock_container.exec_run.return_value = Mock(exit_code=1, output=(None, b"error"))

        api = DockerContainerAPI(mock_client, stream=True)

        exit_code, _ = api.execute_command(mock_container, "invalid-command")

        assert exit_code == 1
        mock_container.exec_run.assert_called_once_with(
            cmd="invalid-command",
            stream=True,
            tty=False,
            stderr=True,
            stdout=True,
            demux=True,
        )

    @patch("tarfile.open")
    @patch("llm_sandbox.docker.Path")
    def test_copy_to_container(self, mock_path: MagicMock, mock_tarfile: MagicMock) -> None:
        """Test copying file to container."""
        mock_client = MagicMock()
        mock_container = MagicMock()

        # Mock Path behavior
        mock_dest_path = MagicMock()
        mock_dest_path.name = "file.txt"
        mock_dest_path.parent = "/container/path"
        mock_path.return_value = mock_dest_path

        # Mock tarfile
        mock_tar = MagicMock()
        mock_tarfile.return_value.__enter__.return_value = mock_tar

        api = DockerContainerAPI(mock_client)
        api.copy_to_container(mock_container, "/host/file.txt", "/container/path/file.txt")

        mock_tar.add.assert_called_once_with("/host/file.txt", arcname="file.txt")
        mock_container.put_archive.assert_called_once()

    def test_copy_from_container(self) -> None:
        """Test copying file from container."""
        mock_client = MagicMock()
        mock_container = MagicMock()
        mock_data = [b"chunk1", b"chunk2"]
        mock_stat = {"size": 12}
        mock_container.get_archive.return_value = (mock_data, mock_stat)

        api = DockerContainerAPI(mock_client)
        data, stat = api.copy_from_container(mock_container, "/container/file.txt")

        assert data == b"chunk1chunk2"
        assert stat == mock_stat
        mock_container.get_archive.assert_called_once_with("/container/file.txt")


class TestSandboxDockerSessionEdgeCases:
    """Test additional edge cases and error conditions."""

    @patch("llm_sandbox.docker.docker.from_env")
    @patch("llm_sandbox.language_handlers.factory.LanguageHandlerFactory.create_handler")
    def test_init_with_deprecated_mounts_parameter(
        self, mock_create_handler: MagicMock, mock_docker_from_env: MagicMock
    ) -> None:
        """Test initialization with deprecated mounts parameter."""
        mock_handler = MagicMock()
        mock_create_handler.return_value = mock_handler

        with pytest.warns(DeprecationWarning, match="The 'mounts' parameter is deprecated"):
            session = SandboxDockerSession(mounts=[{"type": "bind", "source": "/host", "target": "/container"}])

        assert "mounts" in session.config.runtime_configs
        assert len(session.config.runtime_configs["mounts"]) == 1

    @patch("llm_sandbox.docker.docker.from_env")
    @patch("llm_sandbox.language_handlers.factory.LanguageHandlerFactory.create_handler")
    def test_init_with_deprecated_mounts_parameter_single_mount(
        self, mock_create_handler: MagicMock, mock_docker_from_env: MagicMock
    ) -> None:
        """Test initialization with deprecated mounts parameter as single mount."""
        mock_handler = MagicMock()
        mock_create_handler.return_value = mock_handler

        with pytest.warns(DeprecationWarning, match="The 'mounts' parameter is deprecated"):
            session = SandboxDockerSession(mounts={"type": "bind", "source": "/host", "target": "/container"})

        assert "mounts" in session.config.runtime_configs
        assert len(session.config.runtime_configs["mounts"]) == 1

    @patch("llm_sandbox.docker.docker.from_env")
    @patch("llm_sandbox.language_handlers.factory.LanguageHandlerFactory.create_handler")
    def test_init_with_deprecated_mounts_and_existing_runtime_mounts(
        self, mock_create_handler: MagicMock, mock_docker_from_env: MagicMock
    ) -> None:
        """Test initialization with deprecated mounts and existing runtime config mounts."""
        mock_handler = MagicMock()
        mock_create_handler.return_value = mock_handler

        runtime_configs = {"mounts": [{"existing": "mount"}]}

        with pytest.warns(DeprecationWarning, match="The 'mounts' parameter is deprecated"):
            session = SandboxDockerSession(
                runtime_configs=runtime_configs, mounts=[{"type": "bind", "source": "/host", "target": "/container"}]
            )

        assert len(session.config.runtime_configs["mounts"]) == 2

    @patch("llm_sandbox.docker.docker.from_env")
    @patch("llm_sandbox.language_handlers.factory.LanguageHandlerFactory.create_handler")
    def test_ensure_directory_exists_failure(
        self, mock_create_handler: MagicMock, mock_docker_from_env: MagicMock
    ) -> None:
        """Test _ensure_directory_exists with command failure."""
        mock_handler = MagicMock()
        mock_create_handler.return_value = mock_handler

        session = SandboxDockerSession()
        mock_container = MagicMock()
        session.container = mock_container

        # Mock container API to return error
        session.container_api.execute_command = Mock(return_value=(1, (b"", b"Permission denied")))  # type: ignore[method-assign]

        with patch.object(session, "_log") as mock_log:
            session._ensure_directory_exists("/test/path")

        mock_log.assert_called_with("Failed to create directory /test/path: Permission denied", "error")

    @patch("llm_sandbox.docker.docker.from_env")
    @patch("llm_sandbox.language_handlers.factory.LanguageHandlerFactory.create_handler")
    def test_ensure_directory_exists_failure_with_stdout(
        self, mock_create_handler: MagicMock, mock_docker_from_env: MagicMock
    ) -> None:
        """Test _ensure_directory_exists with command failure and stdout error."""
        mock_handler = MagicMock()
        mock_create_handler.return_value = mock_handler

        session = SandboxDockerSession()
        mock_container = MagicMock()
        session.container = mock_container

        # Mock container API to return error with stdout only
        session.container_api.execute_command = Mock(return_value=(1, (b"Directory creation failed", b"")))  # type: ignore[method-assign]

        with patch.object(session, "_log") as mock_log:
            session._ensure_directory_exists("/test/path")

        mock_log.assert_called_with("Failed to create directory /test/path: Directory creation failed", "error")

    @patch("llm_sandbox.docker.docker.from_env")
    @patch("llm_sandbox.language_handlers.factory.LanguageHandlerFactory.create_handler")
    def test_process_stream_output_with_timeout_exception(
        self, mock_create_handler: MagicMock, mock_docker_from_env: MagicMock
    ) -> None:
        """Test _process_stream_output with SandboxTimeoutError exception."""
        from llm_sandbox.exceptions import SandboxTimeoutError

        mock_handler = MagicMock()
        mock_create_handler.return_value = mock_handler

        session = SandboxDockerSession()

        def mock_output_generator() -> Generator[tuple[bytes, bytes | None], None, None]:
            """Mock output generator."""
            yield (b"stdout data", None)
            msg = "Timeout occurred while processing output"
            raise SandboxTimeoutError(msg, 10)

        with pytest.raises(SandboxTimeoutError):
            session._process_stream_output(mock_output_generator())

    @patch("llm_sandbox.docker.docker.from_env")
    @patch("llm_sandbox.language_handlers.factory.LanguageHandlerFactory.create_handler")
    def test_process_stream_output_with_other_exception(
        self, mock_create_handler: MagicMock, mock_docker_from_env: MagicMock
    ) -> None:
        """Test _process_stream_output with other exceptions."""
        mock_handler = MagicMock()
        mock_create_handler.return_value = mock_handler

        session = SandboxDockerSession()

        def mock_output_generator() -> Generator[tuple[bytes, bytes | None], None, None]:
            """Mock output generator."""
            yield (b"stdout data", None)
            raise ValueError

        with patch.object(session, "_log") as mock_log:
            stdout, stderr = session._process_stream_output(mock_output_generator())

        assert stdout == "stdout data"
        assert stderr == ""
        mock_log.assert_called_with("Error processing stream output: ", "warning")

    @patch("llm_sandbox.docker.docker.from_env")
    @patch("llm_sandbox.language_handlers.factory.LanguageHandlerFactory.create_handler")
    def test_handle_timeout(self, mock_create_handler: MagicMock, mock_docker_from_env: MagicMock) -> None:
        """Test _handle_timeout method."""
        mock_handler = MagicMock()
        mock_create_handler.return_value = mock_handler

        session = SandboxDockerSession()
        mock_container = MagicMock()
        mock_container.short_id = "abc123"
        session.container = mock_container
        session.using_existing_container = True  # Set to use existing container

        with patch.object(session, "close") as mock_close:
            session._handle_timeout()

        mock_close.assert_called_once()

    @patch("llm_sandbox.docker.docker.from_env")
    @patch("llm_sandbox.language_handlers.factory.LanguageHandlerFactory.create_handler")
    def test_handle_timeout_kill_failure(self, mock_create_handler: MagicMock, mock_docker_from_env: MagicMock) -> None:
        """Test _handle_timeout with using_existing_container=False (should do nothing)."""
        mock_handler = MagicMock()
        mock_create_handler.return_value = mock_handler

        session = SandboxDockerSession()
        mock_container = MagicMock()
        mock_container.short_id = "abc123"
        session.container = mock_container
        session.using_existing_container = False  # Not using existing container

        with patch.object(session, "close") as mock_close:
            session._handle_timeout()

        # Should not call close() when not using existing container
        mock_close.assert_not_called()

    @patch("llm_sandbox.docker.docker.from_env")
    @patch("llm_sandbox.language_handlers.factory.LanguageHandlerFactory.create_handler")
    def test_handle_timeout_remove_failure(
        self, mock_create_handler: MagicMock, mock_docker_from_env: MagicMock
    ) -> None:
        """Test _handle_timeout with close() raising exception."""
        mock_handler = MagicMock()
        mock_create_handler.return_value = mock_handler

        session = SandboxDockerSession()
        mock_container = MagicMock()
        mock_container.short_id = "abc123"
        session.container = mock_container
        session.using_existing_container = True  # Set to use existing container

        with patch.object(session, "close", side_effect=Exception("Close failed")) as mock_close:
            # Should not raise exception, just call close
            session._handle_timeout()

        mock_close.assert_called_once()

    @patch("llm_sandbox.docker.docker.from_env")
    @patch("llm_sandbox.language_handlers.factory.LanguageHandlerFactory.create_handler")
    def test_prepare_image_with_default_image_selection(
        self, mock_create_handler: MagicMock, mock_docker_from_env: MagicMock
    ) -> None:
        """Test _prepare_image with default image selection."""
        mock_handler = MagicMock()
        mock_create_handler.return_value = mock_handler

        session = SandboxDockerSession(lang="python")  # No explicit image provided

        with patch.object(session, "_get_or_pull_image") as mock_get_or_pull:
            session._prepare_image()

        assert session.config.image == DefaultImage.PYTHON
        mock_get_or_pull.assert_called_once()

    @patch("llm_sandbox.docker.docker.from_env")
    @patch("llm_sandbox.language_handlers.factory.LanguageHandlerFactory.create_handler")
    def test_commit_container_error_handling(
        self, mock_create_handler: MagicMock, mock_docker_from_env: MagicMock
    ) -> None:
        """Test _commit_container with error handling."""
        mock_handler = MagicMock()
        mock_create_handler.return_value = mock_handler

        session = SandboxDockerSession()
        mock_container = MagicMock()
        mock_container.commit.side_effect = Exception("Commit failed")
        mock_image = MagicMock()
        mock_image.tags = ["test:latest"]

        session.container = mock_container
        session.docker_image = mock_image

        with (
            patch.object(session, "_log") as mock_log,
            pytest.raises(Exception, match="Commit failed"),
        ):
            session._commit_container()

        mock_log.assert_called_with("Failed to commit container", "error")

    @patch("llm_sandbox.docker.docker.from_env")
    @patch("llm_sandbox.language_handlers.factory.LanguageHandlerFactory.create_handler")
    def test_commit_container_no_tags(self, mock_create_handler: MagicMock, mock_docker_from_env: MagicMock) -> None:
        """Test _commit_container with image without tags."""
        mock_handler = MagicMock()
        mock_create_handler.return_value = mock_handler

        session = SandboxDockerSession()
        mock_container = MagicMock()
        mock_image = MagicMock()
        mock_image.tags = []  # No tags

        session.container = mock_container
        session.docker_image = mock_image

        # Should not call commit since there are no tags
        session._commit_container()

        mock_container.commit.assert_not_called()

    @patch("llm_sandbox.docker.docker.from_env")
    @patch("llm_sandbox.language_handlers.factory.LanguageHandlerFactory.create_handler")
    def test_cleanup_image_with_containers_using_it(
        self, mock_create_handler: MagicMock, mock_docker_from_env: MagicMock
    ) -> None:
        """Test _cleanup_image when other containers are using the image."""
        mock_handler = MagicMock()
        mock_create_handler.return_value = mock_handler
        mock_client = MagicMock()
        mock_docker_from_env.return_value = mock_client

        session = SandboxDockerSession()
        mock_image = MagicMock()
        mock_image.id = "image123"
        session.docker_image = mock_image

        # Mock containers list to return some containers using the image
        mock_client.containers.list.return_value = [MagicMock()]

        with patch.object(session, "_log") as mock_log:
            session._cleanup_image()

        mock_image.remove.assert_not_called()
        mock_log.assert_called_with("Image in use by other containers, skipping removal")

    @patch("llm_sandbox.docker.docker.from_env")
    @patch("llm_sandbox.language_handlers.factory.LanguageHandlerFactory.create_handler")
    def test_cleanup_image_removal_failure(
        self, mock_create_handler: MagicMock, mock_docker_from_env: MagicMock
    ) -> None:
        """Test _cleanup_image with image removal failure."""
        mock_handler = MagicMock()
        mock_create_handler.return_value = mock_handler
        mock_client = MagicMock()
        mock_docker_from_env.return_value = mock_client

        session = SandboxDockerSession()
        mock_image = MagicMock()
        mock_image.id = "image123"
        mock_image.remove.side_effect = Exception("Remove failed")
        session.docker_image = mock_image

        # Mock containers list to return no containers using the image
        mock_client.containers.list.return_value = []

        with patch.object(session, "_log") as mock_log:
            session._cleanup_image()

        mock_image.remove.assert_called_once_with(force=True)
        mock_log.assert_called_with("Failed to remove image: Remove failed", "error")

    @patch("llm_sandbox.docker.docker.from_env")
    @patch("llm_sandbox.language_handlers.factory.LanguageHandlerFactory.create_handler")
    def test_close_container_cleanup_exception(
        self, mock_create_handler: MagicMock, mock_docker_from_env: MagicMock
    ) -> None:
        """Test close method with container cleanup exception."""
        mock_handler = MagicMock()
        mock_create_handler.return_value = mock_handler

        session = SandboxDockerSession()
        mock_container = MagicMock()
        mock_container.stop.side_effect = Exception("Stop failed")
        session.container = mock_container

        with patch.object(session, "_log") as mock_log:
            session.close()

        assert session.container is None
        mock_log.assert_called_with("Error cleaning up container")

    @patch("llm_sandbox.docker.docker.from_env")
    @patch("llm_sandbox.language_handlers.factory.LanguageHandlerFactory.create_handler")
    def test_close_with_no_commit_but_keep_template_false(
        self, mock_create_handler: MagicMock, mock_docker_from_env: MagicMock
    ) -> None:
        """Test close method without commit but with keep_template=False."""
        mock_handler = MagicMock()
        mock_create_handler.return_value = mock_handler

        session = SandboxDockerSession(commit_container=False, keep_template=False)
        session.is_create_template = True

        mock_container = MagicMock()
        mock_image = MagicMock()
        session.container = mock_container
        session.docker_image = mock_image

        with patch.object(session, "_cleanup_image") as mock_cleanup:
            session.close()

        mock_cleanup.assert_called_once()

    @patch("llm_sandbox.docker.docker.from_env")
    @patch("llm_sandbox.language_handlers.factory.LanguageHandlerFactory.create_handler")
    def test_close_with_commit_and_keep_template_true(
        self, mock_create_handler: MagicMock, mock_docker_from_env: MagicMock
    ) -> None:
        """Test close method with commit and keep_template=True."""
        mock_handler = MagicMock()
        mock_create_handler.return_value = mock_handler

        session = SandboxDockerSession(commit_container=True, keep_template=True)
        session.is_create_template = True

        mock_container = MagicMock()
        mock_image = MagicMock()
        session.container = mock_container
        session.docker_image = mock_image

        with (
            patch.object(session, "_commit_container") as mock_commit,
            patch.object(session, "_cleanup_image") as mock_cleanup,
        ):
            session.close()

        mock_commit.assert_called_once()
        mock_cleanup.assert_not_called()  # Should not cleanup when keep_template=True


class TestSandboxDockerSessionTimeoutEdgeCases:
    """Test timeout-related edge cases."""

    @patch("llm_sandbox.docker.docker.from_env")
    @patch("llm_sandbox.language_handlers.factory.LanguageHandlerFactory.create_handler")
    def test_process_non_stream_output_with_none_output(
        self, mock_create_handler: MagicMock, mock_docker_from_env: MagicMock
    ) -> None:
        """Test _process_non_stream_output with None output."""
        mock_handler = MagicMock()
        mock_create_handler.return_value = mock_handler

        session = SandboxDockerSession()

        stdout, stderr = session._process_non_stream_output(None)

        assert stdout == ""
        assert stderr == ""

    @patch("llm_sandbox.docker.docker.from_env")
    @patch("llm_sandbox.language_handlers.factory.LanguageHandlerFactory.create_handler")
    def test_process_non_stream_output_with_none_components(
        self, mock_create_handler: MagicMock, mock_docker_from_env: MagicMock
    ) -> None:
        """Test _process_non_stream_output with None stdout/stderr."""
        mock_handler = MagicMock()
        mock_create_handler.return_value = mock_handler

        session = SandboxDockerSession()

        stdout, stderr = session._process_non_stream_output((None, None))

        assert stdout == ""
        assert stderr == ""

    @patch("llm_sandbox.docker.docker.from_env")
    @patch("llm_sandbox.language_handlers.factory.LanguageHandlerFactory.create_handler")
    def test_process_stream_output_with_string_chunks(
        self, mock_create_handler: MagicMock, mock_docker_from_env: MagicMock
    ) -> None:
        """Test _process_stream_output with string chunks instead of bytes."""
        mock_handler = MagicMock()
        mock_create_handler.return_value = mock_handler

        session = SandboxDockerSession()

        def mock_output_generator() -> Generator[tuple[str | None, str | None], None, None]:
            """Mock output generator."""
            yield ("stdout string", "stderr string")
            yield (None, None)

        stdout, stderr = session._process_stream_output(mock_output_generator())

        assert stdout == "stdout string"
        assert stderr == "stderr string"

    @patch("llm_sandbox.docker.docker.from_env")
    @patch("llm_sandbox.language_handlers.factory.LanguageHandlerFactory.create_handler")
    def test_handle_timeout_no_container(self, mock_create_handler: MagicMock, mock_docker_from_env: MagicMock) -> None:
        """Test _handle_timeout with no existing container usage."""
        mock_handler = MagicMock()
        mock_create_handler.return_value = mock_handler

        session = SandboxDockerSession()
        session.container = None
        session.using_existing_container = False  # Not using existing container

        with patch.object(session, "close") as mock_close:
            session._handle_timeout()

        # Should not call close() when not using existing container
        mock_close.assert_not_called()


class TestSandboxDockerSessionExistingContainer:
    """Test cases for existing container functionality."""

    @patch("llm_sandbox.docker.docker.from_env")
    @patch("llm_sandbox.language_handlers.factory.LanguageHandlerFactory.create_handler")
    def test_connect_to_existing_container_not_found(
        self, mock_create_handler: MagicMock, mock_docker_from_env: MagicMock
    ) -> None:
        """Test connecting to non-existent container."""
        mock_handler = MagicMock()
        mock_create_handler.return_value = mock_handler

        mock_client = MagicMock()
        mock_docker_from_env.return_value = mock_client
        mock_client.containers.get.side_effect = NotFound("Container not found")

        session = SandboxDockerSession(container_id="non-existent")

        with pytest.raises(ContainerError, match="Container non-existent not found"):
            session.open()

    @patch("llm_sandbox.docker.docker.from_env")
    @patch("llm_sandbox.language_handlers.factory.LanguageHandlerFactory.create_handler")
    def test_connect_to_existing_container_other_error(
        self, mock_create_handler: MagicMock, mock_docker_from_env: MagicMock
    ) -> None:
        """Test connecting to existing container with other errors."""
        mock_handler = MagicMock()
        mock_create_handler.return_value = mock_handler

        mock_client = MagicMock()
        mock_docker_from_env.return_value = mock_client
        mock_client.containers.get.side_effect = Exception("Connection failed")

        session = SandboxDockerSession(container_id="test-container")

        with pytest.raises(ContainerError, match="Failed to connect to container test-container"):
            session.open()

    @patch("llm_sandbox.docker.docker.from_env")
    @patch("llm_sandbox.language_handlers.factory.LanguageHandlerFactory.create_handler")
    def test_connect_to_stopped_container(
        self, mock_create_handler: MagicMock, mock_docker_from_env: MagicMock
    ) -> None:
        """Test connecting to stopped container and starting it."""
        mock_handler = MagicMock()
        mock_create_handler.return_value = mock_handler

        mock_client = MagicMock()
        mock_docker_from_env.return_value = mock_client

        mock_container = MagicMock()
        mock_container.status = "stopped"
        mock_container.short_id = "abc123"
        mock_client.containers.get.return_value = mock_container

        session = SandboxDockerSession(container_id="test-container")

        with patch.object(session, "environment_setup"):
            session.open()

        mock_container.start.assert_called_once()
        assert session.container == mock_container

    @patch("llm_sandbox.docker.docker.from_env")
    @patch("llm_sandbox.language_handlers.factory.LanguageHandlerFactory.create_handler")
    def test_image_pull_error(self, mock_create_handler: MagicMock, mock_docker_from_env: MagicMock) -> None:
        """Test image pull failure."""
        mock_handler = MagicMock()
        mock_create_handler.return_value = mock_handler

        mock_client = MagicMock()
        mock_docker_from_env.return_value = mock_client
        mock_client.images.get.side_effect = ImageNotFound("Image not found")
        mock_client.images.pull.side_effect = Exception("Pull failed")

        session = SandboxDockerSession(image="non-existent-image")

        with pytest.raises(ImagePullError):
            session.open()

    @patch("llm_sandbox.docker.docker.from_env")
    @patch("llm_sandbox.language_handlers.factory.LanguageHandlerFactory.create_handler")
    def test_commit_container_failure(self, mock_create_handler: MagicMock, mock_docker_from_env: MagicMock) -> None:
        """Test container commit failure."""
        mock_handler = MagicMock()
        mock_create_handler.return_value = mock_handler

        mock_client = MagicMock()
        mock_docker_from_env.return_value = mock_client

        mock_image = MagicMock()
        mock_image.tags = ["test-image:latest"]
        mock_client.images.get.return_value = mock_image

        mock_container = MagicMock()
        mock_container.commit.side_effect = Exception("Commit failed")

        session = SandboxDockerSession(keep_template=True)
        session.docker_image = mock_image
        session.container = mock_container

        with pytest.raises(Exception, match="Commit failed"):
            session._commit_container()

    @patch("llm_sandbox.docker.docker.from_env")
    @patch("llm_sandbox.language_handlers.factory.LanguageHandlerFactory.create_handler")
    def test_cleanup_image_with_containers(
        self, mock_create_handler: MagicMock, mock_docker_from_env: MagicMock
    ) -> None:
        """Test image cleanup when other containers are using it."""
        mock_handler = MagicMock()
        mock_create_handler.return_value = mock_handler

        mock_client = MagicMock()
        mock_docker_from_env.return_value = mock_client

        mock_image = MagicMock()
        mock_image.id = "image-id"

        # Mock containers using the image
        mock_client.containers.list.return_value = [MagicMock()]

        session = SandboxDockerSession()
        session.client = mock_client
        session.docker_image = mock_image

        session._cleanup_image()

        # Should not remove image when containers are using it
        mock_image.remove.assert_not_called()

    @patch("llm_sandbox.docker.docker.from_env")
    @patch("llm_sandbox.language_handlers.factory.LanguageHandlerFactory.create_handler")
    def test_cleanup_image_removal_failure(
        self, mock_create_handler: MagicMock, mock_docker_from_env: MagicMock
    ) -> None:
        """Test image cleanup when removal fails."""
        mock_handler = MagicMock()
        mock_create_handler.return_value = mock_handler

        mock_client = MagicMock()
        mock_docker_from_env.return_value = mock_client

        mock_image = MagicMock()
        mock_image.id = "image-id"
        mock_image.remove.side_effect = Exception("Remove failed")

        # No containers using the image
        mock_client.containers.list.return_value = []

        session = SandboxDockerSession()
        session.client = mock_client
        session.docker_image = mock_image

        # Should not raise exception, just log error
        session._cleanup_image()

        mock_image.remove.assert_called_once()
