# ruff: noqa: SLF001, PLR2004, ARG002, PT011

"""Tests for Docker backend implementation."""

import io
import tarfile
import tempfile
from unittest.mock import MagicMock, Mock, patch

import pytest
from pydantic_core import ValidationError

from llm_sandbox.const import DefaultImage, SupportedLanguage
from llm_sandbox.data import ConsoleOutput
from llm_sandbox.docker import SandboxDockerSession
from llm_sandbox.exceptions import CommandEmptyError, ImagePullError, NotOpenSessionError
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
        assert session.stream is True
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
        from docker.errors import ImageNotFound

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

        from docker.errors import ImageNotFound

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
        mock_handler.get_execution_commands.return_value = ["python /sandbox/code.py"]
        mock_create_handler.return_value = mock_handler

        session = SandboxDockerSession()

        mock_container = MagicMock()
        session.container = mock_container
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
