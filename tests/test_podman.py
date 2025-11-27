# ruff: noqa: SLF001, PLR2004, ARG002, PT011

"""Tests for Podman backend implementation."""

from collections.abc import Callable
from typing import TYPE_CHECKING, Any, cast
from unittest.mock import MagicMock, Mock, patch

import pytest
from pydantic_core import ValidationError

from llm_sandbox.const import SupportedLanguage
from llm_sandbox.data import ConsoleOutput
from llm_sandbox.exceptions import CommandEmptyError, ContainerError, ImagePullError, NotOpenSessionError
from llm_sandbox.podman import PodmanImageNotFound, PodmanNotFound, SandboxPodmanSession
from llm_sandbox.security import SecurityPolicy


@pytest.fixture
def podman_session_factory() -> Callable[..., SandboxPodmanSession]:
    """Create SandboxPodmanSession instances with mocked dependencies."""

    def _factory(**kwargs: Any) -> SandboxPodmanSession:
        client = cast("PodmanClient", kwargs.pop("client", MagicMock()))
        with patch("llm_sandbox.language_handlers.factory.LanguageHandlerFactory.create_handler") as mock_handler:
            mock_handler.return_value = MagicMock()
            return SandboxPodmanSession(client=client, **kwargs)

    return _factory


if TYPE_CHECKING:
    from podman import PodmanClient


class TestSandboxPodmanSessionInit:
    """Test SandboxPodmanSession initialization."""

    @patch("llm_sandbox.podman.PodmanClient.from_env")
    @patch("llm_sandbox.language_handlers.factory.LanguageHandlerFactory.create_handler")
    def test_init_with_defaults(self, mock_create_handler: MagicMock, mock_podman_from_env: MagicMock) -> None:
        """Test initialization with default parameters."""
        mock_handler = MagicMock()
        mock_create_handler.return_value = mock_handler
        mock_client = MagicMock()
        mock_podman_from_env.return_value = mock_client

        session = SandboxPodmanSession()

        assert session.config.lang == SupportedLanguage.PYTHON
        assert session.config.verbose is False
        assert session.config.image is None  # Image is set during _prepare_image()
        assert session.keep_template is False
        assert session.commit_container is False
        assert session.stream is False
        assert session.config.workdir == "/sandbox"
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

        assert session.config.image == "custom:latest"
        assert session.config.lang == SupportedLanguage.JAVA
        assert session.keep_template is True
        assert session.commit_container is True
        assert session.config.verbose is False
        assert session.stream is False
        assert session.config.workdir == "/custom"
        assert session.config.security_policy == security_policy

    @patch("llm_sandbox.podman.PodmanClient.from_env")
    @patch("llm_sandbox.language_handlers.factory.LanguageHandlerFactory.create_handler")
    def test_init_with_dockerfile_and_image_raises_error(
        self, mock_create_handler: MagicMock, mock_podman_from_env: MagicMock
    ) -> None:
        """Test initialization fails when both dockerfile and image are provided."""
        mock_handler = MagicMock()
        mock_create_handler.return_value = mock_handler

        with pytest.raises(ValidationError, match="Only one of"):
            SandboxPodmanSession(image="test:latest", dockerfile="/path/to/Containerfile")

    @patch("llm_sandbox.podman.PodmanClient.from_env")
    @patch("llm_sandbox.language_handlers.factory.LanguageHandlerFactory.create_handler")
    def test_init_with_dockerfile_only(self, mock_create_handler: MagicMock, mock_podman_from_env: MagicMock) -> None:
        """Test initialization with dockerfile only."""
        mock_handler = MagicMock()
        mock_create_handler.return_value = mock_handler

        session = SandboxPodmanSession(dockerfile="/path/to/Containerfile")

        assert session.config.dockerfile == "/path/to/Containerfile"
        assert session.config.image is None


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
        assert session.container is None

    @patch("llm_sandbox.podman.PodmanClient.from_env")
    @patch("llm_sandbox.language_handlers.factory.LanguageHandlerFactory.create_handler")
    def test_close_with_commit(self, mock_create_handler: MagicMock, mock_podman_from_env: MagicMock) -> None:
        """Test close with container commit."""
        mock_handler = MagicMock()
        mock_create_handler.return_value = mock_handler
        mock_client = MagicMock()
        mock_podman_from_env.return_value = mock_client

        session = SandboxPodmanSession(commit_container=True)
        session.keep_template = True  # Need to set this for commit to happen

        mock_container = MagicMock()
        mock_image = MagicMock()
        mock_image.tags = ["test:latest"]
        session.container = mock_container
        session.docker_image = mock_image

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
        session.config.image = "test:latest"

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
        mock_create_handler.return_value = mock_handler

        session = SandboxPodmanSession()

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

    @patch("llm_sandbox.podman.PodmanClient.from_env")
    @patch("llm_sandbox.language_handlers.factory.LanguageHandlerFactory.create_handler")
    def test_run_without_open_session(self, mock_create_handler: MagicMock, mock_podman_from_env: MagicMock) -> None:
        """Test run fails when session is not open."""
        mock_handler = MagicMock()
        mock_create_handler.return_value = mock_handler

        session = SandboxPodmanSession()
        session.container = None
        session.is_open = False

        with pytest.raises(NotOpenSessionError):
            session.run("print('hello')")


class TestSandboxPodmanSessionFileOperations:
    """Test SandboxPodmanSession file operations."""

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
    def test_copy_from_runtime_file_not_found(
        self, mock_create_handler: MagicMock, mock_podman_from_env: MagicMock
    ) -> None:
        """Test copy_from_runtime when file not found."""
        mock_handler = MagicMock()
        mock_create_handler.return_value = mock_handler

        session = SandboxPodmanSession()

        mock_container = MagicMock()
        session.container = mock_container

        with (
            patch.object(session.container_api, "copy_from_container", side_effect=FileNotFoundError),
            pytest.raises(FileNotFoundError),
        ):
            session.copy_from_runtime("/container/missing.txt", "/host/file.txt")


class TestPodmanRuntimeNormalization:
    """Tests for Podman-specific runtime normalization helpers."""

    def test_normalize_memory_limit_preserves_podman_format(
        self,
        podman_session_factory: Callable[..., SandboxPodmanSession],
    ) -> None:
        """Existing Podman memory format should be returned unchanged."""
        session = podman_session_factory()
        assert session._normalize_memory_limit("256m") == "256m"

    @pytest.mark.parametrize(
        ("limit", "expected"),
        [
            ("1GB", "1024m"),
            ("2g", "2048m"),
            ("1024k", "1m"),
            ("2048kb", "2m"),
        ],
    )
    def test_normalize_memory_limit_converts_units(
        self,
        podman_session_factory: Callable[..., SandboxPodmanSession],
        limit: str,
        expected: str,
    ) -> None:
        """Memory limits using Docker-style units should be converted to Podman format."""
        session = podman_session_factory()
        assert session._normalize_memory_limit(limit) == expected

    def test_normalize_memory_limit_returns_original_for_unknown_unit(
        self,
        podman_session_factory: Callable[..., SandboxPodmanSession],
    ) -> None:
        """Unknown units should be returned verbatim for Podman to handle."""
        session = podman_session_factory()
        assert session._normalize_memory_limit("5XB") == "5XB"

    def test_normalize_runtime_configs_for_podman_returns_copy(
        self,
        podman_session_factory: Callable[..., SandboxPodmanSession],
    ) -> None:
        """Runtime config normalization should return a new dictionary with converted memory limits."""
        session = podman_session_factory()
        runtime_configs = {"mem_limit": "3GB", "user": "1000:1000"}
        normalized = session._normalize_runtime_configs_for_podman(runtime_configs)

        assert normalized is not runtime_configs
        assert normalized["mem_limit"] == "3072m"
        assert runtime_configs["mem_limit"] == "3GB"

    def test_open_normalizes_runtime_configs_before_super_open(
        self,
        podman_session_factory: Callable[..., SandboxPodmanSession],
    ) -> None:
        """open() should normalize runtime configs before delegating to Docker implementation."""
        session = podman_session_factory(runtime_configs={"mem_limit": "4GB"}, image="my-image:latest")

        with (
            patch.object(
                SandboxPodmanSession,
                "_normalize_runtime_configs_for_podman",
                wraps=session._normalize_runtime_configs_for_podman,
            ) as mock_normalize,
            patch("llm_sandbox.podman.SandboxDockerSession.open", autospec=True) as mock_super_open,
        ):
            session.open()

        mock_normalize.assert_called_once()
        mock_super_open.assert_called_once_with(session)
        assert session.config.runtime_configs["mem_limit"] == "4096m"


class TestPodmanImageManagement:
    """Tests for Podman image retrieval and pulling."""

    def test_get_or_pull_image_uses_local_image(
        self,
        podman_session_factory: Callable[..., SandboxPodmanSession],
    ) -> None:
        """_get_or_pull_image should use a local image when it exists."""
        image = MagicMock(tags=["repo:tag"])
        client = MagicMock()
        client.images.get.return_value = image
        session = podman_session_factory(client=client, image="repo:tag")

        session._get_or_pull_image()

        client.images.get.assert_called_once_with("repo:tag")
        client.images.pull.assert_not_called()
        assert session.docker_image is image
        assert session.is_create_template is False

    def test_get_or_pull_image_pulls_when_missing(
        self,
        podman_session_factory: Callable[..., SandboxPodmanSession],
    ) -> None:
        """_get_or_pull_image should pull the image when it's not present locally."""
        image = MagicMock(tags=["repo:tag"])
        client = MagicMock()
        client.images.get.side_effect = PodmanImageNotFound("missing")
        client.images.pull.return_value = image
        session = podman_session_factory(client=client, image="repo:tag")

        session._get_or_pull_image()

        client.images.pull.assert_called_once_with("repo:tag")
        assert session.docker_image is image
        assert session.is_create_template is True

    def test_get_or_pull_image_raises_image_pull_error(
        self,
        podman_session_factory: Callable[..., SandboxPodmanSession],
    ) -> None:
        """_get_or_pull_image should raise ImagePullError when pull fails."""
        client = MagicMock()
        client.images.get.side_effect = PodmanImageNotFound("missing")
        client.images.pull.side_effect = RuntimeError("boom")
        session = podman_session_factory(client=client, image="repo:tag")

        with pytest.raises(ImagePullError) as exc:
            session._get_or_pull_image()

        client.images.pull.assert_called_once_with("repo:tag")
        assert "repo:tag" in str(exc.value)


class TestPodmanExistingContainerConnection:
    """Tests for connecting to existing Podman containers."""

    def test_connect_to_existing_container_starts_when_stopped(
        self,
        podman_session_factory: Callable[..., SandboxPodmanSession],
    ) -> None:
        """Connecting to an existing stopped container should start it."""
        container = MagicMock()
        container.status = "exited"
        client = MagicMock()
        client.containers.get.return_value = container
        session = podman_session_factory(client=client)

        session._connect_to_existing_container("abc123")

        client.containers.get.assert_called_once_with("abc123")
        container.start.assert_called_once()
        assert session.container is container

    def test_connect_to_existing_container_handles_not_found(
        self,
        podman_session_factory: Callable[..., SandboxPodmanSession],
    ) -> None:
        """Connecting to a missing container should raise ContainerError."""
        client = MagicMock()
        client.containers.get.side_effect = PodmanNotFound("missing")
        session = podman_session_factory(client=client)

        with pytest.raises(ContainerError, match="Container abc123 not found"):
            session._connect_to_existing_container("abc123")

    def test_connect_to_existing_container_wraps_generic_errors(
        self,
        podman_session_factory: Callable[..., SandboxPodmanSession],
    ) -> None:
        """Unexpected errors should be wrapped in ContainerError."""
        client = MagicMock()
        client.containers.get.side_effect = RuntimeError("boom")
        session = podman_session_factory(client=client)

        with pytest.raises(ContainerError, match="Failed to connect to container abc123"):
            session._connect_to_existing_container("abc123")


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
            cmd="ls -l",
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
            (b"stdout chunk 2", b"stderr chunk 2"),
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
