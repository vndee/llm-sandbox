"""Tests for Podman pool manager."""

from unittest.mock import MagicMock, patch

from llm_sandbox.const import SupportedLanguage
from llm_sandbox.pool.config import PoolConfig
from llm_sandbox.pool.podman_pool import PodmanPoolManager


class TestPodmanPoolManagerInitialization:
    """Test PodmanPoolManager initialization."""

    @patch("llm_sandbox.pool.podman_pool.PodmanClient.from_env")
    def test_init_with_default_client(self, mock_podman_client: MagicMock) -> None:
        """Test initialization with default Podman client."""
        mock_client = MagicMock()
        mock_podman_client.return_value = mock_client

        config = PoolConfig(max_pool_size=5, enable_prewarming=False)
        pool = PodmanPoolManager(config=config, lang=SupportedLanguage.PYTHON)

        assert pool.client == mock_client

    def test_init_with_custom_client(self) -> None:
        """Test initialization with custom Podman client."""
        custom_client = MagicMock()
        config = PoolConfig(max_pool_size=5, enable_prewarming=False)

        pool = PodmanPoolManager(config=config, lang=SupportedLanguage.PYTHON, client=custom_client)

        assert pool.client == custom_client

    @patch("llm_sandbox.pool.podman_pool.PodmanClient.from_env")
    def test_init_with_dockerfile(self, mock_podman_client: MagicMock) -> None:
        """Test initialization with Dockerfile."""
        mock_podman_client.from_env.return_value = MagicMock()
        config = PoolConfig(max_pool_size=3, enable_prewarming=False)

        pool = PodmanPoolManager(
            config=config,
            lang=SupportedLanguage.PYTHON,
            dockerfile="/path/to/Dockerfile",
        )

        assert pool.dockerfile == "/path/to/Dockerfile"

    @patch("llm_sandbox.pool.podman_pool.PodmanClient")
    def test_init_with_runtime_configs(self, mock_podman_client: MagicMock) -> None:
        """Test initialization with runtime configs."""
        mock_podman_client.return_value = MagicMock()
        config = PoolConfig(max_pool_size=3, enable_prewarming=False)

        runtime_configs = {"memory": "512m"}
        pool = PodmanPoolManager(
            config=config,
            lang=SupportedLanguage.PYTHON,
            runtime_configs=runtime_configs,
        )

        assert pool.runtime_configs == runtime_configs

    @patch("llm_sandbox.pool.podman_pool.PodmanClient")
    def test_inherits_from_docker_pool(self, mock_podman_client: MagicMock) -> None:
        """Test that PodmanPoolManager inherits from DockerPoolManager."""
        from llm_sandbox.pool.docker_pool import DockerPoolManager

        mock_podman_client.return_value = MagicMock()
        config = PoolConfig(max_pool_size=3, enable_prewarming=False)

        pool = PodmanPoolManager(config=config, lang=SupportedLanguage.PYTHON)

        assert isinstance(pool, DockerPoolManager)


class TestPodmanPoolManagerSessionCreation:
    """Test session creation for Podman pool."""

    @patch("llm_sandbox.pool.podman_pool.PodmanClient.from_env")
    @patch("llm_sandbox.podman.SandboxPodmanSession")
    def test_create_session_for_container(self, mock_session_class: MagicMock, mock_podman_client: MagicMock) -> None:
        """Test creating Podman session for container."""
        mock_client = MagicMock()
        mock_podman_client.return_value = mock_client
        config = PoolConfig(max_pool_size=3, enable_prewarming=False)

        pool = PodmanPoolManager(
            config=config,
            lang=SupportedLanguage.PYTHON,
            image="test:latest",
        )

        pool._create_session_for_container()

        # Verify session was created with correct parameters
        mock_session_class.assert_called_once()
        call_kwargs = mock_session_class.call_args[1]
        assert call_kwargs["client"] == mock_client
        assert call_kwargs["image"] == "test:latest"
        assert call_kwargs["lang"] == "python"

    @patch("llm_sandbox.pool.podman_pool.PodmanClient")
    @patch("llm_sandbox.podman.SandboxPodmanSession")
    def test_create_session_with_dockerfile(self, mock_session_class: MagicMock, mock_podman_client: MagicMock) -> None:
        """Test creating session with Dockerfile."""
        mock_podman_client.return_value = MagicMock()
        config = PoolConfig(max_pool_size=3, enable_prewarming=False)

        pool = PodmanPoolManager(
            config=config,
            lang=SupportedLanguage.PYTHON,
            dockerfile="/path/to/Dockerfile",
        )

        pool._create_session_for_container()

        call_kwargs = mock_session_class.call_args[1]
        assert call_kwargs["dockerfile"] == "/path/to/Dockerfile"

    @patch("llm_sandbox.pool.podman_pool.PodmanClient.from_env")
    @patch("llm_sandbox.podman.SandboxPodmanSession")
    def test_create_session_with_string_lang(self, mock_session_class: MagicMock, mock_podman_client: MagicMock) -> None:
        """Test creating Podman session with string language instead of SupportedLanguage enum."""
        mock_client = MagicMock()
        mock_podman_client.return_value = mock_client
        config = PoolConfig(max_pool_size=3, enable_prewarming=False)

        pool = PodmanPoolManager(
            config=config,
            lang="python",  # String instead of SupportedLanguage enum
            image="test:latest",
        )

        pool._create_session_for_container()

        # Verify session was created with correct parameters
        mock_session_class.assert_called_once()
        call_kwargs = mock_session_class.call_args[1]
        assert call_kwargs["client"] == mock_client
        assert call_kwargs["image"] == "test:latest"
        assert call_kwargs["lang"] == "python"
