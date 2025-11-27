"""Tests for Docker pool manager."""

from unittest.mock import MagicMock, patch

from docker.errors import NotFound

from llm_sandbox.const import SupportedLanguage
from llm_sandbox.pool.config import PoolConfig
from llm_sandbox.pool.docker_pool import DockerPoolManager


class TestDockerPoolManagerInitialization:
    """Test DockerPoolManager initialization."""

    @patch("llm_sandbox.pool.docker_pool.docker.from_env")
    def test_init_with_default_client(self, mock_docker_env: MagicMock) -> None:
        """Test initialization with default Docker client."""
        mock_client = MagicMock()
        mock_docker_env.return_value = mock_client

        config = PoolConfig(max_pool_size=5, enable_prewarming=False)
        pool = DockerPoolManager(config=config, lang=SupportedLanguage.PYTHON)

        assert pool.client == mock_client
        assert pool.config == config
        assert pool.lang == SupportedLanguage.PYTHON
        mock_docker_env.assert_called_once()

    def test_init_with_custom_client(self) -> None:
        """Test initialization with custom Docker client."""
        custom_client = MagicMock()
        config = PoolConfig(max_pool_size=5, enable_prewarming=False)

        pool = DockerPoolManager(config=config, lang=SupportedLanguage.PYTHON, client=custom_client)

        assert pool.client == custom_client

    @patch("llm_sandbox.pool.docker_pool.docker.from_env")
    def test_init_with_custom_image(self, mock_docker_env: MagicMock) -> None:
        """Test initialization with custom image."""
        mock_docker_env.return_value = MagicMock()
        config = PoolConfig(max_pool_size=3, enable_prewarming=False)

        pool = DockerPoolManager(config=config, lang=SupportedLanguage.PYTHON, image="custom-image:tag")

        assert pool.image == "custom-image:tag"

    @patch("llm_sandbox.pool.docker_pool.docker.from_env")
    def test_init_with_dockerfile(self, mock_docker_env: MagicMock) -> None:
        """Test initialization with Dockerfile."""
        mock_docker_env.return_value = MagicMock()
        config = PoolConfig(max_pool_size=3, enable_prewarming=False)

        pool = DockerPoolManager(
            config=config,
            lang=SupportedLanguage.PYTHON,
            dockerfile="/path/to/Dockerfile",
        )

        assert pool.dockerfile == "/path/to/Dockerfile"

    @patch("llm_sandbox.pool.docker_pool.docker.from_env")
    def test_init_with_runtime_configs(self, mock_docker_env: MagicMock) -> None:
        """Test initialization with runtime configs."""
        mock_docker_env.return_value = MagicMock()
        config = PoolConfig(max_pool_size=3, enable_prewarming=False)

        runtime_configs = {"memory": "512m", "cpu_count": 2}
        pool = DockerPoolManager(
            config=config,
            lang=SupportedLanguage.PYTHON,
            runtime_configs=runtime_configs,
        )

        assert pool.runtime_configs == runtime_configs

    @patch("llm_sandbox.pool.docker_pool.docker.from_env")
    def test_init_with_session_kwargs(self, mock_docker_env: MagicMock) -> None:
        """Test initialization passes kwargs to base class."""
        mock_docker_env.return_value = MagicMock()
        config = PoolConfig(max_pool_size=3, enable_prewarming=False)

        pool = DockerPoolManager(
            config=config,
            lang=SupportedLanguage.PYTHON,
            verbose=True,
            keep_template=True,
        )

        assert "verbose" in pool.session_kwargs
        assert "keep_template" in pool.session_kwargs


class TestDockerPoolManagerSessionCreation:
    """Test session creation for Docker pool."""

    @patch("llm_sandbox.pool.docker_pool.docker.from_env")
    @patch("llm_sandbox.docker.SandboxDockerSession")
    def test_create_session_for_container(self, mock_session_class: MagicMock, mock_docker_env: MagicMock) -> None:
        """Test creating Docker session for container."""
        mock_client = MagicMock()
        mock_docker_env.return_value = mock_client
        config = PoolConfig(max_pool_size=3, enable_prewarming=False)

        pool = DockerPoolManager(
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

    @patch("llm_sandbox.pool.docker_pool.docker.from_env")
    @patch("llm_sandbox.docker.SandboxDockerSession")
    def test_create_session_with_dockerfile(self, mock_session_class: MagicMock, mock_docker_env: MagicMock) -> None:
        """Test creating session with Dockerfile."""
        mock_docker_env.return_value = MagicMock()
        config = PoolConfig(max_pool_size=3, enable_prewarming=False)

        pool = DockerPoolManager(
            config=config,
            lang=SupportedLanguage.PYTHON,
            dockerfile="/path/to/Dockerfile",
        )

        pool._create_session_for_container()

        call_kwargs = mock_session_class.call_args[1]
        assert call_kwargs["dockerfile"] == "/path/to/Dockerfile"

    @patch("llm_sandbox.pool.docker_pool.docker.from_env")
    @patch("llm_sandbox.docker.SandboxDockerSession")
    def test_create_session_with_runtime_configs(
        self, mock_session_class: MagicMock, mock_docker_env: MagicMock
    ) -> None:
        """Test creating session with runtime configs."""
        mock_docker_env.return_value = MagicMock()
        config = PoolConfig(max_pool_size=3, enable_prewarming=False)

        runtime_configs = {"memory": "1g"}
        pool = DockerPoolManager(
            config=config,
            lang=SupportedLanguage.PYTHON,
            runtime_configs=runtime_configs,
        )

        pool._create_session_for_container()

        call_kwargs = mock_session_class.call_args[1]
        assert call_kwargs["runtime_configs"] == runtime_configs


class TestDockerPoolManagerContainerDestruction:
    """Test container destruction."""

    @patch("llm_sandbox.pool.docker_pool.docker.from_env")
    def test_destroy_container_success(self, mock_docker_env: MagicMock) -> None:
        """Test successful container destruction."""
        mock_docker_env.return_value = MagicMock()
        config = PoolConfig(max_pool_size=3, enable_prewarming=False)
        pool = DockerPoolManager(config=config, lang=SupportedLanguage.PYTHON)

        # Mock container
        mock_container = MagicMock()
        mock_container.stop = MagicMock()
        mock_container.wait = MagicMock()
        mock_container.remove = MagicMock()

        pool._destroy_container_impl(mock_container)

        mock_container.stop.assert_called_once()
        mock_container.wait.assert_called_once()
        mock_container.remove.assert_called_once_with(force=True)

    @patch("llm_sandbox.pool.docker_pool.docker.from_env")
    def test_destroy_container_handles_exceptions(self, mock_docker_env: MagicMock) -> None:
        """Test container destruction handles exceptions gracefully."""
        mock_docker_env.return_value = MagicMock()
        config = PoolConfig(max_pool_size=3, enable_prewarming=False)
        pool = DockerPoolManager(config=config, lang=SupportedLanguage.PYTHON)

        # Mock container that raises exception
        mock_container = MagicMock()
        mock_container.stop.side_effect = Exception("Test error")

        # Should not raise exception
        pool._destroy_container_impl(mock_container)


class TestDockerPoolManagerContainerId:
    """Test container ID extraction."""

    @patch("llm_sandbox.pool.docker_pool.docker.from_env")
    def test_get_container_id(self, mock_docker_env: MagicMock) -> None:
        """Test getting container ID."""
        mock_docker_env.return_value = MagicMock()
        config = PoolConfig(max_pool_size=3, enable_prewarming=False)
        pool = DockerPoolManager(config=config, lang=SupportedLanguage.PYTHON)

        mock_container = MagicMock()
        mock_container.id = "abc123def456"

        container_id = pool._get_container_id(mock_container)

        assert container_id == "abc123def456"


class TestDockerPoolManagerHealthCheck:
    """Test health check implementation."""

    @patch("llm_sandbox.pool.docker_pool.docker.from_env")
    def test_health_check_healthy_container(self, mock_docker_env: MagicMock) -> None:
        """Test health check for healthy container."""
        mock_docker_env.return_value = MagicMock()
        config = PoolConfig(max_pool_size=3, enable_prewarming=False)
        pool = DockerPoolManager(config=config, lang=SupportedLanguage.PYTHON)

        mock_container = MagicMock()
        mock_container.status = "running"
        mock_container.exec_run.return_value = (0, b"health_check")
        mock_container.reload = MagicMock()

        is_healthy = pool._health_check_impl(mock_container)

        assert is_healthy is True
        mock_container.reload.assert_called_once()
        mock_container.exec_run.assert_called_once_with("echo health_check", tty=False)

    @patch("llm_sandbox.pool.docker_pool.docker.from_env")
    def test_health_check_stopped_container(self, mock_docker_env: MagicMock) -> None:
        """Test health check for stopped container."""
        mock_docker_env.return_value = MagicMock()
        config = PoolConfig(max_pool_size=3, enable_prewarming=False)
        pool = DockerPoolManager(config=config, lang=SupportedLanguage.PYTHON)

        mock_container = MagicMock()
        mock_container.status = "stopped"
        mock_container.reload = MagicMock()

        is_healthy = pool._health_check_impl(mock_container)

        assert is_healthy is False

    @patch("llm_sandbox.pool.docker_pool.docker.from_env")
    def test_health_check_exec_failure(self, mock_docker_env: MagicMock) -> None:
        """Test health check when exec fails."""
        mock_docker_env.return_value = MagicMock()
        config = PoolConfig(max_pool_size=3, enable_prewarming=False)
        pool = DockerPoolManager(config=config, lang=SupportedLanguage.PYTHON)

        mock_container = MagicMock()
        mock_container.status = "running"
        mock_container.exec_run.return_value = (1, b"error")  # Non-zero exit code
        mock_container.reload = MagicMock()

        is_healthy = pool._health_check_impl(mock_container)

        assert is_healthy is False

    @patch("llm_sandbox.pool.docker_pool.docker.from_env")
    def test_health_check_container_not_found(self, mock_docker_env: MagicMock) -> None:
        """Test health check when container not found."""
        mock_docker_env.return_value = MagicMock()
        config = PoolConfig(max_pool_size=3, enable_prewarming=False)
        pool = DockerPoolManager(config=config, lang=SupportedLanguage.PYTHON)

        mock_container = MagicMock()
        mock_container.reload.side_effect = NotFound("Container not found")

        is_healthy = pool._health_check_impl(mock_container)

        assert is_healthy is False

    @patch("llm_sandbox.pool.docker_pool.docker.from_env")
    def test_health_check_generic_exception(self, mock_docker_env: MagicMock) -> None:
        """Test health check handles generic exceptions."""
        mock_docker_env.return_value = MagicMock()
        config = PoolConfig(max_pool_size=3, enable_prewarming=False)
        pool = DockerPoolManager(config=config, lang=SupportedLanguage.PYTHON)

        mock_container = MagicMock()
        mock_container.reload.side_effect = Exception("Unexpected error")

        is_healthy = pool._health_check_impl(mock_container)

        assert is_healthy is False
