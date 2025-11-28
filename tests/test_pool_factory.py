"""Tests for pool factory module."""

from unittest.mock import MagicMock, patch

import pytest

from llm_sandbox.const import SandboxBackend, SupportedLanguage
from llm_sandbox.exceptions import UnsupportedBackendError
from llm_sandbox.pool.config import PoolConfig
from llm_sandbox.pool.factory import create_pool_manager


class TestCreatePoolManager:
    """Test create_pool_manager factory function."""

    @patch("llm_sandbox.pool.docker_pool.docker.from_env")
    def test_create_docker_pool_manager(self, mock_docker_env: MagicMock) -> None:
        """Test creating Docker pool manager."""
        mock_client = MagicMock()
        mock_docker_env.return_value = mock_client

        pool = create_pool_manager(backend=SandboxBackend.DOCKER)

        assert pool is not None
        assert pool.__class__.__name__ == "DockerPoolManager"
        mock_docker_env.assert_called_once()

    @patch("kubernetes.config.load_kube_config")
    @patch("llm_sandbox.pool.kubernetes_pool.CoreV1Api")
    def test_create_kubernetes_pool_manager(self, mock_core_api: MagicMock, mock_load_config: MagicMock) -> None:
        """Test creating Kubernetes pool manager."""
        mock_client = MagicMock()
        mock_core_api.return_value = mock_client

        pool = create_pool_manager(backend=SandboxBackend.KUBERNETES)

        assert pool is not None
        assert pool.__class__.__name__ == "KubernetesPoolManager"
        mock_load_config.assert_called_once()

    @patch("llm_sandbox.pool.podman_pool.PodmanClient.from_env")
    def test_create_podman_pool_manager(self, mock_podman_client: MagicMock) -> None:
        """Test creating Podman pool manager."""
        mock_client = MagicMock()
        mock_podman_client.from_env.return_value = mock_client

        pool = create_pool_manager(backend=SandboxBackend.PODMAN)

        assert pool is not None
        assert pool.__class__.__name__ == "PodmanPoolManager"
        mock_podman_client.assert_called_once()

    @patch("llm_sandbox.pool.docker_pool.docker.from_env")
    def test_create_with_custom_client(self, mock_docker_env: MagicMock) -> None:
        """Test creating pool manager with custom client."""
        custom_client = MagicMock()
        pool = create_pool_manager(backend=SandboxBackend.DOCKER, client=custom_client)

        assert pool is not None
        # Should not call from_env when client is provided
        mock_docker_env.assert_not_called()

    @patch("llm_sandbox.pool.docker_pool.docker.from_env")
    def test_create_with_default_config(self, mock_docker_env: MagicMock) -> None:
        """Test creating pool manager with default config."""
        mock_docker_env.return_value = MagicMock()

        pool = create_pool_manager(backend=SandboxBackend.DOCKER)

        # Should use default PoolConfig values
        assert pool.config.max_pool_size == 10
        assert pool.config.min_pool_size == 0

    @patch("llm_sandbox.pool.docker_pool.docker.from_env")
    def test_create_with_custom_config(self, mock_docker_env: MagicMock) -> None:
        """Test creating pool manager with custom config."""
        mock_docker_env.return_value = MagicMock()

        custom_config = PoolConfig(max_pool_size=20, min_pool_size=5)
        pool = create_pool_manager(backend=SandboxBackend.DOCKER, config=custom_config)

        assert pool.config.max_pool_size == 20
        assert pool.config.min_pool_size == 5

    @patch("llm_sandbox.pool.docker_pool.docker.from_env")
    def test_create_with_default_language(self, mock_docker_env: MagicMock) -> None:
        """Test creating pool manager with default language (Python)."""
        mock_docker_env.return_value = MagicMock()

        pool = create_pool_manager(backend=SandboxBackend.DOCKER)

        assert pool.lang == SupportedLanguage.PYTHON

    @patch("llm_sandbox.pool.docker_pool.docker.from_env")
    def test_create_with_custom_language(self, mock_docker_env: MagicMock) -> None:
        """Test creating pool manager with custom language."""
        mock_docker_env.return_value = MagicMock()

        pool = create_pool_manager(backend=SandboxBackend.DOCKER, lang=SupportedLanguage.JAVASCRIPT)

        assert pool.lang == SupportedLanguage.JAVASCRIPT

    @patch("llm_sandbox.pool.docker_pool.docker.from_env")
    def test_create_with_additional_kwargs(self, mock_docker_env: MagicMock) -> None:
        """Test creating pool manager with additional kwargs."""
        mock_docker_env.return_value = MagicMock()

        pool = create_pool_manager(
            backend=SandboxBackend.DOCKER,
            image="custom:image",
            runtime_configs={"memory": "512m"},
        )

        from llm_sandbox.pool.docker_pool import DockerPoolManager

        assert pool.image == "custom:image"
        assert isinstance(pool, DockerPoolManager)
        assert pool.runtime_configs == {"memory": "512m"}

    @patch("kubernetes.config.load_kube_config")
    @patch("llm_sandbox.pool.kubernetes_pool.CoreV1Api")
    def test_create_kubernetes_with_namespace(self, mock_core_api: MagicMock, _mock_load_config: MagicMock) -> None:
        """Test creating Kubernetes pool with custom namespace."""
        mock_core_api.return_value = MagicMock()

        pool = create_pool_manager(
            backend=SandboxBackend.KUBERNETES,
            namespace="custom-namespace",
        )

        from llm_sandbox.pool.kubernetes_pool import KubernetesPoolManager

        assert isinstance(pool, KubernetesPoolManager)
        assert pool.namespace == "custom-namespace"

    def test_unsupported_backend_raises_error(self) -> None:
        """Test that unsupported backend raises UnsupportedBackendError."""
        # Create a mock backend that doesn't exist
        with pytest.raises(UnsupportedBackendError):
            create_pool_manager(backend="invalid_backend")  # type: ignore[arg-type]

    @patch("llm_sandbox.pool.docker_pool.docker.from_env")
    def test_all_parameters_passed_through(self, mock_docker_env: MagicMock) -> None:
        """Test all parameters are passed through to pool manager."""
        mock_docker_env.return_value = MagicMock()

        config = PoolConfig(max_pool_size=15)
        pool = create_pool_manager(
            backend=SandboxBackend.DOCKER,
            config=config,
            lang=SupportedLanguage.PYTHON,
            image="test:latest",
            verbose=True,
        )

        assert pool.config.max_pool_size == 15
        assert pool.lang == SupportedLanguage.PYTHON
        assert pool.image == "test:latest"

    @patch("llm_sandbox.pool.podman_pool.PodmanClient")
    def test_podman_with_dockerfile(self, mock_podman_client: MagicMock) -> None:
        """Test creating Podman pool with dockerfile."""
        mock_podman_client.return_value = MagicMock()

        pool = create_pool_manager(
            backend=SandboxBackend.PODMAN,
            dockerfile="/path/to/Dockerfile",
        )

        from llm_sandbox.pool.podman_pool import PodmanPoolManager

        assert isinstance(pool, PodmanPoolManager)
        assert pool.dockerfile == "/path/to/Dockerfile"


class TestFactoryEdgeCases:
    """Test edge cases for factory function."""

    @patch("llm_sandbox.pool.docker_pool.docker.from_env")
    def test_none_config_uses_defaults(self, mock_docker_env: MagicMock) -> None:
        """Test that None config creates default PoolConfig."""
        mock_docker_env.return_value = MagicMock()

        pool = create_pool_manager(backend=SandboxBackend.DOCKER, config=None)

        # Should use default values
        assert pool.config.max_pool_size == 10
        assert pool.config.exhaustion_strategy.value == "wait"

    @patch("llm_sandbox.pool.docker_pool.docker.from_env")
    def test_empty_kwargs(self, mock_docker_env: MagicMock) -> None:
        """Test creating pool with no additional kwargs."""
        mock_docker_env.return_value = MagicMock()

        pool = create_pool_manager(backend=SandboxBackend.DOCKER)

        assert pool is not None

    @patch("kubernetes.config.load_kube_config")
    @patch("llm_sandbox.pool.kubernetes_pool.CoreV1Api")
    def test_kubernetes_with_pod_manifest(self, mock_core_api: MagicMock, _mock_load_config: MagicMock) -> None:
        """Test creating Kubernetes pool with custom pod manifest."""
        mock_core_api.return_value = MagicMock()

        custom_manifest = {"apiVersion": "v1", "kind": "Pod"}
        pool = create_pool_manager(
            backend=SandboxBackend.KUBERNETES,
            pod_manifest=custom_manifest,
        )

        from llm_sandbox.pool.kubernetes_pool import KubernetesPoolManager

        assert isinstance(pool, KubernetesPoolManager)
        assert pool.pod_manifest_template == custom_manifest
