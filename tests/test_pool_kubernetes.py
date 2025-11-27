"""Tests for Kubernetes pool manager."""

from unittest.mock import MagicMock, patch

from kubernetes.client.exceptions import ApiException

from llm_sandbox.const import SupportedLanguage
from llm_sandbox.pool.config import PoolConfig
from llm_sandbox.pool.kubernetes_pool import KubernetesPoolManager


class TestKubernetesPoolManagerInitialization:
    """Test KubernetesPoolManager initialization."""

    @patch("kubernetes.config.load_kube_config")
    @patch("llm_sandbox.pool.kubernetes_pool.CoreV1Api")
    def test_init_with_default_client(self, mock_core_api: MagicMock, mock_load_config: MagicMock) -> None:
        """Test initialization with default K8s client."""
        mock_client = MagicMock()
        mock_core_api.return_value = mock_client

        config = PoolConfig(max_pool_size=5, enable_prewarming=False)
        pool = KubernetesPoolManager(config=config, lang=SupportedLanguage.PYTHON)

        assert pool.client == mock_client
        mock_load_config.assert_called_once()

    @patch("kubernetes.config.load_kube_config")
    @patch("llm_sandbox.pool.kubernetes_pool.CoreV1Api")
    def test_init_with_incluster_config_fallback(self, mock_core_api: MagicMock, mock_load_config: MagicMock) -> None:
        """Test initialization falls back to incluster config."""
        mock_load_config.side_effect = Exception("No kubeconfig")
        mock_client = MagicMock()
        mock_core_api.return_value = mock_client

        with patch("kubernetes.config.load_incluster_config") as mock_incluster:
            config = PoolConfig(max_pool_size=5, enable_prewarming=False)
            pool = KubernetesPoolManager(config=config, lang=SupportedLanguage.PYTHON)

            assert pool.client == mock_client
            mock_incluster.assert_called_once()

    def test_init_with_custom_client(self) -> None:
        """Test initialization with custom K8s client."""
        custom_client = MagicMock()
        config = PoolConfig(max_pool_size=5, enable_prewarming=False)

        pool = KubernetesPoolManager(config=config, lang=SupportedLanguage.PYTHON, client=custom_client)

        assert pool.client == custom_client

    @patch("kubernetes.config.load_kube_config")
    @patch("llm_sandbox.pool.kubernetes_pool.CoreV1Api")
    def test_init_with_custom_namespace(self, mock_core_api: MagicMock, _mock_load_config: MagicMock) -> None:
        """Test initialization with custom namespace."""
        mock_core_api.return_value = MagicMock()
        config = PoolConfig(max_pool_size=3, enable_prewarming=False)

        pool = KubernetesPoolManager(
            config=config,
            lang=SupportedLanguage.PYTHON,
            namespace="custom-ns",
        )

        assert pool.namespace == "custom-ns"

    @patch("kubernetes.config.load_kube_config")
    @patch("llm_sandbox.pool.kubernetes_pool.CoreV1Api")
    def test_init_with_pod_manifest(self, mock_core_api: MagicMock, _mock_load_config: MagicMock) -> None:
        """Test initialization with custom pod manifest."""
        mock_core_api.return_value = MagicMock()
        config = PoolConfig(max_pool_size=3, enable_prewarming=False)

        manifest = {"apiVersion": "v1", "kind": "Pod"}
        pool = KubernetesPoolManager(
            config=config,
            lang=SupportedLanguage.PYTHON,
            pod_manifest=manifest,
        )

        assert pool.pod_manifest_template == manifest


class TestKubernetesPoolManagerSessionCreation:
    """Test session creation for Kubernetes pool."""

    @patch("kubernetes.config.load_kube_config")
    @patch("llm_sandbox.pool.kubernetes_pool.CoreV1Api")
    @patch("llm_sandbox.kubernetes.SandboxKubernetesSession")
    def test_create_session_for_container(
        self, mock_session_class: MagicMock, mock_core_api: MagicMock, _mock_load_config: MagicMock
    ) -> None:
        """Test creating Kubernetes session for container."""
        mock_client = MagicMock()
        mock_core_api.return_value = mock_client
        config = PoolConfig(max_pool_size=3, enable_prewarming=False)

        pool = KubernetesPoolManager(
            config=config,
            lang=SupportedLanguage.PYTHON,
            image="test:latest",
            namespace="test-ns",
        )

        pool._create_session_for_container()

        # Verify session was created with correct parameters
        mock_session_class.assert_called_once()
        call_kwargs = mock_session_class.call_args[1]
        assert call_kwargs["client"] == mock_client
        assert call_kwargs["image"] == "test:latest"
        assert call_kwargs["lang"] == "python"
        assert call_kwargs["namespace"] == "test-ns"


class TestKubernetesPoolManagerContainerDestruction:
    """Test pod destruction."""

    @patch("kubernetes.config.load_kube_config")
    @patch("llm_sandbox.pool.kubernetes_pool.CoreV1Api")
    def test_destroy_pod_with_pod_object(self, mock_core_api: MagicMock, _mock_load_config: MagicMock) -> None:
        """Test destroying pod with pod object."""
        mock_client = MagicMock()
        mock_core_api.return_value = mock_client
        config = PoolConfig(max_pool_size=3, enable_prewarming=False)
        pool = KubernetesPoolManager(config=config, lang=SupportedLanguage.PYTHON)

        # Mock pod object
        mock_pod = MagicMock()
        mock_pod.metadata.name = "test-pod-123"

        with patch("llm_sandbox.pool.kubernetes_pool.retry_k8s_api_call") as mock_retry:
            pool._destroy_container_impl(mock_pod)
            mock_retry.assert_called_once()

    @patch("kubernetes.config.load_kube_config")
    @patch("llm_sandbox.pool.kubernetes_pool.CoreV1Api")
    def test_destroy_pod_with_pod_name_string(self, mock_core_api: MagicMock, _mock_load_config: MagicMock) -> None:
        """Test destroying pod with pod name string."""
        mock_client = MagicMock()
        mock_core_api.return_value = mock_client
        config = PoolConfig(max_pool_size=3, enable_prewarming=False)
        pool = KubernetesPoolManager(config=config, lang=SupportedLanguage.PYTHON)

        with patch("llm_sandbox.pool.kubernetes_pool.retry_k8s_api_call") as mock_retry:
            pool._destroy_container_impl("test-pod-name")
            mock_retry.assert_called_once()

    @patch("kubernetes.config.load_kube_config")
    @patch("llm_sandbox.pool.kubernetes_pool.CoreV1Api")
    def test_destroy_pod_not_found(self, mock_core_api: MagicMock, _mock_load_config: MagicMock) -> None:
        """Test destroying pod that doesn't exist."""
        mock_client = MagicMock()
        mock_core_api.return_value = mock_client
        config = PoolConfig(max_pool_size=3, enable_prewarming=False)
        pool = KubernetesPoolManager(config=config, lang=SupportedLanguage.PYTHON)

        # Not found exception (404)
        not_found_error = ApiException(status=404)

        with patch("llm_sandbox.pool.kubernetes_pool.retry_k8s_api_call", side_effect=not_found_error):
            # Should not raise exception
            pool._destroy_container_impl("test-pod")

    @patch("kubernetes.config.load_kube_config")
    @patch("llm_sandbox.pool.kubernetes_pool.CoreV1Api")
    def test_destroy_pod_other_api_exception(self, mock_core_api: MagicMock, _mock_load_config: MagicMock) -> None:
        """Test destroying pod with other API exception."""
        mock_client = MagicMock()
        mock_core_api.return_value = mock_client
        config = PoolConfig(max_pool_size=3, enable_prewarming=False)
        pool = KubernetesPoolManager(config=config, lang=SupportedLanguage.PYTHON)

        # Other API exception
        api_error = ApiException(status=500)

        with patch("llm_sandbox.pool.kubernetes_pool.retry_k8s_api_call", side_effect=api_error):
            # Should not raise exception
            pool._destroy_container_impl("test-pod")


class TestKubernetesPoolManagerContainerId:
    """Test pod name/ID extraction."""

    @patch("kubernetes.config.load_kube_config")
    @patch("llm_sandbox.pool.kubernetes_pool.CoreV1Api")
    def test_get_container_id_from_pod_object(self, mock_core_api: MagicMock, _mock_load_config: MagicMock) -> None:
        """Test getting pod name from pod object."""
        mock_core_api.return_value = MagicMock()
        config = PoolConfig(max_pool_size=3, enable_prewarming=False)
        pool = KubernetesPoolManager(config=config, lang=SupportedLanguage.PYTHON)

        mock_pod = MagicMock()
        mock_pod.metadata.name = "my-pod-123"

        pod_name = pool._get_container_id(mock_pod)
        assert pod_name == "my-pod-123"

    @patch("kubernetes.config.load_kube_config")
    @patch("llm_sandbox.pool.kubernetes_pool.CoreV1Api")
    def test_get_container_id_from_string(self, mock_core_api: MagicMock, _mock_load_config: MagicMock) -> None:
        """Test getting pod name from string."""
        mock_core_api.return_value = MagicMock()
        config = PoolConfig(max_pool_size=3, enable_prewarming=False)
        pool = KubernetesPoolManager(config=config, lang=SupportedLanguage.PYTHON)

        pod_name = pool._get_container_id("pod-name-string")
        assert pod_name == "pod-name-string"


class TestKubernetesPoolManagerHealthCheck:
    """Test health check implementation."""

    @patch("kubernetes.config.load_kube_config")
    @patch("llm_sandbox.pool.kubernetes_pool.CoreV1Api")
    def test_health_check_healthy_pod(self, mock_core_api: MagicMock, _mock_load_config: MagicMock) -> None:
        """Test health check for healthy pod."""
        mock_core_api.return_value = MagicMock()
        config = PoolConfig(max_pool_size=3, enable_prewarming=False)
        pool = KubernetesPoolManager(config=config, lang=SupportedLanguage.PYTHON)

        # Mock healthy pod
        mock_pod = MagicMock()
        mock_pod.status.phase = "Running"
        mock_container_status = MagicMock()
        mock_container_status.ready = True
        mock_pod.status.container_statuses = [mock_container_status]

        with patch("llm_sandbox.pool.kubernetes_pool.retry_k8s_api_call", return_value=mock_pod):
            is_healthy = pool._health_check_impl("test-pod")
            assert is_healthy is True

    @patch("kubernetes.config.load_kube_config")
    @patch("llm_sandbox.pool.kubernetes_pool.CoreV1Api")
    def test_health_check_pod_not_running(self, mock_core_api: MagicMock, _mock_load_config: MagicMock) -> None:
        """Test health check for pod not in Running phase."""
        mock_core_api.return_value = MagicMock()
        config = PoolConfig(max_pool_size=3, enable_prewarming=False)
        pool = KubernetesPoolManager(config=config, lang=SupportedLanguage.PYTHON)

        mock_pod = MagicMock()
        mock_pod.status.phase = "Pending"

        with patch("llm_sandbox.pool.kubernetes_pool.retry_k8s_api_call", return_value=mock_pod):
            is_healthy = pool._health_check_impl("test-pod")
            assert is_healthy is False

    @patch("kubernetes.config.load_kube_config")
    @patch("llm_sandbox.pool.kubernetes_pool.CoreV1Api")
    def test_health_check_container_not_ready(self, mock_core_api: MagicMock, _mock_load_config: MagicMock) -> None:
        """Test health check when container not ready."""
        mock_core_api.return_value = MagicMock()
        config = PoolConfig(max_pool_size=3, enable_prewarming=False)
        pool = KubernetesPoolManager(config=config, lang=SupportedLanguage.PYTHON)

        mock_pod = MagicMock()
        mock_pod.status.phase = "Running"
        mock_container_status = MagicMock()
        mock_container_status.ready = False
        mock_pod.status.container_statuses = [mock_container_status]

        with patch("llm_sandbox.pool.kubernetes_pool.retry_k8s_api_call", return_value=mock_pod):
            is_healthy = pool._health_check_impl("test-pod")
            assert is_healthy is False

    @patch("kubernetes.config.load_kube_config")
    @patch("llm_sandbox.pool.kubernetes_pool.CoreV1Api")
    def test_health_check_pod_not_found(self, mock_core_api: MagicMock, _mock_load_config: MagicMock) -> None:
        """Test health check when pod not found."""
        mock_core_api.return_value = MagicMock()
        config = PoolConfig(max_pool_size=3, enable_prewarming=False)
        pool = KubernetesPoolManager(config=config, lang=SupportedLanguage.PYTHON)

        not_found_error = ApiException(status=404)

        with patch("llm_sandbox.pool.kubernetes_pool.retry_k8s_api_call", side_effect=not_found_error):
            is_healthy = pool._health_check_impl("test-pod")
            assert is_healthy is False

    @patch("kubernetes.config.load_kube_config")
    @patch("llm_sandbox.pool.kubernetes_pool.CoreV1Api")
    def test_health_check_generic_exception(self, mock_core_api: MagicMock, _mock_load_config: MagicMock) -> None:
        """Test health check handles generic exceptions."""
        mock_core_api.return_value = MagicMock()
        config = PoolConfig(max_pool_size=3, enable_prewarming=False)
        pool = KubernetesPoolManager(config=config, lang=SupportedLanguage.PYTHON)

        with patch("llm_sandbox.pool.kubernetes_pool.retry_k8s_api_call", side_effect=Exception("Test error")):
            is_healthy = pool._health_check_impl("test-pod")
            assert is_healthy is False

    @patch("kubernetes.config.load_kube_config")
    @patch("llm_sandbox.pool.kubernetes_pool.CoreV1Api")
    def test_destroy_pod_generic_exception(self, mock_core_api: MagicMock, _mock_load_config: MagicMock) -> None:
        """Test destroying pod handles generic exceptions."""
        mock_core_api.return_value = MagicMock()
        config = PoolConfig(max_pool_size=3, enable_prewarming=False)
        pool = KubernetesPoolManager(config=config, lang=SupportedLanguage.PYTHON)

        with patch("llm_sandbox.pool.kubernetes_pool.retry_k8s_api_call", side_effect=Exception("Generic error")):
            # Should not raise exception
            pool._destroy_container_impl("test-pod")

    @patch("kubernetes.config.load_kube_config")
    @patch("llm_sandbox.pool.kubernetes_pool.CoreV1Api")
    def test_health_check_no_container_statuses(self, mock_core_api: MagicMock, _mock_load_config: MagicMock) -> None:
        """Test health check when pod has no container_statuses."""
        mock_core_api.return_value = MagicMock()
        config = PoolConfig(max_pool_size=3, enable_prewarming=False)
        pool = KubernetesPoolManager(config=config, lang=SupportedLanguage.PYTHON)

        # Mock pod with no container_statuses
        mock_pod = MagicMock()
        mock_pod.status.phase = "Running"
        mock_pod.status.container_statuses = None

        with patch("llm_sandbox.pool.kubernetes_pool.retry_k8s_api_call", return_value=mock_pod):
            is_healthy = pool._health_check_impl("test-pod")
            # Should still be healthy if Running and no container_statuses
            assert is_healthy is True
