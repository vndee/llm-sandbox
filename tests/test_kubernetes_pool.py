# ruff: noqa: SLF001, PLR2004, ARG002, PT011, PT012

"""Tests for Kubernetes Pod Pool implementation."""

import threading
from unittest.mock import MagicMock, patch

import pytest
from kubernetes.client.exceptions import ApiException

from llm_sandbox.const import DefaultImage, SandboxBackend, SupportedLanguage
from llm_sandbox.exceptions import ContainerError
from llm_sandbox.kubernetes_pool import KubernetesPodPool


class TestKubernetesPodPoolInit:
    """Test KubernetesPodPool initialization."""

    @patch("kubernetes.config.load_kube_config")
    @patch("llm_sandbox.kubernetes_pool.CoreV1Api")
    @patch("llm_sandbox.kubernetes_pool.AppsV1Api")
    def test_init_with_defaults(
        self, mock_apps_v1_api: MagicMock, mock_core_v1_api: MagicMock, mock_load_config: MagicMock
    ) -> None:
        """Test initialization with default parameters."""
        mock_core_client = MagicMock()
        mock_core_v1_api.return_value = mock_core_client

        pool = KubernetesPodPool()

        assert pool.namespace == "default"
        assert pool.pool_size == 5
        assert pool.deployment_name == "llm-sandbox-pool"
        assert pool.lang == SupportedLanguage.PYTHON
        assert pool.image == DefaultImage.PYTHON
        assert pool.core_v1 == mock_core_client
        assert pool.verbose is False
        mock_load_config.assert_called_once()

    def test_init_with_custom_client(self) -> None:
        """Test initialization with custom Kubernetes client."""
        custom_client = MagicMock()

        pool = KubernetesPodPool(client=custom_client)

        assert pool.core_v1 == custom_client

    @patch("kubernetes.config.load_kube_config")
    @patch("llm_sandbox.kubernetes_pool.CoreV1Api")
    @patch("llm_sandbox.kubernetes_pool.AppsV1Api")
    def test_init_with_custom_params(
        self, mock_apps_v1_api: MagicMock, mock_core_v1_api: MagicMock, mock_load_config: MagicMock
    ) -> None:
        """Test initialization with custom parameters."""
        mock_core_client = MagicMock()
        mock_core_v1_api.return_value = mock_core_client

        pool = KubernetesPodPool(
            namespace="custom-ns",
            pool_size=10,
            deployment_name="custom-pool",
            image="custom:latest",
            lang="java",
            verbose=True,
            warmup_timeout=600,
            acquisition_timeout=60,
        )

        assert pool.namespace == "custom-ns"
        assert pool.pool_size == 10
        assert pool.deployment_name == "custom-pool"
        assert pool.image == "custom:latest"
        assert pool.lang == "java"
        assert pool.verbose is True
        assert pool.warmup_timeout == 600
        assert pool.acquisition_timeout == 60


class TestKubernetesPodPoolPodTemplate:
    """Test pod template generation."""

    @patch("kubernetes.config.load_kube_config")
    @patch("llm_sandbox.kubernetes_pool.CoreV1Api")
    @patch("llm_sandbox.kubernetes_pool.AppsV1Api")
    def test_default_pod_template(
        self, mock_apps_v1_api: MagicMock, mock_core_v1_api: MagicMock, mock_load_config: MagicMock
    ) -> None:
        """Test default pod template generation."""
        pool = KubernetesPodPool(deployment_name="test-pool", lang="python")

        template = pool._get_default_pod_template()

        assert template["metadata"]["labels"]["app"] == "llm-sandbox-pool"
        assert template["metadata"]["labels"]["pool"] == "test-pool"
        assert template["metadata"]["labels"]["lang"] == "python"

        container = template["spec"]["containers"][0]
        assert container["name"] == "sandbox"
        assert container["image"] == DefaultImage.PYTHON
        assert container["command"] == ["tail", "-f", "/dev/null"]

        # Check security context
        assert container["securityContext"]["runAsUser"] == 0
        assert container["securityContext"]["runAsGroup"] == 0
        assert template["spec"]["securityContext"]["runAsUser"] == 0
        assert template["spec"]["securityContext"]["runAsGroup"] == 0

    @patch("kubernetes.config.load_kube_config")
    @patch("llm_sandbox.kubernetes_pool.CoreV1Api")
    @patch("llm_sandbox.kubernetes_pool.AppsV1Api")
    def test_custom_pod_template(
        self, mock_apps_v1_api: MagicMock, mock_core_v1_api: MagicMock, mock_load_config: MagicMock
    ) -> None:
        """Test custom pod template usage."""
        custom_template = {
            "metadata": {"labels": {"custom": "template"}},
            "spec": {"containers": [{"name": "custom", "image": "custom:latest"}]},
        }

        pool = KubernetesPodPool(pod_template=custom_template)

        manifest = pool._create_deployment_manifest()
        assert manifest["spec"]["template"] == custom_template


class TestKubernetesPodPoolSetup:
    """Test pod pool setup functionality."""

    @patch("kubernetes.config.load_kube_config")
    @patch("llm_sandbox.kubernetes_pool.CoreV1Api")
    @patch("llm_sandbox.kubernetes_pool.AppsV1Api")
    def test_setup_creates_new_deployment(
        self, mock_apps_v1_api: MagicMock, mock_core_v1_api: MagicMock, mock_load_config: MagicMock
    ) -> None:
        """Test setup creates new deployment when it doesn't exist."""
        mock_apps_client = MagicMock()
        mock_apps_v1_api.return_value = mock_apps_client

        # Mock deployment not found
        mock_apps_client.read_namespaced_deployment.side_effect = ApiException(status=404)

        pool = KubernetesPodPool(pool_size=3, verbose=True)

        with (
            patch.object(pool, "_wait_for_ready_pods") as mock_wait,
            patch.object(pool, "_start_health_check") as mock_health,
        ):
            pool.setup()

        mock_apps_client.create_namespaced_deployment.assert_called_once()
        mock_wait.assert_called_once()
        mock_health.assert_called_once()

    @patch("kubernetes.config.load_kube_config")
    @patch("llm_sandbox.kubernetes_pool.CoreV1Api")
    @patch("llm_sandbox.kubernetes_pool.AppsV1Api")
    def test_setup_updates_existing_deployment(
        self, mock_apps_v1_api: MagicMock, mock_core_v1_api: MagicMock, mock_load_config: MagicMock
    ) -> None:
        """Test setup updates existing deployment with different replica count."""
        mock_apps_client = MagicMock()
        mock_apps_v1_api.return_value = mock_apps_client

        # Mock existing deployment with different replica count
        mock_deployment = MagicMock()
        mock_deployment.spec.replicas = 3
        mock_apps_client.read_namespaced_deployment.return_value = mock_deployment

        pool = KubernetesPodPool(pool_size=5)

        with (
            patch.object(pool, "scale") as mock_scale,
            patch.object(pool, "_wait_for_ready_pods"),
            patch.object(pool, "_start_health_check"),
        ):
            pool.setup()

        mock_scale.assert_called_once_with(5)

    @patch("kubernetes.config.load_kube_config")
    @patch("llm_sandbox.kubernetes_pool.CoreV1Api")
    @patch("llm_sandbox.kubernetes_pool.AppsV1Api")
    def test_setup_with_api_exception(
        self, mock_apps_v1_api: MagicMock, mock_core_v1_api: MagicMock, mock_load_config: MagicMock
    ) -> None:
        """Test setup handles API exceptions properly."""
        mock_apps_client = MagicMock()
        mock_apps_v1_api.return_value = mock_apps_client

        mock_apps_client.read_namespaced_deployment.side_effect = ApiException(status=500)

        pool = KubernetesPodPool()

        with pytest.raises(ContainerError, match="Failed to setup pod pool"):
            pool.setup()


class TestKubernetesPodPoolPodManagement:
    """Test pod acquisition and release functionality."""

    @patch("kubernetes.config.load_kube_config")
    @patch("llm_sandbox.kubernetes_pool.CoreV1Api")
    @patch("llm_sandbox.kubernetes_pool.AppsV1Api")
    def test_get_ready_pods_success(
        self, mock_apps_v1_api: MagicMock, mock_core_v1_api: MagicMock, mock_load_config: MagicMock
    ) -> None:
        """Test getting ready pods from the cluster."""
        mock_core_client = MagicMock()
        mock_core_v1_api.return_value = mock_core_client

        # Mock pod list response
        mock_pod1 = MagicMock()
        mock_pod1.metadata.name = "pod-1"
        mock_pod1.status.phase = "Running"
        mock_condition = MagicMock()
        mock_condition.type = "Ready"
        mock_condition.status = "True"
        mock_pod1.status.conditions = [mock_condition]

        mock_pod2 = MagicMock()
        mock_pod2.metadata.name = "pod-2"
        mock_pod2.status.phase = "Pending"

        mock_pods = MagicMock()
        mock_pods.items = [mock_pod1, mock_pod2]
        mock_core_client.list_namespaced_pod.return_value = mock_pods

        pool = KubernetesPodPool(deployment_name="test-pool")
        ready_pods = pool._get_ready_pods()

        assert ready_pods == ["pod-1"]
        mock_core_client.list_namespaced_pod.assert_called_once_with(
            namespace="default", label_selector="app=llm-sandbox-pool,pool=test-pool"
        )

    @patch("kubernetes.config.load_kube_config")
    @patch("llm_sandbox.kubernetes_pool.CoreV1Api")
    @patch("llm_sandbox.kubernetes_pool.AppsV1Api")
    def test_acquire_pod_success(
        self, mock_apps_v1_api: MagicMock, mock_core_v1_api: MagicMock, mock_load_config: MagicMock
    ) -> None:
        """Test successful pod acquisition."""
        pool = KubernetesPodPool(verbose=True)

        with patch.object(pool, "_get_ready_pods", return_value=["pod-1", "pod-2"]):
            pod_name = pool.acquire_pod()

        assert pod_name == "pod-1"

    @patch("kubernetes.config.load_kube_config")
    @patch("llm_sandbox.kubernetes_pool.CoreV1Api")
    @patch("llm_sandbox.kubernetes_pool.AppsV1Api")
    def test_acquire_pod_timeout(
        self, mock_apps_v1_api: MagicMock, mock_core_v1_api: MagicMock, mock_load_config: MagicMock
    ) -> None:
        """Test pod acquisition timeout."""
        pool = KubernetesPodPool(acquisition_timeout=1)

        with (
            patch.object(pool, "_get_ready_pods", return_value=[]),
            patch("time.sleep"),
            pytest.raises(ContainerError, match="No pod available after 1 seconds"),
        ):
            pool.acquire_pod()

    @patch("kubernetes.config.load_kube_config")
    @patch("llm_sandbox.kubernetes_pool.CoreV1Api")
    @patch("llm_sandbox.kubernetes_pool.AppsV1Api")
    def test_release_pod_success(
        self, mock_apps_v1_api: MagicMock, mock_core_v1_api: MagicMock, mock_load_config: MagicMock
    ) -> None:
        """Test successful pod release."""
        mock_core_client = MagicMock()
        mock_core_v1_api.return_value = mock_core_client

        pool = KubernetesPodPool(verbose=True)
        pool.release_pod("test-pod")

        mock_core_client.delete_namespaced_pod.assert_called_once()

    @patch("kubernetes.config.load_kube_config")
    @patch("llm_sandbox.kubernetes_pool.CoreV1Api")
    @patch("llm_sandbox.kubernetes_pool.AppsV1Api")
    def test_release_pod_not_found(
        self, mock_apps_v1_api: MagicMock, mock_core_v1_api: MagicMock, mock_load_config: MagicMock
    ) -> None:
        """Test pod release when pod not found (should not raise)."""
        mock_core_client = MagicMock()
        mock_core_v1_api.return_value = mock_core_client
        mock_core_client.delete_namespaced_pod.side_effect = ApiException(status=404)

        pool = KubernetesPodPool()
        # Should not raise exception
        pool.release_pod("missing-pod")


class TestKubernetesPodPoolSessionManagement:
    """Test session context manager functionality."""

    @patch("kubernetes.config.load_kube_config")
    @patch("llm_sandbox.kubernetes_pool.CoreV1Api")
    @patch("llm_sandbox.kubernetes_pool.AppsV1Api")
    @patch("llm_sandbox.kubernetes_pool.create_session")
    def test_get_session_success(
        self,
        mock_create_session: MagicMock,
        mock_apps_v1_api: MagicMock,
        mock_core_v1_api: MagicMock,
        mock_load_config: MagicMock,
    ) -> None:
        """Test successful session acquisition and release."""
        mock_core_client = MagicMock()
        mock_core_v1_api.return_value = mock_core_client

        mock_session = MagicMock()
        mock_create_session.return_value.__enter__.return_value = mock_session

        pool = KubernetesPodPool(namespace="test-ns", lang="python")

        with (
            patch.object(pool, "acquire_pod", return_value="test-pod") as mock_acquire,
            patch.object(pool, "release_pod") as mock_release,
        ):
            with pool.get_session(custom_arg="value") as session:
                assert session == mock_session

            mock_acquire.assert_called_once()
            mock_release.assert_called_once_with("test-pod")

            # Verify session configuration
            expected_config = {
                "backend": SandboxBackend.KUBERNETES,
                "client": mock_core_client,
                "kube_namespace": "test-ns",
                "container_id": "test-pod",
                "lang": "python",
                "verbose": False,
                "custom_arg": "value",
            }
            mock_create_session.assert_called_once_with(**expected_config)

    @patch("kubernetes.config.load_kube_config")
    @patch("llm_sandbox.kubernetes_pool.CoreV1Api")
    @patch("llm_sandbox.kubernetes_pool.AppsV1Api")
    @patch("llm_sandbox.kubernetes_pool.create_session")
    def test_get_session_with_exception(
        self,
        mock_create_session: MagicMock,
        mock_apps_v1_api: MagicMock,
        mock_core_v1_api: MagicMock,
        mock_load_config: MagicMock,
    ) -> None:
        """Test session release even when exception occurs."""
        mock_create_session.return_value.__enter__.side_effect = Exception("Session failed")

        pool = KubernetesPodPool()

        with (
            patch.object(pool, "acquire_pod", return_value="test-pod"),
            patch.object(pool, "release_pod") as mock_release,
            pytest.raises(Exception, match="Session failed"),
        ):
            with pool.get_session():
                pass

        # Pod should still be released even after exception
        mock_release.assert_called_once_with("test-pod")


class TestKubernetesPodPoolScaling:
    """Test pool scaling functionality."""

    @patch("kubernetes.config.load_kube_config")
    @patch("llm_sandbox.kubernetes_pool.CoreV1Api")
    @patch("llm_sandbox.kubernetes_pool.AppsV1Api")
    def test_scale_success(
        self, mock_apps_v1_api: MagicMock, mock_core_v1_api: MagicMock, mock_load_config: MagicMock
    ) -> None:
        """Test successful pool scaling."""
        mock_apps_client = MagicMock()
        mock_apps_v1_api.return_value = mock_apps_client

        mock_deployment = MagicMock()
        mock_apps_client.read_namespaced_deployment.return_value = mock_deployment

        pool = KubernetesPodPool(deployment_name="test-pool", verbose=True)
        pool.scale(10)

        assert mock_deployment.spec.replicas == 10
        assert pool.pool_size == 10
        mock_apps_client.patch_namespaced_deployment.assert_called_once()

    @patch("kubernetes.config.load_kube_config")
    @patch("llm_sandbox.kubernetes_pool.CoreV1Api")
    @patch("llm_sandbox.kubernetes_pool.AppsV1Api")
    def test_scale_api_error(
        self, mock_apps_v1_api: MagicMock, mock_core_v1_api: MagicMock, mock_load_config: MagicMock
    ) -> None:
        """Test scaling with API error."""
        mock_apps_client = MagicMock()
        mock_apps_v1_api.return_value = mock_apps_client
        mock_apps_client.read_namespaced_deployment.side_effect = ApiException(status=500)

        pool = KubernetesPodPool()

        with pytest.raises(ContainerError, match="Failed to scale pool"):
            pool.scale(10)


class TestKubernetesPodPoolStatus:
    """Test pool status monitoring."""

    @patch("kubernetes.config.load_kube_config")
    @patch("llm_sandbox.kubernetes_pool.CoreV1Api")
    @patch("llm_sandbox.kubernetes_pool.AppsV1Api")
    def test_get_pool_status_success(
        self, mock_apps_v1_api: MagicMock, mock_core_v1_api: MagicMock, mock_load_config: MagicMock
    ) -> None:
        """Test successful pool status retrieval."""
        mock_apps_client = MagicMock()
        mock_apps_v1_api.return_value = mock_apps_client

        mock_deployment = MagicMock()
        mock_deployment.spec.replicas = 5
        mock_deployment.status.ready_replicas = 4
        mock_deployment.status.available_replicas = 4
        mock_apps_client.read_namespaced_deployment.return_value = mock_deployment

        pool = KubernetesPodPool(deployment_name="test-pool", namespace="test-ns")

        with patch.object(pool, "_get_ready_pods", return_value=["pod-1", "pod-2", "pod-3"]):
            status = pool.get_pool_status()

        expected_status = {
            "deployment_name": "test-pool",
            "namespace": "test-ns",
            "desired_replicas": 5,
            "ready_replicas": 4,
            "available_replicas": 4,
            "ready_pods": 3,
            "ready_pod_names": ["pod-1", "pod-2", "pod-3"],
        }

        assert status == expected_status

    @patch("kubernetes.config.load_kube_config")
    @patch("llm_sandbox.kubernetes_pool.CoreV1Api")
    @patch("llm_sandbox.kubernetes_pool.AppsV1Api")
    def test_get_pool_status_error(
        self, mock_apps_v1_api: MagicMock, mock_core_v1_api: MagicMock, mock_load_config: MagicMock
    ) -> None:
        """Test pool status with API error."""
        mock_apps_client = MagicMock()
        mock_apps_v1_api.return_value = mock_apps_client
        mock_apps_client.read_namespaced_deployment.side_effect = ApiException(status=500)

        pool = KubernetesPodPool()
        status = pool.get_pool_status()

        assert "error" in status


class TestKubernetesPodPoolHealthCheck:
    """Test health check functionality."""

    @patch("kubernetes.config.load_kube_config")
    @patch("llm_sandbox.kubernetes_pool.CoreV1Api")
    @patch("llm_sandbox.kubernetes_pool.AppsV1Api")
    def test_health_check_warning(
        self, mock_apps_v1_api: MagicMock, mock_core_v1_api: MagicMock, mock_load_config: MagicMock
    ) -> None:
        """Test health check warning when pods are below threshold."""
        pool = KubernetesPodPool(pool_size=10)

        # Mock low ready pod count (should trigger warning at 80% threshold)
        mock_status = {"ready_pods": 7}  # 70% of 10, below 80% threshold

        with (
            patch.object(pool, "get_pool_status", return_value=mock_status),
            patch("llm_sandbox.kubernetes_pool.logger.warning") as mock_warning,
        ):
            pool._health_check_loop()
            # Stop the health check immediately
            pool._stop_health_check.set()

            mock_warning.assert_called_with("Pool health warning: only 7/10 pods ready")

    @patch("kubernetes.config.load_kube_config")
    @patch("llm_sandbox.kubernetes_pool.CoreV1Api")
    @patch("llm_sandbox.kubernetes_pool.AppsV1Api")
    def test_start_health_check(
        self, mock_apps_v1_api: MagicMock, mock_core_v1_api: MagicMock, mock_load_config: MagicMock
    ) -> None:
        """Test starting health check thread."""
        pool = KubernetesPodPool()

        with patch("threading.Thread") as mock_thread:
            mock_thread_instance = MagicMock()
            mock_thread.return_value = mock_thread_instance

            pool._start_health_check()

            mock_thread.assert_called_once()
            mock_thread_instance.start.assert_called_once()


class TestKubernetesPodPoolTeardown:
    """Test pool teardown functionality."""

    @patch("kubernetes.config.load_kube_config")
    @patch("llm_sandbox.kubernetes_pool.CoreV1Api")
    @patch("llm_sandbox.kubernetes_pool.AppsV1Api")
    def test_teardown_success(
        self, mock_apps_v1_api: MagicMock, mock_core_v1_api: MagicMock, mock_load_config: MagicMock
    ) -> None:
        """Test successful pool teardown."""
        mock_apps_client = MagicMock()
        mock_apps_v1_api.return_value = mock_apps_client

        pool = KubernetesPodPool(deployment_name="test-pool", verbose=True)

        # Mock health check thread
        mock_thread = MagicMock()
        pool._health_check_thread = mock_thread

        pool.teardown()

        pool._stop_health_check.is_set()  # Should be set
        mock_thread.join.assert_called_once_with(timeout=5)
        mock_apps_client.delete_namespaced_deployment.assert_called_once()

    @patch("kubernetes.config.load_kube_config")
    @patch("llm_sandbox.kubernetes_pool.CoreV1Api")
    @patch("llm_sandbox.kubernetes_pool.AppsV1Api")
    def test_teardown_deployment_not_found(
        self, mock_apps_v1_api: MagicMock, mock_core_v1_api: MagicMock, mock_load_config: MagicMock
    ) -> None:
        """Test teardown when deployment not found (should not raise)."""
        mock_apps_client = MagicMock()
        mock_apps_v1_api.return_value = mock_apps_client
        mock_apps_client.delete_namespaced_deployment.side_effect = ApiException(status=404)

        pool = KubernetesPodPool()
        # Should not raise exception
        pool.teardown()


class TestKubernetesPodPoolContextManager:
    """Test context manager functionality."""

    @patch("kubernetes.config.load_kube_config")
    @patch("llm_sandbox.kubernetes_pool.CoreV1Api")
    @patch("llm_sandbox.kubernetes_pool.AppsV1Api")
    def test_context_manager_success(
        self, mock_apps_v1_api: MagicMock, mock_core_v1_api: MagicMock, mock_load_config: MagicMock
    ) -> None:
        """Test using pool as context manager."""
        pool = KubernetesPodPool()

        with patch.object(pool, "setup") as mock_setup, patch.object(pool, "teardown") as mock_teardown:
            with pool as p:
                assert p == pool
                mock_setup.assert_called_once()

            mock_teardown.assert_called_once()

    @patch("kubernetes.config.load_kube_config")
    @patch("llm_sandbox.kubernetes_pool.CoreV1Api")
    @patch("llm_sandbox.kubernetes_pool.AppsV1Api")
    def test_context_manager_with_exception(
        self, mock_apps_v1_api: MagicMock, mock_core_v1_api: MagicMock, mock_load_config: MagicMock
    ) -> None:
        """Test context manager ensures teardown even with exception."""
        pool = KubernetesPodPool()

        with patch.object(pool, "setup"), patch.object(pool, "teardown") as mock_teardown, pytest.raises(ValueError):
            with pool:
                raise ValueError("Test exception")

            mock_teardown.assert_called_once()


class TestKubernetesPodPoolWaitForReady:
    """Test waiting for ready pods functionality."""

    @patch("kubernetes.config.load_kube_config")
    @patch("llm_sandbox.kubernetes_pool.CoreV1Api")
    @patch("llm_sandbox.kubernetes_pool.AppsV1Api")
    def test_wait_for_ready_pods_success(
        self, mock_apps_v1_api: MagicMock, mock_core_v1_api: MagicMock, mock_load_config: MagicMock
    ) -> None:
        """Test successful wait for ready pods."""
        pool = KubernetesPodPool(pool_size=3, verbose=True)

        # Mock progression: first call returns 1 pod, second returns all 3
        with (
            patch.object(pool, "_get_ready_pods", side_effect=[["pod-1"], ["pod-1", "pod-2", "pod-3"]]),
            patch("time.sleep"),
        ):
            pool._wait_for_ready_pods()
            # Should complete without raising

    @patch("kubernetes.config.load_kube_config")
    @patch("llm_sandbox.kubernetes_pool.CoreV1Api")
    @patch("llm_sandbox.kubernetes_pool.AppsV1Api")
    def test_wait_for_ready_pods_timeout(
        self, mock_apps_v1_api: MagicMock, mock_core_v1_api: MagicMock, mock_load_config: MagicMock
    ) -> None:
        """Test wait for ready pods with timeout (should log warning but not fail)."""
        pool = KubernetesPodPool(pool_size=5, warmup_timeout=1)

        with (
            patch.object(pool, "_get_ready_pods", return_value=["pod-1"]),
            patch("time.sleep"),
            patch("llm_sandbox.kubernetes_pool.logger.warning") as mock_warning,
        ):
            pool._wait_for_ready_pods()

            mock_warning.assert_called_with("Only 1/5 pods ready after 1s")


class TestKubernetesPodPoolThreadSafety:
    """Test thread safety of pod acquisition."""

    @patch("kubernetes.config.load_kube_config")
    @patch("llm_sandbox.kubernetes_pool.CoreV1Api")
    @patch("llm_sandbox.kubernetes_pool.AppsV1Api")
    def test_concurrent_pod_acquisition(
        self, mock_apps_v1_api: MagicMock, mock_core_v1_api: MagicMock, mock_load_config: MagicMock
    ) -> None:
        """Test that pod acquisition is thread-safe."""
        pool = KubernetesPodPool()
        acquired_pods = []

        def acquire_pod_worker() -> None:
            try:
                pod = pool.acquire_pod()
                acquired_pods.append(pod)
            except ContainerError:
                pass  # Expected when no pods available

        # Mock limited pod availability
        with patch.object(pool, "_get_ready_pods", side_effect=[["pod-1"], [], []]):
            # Start multiple threads trying to acquire pods
            threads = []
            for _ in range(3):
                thread = threading.Thread(target=acquire_pod_worker)
                threads.append(thread)
                thread.start()

            # Wait for all threads to complete
            for thread in threads:
                thread.join()

        # Only one thread should have successfully acquired the pod
        assert len(acquired_pods) == 1
        assert acquired_pods[0] == "pod-1"
