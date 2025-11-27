"""Comprehensive tests for PooledSandboxSession and artifact session."""

from unittest.mock import MagicMock, patch

import pytest

from llm_sandbox.const import SandboxBackend, SupportedLanguage
from llm_sandbox.data import ConsoleOutput
from llm_sandbox.pool.config import PoolConfig
from llm_sandbox.pool.session import ArtifactPooledSandboxSession, DuplicateClientError, PooledSandboxSession


class TestDuplicateClientError:
    """Test DuplicateClientError exception."""

    def test_error_message_contains_key_info(self) -> None:
        """Test error message."""
        error = DuplicateClientError()
        assert "Cannot specify 'client' parameter" in str(error)
        assert "pool manager" in str(error)


class TestPooledSandboxSessionInit:
    """Test session initialization."""

    @patch("llm_sandbox.pool.docker_pool.docker.from_env")
    def test_backend_inference_docker(self, mock_docker_env: MagicMock) -> None:
        """Test Docker backend inference."""
        from llm_sandbox.pool.docker_pool import DockerPoolManager

        mock_docker_env.return_value = MagicMock()
        pool = DockerPoolManager(config=PoolConfig(enable_prewarming=False), lang=SupportedLanguage.PYTHON)

        try:
            session = PooledSandboxSession(pool_manager=pool)
            assert session.backend == SandboxBackend.DOCKER
        finally:
            pool.close()

    @patch("kubernetes.config.load_kube_config")
    @patch("llm_sandbox.pool.kubernetes_pool.CoreV1Api")
    def test_backend_inference_kubernetes(self, mock_core_api: MagicMock, mock_load_config: MagicMock) -> None:
        """Test Kubernetes backend inference."""
        from llm_sandbox.pool.kubernetes_pool import KubernetesPoolManager

        mock_core_api.return_value = MagicMock()
        pool = KubernetesPoolManager(config=PoolConfig(enable_prewarming=False), lang=SupportedLanguage.PYTHON)

        try:
            session = PooledSandboxSession(pool_manager=pool)
            assert session.backend == SandboxBackend.KUBERNETES
        finally:
            pool.close()

    @patch("llm_sandbox.pool.podman_pool.PodmanClient")
    def test_backend_inference_podman(self, mock_podman_client: MagicMock) -> None:
        """Test Podman backend inference."""
        from llm_sandbox.pool.podman_pool import PodmanPoolManager

        mock_podman_client.return_value = MagicMock()
        pool = PodmanPoolManager(config=PoolConfig(enable_prewarming=False), lang=SupportedLanguage.PYTHON)

        try:
            session = PooledSandboxSession(pool_manager=pool)
            assert session.backend == SandboxBackend.PODMAN
        finally:
            pool.close()


class TestPooledSandboxSessionLifecycle:
    """Test session open/close lifecycle."""

    @patch("llm_sandbox.pool.docker_pool.docker.from_env")
    @patch("llm_sandbox.docker.SandboxDockerSession")
    def test_open_acquires_container(self, mock_session_class: MagicMock, mock_docker_env: MagicMock) -> None:
        """Test opening acquires from pool."""
        from llm_sandbox.pool.base import ContainerState
        from llm_sandbox.pool.docker_pool import DockerPoolManager

        mock_client = MagicMock()
        mock_docker_env.return_value = mock_client

        mock_container = MagicMock()
        mock_container.id = "test-123"

        mock_session_instance = MagicMock()
        mock_session_instance.container = mock_container
        mock_session_class.return_value = mock_session_instance

        pool = DockerPoolManager(config=PoolConfig(enable_prewarming=False), lang=SupportedLanguage.PYTHON)

        try:
            session = PooledSandboxSession(pool_manager=pool)
            session.open()

            assert session._pooled_container is not None
            assert session._pooled_container.state == ContainerState.BUSY
            assert session._backend_session is not None
        finally:
            if hasattr(session, "_pooled_container") and session._pooled_container:
                pool.release(session._pooled_container)
            pool.close()

    @patch("llm_sandbox.pool.docker_pool.docker.from_env")
    @patch("llm_sandbox.docker.SandboxDockerSession")
    def test_close_releases_container(self, mock_session_class: MagicMock, mock_docker_env: MagicMock) -> None:
        """Test closing releases to pool."""
        from llm_sandbox.pool.docker_pool import DockerPoolManager

        mock_client = MagicMock()
        mock_docker_env.return_value = mock_client

        mock_container = MagicMock()
        mock_container.id = "test-123"
        mock_container.status = "running"
        mock_container.exec_run.return_value = (0, b"ok")
        mock_container.reload = MagicMock()

        mock_session_instance = MagicMock()
        mock_session_instance.container = mock_container
        mock_session_class.return_value = mock_session_instance

        pool = DockerPoolManager(config=PoolConfig(enable_prewarming=False), lang=SupportedLanguage.PYTHON)

        try:
            session = PooledSandboxSession(pool_manager=pool)
            session.open()
            session.close()

            assert session._pooled_container is None
            assert session._backend_session is None
        finally:
            pool.close()

    @patch("llm_sandbox.pool.docker_pool.docker.from_env")
    @patch("llm_sandbox.docker.SandboxDockerSession")
    def test_context_manager(self, mock_session_class: MagicMock, mock_docker_env: MagicMock) -> None:
        """Test context manager usage."""
        from llm_sandbox.pool.docker_pool import DockerPoolManager

        mock_client = MagicMock()
        mock_docker_env.return_value = mock_client

        mock_container = MagicMock()
        mock_container.id = "test-123"
        mock_container.status = "running"
        mock_container.exec_run.return_value = (0, b"ok")
        mock_container.reload = MagicMock()

        mock_session_instance = MagicMock()
        mock_session_instance.container = mock_container
        mock_session_class.return_value = mock_session_instance

        pool = DockerPoolManager(config=PoolConfig(enable_prewarming=False), lang=SupportedLanguage.PYTHON)

        try:
            with PooledSandboxSession(pool_manager=pool) as session:
                assert session._backend_session is not None
        finally:
            pool.close()


class TestPooledSandboxSessionExecution:
    """Test code execution."""

    @patch("llm_sandbox.pool.docker_pool.docker.from_env")
    @patch("llm_sandbox.docker.SandboxDockerSession")
    def test_run_delegates(self, mock_session_class: MagicMock, mock_docker_env: MagicMock) -> None:
        """Test run delegates to backend."""
        from llm_sandbox.pool.docker_pool import DockerPoolManager

        mock_docker_env.return_value = MagicMock()

        mock_container = MagicMock()
        mock_container.id = "test-123"

        output = ConsoleOutput()
        mock_session_instance = MagicMock()
        mock_session_instance.container = mock_container
        mock_session_instance.run.return_value = output
        mock_session_class.return_value = mock_session_instance

        pool = DockerPoolManager(config=PoolConfig(enable_prewarming=False), lang=SupportedLanguage.PYTHON)

        try:
            session = PooledSandboxSession(pool_manager=pool)
            session.open()
            result = session.run("print('test')")

            assert result == output
            mock_session_instance.run.assert_called_once()
        finally:
            if hasattr(session, "_pooled_container") and session._pooled_container:
                pool.release(session._pooled_container)
            pool.close()

    @patch("llm_sandbox.pool.docker_pool.docker.from_env")
    @patch("llm_sandbox.docker.SandboxDockerSession")
    def test_execute_command_delegates(self, mock_session_class: MagicMock, mock_docker_env: MagicMock) -> None:
        """Test execute_command delegates."""
        from llm_sandbox.pool.docker_pool import DockerPoolManager

        mock_docker_env.return_value = MagicMock()

        mock_container = MagicMock()
        output = ConsoleOutput()
        mock_session_instance = MagicMock()
        mock_session_instance.container = mock_container
        mock_session_instance.execute_command.return_value = output
        mock_session_class.return_value = mock_session_instance

        pool = DockerPoolManager(config=PoolConfig(enable_prewarming=False), lang=SupportedLanguage.PYTHON)

        try:
            session = PooledSandboxSession(pool_manager=pool)
            session.open()
            result = session.execute_command("ls")

            assert result == output
        finally:
            if hasattr(session, "_pooled_container") and session._pooled_container:
                pool.release(session._pooled_container)
            pool.close()

    @patch("llm_sandbox.pool.docker_pool.docker.from_env")
    @patch("llm_sandbox.docker.SandboxDockerSession")
    def test_file_operations_delegate(self, mock_session_class: MagicMock, mock_docker_env: MagicMock) -> None:
        """Test file copy operations."""
        from llm_sandbox.pool.docker_pool import DockerPoolManager

        mock_docker_env.return_value = MagicMock()

        mock_container = MagicMock()
        mock_session_instance = MagicMock()
        mock_session_instance.container = mock_container
        mock_session_class.return_value = mock_session_instance

        pool = DockerPoolManager(config=PoolConfig(enable_prewarming=False), lang=SupportedLanguage.PYTHON)

        try:
            session = PooledSandboxSession(pool_manager=pool)
            session.open()

            session.copy_to_runtime("/local", "/remote")
            session.copy_from_runtime("/remote", "/local")

            mock_session_instance.copy_to_runtime.assert_called_once()
            mock_session_instance.copy_from_runtime.assert_called_once()
        finally:
            if hasattr(session, "_pooled_container") and session._pooled_container:
                pool.release(session._pooled_container)
            pool.close()

    @patch("llm_sandbox.pool.docker_pool.docker.from_env")
    def test_run_without_open_raises(self, mock_docker_env: MagicMock) -> None:
        """Test run without opening raises error."""
        from llm_sandbox.pool.docker_pool import DockerPoolManager

        mock_docker_env.return_value = MagicMock()
        pool = DockerPoolManager(config=PoolConfig(enable_prewarming=False), lang=SupportedLanguage.PYTHON)

        try:
            session = PooledSandboxSession(pool_manager=pool)

            with pytest.raises(RuntimeError, match="Session not open"):
                session.run("print('test')")
        finally:
            pool.close()


class TestArtifactPooledSandboxSession:
    """Test artifact session."""

    @patch("llm_sandbox.pool.docker_pool.docker.from_env")
    def test_artifact_session_init(self, mock_docker_env: MagicMock) -> None:
        """Test artifact session initialization."""
        from llm_sandbox.pool.docker_pool import DockerPoolManager

        mock_docker_env.return_value = MagicMock()
        pool = DockerPoolManager(config=PoolConfig(enable_prewarming=False), lang=SupportedLanguage.PYTHON)

        try:
            session = ArtifactPooledSandboxSession(pool_manager=pool, enable_plotting=True)
            assert session.enable_plotting is True
            assert session._session is not None
        finally:
            pool.close()

    @patch("llm_sandbox.pool.docker_pool.docker.from_env")
    @patch("llm_sandbox.docker.SandboxDockerSession")
    def test_run_with_plotting(self, mock_session_class: MagicMock, mock_docker_env: MagicMock) -> None:
        """Test run with plotting enabled."""
        from llm_sandbox.pool.docker_pool import DockerPoolManager

        mock_docker_env.return_value = MagicMock()

        mock_container = MagicMock()
        mock_session_instance = MagicMock()
        mock_session_instance.container = mock_container
        # Mock language handler support for plotting
        mock_session_instance.language_handler.is_support_plot_detection = True

        # Mock run_with_artifacts return
        mock_result = MagicMock()
        mock_result.exit_code = 0
        mock_result.stdout = "output"
        mock_result.stderr = ""
        mock_plots = [MagicMock()]
        mock_session_instance.language_handler.run_with_artifacts.return_value = (mock_result, mock_plots)

        mock_session_class.return_value = mock_session_instance

        pool = DockerPoolManager(config=PoolConfig(enable_prewarming=False), lang=SupportedLanguage.PYTHON)
        pool.acquire = MagicMock(return_value=MagicMock(container_id="test-container"))  # type: ignore[method-assign]

        try:
            session = ArtifactPooledSandboxSession(pool_manager=pool, enable_plotting=True)
            session.open()

            result = session.run("import matplotlib.pyplot as plt")

            assert result.exit_code == 0
            assert result.stdout == "output"
            assert result.plots == mock_plots
            mock_session_instance.language_handler.run_with_artifacts.assert_called_once()
        finally:
            pool.close()

    @patch("llm_sandbox.pool.docker_pool.docker.from_env")
    @patch("llm_sandbox.docker.SandboxDockerSession")
    def test_run_plotting_not_supported(self, mock_session_class: MagicMock, mock_docker_env: MagicMock) -> None:
        """Test run raises error if plotting not supported."""
        from llm_sandbox.exceptions import LanguageNotSupportPlotError
        from llm_sandbox.pool.docker_pool import DockerPoolManager

        mock_docker_env.return_value = MagicMock()
        mock_session_instance = MagicMock()
        mock_session_instance.language_handler.is_support_plot_detection = False
        mock_session_class.return_value = mock_session_instance

        pool = DockerPoolManager(config=PoolConfig(enable_prewarming=False), lang=SupportedLanguage.PYTHON)
        pool.acquire = MagicMock(return_value=MagicMock(container_id="test-container"))  # type: ignore[method-assign]  # type: ignore[method-assign]

        try:
            session = ArtifactPooledSandboxSession(pool_manager=pool, enable_plotting=True)
            session.open()

            with pytest.raises(LanguageNotSupportPlotError):
                session.run("code")
        finally:
            pool.close()

    @patch("llm_sandbox.pool.docker_pool.docker.from_env")
    @patch("llm_sandbox.docker.SandboxDockerSession")
    def test_clear_plots(self, mock_session_class: MagicMock, mock_docker_env: MagicMock) -> None:
        """Test clearing plots."""
        from llm_sandbox.pool.docker_pool import DockerPoolManager

        mock_docker_env.return_value = MagicMock()
        mock_session_instance = MagicMock()
        mock_session_class.return_value = mock_session_instance

        pool = DockerPoolManager(config=PoolConfig(enable_prewarming=False), lang=SupportedLanguage.PYTHON)
        pool.acquire = MagicMock(return_value=MagicMock(container_id="test-container"))  # type: ignore[method-assign]  # type: ignore[method-assign]

        try:
            session = ArtifactPooledSandboxSession(pool_manager=pool, enable_plotting=True)
            session.open()

            session.clear_plots()

            # Verify command execution to clear plots
            mock_session_instance.execute_command.assert_called()
            call_args = mock_session_instance.execute_command.call_args[0][0]
            assert "rm -rf /tmp/sandbox_plots" in call_args
        finally:
            pool.close()

    @patch("llm_sandbox.pool.docker_pool.docker.from_env")
    @patch("llm_sandbox.docker.SandboxDockerSession")
    def test_run_with_timeout_override(self, mock_session_class: MagicMock, mock_docker_env: MagicMock) -> None:
        """Test run with explicit timeout."""
        from llm_sandbox.pool.docker_pool import DockerPoolManager

        mock_docker_env.return_value = MagicMock()
        mock_session_instance = MagicMock()
        mock_session_instance.language_handler.is_support_plot_detection = True
        mock_session_instance.language_handler.run_with_artifacts.return_value = (MagicMock(), [])
        mock_session_class.return_value = mock_session_instance

        pool = DockerPoolManager(config=PoolConfig(enable_prewarming=False), lang=SupportedLanguage.PYTHON)
        pool.acquire = MagicMock(return_value=MagicMock(container_id="test-container"))  # type: ignore[method-assign]

        try:
            session = ArtifactPooledSandboxSession(pool_manager=pool, enable_plotting=True)
            session.open()

            session.run("code", timeout=30)

            call_kwargs = mock_session_instance.language_handler.run_with_artifacts.call_args[1]
            assert call_kwargs["timeout"] == 30
        finally:
            pool.close()


class TestPooledSessionAttributeDelegation:
    """Test attribute delegation to backend session."""

    @patch("llm_sandbox.pool.docker_pool.docker.from_env")
    @patch("llm_sandbox.docker.SandboxDockerSession")
    def test_getattr_delegates_to_backend(self, mock_session_class: MagicMock, mock_docker_env: MagicMock) -> None:
        """Test __getattr__ delegates to backend session."""
        from llm_sandbox.pool.docker_pool import DockerPoolManager

        mock_docker_env.return_value = MagicMock()

        mock_container = MagicMock()
        mock_session_instance = MagicMock()
        mock_session_instance.container = mock_container
        mock_session_instance.some_method = MagicMock(return_value="test_value")
        mock_session_class.return_value = mock_session_instance

        pool = DockerPoolManager(config=PoolConfig(enable_prewarming=False), lang=SupportedLanguage.PYTHON)

        try:
            session = PooledSandboxSession(pool_manager=pool)
            session.open()

            # Access method through delegation
            result = session.some_method()
            assert result == "test_value"
            mock_session_instance.some_method.assert_called_once()
        finally:
            if hasattr(session, "_pooled_container") and session._pooled_container:
                pool.release(session._pooled_container)
            pool.close()

    @patch("llm_sandbox.pool.docker_pool.docker.from_env")
    def test_getattr_without_open_raises(self, mock_docker_env: MagicMock) -> None:
        """Test getattr raises error when session not open."""
        from llm_sandbox.pool.docker_pool import DockerPoolManager

        mock_docker_env.return_value = MagicMock()
        pool = DockerPoolManager(config=PoolConfig(enable_prewarming=False), lang=SupportedLanguage.PYTHON)

        try:
            session = PooledSandboxSession(pool_manager=pool)

            with pytest.raises(AttributeError, match="Cannot access.*session not open"):
                _ = session.unknown_attribute
        finally:
            pool.close()

    @patch("llm_sandbox.pool.docker_pool.docker.from_env")
    @patch("llm_sandbox.docker.SandboxDockerSession")
    def test_backend_session_property(self, mock_session_class: MagicMock, mock_docker_env: MagicMock) -> None:
        """Test backend_session property."""
        from llm_sandbox.pool.docker_pool import DockerPoolManager

        mock_docker_env.return_value = MagicMock()

        mock_container = MagicMock()
        mock_session_instance = MagicMock()
        mock_session_instance.container = mock_container
        mock_session_class.return_value = mock_session_instance

        pool = DockerPoolManager(config=PoolConfig(enable_prewarming=False), lang=SupportedLanguage.PYTHON)

        try:
            session = PooledSandboxSession(pool_manager=pool)
            session.open()

            backend = session.backend_session
            assert backend == mock_session_instance
        finally:
            if hasattr(session, "_pooled_container") and session._pooled_container:
                pool.release(session._pooled_container)
            pool.close()

    @patch("llm_sandbox.pool.docker_pool.docker.from_env")
    def test_backend_session_property_not_open_raises(self, mock_docker_env: MagicMock) -> None:
        """Test backend_session property raises when not open."""
        from llm_sandbox.pool.docker_pool import DockerPoolManager

        mock_docker_env.return_value = MagicMock()
        pool = DockerPoolManager(config=PoolConfig(enable_prewarming=False), lang=SupportedLanguage.PYTHON)

        try:
            session = PooledSandboxSession(pool_manager=pool)

            with pytest.raises(RuntimeError, match="Session not open"):
                _ = session.backend_session
        finally:
            pool.close()


class TestPooledSessionBackendCreation:
    """Test backend session creation for different backends."""

    @patch("kubernetes.config.load_kube_config")
    @patch("llm_sandbox.pool.kubernetes_pool.CoreV1Api")
    @patch("llm_sandbox.kubernetes.SandboxKubernetesSession")
    def test_create_kubernetes_backend_session(
        self, mock_k8s_session: MagicMock, mock_core_api: MagicMock, mock_load_config: MagicMock
    ) -> None:
        """Test creating Kubernetes backend session."""
        from llm_sandbox.pool.base import PooledContainer
        from llm_sandbox.pool.kubernetes_pool import KubernetesPoolManager

        mock_core_api.return_value = MagicMock()
        mock_k8s_session.return_value = MagicMock()

        pool = KubernetesPoolManager(
            config=PoolConfig(enable_prewarming=False),
            lang=SupportedLanguage.PYTHON,
            namespace="test-ns",
        )
        # Mock acquire to avoid actual pod creation logic
        pool.acquire = MagicMock(return_value=PooledContainer(container_id="test-pod", container=MagicMock()))  # type: ignore[method-assign]

        try:
            session = PooledSandboxSession(pool_manager=pool, namespace="custom-ns")
            session.open()

            # Verify Kubernetes session was created
            mock_k8s_session.assert_called_once()
            call_kwargs = mock_k8s_session.call_args[1]
            assert call_kwargs["namespace"] == "custom-ns"
            assert call_kwargs["skip_environment_setup"] is True
        finally:
            pool.close()

    @patch("llm_sandbox.pool.podman_pool.PodmanClient")
    @patch("llm_sandbox.podman.SandboxPodmanSession")
    def test_create_podman_backend_session(self, mock_podman_session: MagicMock, mock_podman_client: MagicMock) -> None:
        """Test creating Podman backend session."""
        from llm_sandbox.pool.base import PooledContainer
        from llm_sandbox.pool.podman_pool import PodmanPoolManager

        mock_podman_client.return_value = MagicMock()
        mock_podman_session.return_value = MagicMock()

        pool = PodmanPoolManager(config=PoolConfig(enable_prewarming=False), lang=SupportedLanguage.PYTHON)
        pool.acquire = MagicMock(return_value=PooledContainer(container_id="test-container", container=MagicMock()))  # type: ignore[method-assign]

        try:
            session = PooledSandboxSession(pool_manager=pool)
            session.open()

            # Verify Podman session was created
            mock_podman_session.assert_called_once()
            call_kwargs = mock_podman_session.call_args[1]
            assert call_kwargs["skip_environment_setup"] is True
        finally:
            pool.close()

    def test_infer_backend_failure(self) -> None:
        """Test failure to infer backend from pool manager."""
        pool = MagicMock()
        pool.__class__.__name__ = "UnknownPoolManager"

        # We need to bypass __init__ because it calls _infer_backend_from_pool
        session = PooledSandboxSession.__new__(PooledSandboxSession)
        session._pool_manager = pool

        with pytest.raises(RuntimeError, match="Cannot infer backend"):
            session._infer_backend_from_pool()

    def test_open_no_pool_manager(self) -> None:
        """Test open raises RuntimeError if pool manager is missing."""
        session = PooledSandboxSession.__new__(PooledSandboxSession)
        session._pool_manager = None  # type: ignore[assignment]

        with pytest.raises(RuntimeError, match="Pool manager not initialized"):
            session.open()

    def test_close_idempotent(self) -> None:
        """Test close is idempotent and handles uninitialized state."""
        session = PooledSandboxSession.__new__(PooledSandboxSession)
        session._backend_session = None
        session._pooled_container = None
        session._pool_manager = None

        # Should not raise
        session.close()

    def test_create_backend_session_unsupported(self) -> None:
        """Test _create_backend_session with unsupported backend."""
        session = PooledSandboxSession.__new__(PooledSandboxSession)
        session.backend = "unsupported"  # type: ignore[assignment]
        session._session_kwargs = {}

        with pytest.raises(RuntimeError, match="Unsupported backend"):
            session._create_backend_session("id")

    def test_methods_raise_if_not_open(self) -> None:
        """Test methods raise RuntimeError if session is not open."""
        session = PooledSandboxSession.__new__(PooledSandboxSession)
        session._backend_session = None

        with pytest.raises(RuntimeError, match="Session not open"):
            session.run("code")

        with pytest.raises(RuntimeError, match="Session not open"):
            session.execute_command("cmd")

        with pytest.raises(RuntimeError, match="Session not open"):
            session.copy_to_runtime("src", "dest")

        with pytest.raises(RuntimeError, match="Session not open"):
            session.copy_from_runtime("src", "dest")

        with pytest.raises(RuntimeError, match="Session not open"):
            _ = session.backend_session

        with pytest.raises(AttributeError, match="session not open"):
            _ = session.some_attribute

    @patch("llm_sandbox.pool.kubernetes_pool.CoreV1Api")
    @patch("llm_sandbox.kubernetes.SandboxKubernetesSession")
    def test_kubernetes_duplicate_client_error(self, mock_k8s_session: MagicMock, mock_core_api: MagicMock) -> None:
        """Test DuplicateClientError for Kubernetes backend."""
        from llm_sandbox.pool.base import PooledContainer
        from llm_sandbox.pool.kubernetes_pool import KubernetesPoolManager

        mock_core_api.return_value = MagicMock()
        pool = KubernetesPoolManager(config=PoolConfig(enable_prewarming=False), lang=SupportedLanguage.PYTHON)
        pool.acquire = MagicMock(return_value=PooledContainer(container_id="test", container=MagicMock()))  # type: ignore[method-assign]

        try:
            session = PooledSandboxSession(pool_manager=pool, client=MagicMock())
            with pytest.raises(DuplicateClientError):
                session.open()
        finally:
            pool.close()

    @patch("llm_sandbox.pool.podman_pool.PodmanClient")
    @patch("llm_sandbox.podman.SandboxPodmanSession")
    def test_podman_duplicate_client_error(self, mock_podman_session: MagicMock, mock_podman_client: MagicMock) -> None:
        """Test DuplicateClientError for Podman backend."""
        from llm_sandbox.pool.base import PooledContainer
        from llm_sandbox.pool.podman_pool import PodmanPoolManager

        mock_podman_client.return_value = MagicMock()
        pool = PodmanPoolManager(config=PoolConfig(enable_prewarming=False), lang=SupportedLanguage.PYTHON)
        pool.acquire = MagicMock(return_value=PooledContainer(container_id="test", container=MagicMock()))  # type: ignore[method-assign]

        try:
            session = PooledSandboxSession(pool_manager=pool, client=MagicMock())
            with pytest.raises(DuplicateClientError):
                session.open()
        finally:
            pool.close()

    def test_artifact_session_context_manager(self) -> None:
        """Test ArtifactPooledSandboxSession context manager."""
        mock_pool = MagicMock()
        mock_pool.__class__.__name__ = "DockerPoolManager"

        with patch("llm_sandbox.pool.session.PooledSandboxSession.open") as mock_open:
            with patch("llm_sandbox.pool.session.PooledSandboxSession.close") as mock_close:
                with ArtifactPooledSandboxSession(pool_manager=mock_pool):
                    pass

                mock_open.assert_called_once()
                mock_close.assert_called_once()

    def test_artifact_session_clear_plots_disabled(self) -> None:
        """Test clear_plots does nothing when disabled."""
        mock_pool = MagicMock()
        mock_pool.__class__.__name__ = "DockerPoolManager"

        session = ArtifactPooledSandboxSession(pool_manager=mock_pool, enable_plotting=False)
        # Should not raise or call anything
        session.clear_plots()


class TestPooledSessionParameters:
    """Test session parameter handling."""

    @patch("llm_sandbox.pool.docker_pool.docker.from_env")
    @patch("llm_sandbox.docker.SandboxDockerSession")
    def test_custom_parameters_passed_to_backend(
        self, mock_session_class: MagicMock, mock_docker_env: MagicMock
    ) -> None:
        """Test custom parameters are passed to backend session."""
        from llm_sandbox.pool.docker_pool import DockerPoolManager

        mock_docker_env.return_value = MagicMock()

        mock_container = MagicMock()
        mock_session_instance = MagicMock()
        mock_session_instance.container = mock_container
        mock_session_class.return_value = mock_session_instance

        pool = DockerPoolManager(config=PoolConfig(enable_prewarming=False), lang=SupportedLanguage.PYTHON)

        try:
            session = PooledSandboxSession(
                pool_manager=pool,
                verbose=True,
                stream=True,
                workdir="/custom",
                execution_timeout=120.0,
            )
            session.open()

            # Verify parameters were passed
            call_kwargs = mock_session_class.call_args[1]
            assert call_kwargs["verbose"] is True
            assert call_kwargs["stream"] is True
            assert call_kwargs["workdir"] == "/custom"
            assert call_kwargs["execution_timeout"] == 120.0
        finally:
            if hasattr(session, "_pooled_container") and session._pooled_container:
                pool.release(session._pooled_container)
            pool.close()

    @patch("llm_sandbox.pool.docker_pool.docker.from_env")
    @patch("llm_sandbox.docker.SandboxDockerSession")
    def test_lang_and_image_from_kwargs(self, mock_session_class: MagicMock, mock_docker_env: MagicMock) -> None:
        """Test lang and image can be overridden via kwargs."""
        from llm_sandbox.pool.docker_pool import DockerPoolManager

        mock_docker_env.return_value = MagicMock()

        mock_container = MagicMock()
        mock_session_instance = MagicMock()
        mock_session_instance.container = mock_container
        mock_session_class.return_value = mock_session_instance

        pool = DockerPoolManager(
            config=PoolConfig(enable_prewarming=False),
            lang=SupportedLanguage.PYTHON,
            image="default:image",
        )

        try:
            session = PooledSandboxSession(
                pool_manager=pool,
                lang="javascript",
                image="custom:image",
            )
            session.open()

            # Verify overridden values were used
            call_kwargs = mock_session_class.call_args[1]
            assert call_kwargs["lang"] == "javascript"
            assert call_kwargs["image"] == "custom:image"
        finally:
            if hasattr(session, "_pooled_container") and session._pooled_container:
                pool.release(session._pooled_container)
            pool.close()
