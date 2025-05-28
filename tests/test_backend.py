# ruff: noqa: SLF001, PLR2004, ARG002, FBT003

"""Integration tests for all sandbox backends."""

from unittest.mock import MagicMock, patch

import pytest

from llm_sandbox.const import SandboxBackend, SupportedLanguage
from llm_sandbox.exceptions import MissingDependencyError, UnsupportedBackendError
from llm_sandbox.security import SecurityPolicy
from llm_sandbox.session import create_session


class TestBackendSelection:
    """Test backend selection and creation functionality."""

    @patch("llm_sandbox.session.find_spec")
    @patch("llm_sandbox.docker.SandboxDockerSession")
    def test_create_docker_session(self, mock_docker_session: MagicMock, mock_find_spec: MagicMock) -> None:
        """Test creating Docker session."""
        mock_find_spec.return_value = MagicMock()  # Docker available
        mock_session_instance = MagicMock()
        mock_docker_session.return_value = mock_session_instance

        session = create_session(backend=SandboxBackend.DOCKER, lang="python", verbose=True, image="custom:latest")

        assert session == mock_session_instance
        mock_docker_session.assert_called_once_with(lang="python", verbose=True, image="custom:latest")

    @patch("llm_sandbox.session.find_spec")
    @patch("llm_sandbox.kubernetes.SandboxKubernetesSession")
    def test_create_kubernetes_session(self, mock_k8s_session: MagicMock, mock_find_spec: MagicMock) -> None:
        """Test creating Kubernetes session."""
        mock_find_spec.return_value = MagicMock()  # Kubernetes available
        mock_session_instance = MagicMock()
        mock_k8s_session.return_value = mock_session_instance

        session = create_session(
            backend=SandboxBackend.KUBERNETES, lang="java", verbose=False, kube_namespace="test-ns"
        )

        assert session == mock_session_instance
        mock_k8s_session.assert_called_once_with(lang="java", verbose=False, kube_namespace="test-ns")

    @patch("llm_sandbox.session.find_spec")
    @patch("llm_sandbox.podman.SandboxPodmanSession")
    def test_create_podman_session(self, mock_podman_session: MagicMock, mock_find_spec: MagicMock) -> None:
        """Test creating Podman session."""
        mock_find_spec.return_value = MagicMock()  # Podman available
        mock_session_instance = MagicMock()
        mock_podman_session.return_value = mock_session_instance

        custom_client = MagicMock()
        session = create_session(backend=SandboxBackend.PODMAN, lang="javascript", client=custom_client)

        assert session == mock_session_instance
        mock_podman_session.assert_called_once_with(lang="javascript", client=custom_client)

    @patch("llm_sandbox.session.find_spec")
    @patch("llm_sandbox.micromamba.MicromambaSession")
    def test_create_micromamba_session(self, mock_micromamba_session: MagicMock, mock_find_spec: MagicMock) -> None:
        """Test creating Micromamba session."""
        mock_find_spec.return_value = MagicMock()  # Docker available (required for Micromamba)
        mock_session_instance = MagicMock()
        mock_micromamba_session.return_value = mock_session_instance

        session = create_session(backend=SandboxBackend.MICROMAMBA, lang="python", environment="data_science")

        assert session == mock_session_instance
        mock_micromamba_session.assert_called_once_with(lang="python", environment="data_science")

    def test_create_session_unsupported_backend(self) -> None:
        """Test creating session with unsupported backend."""
        with pytest.raises(UnsupportedBackendError):
            create_session(backend="invalid_backend")  # type: ignore[arg-type]

    @patch("llm_sandbox.session.find_spec")
    def test_create_session_missing_docker_dependency(self, mock_find_spec: MagicMock) -> None:
        """Test creating session when Docker dependency is missing."""
        mock_find_spec.return_value = None  # Docker not available

        with pytest.raises(MissingDependencyError, match="Docker backend requires 'docker' package"):
            create_session(backend=SandboxBackend.DOCKER)

    @patch("llm_sandbox.session.find_spec")
    def test_create_session_missing_kubernetes_dependency(self, mock_find_spec: MagicMock) -> None:
        """Test creating session when Kubernetes dependency is missing."""
        mock_find_spec.return_value = None  # Kubernetes not available

        with pytest.raises(MissingDependencyError, match="Kubernetes backend requires 'kubernetes' package"):
            create_session(backend=SandboxBackend.KUBERNETES)

    @patch("llm_sandbox.session.find_spec")
    def test_create_session_missing_podman_dependency(self, mock_find_spec: MagicMock) -> None:
        """Test creating session when Podman dependency is missing."""
        mock_find_spec.return_value = None  # Podman not available

        with pytest.raises(MissingDependencyError, match="Podman backend requires 'podman' package"):
            create_session(backend=SandboxBackend.PODMAN)

    @patch("llm_sandbox.session.find_spec")
    def test_create_session_missing_micromamba_docker_dependency(self, mock_find_spec: MagicMock) -> None:
        """Test creating Micromamba session when Docker dependency is missing."""
        mock_find_spec.return_value = None  # Docker not available

        with pytest.raises(MissingDependencyError, match="Docker backend requires 'docker' package"):
            create_session(backend=SandboxBackend.MICROMAMBA)


class TestBackendCommonInterface:
    """Test that all backends implement the common interface correctly."""

    @patch("llm_sandbox.session.find_spec")
    def test_all_backends_implement_required_methods(self, mock_find_spec: MagicMock) -> None:
        """Test that all backends implement the required session methods."""
        mock_find_spec.return_value = MagicMock()  # All dependencies available

        backends_to_test = [
            SandboxBackend.DOCKER,
            SandboxBackend.KUBERNETES,
            SandboxBackend.PODMAN,
            SandboxBackend.MICROMAMBA,
        ]

        required_methods = [
            "open",
            "close",
            "run",
            "copy_to_runtime",
            "copy_from_runtime",
            "execute_command",
            "execute_commands",
            "install",
            "environment_setup",
            "get_archive",
            "is_safe",
            "__enter__",
            "__exit__",
        ]

        for backend in backends_to_test:
            mock_session_instance = MagicMock()

            if backend == SandboxBackend.DOCKER:
                patch_path = "llm_sandbox.docker.SandboxDockerSession"
            elif backend == SandboxBackend.KUBERNETES:
                patch_path = "llm_sandbox.kubernetes.SandboxKubernetesSession"
            elif backend == SandboxBackend.PODMAN:
                patch_path = "llm_sandbox.podman.SandboxPodmanSession"
            elif backend == SandboxBackend.MICROMAMBA:
                patch_path = "llm_sandbox.micromamba.MicromambaSession"

            with patch(patch_path, return_value=mock_session_instance):
                session = create_session(backend=backend)

                for method_name in required_methods:
                    assert hasattr(session, method_name), f"{backend} should have {method_name} method"

    @patch("llm_sandbox.session.find_spec")
    def test_all_backends_accept_common_parameters(self, mock_find_spec: MagicMock) -> None:
        """Test that all backends accept common initialization parameters."""
        mock_find_spec.return_value = MagicMock()  # All dependencies available

        common_params = {
            "lang": "python",
            "verbose": True,
            "image": "test:latest",
            "workdir": "/test",
            "security_policy": SecurityPolicy(patterns=[], restricted_modules=[]),
        }

        backends_to_test = [
            SandboxBackend.DOCKER,
            SandboxBackend.KUBERNETES,
            SandboxBackend.PODMAN,
            SandboxBackend.MICROMAMBA,
        ]

        for backend in backends_to_test:
            mock_session_instance = MagicMock()

            if backend == SandboxBackend.DOCKER:
                patch_path = "llm_sandbox.docker.SandboxDockerSession"
            elif backend == SandboxBackend.KUBERNETES:
                patch_path = "llm_sandbox.kubernetes.SandboxKubernetesSession"
            elif backend == SandboxBackend.PODMAN:
                patch_path = "llm_sandbox.podman.SandboxPodmanSession"
            elif backend == SandboxBackend.MICROMAMBA:
                patch_path = "llm_sandbox.micromamba.MicromambaSession"

            with patch(patch_path, return_value=mock_session_instance) as mock_session_class:
                _ = create_session(backend=backend, **common_params)

                # Verify the session was created with the expected parameters
                mock_session_class.assert_called_once()
                call_kwargs = mock_session_class.call_args[1]

                assert call_kwargs["lang"] == "python"
                assert call_kwargs["verbose"] is True
                assert call_kwargs["workdir"] == "/test"


class TestBackendSpecificFeatures:
    """Test backend-specific features and parameters."""

    @patch("llm_sandbox.session.find_spec")
    @patch("llm_sandbox.docker.SandboxDockerSession")
    def test_docker_specific_features(self, mock_docker_session: MagicMock, mock_find_spec: MagicMock) -> None:
        """Test Docker-specific features."""
        mock_find_spec.return_value = MagicMock()
        mock_session_instance = MagicMock()
        mock_docker_session.return_value = mock_session_instance

        from docker.types import Mount

        test_mounts = [Mount("/host", "/container", type="bind")]

        _ = create_session(
            backend=SandboxBackend.DOCKER,
            dockerfile="/path/to/Dockerfile",
            mounts=test_mounts,
            stream=False,
            runtime_configs={"mem_limit": "1g"},
            commit_container=True,
        )

        mock_docker_session.assert_called_once()
        call_kwargs = mock_docker_session.call_args[1]
        assert call_kwargs["dockerfile"] == "/path/to/Dockerfile"
        assert call_kwargs["mounts"] == test_mounts
        assert call_kwargs["stream"] is False
        assert call_kwargs["runtime_configs"] == {"mem_limit": "1g"}
        assert call_kwargs["commit_container"] is True

    @patch("llm_sandbox.session.find_spec")
    @patch("llm_sandbox.kubernetes.SandboxKubernetesSession")
    def test_kubernetes_specific_features(self, mock_k8s_session: MagicMock, mock_find_spec: MagicMock) -> None:
        """Test Kubernetes-specific features."""
        mock_find_spec.return_value = MagicMock()
        mock_session_instance = MagicMock()
        mock_k8s_session.return_value = mock_session_instance

        custom_pod_manifest = {
            "apiVersion": "v1",
            "kind": "Pod",
            "spec": {"containers": [{"name": "test", "image": "test:latest"}]},
        }

        _ = create_session(
            backend=SandboxBackend.KUBERNETES,
            kube_namespace="test-namespace",
            env_vars={"TEST": "value"},
            pod_manifest=custom_pod_manifest,
        )

        mock_k8s_session.assert_called_once()
        call_kwargs = mock_k8s_session.call_args[1]
        assert call_kwargs["kube_namespace"] == "test-namespace"
        assert call_kwargs["env_vars"] == {"TEST": "value"}
        assert call_kwargs["pod_manifest"] == custom_pod_manifest

    @patch("llm_sandbox.session.find_spec")
    @patch("llm_sandbox.podman.SandboxPodmanSession")
    def test_podman_specific_features(self, mock_podman_session: MagicMock, mock_find_spec: MagicMock) -> None:
        """Test Podman-specific features."""
        mock_find_spec.return_value = MagicMock()
        mock_session_instance = MagicMock()
        mock_podman_session.return_value = mock_session_instance

        custom_client = MagicMock()

        _ = create_session(
            backend=SandboxBackend.PODMAN,
            client=custom_client,
            dockerfile="/path/to/Containerfile",
            mounts=["/host:/container"],
        )

        mock_podman_session.assert_called_once()
        call_kwargs = mock_podman_session.call_args[1]
        assert call_kwargs["client"] == custom_client
        assert call_kwargs["dockerfile"] == "/path/to/Containerfile"
        assert call_kwargs["mounts"] == ["/host:/container"]

    @patch("llm_sandbox.session.find_spec")
    @patch("llm_sandbox.micromamba.MicromambaSession")
    def test_micromamba_specific_features(self, mock_micromamba_session: MagicMock, mock_find_spec: MagicMock) -> None:
        """Test Micromamba-specific features."""
        mock_find_spec.return_value = MagicMock()
        mock_session_instance = MagicMock()
        mock_micromamba_session.return_value = mock_session_instance

        _ = create_session(
            backend=SandboxBackend.MICROMAMBA, environment="data_science", image="mambaorg/micromamba:latest"
        )

        mock_micromamba_session.assert_called_once()
        call_kwargs = mock_micromamba_session.call_args[1]
        assert call_kwargs["environment"] == "data_science"
        assert call_kwargs["image"] == "mambaorg/micromamba:latest"


class TestBackendConsistency:
    """Test that all backends behave consistently for common operations."""

    @patch("llm_sandbox.session.find_spec")
    def test_all_backends_support_python_language(self, mock_find_spec: MagicMock) -> None:
        """Test that all backends support Python language."""
        mock_find_spec.return_value = MagicMock()

        backends = [
            SandboxBackend.DOCKER,
            SandboxBackend.KUBERNETES,
            SandboxBackend.PODMAN,
            SandboxBackend.MICROMAMBA,
        ]

        for backend in backends:
            mock_session_instance = MagicMock()

            if backend == SandboxBackend.DOCKER:
                patch_path = "llm_sandbox.docker.SandboxDockerSession"
            elif backend == SandboxBackend.KUBERNETES:
                patch_path = "llm_sandbox.kubernetes.SandboxKubernetesSession"
            elif backend == SandboxBackend.PODMAN:
                patch_path = "llm_sandbox.podman.SandboxPodmanSession"
            elif backend == SandboxBackend.MICROMAMBA:
                patch_path = "llm_sandbox.micromamba.MicromambaSession"

            with patch(patch_path, return_value=mock_session_instance) as mock_session_class:
                _ = create_session(backend=backend, lang=SupportedLanguage.PYTHON)

                # Verify session was created with Python language
                mock_session_class.assert_called_once()
                call_kwargs = mock_session_class.call_args[1]
                assert call_kwargs["lang"] == SupportedLanguage.PYTHON

    @patch("llm_sandbox.session.find_spec")
    def test_all_backends_support_security_policies(self, mock_find_spec: MagicMock) -> None:
        """Test that all backends support security policies."""
        mock_find_spec.return_value = MagicMock()

        security_policy = SecurityPolicy(patterns=[], restricted_modules=[])
        backends = [
            SandboxBackend.DOCKER,
            SandboxBackend.KUBERNETES,
            SandboxBackend.PODMAN,
            SandboxBackend.MICROMAMBA,
        ]

        for backend in backends:
            mock_session_instance = MagicMock()

            if backend == SandboxBackend.DOCKER:
                patch_path = "llm_sandbox.docker.SandboxDockerSession"
            elif backend == SandboxBackend.KUBERNETES:
                patch_path = "llm_sandbox.kubernetes.SandboxKubernetesSession"
            elif backend == SandboxBackend.PODMAN:
                patch_path = "llm_sandbox.podman.SandboxPodmanSession"
            elif backend == SandboxBackend.MICROMAMBA:
                patch_path = "llm_sandbox.micromamba.MicromambaSession"

            with patch(patch_path, return_value=mock_session_instance) as mock_session_class:
                _ = create_session(backend=backend, security_policy=security_policy)

                # Verify session was created with security policy
                mock_session_class.assert_called_once()
                call_kwargs = mock_session_class.call_args[1]
                assert call_kwargs["security_policy"] == security_policy

    @patch("llm_sandbox.session.find_spec")
    def test_all_backends_support_verbose_logging(self, mock_find_spec: MagicMock) -> None:
        """Test that all backends support verbose logging."""
        mock_find_spec.return_value = MagicMock()

        backends = [
            SandboxBackend.DOCKER,
            SandboxBackend.KUBERNETES,
            SandboxBackend.PODMAN,
            SandboxBackend.MICROMAMBA,
        ]

        for verbose_setting in [True, False]:
            for backend in backends:
                mock_session_instance = MagicMock()

                if backend == SandboxBackend.DOCKER:
                    patch_path = "llm_sandbox.docker.SandboxDockerSession"
                elif backend == SandboxBackend.KUBERNETES:
                    patch_path = "llm_sandbox.kubernetes.SandboxKubernetesSession"
                elif backend == SandboxBackend.PODMAN:
                    patch_path = "llm_sandbox.podman.SandboxPodmanSession"
                elif backend == SandboxBackend.MICROMAMBA:
                    patch_path = "llm_sandbox.micromamba.MicromambaSession"

                with patch(patch_path, return_value=mock_session_instance) as mock_session_class:
                    _ = create_session(backend=backend, verbose=verbose_setting)

                    # Verify session was created with verbose setting
                    mock_session_class.assert_called_once()
                    call_kwargs = mock_session_class.call_args[1]
                    assert call_kwargs["verbose"] == verbose_setting


class TestBackendErrorHandling:
    """Test error handling across different backends."""

    @patch("llm_sandbox.session.find_spec")
    def test_backend_import_errors_are_handled(self, mock_find_spec: MagicMock) -> None:
        """Test that import errors for backend modules are handled gracefully."""
        mock_find_spec.return_value = MagicMock()  # Dependencies available

        # Test what happens when a backend module can't be imported
        with patch("builtins.__import__") as mock_import:
            mock_import.side_effect = ImportError("Module not found")

            with pytest.raises(ImportError):
                create_session(backend=SandboxBackend.DOCKER)

    @patch("llm_sandbox.session.find_spec")
    def test_backend_initialization_errors_are_propagated(self, mock_find_spec: MagicMock) -> None:
        """Test that backend initialization errors are properly propagated."""
        mock_find_spec.return_value = MagicMock()

        with patch("llm_sandbox.docker.SandboxDockerSession") as mock_docker_session:
            mock_docker_session.side_effect = RuntimeError("Docker daemon not running")

            with pytest.raises(RuntimeError, match="Docker daemon not running"):
                create_session(backend=SandboxBackend.DOCKER)


class TestCreateSessionBackwardsCompatibility:
    """Test backwards compatibility of create_session function."""

    @patch("llm_sandbox.session.find_spec")
    @patch("llm_sandbox.docker.SandboxDockerSession")
    def test_positional_arguments_still_work(self, mock_docker_session: MagicMock, mock_find_spec: MagicMock) -> None:
        """Test that positional arguments still work for backwards compatibility."""
        mock_find_spec.return_value = MagicMock()
        mock_session_instance = MagicMock()
        mock_docker_session.return_value = mock_session_instance

        # Test with positional arguments (old style)
        session = create_session(SandboxBackend.DOCKER, "python", True)

        mock_docker_session.assert_called_once_with("python", True)
        assert session == mock_session_instance

    @patch("llm_sandbox.session.find_spec")
    @patch("llm_sandbox.docker.SandboxDockerSession")
    def test_mixed_positional_and_keyword_arguments(
        self, mock_docker_session: MagicMock, mock_find_spec: MagicMock
    ) -> None:
        """Test mixing positional and keyword arguments."""
        mock_find_spec.return_value = MagicMock()
        mock_session_instance = MagicMock()
        mock_docker_session.return_value = mock_session_instance

        _ = create_session(SandboxBackend.DOCKER, "python", verbose=True, image="custom:latest")

        mock_docker_session.assert_called_once_with("python", verbose=True, image="custom:latest")

    @patch("llm_sandbox.session.find_spec")
    @patch("llm_sandbox.docker.SandboxDockerSession")
    def test_default_backend_is_docker(self, mock_docker_session: MagicMock, mock_find_spec: MagicMock) -> None:
        """Test that Docker is the default backend when none specified."""
        mock_find_spec.return_value = MagicMock()
        mock_session_instance = MagicMock()
        mock_docker_session.return_value = mock_session_instance

        session = create_session(lang="python")  # No backend specified

        mock_docker_session.assert_called_once_with(lang="python")
        assert session == mock_session_instance


class TestSandboxSessionAlias:
    """Test that SandboxSession alias works correctly."""

    def test_sandbox_session_alias_exists(self) -> None:
        """Test that SandboxSession alias exists and points to create_session."""
        from llm_sandbox.session import SandboxSession, create_session

        assert SandboxSession == create_session

    @patch("llm_sandbox.session.find_spec")
    @patch("llm_sandbox.docker.SandboxDockerSession")
    def test_sandbox_session_alias_works(self, mock_docker_session: MagicMock, mock_find_spec: MagicMock) -> None:
        """Test that SandboxSession alias works the same as create_session."""
        from llm_sandbox.session import SandboxSession

        mock_find_spec.return_value = MagicMock()
        mock_session_instance = MagicMock()
        mock_docker_session.return_value = mock_session_instance

        # Test using the alias
        session = SandboxSession(backend=SandboxBackend.DOCKER, lang="python")

        mock_docker_session.assert_called_once_with(lang="python")
        assert session == mock_session_instance
