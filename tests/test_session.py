# ruff: noqa: SLF001, PLR2004

"""Tests for session module."""

from unittest.mock import MagicMock, patch

import pytest

from llm_sandbox.const import SandboxBackend, SupportedLanguage
from llm_sandbox.core.config import SessionConfig
from llm_sandbox.data import ExecutionResult, FileType, PlotOutput
from llm_sandbox.exceptions import LanguageNotSupportPlotError, MissingDependencyError, UnsupportedBackendError
from llm_sandbox.session import ArtifactSandboxSession, SandboxSession, create_session


class TestCreateSession:
    """Test create_session function."""

    @patch("llm_sandbox.session.find_spec")
    def test_check_dependency_docker_missing(self, mock_find_spec: MagicMock) -> None:
        """Test that MissingDependencyError is raised when docker package is missing."""
        mock_find_spec.return_value = None

        with pytest.raises(MissingDependencyError, match="Docker backend requires 'docker' package"):
            create_session(SandboxBackend.DOCKER)

    @patch("llm_sandbox.session.find_spec")
    def test_check_dependency_kubernetes_missing(self, mock_find_spec: MagicMock) -> None:
        """Test that MissingDependencyError is raised when kubernetes package is missing."""
        mock_find_spec.return_value = None

        with pytest.raises(MissingDependencyError, match="Kubernetes backend requires 'kubernetes' package"):
            create_session(SandboxBackend.KUBERNETES)

    @patch("llm_sandbox.session.find_spec")
    def test_check_dependency_podman_missing(self, mock_find_spec: MagicMock) -> None:
        """Test that MissingDependencyError is raised when podman package is missing."""
        mock_find_spec.return_value = None

        with pytest.raises(MissingDependencyError, match="Podman backend requires 'podman' package"):
            create_session(SandboxBackend.PODMAN)

    @patch("llm_sandbox.session.find_spec")
    @patch("llm_sandbox.docker.SandboxDockerSession")
    def test_create_docker_session(self, mock_docker_session: MagicMock, mock_find_spec: MagicMock) -> None:
        """Test creating a Docker session."""
        mock_find_spec.return_value = MagicMock()
        mock_instance = MagicMock()
        mock_docker_session.return_value = mock_instance

        session = create_session(SandboxBackend.DOCKER, lang="python", verbose=True)

        assert session == mock_instance
        mock_docker_session.assert_called_once_with(lang="python", verbose=True)

    @patch("llm_sandbox.session.find_spec")
    @patch("llm_sandbox.kubernetes.SandboxKubernetesSession")
    def test_create_kubernetes_session(self, mock_k8s_session: MagicMock, mock_find_spec: MagicMock) -> None:
        """Test creating a Kubernetes session."""
        mock_find_spec.return_value = MagicMock()
        mock_instance = MagicMock()
        mock_k8s_session.return_value = mock_instance

        session = create_session(SandboxBackend.KUBERNETES, lang="java")

        assert session == mock_instance
        mock_k8s_session.assert_called_once_with(lang="java")

    @patch("llm_sandbox.session.find_spec")
    @patch("llm_sandbox.podman.SandboxPodmanSession")
    def test_create_podman_session(self, mock_podman_session: MagicMock, mock_find_spec: MagicMock) -> None:
        """Test creating a Podman session."""
        mock_find_spec.return_value = MagicMock()
        mock_instance = MagicMock()
        mock_podman_session.return_value = mock_instance

        session = create_session(SandboxBackend.PODMAN, lang="javascript")

        assert session == mock_instance
        mock_podman_session.assert_called_once_with(lang="javascript")

    @patch("llm_sandbox.session.find_spec")
    @patch("llm_sandbox.micromamba.MicromambaSession")
    def test_create_micromamba_session(self, mock_micromamba_session: MagicMock, mock_find_spec: MagicMock) -> None:
        """Test creating a Micromamba session."""
        mock_find_spec.return_value = MagicMock()
        mock_instance = MagicMock()
        mock_micromamba_session.return_value = mock_instance

        session = create_session(SandboxBackend.MICROMAMBA, lang="cpp")

        assert session == mock_instance
        mock_micromamba_session.assert_called_once_with(lang="cpp")

    def test_unsupported_backend(self) -> None:
        """Test that UnsupportedBackendError is raised for unsupported backend."""
        with pytest.raises(UnsupportedBackendError):
            create_session("invalid_backend")  # type: ignore[arg-type]

    def test_sandbox_session_alias(self) -> None:
        """Test that SandboxSession is an alias for create_session."""
        assert SandboxSession == create_session


class TestArtifactSandboxSession:
    """Test ArtifactSandboxSession class."""

    @patch("llm_sandbox.session.create_session")
    def test_init_with_defaults(self, mock_create_session: MagicMock) -> None:
        """Test ArtifactSandboxSession initialization with default parameters."""
        mock_session = MagicMock()
        mock_create_session.return_value = mock_session

        artifact_session = ArtifactSandboxSession()

        assert artifact_session._session == mock_session
        assert artifact_session.enable_plotting is True
        mock_create_session.assert_called_once_with(
            backend=SandboxBackend.DOCKER,
            image=None,
            dockerfile=None,
            lang=SupportedLanguage.PYTHON,
            keep_template=False,
            commit_container=False,
            verbose=False,
            runtime_configs=None,
            workdir="/sandbox",
            security_policy=None,
            container_id=None,
        )

    @patch("llm_sandbox.session.create_session")
    def test_init_with_custom_params(self, mock_create_session: MagicMock) -> None:
        """Test ArtifactSandboxSession initialization with custom parameters."""
        mock_session = MagicMock()
        mock_create_session.return_value = mock_session

        artifact_session = ArtifactSandboxSession(
            enable_plotting=False,
            backend=SandboxBackend.KUBERNETES,
        )

        assert artifact_session._session == mock_session
        assert artifact_session.enable_plotting is False
        mock_create_session.assert_called_once_with(
            backend=SandboxBackend.KUBERNETES,
            image=None,
            dockerfile=None,
            lang=SupportedLanguage.PYTHON,
            keep_template=False,
            commit_container=False,
            verbose=False,
            runtime_configs=None,
            workdir="/sandbox",
            security_policy=None,
            container_id=None,
        )

    @patch("llm_sandbox.session.create_session")
    def test_context_manager_enter(self, mock_create_session: MagicMock) -> None:
        """Test ArtifactSandboxSession context manager enter."""
        mock_session = MagicMock()
        mock_create_session.return_value = mock_session

        artifact_session = ArtifactSandboxSession()

        result = artifact_session.__enter__()

        assert result == artifact_session
        mock_session.__enter__.assert_called_once()

    @patch("llm_sandbox.session.create_session")
    def test_context_manager_exit(self, mock_create_session: MagicMock) -> None:
        """Test ArtifactSandboxSession context manager exit."""
        mock_session = MagicMock()
        mock_create_session.return_value = mock_session

        artifact_session = ArtifactSandboxSession()

        exc_type = Exception
        exc_val = Exception("test")
        exc_tb = None

        artifact_session.__exit__(exc_type, exc_val, exc_tb)

        mock_session.__exit__.assert_called_once_with(exc_type, exc_val, exc_tb)

    @patch("llm_sandbox.session.create_session")
    def test_getattr_delegation(self, mock_create_session: MagicMock) -> None:
        """Test that unknown attributes are delegated to the underlying session."""
        mock_session = MagicMock()
        mock_session.some_method.return_value = "test_result"
        mock_create_session.return_value = mock_session

        artifact_session = ArtifactSandboxSession()

        result = artifact_session.some_method("arg1", "arg2")

        assert result == "test_result"
        mock_session.some_method.assert_called_once_with("arg1", "arg2")

    @patch("llm_sandbox.session.create_session")
    def test_run_with_plotting_disabled(self, mock_create_session: MagicMock) -> None:
        """Test run method with plotting disabled."""
        mock_session = MagicMock()
        mock_language_handler = MagicMock()
        mock_language_handler.run_with_artifacts.return_value = (MagicMock(exit_code=0, stdout="output", stderr=""), [])
        mock_session.language_handler = mock_language_handler
        mock_create_session.return_value = mock_session

        artifact_session = ArtifactSandboxSession(enable_plotting=False)

        result = artifact_session.run("print('hello')", ["numpy"])

        assert isinstance(result, ExecutionResult)
        assert result.exit_code == 0
        assert result.stdout == "output"
        assert result.stderr == ""
        assert result.plots == []

        mock_language_handler.run_with_artifacts.assert_called_once_with(
            container=mock_session,
            code="print('hello')",
            libraries=["numpy"],
            enable_plotting=False,
            output_dir="/tmp/sandbox_plots",
            timeout=30,
        )

    @patch("llm_sandbox.session.create_session")
    def test_run_with_plotting_enabled_supported_language(self, mock_create_session: MagicMock) -> None:
        """Test run method with plotting enabled and supported language."""
        mock_session = MagicMock()
        mock_language_handler = MagicMock()
        mock_language_handler.is_support_plot_detection = True
        mock_plot = PlotOutput(format=FileType.PNG, content_base64="dGVzdA==")
        mock_language_handler.run_with_artifacts.return_value = (
            MagicMock(exit_code=0, stdout="output", stderr=""),
            [mock_plot],
        )
        mock_session.language_handler = mock_language_handler
        mock_create_session.return_value = mock_session

        artifact_session = ArtifactSandboxSession(enable_plotting=True)

        result = artifact_session.run("import matplotlib.pyplot as plt; plt.plot([1,2,3])")

        assert isinstance(result, ExecutionResult)
        assert result.exit_code == 0
        assert result.stdout == "output"
        assert result.stderr == ""
        assert len(result.plots) == 1
        assert result.plots[0] == mock_plot

        mock_language_handler.run_with_artifacts.assert_called_once_with(
            container=mock_session,
            code="import matplotlib.pyplot as plt; plt.plot([1,2,3])",
            libraries=None,
            enable_plotting=True,
            output_dir="/tmp/sandbox_plots",
            timeout=30,
        )

    @patch("llm_sandbox.session.create_session")
    def test_run_with_plotting_enabled_unsupported_language(self, mock_create_session: MagicMock) -> None:
        """Test run method with plotting enabled but unsupported language."""
        mock_session = MagicMock()
        mock_language_handler = MagicMock()
        mock_language_handler.is_support_plot_detection = False
        mock_language_handler.name = "java"
        mock_session.language_handler = mock_language_handler
        mock_create_session.return_value = mock_session

        artifact_session = ArtifactSandboxSession(enable_plotting=True)

        with pytest.raises(LanguageNotSupportPlotError):
            artifact_session.run("System.out.println('hello');")


class TestSessionConfig:
    """Test SessionConfig validation."""

    def test_image_and_dockerfile_both_provided(self) -> None:
        """Test validation error when both image and dockerfile are provided."""
        with pytest.raises(ValueError, match="Only one of 'image' or 'dockerfile' can be provided"):
            SessionConfig(image="test-image", dockerfile="test-dockerfile")

    def test_container_id_with_dockerfile(self) -> None:
        """Test validation error when container_id is used with dockerfile."""
        with pytest.raises(ValueError, match="Cannot use 'dockerfile' with existing 'container_id'"):
            SessionConfig(container_id="test-container", dockerfile="test-dockerfile")

    def test_valid_config_combinations(self) -> None:
        """Test valid configuration combinations."""
        # Valid: only image
        config1 = SessionConfig(image="test-image")
        assert config1.image == "test-image"
        assert config1.dockerfile is None

        # Valid: only dockerfile
        config2 = SessionConfig(dockerfile="test-dockerfile")
        assert config2.dockerfile == "test-dockerfile"
        assert config2.image is None

        # Valid: container_id with image
        config3 = SessionConfig(container_id="test-container", image="test-image")
        assert config3.container_id == "test-container"
        assert config3.image == "test-image"

        # Valid: container_id alone
        config4 = SessionConfig(container_id="test-container")
        assert config4.container_id == "test-container"
