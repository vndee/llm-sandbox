"""Tests for artifact pooled sandbox session.

These tests verify the ArtifactPooledSandboxSession functionality,
which combines container pooling with artifact extraction.
"""

from unittest.mock import MagicMock, Mock, patch

import pytest

from llm_sandbox.const import SupportedLanguage
from llm_sandbox.data import ConsoleOutput, ExecutionResult, Plot, PlotFormat
from llm_sandbox.exceptions import LanguageNotSupportPlotError
from llm_sandbox.pool.base import ContainerPoolManager
from llm_sandbox.pool.config import PoolConfig
from llm_sandbox.pool.session import ArtifactPooledSandboxSession


class MockPoolManager(ContainerPoolManager):
    """Mock pool manager for testing."""

    def __init__(self, *args, **kwargs):
        """Initialize mock pool manager."""
        self._container_counter = 0
        # Skip parent init to avoid background threads
        self.client = Mock()
        self.config = PoolConfig()
        self.lang = SupportedLanguage.PYTHON
        self.image = "test:latest"
        self.session_kwargs = {}
        self._pool = []
        self._closed = False
        self.logger = Mock()

    def acquire(self):
        """Mock acquire."""
        from llm_sandbox.pool.base import ContainerState, PooledContainer

        self._container_counter += 1
        return PooledContainer(
            container_id=f"mock_{self._container_counter}",
            container=f"mock_container_{self._container_counter}",
            state=ContainerState.BUSY,
        )

    def release(self, container):
        """Mock release."""

    def close(self):
        """Mock close."""
        self._closed = True

    def _create_session_for_container(self):
        """Mock create session."""

    def _destroy_container_impl(self, container):
        """Mock destroy."""

    def _get_container_id(self, container):
        """Mock get container ID."""
        return str(container)

    def _health_check_impl(self, container):
        """Mock health check."""
        return True


class TestArtifactPooledSandboxSession:
    """Tests for ArtifactPooledSandboxSession."""

    @patch("llm_sandbox.pool.session.PooledSandboxSession")
    def test_initialization(self, mock_pooled_session_class):
        """Test artifact pooled session initialization."""
        pool = MockPoolManager()
        mock_session_instance = Mock()
        mock_pooled_session_class.return_value = mock_session_instance

        session = ArtifactPooledSandboxSession(
            pool_manager=pool,
            verbose=True,
            enable_plotting=True,
        )

        # Should create underlying PooledSandboxSession
        mock_pooled_session_class.assert_called_once()
        call_kwargs = mock_pooled_session_class.call_args.kwargs
        assert call_kwargs["pool_manager"] == pool
        assert call_kwargs["verbose"] is True
        assert session.enable_plotting is True

    @patch("llm_sandbox.pool.session.PooledSandboxSession")
    def test_context_manager(self, mock_pooled_session_class):
        """Test artifact pooled session as context manager."""
        pool = MockPoolManager()
        mock_session_instance = Mock()
        mock_pooled_session_class.return_value = mock_session_instance

        with ArtifactPooledSandboxSession(pool_manager=pool) as session:
            pass

        # Should call __enter__ and __exit__ on underlying session
        mock_session_instance.__enter__.assert_called_once()
        mock_session_instance.__exit__.assert_called_once()

    @patch("llm_sandbox.pool.session.PooledSandboxSession")
    def test_run_with_plots(self, mock_pooled_session_class):
        """Test running code with plot extraction."""
        pool = MockPoolManager()
        mock_session_instance = Mock()
        mock_backend_session = Mock()
        mock_language_handler = Mock()

        # Setup mocks
        mock_session_instance.backend_session = mock_backend_session
        mock_backend_session.language_handler = mock_language_handler
        mock_backend_session.config.get_execution_timeout.return_value = 60
        mock_language_handler.is_support_plot_detection = True
        mock_language_handler.name = "python"

        # Mock run_with_artifacts to return result and plots
        console_output = ConsoleOutput(exit_code=0, stdout="output", stderr="")
        test_plot = Plot(
            content_base64="base64data",
            format=PlotFormat.PNG,
            filename="plot.png",
        )
        mock_language_handler.run_with_artifacts.return_value = (console_output, [test_plot])

        mock_pooled_session_class.return_value = mock_session_instance

        session = ArtifactPooledSandboxSession(
            pool_manager=pool,
            enable_plotting=True,
        )
        session._session = mock_session_instance

        result = session.run("import matplotlib.pyplot as plt; plt.plot([1,2,3])")

        # Should return ExecutionResult with plots
        assert isinstance(result, ExecutionResult)
        assert result.exit_code == 0
        assert result.stdout == "output"
        assert len(result.plots) == 1
        assert result.plots[0] == test_plot

    @patch("llm_sandbox.pool.session.PooledSandboxSession")
    def test_run_unsupported_language(self, mock_pooled_session_class):
        """Test running code with language that doesn't support plots."""
        pool = MockPoolManager()
        mock_session_instance = Mock()
        mock_backend_session = Mock()
        mock_language_handler = Mock()

        # Setup mocks - language doesn't support plots
        mock_session_instance.backend_session = mock_backend_session
        mock_backend_session.language_handler = mock_language_handler
        mock_language_handler.is_support_plot_detection = False
        mock_language_handler.name = "java"

        mock_pooled_session_class.return_value = mock_session_instance

        session = ArtifactPooledSandboxSession(
            pool_manager=pool,
            enable_plotting=True,
        )
        session._session = mock_session_instance

        # Should raise LanguageNotSupportPlotError
        with pytest.raises(LanguageNotSupportPlotError):
            session.run("System.out.println('test');")

    @patch("llm_sandbox.pool.session.PooledSandboxSession")
    def test_run_with_clear_plots(self, mock_pooled_session_class):
        """Test running code with clear_plots option."""
        pool = MockPoolManager()
        mock_session_instance = Mock()
        mock_backend_session = Mock()
        mock_language_handler = Mock()

        # Setup mocks
        mock_session_instance.backend_session = mock_backend_session
        mock_session_instance.execute_command = Mock()
        mock_backend_session.language_handler = mock_language_handler
        mock_backend_session.config.get_execution_timeout.return_value = 60
        mock_language_handler.is_support_plot_detection = True
        mock_language_handler.name = "python"

        console_output = ConsoleOutput(exit_code=0, stdout="output", stderr="")
        mock_language_handler.run_with_artifacts.return_value = (console_output, [])

        mock_pooled_session_class.return_value = mock_session_instance

        session = ArtifactPooledSandboxSession(
            pool_manager=pool,
            enable_plotting=True,
        )
        session._session = mock_session_instance

        result = session.run("print('test')", clear_plots=True)

        # Should call execute_command to clear plots
        mock_session_instance.execute_command.assert_called_once()
        call_args = mock_session_instance.execute_command.call_args
        assert "rm -rf /tmp/sandbox_plots/*" in call_args[0][0]
        assert "echo 0 > /tmp/sandbox_plots/.counter" in call_args[0][0]

    @patch("llm_sandbox.pool.session.PooledSandboxSession")
    def test_clear_plots_method(self, mock_pooled_session_class):
        """Test clear_plots method."""
        pool = MockPoolManager()
        mock_session_instance = Mock()
        mock_session_instance.execute_command = Mock()

        mock_pooled_session_class.return_value = mock_session_instance

        session = ArtifactPooledSandboxSession(
            pool_manager=pool,
            enable_plotting=True,
        )
        session._session = mock_session_instance

        session.clear_plots()

        # Should call execute_command to clear plots
        mock_session_instance.execute_command.assert_called_once()

    @patch("llm_sandbox.pool.session.PooledSandboxSession")
    def test_clear_plots_disabled(self, mock_pooled_session_class):
        """Test clear_plots when plotting is disabled."""
        pool = MockPoolManager()
        mock_session_instance = Mock()

        mock_pooled_session_class.return_value = mock_session_instance

        session = ArtifactPooledSandboxSession(
            pool_manager=pool,
            enable_plotting=False,
        )
        session._session = mock_session_instance

        session.clear_plots()

        # Should not call execute_command when plotting disabled
        mock_session_instance.execute_command.assert_not_called()

    @patch("llm_sandbox.pool.session.PooledSandboxSession")
    def test_run_with_custom_timeout(self, mock_pooled_session_class):
        """Test running code with custom timeout."""
        pool = MockPoolManager()
        mock_session_instance = Mock()
        mock_backend_session = Mock()
        mock_language_handler = Mock()

        # Setup mocks
        mock_session_instance.backend_session = mock_backend_session
        mock_backend_session.language_handler = mock_language_handler
        mock_backend_session.config.get_execution_timeout.return_value = 60
        mock_language_handler.is_support_plot_detection = True

        console_output = ConsoleOutput(exit_code=0, stdout="", stderr="")
        mock_language_handler.run_with_artifacts.return_value = (console_output, [])

        mock_pooled_session_class.return_value = mock_session_instance

        session = ArtifactPooledSandboxSession(
            pool_manager=pool,
            enable_plotting=True,
        )
        session._session = mock_session_instance

        session.run("print('test')", timeout=120)

        # Should use custom timeout
        call_args = mock_language_handler.run_with_artifacts.call_args
        assert call_args.kwargs["timeout"] == 120

    @patch("llm_sandbox.pool.session.PooledSandboxSession")
    def test_run_with_libraries(self, mock_pooled_session_class):
        """Test running code with library installation."""
        pool = MockPoolManager()
        mock_session_instance = Mock()
        mock_backend_session = Mock()
        mock_language_handler = Mock()

        # Setup mocks
        mock_session_instance.backend_session = mock_backend_session
        mock_backend_session.language_handler = mock_language_handler
        mock_backend_session.config.get_execution_timeout.return_value = 60
        mock_language_handler.is_support_plot_detection = True

        console_output = ConsoleOutput(exit_code=0, stdout="", stderr="")
        mock_language_handler.run_with_artifacts.return_value = (console_output, [])

        mock_pooled_session_class.return_value = mock_session_instance

        session = ArtifactPooledSandboxSession(
            pool_manager=pool,
            enable_plotting=True,
        )
        session._session = mock_session_instance

        session.run("import numpy", libraries=["numpy"])

        # Should pass libraries to run_with_artifacts
        call_args = mock_language_handler.run_with_artifacts.call_args
        assert call_args.kwargs["libraries"] == ["numpy"]

    @patch("llm_sandbox.pool.session.PooledSandboxSession")
    def test_attribute_delegation(self, mock_pooled_session_class):
        """Test that unknown attributes are delegated to underlying session."""
        pool = MockPoolManager()
        mock_session_instance = Mock()
        mock_session_instance.some_method = Mock(return_value="result")

        mock_pooled_session_class.return_value = mock_session_instance

        session = ArtifactPooledSandboxSession(pool_manager=pool)
        session._session = mock_session_instance

        # Should delegate to underlying session
        result = session.some_method()
        assert result == "result"
        mock_session_instance.some_method.assert_called_once()

    @patch("llm_sandbox.pool.session.PooledSandboxSession")
    def test_run_plotting_disabled(self, mock_pooled_session_class):
        """Test running code with plotting disabled."""
        pool = MockPoolManager()
        mock_session_instance = Mock()
        mock_backend_session = Mock()
        mock_language_handler = Mock()

        # Setup mocks
        mock_session_instance.backend_session = mock_backend_session
        mock_backend_session.language_handler = mock_language_handler
        mock_backend_session.config.get_execution_timeout.return_value = 60
        mock_language_handler.is_support_plot_detection = True

        console_output = ConsoleOutput(exit_code=0, stdout="", stderr="")
        mock_language_handler.run_with_artifacts.return_value = (console_output, [])

        mock_pooled_session_class.return_value = mock_session_instance

        session = ArtifactPooledSandboxSession(
            pool_manager=pool,
            enable_plotting=False,  # Plotting disabled
        )
        session._session = mock_session_instance

        result = session.run("print('test')")

        # Should pass enable_plotting=False
        call_args = mock_language_handler.run_with_artifacts.call_args
        assert call_args.kwargs["enable_plotting"] is False
