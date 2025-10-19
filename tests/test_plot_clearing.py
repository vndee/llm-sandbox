# ruff: noqa: SLF001, PLR2004, ARG001, ARG002, PT011, ANN001, ANN003, ANN202

"""Tests for plot clearing functionality."""

import base64
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from llm_sandbox import ArtifactSandboxSession
from llm_sandbox.data import ConsoleOutput


def create_mock_plot_data() -> bytes:
    """Create a mock plot data (1x1 pixel PNG).

    Returns:
        bytes: Minimal valid PNG data representing a 1x1 transparent pixel

    """
    return base64.b64decode(
        "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg=="
    )


class TestPlotClearing:
    """Test plot clearing functionality in ArtifactSandboxSession."""

    @patch("llm_sandbox.session.create_session")
    def test_clear_plots_parameter(self, mock_create_session: MagicMock) -> None:
        """Test clear_plots parameter in run method."""
        # Mock the internal session
        mock_session = MagicMock()
        mock_handler = MagicMock()
        mock_handler.name = "python"
        mock_handler.is_support_plot_detection = True
        mock_session.language_handler = mock_handler
        mock_session.config = MagicMock()
        mock_session.config.get_execution_timeout.return_value = 60
        mock_create_session.return_value = mock_session

        # Mock plot data
        plot_data = create_mock_plot_data()

        # Track call count for different behaviors
        call_count = [0]

        def mock_run_with_artifacts(**kwargs: Any) -> tuple[ConsoleOutput, list[bytes]]:
            """Mock the run_with_artifacts to simulate plot generation."""
            call_count[0] += 1
            exec_result = ConsoleOutput(exit_code=0, stdout="", stderr="")

            # First run: 1 plot
            if call_count[0] == 1:
                return exec_result, [plot_data]
            # Second run: 2 plots (cumulative)
            if call_count[0] == 2:
                return exec_result, [plot_data, plot_data]
            # Third run with clear: 0 plots
            if call_count[0] == 3:
                return exec_result, []
            # Fourth run after clear: 1 plot
            if call_count[0] == 4:
                return exec_result, [plot_data]
            return exec_result, []

        mock_handler.run_with_artifacts.side_effect = mock_run_with_artifacts

        plot_code = """
import matplotlib.pyplot as plt
plt.figure()
plt.plot([1, 2, 3], [1, 4, 2])
plt.show()
"""

        with ArtifactSandboxSession(lang="python") as session:
            # First run - should generate 1 plot
            result1 = session.run(plot_code)
            assert len(result1.plots) == 1

            # Second run without clearing - should generate another plot
            result2 = session.run(plot_code)
            assert len(result2.plots) == 2  # Previous plot + new plot

            # Third run with clearing - should start fresh
            result3 = session.run("print('no plot')", clear_plots=True)
            assert len(result3.plots) == 0

            # Fourth run after clearing - should generate 1 plot
            result4 = session.run(plot_code, clear_plots=True)
            assert len(result4.plots) == 1

    @patch("llm_sandbox.session.create_session")
    def test_manual_clear_plots(self, mock_create_session: MagicMock) -> None:
        """Test manual clear_plots method."""
        # Mock the internal session
        mock_session = MagicMock()
        mock_handler = MagicMock()
        mock_handler.name = "python"
        mock_handler.is_support_plot_detection = True
        mock_session.language_handler = mock_handler
        mock_session.config = MagicMock()
        mock_session.config.get_execution_timeout.return_value = 60
        mock_create_session.return_value = mock_session

        # Mock plot data
        plot_data = create_mock_plot_data()

        # Track whether plots were cleared
        plots_cleared = [False]

        def mock_execute_command(cmd: str, **kwargs: Any) -> ConsoleOutput:
            """Mock execute_command to detect clear operations."""
            if "clear_plots" in cmd:
                plots_cleared[0] = True
            return ConsoleOutput(exit_code=0, stdout="", stderr="")

        mock_session.execute_command.side_effect = mock_execute_command

        # Track call count for different behaviors
        call_count = [0]

        def mock_run_with_artifacts(**kwargs: Any) -> tuple[ConsoleOutput, list[bytes]]:
            """Mock the run_with_artifacts to simulate plot generation."""
            call_count[0] += 1
            exec_result = ConsoleOutput(exit_code=0, stdout="", stderr="")

            # Always return 1 plot (simulating new plot after clear)
            return exec_result, [plot_data]

        mock_handler.run_with_artifacts.side_effect = mock_run_with_artifacts

        plot_code = """
import matplotlib.pyplot as plt
plt.figure()
plt.plot([1, 2, 3], [1, 4, 2])
plt.show()
"""

        with ArtifactSandboxSession(lang="python") as session:
            # First run - should generate 1 plot
            result1 = session.run(plot_code)
            assert len(result1.plots) == 1

            # Manual clearing
            session.clear_plots()
            assert plots_cleared[0], "clear_plots() should have been called"

            # Run again - should generate fresh plot
            result2 = session.run(plot_code)
            assert len(result2.plots) == 1

    @patch("llm_sandbox.session.create_session")
    def test_clear_plots_with_no_plotting(self, mock_create_session: MagicMock) -> None:
        """Test clear_plots functionality when plotting is disabled."""
        # Mock the internal session
        mock_session = MagicMock()
        mock_handler = MagicMock()
        mock_handler.name = "python"
        mock_handler.is_support_plot_detection = True
        mock_session.language_handler = mock_handler
        mock_session.config = MagicMock()
        mock_session.config.get_execution_timeout.return_value = 60
        mock_create_session.return_value = mock_session

        def mock_run_with_artifacts(**kwargs: Any) -> tuple[ConsoleOutput, list[bytes]]:
            """Mock the run_with_artifacts with no plot generation."""
            exec_result = ConsoleOutput(exit_code=0, stdout="hello", stderr="")
            return exec_result, []

        mock_handler.run_with_artifacts.side_effect = mock_run_with_artifacts

        with ArtifactSandboxSession(lang="python", enable_plotting=False) as session:
            result = session.run("print('hello')", clear_plots=True)
            assert result.stdout.strip() == "hello"
            assert len(result.plots) == 0

    @patch("llm_sandbox.session.create_session")
    def test_clear_plots_multiple_libraries(self, mock_create_session: MagicMock) -> None:
        """Test clear_plots with different plotting libraries."""
        # Mock the internal session
        mock_session = MagicMock()
        mock_handler = MagicMock()
        mock_handler.name = "python"
        mock_handler.is_support_plot_detection = True
        mock_session.language_handler = mock_handler
        mock_session.config = MagicMock()
        mock_session.config.get_execution_timeout.return_value = 60
        mock_create_session.return_value = mock_session

        # Mock plot data
        plot_data = create_mock_plot_data()

        # Track call count for different behaviors
        call_count = [0]

        def mock_run_with_artifacts(**kwargs: Any) -> tuple[ConsoleOutput, list[bytes]]:
            """Mock the run_with_artifacts to simulate plot generation."""
            call_count[0] += 1
            exec_result = ConsoleOutput(exit_code=0, stdout="", stderr="")

            # First run: 1 matplotlib plot
            if call_count[0] == 1 or call_count[0] == 2:
                return exec_result, [plot_data]
            return exec_result, []

        mock_handler.run_with_artifacts.side_effect = mock_run_with_artifacts

        plotly_code = """
try:
    import plotly.graph_objects as go
    fig = go.Figure(data=go.Scatter(x=[1, 2, 3], y=[1, 4, 2]))
    fig.show()
except ImportError:
    print("plotly not available")
"""

        matplotlib_code = """
import matplotlib.pyplot as plt
plt.figure()
plt.plot([1, 2, 3], [1, 4, 2])
plt.show()
"""

        with ArtifactSandboxSession(lang="python") as session:
            # First matplotlib plot
            result1 = session.run(matplotlib_code)
            initial_plots = len(result1.plots)

            # Clear and run plotly
            result2 = session.run(plotly_code, clear_plots=True, libraries=["plotly"])
            plotly_plots = len(result2.plots)

            # Should have reset the counter
            assert plotly_plots <= initial_plots


if __name__ == "__main__":
    pytest.main([__file__])
