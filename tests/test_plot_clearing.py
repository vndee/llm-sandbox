# ruff: noqa: SLF001, PLR2004, ARG001, ARG002, PT011, ANN001, ANN003, ANN202

"""Tests for plot clearing functionality."""

import base64
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from llm_sandbox import ArtifactSandboxSession, SandboxBackend
from llm_sandbox.data import ConsoleOutput, FileType, PlotOutput


def create_mock_plot_data(count: int = 1) -> bytes | list[PlotOutput]:
    """Create mock plot data (1x1 pixel PNG).

    Args:
        count: Number of plot outputs to create. If 1, returns bytes. If > 1, returns list[PlotOutput].

    Returns:
        bytes or list[PlotOutput]: Minimal valid PNG data or list of PlotOutput objects

    """
    png_data = base64.b64decode(
        "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg=="
    )

    if count == 1:
        return png_data

    # Return list of PlotOutput objects
    return [
        PlotOutput(
            format=FileType.PNG,
            content_base64=base64.b64encode(png_data).decode("utf-8"),
        )
        for _ in range(count)
    ]


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
                return exec_result, [plot_data]  # type: ignore[list-item]
            # Second run: 2 plots (cumulative)
            if call_count[0] == 2:
                return exec_result, [plot_data, plot_data]  # type: ignore[list-item]
            # Third run with clear: 0 plots
            if call_count[0] == 3:
                return exec_result, []
            # Fourth run after clear: 1 plot
            if call_count[0] == 4:
                return exec_result, [plot_data]  # type: ignore[list-item]
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
            if "rm -rf /tmp/sandbox_plots" in cmd and ".counter" in cmd:
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
            return exec_result, [plot_data]  # type: ignore[list-item]

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
                return exec_result, [plot_data]  # type: ignore[list-item]
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

    @pytest.mark.parametrize(
        "backend",
        [
            SandboxBackend.DOCKER,
            SandboxBackend.PODMAN,
            SandboxBackend.KUBERNETES,
        ],
    )
    @patch("llm_sandbox.session.create_session")
    def test_cross_backend_compatibility(self, mock_create_session: MagicMock, backend: SandboxBackend) -> None:
        """Test that plot clearing works across all backends (Docker, Podman, Kubernetes)."""
        # Track clear operations
        clear_commands = []

        # Mock session
        mock_session = MagicMock()
        mock_create_session.return_value = mock_session

        # Mock execute_command to capture commands
        def mock_execute_command(cmd: str, **kwargs: Any) -> ConsoleOutput:
            clear_commands.append(cmd)
            return ConsoleOutput(exit_code=0, stdout="", stderr="")

        mock_session.execute_command.side_effect = mock_execute_command
        mock_session.language_handler.is_support_plot_detection = True
        mock_session.language_handler.run_with_artifacts.return_value = (
            ConsoleOutput(exit_code=0, stdout="", stderr=""),
            create_mock_plot_data(2),
        )
        mock_session.config.get_execution_timeout.return_value = 60

        # Create session with specified backend
        with ArtifactSandboxSession(
            lang="python",
            backend=backend,
            enable_plotting=True,
        ) as session:
            # Test auto-clear
            clear_commands.clear()
            session.run("print('test')", clear_plots=True)

            # Verify clear command was executed
            assert len(clear_commands) > 0, f"{backend.value} did not execute clear command"

            # Verify the command structure is correct for all backends
            clear_cmd = clear_commands[0]
            assert "rm -rf /tmp/sandbox_plots/*" in clear_cmd, f"{backend.value} missing rm command"
            assert "echo 0 >" in clear_cmd, f"{backend.value} missing echo command"
            assert ".counter" in clear_cmd, f"{backend.value} missing counter file"

            # All backends should use shell for wildcards and redirection
            assert "sh -c" in clear_cmd or "sh" in clear_cmd, f"{backend.value} not using shell"

    @patch("llm_sandbox.session.create_session")
    def test_shell_command_compatibility(self, mock_create_session: MagicMock) -> None:
        """Test that shell commands (wildcards, redirection) work correctly."""
        executed_commands = []

        # Mock session
        mock_session = MagicMock()
        mock_create_session.return_value = mock_session

        def mock_execute_command(cmd: str, **kwargs: Any) -> ConsoleOutput:
            executed_commands.append(cmd)
            return ConsoleOutput(exit_code=0, stdout="", stderr="")

        mock_session.execute_command.side_effect = mock_execute_command
        mock_session.language_handler.is_support_plot_detection = True
        mock_session.language_handler.run_with_artifacts.return_value = (
            ConsoleOutput(exit_code=0, stdout="", stderr=""),
            [],
        )
        mock_session.config.get_execution_timeout.return_value = 60

        with ArtifactSandboxSession(
            lang="python",
            backend=SandboxBackend.DOCKER,
            enable_plotting=True,
        ) as session:
            session.clear_plots()

        # Verify the command uses shell features correctly
        assert len(executed_commands) > 0
        clear_cmd = executed_commands[0]

        # Must use sh -c for wildcards and redirection to work
        assert "sh -c" in clear_cmd
        # Wildcard for removing all plots
        assert "/tmp/sandbox_plots/*" in clear_cmd
        # Command chaining with &&
        assert "&&" in clear_cmd
        # Output redirection
        assert ">" in clear_cmd
        assert ".counter" in clear_cmd


if __name__ == "__main__":
    pytest.main([__file__])
