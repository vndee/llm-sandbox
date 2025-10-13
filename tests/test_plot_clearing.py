"""Tests for plot clearing functionality."""

import pytest
from llm_sandbox import ArtifactSandboxSession


def test_clear_plots_parameter():
    """Test clear_plots parameter in run method."""
    with ArtifactSandboxSession(lang="python") as session:
        plot_code = """
import matplotlib.pyplot as plt
plt.figure()
plt.plot([1, 2, 3], [1, 4, 2])
plt.show()
"""
        
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


def test_manual_clear_plots():
    """Test manual clear_plots method."""
    with ArtifactSandboxSession(lang="python") as session:
        plot_code = """
import matplotlib.pyplot as plt
plt.figure()
plt.plot([1, 2, 3], [1, 4, 2])
plt.show()
"""
        
        # First run - should generate 1 plot
        result1 = session.run(plot_code)
        assert len(result1.plots) == 1
        
        # Manual clearing
        session.clear_plots()
        
        # Run again - should generate fresh plot
        result2 = session.run(plot_code)
        assert len(result2.plots) == 1


def test_clear_plots_with_no_plotting():
    """Test clear_plots functionality when plotting is disabled."""
    with ArtifactSandboxSession(lang="python", enable_plotting=False) as session:
        result = session.run("print('hello')", clear_plots=True)
        assert result.stdout.strip() == "hello"
        assert len(result.plots) == 0


def test_clear_plots_multiple_libraries():
    """Test clear_plots with different plotting libraries."""
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
        result2 = session.run(plotly_code, clear_plots=True, libraries=['plotly'])
        plotly_plots = len(result2.plots)
        
        # Should have reset the counter
        assert plotly_plots <= initial_plots


if __name__ == "__main__":
    pytest.main([__file__])