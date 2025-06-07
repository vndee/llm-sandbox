"""Tests for llm_sandbox.language_handlers.base module - missing coverage."""

import logging
from unittest.mock import Mock

import pytest

from llm_sandbox.exceptions import CommandFailedError, PackageManagerError
from llm_sandbox.language_handlers.base import AbstractLanguageHandler, LanguageConfig, PlotDetectionConfig, PlotLibrary


class ConcreteLanguageHandler(AbstractLanguageHandler):
    """Concrete implementation for testing."""

    def __init__(self, config: LanguageConfig, logger: logging.Logger | None = None) -> None:
        """Initialize the ConcreteLanguageHandler."""
        super().__init__(logger)
        self.config = config

    def get_import_patterns(self, module: str) -> str:
        """Get import patterns."""
        return f"import {module}"

    @staticmethod
    def get_multiline_comment_patterns() -> str:
        """Get multiline comment patterns."""
        return r'""".*?"""'

    @staticmethod
    def get_inline_comment_patterns() -> str:
        """Get inline comment patterns."""
        return r"#.*$"


class TestMissingCoverage:
    """Test missing coverage lines in base.py."""

    def test_empty_execution_commands_line_78(self) -> None:
        """Test line 78: empty execution commands."""
        config = LanguageConfig(
            name="test",
            file_extension=".test",
            execution_commands=[],  # Empty commands
            package_manager="test-manager",
        )
        handler = ConcreteLanguageHandler(config)

        with pytest.raises(CommandFailedError):
            handler.get_execution_commands("test.py")

    def test_no_package_manager_line_84(self) -> None:
        """Test line 84: no package manager."""
        config = LanguageConfig(
            name="test",
            file_extension=".test",
            execution_commands=["test {file}"],
            package_manager=None,  # No package manager
        )
        handler = ConcreteLanguageHandler(config)

        with pytest.raises(PackageManagerError):
            handler.get_library_installation_command("numpy")

    def test_no_plot_detection_line_212(self) -> None:
        """Test line 212: no plot detection."""
        config = LanguageConfig(
            name="test",
            file_extension=".test",
            execution_commands=["test {file}"],
            package_manager="test-manager",
            plot_detection=None,  # No plot detection
        )
        handler = ConcreteLanguageHandler(config)

        assert handler.supported_plot_libraries == []

    def test_run_with_artifacts_no_plot_support(self) -> None:
        """Test run_with_artifacts without plot support (lines 142-155)."""
        config = LanguageConfig(
            name="test",
            file_extension=".test",
            execution_commands=["test {file}"],
            package_manager="test-manager",
            plot_detection=None,  # No plot detection
        )
        handler = ConcreteLanguageHandler(config)

        container = Mock()
        container.run.return_value = "result"

        code = "print('hello')"
        result, plots = handler.run_with_artifacts(container, code, libraries=["numpy"], enable_plotting=True)

        # Should call container.run without plot detection
        container.run.assert_called_once_with(code, ["numpy"])
        assert result == "result"
        assert plots == []

    def test_plot_library_enum_coverage(self) -> None:
        """Test PlotLibrary enum (lines 12-25)."""
        # Test that all enum values exist
        assert PlotLibrary.MATPLOTLIB.value == "matplotlib"
        assert PlotLibrary.PLOTLY.value == "plotly"
        assert PlotLibrary.SEABORN.value == "seaborn"
        assert PlotLibrary.CHARTJS.value == "chartjs"
        assert PlotLibrary.D3JS.value == "d3js"
        assert PlotLibrary.JFREECHART.value == "jfreechart"
        assert PlotLibrary.XCHART.value == "xchart"
        assert PlotLibrary.ROOT.value == "root"
        assert PlotLibrary.GONUM_PLOT.value == "gonum_plot"
        assert PlotLibrary.GRUFF.value == "gruff"

    def test_plot_detection_config(self) -> None:
        """Test PlotDetectionConfig dataclass."""
        config = PlotDetectionConfig(
            libraries=[PlotLibrary.MATPLOTLIB],
            setup_code="setup",
            cleanup_code="cleanup",
        )
        assert len(config.libraries) == 1
        assert config.setup_code == "setup"
        assert config.cleanup_code == "cleanup"
