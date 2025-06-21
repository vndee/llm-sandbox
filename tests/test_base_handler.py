# ruff: noqa: SLF001, PLR2004

"""Tests for llm_sandbox.language_handlers.base module - missing coverage."""

import io
import logging
import tarfile
from unittest.mock import Mock, patch

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

    def test_extract_plots_exception_handling(self) -> None:
        """Test extract_plots exception handling for lines 140-141."""
        config = LanguageConfig(
            name="test",
            file_extension=".test",
            execution_commands=["test {file}"],
            package_manager="test-manager",
            plot_detection=PlotDetectionConfig(
                libraries=[PlotLibrary.MATPLOTLIB],
                setup_code="setup",
                cleanup_code="cleanup",
            ),
        )
        handler = ConcreteLanguageHandler(config)

        container = Mock()
        # Mock directory exists
        mock_dir_result = Mock()
        mock_dir_result.exit_code = 0

        # Mock files found
        mock_find_result = Mock()
        mock_find_result.exit_code = 0
        mock_find_result.stdout = "/tmp/plots/test.png"

        container.execute_command.side_effect = [mock_dir_result, mock_find_result]

        # Mock _extract_single_plot to raise an exception to trigger lines 140-141
        with patch.object(handler, "_extract_single_plot", side_effect=OSError("Test error")):
            # This should not raise an exception but log it and continue
            plots = handler.extract_plots(container, "/tmp/plots")
            assert plots == []

    def test_extract_plots_runtime_error_handling(self) -> None:
        """Test extract_plots RuntimeError exception handling for lines 140-141."""
        config = LanguageConfig(
            name="test",
            file_extension=".test",
            execution_commands=["test {file}"],
            package_manager="test-manager",
            plot_detection=PlotDetectionConfig(
                libraries=[PlotLibrary.MATPLOTLIB],
                setup_code="setup",
                cleanup_code="cleanup",
            ),
        )
        handler = ConcreteLanguageHandler(config)

        container = Mock()
        # Make execute_command raise RuntimeError to trigger lines 140-141
        container.execute_command.side_effect = RuntimeError("Container error")

        # This should not raise an exception but log it and return empty list
        plots = handler.extract_plots(container, "/tmp/plots")
        assert plots == []

    def test_extract_single_plot_exception_handling(self) -> None:
        """Test _extract_single_plot exception handling for lines 182-183."""
        config = LanguageConfig(
            name="test",
            file_extension=".test",
            execution_commands=["test {file}"],
            package_manager="test-manager",
            plot_detection=PlotDetectionConfig(
                libraries=[PlotLibrary.MATPLOTLIB],
                setup_code="setup",
                cleanup_code="cleanup",
            ),
        )
        handler = ConcreteLanguageHandler(config)

        container = Mock()
        # Mock get_archive to raise exception to trigger lines 182-183
        container.get_archive.side_effect = OSError("Archive error")

        # This should not raise an exception but log it and return None
        result = handler._extract_single_plot(container, "/tmp/plots/test.png")
        assert result is None

    def test_extract_single_plot_tar_error_handling(self) -> None:
        """Test _extract_single_plot TarError exception handling for lines 182-183."""
        config = LanguageConfig(
            name="test",
            file_extension=".test",
            execution_commands=["test {file}"],
            package_manager="test-manager",
            plot_detection=PlotDetectionConfig(
                libraries=[PlotLibrary.MATPLOTLIB],
                setup_code="setup",
                cleanup_code="cleanup",
            ),
        )
        handler = ConcreteLanguageHandler(config)

        container = Mock()

        # Create valid tar data but mock tarfile.open to raise TarError
        valid_tar_data = io.BytesIO()
        container.get_archive.return_value = (valid_tar_data.getvalue(), True)

        with patch("tarfile.open", side_effect=tarfile.TarError("Tar error")):
            # This should not raise an exception but log it and return None
            result = handler._extract_single_plot(container, "/tmp/plots/test.png")
            assert result is None

    def test_extract_single_plot_value_error_handling(self) -> None:
        """Test _extract_single_plot ValueError exception handling for lines 182-183."""
        config = LanguageConfig(
            name="test",
            file_extension=".test",
            execution_commands=["test {file}"],
            package_manager="test-manager",
            plot_detection=PlotDetectionConfig(
                libraries=[PlotLibrary.MATPLOTLIB],
                setup_code="setup",
                cleanup_code="cleanup",
            ),
        )
        handler = ConcreteLanguageHandler(config)

        container = Mock()
        # Mock get_archive to raise ValueError to trigger lines 182-183
        container.get_archive.side_effect = ValueError("Value error")

        # This should not raise an exception but log it and return None
        result = handler._extract_single_plot(container, "/tmp/plots/test.png")
        assert result is None

    def test_extract_single_plot_file_obj_none(self) -> None:
        """Test _extract_single_plot when file_obj is None (lines 180-183)."""
        config = LanguageConfig(
            name="test",
            file_extension=".test",
            execution_commands=["test {file}"],
            package_manager="test-manager",
            plot_detection=PlotDetectionConfig(
                libraries=[PlotLibrary.MATPLOTLIB],
                setup_code="setup",
                cleanup_code="cleanup",
            ),
        )
        handler = ConcreteLanguageHandler(config)

        container = Mock()

        # Create a valid tar with a member that extractfile returns None for
        tar_content = io.BytesIO()
        with tarfile.open(fileobj=tar_content, mode="w") as tar:
            # Add an empty file
            tarinfo = tarfile.TarInfo(name="test.png")
            tarinfo.size = 0
            tar.addfile(tarinfo, io.BytesIO())

        container.get_archive.return_value = (tar_content.getvalue(), True)

        # Mock tar.extractfile to return None
        with patch("tarfile.TarFile.extractfile", return_value=None):
            result = handler._extract_single_plot(container, "/tmp/plots/test.png")
            assert result is None

    def test_extract_single_plot_tarfile_processing_error(self) -> None:
        """Test _extract_single_plot exception handling during file processing (lines 182-183)."""
        config = LanguageConfig(
            name="test",
            file_extension=".test",
            execution_commands=["test {file}"],
            package_manager="test-manager",
            plot_detection=PlotDetectionConfig(
                libraries=[PlotLibrary.MATPLOTLIB],
                setup_code="setup",
                cleanup_code="cleanup",
            ),
        )
        handler = ConcreteLanguageHandler(config)

        container = Mock()

        # Create valid tar content
        tar_content = io.BytesIO()
        with tarfile.open(fileobj=tar_content, mode="w") as tar:
            # Add a file with content
            file_content = b"fake image content"
            tarinfo = tarfile.TarInfo(name="test.png")
            tarinfo.size = len(file_content)
            tar.addfile(tarinfo, io.BytesIO(file_content))

        container.get_archive.return_value = (tar_content.getvalue(), True)

        # Mock the file reading to raise an exception during processing
        with patch("tarfile.TarFile.extractfile") as mock_extractfile:
            mock_file_obj = Mock()
            mock_file_obj.read.side_effect = OSError("Error reading file content")
            mock_extractfile.return_value = mock_file_obj

            # This should trigger the exception handling at lines 182-183
            result = handler._extract_single_plot(container, "/tmp/plots/test.png")
            assert result is None
