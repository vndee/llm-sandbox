# ruff: noqa: SLF001, PLR2004

import base64
import io
import logging
import re
import tarfile
from unittest.mock import MagicMock, patch

import pytest

from llm_sandbox.const import SupportedLanguage
from llm_sandbox.data import FileType, PlotOutput
from llm_sandbox.exceptions import LanguageNotSupportPlotError
from llm_sandbox.language_handlers.base import PlotLibrary
from llm_sandbox.language_handlers.python_handler import PythonHandler


class TestPythonHandler:
    """Test PythonHandler specific functionality."""

    def test_init(self) -> None:
        """Test PythonHandler initialization."""
        handler = PythonHandler()

        assert handler.config.name == SupportedLanguage.PYTHON
        assert handler.config.file_extension == "py"
        assert "/tmp/venv/bin/python {file}" in handler.config.execution_commands
        assert handler.config.package_manager == "/tmp/venv/bin/pip install --cache-dir /tmp/pip_cache"
        assert handler.config.plot_detection is not None
        assert PlotLibrary.MATPLOTLIB in handler.config.plot_detection.libraries
        assert PlotLibrary.PLOTLY in handler.config.plot_detection.libraries
        assert PlotLibrary.SEABORN in handler.config.plot_detection.libraries

    def test_init_with_custom_logger(self) -> None:
        """Test PythonHandler initialization with custom logger."""
        custom_logger = logging.getLogger("custom")
        handler = PythonHandler(custom_logger)
        assert handler.logger == custom_logger

    def test_inject_plot_detection_code(self) -> None:
        """Test plot detection code injection."""
        handler = PythonHandler()
        code = "print('hello')"

        injected_code = handler.inject_plot_detection_code(code)

        assert "print('hello')" in injected_code
        assert len(injected_code) > len(code)  # Should have additional setup code

    def test_inject_plot_detection_code_no_support(self) -> None:
        """Test plot detection code injection when not supported."""
        handler = PythonHandler()
        handler.config.plot_detection = None

        with pytest.raises(LanguageNotSupportPlotError):
            handler.inject_plot_detection_code("print('hello')")

    def test_run_with_artifacts_plotting_enabled(self) -> None:
        """Test run_with_artifacts with plotting enabled."""
        handler = PythonHandler()
        mock_container = MagicMock()
        mock_result = MagicMock()
        mock_container.run.return_value = mock_result

        with (
            patch.object(handler, "inject_plot_detection_code") as mock_inject,
            patch.object(handler, "extract_plots") as mock_extract,
        ):
            mock_inject.return_value = "injected_code"
            mock_extract.return_value = []

            result, plots = handler.run_with_artifacts(
                container=mock_container,
                code="print('hello')",
                libraries=["numpy"],
                enable_plotting=True,
                timeout=30,
                output_dir="/tmp/sandbox_plots",
            )

            assert result == mock_result
            assert plots == []
            mock_inject.assert_called_once_with("print('hello')")
            mock_container.run.assert_called_once_with("injected_code", ["numpy"], 30)
            mock_extract.assert_called_once_with(mock_container, "/tmp/sandbox_plots")

    def test_run_with_artifacts_plotting_disabled(self) -> None:
        """Test run_with_artifacts with plotting disabled."""
        handler = PythonHandler()
        mock_container = MagicMock()
        mock_result = MagicMock()
        mock_container.run.return_value = mock_result

        result, plots = handler.run_with_artifacts(
            container=mock_container,
            code="print('hello')",
            libraries=["numpy"],
            enable_plotting=False,
            timeout=30,
            output_dir="/tmp/sandbox_plots",
        )

        assert result == mock_result
        assert plots == []
        mock_container.run.assert_called_once_with("print('hello')", ["numpy"], 30)

    def test_extract_plots_no_directory(self) -> None:
        """Test extract_plots when output directory doesn't exist."""
        handler = PythonHandler()
        mock_container = MagicMock()
        mock_container.execute_command.return_value = MagicMock(exit_code=1)

        plots = handler.extract_plots(mock_container, "/tmp/sandbox_plots")

        assert plots == []
        mock_container.execute_command.assert_called_once_with("test -d /tmp/sandbox_plots")

    def test_extract_plots_no_files(self) -> None:
        """Test extract_plots when no plot files found."""
        handler = PythonHandler()
        mock_container = MagicMock()
        mock_container.execute_command.side_effect = [
            MagicMock(exit_code=0),  # Directory exists
            MagicMock(exit_code=1),  # No files found
        ]

        plots = handler.extract_plots(mock_container, "/tmp/sandbox_plots")

        assert plots == []

    def test_extract_plots_success(self) -> None:
        """Test extract_plots with successful file extraction."""
        handler = PythonHandler()
        mock_container = MagicMock()
        mock_container.execute_command.side_effect = [
            MagicMock(exit_code=0),  # Directory exists
            MagicMock(exit_code=0, stdout="/tmp/sandbox_plots/plot1.png\n/tmp/sandbox_plots/plot2.svg"),
        ]

        with patch.object(handler, "_extract_single_plot") as mock_extract_single:
            mock_plot1 = PlotOutput(format=FileType.PNG, content_base64="dGVzdDE=")
            mock_plot2 = PlotOutput(format=FileType.SVG, content_base64="dGVzdDI=")
            mock_extract_single.side_effect = [mock_plot1, mock_plot2]

            plots = handler.extract_plots(mock_container, "/tmp/sandbox_plots")

            assert len(plots) == 2
            assert plots[0] == mock_plot1
            assert plots[1] == mock_plot2

            # Should be called with sorted file paths
            calls = mock_extract_single.call_args_list
            assert calls[0][0] == (mock_container, "/tmp/sandbox_plots/plot1.png")
            assert calls[1][0] == (mock_container, "/tmp/sandbox_plots/plot2.svg")

    def test_extract_plots_with_exception(self) -> None:
        """Test extract_plots handling exceptions."""
        handler = PythonHandler()
        mock_container = MagicMock()
        mock_container.execute_command.side_effect = RuntimeError("Connection failed")

        with patch.object(handler.logger, "exception") as mock_log:
            plots = handler.extract_plots(mock_container, "/tmp/sandbox_plots")

            assert plots == []
            mock_log.assert_called_once_with("Error extracting %s plots", handler.config.name)

    def test_extract_single_plot_success(self) -> None:
        """Test _extract_single_plot with successful extraction."""
        handler = PythonHandler()
        mock_container = MagicMock()

        # Create a mock tar file content
        png_content = b"fake png data"
        tar_buffer = io.BytesIO()
        with tarfile.open(fileobj=tar_buffer, mode="w") as tar:
            info = tarfile.TarInfo("plot.png")
            info.size = len(png_content)
            tar.addfile(info, io.BytesIO(png_content))
        tar_data = tar_buffer.getvalue()

        mock_container.get_archive.return_value = (tar_data, {"size": len(tar_data)})

        plot = handler._extract_single_plot(mock_container, "/tmp/sandbox_plots/plot.png")

        assert plot is not None
        assert plot.format == FileType.PNG
        assert plot.content_base64 == base64.b64encode(png_content).decode("utf-8")

    def test_extract_single_plot_no_stat(self) -> None:
        """Test _extract_single_plot when get_archive returns no stat."""
        handler = PythonHandler()
        mock_container = MagicMock()
        mock_container.get_archive.return_value = (b"data", None)

        plot = handler._extract_single_plot(mock_container, "/tmp/sandbox_plots/plot.png")

        assert plot is None

    def test_extract_single_plot_empty_tar(self) -> None:
        """Test _extract_single_plot with empty tar file."""
        handler = PythonHandler()
        mock_container = MagicMock()

        # Create empty tar
        tar_buffer = io.BytesIO()
        with tarfile.open(fileobj=tar_buffer, mode="w"):
            pass
        tar_data = tar_buffer.getvalue()

        mock_container.get_archive.return_value = (tar_data, {"size": len(tar_data)})

        plot = handler._extract_single_plot(mock_container, "/tmp/sandbox_plots/plot.png")

        assert plot is None

    def test_extract_single_plot_exception(self) -> None:
        """Test _extract_single_plot handling exceptions."""
        handler = PythonHandler()
        mock_container = MagicMock()
        mock_container.get_archive.side_effect = OSError("File not found")

        with patch.object(handler.logger, "exception") as mock_log:
            plot = handler._extract_single_plot(mock_container, "/tmp/sandbox_plots/plot.png")

            assert plot is None
            mock_log.assert_called_once_with("Error extracting single plot")

    def test_get_import_patterns(self) -> None:
        """Test get_import_patterns method."""
        handler = PythonHandler()

        pattern = handler.get_import_patterns("os")

        # Should match various import formats
        import_code_samples = [
            "import os",
            "import os as operating_system",
            "from os import path",
            "from os.path import join",
            "from os import path, environ",
        ]

        for code in import_code_samples:
            assert re.search(pattern, code), f"Pattern should match: {code}"

        # Should not match comments or parts of other words
        non_matching_samples = [
            "# import os",
            "import osmodule",
            "from myos import something",
        ]

        for code in non_matching_samples:
            filtered_code = handler.filter_comments(code)
            assert not re.search(pattern, filtered_code), f"Pattern should not match: {code}"

    def test_get_multiline_comment_patterns(self) -> None:
        """Test get_multiline_comment_patterns method."""
        pattern = PythonHandler.get_multiline_comment_patterns()

        comment_samples = [
            "'''This is a comment'''",
            "'''\nMultiline\ncomment\n'''",
            "'''Single line with content'''",
        ]

        for comment in comment_samples:
            assert re.search(pattern, comment), f"Pattern should match: {comment}"

    def test_get_inline_comment_patterns(self) -> None:
        """Test get_inline_comment_patterns method."""
        pattern = PythonHandler.get_inline_comment_patterns()

        comment_samples = [
            "# This is a comment",
            "print('hello')  # Inline comment",
            "    # Indented comment",
        ]

        for comment in comment_samples:
            assert re.search(pattern, comment), f"Pattern should match: {comment}"
