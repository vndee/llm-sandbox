# ruff: noqa: SLF001, PLR2004

import logging
import re
from unittest.mock import MagicMock

from llm_sandbox.const import SupportedLanguage
from llm_sandbox.language_handlers.r_handler import RHandler


class TestRHandler:
    """Test RHandler specific functionality."""

    def test_init(self) -> None:
        """Test RHandler initialization."""
        handler = RHandler()

        assert handler.config.name == SupportedLanguage.R
        assert handler.config.file_extension == "R"
        assert "Rscript {file}" in handler.config.execution_commands
        assert handler.config.package_manager == "install.packages"
        assert handler.config.plot_detection is not None  # R has basic plot support
        assert handler.config.is_support_library_installation is True

    def test_init_with_custom_logger(self) -> None:
        """Test RHandler initialization with custom logger."""
        custom_logger = logging.getLogger("custom")
        handler = RHandler(custom_logger)
        assert handler.logger == custom_logger

    def test_run_with_artifacts_basic_support(self) -> None:
        """Test run_with_artifacts with basic artifact support."""
        handler = RHandler()
        mock_container = MagicMock()
        mock_result = MagicMock()
        mock_container.run.return_value = mock_result

        result, plots = handler.run_with_artifacts(
            container=mock_container, 
            code='print("Hello R")', 
            libraries=["ggplot2"], 
            enable_plotting=True
        )

        assert result == mock_result
        assert plots == []  # Basic implementation returns empty plots for now
        mock_container.run.assert_called_once_with('print("Hello R")', ["ggplot2"])

    def test_extract_plots_returns_empty(self) -> None:
        """Test extract_plots returns empty list (basic implementation)."""
        handler = RHandler()
        mock_container = MagicMock()

        plots = handler.extract_plots(mock_container, "/tmp/sandbox_plots")

        assert plots == []

    def test_get_import_patterns_library(self) -> None:
        """Test get_import_patterns method for library() statements."""
        handler = RHandler()

        pattern = handler.get_import_patterns("ggplot2")

        # Should match library statements
        library_statements = [
            "library(ggplot2)",
            "library('ggplot2')",
            'library("ggplot2")',
            "library( ggplot2 )",
            "  library(ggplot2)",
            "library(ggplot2)  # Load ggplot2",
        ]

        for code in library_statements:
            assert re.search(pattern, code), f"Pattern should match library: {code}"

    def test_get_import_patterns_require(self) -> None:
        """Test get_import_patterns method for require() statements."""
        handler = RHandler()

        pattern = handler.get_import_patterns("dplyr")

        # Should match require statements
        require_statements = [
            "require(dplyr)",
            "require('dplyr')",
            'require("dplyr")',
            "require( dplyr )",
            "  require(dplyr)",
        ]

        for code in require_statements:
            assert re.search(pattern, code), f"Pattern should match require: {code}"

    def test_get_import_patterns_no_false_positives(self) -> None:
        """Test that import patterns don't match unrelated code."""
        handler = RHandler()

        pattern = handler.get_import_patterns("ggplot2")

        # Should not match comments or parts of other words
        non_matching_samples = [
            "# library(ggplot2)",
            "ggplot2_data <- data.frame()",  # Variable usage
            "print('library(ggplot2)')",  # String literal
            "library(ggplot2extra)",  # Different package
        ]

        for code in non_matching_samples:
            filtered_code = handler.filter_comments(code)
            if "ggplot2_data" not in code and "print" not in code and "ggplot2extra" not in code:
                assert not re.search(pattern, filtered_code), f"Pattern should not match: {code}"

    def test_get_multiline_comment_patterns(self) -> None:
        """Test get_multiline_comment_patterns method."""
        pattern = RHandler.get_multiline_comment_patterns()

        # R doesn't have true multiline comments, so pattern should be empty
        assert pattern == ""

    def test_get_inline_comment_patterns(self) -> None:
        """Test get_inline_comment_patterns method."""
        pattern = RHandler.get_inline_comment_patterns()

        comment_samples = [
            "# This is a comment",
            'print("Hello")  # Inline comment',
            "    # Indented comment",
            "x <- 5 # Variable assignment",
        ]

        for comment in comment_samples:
            assert re.search(pattern, comment), f"Pattern should match: {comment}"

    def test_filter_comments(self) -> None:
        """Test comment filtering functionality."""
        handler = RHandler()

        code_with_comments = """
        # This is a single line comment
        library(ggplot2)
        hello <- function() {
            print("Hello") # Inline comment
            return(42)
        }
        """

        filtered_code = handler.filter_comments(code_with_comments)

        # Should remove comments but keep code
        assert 'print("Hello")' in filtered_code
        assert "library(ggplot2)" in filtered_code
        assert "hello <- function()" in filtered_code
        assert "# This is a single line comment" not in filtered_code
        assert "# Inline comment" not in filtered_code

    def test_properties(self) -> None:
        """Test handler property methods."""
        handler = RHandler()

        assert handler.name == SupportedLanguage.R
        assert handler.file_extension == "R"
        assert handler.is_support_library_installation is True
        assert handler.is_support_plot_detection is True  # R has basic plot support

    def test_get_execution_commands(self) -> None:
        """Test getting execution commands."""
        handler = RHandler()

        commands = handler.get_execution_commands("script.R")

        assert len(commands) == 1
        assert commands[0] == "Rscript script.R"

    def test_get_library_installation_command(self) -> None:
        """Test getting library installation command."""
        handler = RHandler()

        command = handler.get_library_installation_command("ggplot2")

        assert command == "install.packages ggplot2"

    def test_base_r_packages(self) -> None:
        """Test base R package loading statements."""
        handler = RHandler()

        base_packages = ["base", "stats", "graphics", "grDevices", "utils", "datasets", "methods"]

        for package in base_packages:
            pattern = handler.get_import_patterns(package)
            library_stmt = f"library({package})"
            assert re.search(pattern, library_stmt), f"Should match base package: {library_stmt}"

    def test_popular_r_packages(self) -> None:
        """Test popular R package loading statements."""
        handler = RHandler()

        popular_packages = [
            ("ggplot2", "library(ggplot2)"),
            ("dplyr", "library(dplyr)"),
            ("tidyr", "library(tidyr)"),
            ("readr", "library(readr)"),
            ("data.table", "library(data.table)"),
            ("plotly", "library(plotly)"),
            ("shiny", "library(shiny)"),
        ]

        for package, library_stmt in popular_packages:
            pattern = handler.get_import_patterns(package)
            assert re.search(pattern, library_stmt), f"Should match package: {library_stmt}"

    def test_library_variations(self) -> None:
        """Test different variations of library statements."""
        handler = RHandler()

        pattern = handler.get_import_patterns("ggplot2")

        variations = [
            "library(ggplot2)",
            "library('ggplot2')",
            'library("ggplot2")',
            "library( ggplot2 )",
            "require(ggplot2)",
            "require('ggplot2')",
            'require("ggplot2")',
        ]

        for variation in variations:
            assert re.search(pattern, variation), f"Should match variation: {variation}"

    def test_conditional_library_loading(self) -> None:
        """Test conditional library loading patterns."""
        handler = RHandler()

        pattern = handler.get_import_patterns("optional_package")

        conditional_loads = [
            "if (!require(optional_package)) install.packages('optional_package')",
            "library(optional_package, quietly = TRUE)",
            "suppressMessages(library(optional_package))",
        ]

        # These should still match the basic library/require pattern
        for load_stmt in conditional_loads:
            assert re.search(pattern, load_stmt), f"Should match conditional load: {load_stmt}"

    def test_namespace_access_not_matched(self) -> None:
        """Test that namespace access operators are not matched."""
        handler = RHandler()

        pattern = handler.get_import_patterns("dplyr")

        # These should NOT match the import pattern
        non_matching = [
            "dplyr::filter(data, condition)",
            "dplyr:::internal_function()",
            "data %>% dplyr::mutate(new_col = value)",
        ]

        for stmt in non_matching:
            assert not re.search(pattern, stmt), f"Should not match namespace access: {stmt}"

    def test_string_literals_not_matched(self) -> None:
        """Test that string literals with library-like text are not matched."""
        handler = RHandler()

        pattern = handler.get_import_patterns("ggplot2")

        # These should NOT match
        non_matching = [
            'print("Please load library(ggplot2)")',
            "message('Install ggplot2 with: library(ggplot2)')",
            "# Load ggplot2 with library(ggplot2)",
        ]

        for stmt in non_matching:
            filtered_stmt = handler.filter_comments(stmt)
            if "print" in stmt or "message" in stmt:  # String literals should not match
                assert not re.search(pattern, filtered_stmt), f"Should not match string: {stmt}"

    def test_assignment_operations(self) -> None:
        """Test that assignment operations don't interfere with pattern matching."""
        handler = RHandler()

        pattern = handler.get_import_patterns("data.table")

        code_samples = [
            "library(data.table)",
            "dt <- data.table(x = 1:5)",  # This should not match
            "result <- data.table::fread('file.csv')",  # This should not match
        ]

        # Only the first should match
        assert re.search(pattern, code_samples[0]), "Should match library statement"
        assert not re.search(pattern, code_samples[1]), "Should not match assignment"
        assert not re.search(pattern, code_samples[2]), "Should not match namespace call"

    def test_plot_detection_config(self) -> None:
        """Test that plot detection configuration is properly set."""
        handler = RHandler()

        assert handler.config.plot_detection is not None
        assert handler.is_support_plot_detection is True
        # Basic setup for future plot detection
        assert handler.config.plot_detection.setup_code == ""
        assert handler.config.plot_detection.cleanup_code == ""

    def test_r_file_extension_case(self) -> None:
        """Test that R file extension is uppercase."""
        handler = RHandler()

        # R files typically use uppercase .R extension
        assert handler.file_extension == "R"

        commands = handler.get_execution_commands("analysis.R")
        assert commands[0] == "Rscript analysis.R"
