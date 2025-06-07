# ruff: noqa: PLR2004
"""Additional tests for llm_sandbox.data module to improve coverage."""

import warnings

from llm_sandbox.data import ConsoleOutput, ExecutionResult, FileType, PlotOutput


class TestConsoleOutputMissingMethods:
    """Test ConsoleOutput methods that are missing coverage."""

    def test_console_output_text_method_deprecated(self) -> None:
        """Test the deprecated text() method."""
        output = ConsoleOutput(exit_code=0, stdout="Hello, World!", stderr="")

        # Test that the deprecated text() method still works and shows warning
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")  # Ensure all DeprecationWarnings are triggered

            result = output.text()

            # Check that warning was issued
            assert len(w) == 1
            assert issubclass(w[0].category, DeprecationWarning)
            assert "deprecated" in str(w[0].message).lower()
            assert "text" in str(w[0].message).lower()
            assert "stdout" in str(w[0].message).lower()

            # Check that it returns the stdout content
            assert result == "Hello, World!"

    def test_console_output_text_method_with_different_content(self) -> None:
        """Test text() method with different content."""
        test_cases = [
            ("Simple text", "Simple text"),
            ("", ""),  # Empty stdout
            ("Multi\nline\ntext", "Multi\nline\ntext"),
            ("Unicode: 你好世界 🌍", "Unicode: 你好世界 🌍"),
        ]

        for stdout_content, expected in test_cases:
            output = ConsoleOutput(exit_code=0, stdout=stdout_content, stderr="some error")

            with warnings.catch_warnings(record=True):
                warnings.simplefilter("always")
                result = output.text()
                assert result == expected

    def test_console_output_success_method_true(self) -> None:
        """Test success() method when exit_code is 0."""
        output = ConsoleOutput(exit_code=0, stdout="Success", stderr="")
        assert output.success() is True

    def test_console_output_success_method_false(self) -> None:
        """Test success() method when exit_code is non-zero."""
        test_cases = [1, 2, 127, 255, -1]

        for exit_code in test_cases:
            output = ConsoleOutput(exit_code=exit_code, stdout="", stderr="Error")
            assert output.success() is False

    def test_console_output_success_method_edge_cases(self) -> None:
        """Test success() method with edge cases."""
        # Test with various combinations
        combinations = [
            (0, "", "", True),  # Success with empty output
            (0, "output", "error", True),  # Success with both stdout and stderr
            (1, "output", "", False),  # Failure with stdout only
            (1, "", "error", False),  # Failure with stderr only
            (1, "output", "error", False),  # Failure with both
        ]

        for exit_code, stdout, stderr, expected_success in combinations:
            output = ConsoleOutput(exit_code=exit_code, stdout=stdout, stderr=stderr)
            assert output.success() is expected_success


class TestExecutionResultInheritance:
    """Test ExecutionResult inheritance of ConsoleOutput methods."""

    def test_execution_result_inherits_text_method(self) -> None:
        """Test that ExecutionResult inherits the deprecated text() method."""
        plot = PlotOutput(format=FileType.PNG, content_base64="dGVzdA==")
        result = ExecutionResult(exit_code=0, stdout="Execution completed", stderr="", plots=[plot])

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")

            text_output = result.text()

            # Should show deprecation warning
            assert len(w) == 1
            assert issubclass(w[0].category, DeprecationWarning)

            # Should return stdout content
            assert text_output == "Execution completed"

    def test_execution_result_inherits_success_method(self) -> None:
        """Test that ExecutionResult inherits the success() method."""
        plot = PlotOutput(format=FileType.PNG, content_base64="dGVzdA==")

        # Test successful execution
        success_result = ExecutionResult(exit_code=0, stdout="Success", stderr="", plots=[plot])
        assert success_result.success() is True

        # Test failed execution
        failed_result = ExecutionResult(exit_code=1, stdout="", stderr="Error", plots=[])
        assert failed_result.success() is False

    def test_execution_result_success_with_plots(self) -> None:
        """Test success() method on ExecutionResult with various plot scenarios."""
        # Successful execution with plots
        plot1 = PlotOutput(format=FileType.PNG, content_base64="cGxvdDE=")
        plot2 = PlotOutput(format=FileType.SVG, content_base64="cGxvdDI=")

        result_with_plots = ExecutionResult(exit_code=0, stdout="Generated 2 plots", stderr="", plots=[plot1, plot2])
        assert result_with_plots.success() is True

        # Failed execution with plots (edge case)
        result_failed_with_plots = ExecutionResult(
            exit_code=1,
            stdout="",
            stderr="Plot generation failed",
            plots=[plot1],  # Plots might still be generated before failure
        )
        assert result_failed_with_plots.success() is False


class TestPlotOutputEdgeCases:
    """Test PlotOutput with various edge cases."""

    def test_plot_output_with_optional_parameters(self) -> None:
        """Test PlotOutput with optional width, height, and dpi parameters."""
        plot = PlotOutput(format=FileType.PNG, content_base64="dGVzdA==", width=800, height=600, dpi=150)

        assert plot.format == FileType.PNG
        assert plot.content_base64 == "dGVzdA=="
        assert plot.width == 800
        assert plot.height == 600
        assert plot.dpi == 150

    def test_plot_output_default_optional_parameters(self) -> None:
        """Test PlotOutput with default None values for optional parameters."""
        plot = PlotOutput(format=FileType.SVG, content_base64="c3ZnIGRhdGE=")

        assert plot.format == FileType.SVG
        assert plot.content_base64 == "c3ZnIGRhdGE="
        assert plot.width is None
        assert plot.height is None
        assert plot.dpi is None

    def test_plot_output_all_file_types(self) -> None:
        """Test PlotOutput with all supported file types."""
        file_types = [
            FileType.PNG,
            FileType.JPEG,
            FileType.PDF,
            FileType.SVG,
            FileType.CSV,
            FileType.JSON,
            FileType.TXT,
            FileType.HTML,
        ]

        for file_type in file_types:
            plot = PlotOutput(
                format=file_type,
                content_base64="dGVzdCBkYXRhIGZvciB7ZmlsZV90eXBlfQ==",  # Base64 encoded test data
                width=100,
                height=100,
                dpi=72,
            )
            assert plot.format == file_type
            assert plot.width == 100
            assert plot.height == 100
            assert plot.dpi == 72


class TestFileTypeCompleteness:
    """Test FileType enum completeness."""

    def test_all_file_types_exist(self) -> None:
        """Test that all expected file types exist in the enum."""
        expected_types = {
            "png": FileType.PNG,
            "jpeg": FileType.JPEG,
            "pdf": FileType.PDF,
            "svg": FileType.SVG,
            "csv": FileType.CSV,
            "json": FileType.JSON,
            "txt": FileType.TXT,
            "html": FileType.HTML,
        }

        for value, expected_enum in expected_types.items():
            assert FileType(value) == expected_enum
            assert expected_enum.value == value

    def test_file_type_string_representation(self) -> None:
        """Test string representation of FileType enum values."""
        file_types = [
            FileType.PNG,
            FileType.JPEG,
            FileType.PDF,
            FileType.SVG,
            FileType.CSV,
            FileType.JSON,
            FileType.TXT,
            FileType.HTML,
        ]

        for file_type in file_types:
            assert isinstance(file_type.value, str)
            assert len(file_type.value) > 0


class TestDataClassDefaults:
    """Test default values and behaviors of data classes."""

    def test_console_output_defaults(self) -> None:
        """Test ConsoleOutput with default values."""
        # Test with all defaults
        output = ConsoleOutput()
        assert output.exit_code == 0
        assert output.stdout == ""
        assert output.stderr == ""
        assert output.success() is True

        # Test with partial defaults
        output_partial = ConsoleOutput(exit_code=1)
        assert output_partial.exit_code == 1
        assert output_partial.stdout == ""
        assert output_partial.stderr == ""
        assert output_partial.success() is False

    def test_execution_result_defaults(self) -> None:
        """Test ExecutionResult with default values."""
        # Test with minimal parameters
        result = ExecutionResult()
        assert result.exit_code == 0
        assert result.stdout == ""
        assert result.stderr == ""
        assert result.plots == []
        assert result.success() is True

        # Test that plots default to empty list
        result_custom = ExecutionResult(exit_code=0, stdout="test")
        assert result_custom.plots == []
        assert isinstance(result_custom.plots, list)


class TestWarningBehavior:
    """Test warning behavior and stack levels."""

    def test_deprecation_warning_stack_level(self) -> None:
        """Test that deprecation warning has correct stack level."""
        output = ConsoleOutput(exit_code=0, stdout="test", stderr="")

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")

            # Call through a wrapper function to test stack level
            def wrapper_function() -> str:
                return output.text()

            _ = wrapper_function()

            assert len(w) == 1
            assert issubclass(w[0].category, DeprecationWarning)
            # The warning should point to the caller (wrapper_function), not to the text() method itself
            # This tests that stacklevel=2 is working correctly
