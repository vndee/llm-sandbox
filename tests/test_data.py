# ruff: noqa: PLR2004, PT011
"""Tests for data models and exceptions."""

from dataclasses import FrozenInstanceError

import pytest

from llm_sandbox.data import ConsoleOutput, ExecutionResult, FileType, PlotOutput
from llm_sandbox.exceptions import (
    CommandEmptyError,
    CommandFailedError,
    ExtraArgumentsError,
    ImageNotFoundError,
    ImagePullError,
    LanguageHandlerNotInitializedError,
    LanguageNotSupportPlotError,
    LibraryInstallationNotSupportedError,
    MissingDependencyError,
    NotOpenSessionError,
    SecurityViolationError,
    UnsupportedBackendError,
)


class TestConsoleOutput:
    """Test ConsoleOutput dataclass."""

    def test_console_output_creation(self) -> None:
        """Test creating a ConsoleOutput instance."""
        output = ConsoleOutput(exit_code=0, stdout="Hello, World!", stderr="")

        assert output.exit_code == 0
        assert output.stdout == "Hello, World!"
        assert output.stderr == ""

    def test_console_output_with_error(self) -> None:
        """Test creating a ConsoleOutput instance with error."""
        output = ConsoleOutput(exit_code=1, stdout="", stderr="Error occurred")

        assert output.exit_code == 1
        assert output.stdout == ""
        assert output.stderr == "Error occurred"

    def test_console_output_immutable(self) -> None:
        """Test that ConsoleOutput is immutable."""
        output = ConsoleOutput(exit_code=0, stdout="test", stderr="")

        with pytest.raises(FrozenInstanceError):
            output.exit_code = 1

    def test_console_output_equality(self) -> None:
        """Test ConsoleOutput equality comparison."""
        output1 = ConsoleOutput(exit_code=0, stdout="test", stderr="")
        output2 = ConsoleOutput(exit_code=0, stdout="test", stderr="")
        output3 = ConsoleOutput(exit_code=1, stdout="test", stderr="")

        assert output1 == output2
        assert output1 != output3


class TestPlotOutput:
    """Test PlotOutput dataclass."""

    def test_plot_output_creation(self) -> None:
        """Test creating a PlotOutput instance."""
        plot = PlotOutput(format=FileType.PNG, content_base64="dGVzdCBpbWFnZSBkYXRh")

        assert plot.format == FileType.PNG
        assert plot.content_base64 == "dGVzdCBpbWFnZSBkYXRh"

    def test_plot_output_different_formats(self) -> None:
        """Test creating PlotOutput with different file formats."""
        formats_and_content = [
            (FileType.PNG, "cG5nIGRhdGE="),
            (FileType.SVG, "c3ZnIGRhdGE="),
            (FileType.PDF, "cGRmIGRhdGE="),
            (FileType.HTML, "aHRtbCBkYXRh"),
        ]

        for file_format, content in formats_and_content:
            plot = PlotOutput(format=file_format, content_base64=content)
            assert plot.format == file_format
            assert plot.content_base64 == content

    def test_plot_output_immutable(self) -> None:
        """Test that PlotOutput is immutable."""
        plot = PlotOutput(format=FileType.PNG, content_base64="dGVzdA==")

        with pytest.raises(FrozenInstanceError):
            plot.format = FileType.SVG

    def test_plot_output_equality(self) -> None:
        """Test PlotOutput equality comparison."""
        plot1 = PlotOutput(format=FileType.PNG, content_base64="dGVzdA==")
        plot2 = PlotOutput(format=FileType.PNG, content_base64="dGVzdA==")
        plot3 = PlotOutput(format=FileType.SVG, content_base64="dGVzdA==")

        assert plot1 == plot2
        assert plot1 != plot3


class TestExecutionResult:
    """Test ExecutionResult dataclass."""

    def test_execution_result_creation_minimal(self) -> None:
        """Test creating an ExecutionResult with minimal parameters."""
        result = ExecutionResult(exit_code=0, stdout="output", stderr="", plots=[])

        assert result.exit_code == 0
        assert result.stdout == "output"
        assert result.stderr == ""
        assert result.plots == []

    def test_execution_result_with_plots(self) -> None:
        """Test creating an ExecutionResult with plots."""
        plot1 = PlotOutput(format=FileType.PNG, content_base64="cGxvdDE=")
        plot2 = PlotOutput(format=FileType.SVG, content_base64="cGxvdDI=")

        result = ExecutionResult(exit_code=0, stdout="Generated plots", stderr="", plots=[plot1, plot2])

        assert result.exit_code == 0
        assert result.stdout == "Generated plots"
        assert result.stderr == ""
        assert len(result.plots) == 2
        assert result.plots[0] == plot1
        assert result.plots[1] == plot2

    def test_execution_result_with_error(self) -> None:
        """Test creating an ExecutionResult with error."""
        result = ExecutionResult(exit_code=1, stdout="", stderr="Runtime error", plots=[])

        assert result.exit_code == 1
        assert result.stdout == ""
        assert result.stderr == "Runtime error"
        assert result.plots == []

    def test_execution_result_immutable(self) -> None:
        """Test that ExecutionResult is immutable."""
        result = ExecutionResult(exit_code=0, stdout="test", stderr="", plots=[])

        with pytest.raises(FrozenInstanceError):
            result.exit_code = 1


class TestFileType:
    """Test FileType enum."""

    def test_file_type_values(self) -> None:
        """Test FileType enum values."""
        assert FileType.PNG.value == "png"
        assert FileType.SVG.value == "svg"
        assert FileType.PDF.value == "pdf"
        assert FileType.HTML.value == "html"

    def test_file_type_from_string(self) -> None:
        """Test creating FileType from string."""
        assert FileType("png") == FileType.PNG
        assert FileType("svg") == FileType.SVG
        assert FileType("pdf") == FileType.PDF
        assert FileType("html") == FileType.HTML

    def test_file_type_invalid_value(self) -> None:
        """Test creating FileType with invalid value."""
        with pytest.raises(ValueError):
            FileType("invalid")


class TestExceptions:
    """Test custom exceptions."""

    def test_command_empty_error(self) -> None:
        """Test CommandEmptyError exception."""
        with pytest.raises(CommandEmptyError, match="Command cannot be empty"):
            raise CommandEmptyError

    def test_command_failed_error(self) -> None:
        """Test CommandFailedError exception."""
        error = CommandFailedError("ls -l", 1, "Permission denied")

        assert "Command ls -l failed with exit code 1" in str(error)
        assert "Permission denied" in str(error)

    def test_extra_arguments_error(self) -> None:
        """Test ExtraArgumentsError exception."""
        msg = "Too many arguments provided"
        with pytest.raises(ExtraArgumentsError, match="Too many arguments provided"):
            raise ExtraArgumentsError(msg)

    def test_image_not_found_error(self) -> None:
        """Test ImageNotFoundError exception."""
        error = ImageNotFoundError("python:3.11")

        assert "Image python:3.11 not found" in str(error)

    def test_image_pull_error(self) -> None:
        """Test ImagePullError exception."""
        error = ImagePullError("python:3.11", "Network timeout")

        assert "Failed to pull image python:3.11" in str(error)
        assert "Network timeout" in str(error)

    def test_language_handler_not_initialized_error(self) -> None:
        """Test LanguageHandlerNotInitializedError exception."""
        error = LanguageHandlerNotInitializedError("python")

        assert "Language handler for python is not initialized" in str(error)

    def test_language_not_support_plot_error(self) -> None:
        """Test LanguageNotSupportPlotError exception."""
        error = LanguageNotSupportPlotError("java")

        assert "Language java does not support plot detection" in str(error)

    def test_library_installation_not_supported_error(self) -> None:
        """Test LibraryInstallationNotSupportedError exception."""
        error = LibraryInstallationNotSupportedError("assembly")

        assert "Library installation is not supported for assembly" in str(error)

    def test_missing_dependency_error(self) -> None:
        """Test MissingDependencyError exception."""
        error = MissingDependencyError("Docker is required but not installed")

        assert "Docker is required but not installed" in str(error)

    def test_not_open_session_error(self) -> None:
        """Test NotOpenSessionError exception."""
        with pytest.raises(NotOpenSessionError, match="Session is not open"):
            raise NotOpenSessionError

    def test_security_violation_error(self) -> None:
        """Test SecurityViolationError exception."""
        error = SecurityViolationError("Dangerous code detected")

        assert "Dangerous code detected" in str(error)

    def test_unsupported_backend_error(self) -> None:
        """Test UnsupportedBackendError exception."""
        error = UnsupportedBackendError("invalid-backend")

        assert "Unsupported backend: invalid-backend" in str(error)

    def test_exception_inheritance(self) -> None:
        """Test that all custom exceptions inherit from appropriate base classes."""
        # Test that all exceptions are instances of Exception
        exceptions_to_test = [
            CommandEmptyError(),
            CommandFailedError("cmd", 1, "output"),
            ExtraArgumentsError("message"),
            ImageNotFoundError("image"),
            ImagePullError("image", "details"),
            LanguageHandlerNotInitializedError("lang"),
            LanguageNotSupportPlotError("lang"),
            LibraryInstallationNotSupportedError("lang"),
            MissingDependencyError("message"),
            NotOpenSessionError(),
            SecurityViolationError("message"),
            UnsupportedBackendError("backend"),
        ]

        for exception in exceptions_to_test:
            assert isinstance(exception, Exception)
            # Test that they can be raised and caught
            with pytest.raises(type(exception)):
                raise exception
