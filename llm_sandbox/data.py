"""Data classes for LLM Sandbox."""

import warnings
from dataclasses import dataclass, field
from enum import Enum


class FileType(Enum):
    """File types supported by the plot extractor."""

    PNG = "png"
    JPEG = "jpeg"
    PDF = "pdf"
    SVG = "svg"
    CSV = "csv"
    JSON = "json"
    TXT = "txt"
    HTML = "html"


@dataclass
class PlotOutput:
    """Represents a plot/chart output."""

    format: FileType
    content_base64: str
    width: int | None = None
    height: int | None = None
    dpi: int | None = None


@dataclass
class ConsoleOutput:
    """Console output from code execution."""

    exit_code: int = 0
    stderr: str = ""
    stdout: str = ""

    @property
    def text(self) -> str:
        """Get the text representation of the console output.

        .. deprecated::
            The `text` property is deprecated and will be removed in a future version.
            Use `stdout` attribute directly instead.
        """
        warnings.warn(
            "The 'text' property is deprecated and will be removed in a future version. "
            "Use 'stdout' attribute directly instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.stdout

    @property
    def success(self) -> bool:
        """Check if the execution was successful."""
        return not self.exit_code


@dataclass
class ExecutionResult(ConsoleOutput):
    """Result of code execution in sandbox."""

    plots: list[PlotOutput] = field(default_factory=list)
