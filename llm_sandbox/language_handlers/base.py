import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum

from llm_sandbox.exceptions import CommandFailedError, PackageManagerError


class PlotLibrary(Enum):
    """Plot libraries supported by the language handler."""

    MATPLOTLIB = "matplotlib"
    PLOTLY = "plotly"
    SEABORN = "seaborn"
    CHARTJS = "chartjs"
    D3JS = "d3js"
    JFREECHART = "jfreechart"
    XCHART = "xchart"
    ROOT = "root"
    GONUM_PLOT = "gonum_plot"
    GRUFF = "gruff"


@dataclass
class PlotDetectionConfig:
    """Configuration for plot detection."""

    libraries: list[PlotLibrary]
    output_formats: list[str]
    detection_patterns: list[str]
    setup_code: str | None = None
    cleanup_code: str | None = None


@dataclass
class LanguageConfig:
    """Language-specific configuration."""

    name: str
    file_extension: str
    execution_commands: list[str]
    package_manager: str
    supported_libraries: list[str] | None = None
    is_support_library_installation: bool = True
    plot_detection: PlotDetectionConfig | None = None


class AbstractLanguageHandler(ABC):
    """Abstract base class for language-specific handlers."""

    def __init__(
        self, config: LanguageConfig, logger: logging.Logger | None = None
    ) -> None:
        """Initialize the language handler."""
        self.config = config
        self.logger = logger or logging.getLogger(__name__)
        self.plot_outputs = []
        self.file_outputs = []

    def get_execution_commands(self, code_file: str) -> list[str]:
        """Get commands to execute code file."""
        if not self.config.execution_commands:
            raise CommandFailedError(self.config.name)
        return [
            command.format(file=code_file) for command in self.config.execution_commands
        ]

    def get_library_installation_command(self, library: str) -> str:
        """Get command to install library."""
        if not self.config.package_manager:
            raise PackageManagerError(self.config.name)
        return f"{self.config.package_manager} {library}"

    @abstractmethod
    def inject_plot_detection_code(self, code: str) -> str:
        """Inject code to detect and capture plots."""

    @abstractmethod
    def safety_check(self, code: str) -> list[str]:
        """Check the code for safety issues."""

    @property
    def name(self) -> str:
        """Get name of the language."""
        return self.config.name

    @property
    def file_extension(self) -> str:
        """Get file extension for language."""
        return self.config.file_extension

    @property
    def supported_libraries(self) -> list[str]:
        """Get supported libraries for language."""
        return self.config.supported_libraries

    @property
    def supported_plot_libraries(self) -> list[PlotLibrary]:
        """Get supported plotting libraries."""
        return self.config.plot_detection.libraries

    @property
    def is_support_library_installation(self) -> bool:
        """Get if the language supports library installation."""
        return self.config.is_support_library_installation
