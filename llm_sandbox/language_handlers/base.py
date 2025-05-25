from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, Protocol

from llm_sandbox.artifact import FileOutput, PlotOutput
from llm_sandbox.base import ConsoleOutput
from llm_sandbox.exceptions import CommandFailedError, PackageManagerError

if TYPE_CHECKING:

    class ContainerProtocol(Protocol):
        """Protocol for container objects (Docker, Podman, K8s)."""

        def execute_command(
            self, command: str, workdir: str | None = None
        ) -> "ConsoleOutput":
            """Execute a command in the container."""
            ...

        def get_archive(self, path: str) -> tuple:
            """Get archive of files from container."""
            ...


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

    def __init__(self, config: LanguageConfig) -> None:
        """Initialize the language handler."""
        self.config = config
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
    def extract_plots(
        self, container: "ContainerProtocol", output_dir: str
    ) -> list[PlotOutput]:
        """Extract plots from execution."""

    @abstractmethod
    def extract_files(
        self, container: "ContainerProtocol", output_dir: str
    ) -> list[FileOutput]:
        """Extract files from execution."""

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
