import logging
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, Any, Protocol

from llm_sandbox.data import PlotOutput

if TYPE_CHECKING:

    class ContainerProtocol(Protocol):
        """Protocol for container objects (Docker, Podman, K8s)."""

        def execute_command(self, command: str, workdir: str | None = None) -> Any:
            """Execute a command in the container."""
            ...

        def get_archive(self, path: str) -> tuple:
            """Get archive of files from container."""
            ...

        def run(self, code: str, libraries: list | None = None) -> Any:
            """Run code in the container."""
            ...


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
    setup_code: str
    cleanup_code: str


@dataclass
class LanguageConfig:
    """Language-specific configuration."""

    name: str
    file_extension: str
    execution_commands: list[str]
    package_manager: str | None
    is_support_library_installation: bool = True
    plot_detection: PlotDetectionConfig | None = None


class AbstractLanguageHandler(ABC):
    """Abstract base class for language-specific handlers."""

    def __init__(self, logger: logging.Logger | None = None) -> None:
        """Initialize the language handler."""
        self.config: LanguageConfig
        self.logger: logging.Logger = logger or logging.getLogger(__name__)

    def get_execution_commands(self, code_file: str) -> list[str]:
        """Get commands to execute code file."""
        if not self.config.execution_commands:
            raise CommandFailedError(self.config.name, 1, "No execution commands found")
        return [command.format(file=code_file) for command in self.config.execution_commands]

    def get_library_installation_command(self, library: str) -> str:
        """Get command to install library."""
        if not self.config.package_manager:
            raise PackageManagerError(self.config.name)
        return f"{self.config.package_manager} {library}"

    def inject_plot_detection_code(self, code: str) -> str:
        """Inject code to detect and capture plots.

        Subclasses should override this method to provide custom plot detection code.

        Args:
            code: The code to inject plot detection code into.

        Returns:
            The code with plot detection code injected.

        """
        return code

    def extract_plots(self, container: "ContainerProtocol", output_dir: str) -> list[PlotOutput]:  # noqa: ARG002
        """Extract plots from the code.

        Subclasses should override this method to provide custom plot extraction logic.

        Args:
            container: The container protocol instance to run code in
            output_dir: Directory where plots should be saved

        Returns:
            list[PlotOutput]: List of plot outputs

        """
        return []

    def run_with_artifacts(
        self,
        container: "ContainerProtocol",
        code: str,
        libraries: list | None = None,
        enable_plotting: bool = True,
        output_dir: str = "/tmp/sandbox_plots",
    ) -> tuple[Any, list[PlotOutput]]:
        """Run code and extract artifacts (plots) in a language-specific manner.

        This method provides a language-specific implementation for running code
        with artifact extraction. Languages that support plot detection can override
        this method to provide custom artifact extraction logic.

        Args:
            container: The container protocol instance to run code in
            code: The code to execute
            libraries: Optional list of libraries to install before running
            enable_plotting: Whether to enable plot detection and extraction
            output_dir: Directory where plots should be saved

        Returns:
            tuple: (execution_result, list_of_plots)

        """
        # Default implementation for languages without plot support
        if enable_plotting and self.is_support_plot_detection:
            # Inject plot detection code
            injected_code = self.inject_plot_detection_code(code)

            # Run the code with plot detection
            result = container.run(injected_code, libraries)

            # Extract plots
            plots = self.extract_plots(container, output_dir)

            return result, plots
        # Run code without plot detection
        result = container.run(code, libraries)
        return result, []

    @abstractmethod
    def get_import_patterns(self, module: str) -> str:
        """Get the regex patterns for import statements."""

    @staticmethod
    @abstractmethod
    def get_multiline_comment_patterns() -> str:
        """Get the regex patterns for multiline comment."""

    @staticmethod
    @abstractmethod
    def get_inline_comment_patterns() -> str:
        """Get the regex for inline comment patterns."""

    def filter_comments(self, code: str) -> str:
        """Filter out comments from code in a language-specific way.

        Args:
            code (str): The code to filter comments from.

        Returns:
            str: The code with comments removed.

        """
        # First remove multi-line comments
        code = re.sub(self.get_multiline_comment_patterns(), "", code)

        # Then handle single-line comments
        filtered_lines = []
        for line in code.split("\n"):
            # Remove inline comments
            clean_line = re.sub(self.get_inline_comment_patterns(), "", line)
            # Keep the line if it has non-whitespace content
            if clean_line.strip():
                filtered_lines.append(clean_line)
            else:
                # Preserve empty lines for readability
                filtered_lines.append("")
        return "\n".join(filtered_lines)

    @property
    def name(self) -> str:
        """Get name of the language."""
        return self.config.name

    @property
    def file_extension(self) -> str:
        """Get file extension for language."""
        return self.config.file_extension

    @property
    def supported_plot_libraries(self) -> list[PlotLibrary]:
        """Get supported plotting libraries."""
        if not self.config.plot_detection:
            return []
        return self.config.plot_detection.libraries

    @property
    def is_support_library_installation(self) -> bool:
        """Get if the language supports library installation."""
        return self.config.is_support_library_installation

    @property
    def is_support_plot_detection(self) -> bool:
        """Get if the language supports plot detection."""
        return self.config.plot_detection is not None
