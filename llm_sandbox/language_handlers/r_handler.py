import logging
import re
from typing import TYPE_CHECKING, Any

from llm_sandbox.const import SupportedLanguage
from llm_sandbox.data import PlotOutput

from .base import AbstractLanguageHandler, LanguageConfig, PlotDetectionConfig, PlotLibrary

if TYPE_CHECKING:
    from .base import ContainerProtocol


class RHandler(AbstractLanguageHandler):
    """Handler for R programming language."""

    def __init__(self, logger: logging.Logger | None = None) -> None:
        """Initialize the R handler."""
        super().__init__(logger)
        self.config = LanguageConfig(
            name=SupportedLanguage.R,
            file_extension="R",
            execution_commands=["Rscript {file}"],
            package_manager="install.packages",
            plot_detection=PlotDetectionConfig(
                libraries=[
                    PlotLibrary.ROOT,  # For basic R plotting
                ],
                setup_code="",  # Future: Add R plot detection code
                cleanup_code="",
            ),
        )

    def run_with_artifacts(
        self,
        container: "ContainerProtocol",
        code: str,
        libraries: list | None = None,
        enable_plotting: bool = True,  # noqa: ARG002
        output_dir: str = "/tmp/sandbox_plots",  # noqa: ARG002
    ) -> tuple[Any, list[PlotOutput]]:
        """Run R code with basic artifact extraction support.

        R supports rich plotting capabilities through base R graphics,
        ggplot2, plotly, and other specialized packages. This implementation
        provides a foundation for future plot detection and extraction.

        Popular R plotting libraries include:
        - Base R graphics (plot, hist, boxplot, etc.)
        - ggplot2 for grammar of graphics
        - plotly for interactive plots
        - lattice for trellis graphics
        - shiny for interactive web applications

        Args:
            container: The container protocol instance to run code in
            code: The R code to execute
            libraries: Optional list of libraries to install before running
            enable_plotting: Whether to enable plot detection (basic support)
            output_dir: Directory where plots should be saved

        Returns:
            tuple: (execution_result, list_of_plots)

        """
        # For now, run code without advanced plot detection
        # Future implementations can add comprehensive R plot extraction
        self.logger.info("Running R code with basic plot support")
        result = container.run(code, libraries)
        return result, []

    def get_import_patterns(self, module: str) -> str:
        """Get the regex patterns for R library loading statements.

        Regex to match library() or require() calls for the given module.
        Covers:
            library(module)
            library("module")
            library('module')
            require(module)
            require("module")
            require('module')
        Handles variations in whitespace and optional parentheses.

        Args:
            module (str): The name of the module to get the import patterns for.

        Returns:
            str: The regex patterns for import statements.

        """
        escaped_module = re.escape(module)
        # Pattern matches library/require calls with quotes or without
        return rf"(?:library|require)\s*\(\s*['\"]?{escaped_module}['\"]?\s*\)"

    @staticmethod
    def get_multiline_comment_patterns() -> str:
        """Get the regex patterns for multiline comments.

        R doesn't have native multiline comments, but roxygen2 style
        comments are commonly used for documentation.
        """
        # R doesn't have true multiline comments, returning empty pattern
        return r""

    @staticmethod
    def get_inline_comment_patterns() -> str:
        """Get the regex patterns for inline comments.

        Regex to match inline comments in R.
        Handles variations in whitespace.
        """
        return r"#.*$"
