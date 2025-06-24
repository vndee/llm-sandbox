import logging
import re

from llm_sandbox.const import SupportedLanguage
from llm_sandbox.exceptions import LanguageNotSupportPlotError, PackageManagerError
from llm_sandbox.language_handlers.artifact_detection import R_PLOT_DETECTION_CODE

from .base import AbstractLanguageHandler, LanguageConfig, PlotDetectionConfig, PlotLibrary


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
                    PlotLibrary.ROOT,  # Base R graphics
                    PlotLibrary.PLOTLY,  # Plotly for R
                ],
                setup_code=R_PLOT_DETECTION_CODE,
                cleanup_code="",
            ),
        )

    def inject_plot_detection_code(self, code: str) -> str:
        """Inject comprehensive plot detection for R."""
        if not self.config.plot_detection:
            raise LanguageNotSupportPlotError(self.config.name)

        return (self.config.plot_detection.setup_code or "") + "\n\n" + code

    def get_import_patterns(self, module: str) -> str:
        """Get the regex patterns for R library loading statements.

        Regex to match library() or require() calls for the given module.
        Covers:
            library(module)
            library("module")
            library('module')
            library(module, quietly = TRUE)
            require(module)
            require("module")
            require('module')
        Handles variations in whitespace, optional additional arguments,
        and avoids matching inside string literals.

        Args:
            module (str): The name of the module to get the import patterns for.

        Returns:
            str: The regex patterns for import statements.

        """
        escaped_module = re.escape(module)
        # Pattern matches library/require calls allowing additional arguments
        # Very restrictive: only matches at start of line or after specific R syntax punctuation
        return rf"(?:^|[(!;,\{{])\s*(?:library|require)\s*\(\s*['\"]?{escaped_module}['\"]?[^)]*\)"

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

    def get_library_installation_command(self, library: str) -> str:
        """Get command to install R library."""
        if not self.config.package_manager:
            raise PackageManagerError(self.config.name)

        return f"R -e \"install.packages('{library}', repos='https://cran.rstudio.com/')\""
