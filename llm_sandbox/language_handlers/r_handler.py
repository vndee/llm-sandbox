import base64
import io
import logging
import re
import tarfile
from pathlib import Path
from typing import TYPE_CHECKING, Any

from llm_sandbox.const import SupportedLanguage
from llm_sandbox.data import FileType, PlotOutput
from llm_sandbox.exceptions import LanguageNotSupportPlotError
from llm_sandbox.language_handlers.artifact_detection import R_PLOT_DETECTION_CODE

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

    def run_with_artifacts(
        self,
        container: "ContainerProtocol",
        code: str,
        libraries: list | None = None,
        enable_plotting: bool = True,
        output_dir: str = "/tmp/sandbox_plots",
    ) -> tuple[Any, list[PlotOutput]]:
        """Run R code and extract artifacts (plots) with R-specific logic.

        This method provides R-specific implementation for running code
        with artifact extraction, including base R graphics, ggplot2, plotly,
        and lattice support.

        Args:
            container: The container protocol instance to run code in
            code: The R code to execute
            libraries: Optional list of libraries to install before running
            enable_plotting: Whether to enable plot detection and extraction
            output_dir: Directory where plots should be saved

        Returns:
            tuple: (execution_result, list_of_plots)

        """
        plots: list[PlotOutput] = []

        if enable_plotting and self.is_support_plot_detection:
            # Inject R-specific plot detection code
            injected_code = self.inject_plot_detection_code(code)

            # Run the code with plot detection
            result = container.run(injected_code, libraries)

            # Extract plots using R-specific logic
            plots = self.extract_plots(container, output_dir)

            return result, plots
        # Run code without plot detection
        result = container.run(code, libraries)
        return result, plots

    def extract_plots(self, container: "ContainerProtocol", output_dir: str) -> list[PlotOutput]:
        """Extract plots from R execution."""
        plots: list[PlotOutput] = []

        try:
            result = container.execute_command(f"test -d {output_dir}")
            if result.exit_code:
                return plots

            result = container.execute_command(
                f"find {output_dir} -name '*.png' -o -name '*.svg' -o -name '*.pdf' -o -name '*.html'"
            )
            if result.exit_code:
                return plots

            file_paths = result.stdout.strip().split("\n")
            file_paths = [path.strip() for path in file_paths if path.strip()]

            for file_path in sorted(file_paths):
                try:
                    plot_output = self._extract_single_plot(container, file_path)
                    if plot_output:
                        plots.append(plot_output)
                except (OSError, tarfile.TarError, ValueError):
                    self.logger.exception("Error extracting plot %s", file_path)

        except (OSError, RuntimeError):
            self.logger.exception("Error extracting R plots")

        return plots

    def _extract_single_plot(self, container: "ContainerProtocol", file_path: str) -> PlotOutput | None:
        """Extract single plot file from container."""
        try:
            bits, stat = container.get_archive(file_path)
            if not stat:
                return None

            with tarfile.open(fileobj=io.BytesIO(bits), mode="r") as tar:
                members = tar.getmembers()
                if not members:
                    return None

                target_filename = Path(file_path).name
                target_member = None

                for member in members:
                    if member.isfile() and Path(member.name).name == target_filename:
                        target_member = member
                        break

                if not target_member:
                    for member in members:
                        if member.isfile():
                            target_member = member
                            break

                if target_member:
                    file_obj = tar.extractfile(target_member)
                    if file_obj:
                        content = file_obj.read()

                        # Get file info
                        filename = Path(file_path).name
                        file_ext = Path(filename).suffix.lower().lstrip(".")

                        return PlotOutput(
                            format=FileType(file_ext) if file_ext in ["png", "svg", "pdf", "html"] else FileType.PNG,
                            content_base64=base64.b64encode(content).decode("utf-8"),
                        )

        except (OSError, tarfile.TarError, ValueError):
            self.logger.exception("Error extracting single plot")

        return None

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
