import base64
import io
import logging
import re
import tarfile
from pathlib import Path
from typing import TYPE_CHECKING

from llm_sandbox.const import SupportedLanguage
from llm_sandbox.data import FileType, PlotOutput
from llm_sandbox.exceptions import LanguageNotSupportPlotError
from llm_sandbox.language_handlers.artifact_detection import PYTHON_PLOT_DETECTION_CODE
from llm_sandbox.language_handlers.base import AbstractLanguageHandler, LanguageConfig, PlotDetectionConfig, PlotLibrary

if TYPE_CHECKING:
    from .base import ContainerProtocol


class PythonHandler(AbstractLanguageHandler):
    """Handler for Python language."""

    def __init__(self, logger: logging.Logger | None = None) -> None:
        """Initialize the Python handler."""
        super().__init__()

        self.config = LanguageConfig(
            name=SupportedLanguage.PYTHON,
            file_extension="py",
            execution_commands=["/tmp/venv/bin/python {file}"],
            package_manager="/tmp/venv/bin/pip install --cache-dir /tmp/pip_cache",
            plot_detection=PlotDetectionConfig(
                libraries=[
                    PlotLibrary.MATPLOTLIB,
                    PlotLibrary.PLOTLY,
                    PlotLibrary.SEABORN,
                ],
                setup_code=PYTHON_PLOT_DETECTION_CODE,
                cleanup_code="",
            ),
        )
        self.logger = logger or logging.getLogger(__name__)

    def inject_plot_detection_code(self, code: str) -> str:
        """Inject comprehensive plot detection for Python."""
        if not self.config.plot_detection:
            raise LanguageNotSupportPlotError(self.config.name)

        return (self.config.plot_detection.setup_code or "") + "\n\n" + code

    def extract_plots(self, container: "ContainerProtocol", output_dir: str) -> list[PlotOutput]:
        """Extract plots from Python execution."""
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
            self.logger.exception("Error extracting Python plots")

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
        """Get the regex patterns for import statements.

        Regex to match various import styles for the given module
        Covers:
            import module
            import module as alias
            from module import ...
            from module.submodule import ... (if module is specified like module.submodule)
        Handles variations in whitespace.
        Negative lookbehind and lookahead to avoid matching comments or parts of other words.

        Args:
            module (str): The name of the module to get the import patterns for.

        Returns:
            str: The regex patterns for import statements.

        """
        return (
            rf"(?:^|\s)(?:import\s+{module}(?:\s+as\s+\w+)?|from\s+{module}\s+import\s+(?:\*|\w+(?:\s+as\s+\w+)"
            rf"?(?:,\s*\w+(?:\s+as\s+\w+)?)*))(?=[\s;(#]|$)"
        )

    def filter_comments(self, code: str) -> str:
        """Filter out Python comments from code.

        Handles:
        - Single line comments starting with #
        - Inline comments after code
        - Preserves empty lines for readability

        Args:
            code (str): The code to filter comments from.

        Returns:
            str: The code with comments removed.

        """
        # First remove multi-line comments
        code = re.sub(r"'''[\s\S]*?'''", "", code)
        code = re.sub(r'"""[\s\S]*?"""', "", code)

        filtered_lines = []
        for line in code.split("\n"):
            # Remove inline comments
            line = re.sub(r"#.*$", "", line)
            # Keep the line if it has non-whitespace content
            if line.strip():
                filtered_lines.append(line)
            else:
                # Preserve empty lines for readability
                filtered_lines.append("")
        return "\n".join(filtered_lines)
