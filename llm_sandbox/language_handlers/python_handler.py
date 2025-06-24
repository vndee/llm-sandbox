import logging
import re

from llm_sandbox.const import SupportedLanguage
from llm_sandbox.exceptions import LanguageNotSupportPlotError
from llm_sandbox.language_handlers.artifact_detection import PYTHON_PLOT_DETECTION_CODE
from llm_sandbox.language_handlers.base import AbstractLanguageHandler, LanguageConfig, PlotDetectionConfig, PlotLibrary


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
            r"\s*(from\s+" + re.escape(module) + r"(?:\s|$|\.|import)|import\s+" + re.escape(module) + r"(?:\s|$|\.))"
        )

    @staticmethod
    def get_multiline_comment_patterns() -> str:
        """Get the regex patterns for multiline comments.

        Regex to match multiline comments.
        Handles variations in whitespace.
        """
        return r"'''[\s\S]*?'''"

    @staticmethod
    def get_inline_comment_patterns() -> str:
        """Get the regex patterns for inline comments.

        Regex to match inline comments.
        Handles variations in whitespace.
        """
        return r"#.*$"
