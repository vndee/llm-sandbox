import logging
import re

from llm_sandbox.const import SupportedLanguage
from llm_sandbox.exceptions import LanguageNotSupportPlotError
from llm_sandbox.language_handlers.artifact_detection import PYTHON_PLOT_DETECTION_CODE
from llm_sandbox.language_handlers.base import AbstractLanguageHandler, LanguageConfig, PlotDetectionConfig, PlotLibrary
from llm_sandbox.language_handlers.runtime_context import RuntimeContext


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

    def get_execution_commands(self, code_file: str, runtime_context: RuntimeContext | None = None) -> list[str]:
        """Get execution commands with runtime context paths.

        Args:
            code_file: Path to the Python file to execute
            runtime_context: Optional runtime context containing dynamic paths

        Returns:
            List of commands to execute the Python file

        """
        if runtime_context and runtime_context.python_executable_path:
            return [f"{runtime_context.python_executable_path} {code_file}"]

        # Fall back to static config for backwards compatibility
        return super().get_execution_commands(code_file)

    def get_library_installation_command(self, library: str, runtime_context: RuntimeContext | None = None) -> str:
        """Get library installation command with runtime context paths.

        Args:
            library: Name of the library to install
            runtime_context: Optional runtime context containing dynamic paths

        Returns:
            Command string to install the library

        """
        if runtime_context and runtime_context.pip_executable_path and runtime_context.pip_cache_dir:
            return (
                f"{runtime_context.pip_executable_path} install {library} --cache-dir {runtime_context.pip_cache_dir}"
            )

        # Fall back to static config for backwards compatibility
        return super().get_library_installation_command(library)

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
