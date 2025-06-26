import logging
import re
from typing import TYPE_CHECKING, Any

from llm_sandbox.const import SupportedLanguage
from llm_sandbox.data import PlotOutput

from .base import AbstractLanguageHandler, LanguageConfig

if TYPE_CHECKING:
    from .base import ContainerProtocol


class JavaHandler(AbstractLanguageHandler):
    """Handler for Java."""

    def __init__(self, logger: logging.Logger | None = None) -> None:
        """Initialize the Java handler."""
        super().__init__(logger)

        self.config = LanguageConfig(
            name=SupportedLanguage.JAVA,
            file_extension="java",
            execution_commands=["java {file}"],
            package_manager="mvn",
            plot_detection=None,
            is_support_library_installation=False,
        )

    def inject_plot_detection_code(self, code: str) -> str:
        """Java does not support plot detection directly in this manner."""
        return code

    def run_with_artifacts(
        self,
        container: "ContainerProtocol",
        code: str,
        libraries: list | None = None,
        enable_plotting: bool = True,  # noqa: ARG002
        output_dir: str = "/tmp/sandbox_plots",  # noqa: ARG002
        timeout: int = 30,
    ) -> tuple[Any, list[PlotOutput]]:
        """Run Java code without artifact extraction.

        Java plot detection is not currently supported. This method
        runs the code normally and returns an empty list of plots.

        Future implementations could support:
        - JFreeChart for chart generation
        - XChart for plotting
        - JavaFX for graphics

        Args:
            container: The container protocol instance to run code in
            code: The Java code to execute
            libraries: Optional list of libraries to install before running
            enable_plotting: Whether to enable plot detection (ignored for Java)
            output_dir: Directory where plots should be saved (unused)
            timeout: Timeout for the code execution

        Returns:
            tuple: (execution_result, empty_list_of_plots)

        """
        self.logger.warning("Java does not support plot extraction yet")
        result = container.run(code, libraries, timeout=timeout)
        return result, []

    def get_import_patterns(self, module: str) -> str:
        """Get the regex patterns for import statements.

        Regex to match import statements for the given module/package.
        Covers:
            import module.Class;
            import module.*;
        Handles variations in whitespace.
        Negative lookbehind and lookahead to avoid matching comments or parts of other words.

        Args:
            module (str): The name of the module (package) to get import patterns for.
                        Can be a full path like com.example.Mypackage

        Returns:
            str: The regex patterns for import statements.

        """
        # Java packages can have dots, need to escape them for regex
        # Module can be something like "java.util" or "com.google.common.collect"
        # We want to match "import java.util.List;" or "import java.util.*;"
        # or "import com.google.common.collect.Lists;" or "import com.google.common.collect.*;"
        escaped_module = re.escape(module)
        return rf"(?:^|\s)import\s+{escaped_module}(?:\.\*|\.\w+|\w*);(?=[\s\S]|$)"

    @staticmethod
    def get_multiline_comment_patterns() -> str:
        """Get the regex patterns for multiline comments.

        Regex to match multiline comments.
        Handles variations in whitespace.
        """
        return r"/\*[\s\S]*?\*/"

    @staticmethod
    def get_inline_comment_patterns() -> str:
        """Get the regex patterns for inline comments.

        Regex to match inline comments.
        Handles variations in whitespace.
        """
        return r"//.*$"
