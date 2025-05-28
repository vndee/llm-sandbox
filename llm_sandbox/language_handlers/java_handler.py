import logging
import re
from typing import TYPE_CHECKING

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

    def extract_plots(
        self,
        container: "ContainerProtocol",  # noqa: ARG002
        output_dir: str,  # noqa: ARG002
    ) -> list[PlotOutput]:
        """Java does not support plot extraction in this manner."""
        return []

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

    def filter_comments(self, code: str) -> str:
        """Filter out Java comments from code.

        Handles:
        - Single line comments starting with //
        - Multi-line comments between /* and */
        - Preserves empty lines for readability

        Args:
            code (str): The code to filter comments from.

        Returns:
            str: The code with comments removed.

        """
        # First remove multi-line comments
        code = re.sub(r"/\*[\s\S]*?\*/", "", code)

        # Then handle single-line comments
        filtered_lines = []
        for line in code.split("\n"):
            # Remove inline comments
            line = re.sub(r"//.*$", "", line)
            # Keep the line if it has non-whitespace content
            if line.strip():
                filtered_lines.append(line)
            else:
                # Preserve empty lines for readability
                filtered_lines.append("")
        return "\n".join(filtered_lines)
