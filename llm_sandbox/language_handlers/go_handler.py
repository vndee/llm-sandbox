import logging
import re
from typing import TYPE_CHECKING

from llm_sandbox.const import SupportedLanguage
from llm_sandbox.data import PlotOutput

from .base import AbstractLanguageHandler, LanguageConfig

if TYPE_CHECKING:
    from .base import ContainerProtocol


class GoHandler(AbstractLanguageHandler):
    """Handler for Go."""

    def __init__(self, logger: logging.Logger | None = None) -> None:
        """Initialize the Go handler."""
        super().__init__(logger)

        self.config = LanguageConfig(
            name=SupportedLanguage.GO,
            file_extension="go",
            execution_commands=["go run {file}"],
            package_manager="go get",
            plot_detection=None,
        )

    def inject_plot_detection_code(self, code: str) -> str:
        """Go does not support plot detection directly in this manner."""
        return code

    def extract_plots(
        self,
        container: "ContainerProtocol",  # noqa: ARG002
        output_dir: str,  # noqa: ARG002
    ) -> list[PlotOutput]:
        """Go does not support plot extraction in this manner."""
        return []

    def get_import_patterns(self, module: str) -> str:
        """Get the regex patterns for Go language import statements.

        Regex to match import statements for the given module/package.
        Covers:
            import "module"
            import (
                "module"
                alias "module"
            )
        Handles variations in whitespace and comments.
        Negative lookbehind and lookahead to avoid matching comments or parts of other words.

        Args:
            module (str): The name of the module (package) to get import patterns for.
                        Can be a path like "fmt" or "github.com/user/repo".

        Returns:
            str: The regex patterns for import statements.

        """
        escaped_module = re.escape(module)
        # Matches: import "module"  OR  import alias "module" (potentially inside import block)
        # The lookbehind (?<![\w\d_]) and lookahead (?![\w\d_]) ensure "module" is not part of a larger identifier.
        return (
            rf'(?:^|\s)import\s+(?:\(\s*(?:\w+\s+)?"(?:[^"]*?/)'
            rf'?{escaped_module}"(?:\s*//[^\n]*)?|\w*\s*"(?:[^"]*?/)?{escaped_module}")(?=["\s\(\)]|$)'
        )

    def filter_comments(self, code: str) -> str:
        """Filter out Go comments from code.

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
