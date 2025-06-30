import logging
import re
from typing import TYPE_CHECKING, Any

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

    def run_with_artifacts(
        self,
        container: "ContainerProtocol",
        code: str,
        libraries: list | None = None,
        enable_plotting: bool = True,  # noqa: ARG002
        output_dir: str = "/tmp/sandbox_plots",  # noqa: ARG002
        timeout: int = 30,
    ) -> tuple[Any, list[PlotOutput]]:
        """Run Go code without artifact extraction.

        Go plot detection is not currently supported. This method
        runs the code normally and returns an empty list of plots.

        Future implementations could support:
        - Gonum/plot for scientific plotting
        - go-chart for chart generation
        - Custom plotting libraries that generate image files

        Args:
            container: The container protocol instance to run code in
            code: The Go code to execute
            libraries: Optional list of libraries to install before running
            enable_plotting: Whether to enable plot detection (ignored for Go)
            output_dir: Directory where plots should be saved (unused)
            timeout: Timeout for the code execution

        Returns:
            tuple: (execution_result, empty_list_of_plots)

        """
        self.logger.warning("Go does not support plot extraction yet")
        result = container.run(code, libraries, timeout=timeout)
        return result, []

    def get_import_patterns(self, module: str) -> str:
        """Get the regex patterns for Go language import statements.

        Regex to match import statements for the given module/package.
        Covers:
            import "module"
            import alias "module"
            import . "module"
            import _ "module"
            import (
                "module"
                alias "module"
            )
        Handles variations in whitespace and comments.

        Args:
            module (str): The name of the module (package) to get import patterns for.
                        Can be a path like "fmt" or "github.com/user/repo".

        Returns:
            str: The regex patterns for import statements.

        """
        escaped_module = re.escape(module)

        # Pattern for single import: import [alias] "module"
        # Alias can be identifier, dot, or underscore
        single_import = rf'import\s+(?:[a-zA-Z_][a-zA-Z0-9_]*\s+|[._]\s+)?"(?:[^"]*?/)?{escaped_module}"'

        # Pattern for import block: import ( ... "module" ... )
        # This matches the module within an import block, with optional alias
        block_import = (
            rf'import\s*\(\s*(?:[^)]*?(?:[a-zA-Z_][a-zA-Z0-9_]*\s+|[._]\s+)?"(?:[^"]*?/)?{escaped_module}"[^)]*?)\s*\)'
        )

        return rf"(?:{single_import}|{block_import})"

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
