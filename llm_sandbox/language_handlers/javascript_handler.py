import logging
import re
from typing import TYPE_CHECKING, Any

from llm_sandbox.const import SupportedLanguage
from llm_sandbox.data import PlotOutput

from .base import AbstractLanguageHandler, LanguageConfig

if TYPE_CHECKING:
    from .base import ContainerProtocol


class JavaScriptHandler(AbstractLanguageHandler):
    """Handler for JavaScript/NodeJS."""

    def __init__(self, logger: logging.Logger | None = None) -> None:
        """Initialize the JavaScript handler."""
        super().__init__(logger)

        self.config = LanguageConfig(
            name=SupportedLanguage.JAVASCRIPT,
            file_extension="js",
            execution_commands=["node {file}"],
            package_manager="npm install",
            plot_detection=None,
        )

    def inject_plot_detection_code(self, code: str) -> str:
        """JavaScript plot detection is not directly supported here.

        Consider client-side libraries or specific Node.js plotting libraries.
        """
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
        """Run JavaScript code without artifact extraction.

        JavaScript plot detection is not currently supported. This method
        runs the code normally and returns an empty list of plots.

        Future implementations could support:
        - Chart.js for client-side plotting
        - D3.js for data visualization
        - Puppeteer for server-side rendering
        - Canvas-based plotting libraries

        Args:
            container: The container protocol instance to run code in
            code: The JavaScript code to execute
            libraries: Optional list of libraries to install before running
            enable_plotting: Whether to enable plot detection (ignored for JavaScript)
            output_dir: Directory where plots should be saved (unused)
            timeout: Timeout for the code execution

        Returns:
            tuple: (execution_result, empty_list_of_plots)

        """
        self.logger.warning("JavaScript does not support plot extraction yet")
        result = container.run(code, libraries, timeout=timeout)
        return result, []

    def get_import_patterns(self, module: str) -> str:
        """Get the regex patterns for JavaScript import statements.

        Regex to match import (ES6) and require (CommonJS) statements.
        Covers:
            import ... from 'module';
            import ... from "module";
            const ... = require('module');
            const ... = require("module");
            require('module')
        Handles variations in whitespace, aliasing, and destructured imports.

        Args:
            module (str): The name of the module to get import patterns for.

        Returns:
            str: The regex patterns for import statements.

        """
        escaped_module = re.escape(module)
        # Pattern for ES6 imports: import ... from "module" or 'module'
        es6_pattern = r"import(?:\s+.*\s+from)?\s*['\"]" + escaped_module + r"['\"];?"
        # Pattern for CommonJS requires: require("module") or require('module')
        commonjs_pattern = r"require\s*\(\s*['\"]" + escaped_module + r"[\'\"]\s*\);?"
        return r"(?:" + es6_pattern + r"|" + commonjs_pattern + r")"

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
