import logging
import re
from typing import TYPE_CHECKING, Any

from llm_sandbox.const import SupportedLanguage
from llm_sandbox.data import PlotOutput

from .base import AbstractLanguageHandler, LanguageConfig

if TYPE_CHECKING:
    from .base import ContainerProtocol


class RubyHandler(AbstractLanguageHandler):
    """Handler for Ruby."""

    def __init__(self, logger: logging.Logger | None = None) -> None:
        """Initialize the Ruby handler."""
        super().__init__(logger)
        self.config = LanguageConfig(
            name=SupportedLanguage.RUBY,
            file_extension="rb",
            execution_commands=["ruby {file}"],
            package_manager="gem install",
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
        """Run Ruby code without artifact extraction.

        Ruby plot detection is not currently supported. This method
        runs the code normally and returns an empty list of plots.

        Future implementations could support:
        - Gruff for chart generation
        - Ruby bindings for plotting libraries
        - Custom plotting libraries that generate image files

        Args:
            container: The container protocol instance to run code in
            code: The Ruby code to execute
            libraries: Optional list of libraries to install before running
            enable_plotting: Whether to enable plot detection (ignored for Ruby)
            output_dir: Directory where plots should be saved (unused)
            timeout: Timeout for the code execution

        Returns:
            tuple: (execution_result, empty_list_of_plots)

        """
        self.logger.warning("Ruby does not support plot extraction yet")
        result = container.run(code, libraries, timeout=timeout)
        return result, []

    def get_import_patterns(self, module: str) -> str:
        """Get the regex patterns for import statements.

        Regex to match require or require_relative for the given module.
        Covers:
            require 'module'
            require "module"
            require 'module/submodule'  # For bundler-style requires
            require_relative 'module'
            require_relative "module"
        Handles variations in whitespace and optional parentheses.
        Uses negative lookbehind to avoid matching inside string literals.

        Args:
            module (str): The name of the module to get the import patterns for.

        Returns:
            str: The regex patterns for import statements.

        """
        escaped_module = re.escape(module)
        # Pattern matches require/require_relative with negative lookbehind for string context
        # (?<![^\\s]") avoids matching when preceded by a quote (not preceded by whitespace + quote)
        # The module can be followed by / for submodules (bundler-style)
        return rf"(?<![^\s\"'])(?:^|\s)(?:require|require_relative)\s*\(?\s*['\"](?:{escaped_module}(?:/[^'\"]*)?)['\"]"

    @staticmethod
    def get_multiline_comment_patterns() -> str:
        """Get the regex patterns for multiline comments.

        Regex to match multiline comments.
        Handles variations in whitespace.
        """
        return r"=begin[\s\S]*?=end"

    @staticmethod
    def get_inline_comment_patterns() -> str:
        """Get the regex patterns for inline comments.

        Regex to match inline comments.
        Handles variations in whitespace.
        """
        return r"#.*$"
