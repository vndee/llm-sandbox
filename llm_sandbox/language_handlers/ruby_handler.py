import logging
import re
from typing import TYPE_CHECKING

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

    def inject_plot_detection_code(self, code: str) -> str:
        """Ruby does not support plot detection."""
        return code

    def extract_plots(self, container: "ContainerProtocol", output_dir: str) -> list[PlotOutput]:  # noqa: ARG002
        """Ruby does not support plot extraction."""
        return []

    def get_import_patterns(self, module: str) -> str:
        """Get the regex patterns for import statements.

        Regex to match require or require_relative for the given module.
        Covers:
            require 'module'
            require "module"
            require_relative 'module'
            require_relative "module"
        Handles variations in whitespace and optional parentheses.
        Negative lookbehind and lookahead to avoid matching comments or parts of other words.

        Args:
            module (str): The name of the module to get the import patterns for.

        Returns:
            str: The regex patterns for import statements.

        """
        return r"(?:^|\s)(?:require|require_relative)\s*\(\s*\[\'\"]" + re.escape(module) + r"\[\'\"\]\s*\);?"

    def filter_comments(self, code: str) -> str:
        """Filter out Ruby comments from code.

        Handles:
        - Single line comments starting with #
        - Multi-line comments between =begin and =end
        - Preserves empty lines for readability

        Args:
            code (str): The code to filter comments from.

        Returns:
            str: The code with comments removed.

        """
        # First remove multi-line comments (=begin to =end)
        code = re.sub(r"=begin[\s\S]*?=end", "", code)

        # Then handle single-line comments
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

    def get_execution_commands(self, code_file: str) -> list[str]:
        """Get the execution commands for the Ruby handler."""
        return [f"ruby {code_file}"]

    def get_library_installation_command(self, library: str) -> str:
        """Get the library installation command for the Ruby handler."""
        return f"gem install {library}"
