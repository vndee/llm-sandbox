import logging
import re
from typing import TYPE_CHECKING

from llm_sandbox.const import SupportedLanguage
from llm_sandbox.data import PlotOutput

from .base import AbstractLanguageHandler, LanguageConfig

if TYPE_CHECKING:
    from .base import ContainerProtocol


class CppHandler(AbstractLanguageHandler):
    """Handler for C++ language."""

    def __init__(self, logger: logging.Logger | None = None) -> None:
        """Initialize the C++ handler."""
        super().__init__(logger)

        self.config = LanguageConfig(
            name=SupportedLanguage.CPP,
            file_extension="cpp",
            execution_commands=["g++ -std=c++17 {file} -o /tmp/a.out && /tmp/a.out"],
            package_manager="apt-get install",
            plot_detection=None,
        )

    def get_execution_commands(self, code_file: str) -> list[str]:
        """Get the execution commands for the C++ handler."""
        return [f"g++ -o a.out {code_file}", "./a.out"]

    def get_library_installation_command(self, library: str) -> str:
        """Get the library installation command for the C++ handler."""
        return f"apt-get install {library}"

    def inject_plot_detection_code(self, code: str) -> str:
        """Inject plot detection code for the C++ handler."""
        return code

    def extract_plots(
        self,
        container: "ContainerProtocol",  # noqa: ARG002
        output_dir: str,  # noqa: ARG002
    ) -> list[PlotOutput]:
        """Extract plots from the C++ handler."""
        return []

    def scan(self, code: str) -> list[str]:  # noqa: ARG002
        """Scan the code for safety issues."""
        return []

    def get_import_patterns(self, module: str) -> str:
        """Get the regex patterns for import statements.

        Regex to match #include directives for the given module.
        Covers:
            #include <module>
            #include "module"
        Handles variations in whitespace.
        Negative lookbehind and lookahead to avoid matching comments or parts of other words.

        Args:
            module (str): The name of the module to get the import patterns for.

        Returns:
            str: The regex patterns for import statements.

        """
        # For C++, module often refers to header files.
        # The pattern handles <header.h> or "header.h"
        # It also considers variations with .h, .hpp, or no extension if specified in module string
        module_name = re.escape(module)
        return rf'(?:^|\s)#include\s*(?:<{module_name}>|"{module_name}")(?=[\s;(#]|//|/\*|$)'

    @staticmethod
    def get_multiline_comment_patterns() -> str:
        """Get the regex patterns for multiline comments.

        Regex to match multiline comments.
        Handles variations in whitespace.

        Returns:
            str: The regex patterns for multiline comments.

        """
        return r"/\*[\s\S]*?\*/"

    @staticmethod
    def get_single_line_comment_patterns() -> str:
        """Get the regex patterns for single line comments.

        Regex to match single line comments.
        Handles variations in whitespace.

        Returns:
            str: The regex patterns for single line comments.

        """
        return r"//.*$"
