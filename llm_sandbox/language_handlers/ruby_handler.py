from .base import AbstractLanguageHandler, LanguageConfig


class RubyHandler(AbstractLanguageHandler):
    """Handler for Ruby."""

    def __init__(self) -> None:
        """Initialize the C++ handler."""
        config = LanguageConfig(
            name="ruby",
            file_extension="rb",
            execution_commands=["ruby {file}"],
            package_manager="gem",
            plot_detection=None,
        )
        super().__init__(config)

    def get_execution_commands(self, code_file: str) -> list[str]:
        """Get the execution commands for the Ruby handler."""
        return [f"ruby {code_file}"]

    def get_library_installation_command(self, library: str) -> str:
        """Get the library installation command for the Ruby handler."""
        return f"gem install {library}"
