from typing import TYPE_CHECKING

from llm_sandbox.artifact import FileOutput, PlotOutput

from .base import AbstractLanguageHandler, LanguageConfig

if TYPE_CHECKING:
    from .base import ContainerProtocol


class GoHandler(AbstractLanguageHandler):
    """Handler for Go."""

    def __init__(self) -> None:
        """Initialize the Go handler."""
        config = LanguageConfig(
            name="go",
            file_extension="go",
            execution_commands=["go run {file}"],
            package_manager="go get",
            plot_detection=None,
        )
        super().__init__(config)

    def get_execution_commands(self, code_file: str) -> list[str]:
        """Get the execution commands for the Go handler."""
        return [f"go run {code_file}"]

    def get_library_installation_command(self, library: str) -> str:
        """Get the library installation command for the Go handler."""
        return f"go get {library}"

    def inject_plot_detection_code(self, code: str) -> str:
        """Inject plot detection code for the Go handler."""
        return code

    def extract_plots(
        self,
        container: "ContainerProtocol",  # noqa: ARG002
        output_dir: str,  # noqa: ARG002
    ) -> list[PlotOutput]:
        """Extract plots from the Go handler."""
        return []

    def extract_files(
        self,
        container: "ContainerProtocol",  # noqa: ARG002
        output_dir: str,  # noqa: ARG002
    ) -> list[FileOutput]:
        """Extract files from the Go handler."""
        return []
