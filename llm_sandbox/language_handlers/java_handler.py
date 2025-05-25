from typing import TYPE_CHECKING, Any

from llm_sandbox.artifact import PlotOutput

from .base import AbstractLanguageHandler, LanguageConfig

if TYPE_CHECKING:
    from .base import ContainerProtocol


class JavaHandler(AbstractLanguageHandler):
    """Handler for Java."""

    def __init__(self, *args: Any, **kwargs: dict[str, Any]) -> None:
        """Initialize the Java handler."""
        config = LanguageConfig(
            name="java",
            file_extension="java",
            execution_commands=["java {file}"],
            package_manager="mvn",
            plot_detection=None,
            is_support_library_installation=False,
        )
        super().__init__(config, *args, **kwargs)

    def get_execution_commands(self, code_file: str) -> list[str]:
        """Get the execution commands for the Java handler."""
        return [f"java {code_file}"]

    def get_library_installation_command(self, library: str) -> str:
        """Get the library installation command for the Java handler."""
        return f"mvn install:install-file -Dfile={library}"

    def inject_plot_detection_code(self, code: str) -> str:
        """Inject plot detection code for the Java handler."""
        return code

    def extract_plots(
        self,
        container: "ContainerProtocol",  # noqa: ARG002
        output_dir: str,  # noqa: ARG002
    ) -> list[PlotOutput]:
        """Extract plots from the Go handler."""
        return []

    def safety_check(self, code: str) -> list[str]:
        """Check the code for safety issues."""
