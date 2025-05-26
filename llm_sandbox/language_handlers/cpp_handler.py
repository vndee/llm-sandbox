from typing import TYPE_CHECKING, Any

from llm_sandbox.artifact import PlotOutput

from .base import AbstractLanguageHandler, LanguageConfig

if TYPE_CHECKING:
    from .base import ContainerProtocol


class CppHandler(AbstractLanguageHandler):
    """Handler for C++."""

    def __init__(self, *args: Any, **kwargs: dict[str, Any]) -> None:
        """Initialize the C++ handler."""
        config = LanguageConfig(
            name="cpp",
            file_extension="cpp",
            execution_commands=["g++ {file}"],
            package_manager="apt-get",
            plot_detection=None,
        )
        super().__init__(config, *args, **kwargs)  # type: ignore[arg-type]

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

    def scan(self, code: str) -> list[str]:
        """Scan the code for safety issues."""
        return []
