from typing import TYPE_CHECKING, Any

from llm_sandbox.artifact import PlotOutput

from .base import (
    AbstractLanguageHandler,
    LanguageConfig,
    PlotDetectionConfig,
    PlotLibrary,
)

if TYPE_CHECKING:
    from .base import ContainerProtocol


class JavaScriptHandler(AbstractLanguageHandler):
    """Handler for JavaScript/NodeJS."""

    def __init__(self, *args: Any, **kwargs: dict[str, Any]) -> None:
        """Initialize the JavaScript handler."""
        config = LanguageConfig(
            name="javascript",
            file_extension="js",
            execution_commands=["node {file}"],
            package_manager="yarn add",
            plot_detection=PlotDetectionConfig(
                libraries=[PlotLibrary.CHARTJS, PlotLibrary.D3JS, PlotLibrary.PLOTLY],
                output_formats=["png", "svg", "html"],
                detection_patterns=["chart.save(", "d3.select(", "Plotly.newPlot("],
                cleanup_code="",
            ),
            supported_libraries=["chart.js", "d3", "plotly.js", "canvas", "jsdom"],
        )
        super().__init__(config, *args, **kwargs)

    def inject_plot_detection_code(self, code: str) -> str:
        """Inject plot detection for JavaScript."""
        return code

    def extract_plots(
        self,
        container: "ContainerProtocol",
        output_dir: str,  # noqa: ARG002
    ) -> list[PlotOutput]:
        """Extract plots from JavaScript execution."""
        return self._extract_files_from_path(container, "/tmp/sandbox_plots")

    def safety_check(self, code: str) -> list[str]:
        """Check the code for safety issues."""
