import logging
from typing import TYPE_CHECKING

from llm_sandbox.data import PlotOutput

from .base import AbstractLanguageHandler, LanguageConfig, PlotDetectionConfig, PlotLibrary

if TYPE_CHECKING:
    from .base import ContainerProtocol


class JavaScriptHandler(AbstractLanguageHandler):
    """Handler for JavaScript/NodeJS."""

    def __init__(self, logger: logging.Logger | None = None) -> None:
        """Initialize the JavaScript handler."""
        super().__init__()

        self.config = LanguageConfig(
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
        self.logger = logger or logging.getLogger(__name__)

    def inject_plot_detection_code(self, code: str) -> str:
        """Inject plot detection for JavaScript."""
        return code

    def extract_plots(
        self,
        container: "ContainerProtocol",
        output_dir: str,
    ) -> list[PlotOutput]:
        """Extract plots from JavaScript execution."""
        raise NotImplementedError

    def scan(self, code: str) -> list[str]:  # noqa: ARG002
        """Check the code for safety issues."""
        return []
