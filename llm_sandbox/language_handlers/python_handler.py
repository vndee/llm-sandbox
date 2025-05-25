import base64
import io
import tarfile
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol

from llm_sandbox.artifact import FileType, PlotOutput
from llm_sandbox.const import SupportedLanguage
from llm_sandbox.language_handlers.artifact_detection import PYTHON_PLOT_DETECTION_CODE

from .base import (
    AbstractLanguageHandler,
    LanguageConfig,
    PlotDetectionConfig,
    PlotLibrary,
)

if TYPE_CHECKING:

    class ContainerProtocol(Protocol):
        """Protocol for container objects (Docker, Podman, K8s)."""

        def execute_command(self, command: str, workdir: str | None = None) -> Any:
            """Execute a command in the container."""
            ...

        def get_archive(self, path: str) -> tuple:
            """Get archive of files from container."""
            ...


class PythonHandler(AbstractLanguageHandler):
    """Handler for Python language."""

    def __init__(self, *args: Any, **kwargs: dict[str, Any]) -> None:
        """Initialize the Python handler."""
        config = LanguageConfig(
            name=SupportedLanguage.PYTHON,
            file_extension="py",
            execution_commands=["/tmp/venv/bin/python {file}"],
            package_manager="/tmp/venv/bin/pip install --cache-dir /tmp/pip_cache",
            plot_detection=PlotDetectionConfig(
                libraries=[
                    PlotLibrary.MATPLOTLIB,
                    PlotLibrary.PLOTLY,
                    PlotLibrary.SEABORN,
                ],
                output_formats=["png", "svg", "pdf", "html"],
                detection_patterns=[
                    "plt.show()",
                    "plt.savefig(",
                    "fig.write_html(",
                    "fig.write_image(",
                ],
                setup_code=PYTHON_PLOT_DETECTION_CODE,
                cleanup_code="plt.close('all')",
            ),
        )
        super().__init__(config, *args, **kwargs)

    def inject_plot_detection_code(self, code: str) -> str:
        """Inject comprehensive plot detection for Python."""
        return self.config.plot_detection.setup_code + "\n\n" + code

    def extract_plots(
        self, container: "ContainerProtocol", output_dir: str
    ) -> list[PlotOutput]:
        """Extract plots from Python execution."""
        plots = []

        try:
            result = container.execute_command(f"test -d {output_dir}")
            if result.exit_code:
                return plots

            result = container.execute_command(
                f"find {output_dir} -name '*.png' -o -name '*.svg' -o -name '*.pdf' -o -name '*.html'"  # noqa: E501
            )
            if result.exit_code:
                return plots

            file_paths = result.stdout.strip().split("\n")
            file_paths = [path.strip() for path in file_paths if path.strip()]

            for file_path in sorted(file_paths):
                try:
                    plot_output = self._extract_single_plot(container, file_path)
                    if plot_output:
                        plots.append(plot_output)
                except (OSError, tarfile.TarError, ValueError):
                    self.logger.exception("Error extracting plot %s", file_path)

        except (OSError, RuntimeError):
            self.logger.exception("Error extracting Python plots")

        return plots

    def _extract_single_plot(
        self, container: "ContainerProtocol", file_path: str
    ) -> PlotOutput | None:
        """Extract single plot file from container."""
        try:
            bits, stat = container.get_archive(file_path)
            if not stat:
                return None

            with tarfile.open(fileobj=io.BytesIO(bits), mode="r") as tar:
                members = tar.getmembers()
                if not members:
                    return None

                target_filename = Path(file_path).name
                target_member = None

                for member in members:
                    if member.isfile() and Path(member.name).name == target_filename:
                        target_member = member
                        break

                if not target_member:
                    for member in members:
                        if member.isfile():
                            target_member = member
                            break

                if target_member:
                    file_obj = tar.extractfile(target_member)
                    if file_obj:
                        content = file_obj.read()

                        # Get file info
                        filename = Path(file_path).name
                        file_ext = Path(filename).suffix.lower().lstrip(".")

                        return PlotOutput(
                            format=FileType(file_ext)
                            if file_ext in ["png", "svg", "pdf", "html"]
                            else FileType.PNG,
                            content_base64=base64.b64encode(content).decode("utf-8"),
                        )

        except (OSError, tarfile.TarError, ValueError):
            self.logger.exception("Error extracting single plot")

        return None

    def safety_check(self, code: str) -> list[str]:
        """Check the code for safety issues."""
