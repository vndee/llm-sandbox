import base64
import json
import os
from typing import TYPE_CHECKING

from llm_sandbox.artifact import ChartData, FileOutput, FileType, PlotOutput
from llm_sandbox.const import SupportedLanguage
from llm_sandbox.language_handlers.artifact_detection import PYTHON_PLOT_DETECTION_CODE

from .base import (
    AbstractLanguageHandler,
    LanguageConfig,
    PlotDetectionConfig,
    PlotLibrary,
)

if TYPE_CHECKING:
    from .base import ContainerProtocol


class PythonHandler(AbstractLanguageHandler):
    """Handler for Python language."""

    def __init__(self) -> None:
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
        super().__init__(config)

    def inject_plot_detection_code(self, code: str) -> str:
        """Inject comprehensive plot detection for Python."""
        return self.config.plot_detection.setup_code + "\n\n" + code

    def extract_plots(
        self, container: "ContainerProtocol", output_dir: str
    ) -> list[PlotOutput]:
        """Extract plots from Python execution."""
        plots = []

        try:
            result = container.execute_command("test -d /tmp/sandbox_plots")
            if result.exit_code:
                return plots

            result = container.execute_command(
                "find /tmp/sandbox_plots -name '*.png' -o -name '*.svg' -o -name '*.pdf' -o -name '*.html'"
            )
            if result.exit_code:
                return plots

            file_paths = result.stdout.strip().split("\n")
            file_paths = [path.strip() for path in file_paths if path.strip()]

            for file_path in file_paths:
                try:
                    plot_output = self._extract_single_plot(container, file_path)
                    if plot_output:
                        plots.append(plot_output)
                except Exception as e:
                    print(f"Error extracting plot {file_path}: {e}")

        except Exception as e:
            print(f"Error extracting Python plots: {e}")

        return plots

    def _extract_single_plot(
        self, container: "ContainerProtocol", file_path: str
    ) -> PlotOutput | None:
        """Extract single plot file from container."""
        try:
            # Get file content
            bits, stat = container.get_archive(file_path)

            import io
            import tarfile

            tarstream = io.BytesIO(b"".join(bits))
            with tarfile.open(fileobj=tarstream, mode="r") as tar:
                member = tar.getmembers()[0]
                if member.isfile():
                    file_obj = tar.extractfile(member)
                    if file_obj:
                        content = file_obj.read()

                        # Get file info
                        filename = os.path.basename(file_path)
                        file_ext = os.path.splitext(filename)[1].lower().lstrip(".")

                        # Try to get metadata
                        metadata_path = file_path.replace(f".{file_ext}", "_meta.json")
                        chart_data = self._extract_metadata(container, metadata_path)

                        return PlotOutput(
                            format=FileType(file_ext)
                            if file_ext in ["png", "svg", "pdf"]
                            else FileType.PNG,
                            content_base64=base64.b64encode(content).decode("utf-8"),
                            chart_data=chart_data,
                        )

        except Exception as e:
            print(f"Error extracting single plot: {e}")

        return None

    def _extract_metadata(
        self, container: "ContainerProtocol", metadata_path: str
    ) -> ChartData | None:
        """Extract chart metadata if available"""
        try:
            result = container.execute_command(f"cat {metadata_path}")
            if result.exit_code == 0:
                metadata = json.loads(result.stdout)
                return ChartData(
                    chart_type=metadata.get("chart_type", "unknown"),
                    title=metadata.get("title"),
                    metadata=metadata,
                )
        except:
            pass

        return None

    def extract_files(
        self, container: "ContainerProtocol", output_dir: str
    ) -> list[FileOutput]:
        """Extract files from Python execution."""
        return self._extract_files_from_path(container, "/tmp/sandbox_output")

    def _extract_files_from_path(
        self, container: "ContainerProtocol", path: str
    ) -> list[FileOutput]:
        """Extract files from a specific path in the container."""
        files = []

        try:
            # Check if directory exists
            result = container.execute_command(f"test -d {path}")
            if result.exit_code != 0:
                return files

            # Find all files in the directory
            result = container.execute_command(f"find {path} -type f")
            if result.exit_code != 0:
                return files

            file_paths = result.stdout.strip().split("\n")
            file_paths = [path.strip() for path in file_paths if path.strip()]

            for file_path in file_paths:
                try:
                    file_output = self._extract_single_file(container, file_path)
                    if file_output:
                        files.append(file_output)
                except Exception as e:
                    print(f"Error extracting file {file_path}: {e}")

        except Exception as e:
            print(f"Error extracting files from {path}: {e}")

        return files

    def _extract_single_file(
        self, container: "ContainerProtocol", file_path: str
    ) -> FileOutput | None:
        """Extract single file from container."""
        try:
            # Get file content
            bits, stat = container.get_archive(file_path)

            import io
            import mimetypes
            import tarfile

            tarstream = io.BytesIO(b"".join(bits))
            with tarfile.open(fileobj=tarstream, mode="r") as tar:
                member = tar.getmembers()[0]
                if member.isfile():
                    file_obj = tar.extractfile(member)
                    if file_obj:
                        content = file_obj.read()

                        # Get file info
                        filename = os.path.basename(file_path)
                        file_ext = os.path.splitext(filename)[1].lower().lstrip(".")
                        mime_type, _ = mimetypes.guess_type(file_path)

                        try:
                            file_type = FileType(file_ext)
                        except ValueError:
                            file_type = FileType.TXT

                        return FileOutput(
                            filename=filename,
                            content_base64=base64.b64encode(content).decode("utf-8"),
                            file_type=file_type,
                            mime_type=mime_type or "application/octet-stream",
                            size=len(content),
                        )

        except Exception as e:
            print(f"Error extracting single file: {e}")

        return None

    def safety_check(self, code: str) -> list[str]:
        """Check the code for safety issues."""
        warnings = []

        # Check for potentially dangerous imports
        dangerous_imports = [
            "subprocess",
            "os.system",
            "eval",
            "exec",
            "compile",
            "__import__",
            "open",
            "file",
            "input",
            "raw_input",
        ]

        for dangerous in dangerous_imports:
            if dangerous in code:
                warnings.append(
                    f"Potentially dangerous operation detected: {dangerous}"
                )

        # Check for file system operations
        file_operations = ["open(", "file(", "os.remove", "os.unlink", "shutil."]
        for op in file_operations:
            if op in code:
                warnings.append(f"File system operation detected: {op}")

        # Check for network operations
        network_operations = ["urllib", "requests", "socket", "http"]
        for op in network_operations:
            if op in code:
                warnings.append(f"Network operation detected: {op}")

        return warnings
