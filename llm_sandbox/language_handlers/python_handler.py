import base64
import json
import os
from typing import TYPE_CHECKING

from llm_sandbox.artifact import ChartData, FileOutput, FileType, PlotOutput
from llm_sandbox.const import SupportedLanguage

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
                setup_code="",
                cleanup_code="plt.close('all')",
            ),
            supported_libraries=["matplotlib", "plotly", "seaborn", "pandas", "numpy"],
        )
        super().__init__(config)

    def inject_plot_detection_code(self, code: str) -> str:
        """Inject comprehensive plot detection for Python."""
        setup_code = """
# Multi-library plot detection setup
import os
import sys
import base64
import json
from pathlib import Path

# Setup output directories
os.makedirs('/tmp/sandbox_plots', exist_ok=True)
os.makedirs('/tmp/sandbox_output', exist_ok=True)

# Global plot counter
_plot_counter = 0

# === MATPLOTLIB SUPPORT ===
try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    _original_show = plt.show
    _original_savefig = plt.savefig

    def _enhanced_show(*args, **kwargs):
        global _plot_counter
        try:
            fig = plt.gcf()
            if fig and fig.get_axes():
                # Save in multiple formats
                for fmt in ['png', 'svg', 'pdf']:
                    filename = f'/tmp/sandbox_plots/matplotlib_plot_{_plot_counter}.{fmt}'
                    fig.savefig(filename, format=fmt, dpi=100, bbox_inches='tight')

                # Save metadata
                metadata = {
                    'library': 'matplotlib',
                    'plot_id': _plot_counter,
                    'title': fig._suptitle.get_text() if fig._suptitle else None,
                    'size': fig.get_size_inches().tolist(),
                    'axes_count': len(fig.get_axes())
                }

                with open(f'/tmp/sandbox_plots/matplotlib_plot_{_plot_counter}_meta.json', 'w') as f:
                    json.dump(metadata, f)

                _plot_counter += 1
        except Exception as e:
            print(f"Matplotlib capture error: {e}")
        finally:
            plt.clf()

    def _enhanced_savefig(filename, *args, **kwargs):
        global _plot_counter
        result = _original_savefig(filename, *args, **kwargs)

        try:
            # Copy to our output directory
            import shutil
            base_name = Path(filename).stem
            ext = Path(filename).suffix
            output_file = f'/tmp/sandbox_plots/matplotlib_saved_{_plot_counter}{ext}'
            shutil.copy2(filename, output_file)
            _plot_counter += 1
        except Exception as e:
            print(f"Matplotlib savefig capture error: {e}")

        return result

    plt.show = _enhanced_show
    plt.savefig = _enhanced_savefig

except ImportError:
    pass

# === PLOTLY SUPPORT ===
try:
    import plotly.graph_objects as go
    import plotly.offline as pyo
    from plotly import io as pio

    _original_write_html = None
    _original_write_image = None

    # Monkey patch Figure methods
    def _enhanced_write_html(self, file, *args, **kwargs):
        global _plot_counter

        # Call original method
        if hasattr(go.Figure, '_original_write_html'):
            result = self._original_write_html(file, *args, **kwargs)
        else:
            result = super(go.Figure, self).write_html(file, *args, **kwargs)

        try:
            # Copy to our output directory
            import shutil
            output_file = f'/tmp/sandbox_plots/plotly_plot_{_plot_counter}.html'
            shutil.copy2(file, output_file)

            # Save as PNG too if possible
            try:
                png_file = f'/tmp/sandbox_plots/plotly_plot_{_plot_counter}.png'
                self.write_image(png_file)
            except:
                pass

            _plot_counter += 1
        except Exception as e:
            print(f"Plotly HTML capture error: {e}")

        return result

    def _enhanced_write_image(self, file, *args, **kwargs):
        global _plot_counter

        # Call original method
        if hasattr(go.Figure, '_original_write_image'):
            result = self._original_write_image(file, *args, **kwargs)
        else:
            result = super(go.Figure, self).write_image(file, *args, **kwargs)

        try:
            # Copy to our output directory
            import shutil
            ext = Path(file).suffix
            output_file = f'/tmp/sandbox_plots/plotly_img_{_plot_counter}{ext}'
            shutil.copy2(file, output_file)
            _plot_counter += 1
        except Exception as e:
            print(f"Plotly image capture error: {e}")

        return result

    # Apply patches
    go.Figure._original_write_html = go.Figure.write_html
    go.Figure._original_write_image = go.Figure.write_image
    go.Figure.write_html = _enhanced_write_html
    go.Figure.write_image = _enhanced_write_image

except ImportError:
    pass

# === SEABORN SUPPORT ===
try:
    import seaborn as sns
    # Seaborn uses matplotlib backend, so it's already covered
    print("Seaborn plotting enabled via matplotlib backend")
except ImportError:
    pass

print("Python plot detection setup complete")
"""

        return setup_code + "\n\n" + code

    def extract_plots(
        self, container: "ContainerProtocol", output_dir: str
    ) -> list[PlotOutput]:
        """Extract plots from Python execution."""
        plots = []

        try:
            exit_code, _ = container.exec_run("test -d /tmp/sandbox_plots")
            if exit_code != 0:
                return plots

            exit_code, output = container.exec_run(
                "find /tmp/sandbox_plots -name '*.png' -o -name '*.svg' -o -name '*.pdf' -o -name '*.html'"
            )
            if exit_code != 0:
                return plots

            file_paths = output.decode().strip().split("\n")
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
            exit_code, output = container.exec_run(f"cat {metadata_path}")
            if exit_code == 0:
                metadata = json.loads(output.decode())
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
