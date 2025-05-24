import base64
import io
import json
import os
import shutil
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

matplotlib.use("Agg")


class FileType(Enum):
    """File types supported by the plot extractor."""

    PNG = "png"
    JPEG = "jpeg"
    PDF = "pdf"
    SVG = "svg"
    CSV = "csv"
    JSON = "json"
    TXT = "txt"
    HTML = "html"


@dataclass
class ChartData:
    """Structured data extracted from matplotlib charts."""

    chart_type: str
    title: str | None = None
    x_label: str | None = None
    y_label: str | None = None
    x_data: list[Any] | None = None
    y_data: list[Any] | None = None
    legend: list[str] | None = None
    colors: list[str] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class PlotOutput:
    """Represents a plot/chart output."""

    format: FileType
    content_base64: str
    width: int | None = None
    height: int | None = None
    dpi: int | None = None
    chart_data: ChartData | None = None

    def __repr__(self) -> str:
        """Return the string representation of the plot output."""
        return f"PlotOutput(format={self.format}, content_base64={self.content_base64}, width={self.width}, height={self.height}, dpi={self.dpi}, chart_data={self.chart_data})"

    def __str__(self) -> str:
        """Return the string representation of the plot output."""
        return self.__repr__()


@dataclass
class FileOutput:
    """Represents a file output from code execution."""

    filename: str
    content_base64: str
    file_type: FileType
    mime_type: str
    size: int

    @classmethod
    def from_path(cls, path: str) -> "FileOutput":
        """Create FileOutput from file path."""
        import mimetypes

        with open(path, "rb") as f:
            content = f.read()

        filename = os.path.basename(path)
        mime_type, _ = mimetypes.guess_type(path)
        file_extension = os.path.splitext(path)[1].lower().lstrip(".")

        try:
            file_type = FileType(file_extension)
        except ValueError:
            file_type = FileType.TXT

        return cls(
            filename=filename,
            content_base64=base64.b64encode(content).decode("utf-8"),
            file_type=file_type,
            mime_type=mime_type or "application/octet-stream",
            size=len(content),
        )

    def __repr__(self) -> str:
        """Return the string representation of the file output."""
        return f"FileOutput(filename={self.filename}, file_type={self.file_type}, mime_type={self.mime_type}, size={self.size})"

    def __str__(self) -> str:
        """Return the string representation of the file output."""
        return self.__repr__()


class PlotExtractor:
    """Extract plots and charts from matplotlib with E2B-style functionality."""

    def __init__(self) -> None:
        """Initialize the plot extractor."""
        self.detected_plots = []
        self.plot_counter = 0
        self.original_show = plt.show
        self.original_savefig = plt.savefig

        # Setup output directory
        self.output_dir = "/tmp/sandbox_plots"
        os.makedirs(self.output_dir, exist_ok=True)

        # Monkey patch matplotlib
        self._setup_matplotlib_hooks()

    def _setup_matplotlib_hooks(self):
        """Setup matplotlib hooks to capture plots."""

        def enhanced_show(*args, **kwargs):
            """Enhanced plt.show() that captures plots."""
            try:
                fig = plt.gcf()
                if fig and fig.get_axes():
                    plot_data = self._extract_plot_data(fig)
                    self.detected_plots.append(plot_data)

                    # Save plot file
                    filename = f"{self.output_dir}/plot_{self.plot_counter}.png"
                    fig.savefig(filename, dpi=100, bbox_inches="tight")
                    self.plot_counter += 1

            except Exception as e:
                print(f"Plot capture error: {e}")
            finally:
                plt.clf()

        def enhanced_savefig(filename, *args, **kwargs):
            """Enhanced plt.savefig() that also captures plot data"""
            # Call original savefig
            result = self.original_savefig(filename, *args, **kwargs)

            try:
                fig = plt.gcf()
                if fig and fig.get_axes():
                    plot_data = self._extract_plot_data(fig)
                    plot_data["saved_filename"] = filename
                    self.detected_plots.append(plot_data)

                    # Also save to our output directory
                    output_filename = f"{self.output_dir}/{os.path.basename(filename)}"
                    shutil.copy2(filename, output_filename)

            except Exception as e:
                print(f"Savefig capture error: {e}")

            return result

        # Apply patches
        plt.show = enhanced_show
        plt.savefig = enhanced_savefig

    def _extract_plot_data(self, fig) -> dict[str, Any]:
        """Extract comprehensive data from matplotlib figure"""
        plot_data = {"formats": {}, "chart_data": {}, "metadata": {}}

        try:
            # Basic metadata
            plot_data["metadata"] = {
                "title": fig._suptitle.get_text() if fig._suptitle else None,
                "size": fig.get_size_inches().tolist(),
                "dpi": fig.dpi,
                "facecolor": fig.get_facecolor(),
                "num_axes": len(fig.get_axes()),
            }

            # Save in multiple formats
            formats_to_extract = ["png", "jpeg", "pdf", "svg"]
            for fmt in formats_to_extract:
                try:
                    buf = io.BytesIO()
                    fig.savefig(buf, format=fmt, bbox_inches="tight", dpi=100)
                    buf.seek(0)
                    content = buf.getvalue()
                    plot_data["formats"][fmt] = base64.b64encode(content).decode(
                        "utf-8"
                    )
                    buf.close()
                except Exception as e:
                    print(f"Error saving format {fmt}: {e}")

            # Extract chart data from axes
            if fig.get_axes():
                ax = fig.get_axes()[0]  # Use first axes
                plot_data["chart_data"] = self._extract_chart_data(ax)

        except Exception as e:
            plot_data["error"] = str(e)

        return plot_data

    def _extract_chart_data(self, ax) -> dict[str, Any]:
        """Extract structured data from matplotlib axes"""
        chart_data = {
            "x_label": ax.get_xlabel() or None,
            "y_label": ax.get_ylabel() or None,
            "title": ax.get_title() or None,
            "legend": [],
            "series": [],
            "chart_type": "unknown",
        }

        try:
            # Extract legend
            legend = ax.get_legend()
            if legend:
                chart_data["legend"] = [t.get_text() for t in legend.get_texts()]

            # Extract line plot data
            lines = ax.get_lines()
            if lines:
                chart_data["chart_type"] = "line"
                for line in lines:
                    x_data = line.get_xdata()
                    y_data = line.get_ydata()

                    # Convert numpy arrays to lists
                    x_list = (
                        x_data.tolist() if hasattr(x_data, "tolist") else list(x_data)
                    )
                    y_list = (
                        y_data.tolist() if hasattr(y_data, "tolist") else list(y_data)
                    )

                    series_data = {
                        "type": "line",
                        "x_data": x_list,
                        "y_data": y_list,
                        "label": line.get_label()
                        if line.get_label() != "_nolegend_"
                        else None,
                        "color": line.get_color(),
                        "linewidth": line.get_linewidth(),
                        "linestyle": line.get_linestyle(),
                    }
                    chart_data["series"].append(series_data)

            # Extract bar chart data
            patches = ax.patches
            if patches and not lines:  # Bar chart if patches but no lines
                chart_data["chart_type"] = "bar"
                bar_data = []
                for patch in patches:
                    if hasattr(patch, "get_height") and patch.get_height() > 0:
                        bar_info = {
                            "x": patch.get_x(),
                            "width": patch.get_width(),
                            "height": patch.get_height(),
                            "color": patch.get_facecolor(),
                        }
                        bar_data.append(bar_info)
                chart_data["bar_data"] = bar_data

            # Extract scatter plot data
            collections = ax.collections
            if collections:
                for collection in collections:
                    if hasattr(collection, "get_offsets"):
                        chart_data["chart_type"] = "scatter"
                        offsets = collection.get_offsets()
                        if len(offsets) > 0:
                            scatter_data = {
                                "x_data": [point[0] for point in offsets],
                                "y_data": [point[1] for point in offsets],
                                "colors": collection.get_facecolors().tolist()
                                if hasattr(collection.get_facecolors(), "tolist")
                                else None,
                            }
                            chart_data["scatter_data"] = scatter_data

        except Exception as e:
            chart_data["extraction_error"] = str(e)

        return chart_data

    def get_plots(self) -> list[PlotOutput]:
        """Convert detected plots to PlotOutput objects"""
        plots = []

        for plot_data in self.detected_plots:
            formats = plot_data.get("formats", {})

            for fmt, content_base64 in formats.items():
                # Create ChartData object
                chart_data = None
                if "chart_data" in plot_data:
                    cd = plot_data["chart_data"]
                    chart_data = ChartData(
                        chart_type=cd.get("chart_type", "unknown"),
                        title=cd.get("title"),
                        x_label=cd.get("x_label"),
                        y_label=cd.get("y_label"),
                        legend=cd.get("legend"),
                        metadata=plot_data.get("metadata", {}),
                    )

                    # Add series data
                    if cd.get("series"):
                        chart_data.x_data = cd["series"][0].get("x_data")
                        chart_data.y_data = cd["series"][0].get("y_data")
                        chart_data.colors = [s.get("color") for s in cd["series"]]

                # Create PlotOutput
                plot = PlotOutput(
                    format=FileType(fmt),
                    content_base64=content_base64,
                    width=int(
                        plot_data.get("metadata", {}).get("size", [None, None])[0] * 100
                    )
                    if plot_data.get("metadata", {}).get("size", [None, None])[0]
                    else None,
                    height=int(
                        plot_data.get("metadata", {}).get("size", [None, None])[1] * 100
                    )
                    if plot_data.get("metadata", {}).get("size", [None, None])[1]
                    else None,
                    dpi=plot_data.get("metadata", {}).get("dpi"),
                    chart_data=chart_data,
                )
                plots.append(plot)

        return plots

    def clear(self):
        """Clear detected plots"""
        self.detected_plots.clear()
        self.plot_counter = 0


class FileHandler:
    """Handle file outputs from code execution."""

    def __init__(self, output_dir: str = "/tmp/sandbox_output") -> None:
        """Initialize the file handler."""
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        self.initial_files = set()
        self._scan_initial_files()

    def _scan_initial_files(self):
        """Scan existing files"""
        try:
            if os.path.exists(self.output_dir):
                for filename in os.listdir(self.output_dir):
                    self.initial_files.add(filename)
        except Exception as e:
            print(f"Error scanning initial files: {e}")

    def get_new_files(self) -> list[FileOutput]:
        """Get newly created files"""
        file_outputs = []

        try:
            if os.path.exists(self.output_dir):
                current_files = set(os.listdir(self.output_dir))
                new_files = current_files - self.initial_files

                for filename in new_files:
                    file_path = os.path.join(self.output_dir, filename)
                    if os.path.isfile(file_path):
                        try:
                            file_output = FileOutput.from_path(file_path)
                            file_outputs.append(file_output)
                        except Exception as e:
                            print(f"Error processing file {filename}: {e}")

        except Exception as e:
            print(f"Error getting new files: {e}")

        return file_outputs

    def clear(self):
        """Update initial files list"""
        self._scan_initial_files()


# Example usage and testing
if __name__ == "__main__":
    # Test the plot extractor
    extractor = PlotExtractor()

    # Create a test plot
    import numpy as np

    x = np.linspace(0, 10, 100)
    y = np.sin(x)

    plt.figure(figsize=(10, 6))
    plt.plot(x, y, label="sin(x)", color="blue", linewidth=2)
    plt.xlabel("X values")
    plt.ylabel("Y values")
    plt.title("Sine Wave Plot")
    plt.legend()
    plt.grid(True)
    plt.show()  # This will be captured

    # Get the plots
    plots = extractor.get_plots()

    if plots:
        plot = plots[0]
        print("Captured plot:")
        print(f"- Format: {plot.format}")
        print(f"- Chart type: {plot.chart_data.chart_type}")
        print(f"- Title: {plot.chart_data.title}")
        print(f"- X label: {plot.chart_data.x_label}")
        print(f"- Y label: {plot.chart_data.y_label}")
        print(
            f"- Data points: {len(plot.chart_data.x_data) if plot.chart_data.x_data else 0}"
        )

        # Save the plot
        with open("test_output.png", "wb") as f:
            f.write(base64.b64decode(plot.content_base64))
        print("Plot saved as test_output.png")

    # Test file handler
    file_handler = FileHandler()

    # Create a test file
    test_data = {"message": "Hello from LLM Sandbox!", "numbers": [1, 2, 3, 4, 5]}
    with open("/tmp/sandbox_output/test.json", "w") as f:
        json.dump(test_data, f)

    # Get new files
    files = file_handler.get_new_files()

    if files:
        file_output = files[0]
        print("\nCaptured file:")
        print(f"- Filename: {file_output.filename}")
        print(f"- Type: {file_output.file_type}")
        print(f"- Size: {file_output.size} bytes")

        # Decode and verify content
        content = base64.b64decode(file_output.content_base64)
        print(f"- Content: {content.decode('utf-8')}")
