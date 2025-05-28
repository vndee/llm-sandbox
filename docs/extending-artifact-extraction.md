# Extending Artifact Extraction

This guide explains how to extend the LLM Sandbox artifact extraction system to support new programming languages and plotting libraries.

## Architecture Overview

The artifact extraction system uses a language-specific approach where each programming language handler implements its own artifact extraction logic. This design provides flexibility and allows for language-specific optimizations.

### Key Components

1. **AbstractLanguageHandler**: Base class defining the artifact extraction interface
2. **Language Handlers**: Language-specific implementations (PythonHandler, JavaScriptHandler, etc.)
3. **ArtifactSandboxSession**: Main session class that delegates to language handlers
4. **Plot Detection Code**: Language-specific code injected to capture artifacts

## Adding Support for a New Language

### Step 1: Create Plot Detection Code

Create a new file in `llm_sandbox/language_handlers/artifact_detection/` for your language's plot detection code:

```python
# llm_sandbox/language_handlers/artifact_detection/javascript_plot_detection.py

JAVASCRIPT_PLOT_DETECTION_CODE = """
// JavaScript plot detection setup for Node.js environment
const fs = require('fs');
const path = require('path');

// Setup output directories
const plotsDir = '/tmp/sandbox_plots';
if (!fs.existsSync(plotsDir)) {
    fs.mkdirSync(plotsDir, { recursive: true });
}

// Global plot counter
let plotCounter = 0;

// === CHART.JS SUPPORT ===
try {
    const { ChartJSNodeCanvas } = require('chartjs-node-canvas');

    // Override Chart.js rendering to save plots
    const originalRender = ChartJSNodeCanvas.prototype.renderToBuffer;
    ChartJSNodeCanvas.prototype.renderToBuffer = function(configuration, mimeType = 'image/png') {
        const result = originalRender.call(this, configuration, mimeType);

        // Save the plot
        plotCounter++;
        const filename = `${plotsDir}/${String(plotCounter).padStart(6, '0')}.png`;

        result.then(buffer => {
            fs.writeFileSync(filename, buffer);
        }).catch(err => {
            console.error('Chart.js capture error:', err);
        });

        return result;
    };
} catch (e) {
    // Chart.js not available
}

console.log('JavaScript plot detection setup complete');
"""
```

### Step 2: Update Language Handler Configuration

Modify your language handler to include plot detection configuration:

```python
# llm_sandbox/language_handlers/javascript_handler.py

from llm_sandbox.language_handlers.artifact_detection.javascript_plot_detection import JAVASCRIPT_PLOT_DETECTION_CODE

class JavaScriptHandler(AbstractLanguageHandler):
    def __init__(self, logger: logging.Logger | None = None) -> None:
        super().__init__(logger)

        self.config = LanguageConfig(
            name=SupportedLanguage.JAVASCRIPT,
            file_extension="js",
            execution_commands=["node {file}"],
            package_manager="npm install",
            plot_detection=PlotDetectionConfig(
                libraries=[
                    PlotLibrary.CHARTJS,
                    PlotLibrary.D3JS,
                    PlotLibrary.PLOTLY,
                ],
                setup_code=JAVASCRIPT_PLOT_DETECTION_CODE,
                cleanup_code="",
            ),
        )
```

### Step 3: Override run_with_artifacts()

Implement language-specific artifact extraction logic:

```python
def run_with_artifacts(
    self,
    container: "ContainerProtocol",
    code: str,
    libraries: list | None = None,
    enable_plotting: bool = True,
    output_dir: str = "/tmp/sandbox_plots"
) -> tuple[Any, list[PlotOutput]]:
    """Run JavaScript code and extract artifacts with JS-specific logic."""
    plots: list[PlotOutput] = []

    if enable_plotting and self.is_support_plot_detection:
        # Inject JavaScript-specific plot detection code
        injected_code = self.inject_plot_detection_code(code)

        # Install additional dependencies for server-side rendering
        if libraries:
            libraries = list(libraries)  # Make a copy
            libraries.extend(['jsdom', 'canvas', 'chartjs-node-canvas'])
        else:
            libraries = ['jsdom', 'canvas', 'chartjs-node-canvas']

        # Run the code with plot detection
        result = container.run(injected_code, libraries)

        # Extract plots using JavaScript-specific logic
        plots = self.extract_plots(container, output_dir)

        return result, plots
    else:
        # Run code without plot detection
        result = container.run(code, libraries)
        return result, plots
```

### Step 4: Implement extract_plots()

Create language-specific plot extraction logic:

```python
def extract_plots(self, container: "ContainerProtocol", output_dir: str) -> list[PlotOutput]:
    """Extract plots from JavaScript execution."""
    plots: list[PlotOutput] = []

    try:
        # Check for generated plot files
        result = container.execute_command(f"test -d {output_dir}")
        if result.exit_code:
            return plots

        # Look for various image formats that JS libraries might generate
        result = container.execute_command(
            f"find {output_dir} -name '*.png' -o -name '*.svg' -o -name '*.html' -o -name '*.pdf'"
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
            except Exception:
                self.logger.exception("Error extracting plot %s", file_path)

    except Exception:
        self.logger.exception("Error extracting JavaScript plots")

    return plots

def _extract_single_plot(self, container: "ContainerProtocol", file_path: str) -> PlotOutput | None:
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
                        format=FileType(file_ext) if file_ext in ["png", "svg", "pdf", "html"] else FileType.PNG,
                        content_base64=base64.b64encode(content).decode("utf-8"),
                    )

    except (OSError, tarfile.TarError, ValueError):
        self.logger.exception("Error extracting single plot")

    return None
```

## Language-Specific Strategies

### Python Strategy: Monkey Patching

Python uses monkey patching to intercept plotting library calls:

```python
# Override matplotlib's show() function
_original_show = plt.show
def _enhanced_show(*args, **kwargs):
    global _plot_counter
    try:
        fig = plt.gcf()
        if fig and fig.get_axes():
            _plot_counter += 1
            filename = f'/tmp/sandbox_plots/{_plot_counter:06d}.png'
            fig.savefig(filename, format='png', dpi=100, bbox_inches='tight')
    except Exception as e:
        print(f"Plot capture error: {e}")
    finally:
        plt.clf()

plt.show = _enhanced_show
```

### JavaScript Strategy: Library Wrapping

JavaScript wraps plotting library constructors and methods:

```javascript
// Override Chart.js constructor
const OriginalChart = Chart;
Chart = function(ctx, config) {
    const chart = new OriginalChart(ctx, config);

    // Add save functionality
    chart.save = function(filename) {
        const canvas = this.canvas;
        const buffer = canvas.toBuffer('image/png');
        fs.writeFileSync(filename || generateFilename(), buffer);
    };

    return chart;
};
```

### Java Strategy: Output Interception

Java monitors file system for generated plot files:

```java
// Monitor output directory for new files
Path plotsDir = Paths.get("/tmp/sandbox_plots");
WatchService watchService = FileSystems.getDefault().newWatchService();
plotsDir.register(watchService, StandardWatchEventKinds.ENTRY_CREATE);

// JFreeChart integration
ChartUtils.saveChartAsPNG(new File(plotsDir + "/plot.png"), chart, 800, 600);
```

### C++ Strategy: Compilation Hooks

C++ uses compilation flags and library linking:

```cpp
// ROOT framework integration
#include "TCanvas.h"
#include "TH1F.h"

void setupPlotCapture() {
    gROOT->SetBatch(kTRUE);  // Disable interactive mode

    // Override canvas save behavior
    TCanvas::Class()->SetNew(&captureCanvas);
}

void captureCanvas(void* p) {
    TCanvas* canvas = (TCanvas*)p;
    std::string filename = generateFilename();
    canvas->SaveAs(filename.c_str());
}
```

## Plot Detection Patterns

### File-Based Detection

Most languages generate files that can be detected:

```python
def extract_plots(self, container, output_dir):
    # Find all image files
    result = container.execute_command(
        f"find {output_dir} -type f \\( -name '*.png' -o -name '*.svg' -o -name '*.pdf' \\)"
    )

    file_paths = result.stdout.strip().split("\n")
    return [self._extract_single_plot(container, path) for path in file_paths]
```

### Memory-Based Detection

Some languages can capture plots directly from memory:

```python
def capture_plot_from_memory(self, plot_object):
    """Capture plot directly from memory without file I/O."""
    buffer = io.BytesIO()
    plot_object.save(buffer, format='png')
    buffer.seek(0)

    return PlotOutput(
        format=FileType.PNG,
        content_base64=base64.b64encode(buffer.read()).decode('utf-8')
    )
```

### Stream-Based Detection

For real-time plotting:

```python
def setup_plot_streaming(self):
    """Setup real-time plot streaming."""
    self.plot_queue = queue.Queue()

    def plot_callback(plot_data):
        self.plot_queue.put(plot_data)

    # Register callback with plotting library
    plotting_lib.register_callback(plot_callback)
```

## Testing Your Implementation

### Unit Tests

Create tests for your language handler:

```python
def test_javascript_plot_extraction():
    """Test JavaScript plot extraction."""
    with ArtifactSandboxSession(lang="javascript") as session:
        code = """
        const Chart = require('chart.js');
        const canvas = createCanvas(800, 600);
        const ctx = canvas.getContext('2d');

        new Chart(ctx, {
            type: 'bar',
            data: {
                labels: ['A', 'B', 'C'],
                datasets: [{
                    data: [1, 2, 3]
                }]
            }
        });
        """

        result = session.run(code, libraries=['chart.js'])
        assert len(result.plots) >= 1
        assert result.plots[0].format in [FileType.PNG, FileType.SVG]
```

### Integration Tests

Test the full pipeline:

```python
def test_end_to_end_plotting():
    """Test complete plotting pipeline."""
    languages = ['python', 'javascript', 'java']

    for lang in languages:
        with ArtifactSandboxSession(lang=lang) as session:
            if session._session.language_handler.is_support_plot_detection:
                code = get_sample_plotting_code(lang)
                result = session.run(code)
                assert len(result.plots) > 0
```

## Best Practices

### 1. Error Handling

Always handle errors gracefully:

```python
def extract_plots(self, container, output_dir):
    plots = []
    try:
        # Plot extraction logic
        pass
    except Exception as e:
        self.logger.warning(f"Plot extraction failed: {e}")
        # Return empty list instead of crashing
    return plots
```

### 2. Resource Management

Clean up temporary files and resources:

```python
def cleanup_plots(self, container, output_dir):
    """Clean up temporary plot files."""
    try:
        container.execute_command(f"rm -rf {output_dir}/*")
    except Exception:
        pass  # Ignore cleanup errors
```

### 3. Format Detection

Automatically detect plot formats:

```python
def detect_format(self, file_path):
    """Detect plot format from file extension and content."""
    ext = Path(file_path).suffix.lower()

    format_map = {
        '.png': FileType.PNG,
        '.svg': FileType.SVG,
        '.pdf': FileType.PDF,
        '.html': FileType.HTML,
    }

    return format_map.get(ext, FileType.PNG)
```

### 4. Performance Optimization

Optimize for common use cases:

```python
def extract_plots_optimized(self, container, output_dir):
    """Optimized plot extraction with caching."""
    # Check if directory exists first
    if not self._directory_exists(container, output_dir):
        return []

    # Use efficient file listing
    result = container.execute_command(f"ls -1 {output_dir}/*.{{png,svg,pdf}}")
    if result.exit_code != 0:
        return []

    # Process files in parallel if many
    file_paths = result.stdout.strip().split('\n')
    if len(file_paths) > 10:
        return self._extract_plots_parallel(container, file_paths)
    else:
        return self._extract_plots_sequential(container, file_paths)
```

## Common Pitfalls

### 1. File Permissions

Ensure the sandbox can write to the output directory:

```python
def setup_output_directory(self, container, output_dir):
    """Setup output directory with proper permissions."""
    container.execute_command(f"mkdir -p {output_dir}")
    container.execute_command(f"chmod 777 {output_dir}")
```

### 2. Library Dependencies

Handle missing plotting libraries gracefully:

```python
def inject_plot_detection_code(self, code):
    """Inject plot detection with dependency checking."""
    detection_code = """
    try:
        import plotting_library
        # Setup plot detection
    except ImportError:
        print("Plotting library not available")
        # Provide fallback behavior
    """
    return detection_code + "\n\n" + code
```

### 3. Format Compatibility

Ensure generated formats are supported:

```python
SUPPORTED_FORMATS = {FileType.PNG, FileType.SVG, FileType.PDF, FileType.HTML}

def validate_plot_format(self, plot_format):
    """Validate that plot format is supported."""
    if plot_format not in SUPPORTED_FORMATS:
        self.logger.warning(f"Unsupported format: {plot_format}")
        return FileType.PNG  # Default fallback
    return plot_format
```

This guide provides the foundation for extending artifact extraction to any programming language. The key is to understand how plotting libraries work in your target language and implement appropriate interception mechanisms.
