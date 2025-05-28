# Artifact Extraction

The LLM Sandbox supports automatic extraction of artifacts (such as plots, charts, and visualizations) generated during code execution. This feature uses a language-specific approach, allowing each programming language to implement artifact extraction in the most appropriate way.

## Overview

Artifact extraction automatically captures and returns visual outputs created by your code, such as:

- **Plots and Charts**: matplotlib, plotly, seaborn (Python)
- **Data Visualizations**: D3.js, Chart.js (JavaScript - future)
- **Scientific Plots**: ROOT framework (C++ - future)
- **Statistical Graphics**: ggplot2 (R - future)

## Quick Start

### Basic Usage

```python
from llm_sandbox import ArtifactSandboxSession

# Create a session with artifact extraction enabled
with ArtifactSandboxSession(lang="python", enable_plotting=True) as session:
    code = """
    import matplotlib.pyplot as plt
    import numpy as np

    x = np.linspace(0, 10, 100)
    y = np.sin(x)

    plt.figure(figsize=(10, 6))
    plt.plot(x, y, label='sin(x)')
    plt.title('Sine Wave')
    plt.xlabel('x')
    plt.ylabel('sin(x)')
    plt.legend()
    plt.grid(True)
    plt.show()
    """

    result = session.run(code)

    print(f"Generated {len(result.plots)} plots")
    for i, plot in enumerate(result.plots):
        print(f"Plot {i+1}: {plot.format.value} format")
```

### Saving Extracted Plots

```python
import base64
from pathlib import Path

with ArtifactSandboxSession(lang="python") as session:
    result = session.run(plotting_code)

    # Save all generated plots
    output_dir = Path("generated_plots")
    output_dir.mkdir(exist_ok=True)

    for i, plot in enumerate(result.plots):
        filename = output_dir / f"plot_{i+1:03d}.{plot.format.value}"
        with open(filename, "wb") as f:
            f.write(base64.b64decode(plot.content_base64))
        print(f"Saved: {filename}")
```

## Supported Languages

### Python (Fully Supported)

Python has comprehensive artifact extraction support for multiple plotting libraries:

#### Supported Libraries
- **matplotlib**: Static plots (PNG, SVG, PDF)
- **plotly**: Interactive plots (HTML, PNG)
- **seaborn**: Statistical visualizations (PNG, SVG)

#### Example: Multiple Plot Types

```python
with ArtifactSandboxSession(lang="python") as session:
    code = """
    import matplotlib.pyplot as plt
    import seaborn as sns
    import plotly.express as px
    import pandas as pd
    import numpy as np

    # Create sample data
    data = pd.DataFrame({
        'x': np.random.randn(100),
        'y': np.random.randn(100),
        'category': np.random.choice(['A', 'B', 'C'], 100)
    })

    # Matplotlib plot
    plt.figure(figsize=(8, 6))
    plt.scatter(data['x'], data['y'], c=data['category'].astype('category').cat.codes)
    plt.title('Matplotlib Scatter Plot')
    plt.show()

    # Seaborn plot
    plt.figure(figsize=(8, 6))
    sns.boxplot(data=data, x='category', y='y')
    plt.title('Seaborn Box Plot')
    plt.show()

    # Plotly plot
    fig = px.scatter(data, x='x', y='y', color='category',
                     title='Plotly Interactive Scatter Plot')
    fig.show()
    """

    result = session.run(code, libraries=['plotly'])
    print(f"Generated {len(result.plots)} plots")
```

#### Installing Additional Libraries

```python
with ArtifactSandboxSession(lang="python") as session:
    # Libraries are automatically installed when specified
    result = session.run(code, libraries=['plotly', 'seaborn', 'pandas'])
```

### Other Languages (Future Support)

Other programming languages have placeholder implementations ready for future development:

#### JavaScript (Planned)
- **Chart.js**: Canvas-based charts
- **D3.js**: SVG-based visualizations
- **Plotly.js**: Interactive plots

#### Java (Planned)
- **JFreeChart**: Chart generation
- **XChart**: Lightweight plotting

#### C++ (Planned)
- **ROOT**: Scientific data analysis
- **matplotlib-cpp**: Python matplotlib bindings

#### Go (Planned)
- **Gonum/plot**: Scientific plotting
- **go-chart**: Chart generation

## Configuration

### ArtifactSandboxSession Parameters

```python
ArtifactSandboxSession(
    backend=SandboxBackend.DOCKER,     # Container backend
    lang="python",                     # Programming language
    enable_plotting=True,              # Enable artifact extraction
    image="custom-image",              # Custom container image
    verbose=True,                      # Enable verbose logging
    keep_template=False,               # Keep container after session
    workdir="/sandbox"                 # Working directory
)
```

### Disabling Artifact Extraction

```python
# Disable plotting for faster execution
with ArtifactSandboxSession(lang="python", enable_plotting=False) as session:
    result = session.run(code)
    # result.plots will be empty
```

## Advanced Usage

### Custom Output Directory

The artifact extraction system uses `/tmp/sandbox_plots` as the default output directory within the container. This is automatically managed and doesn't require user configuration.

### Error Handling

```python
from llm_sandbox.exceptions import LanguageNotSupportPlotError

try:
    with ArtifactSandboxSession(lang="java", enable_plotting=True) as session:
        result = session.run(code)
except LanguageNotSupportPlotError as e:
    print(f"Plot extraction not supported: {e}")
    # Fall back to running without plotting
    with ArtifactSandboxSession(lang="java", enable_plotting=False) as session:
        result = session.run(code)
```

### Checking Language Support

```python
session = ArtifactSandboxSession(lang="python")
handler = session._session.language_handler

print(f"Supports plotting: {handler.is_support_plot_detection}")
print(f"Supported libraries: {[lib.value for lib in handler.supported_plot_libraries]}")
```

## Plot Formats

The system supports multiple output formats depending on the plotting library:

| Format | Extension | Description | Libraries |
|--------|-----------|-------------|-----------|
| PNG | `.png` | Raster image | matplotlib, plotly |
| SVG | `.svg` | Vector graphics | matplotlib, seaborn |
| PDF | `.pdf` | Portable document | matplotlib |
| HTML | `.html` | Interactive web | plotly |

## Best Practices

### 1. Explicit Plot Display

Always call the appropriate display function for your plotting library:

```python
# matplotlib
plt.show()

# plotly
fig.show()

# seaborn (uses matplotlib backend)
plt.show()
```

### 2. Figure Management

Close figures explicitly to avoid memory issues:

```python
import matplotlib.pyplot as plt

plt.figure()
plt.plot([1, 2, 3], [1, 4, 9])
plt.show()
plt.close()  # Clean up
```

### 3. Library Installation

Specify all required libraries upfront:

```python
libraries = ['matplotlib', 'seaborn', 'plotly', 'pandas', 'numpy']
result = session.run(code, libraries=libraries)
```

### 4. Error Handling

Handle cases where plotting libraries might not be available:

```python
code = """
try:
    import matplotlib.pyplot as plt
    plt.plot([1, 2, 3])
    plt.show()
except ImportError:
    print("Matplotlib not available")
"""
```

## Troubleshooting

### Common Issues

#### No Plots Generated
- Ensure `enable_plotting=True` is set
- Check that plotting code calls display functions (`plt.show()`, `fig.show()`)
- Verify the language supports plot extraction

#### Missing Libraries
- Add required libraries to the `libraries` parameter
- Check library names are correct (e.g., `plotly` not `plotly.express`)

#### Format Issues
- Some libraries may generate different formats than expected
- Check the `plot.format` attribute to see what was actually generated

### Debug Mode

Enable verbose logging to see detailed information:

```python
with ArtifactSandboxSession(lang="python", verbose=True) as session:
    result = session.run(code)
```

## API Reference

### ArtifactSandboxSession

The main class for running code with artifact extraction.

#### Methods

##### `run(code: str, libraries: list[str] | None = None) -> ExecutionResult`

Execute code and extract artifacts.

**Parameters:**
- `code`: The code to execute
- `libraries`: Optional list of libraries to install

**Returns:**
- `ExecutionResult`: Contains stdout, stderr, exit_code, and plots

### ExecutionResult

Result object containing execution output and extracted artifacts.

#### Attributes

- `exit_code: int` - Process exit code
- `stdout: str` - Standard output
- `stderr: str` - Standard error
- `plots: list[PlotOutput]` - Extracted plots/artifacts

### PlotOutput

Represents an extracted plot or visualization.

#### Attributes

- `format: FileType` - Plot format (PNG, SVG, PDF, HTML)
- `content_base64: str` - Base64-encoded plot data

## Examples

### Scientific Computing

```python
with ArtifactSandboxSession(lang="python") as session:
    code = """
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy import signal

    # Generate signal
    t = np.linspace(0, 1, 1000)
    sig = np.sin(2 * np.pi * 5 * t) + np.sin(2 * np.pi * 10 * t)

    # Add noise
    sig_noise = sig + np.random.normal(0, 0.1, sig.shape)

    # Apply filter
    b, a = signal.butter(4, 0.2)
    sig_filtered = signal.filtfilt(b, a, sig_noise)

    # Plot results
    plt.figure(figsize=(12, 8))

    plt.subplot(3, 1, 1)
    plt.plot(t, sig)
    plt.title('Original Signal')

    plt.subplot(3, 1, 2)
    plt.plot(t, sig_noise)
    plt.title('Noisy Signal')

    plt.subplot(3, 1, 3)
    plt.plot(t, sig_filtered)
    plt.title('Filtered Signal')

    plt.tight_layout()
    plt.show()
    """

    result = session.run(code, libraries=['scipy'])
```

### Data Analysis

```python
with ArtifactSandboxSession(lang="python") as session:
    code = """
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns

    # Create sample dataset
    np.random.seed(42)
    data = pd.DataFrame({
        'sales': np.random.normal(1000, 200, 100),
        'marketing': np.random.normal(500, 100, 100),
        'region': np.random.choice(['North', 'South', 'East', 'West'], 100)
    })

    # Correlation analysis
    plt.figure(figsize=(10, 8))

    plt.subplot(2, 2, 1)
    plt.scatter(data['marketing'], data['sales'])
    plt.xlabel('Marketing Spend')
    plt.ylabel('Sales')
    plt.title('Sales vs Marketing')

    plt.subplot(2, 2, 2)
    sns.boxplot(data=data, x='region', y='sales')
    plt.title('Sales by Region')

    plt.subplot(2, 2, 3)
    data['sales'].hist(bins=20)
    plt.xlabel('Sales')
    plt.title('Sales Distribution')

    plt.subplot(2, 2, 4)
    correlation = data[['sales', 'marketing']].corr()
    sns.heatmap(correlation, annot=True, cmap='coolwarm')
    plt.title('Correlation Matrix')

    plt.tight_layout()
    plt.show()
    """

    result = session.run(code, libraries=['pandas', 'seaborn'])
```

This artifact extraction system provides a powerful way to automatically capture and work with visualizations generated by your code, making it ideal for data analysis, scientific computing, and educational applications.
