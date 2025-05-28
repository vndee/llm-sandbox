# Tutorials

This section provides step-by-step tutorials for common use cases with LLM Sandbox artifact extraction.

## Tutorial 1: Data Visualization with Python

Learn how to create and extract data visualizations using Python's popular plotting libraries.

### Prerequisites

- LLM Sandbox installed
- Docker running on your system

### Step 1: Basic Matplotlib Plot

```python
from llm_sandbox import ArtifactSandboxSession
import base64
from pathlib import Path

# Create output directory
output_dir = Path("tutorial_plots")
output_dir.mkdir(exist_ok=True)

with ArtifactSandboxSession(lang="python", verbose=True) as session:
    code = """
    import matplotlib.pyplot as plt
    import numpy as np

    # Generate data
    x = np.linspace(0, 2*np.pi, 100)
    y1 = np.sin(x)
    y2 = np.cos(x)

    # Create plot
    plt.figure(figsize=(10, 6))
    plt.plot(x, y1, label='sin(x)', linewidth=2)
    plt.plot(x, y2, label='cos(x)', linewidth=2)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Trigonometric Functions')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()
    """

    result = session.run(code)

    # Save the plot
    if result.plots:
        plot = result.plots[0]
        filename = output_dir / f"trigonometric.{plot.format.value}"
        with open(filename, "wb") as f:
            f.write(base64.b64decode(plot.content_base64))
        print(f"Saved plot: {filename}")
```

### Step 2: Multiple Subplots

```python
with ArtifactSandboxSession(lang="python") as session:
    code = """
    import matplotlib.pyplot as plt
    import numpy as np

    # Generate sample data
    np.random.seed(42)
    data1 = np.random.normal(0, 1, 1000)
    data2 = np.random.exponential(2, 1000)

    # Create subplots
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))

    # Histogram
    ax1.hist(data1, bins=30, alpha=0.7, color='blue')
    ax1.set_title('Normal Distribution')
    ax1.set_xlabel('Value')
    ax1.set_ylabel('Frequency')

    # Scatter plot
    ax2.scatter(data1[:100], data2[:100], alpha=0.6)
    ax2.set_title('Scatter Plot')
    ax2.set_xlabel('Normal Data')
    ax2.set_ylabel('Exponential Data')

    # Line plot
    x = np.linspace(0, 10, 100)
    ax3.plot(x, np.sin(x), 'r-', label='sin(x)')
    ax3.plot(x, np.cos(x), 'b--', label='cos(x)')
    ax3.set_title('Trigonometric Functions')
    ax3.legend()

    # Box plot
    ax4.boxplot([data1, data2], labels=['Normal', 'Exponential'])
    ax4.set_title('Box Plot Comparison')

    plt.tight_layout()
    plt.show()
    """

    result = session.run(code)
    print(f"Generated {len(result.plots)} plots")
```

### Step 3: Interactive Plotly Visualization

```python
with ArtifactSandboxSession(lang="python") as session:
    code = """
    import plotly.express as px
    import plotly.graph_objects as go
    import pandas as pd
    import numpy as np

    # Create sample dataset
    np.random.seed(42)
    n_points = 200

    df = pd.DataFrame({
        'x': np.random.randn(n_points),
        'y': np.random.randn(n_points),
        'size': np.random.randint(10, 50, n_points),
        'category': np.random.choice(['A', 'B', 'C', 'D'], n_points),
        'value': np.random.uniform(0, 100, n_points)
    })

    # Create interactive scatter plot
    fig = px.scatter(
        df,
        x='x',
        y='y',
        size='size',
        color='category',
        hover_data=['value'],
        title='Interactive Scatter Plot',
        width=800,
        height=600
    )

    fig.update_layout(
        showlegend=True,
        hovermode='closest'
    )

    fig.show()
    """

    result = session.run(code, libraries=['plotly', 'pandas'])

    # Save interactive plot
    if result.plots:
        for i, plot in enumerate(result.plots):
            if plot.format.value == 'html':
                filename = output_dir / f"interactive_plot_{i}.html"
                with open(filename, "wb") as f:
                    f.write(base64.b64decode(plot.content_base64))
                print(f"Saved interactive plot: {filename}")
```

## Tutorial 2: Statistical Analysis with Seaborn

Learn how to create statistical visualizations using Seaborn.

### Step 1: Distribution Analysis

```python
with ArtifactSandboxSession(lang="python") as session:
    code = """
    import seaborn as sns
    import matplotlib.pyplot as plt
    import pandas as pd
    import numpy as np

    # Set style
    sns.set_style("whitegrid")

    # Generate sample data
    np.random.seed(42)
    data = pd.DataFrame({
        'group': np.repeat(['A', 'B', 'C'], 100),
        'value': np.concatenate([
            np.random.normal(0, 1, 100),
            np.random.normal(2, 1.5, 100),
            np.random.normal(-1, 0.5, 100)
        ]),
        'category': np.tile(['X', 'Y'], 150)
    })

    # Create distribution plots
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Histogram with KDE
    sns.histplot(data=data, x='value', hue='group', kde=True, ax=axes[0,0])
    axes[0,0].set_title('Distribution by Group')

    # Box plot
    sns.boxplot(data=data, x='group', y='value', ax=axes[0,1])
    axes[0,1].set_title('Box Plot by Group')

    # Violin plot
    sns.violinplot(data=data, x='group', y='value', hue='category', ax=axes[1,0])
    axes[1,0].set_title('Violin Plot by Group and Category')

    # Pair plot data preparation
    pivot_data = data.pivot_table(values='value', index=data.index//2,
                                  columns='group', aggfunc='mean').dropna()

    # Correlation heatmap
    correlation = pivot_data.corr()
    sns.heatmap(correlation, annot=True, cmap='coolwarm', center=0, ax=axes[1,1])
    axes[1,1].set_title('Correlation Heatmap')

    plt.tight_layout()
    plt.show()
    """

    result = session.run(code, libraries=['seaborn', 'pandas'])
    print(f"Generated {len(result.plots)} statistical plots")
```

### Step 2: Regression Analysis

```python
with ArtifactSandboxSession(lang="python") as session:
    code = """
    import seaborn as sns
    import matplotlib.pyplot as plt
    import pandas as pd
    import numpy as np

    # Generate regression data
    np.random.seed(42)
    n = 100
    x = np.random.uniform(0, 10, n)
    y = 2 * x + 1 + np.random.normal(0, 2, n)
    category = np.random.choice(['Type1', 'Type2', 'Type3'], n)

    df = pd.DataFrame({'x': x, 'y': y, 'category': category})

    # Create regression plots
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))

    # Scatter plot with regression line
    sns.scatterplot(data=df, x='x', y='y', hue='category', ax=axes[0])
    sns.regplot(data=df, x='x', y='y', scatter=False, ax=axes[0], color='red')
    axes[0].set_title('Scatter Plot with Regression Line')

    # Regression plot by category
    sns.lmplot(data=df, x='x', y='y', hue='category', height=6, aspect=1.2)
    plt.suptitle('Regression by Category')
    plt.show()

    # Residual plot
    plt.figure(figsize=(8, 6))
    sns.residplot(data=df, x='x', y='y')
    plt.title('Residual Plot')
    plt.show()
    """

    result = session.run(code, libraries=['seaborn', 'pandas'])
    print(f"Generated {len(result.plots)} regression plots")
```

## Tutorial 3: Scientific Computing Visualization

Learn how to create scientific plots for data analysis.

### Step 1: Signal Processing

```python
with ArtifactSandboxSession(lang="python") as session:
    code = """
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy import signal
    from scipy.fft import fft, fftfreq

    # Generate signal
    fs = 1000  # Sampling frequency
    t = np.linspace(0, 1, fs)

    # Create composite signal
    freq1, freq2, freq3 = 50, 120, 200
    signal_clean = (np.sin(2*np.pi*freq1*t) +
                   0.5*np.sin(2*np.pi*freq2*t) +
                   0.3*np.sin(2*np.pi*freq3*t))

    # Add noise
    noise = 0.2 * np.random.randn(len(t))
    signal_noisy = signal_clean + noise

    # Apply filter
    b, a = signal.butter(4, 150/(fs/2), 'low')
    signal_filtered = signal.filtfilt(b, a, signal_noisy)

    # FFT analysis
    fft_clean = fft(signal_clean)
    fft_noisy = fft(signal_noisy)
    fft_filtered = fft(signal_filtered)
    freqs = fftfreq(len(t), 1/fs)

    # Create comprehensive plot
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))

    # Time domain plots
    axes[0,0].plot(t[:200], signal_clean[:200], 'b-', label='Clean')
    axes[0,0].set_title('Clean Signal')
    axes[0,0].set_xlabel('Time (s)')
    axes[0,0].set_ylabel('Amplitude')
    axes[0,0].grid(True)

    axes[0,1].plot(t[:200], signal_noisy[:200], 'r-', label='Noisy')
    axes[0,1].set_title('Noisy Signal')
    axes[0,1].set_xlabel('Time (s)')
    axes[0,1].set_ylabel('Amplitude')
    axes[0,1].grid(True)

    axes[0,2].plot(t[:200], signal_filtered[:200], 'g-', label='Filtered')
    axes[0,2].set_title('Filtered Signal')
    axes[0,2].set_xlabel('Time (s)')
    axes[0,2].set_ylabel('Amplitude')
    axes[0,2].grid(True)

    # Frequency domain plots
    mask = freqs > 0
    axes[1,0].semilogy(freqs[mask], np.abs(fft_clean[mask]), 'b-')
    axes[1,0].set_title('Clean Signal FFT')
    axes[1,0].set_xlabel('Frequency (Hz)')
    axes[1,0].set_ylabel('Magnitude')
    axes[1,0].grid(True)

    axes[1,1].semilogy(freqs[mask], np.abs(fft_noisy[mask]), 'r-')
    axes[1,1].set_title('Noisy Signal FFT')
    axes[1,1].set_xlabel('Frequency (Hz)')
    axes[1,1].set_ylabel('Magnitude')
    axes[1,1].grid(True)

    axes[1,2].semilogy(freqs[mask], np.abs(fft_filtered[mask]), 'g-')
    axes[1,2].set_title('Filtered Signal FFT')
    axes[1,2].set_xlabel('Frequency (Hz)')
    axes[1,2].set_ylabel('Magnitude')
    axes[1,2].grid(True)

    plt.tight_layout()
    plt.show()
    """

    result = session.run(code, libraries=['scipy'])
    print(f"Generated {len(result.plots)} signal processing plots")
```

### Step 2: 3D Surface Plot

```python
with ArtifactSandboxSession(lang="python") as session:
    code = """
    import numpy as np
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D

    # Create 3D surface data
    x = np.linspace(-5, 5, 50)
    y = np.linspace(-5, 5, 50)
    X, Y = np.meshgrid(x, y)
    Z = np.sin(np.sqrt(X**2 + Y**2)) * np.exp(-0.1*np.sqrt(X**2 + Y**2))

    # Create 3D plots
    fig = plt.figure(figsize=(15, 10))

    # Surface plot
    ax1 = fig.add_subplot(2, 2, 1, projection='3d')
    surf = ax1.plot_surface(X, Y, Z, cmap='viridis', alpha=0.8)
    ax1.set_title('3D Surface Plot')
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_zlabel('Z')

    # Contour plot
    ax2 = fig.add_subplot(2, 2, 2)
    contour = ax2.contour(X, Y, Z, levels=20)
    ax2.clabel(contour, inline=True, fontsize=8)
    ax2.set_title('Contour Plot')
    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')

    # Filled contour plot
    ax3 = fig.add_subplot(2, 2, 3)
    contourf = ax3.contourf(X, Y, Z, levels=20, cmap='viridis')
    plt.colorbar(contourf, ax=ax3)
    ax3.set_title('Filled Contour Plot')
    ax3.set_xlabel('X')
    ax3.set_ylabel('Y')

    # Wireframe plot
    ax4 = fig.add_subplot(2, 2, 4, projection='3d')
    ax4.plot_wireframe(X, Y, Z, alpha=0.6)
    ax4.set_title('Wireframe Plot')
    ax4.set_xlabel('X')
    ax4.set_ylabel('Y')
    ax4.set_zlabel('Z')

    plt.tight_layout()
    plt.show()
    """

    result = session.run(code)
    print(f"Generated {len(result.plots)} 3D plots")
```

## Tutorial 4: Batch Processing Multiple Plots

Learn how to generate and save multiple plots efficiently.

```python
from llm_sandbox import ArtifactSandboxSession
import base64
from pathlib import Path

def save_plots(result, output_dir, prefix="plot"):
    """Helper function to save all plots from a result."""
    saved_files = []
    for i, plot in enumerate(result.plots):
        filename = output_dir / f"{prefix}_{i+1:03d}.{plot.format.value}"
        with open(filename, "wb") as f:
            f.write(base64.b64decode(plot.content_base64))
        saved_files.append(filename)
    return saved_files

# Create output directory
output_dir = Path("batch_plots")
output_dir.mkdir(exist_ok=True)

with ArtifactSandboxSession(lang="python", verbose=True) as session:
    # Generate multiple different plot types
    code = """
    import matplotlib.pyplot as plt
    import seaborn as sns
    import plotly.express as px
    import pandas as pd
    import numpy as np

    # Set random seed for reproducibility
    np.random.seed(42)

    # Generate sample dataset
    n_samples = 200
    data = pd.DataFrame({
        'x': np.random.randn(n_samples),
        'y': np.random.randn(n_samples),
        'category': np.random.choice(['A', 'B', 'C'], n_samples),
        'value': np.random.uniform(0, 100, n_samples),
        'size': np.random.randint(10, 100, n_samples)
    })

    # Plot 1: Matplotlib scatter
    plt.figure(figsize=(10, 8))
    colors = {'A': 'red', 'B': 'blue', 'C': 'green'}
    for cat in data['category'].unique():
        subset = data[data['category'] == cat]
        plt.scatter(subset['x'], subset['y'],
                   c=colors[cat], label=cat, alpha=0.6, s=50)
    plt.xlabel('X values')
    plt.ylabel('Y values')
    plt.title('Matplotlib: Scatter Plot by Category')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()

    # Plot 2: Seaborn distribution
    plt.figure(figsize=(12, 8))
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    sns.histplot(data=data, x='value', hue='category', kde=True, ax=axes[0,0])
    axes[0,0].set_title('Distribution by Category')

    sns.boxplot(data=data, x='category', y='value', ax=axes[0,1])
    axes[0,1].set_title('Box Plot by Category')

    sns.scatterplot(data=data, x='x', y='y', hue='category',
                   size='size', sizes=(20, 200), ax=axes[1,0])
    axes[1,0].set_title('Scatter with Size')

    correlation = data[['x', 'y', 'value', 'size']].corr()
    sns.heatmap(correlation, annot=True, cmap='coolwarm', ax=axes[1,1])
    axes[1,1].set_title('Correlation Matrix')

    plt.tight_layout()
    plt.show()

    # Plot 3: Plotly interactive
    fig = px.scatter_3d(data, x='x', y='y', z='value',
                       color='category', size='size',
                       title='3D Interactive Scatter Plot',
                       width=800, height=600)
    fig.show()

    # Plot 4: Time series simulation
    dates = pd.date_range('2023-01-01', periods=100, freq='D')
    ts_data = pd.DataFrame({
        'date': dates,
        'value': np.cumsum(np.random.randn(100)) + 100,
        'trend': np.linspace(95, 105, 100) + np.random.randn(100) * 2
    })

    plt.figure(figsize=(12, 6))
    plt.plot(ts_data['date'], ts_data['value'], label='Actual', linewidth=2)
    plt.plot(ts_data['date'], ts_data['trend'], label='Trend',
             linestyle='--', linewidth=2)
    plt.xlabel('Date')
    plt.ylabel('Value')
    plt.title('Time Series Analysis')
    plt.legend()
    plt.xticks(rotation=45)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
    """

    result = session.run(code, libraries=['plotly', 'seaborn', 'pandas'])

    # Save all generated plots
    saved_files = save_plots(result, output_dir, "tutorial_batch")

    print(f"Generated and saved {len(saved_files)} plots:")
    for file in saved_files:
        print(f"  - {file}")

print("\nTutorial completed! Check the output directories for generated plots.")
```

## Tips and Best Practices

### 1. Memory Management

When generating many plots, clear figures to avoid memory issues:

```python
import matplotlib.pyplot as plt

# After each plot
plt.show()
plt.clf()  # Clear the current figure
plt.close()  # Close the figure window
```

### 2. Error Handling

Always handle potential errors in plotting code:

```python
code = """
try:
    import matplotlib.pyplot as plt
    # Your plotting code here
    plt.show()
except ImportError as e:
    print(f"Required library not available: {e}")
except Exception as e:
    print(f"Plotting error: {e}")
"""
```

### 3. Optimizing Plot Quality

Set appropriate DPI and figure size for better quality:

```python
code = """
import matplotlib.pyplot as plt

# Set high DPI for better quality
plt.rcParams['figure.dpi'] = 150
plt.rcParams['savefig.dpi'] = 150

# Set appropriate figure size
plt.figure(figsize=(10, 6))
# Your plotting code
plt.show()
"""
```

### 4. Batch Processing

For multiple plots, use a systematic approach:

```python
def generate_analysis_plots(data_description):
    with ArtifactSandboxSession(lang="python") as session:
        code = f"""
        # Data generation based on: {data_description}
        # Multiple plot generation code here
        """
        return session.run(code, libraries=['matplotlib', 'seaborn', 'plotly'])

# Process multiple datasets
datasets = ["financial_data", "scientific_measurements", "user_analytics"]
all_results = []

for dataset in datasets:
    result = generate_analysis_plots(dataset)
    all_results.append(result)
    print(f"Generated {len(result.plots)} plots for {dataset}")
```

These tutorials provide a comprehensive introduction to using LLM Sandbox for data visualization and analysis. The artifact extraction feature makes it easy to capture and work with generated plots programmatically.
