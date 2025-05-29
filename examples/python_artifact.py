# ruff: noqa: E501

import base64
import logging
from pathlib import Path

from podman import PodmanClient

from llm_sandbox import ArtifactSandboxSession, SandboxBackend

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

podman_client = PodmanClient(
    base_url="unix:///var/folders/lh/rjbzw60n1fv7xr9kffn7gr840000gn/T/podman/podman-machine-default-api.sock"
)

code = """
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

# Set style for better looking plots
plt.style.use('default')
sns.set_palette("husl")

# Generate sample data
np.random.seed(42)
x = np.linspace(0, 10, 100)
y1 = np.sin(x) + np.random.normal(0, 0.1, 100)
y2 = np.cos(x) + np.random.normal(0, 0.1, 100)

# Test 1: Basic matplotlib plot
plt.figure(figsize=(10, 6))
plt.plot(x, y1, label='sin(x) + noise', linewidth=2)
plt.plot(x, y2, label='cos(x) + noise', linewidth=2)
plt.title('Matplotlib: Trigonometric Functions with Noise')
plt.xlabel('X values')
plt.ylabel('Y values')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

# Test 2: Matplotlib subplots
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
axes[0, 0].plot(x, y1, 'b-', alpha=0.7)
axes[0, 0].set_title('Sine Wave')
axes[0, 1].scatter(x[::5], y2[::5], c='red', alpha=0.6)
axes[0, 1].set_title('Cosine Scatter')
axes[1, 0].hist(y1, bins=20, alpha=0.7, color='green')
axes[1, 0].set_title('Sine Distribution')
axes[1, 1].bar(range(10), np.random.rand(10), alpha=0.7)
axes[1, 1].set_title('Random Bar Chart')
plt.tight_layout()
plt.show()

# Test 3: Seaborn plots
# Create sample dataset
data = pd.DataFrame({
    'x': np.random.randn(200),
    'y': np.random.randn(200),
    'category': np.random.choice(['A', 'B', 'C'], 200),
    'value': np.random.exponential(2, 200)
})

# Seaborn scatter plot with hue
plt.figure(figsize=(10, 6))
sns.scatterplot(data=data, x='x', y='y', hue='category', size='value', alpha=0.7)
plt.title('Seaborn: Scatter Plot with Categories and Sizes')
plt.show()

# Seaborn distribution plots
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
sns.histplot(data=data, x='value', kde=True, ax=axes[0])
axes[0].set_title('Distribution with KDE')
sns.boxplot(data=data, x='category', y='value', ax=axes[1])
axes[1].set_title('Box Plot by Category')
sns.violinplot(data=data, x='category', y='value', ax=axes[2])
axes[2].set_title('Violin Plot by Category')
plt.tight_layout()
plt.show()

# Test 4: Seaborn correlation heatmap
correlation_data = pd.DataFrame(
    np.random.randn(50, 6),
    columns=[
        "Feature_A",
        "Feature_B",
        "Feature_C",
        "Feature_D",
        "Feature_E",
        "Feature_F",
    ],
)
plt.figure(figsize=(8, 6))
sns.heatmap(correlation_data.corr(), annot=True, cmap='coolwarm', center=0)
plt.title('Seaborn: Correlation Heatmap')
plt.show()

# Test 5: Pandas plotting
# Time series data
dates = pd.date_range('2023-01-01', periods=100, freq='D')
ts_data = pd.DataFrame({
    'date': dates,
    'sales': np.cumsum(np.random.randn(100)) + 100,
    'profit': np.cumsum(np.random.randn(100)) + 50,
    'expenses': np.cumsum(np.random.randn(100)) + 30
})
ts_data.set_index('date', inplace=True)

# Pandas line plot
ax = ts_data.plot(figsize=(12, 6), title='Pandas: Time Series Data')
ax.set_ylabel('Values')
plt.show()

# Pandas multiple plot types
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
ts_data['sales'].plot(kind='line', ax=axes[0, 0], title='Sales Line Plot')
ts_data['profit'].plot(kind='bar', ax=axes[0, 1], title='Profit Bar Plot')
ts_data[['sales', 'profit']].plot(kind='scatter', x='sales', y='profit', ax=axes[1, 0], title='Sales vs Profit Scatter')
ts_data['expenses'].plot(kind='hist', ax=axes[1, 1], bins=20,
                        title='Expenses Distribution')
plt.tight_layout()
plt.show()

# Test 6: Combined matplotlib + seaborn styling
plt.figure(figsize=(12, 8))
with sns.axes_style("whitegrid"):
    plt.subplot(2, 2, 1)
    plt.plot(x, np.sin(x), 'o-', alpha=0.7)
    plt.title('Sine with Seaborn Style')

    plt.subplot(2, 2, 2)
    sns.lineplot(x=x, y=np.cos(x))
    plt.title('Cosine with Seaborn')

    plt.subplot(2, 2, 3)
    plt.hist(np.random.normal(0, 1, 1000), bins=30, alpha=0.7)
    plt.title('Normal Distribution')

    plt.subplot(2, 2, 4)
    sns.boxplot(y=np.random.exponential(1, 100))
    plt.title('Exponential Box Plot')

plt.tight_layout()
plt.show()

# Test 7: Plotly Express plots
# Interactive scatter plot
fig = px.scatter(data, x='x', y='y', color='category', size='value',
                title='Plotly Express: Interactive Scatter Plot',
                hover_data=['value'])
fig.show()

# Interactive line plot with time series
fig = px.line(ts_data.reset_index(), x='date', y=['sales', 'profit', 'expenses'],
                title='Plotly Express: Interactive Time Series')
fig.show()

# 3D scatter plot
fig = px.scatter_3d(x=np.random.randn(100), y=np.random.randn(100), z=np.random.randn(100),
                    color=np.random.choice(['A', 'B', 'C'], 100),
                    title='Plotly Express: 3D Scatter Plot')
fig.show()

# Test 8: Plotly Graph Objects
# Custom interactive plot
fig = go.Figure()
fig.add_trace(go.Scatter(x=x, y=y1, mode='lines+markers', name='sin(x) + noise',
                         line=dict(color='blue', width=3)))
fig.add_trace(go.Scatter(x=x, y=y2, mode='lines+markers', name='cos(x) + noise',
                         line=dict(color='red', width=3)))
fig.update_layout(title='Plotly Graph Objects: Custom Interactive Plot',
                  xaxis_title='X values',
                  yaxis_title='Y values',
                  hovermode='x unified')
fig.show()

# Test 9: Plotly subplots
fig = make_subplots(rows=2, cols=2,
                    subplot_titles=('Line Plot', 'Bar Chart', 'Histogram', 'Box Plot'),
                    specs=[[{"secondary_y": False}, {"secondary_y": False}],
                           [{"secondary_y": False}, {"secondary_y": False}]])

# Add traces
fig.add_trace(go.Scatter(x=x[:20], y=y1[:20], mode='lines+markers', name='Line'),
              row=1, col=1)
fig.add_trace(go.Bar(x=list('ABCDE'), y=np.random.rand(5), name='Bar'),
              row=1, col=2)
fig.add_trace(go.Histogram(x=np.random.normal(0, 1, 200), name='Histogram'),
              row=2, col=1)
fig.add_trace(go.Box(y=np.random.exponential(1, 100), name='Box'),
              row=2, col=2)

fig.update_layout(title_text="Plotly Subplots: Multiple Chart Types", showlegend=False)
fig.show()

# Test 10: Plotly heatmap
z = np.random.randn(20, 20)
fig = go.Figure(data=go.Heatmap(z=z, colorscale='Viridis'))
fig.update_layout(title='Plotly: Interactive Heatmap')
fig.show()

# Test 11: Plotly surface plot (3D)
x_surf = np.linspace(-5, 5, 50)
y_surf = np.linspace(-5, 5, 50)
X_surf, Y_surf = np.meshgrid(x_surf, y_surf)
Z_surf = np.sin(np.sqrt(X_surf**2 + Y_surf**2))

fig = go.Figure(data=[go.Surface(z=Z_surf, x=X_surf, y=Y_surf)])
fig.update_layout(title='Plotly: 3D Surface Plot',
                  scene=dict(xaxis_title='X', yaxis_title='Y', zaxis_title='Z'))
fig.show()

# Test 12: Plotly animated plot
frames = []
for i in range(0, 100, 5):
    frame_data = go.Scatter(x=x[:i], y=y1[:i], mode='lines+markers', name=f'Frame {i//5}')
    frames.append(go.Frame(data=[frame_data], name=f'Frame {i//5}'))

fig = go.Figure(
    data=[go.Scatter(x=x[:5], y=y1[:5], mode='lines+markers')],
    frames=frames
)

fig.update_layout(
    title='Plotly: Animated Line Plot',
    updatemenus=[{
        'type': 'buttons',
        'showactive': False,
        'buttons': [
            {'label': 'Play', 'method': 'animate', 'args': [None, {'frame': {'duration': 100}}]},
            {'label': 'Pause', 'method': 'animate', 'args': [[None], {'frame': {'duration': 0}, 'mode': 'immediate'}]}
        ]
    }]
)
fig.show()

print("All plotting tests completed successfully!")
print("Generated plots from: matplotlib, seaborn, pandas, and plotly!")
"""

Path("plots").mkdir(exist_ok=True)

with ArtifactSandboxSession(
    lang="python",
    verbose=True,
    image="ghcr.io/vndee/sandbox-python-311-bullseye",
    backend=SandboxBackend.KUBERNETES,
) as session:
    result = session.run(code)
    logger.info("Captured %d plots", len(result.plots))

    # Create plots directory if it doesn't exist
    Path("plots/kubernetes").mkdir(exist_ok=True)

    # save plots to files
    for i, plot in enumerate(result.plots):
        plot_path = Path("plots/kubernetes") / f"{i + 1:06d}.{plot.format.value}"
        with plot_path.open("wb") as f:
            f.write(base64.b64decode(plot.content_base64))

with ArtifactSandboxSession(
    lang="python",
    verbose=True,
    image="ghcr.io/vndee/sandbox-python-311-bullseye",
    backend=SandboxBackend.DOCKER,
) as session:
    result = session.run(code)
    logger.info("Captured %d plots", len(result.plots))

    # Create plots directory if it doesn't exist
    Path("plots/docker").mkdir(exist_ok=True)

    # save plots to files
    for i, plot in enumerate(result.plots):
        plot_path = Path("plots/docker") / f"{i + 1:06d}.{plot.format.value}"
        with plot_path.open("wb") as f:
            f.write(base64.b64decode(plot.content_base64))

with ArtifactSandboxSession(
    client=podman_client,
    lang="python",
    verbose=True,
    image="ghcr.io/vndee/sandbox-python-311-bullseye",
    backend=SandboxBackend.PODMAN,
) as session:
    result = session.run(code)
    logger.info("Captured %d plots", len(result.plots))

    # Create plots directory if it doesn't exist
    Path("plots/podman").mkdir(exist_ok=True)

    # save plots to files
    for i, plot in enumerate(result.plots):
        plot_path = Path("plots/podman") / f"{i + 1:06d}.{plot.format.value}"
        with plot_path.open("wb") as f:
            f.write(base64.b64decode(plot.content_base64))
