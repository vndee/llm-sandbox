# ruff: noqa: E501

PYTHON_PLOT_DETECTION_CODE = """
# Multi-library plot detection setup
import os
import sys
import base64
from pathlib import Path

# Setup output directories
os.makedirs('/tmp/sandbox_plots', exist_ok=True)
os.makedirs('/tmp/sandbox_output', exist_ok=True)

# Global plot counter - persists across runs using a file
_counter_file = '/tmp/sandbox_plots/.counter'
if os.path.exists(_counter_file):
    try:
        with open(_counter_file, 'r') as f:
            content = f.read().strip()
            _plot_counter = int(content) if content else 0
    except (ValueError, IOError):
        _plot_counter = 0
else:
    _plot_counter = 0

def _save_counter():
    with open(_counter_file, 'w') as f:
        f.write(str(_plot_counter))

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
            # Only save if there are axes AND they contain actual plot elements
            if fig and fig.get_axes():
                # Check if any axes have content (lines, patches, collections, images, etc.)
                has_content = any(
                    ax.lines or ax.patches or ax.collections or ax.images or ax.texts
                    for ax in fig.get_axes()
                )

                if has_content:
                    # Save as PNG with sequential numbering
                    _plot_counter += 1
                    _save_counter()
                    filename = f'/tmp/sandbox_plots/{_plot_counter:06d}.png'
                    fig.savefig(filename, format='png', dpi=100, bbox_inches='tight')
        except Exception as e:
            print(f"Matplotlib capture error: {e}")
        finally:
            plt.clf()

    def _enhanced_savefig(filename, *args, **kwargs):
        global _plot_counter
        result = _original_savefig(filename, *args, **kwargs)

        try:
            # Copy to our output directory with sequential numbering
            import shutil
            ext = Path(filename).suffix
            _plot_counter += 1
            _save_counter()
            output_file = f'/tmp/sandbox_plots/{_plot_counter:06d}{ext}'
            shutil.copy2(filename, output_file)
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

    # Monkey patch Figure methods
    def _enhanced_show(self, *args, **kwargs):
        global _plot_counter

        # Call original show method if it exists
        if hasattr(go.Figure, '_original_show'):
            result = self._original_show(*args, **kwargs)
        else:
            result = None

        try:
            # Save the figure as HTML with sequential numbering
            _plot_counter += 1
            _save_counter()
            html_file = f'/tmp/sandbox_plots/{_plot_counter:06d}.html'
            self.write_html(html_file)
        except Exception as e:
            print(f"Plotly show capture error: {e}")

        return result

    def _enhanced_write_html(self, file, *args, **kwargs):
        global _plot_counter

        # Call original method
        if hasattr(go.Figure, '_original_write_html'):
            result = self._original_write_html(file, *args, **kwargs)
        else:
            result = super(go.Figure, self).write_html(file, *args, **kwargs)

        try:
            # Copy to our output directory with sequential numbering
            import shutil
            _plot_counter += 1
            _save_counter()
            output_file = f'/tmp/sandbox_plots/{_plot_counter:06d}.html'
            shutil.copy2(file, output_file)
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
            # Copy to our output directory with sequential numbering
            import shutil
            ext = Path(file).suffix
            _plot_counter += 1
            _save_counter()
            output_file = f'/tmp/sandbox_plots/{_plot_counter:06d}{ext}'
            shutil.copy2(file, output_file)
        except Exception as e:
            print(f"Plotly image capture error: {e}")

        return result

    # Apply patches
    go.Figure._original_show = getattr(go.Figure, 'show', None)
    go.Figure._original_write_html = go.Figure.write_html
    go.Figure._original_write_image = go.Figure.write_image
    go.Figure.show = _enhanced_show
    go.Figure.write_html = _enhanced_write_html
    go.Figure.write_image = _enhanced_write_image

except ImportError:
    pass

# === SEABORN SUPPORT ===
try:
    import seaborn as sns
    # Seaborn uses matplotlib backend, so it's already covered
    # No need to print anything as it's just using matplotlib hooks
except ImportError:
    pass

print("Python plot detection setup complete")
"""
