PYTHON_PLOT_DETECTION_CODE = """
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
