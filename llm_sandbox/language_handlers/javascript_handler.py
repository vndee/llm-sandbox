from typing import TYPE_CHECKING

from llm_sandbox.artifact import FileOutput, PlotOutput

from .base import (
    AbstractLanguageHandler,
    LanguageConfig,
    PlotDetectionConfig,
    PlotLibrary,
)

if TYPE_CHECKING:
    from .base import ContainerProtocol


class JavaScriptHandler(AbstractLanguageHandler):
    """Handler for JavaScript/NodeJS."""

    def __init__(self) -> None:
        """Initialize the JavaScript handler."""
        config = LanguageConfig(
            name="javascript",
            file_extension="js",
            execution_commands=["node {file}"],
            package_manager="yarn add",
            plot_detection=PlotDetectionConfig(
                libraries=[PlotLibrary.CHARTJS, PlotLibrary.D3JS, PlotLibrary.PLOTLY],
                output_formats=["png", "svg", "html"],
                detection_patterns=["chart.save(", "d3.select(", "Plotly.newPlot("],
                cleanup_code="",
            ),
            supported_libraries=["chart.js", "d3", "plotly.js", "canvas", "jsdom"],
        )
        super().__init__(config)

    def inject_plot_detection_code(self, code: str) -> str:
        """Inject plot detection for JavaScript"""
        setup_code = """
// JavaScript plot detection setup
const fs = require('fs');
const path = require('path');

// Setup output directories
const plotsDir = '/tmp/sandbox_plots';
const outputDir = '/tmp/sandbox_output';

if (!fs.existsSync(plotsDir)) fs.mkdirSync(plotsDir, { recursive: true });
if (!fs.existsSync(outputDir)) fs.mkdirSync(outputDir, { recursive: true });

let plotCounter = 0;

// === CHART.JS SUPPORT (with canvas) ===
try {
    const { createCanvas } = require('canvas');
    const Chart = require('chart.js/auto');

    // Override Chart creation to capture charts
    const originalChart = Chart;
    global.Chart = class extends originalChart {
        constructor(ctx, config) {
            super(ctx, config);
            this._capture_chart();
        }

        _capture_chart() {
            setTimeout(() => {
                try {
                    if (this.canvas) {
                        const buffer = this.canvas.toBuffer('image/png');
                        const filename = path.join(plotsDir, `chartjs_plot_${plotCounter}.png`);
                        fs.writeFileSync(filename, buffer);

                        // Save metadata
                        const metadata = {
                            library: 'chartjs',
                            plot_id: plotCounter,
                            type: this.config.type,
                            title: this.config.options?.title?.text || null
                        };

                        fs.writeFileSync(
                            path.join(plotsDir, `chartjs_plot_${plotCounter}_meta.json`),
                            JSON.stringify(metadata, null, 2)
                        );

                        plotCounter++;
                    }
                } catch (e) {
                    console.error('Chart.js capture error:', e);
                }
            }, 100);
        }
    };

} catch (e) {
    // Chart.js not available
}

// === D3.JS SUPPORT (with jsdom) ===
try {
    const { JSDOM } = require('jsdom');
    const d3 = require('d3');

    // Create virtual DOM
    const dom = new JSDOM('<!DOCTYPE html><body></body>');
    global.window = dom.window;
    global.document = dom.window.document;

    // Override d3 select to track SVG creation
    const originalSelect = d3.select;
    d3.select = function(selector) {
        const selection = originalSelect.call(this, selector);

        // If creating SVG, set up capture
        if (selector === 'body' || selector === document.body) {
            const originalAppend = selection.append;
            selection.append = function(elementType) {
                const element = originalAppend.call(this, elementType);

                if (elementType === 'svg') {
                    // Setup SVG capture after some delay
                    setTimeout(() => {
                        try {
                            const svgElement = element.node();
                            if (svgElement) {
                                const svgString = new XMLSerializer().serializeToString(svgElement);
                                const filename = path.join(plotsDir, `d3_plot_${plotCounter}.svg`);
                                fs.writeFileSync(filename, svgString);
                                plotCounter++;
                            }
                        } catch (e) {
                            console.error('D3.js capture error:', e);
                        }
                    }, 500);
                }

                return element;
            };
        }

        return selection;
    };

} catch (e) {
    // D3.js or jsdom not available
}

// === PLOTLY.JS SUPPORT ===
try {
    const Plotly = require('plotly.js');

    // Override Plotly.newPlot
    const originalNewPlot = Plotly.newPlot;
    Plotly.newPlot = function(graphDiv, data, layout, config) {
        const result = originalNewPlot.call(this, graphDiv, data, layout, config);

        // Capture plot
        setTimeout(() => {
            try {
                const filename = path.join(plotsDir, `plotly_plot_${plotCounter}.html`);

                // Create HTML with plot
                const html = `
<!DOCTYPE html>
<html>
<head>
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
</head>
<body>
    <div id="plot" style="width:100%;height:500px;"></div>
    <script>
        Plotly.newPlot('plot', ${JSON.stringify(data)}, ${JSON.stringify(layout)});
    </script>
</body>
</html>`;

                fs.writeFileSync(filename, html);
                plotCounter++;
            } catch (e) {
                console.error('Plotly.js capture error:', e);
            }
        }, 100);

        return result;
    };

} catch (e) {
    // Plotly.js not available
}

console.log('JavaScript plot detection setup complete');
"""

        return setup_code + "\n\n" + code

    def extract_plots(
        self,
        container: "ContainerProtocol",
        output_dir: str,  # noqa: ARG002
    ) -> list[PlotOutput]:
        """Extract plots from JavaScript execution."""
        return self._extract_files_from_path(container, "/tmp/sandbox_plots")

    def extract_files(
        self,
        container: "ContainerProtocol",
        output_dir: str,  # noqa: ARG002
    ) -> list[FileOutput]:
        """Extract files from JavaScript execution."""
        return self._extract_files_from_path(container, "/tmp/sandbox_output")
