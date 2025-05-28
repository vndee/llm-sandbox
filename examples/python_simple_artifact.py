# ruff: noqa: T201

import base64
from pathlib import Path

from llm_sandbox import ArtifactSandboxSession

code = """
import matplotlib.pyplot as plt
import numpy as np

plt.style.use('default')

# Generate data
x = np.linspace(0, 10, 100)
y1 = np.sin(x) + np.random.normal(0, 0.1, 100)
y2 = np.cos(x) + np.random.normal(0, 0.1, 100)

# Create plot
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

print('Plot generated successfully!')
"""

# Create a sandbox session
with ArtifactSandboxSession(lang="python", verbose=True) as session:
    # Run Python code safely
    result = session.run(code)

    print(result.stdout)  # Output: Plot generated successfully!

    for plot in result.plots:
        with Path("docs/assets/example.png").open("wb") as f:
            f.write(base64.b64decode(plot.content_base64))
