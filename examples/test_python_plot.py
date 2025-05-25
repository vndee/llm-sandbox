import logging

from llm_sandbox.session import ArtifactSandboxSession

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

code = """
import matplotlib.pyplot as plt
import numpy as np

x = np.linspace(0, 10, 100)
y = np.sin(x)

plt.plot(x, y)
plt.title('Sine Wave')
plt.show()  # Automatically captured
"""

with ArtifactSandboxSession(lang="python", verbose=True) as session:
    result = session.run(code)
    logger.info("Captured %d plots", len(result.plots))
    if len(result.plots) > 0:
        logger.info("Plot: %s", result)
