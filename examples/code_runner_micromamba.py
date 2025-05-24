import logging

import docker

from llm_sandbox.micromamba import MicromambaSession

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

logger = logging.getLogger(__name__)


def run_python_code() -> None:
    """Run Python code in the sandbox."""
    client = docker.DockerClient(base_url="unix:///Users/vndee/.docker/run/docker.sock")

    with MicromambaSession(
        client=client,
        lang="python",
        keep_template=True,
        verbose=True,
        backend="micromamba",
        image="ghcr.io/longevity-genie/just-agents/biosandbox:main",
    ) as session:
        output = session.run("print('Hello, World!')")
        logger.info(output)

        output = session.run(
            "import numpy as np\nprint(np.random.rand())", libraries=["numpy"]
        )
        logger.info(output)

        session.install(["pandas"])
        output = session.run("import pandas as pd\nprint(pd.__version__)")
        logger.info(output)

        session.copy_to_runtime("README.md", "/sandbox/data.csv")


if __name__ == "__main__":
    run_python_code()
