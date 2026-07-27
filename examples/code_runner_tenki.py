import logging

from llm_sandbox import SandboxBackend, SandboxSession

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

logger = logging.getLogger(__name__)

WORKDIR = "/home/tenki"


def run_python_code() -> None:
    """Run Python code in a Tenki sandbox."""
    with SandboxSession(
        backend=SandboxBackend.TENKI,
        lang="python",
        verbose=True,
        runtime_configs={"allow_outbound": True},
    ) as session:
        output = session.run("print('Hello from Tenki!')")
        logger.info(output)

        output = session.run("import numpy as np\nprint(np.random.rand())", libraries=["numpy"])
        logger.info(output)

        session.install(["pandas"])
        output = session.run("import pandas as pd\nprint(pd.__version__)")
        logger.info(output)

        session.copy_to_runtime("README.md", f"{WORKDIR}/data.csv")
        output = session.run(f"print(open('{WORKDIR}/data.csv').readline())")
        logger.info(output)


if __name__ == "__main__":
    run_python_code()
