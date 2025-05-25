"""Debug exit codes for virtual environment tests."""

import logging

from llm_sandbox import SandboxSession

logging.basicConfig(
    level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(message)s"
)

logger = logging.getLogger(__name__)


def debug_exit_codes() -> None:
    """Debug what exit codes we're getting."""
    logger.info("=== Debugging Exit Codes ===")

    with SandboxSession(backend="docker", lang="python", verbose=True) as session:
        result = session.run("print('Test successful!')", libraries=["urllib3"])
        logger.info("Exit code: %s", result.exit_code)
        logger.info("Exit code type: %s", type(result.exit_code))
        logger.info("Stdout: %s", result.stdout)
        logger.info("Stderr: %s", result.stderr)
        logger.info("Comparison result.exit_code == 0: %s", result.exit_code == 0)
        logger.info("Comparison result.exit_code is None: %s", result.exit_code is None)


if __name__ == "__main__":
    debug_exit_codes()
