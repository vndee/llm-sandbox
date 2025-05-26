"""Test script to verify pip warnings are suppressed."""

import logging

from llm_sandbox import SandboxSession

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

logger = logging.getLogger(__name__)


def test_pip_no_warning() -> None:
    """Test that pip doesn't show root user warnings."""
    logger.info("=== Testing Pip Root User Warning Suppression ===")

    with SandboxSession(lang="python", verbose=True) as session:
        # Test installing a simple package
        try:
            result = session.run(
                "print('Testing package installation...')",
                libraries=["requests"],  # Common package that's likely not pre-installed
            )
            logger.info("Installation result: %s", result.stdout.strip())
            logger.info("Any warnings/errors: %s", result.stderr.strip())

            # Verify the package can be imported
            result = session.run(
                """
import requests
print("requests imported successfully!")
print(f"requests version: {requests.__version__}")
"""
            )
            logger.info("Import test: %s", result.stdout.strip())

        except Exception:
            logger.exception("Test failed")


def test_custom_user_pip() -> None:
    """Test pip with custom user (should still work)."""
    logger.info("\n=== Testing Pip with Custom User ===")

    try:
        with SandboxSession(
            lang="python",
            runtime_configs={"user": "1000:1000"},
            workdir="/tmp/sandbox",
            verbose=True,
        ) as session:
            # Test installing a package as non-root user
            result = session.run(
                "print('Testing non-root package installation...')",
                libraries=["urllib3"],
            )
            logger.info("Custom user installation: %s", result.stdout.strip())
            logger.info("Any errors: %s", result.stderr.strip())

    except Exception:
        logger.exception("Custom user test failed")


if __name__ == "__main__":
    test_pip_no_warning()
    test_custom_user_pip()
    logger.info("\n=== Pip warning test completed ===")
