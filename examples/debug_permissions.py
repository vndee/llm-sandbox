"""Debug script to understand permission issues."""

import logging

from llm_sandbox import SandboxSession

logging.basicConfig(
    level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(message)s"
)

logger = logging.getLogger(__name__)


def debug_file_permissions() -> None:
    """Debug what's happening with file permissions."""
    logger.info("=== Debugging File Permissions ===")

    with SandboxSession(
        lang="python",
        runtime_configs={"user": "1000:1000"},
        workdir="/tmp/sandbox",
        verbose=True,
    ) as session:
        # Check current user
        result = session.execute_command("id")
        logger.info("Current user info: %s", result.stdout.strip())

        # Check if user 1000 exists
        result = session.execute_command(
            "getent passwd 1000 2>/dev/null || echo 'User 1000 does not exist'"
        )
        logger.info("User 1000 existence: %s", result.stdout.strip())

        # Try to run some code to see where it fails
        logger.info("\n=== Attempting to run code ===")
        try:
            result = session.run("print('Hello from user 1000!')")
            logger.info("Code execution result: %s", result.stdout.strip())
            logger.info("Any errors: %s", result.stderr.strip())
        except Exception as e:
            logger.info("Code execution failed: %s", e)

        # Check workdir after attempting to run code
        result = session.execute_command("ls -la /tmp/")
        logger.info("Contents of /tmp/ after run attempt:\n%s", result.stdout)

        # Check if sandbox dir exists and its permissions
        result = session.execute_command(
            "test -d /tmp/sandbox && ls -la /tmp/sandbox/ || echo 'Directory does not exist'"
        )
        logger.info("Contents of /tmp/sandbox/:\n%s", result.stdout)

        # Check specific file if it exists
        result = session.execute_command(
            "test -f /tmp/sandbox/code.py && ls -la /tmp/sandbox/code.py || echo 'code.py does not exist'"
        )
        logger.info("code.py file info:\n%s", result.stdout)

        # Try to manually create directory and see if we can write
        logger.info("\n=== Manual directory creation test ===")
        result = session.execute_command(
            "mkdir -p /tmp/sandbox_test && echo 'Directory created successfully'"
        )
        logger.info("Manual mkdir result: %s", result.stdout.strip())

        result = session.execute_command(
            "echo 'test content' > /tmp/sandbox_test/test.txt && echo 'File created successfully'"
        )
        logger.info("Manual file creation: %s", result.stdout.strip())

        result = session.execute_command("ls -la /tmp/sandbox_test/")
        logger.info("Manual test directory contents:\n%s", result.stdout)


if __name__ == "__main__":
    debug_file_permissions()
