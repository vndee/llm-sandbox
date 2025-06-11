import logging

from podman import PodmanClient

from llm_sandbox import SandboxBackend, SandboxSession

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

logger = logging.getLogger(__name__)

client = PodmanClient(
    base_url="unix:///var/folders/lh/rjbzw60n1fv7xr9kffn7gr840000gn/T/podman/podman-machine-default-api.sock"
)


def test_simple_shell_commands() -> None:
    """Test simple shell commands that should return non-zero exit codes."""
    logger.info("=== Testing simple shell commands ===")

    with SandboxSession(
        lang="python", keep_template=True, verbose=True, backend=SandboxBackend.PODMAN, client=client
    ) as session:
        # Test 1: Command that should return exit code 1
        result = session.execute_command("false")
        logger.info("false command result: %s", result)

        # Test 2: Command that should return exit code 2
        result = session.execute_command("exit 2")
        logger.info("exit 2 command result: %s", result)

        # Test 3: Non-existent command
        result = session.execute_command("nonexistent_command_xyz")
        logger.info("nonexistent command result: %s", result)


def test_python_direct_execution() -> None:
    """Test Python executed directly vs through virtual environment."""
    logger.info("=== Testing Python direct vs venv execution ===")

    with SandboxSession(
        lang="python", keep_template=True, verbose=True, backend=SandboxBackend.PODMAN, client=client
    ) as session:
        # Test 1: Direct python execution with error
        result = session.execute_command("python -c 'raise ValueError()'")
        logger.info("Direct python error result: %s", result)

        # Test 2: Direct python execution with exit code
        result = session.execute_command("python -c 'import sys; sys.exit(5)'")
        logger.info("Direct python sys.exit(5) result: %s", result)

        # Test 3: Venv python execution with error
        result = session.execute_command("/tmp/venv/bin/python -c 'raise ValueError()'")
        logger.info("Venv python error result: %s", result)

        # Test 4: Venv python execution with exit code
        result = session.execute_command("/tmp/venv/bin/python -c 'import sys; sys.exit(3)'")
        logger.info("Venv python sys.exit(3) result: %s", result)


def test_python_code_through_run() -> None:
    """Test Python code through the run() method."""
    logger.info("=== Testing Python code through run() method ===")

    with SandboxSession(
        lang="python", keep_template=True, verbose=True, backend=SandboxBackend.PODMAN, client=client
    ) as session:
        # Test 1: Exception in run()
        result = session.run("raise ValueError()")
        logger.info("run() with ValueError result: %s", result)

        # Test 2: sys.exit() in run()
        result = session.run("import sys; sys.exit(7)")
        logger.info("run() with sys.exit(7) result: %s", result)


if __name__ == "__main__":
    test_simple_shell_commands()
    test_python_direct_execution()
    test_python_code_through_run()
