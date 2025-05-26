"""Comprehensive test for pip warnings across all backends."""

import logging

from llm_sandbox import SandboxSession

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

logger = logging.getLogger(__name__)


def test_docker_pip() -> None:
    """Test Docker backend with virtual environment."""
    logger.info("=== Testing Docker Backend ===")

    try:
        with SandboxSession(backend="docker", lang="python", verbose=True) as session:
            result = session.run("print('Docker virtual env test!')", libraries=["urllib3"])
            logger.info("Docker result: %s", result.stdout.strip())
            logger.info("Docker errors: %s", result.stderr.strip())

            has_output = "Docker virtual env test!" in result.stdout
            no_critical_errors = "Error" not in result.stderr and "Failed" not in result.stderr
            success = has_output and no_critical_errors

            logger.info("✅ Docker: SUCCESS" if success else "❌ Docker: FAILED")
    except Exception:
        logger.exception("❌ Docker test failed")


def test_podman_pip() -> None:
    """Test Podman backend with virtual environment."""
    logger.info("\n=== Testing Podman Backend ===")

    try:
        from podman import PodmanClient

        client = PodmanClient(
            base_url="unix:///var/folders/lh/rjbzw60n1fv7xr9kffn7gr840000gn/T/podman/podman-machine-default-api.sock"
        )

        with SandboxSession(
            backend="podman", lang="python", verbose=True, client=client
        ) as session:
            result = session.run("print('Podman virtual env test!')", libraries=["urllib3"])
            logger.info("Podman result: %s", result.stdout.strip())
            logger.info("Podman errors: %s", result.stderr.strip())

            # Check for success
            has_output = "Podman virtual env test!" in result.stdout
            no_critical_errors = "Error" not in result.stderr and "Failed" not in result.stderr
            success = has_output and no_critical_errors

            logger.info("✅ Podman: SUCCESS" if success else "❌ Podman: FAILED")
    except Exception:
        logger.exception("❌ Podman test skipped")


def test_kubernetes_pip() -> None:
    """Test Kubernetes backend with virtual environment."""
    logger.info("\n=== Testing Kubernetes Backend ===")

    try:
        with SandboxSession(backend="kubernetes", lang="python", verbose=True) as session:
            result = session.run("print('Kubernetes virtual env test!')", libraries=["urllib3"])
            logger.info("Kubernetes result: %s", result.stdout.strip())
            logger.info("Kubernetes errors: %s", result.stderr.strip())

            # Check for success
            has_output = "Kubernetes virtual env test!" in result.stdout
            no_critical_errors = "Error" not in result.stderr and "Failed" not in result.stderr
            success = (
                (result.exit_code == 0 or result.exit_code is None)
                and has_output
                and no_critical_errors
            )

            logger.info("✅ Kubernetes: SUCCESS" if success else "❌ Kubernetes: FAILED")
    except Exception:
        logger.exception("❌ Kubernetes test skipped")


def test_custom_user_docker() -> None:
    """Test Docker with custom user and virtual environment."""
    logger.info("\n=== Testing Docker Custom User ===")

    try:
        with SandboxSession(
            backend="docker",
            lang="python",
            runtime_configs={"user": "1000:1000"},
            workdir="/tmp/sandbox",
            verbose=True,
        ) as session:
            result = session.run(
                "print('Docker custom user virtual env test!')", libraries=["urllib3"]
            )
            logger.info("Docker custom user result: %s", result.stdout.strip())
            logger.info("Docker custom user errors: %s", result.stderr.strip())

            # Check for success
            has_output = "Docker custom user virtual env test!" in result.stdout
            no_critical_errors = "Error" not in result.stderr and "Failed" not in result.stderr
            success = has_output and no_critical_errors

            logger.info(
                "✅ Docker Custom User: SUCCESS" if success else "❌ Docker Custom User: FAILED"
            )
    except Exception:
        logger.exception("❌ Docker custom user test failed")


if __name__ == "__main__":
    logger.info("🧪 Comprehensive Virtual Environment Test Across All Backends")
    logger.info("=" * 60)

    test_docker_pip()
    test_podman_pip()
    test_kubernetes_pip()
    test_custom_user_docker()

    logger.info("🎯 All backend tests completed!")
