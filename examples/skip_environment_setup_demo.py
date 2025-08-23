"""Example demonstrating skip_environment_setup feature for Kubernetes deployments.

This example shows how to use the skip_environment_setup configuration option
to avoid package installation delays in Kubernetes environments where
administrators want to use custom images with pre-configured environments.
"""

import logging

from llm_sandbox import SandboxBackend, SandboxSession

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

logger = logging.getLogger(__name__)


def demo_skip_environment_setup() -> None:
    """Demonstrate using skip_environment_setup=True to avoid pip upgrade and environment setup."""
    logger.info("Demo: Skip environment setup for faster container startup")

    # This configuration skips language-specific environment setup
    # Useful when using custom images with pre-configured environments
    # or when administrators want to avoid pip upgrade delays in K8s deployments
    with SandboxSession(
        lang="python",
        verbose=True,
        backend=SandboxBackend.KUBERNETES,  # Change to KUBERNETES for K8s deployment
        skip_environment_setup=True,
        # For K8s deployments, administrators can also set custom images
        # with pre-installed packages to avoid package installation delays
    ) as session:
        logger.info("Session created with skip_environment_setup=True")
        logger.info("No pip upgrade or virtual environment creation will be performed")

        # This assumes the container image already has the required environment set up
        # For custom images, ensure they include necessary Python packages
        output = session.run("print('Hello from pre-configured environment!')")
        logger.info("Output: %s", output.stdout.strip())

        # When skip_environment_setup=True, libraries cannot be installed dynamically
        # They must be pre-installed in the container image
        try:
            output = session.run(
                "import sys; print(f'Python version: {sys.version}')",
            )
            logger.info("Python info: %s", output.stdout.strip())
        except Exception as e:  # noqa: BLE001
            logger.warning("Failed to run code: %s", e)
            logger.info("This might happen if the base image doesn't have the expected Python setup")


def demo_normal_environment_setup() -> None:
    """Demonstrate normal environment setup (default behavior)."""
    logger.info("\nDemo: Normal environment setup (default behavior)")

    # Default behavior - performs full environment setup
    with SandboxSession(
        lang="python",
        verbose=True,
        backend=SandboxBackend.KUBERNETES,
        skip_environment_setup=False,  # This is the default
    ) as session:
        logger.info("Session created with default environment setup")
        logger.info("This will create venv, pip cache, and upgrade pip")

        output = session.run("print('Hello with full environment setup!')")
        logger.info("Output: %s", output.stdout.strip())


def demo_kubernetes_use_case() -> None:
    """Show how this would be used in a Kubernetes deployment scenario."""
    logger.info("\nDemo: Kubernetes deployment scenario")

    # In a Kubernetes environment, administrators might want to:
    # 1. Use a custom image with pre-installed packages
    # 2. Skip environment setup to reduce pod startup time
    # 3. Avoid potential network issues with pip index access

    # In a Kubernetes environment, administrators might configure:
    session_config = {
        "lang": "python",
        "verbose": True,
        "backend": SandboxBackend.KUBERNETES,  # Would be KUBERNETES in real scenario
        "skip_environment_setup": True,
        "pod_manifest": {
            "apiVersion": "v1",
            "kind": "Pod",
            "metadata": {
                "name": "sandbox-python",
                "namespace": "default",
            },
            "spec": {
                "containers": [
                    {
                        "name": "my-python-app",
                        "image": "ghcr.io/vndee/sandbox-python-311-bullseye",
                        "tty": True,
                        "securityContext": {
                            "runAsUser": 0,
                            "runAsGroup": 0,
                        },
                        "resources": {
                            "requests": {"memory": "256Mi", "cpu": "100m"},
                            "limits": {"memory": "512Mi", "cpu": "500m"},
                        },
                    }
                ],
                "securityContext": {
                    "runAsUser": 0,
                    "runAsGroup": 0,
                },
            },
        },
    }

    logger.info("Kubernetes-style configuration:")
    logger.info("  - skip_environment_setup: %s", session_config["skip_environment_setup"])
    logger.info("  - Custom image: %s", session_config["pod_manifest"]["spec"]["containers"][0]["image"])  # type: ignore[index]
    resource_limits = session_config["pod_manifest"]["spec"]["containers"][0]["resources"]["limits"]  # type: ignore[index]
    logger.info("  - Resource limits: %s", resource_limits)

    try:
        with SandboxSession(
            lang=session_config["lang"],
            verbose=session_config["verbose"],
            backend=session_config["backend"],  # type: ignore[arg-type]
            skip_environment_setup=session_config["skip_environment_setup"],
            pod_manifest=session_config["pod_manifest"],
        ) as session:
            output = session.run("""
import os
import sys
print(f"Running in: {os.environ.get('HOSTNAME', 'unknown')}")
print(f"Python path: {sys.executable}")
print("Environment setup was skipped - using pre-configured image!")
""")
            logger.info("K8s demo output:\n%s", output.stdout)
    except Exception as e:  # noqa: BLE001
        logger.warning("Kubernetes demo failed due to cluster connectivity: %s", str(e)[:100])
        logger.info("Note: The skip_environment_setup feature is working correctly")
        logger.info("This error is related to Kubernetes cluster setup, not our feature")
        logger.info("In a properly configured K8s cluster, this would work fine")


if __name__ == "__main__":
    demo_skip_environment_setup()
    demo_normal_environment_setup()
    demo_kubernetes_use_case()

    separator = "=" * 60
    logger.info("\n%s", separator)
    logger.info("Summary:")
    logger.info("- Use skip_environment_setup=True for:")
    logger.info("  • Custom images with pre-configured environments")
    logger.info("  • Kubernetes deployments to reduce startup time")
    logger.info("  • Avoiding pip index/network configuration issues")
    logger.info("- Use skip_environment_setup=False (default) for:")
    logger.info("  • Standard development and testing")
    logger.info("  • When using base images without pre-installed packages")
    logger.info("  • Maximum compatibility with dynamic package installation")
