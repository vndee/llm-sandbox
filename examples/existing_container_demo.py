"""Example demonstrating connecting to existing containers/pods.

This example shows how to use LLM Sandbox with existing containers instead of creating new ones.
This is useful for:
- Reusing containers with complex setups
- Working with long-running services
- Debugging and troubleshooting
- Connecting to containers managed by external systems
"""

import logging
import time

from docker import DockerClient

from llm_sandbox import SandboxBackend, SandboxSession
from llm_sandbox.exceptions import ContainerError

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def create_and_setup_container() -> str:
    """Create a container with custom setup and return its ID.

    This simulates having an existing container with custom environment.

    Returns:
        str: Container ID

    """
    client = DockerClient(base_url="unix:///Users/vndee/.docker/run/docker.sock")
    logger.info("🚀 Creating a container with custom setup...")

    # Create a new container with custom environment
    # Use commit_container=True to save the container state
    sandbox = SandboxSession(
        client=client,
        lang="python",
        verbose=True,
        image="ghcr.io/vndee/sandbox-python-311-bullseye",
    )

    sandbox.open()

    # Install some packages and setup environment
    logger.info("📦 Installing packages...")
    sandbox.install(["numpy", "pandas", "matplotlib"])

    # Create some files
    logger.info("📁 Setting up files...")
    # Use Python code to create files instead of shell commands
    sandbox.run("""
# Create hello.py file
with open('/sandbox/hello.py', 'w') as f:
    f.write('print("Hello from existing container!")')

# Create data.txt file
with open('/sandbox/data.txt', 'w') as f:
    f.write('Custom environment data')

print("Files created successfully!")
""")

    # Verify files were created
    result = sandbox.execute_command("ls -la /sandbox/")
    logger.info("📋 Created files:")
    logger.info(result.stdout)

    # Get container ID before closing
    container_id = sandbox.container.id
    logger.info("✅ Container created with ID: %s...", container_id[:12])
    return str(container_id)


def demo_connect_to_existing_docker_container() -> None:
    """Demo connecting to an existing Docker container."""
    # Demo 1: Create a container for demonstration
    try:
        container_id = create_and_setup_container()
    except Exception:
        logger.exception("❌ Failed to create demo container")
        return

    logger.info("\n%s", "=" * 60)
    logger.info("🐳 Demo: Connecting to Existing Docker Container")
    logger.info("%s", "=" * 60)

    try:
        client = DockerClient(base_url="unix:///Users/vndee/.docker/run/docker.sock")
        # Connect to existing container - no environment setup needed
        with SandboxSession(client=client, container_id=container_id, lang="python", verbose=True) as sandbox:
            logger.info("✅ Connected to existing container successfully!")

            # Run code that uses pre-installed packages
            result = sandbox.run("""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

print("Running in existing container!")
print(f"NumPy version: {np.__version__}")
print(f"Pandas version: {pd.__version__}")

# Use the pre-existing data
with open('/sandbox/data.txt', 'r') as f:
    data = f.read().strip()
    print(f"Found existing data: {data}")

# Generate some data and create a simple plot
x = np.linspace(0, 10, 100)
y = np.sin(x)

plt.figure(figsize=(8, 6))
plt.plot(x, y)
plt.title('Plot from Existing Container')
plt.xlabel('X')
plt.ylabel('sin(X)')
plt.grid(True)
plt.savefig('/sandbox/plot.png')
print("Plot saved as /sandbox/plot.png")
""")

            logger.info("📊 Code execution output:")
            logger.info(result.stdout)

            # Execute the pre-existing script
            result = sandbox.execute_command("python /sandbox/hello.py")
            logger.info("📝 Pre-existing script output:")
            logger.info(result.stdout)

            # List files to show existing content
            result = sandbox.execute_command("ls -la /sandbox/")
            logger.info("📁 Container contents:")
            logger.info(result.stdout)

    except ContainerError:
        logger.exception("❌ Failed to connect to container")
    except Exception:
        logger.exception("❌ Unexpected error")


def demo_connect_to_existing_kubernetes_pod() -> None:
    """Demo connecting to an existing Kubernetes pod."""
    logger.info("\n%s", "=" * 60)
    logger.info("☸️  Demo: Connecting to Existing Kubernetes Pod")
    logger.info("%s", "=" * 60)

    # For demo purposes, we'll create a pod first
    # In practice, you would have a pod already running
    try:
        # First create a pod (simulating existing pod)
        logger.info("📦 Creating a demo pod (simulating existing pod)...")
        sandbox = SandboxSession(backend=SandboxBackend.KUBERNETES, lang="python", verbose=True)
        sandbox.open()
        pod_id = sandbox.container  # Get pod name

        # Setup some environment
        sandbox.execute_command("echo 'Pod environment ready' > /sandbox/pod_info.txt")
        logger.info("✅ Demo pod created: %s", pod_id)

        # Now connect to the "existing" pod
        logger.info("🔗 Connecting to existing pod: %s", pod_id)

        # Connect to the existing pod
        with SandboxSession(
            backend=SandboxBackend.KUBERNETES,
            container_id=pod_id,  # Connect to existing pod
            lang="python",
            verbose=True,
        ) as sandbox:
            logger.info("✅ Connected to existing pod successfully!")

            # Run code in the existing pod
            result = sandbox.run("""
import sys
print(f"Python version: {sys.version}")
print("Running in existing Kubernetes pod!")

# Read the existing file
try:
    with open('/sandbox/pod_info.txt', 'r') as f:
        info = f.read().strip()
        print(f"Pod info: {info}")
except FileNotFoundError:
    print("Pod info file not found")

# Show current working directory
import os
print(f"Current directory: {os.getcwd()}")
print(f"Directory contents: {os.listdir('/sandbox')}")
""")

            logger.info("📊 Pod execution output:")
            logger.info(result.stdout)

    except ContainerError:
        logger.exception("❌ Failed to connect to pod")
    except Exception:
        logger.exception("❌ Error in Kubernetes demo (cluster may not be available)")


def demo_connect_to_existing_podman_container() -> None:
    """Demo connecting to an existing Podman container."""
    logger.info("\n%s", "=" * 60)
    logger.info("🦭 Demo: Connecting to Existing Podman Container")
    logger.info("%s", "=" * 60)

    try:
        from podman import PodmanClient

        client = PodmanClient(
            base_url="unix:///var/folders/lh/rjbzw60n1fv7xr9kffn7gr840000gn/T/podman/podman-machine-default-api.sock"
        )

        # First create a container (simulating existing container)
        logger.info("📦 Creating a demo Podman container...")
        sandbox = SandboxSession(
            backend=SandboxBackend.PODMAN, client=client, lang="python", verbose=True, keep_template=True
        )
        sandbox.open()
        container_id = sandbox.container.id

        # Setup some environment
        sandbox.run("""
with open('/sandbox/podman_info.txt', 'w') as f:
    f.write('Podman environment ready')
""")
        logger.info("✅ Demo Podman container created: %s...", container_id[:12])

        # Connect to the existing container
        logger.info("🔗 Connecting to existing Podman container...")
        with SandboxSession(
            backend=SandboxBackend.PODMAN, client=client, container_id=container_id, lang="python", verbose=True
        ) as sandbox:
            logger.info("✅ Connected to existing Podman container successfully!")

            # Run code in the existing container
            result = sandbox.run("""
import platform
print(f"Platform: {platform.platform()}")
print("Running in existing Podman container!")

# Read the existing file
try:
    with open('/sandbox/podman_info.txt', 'r') as f:
        info = f.read().strip()
        print(f"Container info: {info}")
except FileNotFoundError:
    print("Container info file not found")
""")

            logger.info("📊 Podman execution output:")
            logger.info(result.stdout)

    except ImportError:
        logger.warning("⚠️  Podman not available, skipping Podman demo")
    except ContainerError:
        logger.exception("❌ Failed to connect to Podman container")
    except Exception:
        logger.exception("❌ Error in Podman demo")


def demo_error_handling() -> None:
    """Demo error handling when connecting to non-existent containers."""
    logger.info("\n%s", "=" * 60)
    logger.info("🛡️  Demo: Error Handling")
    logger.info("%s", "=" * 60)

    # Try connecting to non-existent container
    logger.info("🧪 Testing connection to non-existent container...")
    try:
        with SandboxSession(container_id="non-existent-container-id", lang="python", verbose=True) as sandbox:
            sandbox.run("print('This should not work')")

    except ContainerError:
        logger.info("✅ Correctly caught ContainerError")
    except Exception:
        logger.exception("❌ Unexpected error type")

    # Try with invalid pod name
    logger.info("🧪 Testing connection to non-existent pod...")
    try:
        with SandboxSession(
            backend=SandboxBackend.KUBERNETES, container_id="non-existent-pod", lang="python", verbose=True
        ) as sandbox:
            sandbox.run("print('This should not work')")

    except ContainerError:
        logger.info("✅ Correctly caught ContainerError for K8s")
    except Exception:
        logger.exception("⚠️  K8s error (cluster may not be available)")


def main() -> None:
    """Run all demos."""
    logger.info("🎯 LLM Sandbox - Existing Container Support Demo")
    logger.info("=" * 80)
    logger.info("This demo shows how to connect to existing containers/pods")
    logger.info("instead of creating new ones from scratch.")
    logger.info("=" * 80)

    # Give container a moment to settle
    time.sleep(2)

    # Demo 1: Connect to existing Docker container
    demo_connect_to_existing_docker_container()

    # Demo 2: Connect to existing Kubernetes pod
    demo_connect_to_existing_kubernetes_pod()

    # # Demo 3: Connect to existing Podman container
    demo_connect_to_existing_podman_container()

    # Demo 4: Error handling
    demo_error_handling()

    logger.info("\n%s", "=" * 80)
    logger.info("🎉 Demo completed!")
    logger.info("Key benefits of existing container support:")
    logger.info("• 🚀 Faster startup (no environment setup)")
    logger.info("• 🔧 Work with pre-configured environments")
    logger.info("• 🔄 Reuse containers across multiple sandboxs")
    logger.info("• 🐛 Connect to running containers for debugging")
    logger.info("• 📦 Integrate with external container management")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
