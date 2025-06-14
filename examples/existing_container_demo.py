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
    logger.info("🚀 Creating a container with custom setup...")
    
    # Create a new container with custom environment
    with SandboxSession(
        lang="python",
        verbose=True,
        keep_template=True,
        image="ghcr.io/vndee/sandbox-python-311-bullseye"
    ) as session:
        # Install some packages and setup environment
        logger.info("📦 Installing packages...")
        session.install(["numpy", "pandas", "matplotlib"])
        
        # Create some files
        logger.info("📁 Setting up files...")
        session.execute_command("echo 'print(\"Hello from existing container!\")' > /sandbox/hello.py")
        session.execute_command("echo 'Custom environment data' > /sandbox/data.txt")
        
        # Get container ID before closing
        container_id = session.container.id
        logger.info(f"✅ Container created with ID: {container_id[:12]}...")
        
        # Important: We need to keep the container running
        # In practice, you would have a container already running
        # For demo purposes, we'll use commit to save state
        session.commit_container = True
        
    return container_id


def demo_connect_to_existing_docker_container(container_id: str) -> None:
    """Demo connecting to an existing Docker container."""
    logger.info("\n" + "="*60)
    logger.info("🐳 Demo: Connecting to Existing Docker Container")
    logger.info("="*60)
    
    try:
        # Connect to existing container - no environment setup needed
        with SandboxSession(
            container_id=container_id,
            lang="python",
            verbose=True
        ) as session:
            logger.info("✅ Connected to existing container successfully!")
            
            # Run code that uses pre-installed packages
            result = session.run("""
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
            print(result.stdout)
            
            # Execute the pre-existing script
            result = session.execute_command("python /sandbox/hello.py")
            logger.info("📝 Pre-existing script output:")
            print(result.stdout)
            
            # List files to show existing content
            result = session.execute_command("ls -la /sandbox/")
            logger.info("📁 Container contents:")
            print(result.stdout)
            
    except ContainerError as e:
        logger.error(f"❌ Failed to connect to container: {e}")
    except Exception as e:
        logger.error(f"❌ Unexpected error: {e}")


def demo_connect_to_existing_kubernetes_pod() -> None:
    """Demo connecting to an existing Kubernetes pod."""
    logger.info("\n" + "="*60)
    logger.info("☸️  Demo: Connecting to Existing Kubernetes Pod")
    logger.info("="*60)
    
    # For demo purposes, we'll create a pod first
    # In practice, you would have a pod already running
    try:
        # First create a pod (simulating existing pod)
        logger.info("📦 Creating a demo pod (simulating existing pod)...")
        with SandboxSession(
            backend=SandboxBackend.KUBERNETES,
            lang="python",
            verbose=True
        ) as session:
            pod_id = session.container  # Get pod name
            
            # Setup some environment
            session.execute_command("echo 'Pod environment ready' > /sandbox/pod_info.txt")
            logger.info(f"✅ Demo pod created: {pod_id}")
            
            # Now connect to the "existing" pod
            logger.info(f"🔗 Connecting to existing pod: {pod_id}")
            
        # Connect to the existing pod
        with SandboxSession(
            backend=SandboxBackend.KUBERNETES,
            container_id=pod_id,  # Connect to existing pod
            lang="python",
            verbose=True
        ) as session:
            logger.info("✅ Connected to existing pod successfully!")
            
            # Run code in the existing pod
            result = session.run("""
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
            print(result.stdout)
            
    except ContainerError as e:
        logger.error(f"❌ Failed to connect to pod: {e}")
    except Exception as e:
        logger.error(f"❌ Error in Kubernetes demo (cluster may not be available): {e}")


def demo_connect_to_existing_podman_container() -> None:
    """Demo connecting to an existing Podman container."""
    logger.info("\n" + "="*60)
    logger.info("🦭 Demo: Connecting to Existing Podman Container")
    logger.info("="*60)
    
    try:
        from podman import PodmanClient
        
        client = PodmanClient()
        
        # First create a container (simulating existing container)
        logger.info("📦 Creating a demo Podman container...")
        with SandboxSession(
            backend=SandboxBackend.PODMAN,
            client=client,
            lang="python",
            verbose=True,
            keep_template=True
        ) as session:
            container_id = session.container.id
            
            # Setup some environment
            session.execute_command("echo 'Podman environment ready' > /sandbox/podman_info.txt")
            logger.info(f"✅ Demo Podman container created: {container_id[:12]}...")
            
        # Connect to the existing container
        logger.info(f"🔗 Connecting to existing Podman container...")
        with SandboxSession(
            backend=SandboxBackend.PODMAN,
            client=client,
            container_id=container_id,
            lang="python",
            verbose=True
        ) as session:
            logger.info("✅ Connected to existing Podman container successfully!")
            
            # Run code in the existing container
            result = session.run("""
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
            print(result.stdout)
            
    except ImportError:
        logger.warning("⚠️  Podman not available, skipping Podman demo")
    except ContainerError as e:
        logger.error(f"❌ Failed to connect to Podman container: {e}")
    except Exception as e:
        logger.error(f"❌ Error in Podman demo: {e}")


def demo_error_handling() -> None:
    """Demo error handling when connecting to non-existent containers."""
    logger.info("\n" + "="*60)
    logger.info("🛡️  Demo: Error Handling")
    logger.info("="*60)
    
    # Try connecting to non-existent container
    logger.info("🧪 Testing connection to non-existent container...")
    try:
        with SandboxSession(
            container_id="non-existent-container-id",
            lang="python",
            verbose=True
        ) as session:
            session.run("print('This should not work')")
            
    except ContainerError as e:
        logger.info(f"✅ Correctly caught ContainerError: {e}")
    except Exception as e:
        logger.error(f"❌ Unexpected error type: {e}")
    
    # Try with invalid pod name
    logger.info("🧪 Testing connection to non-existent pod...")
    try:
        with SandboxSession(
            backend=SandboxBackend.KUBERNETES,
            container_id="non-existent-pod",
            lang="python",
            verbose=True
        ) as session:
            session.run("print('This should not work')")
            
    except ContainerError as e:
        logger.info(f"✅ Correctly caught ContainerError for K8s: {e}")
    except Exception as e:
        logger.warning(f"⚠️  K8s error (cluster may not be available): {e}")


def main() -> None:
    """Run all demos."""
    logger.info("🎯 LLM Sandbox - Existing Container Support Demo")
    logger.info("=" * 80)
    logger.info("This demo shows how to connect to existing containers/pods")
    logger.info("instead of creating new ones from scratch.")
    logger.info("=" * 80)
    
    # Demo 1: Create a container for demonstration
    container_id = create_and_setup_container()
    
    # Give container a moment to settle
    time.sleep(2)
    
    # Demo 2: Connect to existing Docker container
    demo_connect_to_existing_docker_container(container_id)
    
    # Demo 3: Connect to existing Kubernetes pod
    demo_connect_to_existing_kubernetes_pod()
    
    # Demo 4: Connect to existing Podman container
    demo_connect_to_existing_podman_container()
    
    # Demo 5: Error handling
    demo_error_handling()
    
    logger.info("\n" + "="*80)
    logger.info("🎉 Demo completed!")
    logger.info("Key benefits of existing container support:")
    logger.info("• 🚀 Faster startup (no environment setup)")
    logger.info("• 🔧 Work with pre-configured environments")
    logger.info("• 🔄 Reuse containers across multiple sessions")
    logger.info("• 🐛 Connect to running containers for debugging")
    logger.info("• 📦 Integrate with external container management")
    logger.info("="*80)


if __name__ == "__main__":
    main()
