"""Example demonstrating Firecracker backend usage for LLM Sandbox.

This example shows how to use the Firecracker backend for running code
in microVMs, which provides better startup time and minimal resource usage
compared to traditional containers.

Note: This example requires:
1. Firecracker binary installed and in PATH
2. Kernel image (vmlinux) available at /usr/share/firecracker/vmlinux
3. Root filesystem images for each language
4. requests-unixsocket package: pip install requests-unixsocket
"""

from pathlib import Path

from llm_sandbox import SandboxSession
from llm_sandbox.const import SandboxBackend


def run_python_code_in_firecracker():
    """Run Python code in a Firecracker microVM."""
    print("🔥 Running Python code in Firecracker microVM...")
    
    # Python code to execute
    python_code = """
import sys
import os
import platform

print(f"Python version: {sys.version}")
print(f"Platform: {platform.platform()}")
print(f"Architecture: {platform.machine()}")
print(f"Processor: {platform.processor()}")
print(f"Current working directory: {os.getcwd()}")

# Simple computation
numbers = [1, 2, 3, 4, 5]
squared = [x**2 for x in numbers]
print(f"Original numbers: {numbers}")
print(f"Squared numbers: {squared}")

# Test networking (if available)
try:
    import socket
    hostname = socket.gethostname()
    print(f"Hostname: {hostname}")
except Exception as e:
    print(f"Networking error: {e}")
"""

    try:
        with SandboxSession(
            backend=SandboxBackend.FIRECRACKER,
            lang="python",
            verbose=True,
            vcpu_count=1,         # Single CPU core
            mem_size_mib=128,     # 128MB RAM - minimal footprint
            workdir="/sandbox"
        ) as session:
            print("✅ Firecracker microVM started successfully")
            
            # Run the Python code
            result = session.run(python_code)
            
            print(f"\n📋 Execution Results:")
            print(f"Exit Code: {result.exit_code}")
            print(f"Stdout:\n{result.stdout}")
            if result.stderr:
                print(f"Stderr:\n{result.stderr}")
                
            # Test library installation
            print("\n📦 Testing library installation...")
            library_test_code = """
import numpy as np
import requests

print(f"NumPy version: {np.__version__}")
print(f"Requests version: {requests.__version__}")

# Simple NumPy operation
arr = np.array([1, 2, 3, 4, 5])
print(f"Array: {arr}")
print(f"Mean: {np.mean(arr)}")
print(f"Sum: {np.sum(arr)}")
"""
            
            result = session.run(library_test_code, libraries=["numpy", "requests"])
            print(f"Library test exit code: {result.exit_code}")
            print(f"Library test output:\n{result.stdout}")
            
    except Exception as e:
        print(f"❌ Error running Firecracker session: {e}")
        print("\nTroubleshooting tips:")
        print("1. Ensure Firecracker binary is installed: https://github.com/firecracker-microvm/firecracker")
        print("2. Check kernel image exists at /usr/share/firecracker/vmlinux")  
        print("3. Verify rootfs image exists at /usr/share/firecracker/rootfs-python.ext4")
        print("4. Install required dependencies: pip install requests-unixsocket")


def run_with_existing_microvm():
    """Connect to an existing Firecracker microVM."""
    print("\n🔗 Connecting to existing Firecracker microVM...")
    
    # This assumes you have a running microVM with ID 'my-existing-vm'
    # In practice, you would get this ID from your microVM management system
    existing_vm_id = "my-existing-vm"
    
    try:
        with SandboxSession(
            backend=SandboxBackend.FIRECRACKER,
            container_id=existing_vm_id,  # Connect to existing microVM
            lang="python",
            verbose=True
        ) as session:
            print(f"✅ Connected to existing microVM: {existing_vm_id}")
            
            # Run some code in the existing environment
            code = """
import os
print("Running in existing microVM!")
print(f"Process ID: {os.getpid()}")
print(f"Environment variables: {len(os.environ)}")
"""
            
            result = session.run(code)
            print(f"Exit Code: {result.exit_code}")
            print(f"Output:\n{result.stdout}")
            
    except Exception as e:
        print(f"❌ Could not connect to existing microVM: {e}")
        print("Note: This example requires a pre-existing running microVM")


def demonstrate_firecracker_benefits():
    """Demonstrate the benefits of Firecracker microVMs."""
    print("\n🚀 Firecracker MicroVM Benefits:")
    print("=" * 50)
    
    benefits = [
        "⚡ Fast startup: microVMs boot in ~125ms",
        "💾 Low memory overhead: <5MB per microVM",
        "🔒 Strong isolation: Hardware-level virtualization",
        "📊 High density: 150+ microVMs per second per host",
        "🛡️ Security: Process-level and VM-level isolation",
        "🎯 Serverless optimized: Designed for AWS Lambda/Fargate",
        "⚙️ Minimal attack surface: Reduced guest kernel",
        "🔥 Resource efficiency: No container runtime overhead"
    ]
    
    for benefit in benefits:
        print(f"  {benefit}")
    
    print("\n📋 Use Cases:")
    use_cases = [
        "Serverless function execution",
        "Multi-tenant code execution",
        "CI/CD pipeline isolation",
        "Security-sensitive workloads",
        "Edge computing applications",
        "Development sandboxes"
    ]
    
    for use_case in use_cases:
        print(f"  • {use_case}")


def main():
    """Main function to run all examples."""
    print("🔥 Firecracker Backend Examples for LLM Sandbox")
    print("=" * 55)
    
    # Check if Firecracker binary exists
    firecracker_path = Path("/usr/bin/firecracker")
    if not firecracker_path.exists():
        print("⚠️  Firecracker binary not found at /usr/bin/firecracker")
        print("Please install Firecracker first: https://github.com/firecracker-microvm/firecracker")
        return
    
    # Demonstrate benefits first
    demonstrate_firecracker_benefits()
    
    # Run basic Python example
    run_python_code_in_firecracker()
    
    # Show existing microVM connection (will fail but demonstrates usage)
    run_with_existing_microvm()
    
    print("\n✨ Firecracker examples completed!")
    print("\nNext steps:")
    print("1. Set up kernel and rootfs images for production use")
    print("2. Configure networking and security policies")
    print("3. Integrate with your application infrastructure")


if __name__ == "__main__":
    main()