#!/usr/bin/env python3
"""Test script to demonstrate root user functionality in llm-sandbox."""

from llm_sandbox import SandboxSession


def test_default_root_user():
    """Test that containers run as root by default."""
    print("=== Testing Default Root User ===")

    with SandboxSession(lang="python", verbose=True) as session:
        # Test 1: Check if running as root
        result = session.run(
            "import os; print(f'UID: {os.getuid()}, GID: {os.getgid()}')"
        )
        print(f"Result: {result.stdout.strip()}")

        # Test 2: Create directory in root filesystem
        result = session.run("""
import os
os.makedirs('/test_dir', exist_ok=True)
with open('/test_dir/test_file.txt', 'w') as f:
    f.write('Hello from root!')
print('Successfully created file in /test_dir/')
""")
        print(f"File creation result: {result.stdout.strip()}")

        # Test 3: Check if we can access system directories
        result = session.run(
            "import os; print('Can access /etc:', os.path.exists('/etc'))"
        )
        print(f"System access: {result.stdout.strip()}")


def test_custom_user():
    """Test running with a custom user."""
    print("\n=== Testing Custom User ===")

    # Note: This might fail in some containers that don't have user 1000
    try:
        with SandboxSession(
            lang="python",
            runtime_configs={"user": "1000:1000"},
            verbose=True,
            workdir="/tmp/sandbox",
        ) as session:
            result = session.run(
                "import os; print(f'UID: {os.getuid()}, GID: {os.getgid()}')"
            )
            print(f"Custom user result: {result.stdout.strip()}")

            # Test if we can still write to /tmp
            result = session.run("""
import os
import tempfile
with tempfile.NamedTemporaryFile(mode='w', delete=False, dir='/tmp') as f:
    f.write('Hello from user 1000!')
    temp_file = f.name
print(f'Successfully created file: {temp_file}')
""")
            print(f"Temp file creation: {result.stdout.strip()}")

    except Exception as e:
        print(
            f"Custom user test failed (this is expected if user 1000 doesn't exist): {e}"
        )


def test_permission_comparison():
    """Compare what we can do as root vs non-root."""
    print("\n=== Permission Comparison ===")

    print("As root:")
    with SandboxSession(lang="python") as session:
        result = session.run("""
import os
try:
    # Try to write to /etc (should work as root)
    with open('/etc/test_root_file', 'w') as f:
        f.write('test')
    print("✓ Can write to /etc")
    os.remove('/etc/test_root_file')
except Exception as e:
    print(f"✗ Cannot write to /etc: {e}")

try:
    # Try to create directory in /
    os.makedirs('/root_test_dir', exist_ok=True)
    print("✓ Can create directory in /")
    os.rmdir('/root_test_dir')
except Exception as e:
    print(f"✗ Cannot create directory in /: {e}")
""")
        print(result.stdout.strip())


def test_kubernetes_root_user():
    """Test Kubernetes root user behavior (requires k8s cluster)."""
    print("\n=== Testing Kubernetes Root User (requires cluster) ===")

    try:
        with SandboxSession(
            backend="kubernetes", lang="python", verbose=True
        ) as session:
            result = session.run(
                "import os; print(f'K8s UID: {os.getuid()}, GID: {os.getgid()}')"
            )
            print(f"Kubernetes root result: {result.stdout.strip()}")

            # Test creating directory in root filesystem
            result = session.run("""
import os
os.makedirs('/k8s_test_dir', exist_ok=True)
with open('/k8s_test_dir/test.txt', 'w') as f:
    f.write('Hello from K8s root!')
print('✓ Successfully created file in /k8s_test_dir/')
""")
            print(f"K8s file creation: {result.stdout.strip()}")

    except Exception as e:
        print(f"Kubernetes test skipped (cluster not available): {e}")


def test_kubernetes_custom_security():
    """Test Kubernetes with custom security context."""
    print("\n=== Testing Kubernetes Custom Security Context ===")

    try:
        # Custom pod manifest with non-root user
        pod_manifest = {
            "apiVersion": "v1",
            "kind": "Pod",
            "metadata": {
                "name": "test-non-root",
                "labels": {"app": "sandbox"},
            },
            "spec": {
                "containers": [
                    {
                        "name": "sandbox-container",
                        "image": "python:3.11-bullseye",
                        "tty": True,
                        "securityContext": {
                            "runAsUser": 1000,
                            "runAsGroup": 1000,
                        },
                    }
                ],
                "securityContext": {
                    "runAsUser": 1000,
                    "runAsGroup": 1000,
                },
            },
        }

        with SandboxSession(
            backend="kubernetes",
            lang="python",
            pod_manifest=pod_manifest,
            workdir="/tmp/sandbox",  # Use writable directory for non-root
            verbose=True,
        ) as session:
            result = session.run(
                "import os; print(f'Custom K8s UID: {os.getuid()}, GID: {os.getgid()}')"
            )
            print(f"Custom security context result: {result.stdout.strip()}")

    except Exception as e:
        print(f"Kubernetes custom security test skipped: {e}")


def test_podman_root_user():
    """Test Podman root user behavior (requires podman)."""
    print("\n=== Testing Podman Root User (requires podman) ===")
    from podman import PodmanClient

    client = PodmanClient(
        base_url="unix:///var/folders/lh/rjbzw60n1fv7xr9kffn7gr840000gn/T/podman/podman-machine-default-api.sock"
    )
    try:
        with SandboxSession(
            backend="podman", lang="python", verbose=True, client=client
        ) as session:
            result = session.run(
                "import os; print(f'Podman UID: {os.getuid()}, GID: {os.getgid()}')"
            )
            print(f"Podman root result: {result.stdout.strip()}")

            # Test creating directory in root filesystem
            result = session.run("""
import os
os.makedirs('/podman_test_dir', exist_ok=True)
with open('/podman_test_dir/test.txt', 'w') as f:
    f.write('Hello from Podman root!')
print('✓ Successfully created file in /podman_test_dir/')
""")
            print(f"Podman file creation: {result.stdout.strip()}")

    except Exception as e:
        print(f"Podman test skipped (podman not available): {e}")


def test_podman_custom_user():
    """Test Podman with custom user."""
    print("\n=== Testing Podman Custom User ===")
    from podman import PodmanClient

    client = PodmanClient(
        base_url="unix:///var/folders/lh/rjbzw60n1fv7xr9kffn7gr840000gn/T/podman/podman-machine-default-api.sock"
    )
    try:
        with SandboxSession(
            backend="podman",
            lang="python",
            runtime_configs={"user": "1000:1000"},
            workdir="/tmp/sandbox",
            verbose=True,
            client=client,
        ) as session:
            result = session.run(
                "import os; print(f'Podman Custom UID: {os.getuid()}, GID: {os.getgid()}')"
            )
            print(f"Podman custom user result: {result.stdout.strip()}")

            # Test if we can still write to /tmp
            result = session.run("""
import os
import tempfile
with tempfile.NamedTemporaryFile(mode='w', delete=False, dir='/tmp') as f:
    f.write('Hello from Podman user 1000!')
    temp_file = f.name
print(f'Successfully created file: {temp_file}')
""")
            print(f"Podman temp file creation: {result.stdout.strip()}")

    except Exception as e:
        print(f"Podman custom user test skipped: {e}")


if __name__ == "__main__":
    test_default_root_user()
    test_custom_user()
    test_permission_comparison()
    test_kubernetes_root_user()
    test_kubernetes_custom_security()
    test_podman_root_user()
    test_podman_custom_user()
    print("\n=== All tests completed ===")
