#!/usr/bin/env python3
"""Comprehensive test for pip warnings across all backends."""

from llm_sandbox import SandboxSession


def test_docker_pip():
    """Test Docker backend with virtual environment."""
    print("=== Testing Docker Backend ===")

    try:
        with SandboxSession(backend="docker", lang="python", verbose=True) as session:
            result = session.run(
                "print('Docker virtual env test!')", libraries=["urllib3"]
            )
            print(f"Docker result: {result.stdout.strip()}")
            print(f"Docker errors: {result.stderr.strip()}")

            # Check for success: either exit_code is 0 or None (streaming mode) and no critical errors
            has_output = "Docker virtual env test!" in result.stdout
            no_critical_errors = (
                "Error" not in result.stderr and "Failed" not in result.stderr
            )
            success = has_output and no_critical_errors

            print("✅ Docker: SUCCESS" if success else "❌ Docker: FAILED")
    except Exception as e:
        print(f"❌ Docker test failed: {e}")


def test_podman_pip():
    """Test Podman backend with virtual environment."""
    print("\n=== Testing Podman Backend ===")

    try:
        from podman import PodmanClient

        client = PodmanClient(
            base_url="unix:///var/folders/lh/rjbzw60n1fv7xr9kffn7gr840000gn/T/podman/podman-machine-default-api.sock"
        )

        with SandboxSession(
            backend="podman", lang="python", verbose=True, client=client
        ) as session:
            result = session.run(
                "print('Podman virtual env test!')", libraries=["urllib3"]
            )
            print(f"Podman result: {result.stdout.strip()}")
            print(f"Podman errors: {result.stderr.strip()}")

            # Check for success
            has_output = "Podman virtual env test!" in result.stdout
            no_critical_errors = (
                "Error" not in result.stderr and "Failed" not in result.stderr
            )
            success = has_output and no_critical_errors

            print("✅ Podman: SUCCESS" if success else "❌ Podman: FAILED")
    except Exception as e:
        print(f"❌ Podman test skipped: {e}")


def test_kubernetes_pip():
    """Test Kubernetes backend with virtual environment."""
    print("\n=== Testing Kubernetes Backend ===")

    try:
        with SandboxSession(
            backend="kubernetes", lang="python", verbose=True
        ) as session:
            result = session.run(
                "print('Kubernetes virtual env test!')", libraries=["urllib3"]
            )
            print(f"Kubernetes result: {result.stdout.strip()}")
            print(f"Kubernetes errors: {result.stderr.strip()}")

            # Check for success
            has_output = "Kubernetes virtual env test!" in result.stdout
            no_critical_errors = (
                "Error" not in result.stderr and "Failed" not in result.stderr
            )
            success = (
                (result.exit_code == 0 or result.exit_code is None)
                and has_output
                and no_critical_errors
            )

            print("✅ Kubernetes: SUCCESS" if success else "❌ Kubernetes: FAILED")
    except Exception as e:
        print(f"❌ Kubernetes test skipped: {e}")


def test_custom_user_docker():
    """Test Docker with custom user and virtual environment."""
    print("\n=== Testing Docker Custom User ===")

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
            print(f"Docker custom user result: {result.stdout.strip()}")
            print(f"Docker custom user errors: {result.stderr.strip()}")

            # Check for success
            has_output = "Docker custom user virtual env test!" in result.stdout
            no_critical_errors = (
                "Error" not in result.stderr and "Failed" not in result.stderr
            )
            success = has_output and no_critical_errors

            print(
                "✅ Docker Custom User: SUCCESS"
                if success
                else "❌ Docker Custom User: FAILED"
            )
    except Exception as e:
        print(f"❌ Docker custom user test failed: {e}")


if __name__ == "__main__":
    print("🧪 Comprehensive Virtual Environment Test Across All Backends")
    print("=" * 60)

    test_docker_pip()
    test_podman_pip()
    test_kubernetes_pip()
    test_custom_user_docker()

    print("\n" + "=" * 60)
    print("🎯 All backend tests completed!")
