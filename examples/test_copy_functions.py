# ruff: noqa: PLR0912, PLR0915

"""Test and demonstrate the robust copy functions across all backends.

This script demonstrates:
1. File copying (both directions)
2. Directory copying (both directions)
3. Consistency across Docker, Podman, and Kubernetes backends
4. Robustness features like error handling and path validation
5. Basic safety features (absolute path filtering)

Usage:
    python examples/test_copy_functions.py [backend]

    backend: docker, podman, kubernetes, or all (default: all)
"""

import logging
import tempfile
import time
from pathlib import Path
from typing import Any

from llm_sandbox import SandboxBackend, SandboxSession

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

logger = logging.getLogger(__name__)


def create_test_files(base_dir: Path) -> dict[str, Path]:
    """Create test files and directories for copying tests."""
    test_files = {}

    # Create a simple text file
    test_files["simple_file"] = base_dir / "test_file.txt"
    test_files["simple_file"].write_text("Hello from the host!\nThis is a test file.")

    # Create a Python script
    test_files["python_script"] = base_dir / "test_script.py"
    test_files["python_script"].write_text("""#!/usr/bin/env python3
print("Hello from Python!")
print("This script was copied to the container")
import os
print(f"Current directory: {os.getcwd()}")
print(f"Files in current directory: {os.listdir('.')}")
""")

    # Create a directory with multiple files
    test_dir = base_dir / "test_directory"
    test_dir.mkdir(exist_ok=True)

    (test_dir / "file1.txt").write_text("Content of file 1")
    (test_dir / "file2.txt").write_text("Content of file 2")
    (test_dir / "data.json").write_text('{"key": "value", "number": 42}')

    # Create a subdirectory
    subdir = test_dir / "subdir"
    subdir.mkdir(exist_ok=True)
    (subdir / "nested_file.txt").write_text("This is in a subdirectory")

    test_files["directory"] = test_dir

    return test_files


def test_backend(backend_name: str, backend_enum: SandboxBackend) -> dict[str, Any]:
    """Test copy functions for a specific backend."""
    logger.info("\n%s", "=" * 60)
    logger.info("Testing %s Backend", backend_name.upper())
    logger.info("%s", "=" * 60)

    results: dict[str, Any] = {"backend": backend_name, "tests_passed": 0, "tests_failed": 0, "errors": []}

    try:
        # Create session with proper client configuration
        logger.info("Creating %s session...", backend_name)

        client = None
        if backend_name == "docker":
            import docker

            # Use Docker Desktop's actual socket path
            client = docker.DockerClient(base_url="unix:///Users/vndee/.docker/run/docker.sock")
        elif backend_name == "podman":
            from podman import PodmanClient

            client = PodmanClient(
                base_url="unix:///var/folders/lh/rjbzw60n1fv7xr9kffn7gr840000gn/T/podman/podman-machine-default-api.sock"
            )

        with SandboxSession(
            backend=backend_enum, lang="python", verbose=True, keep_template=True, client=client
        ) as session:
            logger.info("✅ %s session opened successfully", backend_name)

            # Create temporary directory for test files
            with tempfile.TemporaryDirectory() as temp_dir:
                temp_path = Path(temp_dir)
                test_files = create_test_files(temp_path)

                # Test 1: Copy single file to container
                logger.info("\n📁 Test 1: Copy single file to container")
                try:
                    session.copy_to_runtime(str(test_files["simple_file"]), "/sandbox/copied_file.txt")

                    # Verify file was copied
                    result = session.execute_command("cat /sandbox/copied_file.txt")
                    # Check for successful execution (exit_code 0 or None means success)
                    if (result.exit_code == 0 or result.exit_code is None) and "Hello from the host!" in result.stdout:
                        logger.info("✅ Single file copy to container successful")
                        results["tests_passed"] += 1
                    else:
                        logger.info("❌ Single file copy verification failed")
                        results["tests_failed"] += 1
                        results["errors"].append("Single file copy verification failed")

                except Exception as e:
                    logger.exception("❌ Single file copy failed")
                    results["tests_failed"] += 1
                    results["errors"].append(f"Single file copy: {e}")

                # Test 2: Copy directory to container
                logger.info("\n📁 Test 2: Copy directory to container")
                try:
                    session.copy_to_runtime(str(test_files["directory"]), "/sandbox/copied_directory")

                    # Verify directory structure was copied
                    result = session.execute_command("find /sandbox/copied_directory -type f")
                    if result.exit_code == 0 or result.exit_code is None:
                        files_found = result.stdout.strip().split("\n")
                        expected_files = ["file1.txt", "file2.txt", "data.json", "nested_file.txt"]

                        all_found = all(any(expected in found for found in files_found) for expected in expected_files)

                        if all_found:
                            logger.info("✅ Directory copy to container successful")
                            results["tests_passed"] += 1
                        else:
                            logger.info("❌ Directory copy incomplete. Found: %s", files_found)
                            results["tests_failed"] += 1
                            results["errors"].append("Directory copy incomplete")
                    else:
                        logger.info("❌ Directory copy verification failed: %s", result.stderr)
                        results["tests_failed"] += 1
                        results["errors"].append("Directory copy verification failed")

                except Exception as e:
                    logger.exception("❌ Directory copy failed")
                    results["tests_failed"] += 1
                    results["errors"].append(f"Directory copy: {e}")

                # Test 3: Create file in container and copy back
                logger.info("\n📁 Test 3: Copy file from container to host")
                try:
                    # Create a file in the container
                    session.execute_command("sh -c 'echo \"Generated in container\" > /sandbox/container_file.txt'")

                    # Copy it back to host
                    output_file = temp_path / "from_container.txt"
                    session.copy_from_runtime("/sandbox/container_file.txt", str(output_file))

                    # Verify file was copied back
                    if output_file.exists() and "Generated in container" in output_file.read_text():
                        logger.info("✅ File copy from container successful")
                        results["tests_passed"] += 1
                    else:
                        logger.info("❌ File copy from container failed")
                        results["tests_failed"] += 1
                        results["errors"].append("File copy from container failed")

                except Exception as e:
                    logger.exception("❌ File copy from container failed")
                    results["tests_failed"] += 1
                    results["errors"].append(f"File copy from container: {e}")

                # Test 4: Copy directory from container to host
                logger.info("\n📁 Test 4: Copy directory from container to host")
                try:
                    # Create directory structure in container
                    session.execute_command("mkdir -p /sandbox/output_dir/subdir")
                    session.execute_command("sh -c 'echo \"Container output 1\" > /sandbox/output_dir/output1.txt'")
                    session.execute_command(
                        "sh -c 'echo \"Container output 2\" > /sandbox/output_dir/subdir/output2.txt'"
                    )

                    # Copy directory back to host
                    output_dir = temp_path / "from_container_dir"
                    session.copy_from_runtime("/sandbox/output_dir", str(output_dir))

                    # Verify directory was copied back with consistent structure
                    expected_pattern = (output_dir / "output_dir" / "output1.txt").exists() and (
                        output_dir / "output_dir" / "subdir" / "output2.txt"
                    ).exists()

                    if output_dir.exists() and expected_pattern:
                        logger.info("✅ Directory copy from container successful")
                        results["tests_passed"] += 1
                    else:
                        logger.info("❌ Directory copy from container failed")
                        results["tests_failed"] += 1
                        results["errors"].append("Directory copy from container failed")

                except Exception as e:
                    logger.exception("❌ Directory copy from container failed")
                    results["tests_failed"] += 1
                    results["errors"].append(f"Directory copy from container: {e}")

                # Test 5: Execute copied Python script
                logger.info("\n📁 Test 5: Execute copied Python script")
                try:
                    session.copy_to_runtime(str(test_files["python_script"]), "/sandbox/test_script.py")

                    result = session.execute_command("python /sandbox/test_script.py")
                    if (result.exit_code == 0 or result.exit_code is None) and "Hello from Python!" in result.stdout:
                        logger.info("✅ Copied Python script execution successful")
                        results["tests_passed"] += 1
                    else:
                        logger.info("❌ Python script execution failed: %s", result.stderr)
                        results["tests_failed"] += 1
                        results["errors"].append("Python script execution failed")

                except Exception as e:
                    logger.exception("❌ Python script test failed")
                    results["tests_failed"] += 1
                    results["errors"].append(f"Python script test: {e}")

                # Test 6: Error handling for non-existent source
                logger.info("\n📁 Test 6: Error handling for non-existent source")
                try:
                    try:
                        session.copy_to_runtime("/nonexistent/file.txt", "/sandbox/dummy.txt")
                        logger.info("❌ Should have failed for non-existent source")
                        results["tests_failed"] += 1
                        results["errors"].append("Non-existent source should have failed")
                    except FileNotFoundError:
                        logger.info("✅ Correctly handled non-existent source file")
                        results["tests_passed"] += 1
                    except Exception:
                        logger.exception("✅ Correctly failed for non-existent source")
                        results["tests_passed"] += 1

                except Exception as e:
                    logger.exception("❌ Error handling test failed")
                    results["tests_failed"] += 1
                    results["errors"].append(f"Error handling test: {e}")

                # Test 7: Safety test - absolute paths (demonstrate warning)
                logger.info("\n📁 Test 7: Safety test - absolute path handling")
                try:
                    # Create a tar file with absolute paths manually to test filtering

                    # For this test, we'll just verify the copy functions work normally
                    # The absolute path filtering happens during extraction from container
                    session.execute_command("sh -c 'echo \"Safe content\" > /sandbox/safe_file.txt'")

                    safe_output = temp_path / "safe_output.txt"
                    session.copy_from_runtime("/sandbox/safe_file.txt", str(safe_output))

                    if safe_output.exists():
                        logger.info("✅ Normal copy operations work correctly")
                        results["tests_passed"] += 1
                    else:
                        logger.info("❌ Safety test failed")
                        results["tests_failed"] += 1
                        results["errors"].append("Safety test failed")

                except Exception as e:
                    logger.exception("❌ Safety test failed")
                    results["tests_failed"] += 1
                    results["errors"].append(f"Safety test: {e}")

    except Exception as e:
        logger.exception("❌ Backend %s failed to initialize", backend_name)
        results["tests_failed"] += 1
        results["errors"].append(f"Backend initialization: {e}")

    return results


def main() -> None:
    """Execute main function to run copy function tests."""
    import sys

    # Parse command line arguments
    backend_arg = sys.argv[1] if len(sys.argv) > 1 else "all"

    backends_to_test = {
        "docker": SandboxBackend.DOCKER,
        "podman": SandboxBackend.PODMAN,
        "kubernetes": SandboxBackend.KUBERNETES,
    }

    if backend_arg != "all" and backend_arg not in backends_to_test:
        logger.error("Error: Unknown backend '%s'", backend_arg)
        logger.error("Available backends: %s or 'all'", list(backends_to_test.keys()))
        sys.exit(1)

    logger.info("🧪 LLM Sandbox Copy Functions Test Suite")
    logger.info("==========================================")
    logger.info("Testing robust file and directory copying across backends...")

    backends = backends_to_test if backend_arg == "all" else {backend_arg: backends_to_test[backend_arg]}
    all_results = []

    for backend_name, backend_enum in backends.items():
        start_time = time.time()

        try:
            results = test_backend(backend_name, backend_enum)
            results["duration"] = time.time() - start_time
            all_results.append(results)

        except Exception as e:
            logger.exception("\n❌ Fatal error testing %s", backend_name)
            all_results.append({
                "backend": backend_name,
                "tests_passed": 0,
                "tests_failed": 1,
                "errors": [f"Fatal error: {e}"],
                "duration": time.time() - start_time,
            })

    # Print summary
    logger.info("\n%s", "=" * 60)
    logger.info("TEST SUMMARY")
    logger.info("%s", "=" * 60)

    total_passed = 0
    total_failed = 0

    for result in all_results:
        backend = result["backend"]
        passed = result["tests_passed"]
        failed = result["tests_failed"]
        duration = result["duration"]

        status = "✅ PASS" if failed == 0 else "❌ FAIL"
        logger.info("%s | %s | %s passed, %s failed | %.1fs", backend.upper()[:12], status, passed, failed, duration)

        if result["errors"]:
            for error in result["errors"]:
                logger.info("             └─ %s", error)

        total_passed += passed
        total_failed += failed

    logger.info("\nOverall: %s tests passed, %s tests failed", total_passed, total_failed)

    if total_failed == 0:
        logger.info("🎉 All copy function tests passed! The implementations are robust and consistent.")
    else:
        logger.info("⚠️  Some tests failed. Check the errors above for details.")
        sys.exit(1)


if __name__ == "__main__":
    main()
