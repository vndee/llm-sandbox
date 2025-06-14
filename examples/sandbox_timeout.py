# ruff: noqa: PLR0912, PLR0915, BLE001
import logging
import sys
import time
from typing import Any

from llm_sandbox import SandboxBackend, SandboxSession
from llm_sandbox.core.session_base import BaseSession
from llm_sandbox.exceptions import SandboxTimeoutError

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def create_test_session(backend_enum: SandboxBackend, client: Any = None) -> BaseSession:
    """Create a new test session for the given backend."""
    return SandboxSession(
        backend=backend_enum,
        lang="python",
        execution_timeout=10.0,  # Default execution timeout
        session_timeout=120.0,  # Session timeout
        verbose=True,
        keep_template=False,
        client=client,
    )


def test_execution_timeout(backend_enum: SandboxBackend, client: Any = None) -> tuple[bool, str]:
    """Test basic execution timeout functionality."""
    try:
        # Test 1: Fast operation should complete
        with create_test_session(backend_enum, client) as session:
            result = session.run(
                """
import time
print("Starting fast operation...")
time.sleep(1)
print("Fast operation completed!")
""",
                timeout=5.0,
            )

            if "Fast operation completed!" not in result.stdout:
                return False, "Fast operation output not found"

        # Test 2: Slow operation should timeout
        try:
            with create_test_session(backend_enum, client) as session:
                session.run(
                    """
import time
print("Starting slow operation...")
time.sleep(10)  # This will timeout after 3 seconds
print("Slow operation completed!")
""",
                    timeout=3.0,
                )
                return False, "Slow operation should have timed out"
        except SandboxTimeoutError:
            return True, "Execution timeout working correctly"

    except SandboxTimeoutError:
        return False, "Unexpected timeout on fast operation"
    except Exception as e:
        return False, f"Execution timeout test failed: {e}"


def test_per_execution_timeout_override(backend_enum: SandboxBackend, client: Any = None) -> tuple[bool, str]:
    """Test per-execution timeout override."""
    try:
        # Test 1: Use longer timeout - should complete
        with create_test_session(backend_enum, client) as session:
            # For Kubernetes, use a shorter sleep time due to slower execution
            sleep_time = 1 if backend_enum == SandboxBackend.KUBERNETES else 2
            result = session.run(
                f"""
import time
print("Using long timeout...")
time.sleep({sleep_time})
print("Completed with long timeout!")
""",
                timeout=10.0,
            )  # Increase timeout for Kubernetes

            if "Completed with long timeout!" not in result.stdout:
                return False, "Long timeout execution failed"

        # Test 2: Override with shorter timeout - should timeout
        try:
            with create_test_session(backend_enum, client) as session:
                # For Kubernetes, use a longer sleep time to ensure timeout
                sleep_time = 6 if backend_enum == SandboxBackend.KUBERNETES else 4
                session.run(
                    f"""
import time
print("Using short timeout...")
time.sleep({sleep_time})
print("This won't print!")
""",
                    timeout=2.0,
                )
                return False, "Short timeout should have failed"
        except SandboxTimeoutError:
            return True, "Per-execution timeout override working correctly"

    except Exception as e:
        return False, f"Per-execution timeout test failed: {e}"


def test_infinite_loop_protection(backend_enum: SandboxBackend, client: Any = None) -> tuple[bool, str]:
    """Test protection against infinite loops."""
    try:
        with create_test_session(backend_enum, client) as session:
            session.run(
                """
print("Starting infinite loop...")
i = 0
while True:  # Infinite loop
    i += 1
    if i % 100000 == 0:
        print(f"Loop iteration: {i}")
print("This will never print!")
""",
                timeout=3.0,
            )
            return False, "Infinite loop should have been stopped"
    except SandboxTimeoutError:
        return True, "Infinite loop protection working correctly"
    except Exception as e:
        return False, f"Infinite loop test failed: {e}"


def test_resource_intensive_code(backend_enum: SandboxBackend, client: Any = None) -> tuple[bool, str]:
    """Test timeout for resource-intensive operations."""
    try:
        with create_test_session(backend_enum, client) as session:
            session.run(
                """
import time
print("Starting CPU-intensive operation...")

# Simulate heavy computation
total = 0
for i in range(10**8):  # This will take a while
    total += i * i
    if i % 10**7 == 0:
        print(f"Progress: {i/10**8*100:.1f}%")

print(f"Final result: {total}")
""",
                timeout=3.0,
            )
            return False, "Resource-intensive operation should have timed out"
    except SandboxTimeoutError:
        return True, "Resource-intensive operation timeout working correctly"
    except Exception as e:
        return False, f"Resource-intensive test failed: {e}"


def test_timeout_with_libraries(backend_enum: SandboxBackend, client: Any = None) -> tuple[bool, str]:
    """Test timeout with library installation and usage."""
    try:
        with create_test_session(backend_enum, client) as session:
            result = session.run(
                """
import numpy as np
import time

print("Creating array...")
arr = np.random.rand(100, 100)

print("Performing operations...")
result = np.dot(arr, arr.T)

print("Simulating processing...")
time.sleep(1)

print(f"Matrix shape: {result.shape}")
print("Library operation completed!")
""",
                libraries=["numpy"],
                timeout=30.0,
            )

            if "Library operation completed!" in result.stdout:
                return True, "Library timeout test working correctly"
            return False, "Library operation did not complete as expected"

    except SandboxTimeoutError:
        return False, "Library operation timed out unexpectedly"
    except Exception as e:
        return False, f"Library timeout test failed: {e}"


def test_backend_timeouts(backend_name: str, backend_enum: SandboxBackend) -> dict[str, Any]:
    """Test timeout functionality for a specific backend."""
    logger.info("\n%s", "=" * 60)
    logger.info("Testing %s Backend Timeouts", backend_name.upper())
    logger.info("%s", "=" * 60)

    results: dict[str, Any] = {"backend": backend_name, "tests_passed": 0, "tests_failed": 0, "errors": []}

    try:
        # Get client configuration
        client = None
        if backend_name == "docker":
            try:
                import docker

                client = docker.DockerClient(base_url="unix:///Users/vndee/.docker/run/docker.sock")
            except ImportError:
                logger.warning("Docker client not available")
                results["tests_failed"] += 1
                results["errors"].append("Docker client not available")
                return results
        elif backend_name == "podman":
            try:
                from podman import PodmanClient

                client = PodmanClient(
                    base_url="unix:///var/folders/lh/rjbzw60n1fv7xr9kffn7gr840000gn/T/podman/podman-machine-default-api.sock"
                )
            except ImportError:
                logger.warning("Podman client not available")
                results["tests_failed"] += 1
                results["errors"].append("Podman client not available")
                return results

        logger.info("Creating %s session with timeout settings...", backend_name)

        # Test 1: Basic execution timeout
        logger.info("\n🕐 Test 1: Basic execution timeout")
        try:
            success, message = test_execution_timeout(backend_enum, client)
            if success:
                logger.info("✅ %s", message)
                results["tests_passed"] += 1
            else:
                logger.info("❌ %s", message)
                results["tests_failed"] += 1
                results["errors"].append(f"Execution timeout: {message}")
        except Exception as e:
            logger.exception("❌ Execution timeout test failed")
            results["tests_failed"] += 1
            results["errors"].append(f"Execution timeout: {e}")

        # Test 2: Per-execution timeout override
        logger.info("\n🕐 Test 2: Per-execution timeout override")
        try:
            success, message = test_per_execution_timeout_override(backend_enum, client)
            if success:
                logger.info("✅ %s", message)
                results["tests_passed"] += 1
            else:
                logger.info("❌ %s", message)
                results["tests_failed"] += 1
                results["errors"].append(f"Timeout override: {message}")
        except Exception as e:
            logger.exception("❌ Per-execution timeout test failed")
            results["tests_failed"] += 1
            results["errors"].append(f"Timeout override: {e}")

        # Test 3: Infinite loop protection
        logger.info("\n🕐 Test 3: Infinite loop protection")
        try:
            success, message = test_infinite_loop_protection(backend_enum, client)
            if success:
                logger.info("✅ %s", message)
                results["tests_passed"] += 1
            else:
                logger.info("❌ %s", message)
                results["tests_failed"] += 1
                results["errors"].append(f"Infinite loop protection: {message}")
        except Exception as e:
            logger.exception("❌ Infinite loop protection test failed")
            results["tests_failed"] += 1
            results["errors"].append(f"Infinite loop protection: {e}")

        # Test 4: Resource-intensive code timeout
        logger.info("\n🕐 Test 4: Resource-intensive code timeout")
        try:
            success, message = test_resource_intensive_code(backend_enum, client)
            if success:
                logger.info("✅ %s", message)
                results["tests_passed"] += 1
            else:
                logger.info("❌ %s", message)
                results["tests_failed"] += 1
                results["errors"].append(f"Resource-intensive timeout: {message}")
        except Exception as e:
            logger.exception("❌ Resource-intensive timeout test failed")
            results["tests_failed"] += 1
            results["errors"].append(f"Resource-intensive timeout: {e}")

        # Test 5: Timeout with libraries
        logger.info("\n🕐 Test 5: Timeout with libraries")
        try:
            success, message = test_timeout_with_libraries(backend_enum, client)
            if success:
                logger.info("✅ %s", message)
                results["tests_passed"] += 1
            else:
                logger.info("❌ %s", message)
                results["tests_failed"] += 1
                results["errors"].append(f"Library timeout: {message}")
        except Exception as e:
            logger.exception("❌ Library timeout test failed")
            results["tests_failed"] += 1
            results["errors"].append(f"Library timeout: {e}")

    except Exception as e:
        logger.exception("❌ Backend %s failed to initialize", backend_name)
        results["tests_failed"] += 1
        results["errors"].append(f"Backend initialization: {e}")

    return results


def example_timeout_error_handling() -> None:
    """Demonstrate timeout error handling with retries."""
    logger.info("\n%s", "=" * 60)
    logger.info("Timeout Error Handling Example")
    logger.info("%s", "=" * 60)

    import docker

    client = docker.DockerClient(base_url="unix:///Users/vndee/.docker/run/docker.sock")

    # This code has variable execution time
    variable_code = """
import time
import random

sleep_time = random.choice([1, 3, 5]) # Short, medium, or long execution
print(f"Sleeping for {sleep_time} seconds...")
time.sleep(sleep_time)
print("Done sleeping.")
"""

    max_retries = 3
    for attempt in range(max_retries):
        logger.info("Attempt %s/%s", attempt + 1, max_retries)
        try:
            # Create a new session for each attempt
            with SandboxSession(lang="python", execution_timeout=2.0, verbose=True, client=client) as session:
                result = session.run(variable_code)
                logger.info("✅ Attempt %s succeeded.", attempt + 1)
                logger.info("Output: %s", result.stdout)
                return  # Exit if successful
        except SandboxTimeoutError as e:
            logger.warning("Attempt %s timed out: %s", attempt + 1, e)
            if attempt < max_retries - 1:
                logger.info("Retrying with a new session...")
            else:
                logger.exception("All attempts failed due to timeout.")
        except Exception:
            logger.exception("❌ Example failed with an unexpected error")
            break


def main() -> None:
    """Execute main function to run timeout tests across all backends."""
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

    logger.info("🕐 LLM Sandbox Timeout Test Suite")
    logger.info("=================================")
    logger.info("Testing timeout functionality across backends...")

    backends = backends_to_test if backend_arg == "all" else {backend_arg: backends_to_test[backend_arg]}
    all_results = []

    for backend_name, backend_enum in backends.items():
        start_time = time.time()

        try:
            results = test_backend_timeouts(backend_name, backend_enum)
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

    # Show timeout error handling example
    try:
        example_timeout_error_handling()
    except Exception:
        logger.exception("❌ Error handling example failed")

    # Print summary
    logger.info("\n%s", "=" * 60)
    logger.info("TIMEOUT TEST SUMMARY")
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
        logger.info("🎉 All timeout tests passed! Timeout functionality is working correctly across all backends.")
    else:
        logger.info("⚠️  Some tests failed. Check the errors above for details.")
        sys.exit(1)


if __name__ == "__main__":
    main()
