"""Examples demonstrating timeout functionality in LLM Sandbox.
These examples show how to use various timeout configurations.
"""

import logging
import time

from llm_sandbox import SandboxBackend, SandboxSession
from llm_sandbox.exceptions import SandboxTimeoutError

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def example_execution_timeout() -> None:
    """Execute timeout example."""
    logger.info("=== Execution Timeout Example ===")

    # Create session with 5-second execution timeout
    with SandboxSession(lang="python", execution_timeout=5.0, verbose=True) as session:
        # This should complete within timeout
        try:
            result = session.run("""
import time
print("Starting fast operation...")
time.sleep(2)
print("Fast operation completed!")
""")
            logger.info("✅ Fast operation succeeded")
            logger.info(f"Output: {result.stdout}")
        except SandboxTimeoutError as e:
            logger.error(f"❌ Unexpected timeout: {e}")

        # This should timeout
        try:
            result = session.run("""
import time
print("Starting slow operation...")
time.sleep(10)  # This will timeout after 5 seconds
print("Slow operation completed!")
""")
            logger.error("❌ This should have timed out!")
        except SandboxTimeoutError as e:
            logger.info(f"✅ Expected timeout occurred: {e}")


def example_per_execution_timeout() -> None:
    """Example: Override timeout per execution."""
    logger.info("\n=== Per-Execution Timeout Example ===")

    with SandboxSession(
        lang="python",
        execution_timeout=10.0,  # Default 10 seconds
        verbose=True,
    ) as session:
        # Use default timeout (10 seconds) - should complete
        try:
            result = session.run("""
import time
print("Using default timeout...")
time.sleep(3)
print("Completed with default timeout!")
""")
            logger.info("✅ Default timeout execution succeeded")
        except SandboxTimeoutError:
            logger.error("❌ Unexpected timeout with default")

        # Override with shorter timeout (2 seconds) - should timeout
        try:
            result = session.run(
                """
import time
print("Using short timeout...")
time.sleep(5)
print("This won't print!")
""",
                timeout=2.0,
            )  # Override to 2 seconds
            logger.error("❌ This should have timed out!")
        except SandboxTimeoutError as e:
            logger.info(f"✅ Expected timeout with override: {e}")


def example_session_timeout() -> None:
    """Example: Session-level timeout."""
    logger.info("\n=== Session Timeout Example ===")

    # Session will automatically close after 8 seconds
    try:
        with SandboxSession(
            lang="python",
            session_timeout=8.0,  # Session expires after 8 seconds
            execution_timeout=3.0,  # Individual executions timeout after 3 seconds
            verbose=True,
        ) as session:
            # First execution (should work)
            result = session.run("print('First execution')")
            logger.info("✅ First execution completed")

            # Wait a bit
            time.sleep(2)

            # Second execution (should work)
            result = session.run("print('Second execution')")
            logger.info("✅ Second execution completed")

            # Wait longer to trigger session timeout
            logger.info("Waiting for session timeout...")
            time.sleep(7)  # This should trigger session timeout

            # This execution should fail due to session timeout
            result = session.run("print('This should not work')")
            logger.error("❌ This should have failed due to session timeout!")

    except SandboxTimeoutError as e:
        logger.info(f"✅ Expected session timeout: {e}")


def example_infinite_loop_protection() -> None:
    """Example: Protecting against infinite loops."""
    logger.info("\n=== Infinite Loop Protection Example ===")

    with SandboxSession(
        lang="python",
        execution_timeout=3.0,  # 3 second timeout
        verbose=True,
    ) as session:
        try:
            result = session.run("""
print("Starting infinite loop...")
i = 0
while True:  # Infinite loop
    i += 1
    if i % 100000 == 0:
        print(f"Loop iteration: {i}")
print("This will never print!")
""")
            logger.error("❌ Infinite loop should have been stopped!")
        except SandboxTimeoutError as e:
            logger.info(f"✅ Infinite loop stopped by timeout: {e}")


def example_resource_intensive_code() -> None:
    """Example: Timeout for resource-intensive operations."""
    logger.info("\n=== Resource Intensive Code Example ===")

    with SandboxSession(lang="python", execution_timeout=5.0, verbose=True) as session:
        # CPU-intensive operation that should timeout
        try:
            result = session.run("""
import time
print("Starting CPU-intensive operation...")

# Simulate heavy computation
total = 0
for i in range(10**8):  # This will take a while
    total += i * i
    if i % 10**7 == 0:
        print(f"Progress: {i/10**8*100:.1f}%")

print(f"Final result: {total}")
""")
            logger.info("✅ CPU-intensive operation completed")
            logger.info(f"Output: {result.stdout}")
        except SandboxTimeoutError as e:
            logger.info(f"✅ CPU-intensive operation timed out as expected: {e}")


def example_timeout_with_libraries() -> None:
    """Example: Timeout with library installation."""
    logger.info("\n=== Timeout with Libraries Example ===")

    with SandboxSession(
        lang="python",
        execution_timeout=30.0,  # Longer timeout for library installation
        verbose=True,
    ) as session:
        try:
            result = session.run(
                """
import numpy as np
import time

print("Creating large array...")
arr = np.random.rand(1000, 1000)

print("Performing matrix operations...")
result = np.dot(arr, arr.T)

print("Simulating some processing time...")
time.sleep(2)

print(f"Matrix shape: {result.shape}")
print(f"Matrix sum: {np.sum(result):.2f}")
""",
                libraries=["numpy"],
            )

            logger.info("✅ Library operation completed")
            logger.info(f"Output: {result.stdout}")
        except SandboxTimeoutError as e:
            logger.error(f"❌ Library operation timed out: {e}")


def example_kubernetes_timeout() -> None:
    """Example: Timeout with Kubernetes backend."""
    logger.info("\n=== Kubernetes Timeout Example ===")

    try:
        with SandboxSession(
            lang="python", backend=SandboxBackend.KUBERNETES, execution_timeout=10.0, session_timeout=60.0, verbose=True
        ) as session:
            result = session.run("""
import time
print("Running in Kubernetes...")
time.sleep(3)
print("Kubernetes execution completed!")
""")
            logger.info("✅ Kubernetes execution completed")

    except ImportError:
        logger.warning("⚠️ Kubernetes client not available, skipping example")
    except Exception as e:
        logger.warning(f"⚠️ Kubernetes not configured: {e}")


def example_timeout_error_handling() -> None:
    """Proper timeout error handling."""
    logger.info("\n=== Timeout Error Handling Example ===")

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
        logger.info(f"Attempt {attempt + 1}/{max_retries}")
        try:
            # Create a new session for each attempt
            with SandboxSession(lang="python", execution_timeout=2.0, verbose=True) as session:
                result = session.run(variable_code)
                logger.info(f"✅ Attempt {attempt + 1} succeeded.")
                logger.info(f"Output: {result.stdout}")
                return  # Exit if successful
        except SandboxTimeoutError as e:
            logger.warning(f"Attempt {attempt + 1} timed out: {e}")
            if attempt < max_retries - 1:
                logger.info("Retrying with a new session...")
            else:
                logger.error("All attempts failed due to timeout.")
                # Optionally, re-raise the last timeout error or a custom error
                # raise
        except Exception as e:
            logger.error(f"❌ Example failed with an unexpected error: {e}")
            # Depending on the desired behavior, you might want to break or re-raise
            break


def main() -> None:
    """Run all timeout examples."""
    logger.info("LLM Sandbox Timeout Feature Examples")
    logger.info("=" * 50)

    try:
        example_execution_timeout()
        example_per_execution_timeout()
        example_session_timeout()
        example_infinite_loop_protection()
        example_resource_intensive_code()
        example_timeout_with_libraries()
        example_kubernetes_timeout()
        example_timeout_error_handling()

        logger.info("\n🎉 All timeout examples completed!")

    except Exception as e:
        logger.error(f"❌ Example failed: {e}")
        raise


if __name__ == "__main__":
    main()
