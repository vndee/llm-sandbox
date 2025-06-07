"""Demonstration of LLM Sandbox timeout functionality.

This example shows various timeout scenarios and how to handle them properly
in real-world applications where LLM-generated code needs to be executed safely.
"""

import logging
import time

from llm_sandbox import SandboxSession
from llm_sandbox.exceptions import SandboxTimeoutError

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def demo_basic_timeout() -> None:
    """Demonstrate basic timeout functionality."""
    logger.info("🕐 Demo 1: Basic Timeout Functionality")
    logger.info("=" * 50)

    # Test fast operation (should complete)
    logger.info("Testing fast operation...")
    with SandboxSession(lang="python", execution_timeout=5.0, verbose=True) as session:
        result = session.run("""
import time
print("Starting fast operation...")
time.sleep(1)
print("Fast operation completed!")
""")
        logger.info("✅ Fast operation result: %s", result.stdout.strip())

    # Test slow operation (should timeout)
    logger.info("\nTesting slow operation (will timeout)...")
    try:
        with SandboxSession(lang="python", execution_timeout=3.0, verbose=True) as session:
            session.run("""
import time
print("Starting slow operation...")
time.sleep(10)  # This will timeout
print("This won't print!")
""")
        logger.error("❌ Slow operation should have timed out!")
    except SandboxTimeoutError as e:
        logger.info("✅ Successfully caught timeout: %s", e)


def demo_timeout_override() -> None:
    """Demonstrate per-execution timeout override."""
    logger.info("\n🕐 Demo 2: Per-Execution Timeout Override")
    logger.info("=" * 50)

    with SandboxSession(lang="python", execution_timeout=5.0, verbose=True) as session:
        # Use default timeout (5 seconds)
        logger.info("Using default timeout (5s)...")
        result1 = session.run("""
import time
time.sleep(2)
print("Completed with default timeout!")
""")
        logger.info("✅ Default timeout result: %s", result1.stdout.strip())

        # Override with longer timeout
        logger.info("Overriding with longer timeout (10s)...")
        result2 = session.run(
            """
import time
time.sleep(7)
print("Completed with extended timeout!")
""",
            timeout=10.0,
        )
        logger.info("✅ Extended timeout result: %s", result2.stdout.strip())

        # Override with shorter timeout (should fail)
        logger.info("Overriding with shorter timeout (2s) - should timeout...")
        try:
            session.run(
                """
import time
time.sleep(5)
print("This won't print!")
""",
                timeout=2.0,
            )
            logger.error("❌ Should have timed out!")
        except SandboxTimeoutError as e:
            logger.info("✅ Successfully caught timeout override: %s", e)


def demo_infinite_loop_protection() -> None:
    """Demonstrate protection against infinite loops."""
    logger.info("\n🕐 Demo 3: Infinite Loop Protection")
    logger.info("=" * 50)

    logger.info("Testing infinite loop protection...")
    try:
        with SandboxSession(lang="python", execution_timeout=3.0, verbose=True) as session:
            session.run("""
print("Starting infinite loop...")
i = 0
while True:  # Infinite loop
    i += 1
    if i % 100000 == 0:
        print(f"Loop iteration: {i}")
""")
        logger.error("❌ Infinite loop should have been stopped!")
    except SandboxTimeoutError as e:
        logger.info("✅ Infinite loop protection working: %s", e)


def demo_resource_intensive_code() -> None:
    """Demonstrate timeout for resource-intensive operations."""
    logger.info("\n🕐 Demo 4: Resource-Intensive Code Timeout")
    logger.info("=" * 50)

    logger.info("Testing resource-intensive operation timeout...")
    try:
        with SandboxSession(lang="python", execution_timeout=3.0, verbose=True) as session:
            session.run("""
print("Starting CPU-intensive operation...")

# Simulate heavy computation
total = 0
for i in range(10**8):  # This will take a while
    total += i * i
    if i % 10**7 == 0:
        print(f"Progress: {i/10**8*100:.1f}%")

print(f"Final result: {total}")
""")
        logger.error("❌ Resource-intensive operation should have timed out!")
    except SandboxTimeoutError as e:
        logger.info("✅ Resource-intensive operation timeout working: %s", e)


def demo_session_timeout() -> None:
    """Demonstrate session-level timeout."""
    logger.info("\n🕐 Demo 5: Session Timeout")
    logger.info("=" * 50)

    logger.info("Testing session timeout (10s session lifetime)...")
    try:
        with SandboxSession(
            lang="python",
            execution_timeout=30.0,  # Long execution timeout
            session_timeout=10.0,  # But short session lifetime
            verbose=True,
        ) as session:
            # First execution should work
            result1 = session.run("print('First execution successful')")
            logger.info("✅ First execution: %s", result1.stdout.strip())

            # Wait to approach session timeout
            time.sleep(6)

            # Second execution should still work
            result2 = session.run("print('Second execution successful')")
            logger.info("✅ Second execution: %s", result2.stdout.strip())

            # Wait to exceed session timeout
            time.sleep(6)  # Total wait: 12 seconds > 10s session timeout

            # This should fail due to session timeout
            session.run("print('This should fail')")
            logger.error("❌ Should have hit session timeout!")

    except SandboxTimeoutError as e:
        logger.info("✅ Session timeout working: %s", e)
    except Exception as e:
        # Session timeout can manifest as NotOpenSessionError when the session is auto-closed
        from llm_sandbox.exceptions import NotOpenSessionError

        if isinstance(e, NotOpenSessionError):
            logger.info("✅ Session timeout working: Session was closed due to timeout")
        else:
            raise


def demo_timeout_error_handling() -> None:
    """Demonstrate proper timeout error handling with retries."""
    logger.info("\n🕐 Demo 6: Timeout Error Handling with Retries")
    logger.info("=" * 50)

    # Code with variable execution time
    variable_code = """
import time
import random

# Simulate variable processing time
sleep_time = random.choice([1, 3, 6])  # Short, medium, or long
print(f"Processing for {sleep_time} seconds...")
time.sleep(sleep_time)
print("Processing completed!")
"""

    max_retries = 3
    timeout = 4.0  # Medium timeout

    for attempt in range(max_retries):
        logger.info("Attempt %d/%d with %ss timeout...", attempt + 1, max_retries, timeout)

        try:
            with SandboxSession(lang="python", execution_timeout=timeout, verbose=True) as session:
                result = session.run(variable_code)
                logger.info("✅ Attempt %d succeeded: %s", attempt + 1, result.stdout.strip())
                return  # Exit on success

        except SandboxTimeoutError as e:
            logger.warning("⚠️ Attempt %d timed out: %s", attempt + 1, e)
            if attempt < max_retries - 1:
                logger.info("Retrying with a new session...")
            else:
                logger.exception("❌ All attempts failed due to timeout")

        except Exception:
            logger.exception("❌ Unexpected error")
            break


def demo_library_timeout() -> None:
    """Demonstrate timeout with library installation and usage."""
    logger.info("\n🕐 Demo 7: Library Installation and Usage Timeout")
    logger.info("=" * 50)

    logger.info("Testing library operations with timeout...")
    try:
        with SandboxSession(lang="python", execution_timeout=30.0, verbose=True) as session:
            result = session.run(
                """
import numpy as np
import time

print("Creating large array...")
arr = np.random.rand(1000, 1000)

print("Performing matrix operations...")
result = np.dot(arr, arr.T)

print("Simulating additional processing...")
time.sleep(2)

print(f"Matrix computation completed! Shape: {result.shape}")
""",
                libraries=["numpy"],
            )

            if "Matrix computation completed!" in result.stdout:
                logger.info("✅ Library operations completed successfully")
            else:
                logger.warning("⚠️ Unexpected output: %s", result.stdout)

    except SandboxTimeoutError:
        logger.exception("❌ Library operation timed out")
    except Exception:
        logger.exception("❌ Library operation failed")


def main() -> None:
    """Run all timeout demonstrations."""
    logger.info("🚀 LLM Sandbox Timeout Functionality Demo")
    logger.info("=" * 60)
    logger.info("This demo shows various timeout scenarios and how to handle them properly.")
    logger.info("=" * 60)

    try:
        demo_basic_timeout()
        demo_timeout_override()
        demo_infinite_loop_protection()
        demo_resource_intensive_code()
        demo_session_timeout()
        demo_timeout_error_handling()
        demo_library_timeout()

        logger.info("\n🎉 All timeout demos completed successfully!")
        logger.info("Key takeaways:")
        logger.info("1. Use appropriate timeouts based on expected execution time")
        logger.info("2. Override timeouts for specific operations when needed")
        logger.info("3. Implement proper error handling and retry logic")
        logger.info("4. Consider both execution and session timeouts")
        logger.info("5. Monitor resource usage patterns for timeout optimization")

    except Exception:
        logger.exception("❌ Demo failed with unexpected error")


if __name__ == "__main__":
    main()
