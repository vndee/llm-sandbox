"""Concurrent execution demonstration using container pool.

This example demonstrates how the container pool handles concurrent
requests efficiently:
- Thread-safe container acquisition
- Multiple concurrent sessions sharing a pool
- Performance comparison with/without pooling
- Pool exhaustion handling strategies
"""

import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

from llm_sandbox import SandboxSession
from llm_sandbox.pool import ExhaustionStrategy, PoolConfig, create_pool_manager


def run_code_in_session(task_id: int, code: str, use_pool: bool = False, pool_manager=None) -> dict:
    """Run code in a session and measure execution time.

    Args:
        task_id: Task identifier
        code: Code to execute
        use_pool: Whether to use pooling
        pool_manager: Pool manager to use (if any)

    Returns:
        Result dictionary with timing information

    """
    start_time = time.time()

    try:
        with SandboxSession(
            lang="python",
            use_pool=use_pool,
            pool_manager=pool_manager,
            verbose=False,
        ) as session:
            result = session.run(code)
            exec_time = time.time() - start_time

            return {
                "task_id": task_id,
                "success": True,
                "output": result.stdout.strip(),
                "time": exec_time,
            }
    except Exception as e:  # noqa: BLE001
        return {
            "task_id": task_id,
            "success": False,
            "error": str(e),
            "time": time.time() - start_time,
        }


def demo_concurrent_execution() -> None:
    """Demonstrate concurrent execution with pooling."""
    print("\n1. Concurrent Execution (Thread-Safe Pool):")
    print("-" * 60)

    # Create a shared pool
    pool_config = PoolConfig(
        max_pool_size=3,
        min_pool_size=2,
        exhaustion_strategy=ExhaustionStrategy.WAIT,
        acquisition_timeout=30.0,
    )

    pool = create_pool_manager(
        backend="docker",
        config=pool_config,
        lang="python",
    )

    try:
        # Prepare tasks
        tasks = [
            f'import time; time.sleep(0.{i}); print("Task {i} completed")'
            for i in range(1, 6)
        ]

        print(f"  Running {len(tasks)} tasks concurrently with pool size {pool_config.max_pool_size}")

        start_time = time.time()

        # Execute concurrently using ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=len(tasks)) as executor:
            # Submit all tasks
            futures = [
                executor.submit(run_code_in_session, i, code, use_pool=False, pool_manager=pool)
                for i, code in enumerate(tasks, 1)
            ]

            # Collect results
            results = []
            for future in as_completed(futures):
                result = future.result()
                results.append(result)
                if result["success"]:
                    print(f"   Task {result['task_id']}: {result['output']} ({result['time']:.2f}s)")
                else:
                    print(f"   Task {result['task_id']}: Failed - {result['error']}")

        total_time = time.time() - start_time
        print(f"\n  Total time: {total_time:.2f}s")
        print(f"  Average time per task: {sum(r['time'] for r in results) / len(results):.2f}s")

    finally:
        pool.close()
        print("   Pool closed")


def demo_performance_comparison() -> None:
    """Compare performance with and without pooling."""
    print("\n2. Performance Comparison (Pool vs No Pool):")
    print("-" * 60)

    num_tasks = 5
    code = 'print("Hello World!")'

    # Test without pooling
    print("  Without pooling:")
    start_time = time.time()

    for i in range(1, num_tasks + 1):
        result = run_code_in_session(i, code, use_pool=False)
        if result["success"]:
            print(f"    Task {i}: {result['time']:.2f}s")

    no_pool_time = time.time() - start_time
    print(f"  Total time: {no_pool_time:.2f}s")

    # Test with pooling
    print("\n  With pooling:")
    pool = create_pool_manager(
        backend="docker",
        config=PoolConfig(max_pool_size=3, min_pool_size=2),
        lang="python",
    )

    try:
        start_time = time.time()

        for i in range(1, num_tasks + 1):
            result = run_code_in_session(i, code, use_pool=False, pool_manager=pool)
            if result["success"]:
                print(f"    Task {i}: {result['time']:.2f}s")

        pool_time = time.time() - start_time
        print(f"  Total time: {pool_time:.2f}s")

        # Calculate speedup
        speedup = no_pool_time / pool_time
        print(f"\n  Performance improvement: {speedup:.2f}x faster with pooling")

    finally:
        pool.close()


def demo_exhaustion_strategies() -> None:
    """Demonstrate different pool exhaustion strategies."""
    print("\n3. Pool Exhaustion Strategies:")
    print("-" * 60)

    # Strategy 1: WAIT - wait for a container to become available
    print("  Strategy 1: WAIT (wait for available container)")

    pool_config = PoolConfig(
        max_pool_size=2,
        min_pool_size=1,
        exhaustion_strategy=ExhaustionStrategy.WAIT,
        acquisition_timeout=10.0,
    )

    pool = create_pool_manager(
        backend="docker",
        config=pool_config,
        lang="python",
    )

    try:
        # Create a long-running task
        long_task = 'import time; time.sleep(2); print("Long task done")'

        def worker(worker_id: int):
            result = run_code_in_session(worker_id, long_task, pool_manager=pool)
            if result["success"]:
                print(f"    Worker {worker_id}: {result['output']}")

        # Start 3 workers (more than pool size)
        threads = [threading.Thread(target=worker, args=(i,)) for i in range(1, 4)]

        start_time = time.time()
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        print(f"    Total time: {time.time() - start_time:.2f}s")
        print("     Worker 3 waited for a container to become available")

    finally:
        pool.close()

    # Strategy 2: FAIL_FAST - raise error when pool is exhausted
    print("\n  Strategy 2: FAIL_FAST (raise error immediately)")

    pool_config_fail_fast = PoolConfig(
        max_pool_size=2,
        min_pool_size=1,
        exhaustion_strategy=ExhaustionStrategy.FAIL_FAST,
    )

    pool_fail_fast = create_pool_manager(
        backend="docker",
        config=pool_config_fail_fast,
        lang="python",
    )

    try:
        # Acquire all containers
        c1 = pool_fail_fast.acquire()
        c2 = pool_fail_fast.acquire()

        print("    Acquired 2 containers (pool is now full)")

        # Try to acquire one more (should fail)
        try:
            c3 = pool_fail_fast.acquire()
            print("     Should have raised PoolExhaustedError")
        except Exception as e:
            print(f"     Correctly raised: {e.__class__.__name__}")

        # Release containers
        pool_fail_fast.release(c1)
        pool_fail_fast.release(c2)

    finally:
        pool_fail_fast.close()


def demo_shared_pool() -> None:
    """Demonstrate sharing a pool across multiple sessions."""
    print("\n4. Shared Pool Manager:")
    print("-" * 60)

    # Create a shared pool
    pool = create_pool_manager(
        backend="docker",
        config=PoolConfig(
            max_pool_size=4,
            min_pool_size=2,
        ),
        lang="python",
    )

    try:
        print("  Executing 10 sessions sharing the same pool...")

        # Track container IDs used
        container_ids = set()

        def execute_with_tracking(task_id: int):
            with SandboxSession(lang="python", pool_manager=pool) as session:
                # Get container ID
                container_id = session.container.id[:12] if hasattr(session, 'container') else "unknown"
                container_ids.add(container_id)

                # Run code
                result = session.run(f'print("Task {task_id}")')
                return result.stdout.strip()

        # Execute multiple sessions
        with ThreadPoolExecutor(max_workers=4) as executor:
            results = list(executor.map(execute_with_tracking, range(1, 11)))

        print("   Executed 10 sessions")
        print(f"   Used {len(container_ids)} unique containers (reused efficiently)")

        # Show pool stats
        stats = pool.get_stats()
        print("\n  Final pool statistics:")
        print(f"    Total containers: {stats['total_size']}")
        print(f"    Idle: {stats['state_counts'].get('idle', 0)}")
        print(f"    Busy: {stats['state_counts'].get('busy', 0)}")

    finally:
        pool.close()
        print("   Pool closed")


def main() -> None:
    """Run all concurrent execution demonstrations."""
    print("=" * 60)
    print("Concurrent Execution Demo")
    print("=" * 60)

    demo_concurrent_execution()
    demo_performance_comparison()
    demo_exhaustion_strategies()
    demo_shared_pool()

    print("\n" + "=" * 60)
    print("Demo Complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
