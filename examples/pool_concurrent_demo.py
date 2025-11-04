# ruff: noqa: T201, F841, S110

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

import docker

from llm_sandbox import SandboxBackend, SandboxSession, SupportedLanguage
from llm_sandbox.pool import ContainerPoolManager, ExhaustionStrategy, PoolConfig, create_pool_manager

client = docker.DockerClient(base_url="unix:///Users/vndee/.docker/run/docker.sock")


def run_code_in_session(task_id: int, code: str, pool_manager: ContainerPoolManager | None = None) -> dict:
    """Run code in a session and measure execution time.

    Args:
        task_id: Task identifier
        code: Code to execute
        pool_manager: Pool manager to use (if any)

    Returns:
        Result dictionary with timing information

    """
    start_time = time.time()

    if pool_manager is None:
        with SandboxSession(
            lang="python",
            verbose=False,
            client=client,
        ) as session:
            result = session.run(code)
    else:
        with SandboxSession(
            lang="python",
            pool=pool_manager,
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
        client=client,
        backend=SandboxBackend.DOCKER,
        config=pool_config,
        lang=SupportedLanguage.PYTHON,
    )

    try:
        # Prepare tasks
        tasks = [f'import time; time.sleep(0.{i}); print("Task {i} completed")' for i in range(1, 6)]

        print(f"  Running {len(tasks)} tasks concurrently with pool size {pool_config.max_pool_size}")

        start_time = time.time()

        # Execute concurrently using ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=len(tasks)) as executor:
            # Submit all tasks
            futures = [executor.submit(run_code_in_session, i, code, pool) for i, code in enumerate(tasks, 1)]

            # Collect results
            results = []
            for future in as_completed(futures):
                result = future.result()
                results.append(result)
                if result["success"]:
                    print(f"  Task {result['task_id']}: {result['output']} ({result['time']:.2f}s)")
                else:
                    print(f"  Task {result['task_id']}: Failed - {result['error']}")

        total_time = time.time() - start_time
        print(f"\n  Total time: {total_time:.2f}s")
        print(f"  Average time per task: {sum(r['time'] for r in results) / len(results):.2f}s")

    finally:
        pool.close()
        print("  Pool closed and all containers cleaned up")


def demo_performance_comparison() -> None:
    """Compare performance with and without pooling."""
    print("\n2. Performance Comparison (Pool vs No Pool):")
    print("-" * 60)

    num_tasks = 5
    code = 'print("Hello World!")'

    # Task without pooling - measure wall clock time
    print("  Without pooling (creates new container each time):")
    start_time = time.time()
    no_pool_results = []
    for i in range(1, num_tasks + 1):
        result = run_code_in_session(i, code)
        no_pool_results.append(result)
        if result["success"]:
            print(f"    Task {i}: {result['time']:.2f}s")
        else:
            print(f"    Task {i}: FAILED - {result.get('error', 'Unknown error')}")

    no_pool_time = time.time() - start_time
    successful_tasks = [r for r in no_pool_results if r["success"]]
    print(f"  Total wall clock time: {no_pool_time:.2f}s")
    print(f"  Successful tasks: {len(successful_tasks)}/{num_tasks}")
    if successful_tasks:
        print(f"  Average time per task: {no_pool_time / len(successful_tasks):.2f}s")

    # Test with pooling
    print("\n  With pooling (reuses containers):")
    pool_manager = create_pool_manager(
        client=client,
        backend=SandboxBackend.DOCKER,
        config=PoolConfig(max_pool_size=3, min_pool_size=2, enable_prewarming=True),
        lang=SupportedLanguage.PYTHON,
    )

    pool_results = []
    start_time = time.time()
    for i in range(1, num_tasks + 1):
        result = run_code_in_session(i, code, pool_manager)
        pool_results.append(result)
        if result["success"]:
            print(f"    Task {i}: {result['time']:.2f}s")
        else:
            print(f"    Task {i}: FAILED - {result.get('error', 'Unknown error')}")

    pool_time = time.time() - start_time
    successful_pool_tasks = [r for r in pool_results if r["success"]]
    print(f"  Total wall clock time: {pool_time:.2f}s")
    print(f"  Successful tasks: {len(successful_pool_tasks)}/{num_tasks}")
    if successful_pool_tasks:
        print(f"  Average time per task: {pool_time / len(successful_pool_tasks):.2f}s")

    # Calculate speedup
    if no_pool_time > 0 and pool_time > 0 and successful_tasks and successful_pool_tasks:
        speedup = no_pool_time / pool_time
        print(f"\n  Speedup with pooling: {speedup:.2f}x")
        print(f"  Time saved: {no_pool_time - pool_time:.2f}s")

    pool_manager.close()


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

    pool_manager = create_pool_manager(
        client=client,
        backend=SandboxBackend.DOCKER,
        config=pool_config,
        lang=SupportedLanguage.PYTHON,
    )

    try:
        # Create a long-running task
        long_task = 'import time; time.sleep(2); print("Long task done")'

        def worker(worker_id: int) -> None:
            result = run_code_in_session(worker_id, long_task, pool_manager)
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
        print("     Worker 3 waited for a container to become available")

    finally:
        pool_manager.close()

    # Strategy 2: FAIL_FAST - raise error when pool is exhausted
    print("\n  Strategy 2: FAIL_FAST (raise error immediately)")

    pool_config_fail_fast = PoolConfig(
        max_pool_size=2,
        min_pool_size=2,  # Pre-warm both containers
        exhaustion_strategy=ExhaustionStrategy.FAIL_FAST,
        enable_prewarming=True,
    )

    pool_manager_fail_fast = create_pool_manager(
        client=client,
        backend=SandboxBackend.DOCKER,
        config=pool_config_fail_fast,
        lang=SupportedLanguage.PYTHON,
    )

    try:
        # First, trigger pre-warming by acquiring one container and releasing it
        # This ensures containers are created before the actual test
        print("    Initializing pool...")
        temp = pool_manager_fail_fast.acquire()
        pool_manager_fail_fast.release(temp)

        # Now acquire all containers
        c1 = pool_manager_fail_fast.acquire()
        c2 = pool_manager_fail_fast.acquire()

        print("    Acquired 2 containers (pool is now full)")

        # Try to acquire one more (should fail)
        try:
            c3 = pool_manager_fail_fast.acquire()
            print("    ERROR: Should have raised PoolExhaustedError")
        except Exception as e:  # noqa: BLE001
            print(f"    ✓ Correctly raised: {type(e).__name__}")

        # Release containers
        pool_manager_fail_fast.release(c1)
        pool_manager_fail_fast.release(c2)

    finally:
        pool_manager_fail_fast.close()


def demo_shared_pool() -> None:
    """Demonstrate sharing a pool across multiple sessions."""
    print("\n4. Shared Pool Manager:")
    print("-" * 60)

    # Create a shared pool with enough capacity
    # Using fewer max_workers than pool size to avoid contention
    pool_manager = create_pool_manager(
        client=client,
        backend=SandboxBackend.DOCKER,
        config=PoolConfig(
            max_pool_size=5,  # Slightly larger than max_workers
            min_pool_size=2,
            exhaustion_strategy=ExhaustionStrategy.WAIT,
            acquisition_timeout=30.0,
        ),
        lang=SupportedLanguage.PYTHON,
    )

    try:
        num_tasks = 10
        max_concurrent = 3  # Fewer concurrent workers than pool size
        print(f"  Executing {num_tasks} sessions sharing the same pool (max {max_concurrent} concurrent)...")

        # Track container IDs used
        container_ids = set()
        successful_tasks = 0

        def execute_with_tracking(task_id: int) -> dict:
            nonlocal successful_tasks
            try:
                with SandboxSession(lang="python", pool=pool_manager, verbose=False) as session:
                    # Get container ID from the pooled container via backend session
                    try:
                        backend_session = getattr(session, "_backend_session", None)
                        if backend_session and hasattr(backend_session, "container") and backend_session.container:
                            container_id = backend_session.container.id[:12]
                            container_ids.add(container_id)
                    except Exception:  # noqa: BLE001
                        # Container ID tracking is optional, just skip if it fails
                        pass

                    # Run code with a small delay to simulate real work
                    result = session.run(f'import time; time.sleep(0.1); print("Task {task_id}")')
                    output = result.stdout.strip()
                    successful_tasks += 1
                    return {"task_id": task_id, "success": True, "output": output}
            except Exception as e:  # noqa: BLE001
                return {"task_id": task_id, "success": False, "error": str(e)}

        # Execute multiple sessions with controlled concurrency
        start_time = time.time()
        with ThreadPoolExecutor(max_workers=max_concurrent) as executor:
            results = list(executor.map(execute_with_tracking, range(1, num_tasks + 1)))

        total_time = time.time() - start_time

        # Report results
        failed_tasks = [r for r in results if not r["success"]]
        print(f"  ✓ Completed {successful_tasks}/{num_tasks} sessions in {total_time:.2f}s")

        if failed_tasks:
            print(f"  ✗ {len(failed_tasks)} tasks failed:")
            for task in failed_tasks:
                print(f"    Task {task['task_id']}: {task.get('error', 'Unknown error')}")

        if container_ids:
            print(f"  Used {len(container_ids)} unique containers (efficient reuse)")

        # Show pool stats
        stats = pool_manager.get_stats()
        print("\n  Final pool statistics:")
        print(f"    Total containers: {stats['total_size']}")
        print(f"    Idle: {stats['state_counts'].get('idle', 0)}")
        print(f"    Busy: {stats['state_counts'].get('busy', 0)}")

    finally:
        pool_manager.close()
        print("  Pool closed")


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
