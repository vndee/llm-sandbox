# ruff: noqa: T201, F841

"""Kubernetes-specific container pool demonstration.

This example showcases container pooling benefits specifically for Kubernetes,
where pod creation is slower (~7-8s) compared to Docker containers (~2-3s).

Key demonstrations:
1. Amortization of setup costs over many tasks
2. Throughput improvements for production workloads
3. Optimal pool sizing for different scenarios
"""

import statistics
import time

from llm_sandbox import SandboxBackend, SandboxSession, SupportedLanguage
from llm_sandbox.pool import ExhaustionStrategy, PoolConfig, create_pool_manager


def demo_large_batch_sequential() -> None:
    """Demonstrate pooling benefits for large batch of sequential tasks.

    This shows how pooling amortizes the setup cost over many tasks,
    which is ideal for batch processing scenarios.
    """
    print("\n1. Large Batch Processing (30 Sequential Tasks):")
    print("-" * 60)

    num_tasks = 30
    simple_code = 'print("Task completed")'

    # Without pooling
    print(f"  Running {num_tasks} tasks WITHOUT pooling...")
    start_time = time.time()
    times_no_pool = []

    for i in range(1, num_tasks + 1):
        task_start = time.time()
        try:
            with SandboxSession(
                lang="python",
                verbose=False,
                backend=SandboxBackend.KUBERNETES,
            ) as session:
                session.run(simple_code)
            task_time = time.time() - task_start
            times_no_pool.append(task_time)
            if i == 1 or i % 10 == 0:
                print(f"    Task {i:2d}: {task_time:.2f}s")
        except Exception as e:  # noqa: BLE001
            print(f"    Task {i:2d}: FAILED - {e!s}")

    total_no_pool = time.time() - start_time
    print(f"\n  Total time: {total_no_pool:.2f}s")
    print(f"  Average per task: {statistics.mean(times_no_pool):.2f}s")
    print(f"  Throughput: {num_tasks / total_no_pool:.2f} tasks/second")

    # With pooling
    print(f"\n  Running {num_tasks} tasks WITH pooling (max_pool_size=3)...")
    pool_manager = create_pool_manager(
        backend=SandboxBackend.KUBERNETES,
        config=PoolConfig(
            max_pool_size=3,
            min_pool_size=2,
            enable_prewarming=True,
        ),
        lang=SupportedLanguage.PYTHON,
    )

    try:
        start_time = time.time()
        times_with_pool = []

        for i in range(1, num_tasks + 1):
            task_start = time.time()
            try:
                with SandboxSession(
                    lang="python",
                    pool=pool_manager,
                    verbose=False,
                ) as session:
                    session.run(simple_code)
                task_time = time.time() - task_start
                times_with_pool.append(task_time)
                if i == 1 or i % 10 == 0:
                    print(f"    Task {i:2d}: {task_time:.2f}s")
            except Exception as e:  # noqa: BLE001
                print(f"    Task {i:2d}: FAILED - {e!s}")

        total_with_pool = time.time() - start_time
        print(f"\n  Total time: {total_with_pool:.2f}s")
        print(f"  Average per task: {statistics.mean(times_with_pool):.2f}s")
        print(f"  Throughput: {num_tasks / total_with_pool:.2f} tasks/second")

        # Comparison
        speedup = total_no_pool / total_with_pool
        time_saved = total_no_pool - total_with_pool
        print("\n  📊 Results:")
        print(f"     Speedup: {speedup:.2f}x faster")
        print(f"     Time saved: {time_saved:.2f}s ({time_saved / 60:.1f} minutes)")
        print(f"     Throughput improvement: {(speedup - 1) * 100:.1f}%")

    finally:
        pool_manager.close()


def demo_warmup_vs_steady_state() -> None:
    """Demonstrate the difference between warmup phase and steady-state performance.

    This shows that after the initial warmup, pooled sessions are much faster.
    """
    print("\n2. Warmup vs Steady-State Performance:")
    print("-" * 60)

    num_warmup_tasks = 5
    num_steady_tasks = 15
    simple_code = 'print("Hello")'

    pool_manager = create_pool_manager(
        backend=SandboxBackend.KUBERNETES,
        config=PoolConfig(
            max_pool_size=3,
            min_pool_size=3,  # Pre-create all containers
            enable_prewarming=True,
        ),
        lang=SupportedLanguage.PYTHON,
    )

    try:
        print(f"  Warmup phase ({num_warmup_tasks} tasks):")
        warmup_times = []
        for i in range(1, num_warmup_tasks + 1):
            task_start = time.time()
            with SandboxSession(
                lang="python",
                pool=pool_manager,
                verbose=False,
            ) as session:
                session.run(simple_code)
            task_time = time.time() - task_start
            warmup_times.append(task_time)
            print(f"    Task {i}: {task_time:.2f}s")

        print(f"  Average warmup time: {statistics.mean(warmup_times):.2f}s")

        print(f"\n  Steady-state phase ({num_steady_tasks} tasks):")
        steady_times = []
        for i in range(1, num_steady_tasks + 1):
            task_start = time.time()
            with SandboxSession(
                lang="python",
                pool=pool_manager,
                verbose=False,
            ) as session:
                session.run(simple_code)
            task_time = time.time() - task_start
            steady_times.append(task_time)
            if i == 1 or i % 5 == 0:
                print(f"    Task {i:2d}: {task_time:.2f}s")

        print(f"  Average steady-state time: {statistics.mean(steady_times):.2f}s")

        improvement = statistics.mean(warmup_times) / statistics.mean(steady_times)
        print("\n  📊 Results:")
        print(f"     Steady-state is {improvement:.2f}x faster than warmup")
        print(f"     After warmup, each task takes only {statistics.mean(steady_times):.2f}s")
        print(f"     Consistent performance: std dev = {statistics.stdev(steady_times):.2f}s")

    finally:
        pool_manager.close()


def demo_pool_size_impact() -> None:
    """Demonstrate how pool size affects performance for sequential workloads."""
    print("\n3. Pool Size Impact (15 Sequential Tasks):")
    print("-" * 60)

    num_tasks = 15
    simple_code = 'import time; time.sleep(0.1); print("Done")'

    pool_sizes = [1, 2, 3]
    results = {}

    for pool_size in pool_sizes:
        print(f"\n  Testing with pool_size={pool_size}...")
        pool_manager = create_pool_manager(
            backend=SandboxBackend.KUBERNETES,
            config=PoolConfig(
                max_pool_size=pool_size,
                min_pool_size=pool_size,
                enable_prewarming=True,
            ),
            lang=SupportedLanguage.PYTHON,
        )

        try:
            start_time = time.time()
            task_times = []

            for _ in range(1, num_tasks + 1):
                task_start = time.time()
                with SandboxSession(
                    lang="python",
                    pool=pool_manager,
                    verbose=False,
                ) as session:
                    session.run(simple_code)
                task_time = time.time() - task_start
                task_times.append(task_time)

            total_time = time.time() - start_time

            # Separate warmup from steady-state
            warmup_avg = statistics.mean(task_times[:pool_size])
            steady_avg = statistics.mean(task_times[pool_size:]) if len(task_times) > pool_size else warmup_avg

            results[pool_size] = {
                "total": total_time,
                "warmup_avg": warmup_avg,
                "steady_avg": steady_avg,
                "throughput": num_tasks / total_time,
            }

            print(f"    Total time: {total_time:.2f}s")
            print(f"    Warmup avg: {warmup_avg:.2f}s")
            print(f"    Steady-state avg: {steady_avg:.2f}s")
            print(f"    Throughput: {results[pool_size]['throughput']:.2f} tasks/s")

        finally:
            pool_manager.close()

    print("\n  📊 Comparison:")
    print(f"     {'Pool Size':<12} {'Total Time':<12} {'Throughput':<15} {'Steady Avg':<12}")
    print(f"     {'-' * 12} {'-' * 12} {'-' * 15} {'-' * 12}")
    for size in pool_sizes:
        r = results[size]
        print(f"     {size:<12} {r['total']:.2f}s{'':<6} {r['throughput']:.2f} tasks/s{'':<3} {r['steady_avg']:.2f}s")


def demo_production_simulation() -> None:
    """Simulate a production service handling incoming requests.

    This demonstrates the real-world benefit: after warmup, the pool
    provides consistently fast response times.
    """
    print("\n4. Production Service Simulation (50 Requests):")
    print("-" * 60)

    num_requests = 50
    request_code = 'result = sum(range(1000)); print(f"Result: {result}")'

    pool_manager = create_pool_manager(
        backend=SandboxBackend.KUBERNETES,
        config=PoolConfig(
            max_pool_size=5,
            min_pool_size=3,
            enable_prewarming=True,
            exhaustion_strategy=ExhaustionStrategy.WAIT,
            acquisition_timeout=30.0,
        ),
        lang=SupportedLanguage.PYTHON,
    )

    try:
        print(f"  Processing {num_requests} requests...")
        all_times = []
        start_time = time.time()

        for i in range(1, num_requests + 1):
            request_start = time.time()
            try:
                with SandboxSession(
                    lang="python",
                    pool=pool_manager,
                    verbose=False,
                ) as session:
                    session.run(request_code)
                request_time = time.time() - request_start
                all_times.append(request_time)

                if i in [1, 10, 20, 30, 40, 50]:
                    print(f"    Request {i:2d}: {request_time:.2f}s")

            except Exception as e:  # noqa: BLE001
                print(f"    Request {i:2d}: FAILED - {e!s}")

        total_time = time.time() - start_time

        # Split into warmup and steady-state (after first 10 requests)
        warmup_times = all_times[:10]
        steady_times = all_times[10:]

        print("\n  📊 Results:")
        print(f"     Total time: {total_time:.2f}s")
        print(f"     Total requests processed: {len(all_times)}")
        print(f"     Average throughput: {len(all_times) / total_time:.2f} requests/s")
        print("\n     Warmup phase (first 10 requests):")
        print(f"       Average: {statistics.mean(warmup_times):.2f}s")
        print(f"       Min: {min(warmup_times):.2f}s, Max: {max(warmup_times):.2f}s")
        print(f"\n     Steady-state (requests 11-{num_requests}):")
        print(f"       Average: {statistics.mean(steady_times):.2f}s")
        print(f"       Min: {min(steady_times):.2f}s, Max: {max(steady_times):.2f}s")
        print(f"       Std dev: {statistics.stdev(steady_times):.2f}s")
        print(f"       P95 latency: {sorted(steady_times)[int(len(steady_times) * 0.95)]:.2f}s")
        print(f"       P99 latency: {sorted(steady_times)[int(len(steady_times) * 0.99)]:.2f}s")

        # Show pool stats
        stats = pool_manager.get_stats()
        print("\n     Final pool state:")
        print(f"       Total containers: {stats['total_size']}")
        print(f"       Containers used: {stats['max_pool_size']}")
        print(f"       Current idle: {stats['state_counts'].get('idle', 0)}")

    finally:
        pool_manager.close()


def main() -> None:
    """Run all Kubernetes-specific demonstrations."""
    print("=" * 60)
    print("Kubernetes Container Pooling Demo")
    print("=" * 60)
    print("\nNote: Kubernetes pod creation is slower (~7-8s) than Docker (~2-3s).")
    print("Pooling benefits are most visible with:")
    print("  - Large batches of tasks (20+)")
    print("  - Long-running services")
    print("  - Steady-state workloads after warmup")
    print("=" * 60)

    demo_large_batch_sequential()
    demo_warmup_vs_steady_state()
    demo_pool_size_impact()
    demo_production_simulation()

    print("\n" + "=" * 60)
    print("Demo Complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
