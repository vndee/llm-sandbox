"""Pool monitoring and health management demonstration.

This example demonstrates pool monitoring capabilities:
- Real-time pool statistics
- Container health checking
- Automatic container recycling
- Idle timeout handling
- Container lifecycle management
"""

import threading
import time
from datetime import datetime

from llm_sandbox import SandboxSession
from llm_sandbox.pool import PoolConfig, create_pool_manager


def print_pool_stats(pool, title: str = "Pool Statistics") -> None:
    """Print formatted pool statistics.

    Args:
        pool: Pool manager instance
        title: Title for the statistics output

    """
    stats = pool.get_stats()

    print(f"\n{title}:")
    print("-" * 50)
    print(f"  Total containers: {stats['total_size']}/{stats['max_size']}")
    print(f"  Minimum size: {stats['min_size']}")
    print("  Container states:")

    for state, count in stats['state_counts'].items():
        if count > 0:
            print(f"    {state}: {count}")

    print(f"  Pool closed: {stats['closed']}")


def demo_real_time_monitoring() -> None:
    """Demonstrate real-time pool monitoring."""
    print("\n1. Real-Time Pool Monitoring:")
    print("=" * 60)

    pool = create_pool_manager(
        backend="docker",
        config=PoolConfig(
            max_pool_size=3,
            min_pool_size=2,
            health_check_interval=5.0,
        ),
        lang="python",
    )

    try:
        # Initial state
        print_pool_stats(pool, "Initial State")

        # Acquire a container
        print("\nAcquiring container...")
        container1 = pool.acquire()
        print(f"   Acquired: {container1.container_id[:12]}")
        print_pool_stats(pool, "After Acquiring 1 Container")

        # Acquire another
        print("\nAcquiring another container...")
        container2 = pool.acquire()
        print(f"   Acquired: {container2.container_id[:12]}")
        print_pool_stats(pool, "After Acquiring 2 Containers")

        # Release one
        print("\nReleasing first container...")
        pool.release(container1)
        print(f"   Released: {container1.container_id[:12]}")
        print_pool_stats(pool, "After Releasing 1 Container")

        # Release second
        print("\nReleasing second container...")
        pool.release(container2)
        print(f"   Released: {container2.container_id[:12]}")
        print_pool_stats(pool, "Final State")

    finally:
        pool.close()
        print("\n Pool closed")


def demo_health_checking() -> None:
    """Demonstrate automatic health checking."""
    print("\n2. Automatic Health Checking:")
    print("=" * 60)

    pool = create_pool_manager(
        backend="docker",
        config=PoolConfig(
            max_pool_size=3,
            min_pool_size=2,
            health_check_interval=3.0,  # Check every 3 seconds
        ),
        lang="python",
    )

    try:
        print("  Pool created with health check interval: 3 seconds")
        print_pool_stats(pool, "Initial State")

        # Let health checks run for a bit
        print("\n  Waiting 10 seconds to observe health checks...")
        for i in range(10):
            time.sleep(1)
            print(f"    {i + 1}s...", end="", flush=True)

        print("\n")
        print_pool_stats(pool, "After Health Checks")

        # Use a container to ensure it stays healthy
        print("\n  Using a container...")
        with SandboxSession(lang="python", pool_manager=pool) as session:
            result = session.run('print("Health check test")')
            print(f"    Output: {result.stdout.strip()}")

        print_pool_stats(pool, "After Container Use")

    finally:
        pool.close()
        print("\n Pool closed")


def demo_idle_timeout() -> None:
    """Demonstrate idle timeout handling."""
    print("\n3. Idle Timeout Management:")
    print("=" * 60)

    pool = create_pool_manager(
        backend="docker",
        config=PoolConfig(
            max_pool_size=4,
            min_pool_size=1,
            idle_timeout=8.0,  # 8 second idle timeout
            health_check_interval=2.0,
        ),
        lang="python",
    )

    try:
        print("  Pool created with 8-second idle timeout")

        # Create some containers
        print("\n  Acquiring 3 containers...")
        containers = []
        for i in range(3):
            c = pool.acquire()
            containers.append(c)
            print(f"     Acquired: {c.container_id[:12]}")

        print_pool_stats(pool, "After Acquiring 3 Containers")

        # Release them all
        print("\n  Releasing all containers...")
        for c in containers:
            pool.release(c)
            print(f"     Released: {c.container_id[:12]}")

        print_pool_stats(pool, "After Releasing (All Idle)")

        # Wait for idle timeout
        print("\n  Waiting for idle timeout (8s) + health check...")
        for i in range(12):
            time.sleep(1)
            print(f"    {i + 1}s...", end="", flush=True)

        print("\n")
        print_pool_stats(pool, "After Idle Timeout")
        print("\n   Idle containers were recycled, min_pool_size maintained")

    finally:
        pool.close()
        print("\n Pool closed")


def demo_container_lifecycle() -> None:
    """Demonstrate container lifecycle management."""
    print("\n4. Container Lifecycle Management:")
    print("=" * 60)

    pool = create_pool_manager(
        backend="docker",
        config=PoolConfig(
            max_pool_size=3,
            min_pool_size=1,
            max_container_lifetime=15.0,  # 15 second lifetime
            max_container_uses=5,  # Max 5 uses
            health_check_interval=3.0,
        ),
        lang="python",
    )

    try:
        print("  Pool created with:")
        print("    - Max lifetime: 15 seconds")
        print("    - Max uses: 5")

        # Get a container and use it multiple times
        container = pool.acquire()
        print(f"\n  Acquired container: {container.container_id[:12]}")
        print(f"    Created at: {datetime.fromtimestamp(container.created_at).strftime('%H:%M:%S')}")
        print(f"    Use count: {container.use_count}")

        # Use it several times
        print("\n  Using container 5 times...")
        for i in range(5):
            pool.release(container)
            time.sleep(0.5)
            container = pool.acquire()
            print(f"    Use #{container.use_count}")

        # Release and try to get it again
        pool.release(container)
        print(f"\n  Released container (use_count={container.use_count})")

        # Next acquisition should potentially get a new container
        print("\n  Acquiring container again...")
        new_container = pool.acquire()
        print(f"  Got container: {new_container.container_id[:12]}")
        print(f"    Use count: {new_container.use_count}")

        if new_container.container_id != container.container_id:
            print("   Got a different container (old one was recycled)")
        else:
            print("   Got same container (not yet recycled)")

        pool.release(new_container)

        print_pool_stats(pool, "Final State")

    finally:
        pool.close()
        print("\n Pool closed")


def demo_concurrent_monitoring() -> None:
    """Demonstrate monitoring during concurrent operations."""
    print("\n5. Monitoring During Concurrent Operations:")
    print("=" * 60)

    pool = create_pool_manager(
        backend="docker",
        config=PoolConfig(
            max_pool_size=3,
            min_pool_size=1,
        ),
        lang="python",
    )

    # Flag to control monitoring thread
    stop_monitoring = threading.Event()
    stats_history = []

    def monitor_pool():
        """Background thread to monitor pool stats."""
        while not stop_monitoring.is_set():
            stats = pool.get_stats()
            stats_history.append({
                "time": datetime.now().strftime("%H:%M:%S"),
                "busy": stats['state_counts'].get('busy', 0),
                "idle": stats['state_counts'].get('idle', 0),
                "total": stats['total_size'],
            })
            time.sleep(0.5)

    try:
        # Start monitoring
        monitor_thread = threading.Thread(target=monitor_pool, daemon=True)
        monitor_thread.start()

        print("  Starting concurrent operations...")

        # Simulate concurrent workload
        def worker(worker_id: int):
            for i in range(2):
                with SandboxSession(lang="python", pool_manager=pool) as session:
                    time.sleep(0.3)
                    session.run(f'print("Worker {worker_id}, iteration {i}")')

        # Run workers
        threads = [threading.Thread(target=worker, args=(i,)) for i in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # Stop monitoring
        stop_monitoring.set()
        monitor_thread.join(timeout=2)

        # Print statistics history
        print("\n  Pool activity over time:")
        print("  " + "-" * 46)
        print("  Time     | Busy | Idle | Total")
        print("  " + "-" * 46)

        for stat in stats_history[::3]:  # Sample every 3rd entry
            print(f"  {stat['time']} |   {stat['busy']}  |   {stat['idle']}  |   {stat['total']}")

        print("  " + "-" * 46)
        print("\n   Pool handled concurrent requests efficiently")

    finally:
        pool.close()
        print("\n Pool closed")


def main() -> None:
    """Run all monitoring demonstrations."""
    print("=" * 60)
    print("Pool Monitoring & Health Management Demo")
    print("=" * 60)

    demo_real_time_monitoring()
    time.sleep(1)

    demo_health_checking()
    time.sleep(1)

    demo_idle_timeout()
    time.sleep(1)

    demo_container_lifecycle()
    time.sleep(1)

    demo_concurrent_monitoring()

    print("\n" + "=" * 60)
    print("Demo Complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
