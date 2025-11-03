"""Basic container pool usage demonstration.

This example demonstrates the core functionality of the container pool manager:
- Creating a pool with configuration
- Acquiring and releasing containers
- Automatic container reuse
- Pool statistics monitoring
"""

from llm_sandbox import SandboxSession
from llm_sandbox.pool import PoolConfig


def main() -> None:
    """Demonstrate basic pool usage."""
    # Configure the container pool
    pool_config = PoolConfig(
        max_pool_size=5,  # Maximum 5 containers
        min_pool_size=2,  # Keep at least 2 warm containers
        idle_timeout=60.0,  # Recycle containers idle for > 60 seconds
        health_check_interval=30.0,  # Check health every 30 seconds
        enable_prewarming=True,  # Pre-warm containers with environment setup
    )

    print("=" * 60)
    print("Basic Container Pool Demo")
    print("=" * 60)

    # Example 1: Simple pooled session
    print("\n1. Simple pooled session:")
    print("-" * 40)

    with SandboxSession(
        lang="python",
        use_pool=True,
        pool_config=pool_config,
        verbose=False,
    ) as session:
        # Run some code
        code = """
print("Hello from pooled container!")
print(f"2 + 2 = {2 + 2}")
"""
        result = session.run(code)
        print(f"Output: {result.stdout.strip()}")

    print("\n Container returned to pool automatically")

    # Example 2: Multiple executions reusing containers
    print("\n2. Multiple executions (containers are reused):")
    print("-" * 40)

    for i in range(3):
        with SandboxSession(
            lang="python",
            use_pool=True,
            pool_config=pool_config,
            verbose=False,
        ) as session:
            code = f'print("Execution #{i + 1}")'
            result = session.run(code)
            print(f"  {result.stdout.strip()}")

    print("\n All 3 executions reused containers from the pool")

    # Example 3: Pre-installed libraries
    print("\n3. Pre-installed libraries in pool:")
    print("-" * 40)

    # Create a pool with pre-installed libraries
    pool_config_with_libs = PoolConfig(
        max_pool_size=3,
        min_pool_size=1,
        enable_prewarming=True,
        prewarm_libraries=["requests"],  # Pre-install requests
    )

    with SandboxSession(
        lang="python",
        use_pool=True,
        pool_config=pool_config_with_libs,
        verbose=False,
    ) as session:
        # This will be fast because requests is already installed
        code = """
import requests
print(f"requests version: {requests.__version__}")
print(" Library already installed in pool!")
"""
        result = session.run(code)
        print(f"  {result.stdout.strip()}")

    # Example 4: Checking pool statistics
    print("\n4. Pool statistics:")
    print("-" * 40)

    from llm_sandbox.pool import create_pool_manager

    # Create a pool manager directly to access statistics
    pool = create_pool_manager(
        backend="docker",
        config=PoolConfig(
            max_pool_size=4,
            min_pool_size=2,
        ),
        lang="python",
    )

    try:
        # Check initial stats
        stats = pool.get_stats()
        print(f"  Total containers: {stats['total_size']}")
        print(f"  Max pool size: {stats['max_size']}")
        print(f"  Min pool size: {stats['min_size']}")
        print("  State counts:")
        for state, count in stats['state_counts'].items():
            if count > 0:
                print(f"    {state}: {count}")

        # Use the pool
        container = pool.acquire()
        print(f"\n  Acquired container: {container.container_id[:12]}...")

        # Check stats after acquisition
        stats = pool.get_stats()
        print(f"  Busy containers: {stats['state_counts'].get('busy', 0)}")

        # Release back to pool
        pool.release(container)
        print("  Released container back to pool")

        # Final stats
        stats = pool.get_stats()
        print(f"  Idle containers: {stats['state_counts'].get('idle', 0)}")

    finally:
        pool.close()
        print("\n Pool closed and all containers cleaned up")

    print("\n" + "=" * 60)
    print("Demo Complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
