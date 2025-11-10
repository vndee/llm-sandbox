# ruff: noqa: T201, F841

"""Basic container pool usage demonstration.

This example demonstrates the core functionality of the container pool manager:
- Creating a pool with configuration
- Acquiring and releasing containers
- Automatic container reuse
- Pool statistics monitoring
"""

import logging

import docker

from llm_sandbox import SandboxBackend, SandboxSession, SupportedLanguage
from llm_sandbox.pool import ContainerPoolManager, PoolConfig, create_pool_manager

logging.basicConfig(level=logging.INFO)

client = docker.DockerClient(base_url="unix:///Users/vndee/.docker/run/docker.sock")


def _print_demo_header() -> None:
    print("=" * 60)
    print("Basic Container Pool Demo")
    print("=" * 60)


def _print_demo_footer() -> None:
    print("\n" + "=" * 60)
    print("Demo Complete!")
    print("=" * 60)


def _example_simple_pooled_session(pool_manager: ContainerPoolManager) -> None:
    print("\n1. Simple pooled session:")
    print("-" * 40)

    with SandboxSession(
        lang="python",
        pool=pool_manager,
    ) as session:
        code = """
print("Hello from pooled container!")
print(f"2 + 2 = {2 + 2}")
"""
        result = session.run(code)
        print(f"Output: {result.stdout.strip()}")

    print("\nContainer returned to pool automatically")


def _example_multiple_executions(pool_manager: ContainerPoolManager) -> None:
    print("\n2. Multiple executions (containers are reused):")
    print("-" * 40)

    for i in range(3):
        with SandboxSession(lang="python", pool=pool_manager) as session:
            code = f'print("Execution #{i + 1}")'
            result = session.run(code)
            print(f"  {result.stdout.strip()}")

    print("\nAll 3 executions reused containers from the pool")


def _example_preinstalled_libraries() -> None:
    print("\n3. Pre-installed libraries in pool:")
    print("-" * 40)

    pool_config = PoolConfig(
        max_pool_size=3,
        min_pool_size=1,
        enable_prewarming=True,
    )

    pool = create_pool_manager(
        backend=SandboxBackend.DOCKER,
        config=pool_config,
        lang=SupportedLanguage.PYTHON,
        client=client,
        libraries=["numpy", "pandas"],
    )

    with SandboxSession(
        pool=pool,
        lang=SupportedLanguage.PYTHON,
        verbose=False,
    ) as session:
        result = session.run("import numpy; print(numpy.__version__)")
        print(f"  {result.stdout.strip()}")
        result = session.run("import pandas; print(pandas.__version__)")
        print(f"  {result.stdout.strip()}")

    pool.close()


def _example_pool_statistics() -> None:
    print("\n4. Pool statistics:")
    print("-" * 40)

    from llm_sandbox.pool import create_pool_manager

    pool = create_pool_manager(
        backend=SandboxBackend.DOCKER,
        config=PoolConfig(
            max_pool_size=4,
            min_pool_size=2,
        ),
        lang=SupportedLanguage.PYTHON,
        client=client,
    )

    try:
        stats = pool.get_stats()
        print(f"  Total containers: {stats['total_size']}")
        print(f"  Max pool size: {stats['max_size']}")
        print(f"  Min pool size: {stats['min_size']}")
        print("  State counts:")
        for state, count in stats["state_counts"].items():
            if count > 0:
                print(f"    {state}: {count}")

        container = pool.acquire()
        print(f"\n  Acquired container: {container.container_id[:12]}...")

        stats = pool.get_stats()
        print(f"  Busy containers: {stats['state_counts'].get('busy', 0)}")

        pool.release(container)
        print("  Released container back to pool")

        stats = pool.get_stats()
        print(f"  Idle containers: {stats['state_counts'].get('idle', 0)}")

    finally:
        pool.close()
        print("\nPool closed and all containers cleaned up")


def _example_shared_pool_manager() -> None:
    print("\n5. Shared pool manager:")
    print("-" * 40)

    from llm_sandbox.pool import create_pool_manager

    pool = create_pool_manager(
        backend=SandboxBackend.DOCKER,
        config=PoolConfig(
            max_pool_size=10,
            min_pool_size=3,
        ),
        lang=SupportedLanguage.PYTHON,
        libraries=["numpy", "pandas"],
        client=client,
    )

    try:
        with SandboxSession(lang="python", pool=pool) as session1:
            result1 = session1.run("import pandas; print(pandas.__version__)")
            _ = result1

        with SandboxSession(lang="python", pool=pool) as session2:
            result2 = session2.run("import numpy; print(numpy.__version__)")
            _ = result2
    finally:
        pool.close()


def main() -> None:
    """Demonstrate basic pool usage."""
    pool_config = PoolConfig(
        max_pool_size=5,
        min_pool_size=2,
        idle_timeout=60.0,
        health_check_interval=30.0,
        enable_prewarming=True,
    )

    pool_manager = create_pool_manager(
        backend=SandboxBackend.DOCKER,
        config=pool_config,
        lang=SupportedLanguage.PYTHON,
        client=client,
    )

    _print_demo_header()
    _example_simple_pooled_session(pool_manager)
    _example_multiple_executions(pool_manager)
    _example_preinstalled_libraries()
    _example_pool_statistics()
    _example_shared_pool_manager()
    _print_demo_footer()

    pool_manager.close()


if __name__ == "__main__":
    main()
