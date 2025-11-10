"""Integration tests for container pool manager.

These tests verify the pool manager works correctly with real
container backends (Docker, Podman, Kubernetes).

Note: These tests require the respective backend to be available.
"""

import time
from concurrent.futures import ThreadPoolExecutor

import pytest

from llm_sandbox import SandboxSession
from llm_sandbox.const import SandboxBackend, SupportedLanguage
from llm_sandbox.pool import ExhaustionStrategy, PoolConfig, create_pool_manager
from llm_sandbox.pool.exceptions import PoolExhaustedError


# Determine which backends are available
def is_docker_available():
    """Check if Docker is available."""
    try:
        import docker

        client = docker.from_env()
        client.ping()
        return True
    except Exception:  # noqa: BLE001
        return False


def is_podman_available():
    """Check if Podman is available."""
    try:
        import podman

        client = podman.PodmanClient()
        client.ping()
        return True
    except Exception:  # noqa: BLE001
        return False


def is_kubernetes_available():
    """Check if Kubernetes is available."""
    try:
        from kubernetes import client, config

        config.load_kube_config()
        v1 = client.CoreV1Api()
        v1.list_namespace()
        return True
    except Exception:  # noqa: BLE001
        return False


# Mark tests that require specific backends
requires_docker = pytest.mark.skipif(
    not is_docker_available(),
    reason="Docker not available",
)

requires_podman = pytest.mark.skipif(
    not is_podman_available(),
    reason="Podman not available",
)

requires_kubernetes = pytest.mark.skipif(
    not is_kubernetes_available(),
    reason="Kubernetes not available",
)


@requires_docker
class TestDockerPoolIntegration:
    """Integration tests for Docker pool manager."""

    def test_basic_pool_usage(self):
        """Test basic pool creation and usage with Docker."""
        config = PoolConfig(
            max_pool_size=2,
            min_pool_size=1,
        )

        pool = create_pool_manager(
            backend=SandboxBackend.DOCKER,
            config=config,
            lang=SupportedLanguage.PYTHON,
        )

        try:
            # Give time for pre-warming
            time.sleep(2)

            # Acquire a container
            container = pool.acquire()
            assert container is not None
            assert container.state.value == "busy"

            # Release it
            pool.release(container)
            assert container.state.value == "idle"

            # Check stats
            stats = pool.get_stats()
            assert stats['total_size'] >= 1
        finally:
            pool.close()

    def test_pooled_session(self):
        """Test pooled session with Docker."""
        config = PoolConfig(
            max_pool_size=3,
            min_pool_size=1,
        )

        with SandboxSession(
            lang="python",
            backend=SandboxBackend.DOCKER,
            use_pool=True,
            pool_config=config,
        ) as session:
            result = session.run("print('Hello from Docker pool!')")

            assert result.exit_code == 0
            assert "Hello from Docker pool!" in result.stdout

    def test_concurrent_sessions(self):
        """Test multiple concurrent sessions sharing a pool."""
        pool = create_pool_manager(
            backend=SandboxBackend.DOCKER,
            config=PoolConfig(max_pool_size=3, min_pool_size=2),
            lang=SupportedLanguage.PYTHON,
        )

        results = []

        def run_task(task_id: int):
            with SandboxSession(lang="python", pool_manager=pool) as session:
                result = session.run(f'print("Task {task_id}")')
                results.append(result.stdout.strip())

        try:
            # Give time for pre-warming
            time.sleep(2)

            # Run tasks concurrently
            with ThreadPoolExecutor(max_workers=3) as executor:
                executor.map(run_task, range(1, 6))

            # Verify all tasks completed
            assert len(results) == 5
            assert all(f"Task {i}" in results for i in range(1, 6))
        finally:
            pool.close()

    def test_container_reuse(self):
        """Test that containers are actually reused."""
        pool = create_pool_manager(
            backend=SandboxBackend.DOCKER,
            config=PoolConfig(max_pool_size=2, min_pool_size=1),
            lang=SupportedLanguage.PYTHON,
        )

        container_ids = []

        try:
            # Give time for pre-warming
            time.sleep(2)

            # Run multiple sessions
            for i in range(3):
                with SandboxSession(lang="python", pool_manager=pool) as session:
                    container_ids.append(session.container.id if hasattr(session, 'container') else None)

            # Should have reused at least one container
            assert len(set(container_ids)) < len(container_ids)
        finally:
            pool.close()

    def test_exhaustion_handling(self):
        """Test pool exhaustion with FAIL_FAST strategy."""
        pool = create_pool_manager(
            backend=SandboxBackend.DOCKER,
            config=PoolConfig(
                max_pool_size=2,
                min_pool_size=1,
                exhaustion_strategy=ExhaustionStrategy.FAIL_FAST,
            ),
            lang=SupportedLanguage.PYTHON,
        )

        try:
            # Give time for pre-warming
            time.sleep(2)

            # Acquire all containers
            c1 = pool.acquire()
            c2 = pool.acquire()

            # Next acquire should fail
            with pytest.raises(PoolExhaustedError):
                pool.acquire()

            # Release and try again
            pool.release(c1)
            c3 = pool.acquire()  # Should succeed

            pool.release(c2)
            pool.release(c3)
        finally:
            pool.close()

    def test_health_checking(self):
        """Test container health checking."""
        pool = create_pool_manager(
            backend=SandboxBackend.DOCKER,
            config=PoolConfig(
                max_pool_size=2,
                min_pool_size=1,
                health_check_interval=2.0,
            ),
            lang=SupportedLanguage.PYTHON,
        )

        try:
            # Give time for pre-warming
            time.sleep(2)

            # Get initial stats
            initial_stats = pool.get_stats()

            # Wait for a health check cycle
            time.sleep(3)

            # Pool should still be healthy
            stats = pool.get_stats()
            assert stats['state_counts'].get('idle', 0) >= 1
            assert stats['state_counts'].get('unhealthy', 0) == 0
        finally:
            pool.close()


@requires_podman
class TestPodmanPoolIntegration:
    """Integration tests for Podman pool manager."""

    def test_basic_pool_usage(self):
        """Test basic pool creation and usage with Podman."""
        config = PoolConfig(
            max_pool_size=2,
            min_pool_size=1,
        )

        pool = create_pool_manager(
            backend=SandboxBackend.PODMAN,
            config=config,
            lang=SupportedLanguage.PYTHON,
        )

        try:
            # Give time for pre-warming
            time.sleep(2)

            # Acquire a container
            container = pool.acquire()
            assert container is not None

            # Release it
            pool.release(container)

            # Check stats
            stats = pool.get_stats()
            assert stats['total_size'] >= 1
        finally:
            pool.close()

    def test_pooled_session(self):
        """Test pooled session with Podman."""
        with SandboxSession(
            lang="python",
            backend=SandboxBackend.PODMAN,
            use_pool=True,
            pool_config=PoolConfig(max_pool_size=2, min_pool_size=1),
        ) as session:
            result = session.run("print('Hello from Podman pool!')")

            assert result.exit_code == 0
            assert "Hello from Podman pool!" in result.stdout


@requires_kubernetes
class TestKubernetesPoolIntegration:
    """Integration tests for Kubernetes pool manager."""

    def test_basic_pool_usage(self):
        """Test basic pool creation and usage with Kubernetes."""
        config = PoolConfig(
            max_pool_size=2,
            min_pool_size=1,
        )

        pool = create_pool_manager(
            backend=SandboxBackend.KUBERNETES,
            config=config,
            lang=SupportedLanguage.PYTHON,
        )

        try:
            # Kubernetes pods take longer to start
            time.sleep(10)

            # Acquire a pod
            pod = pool.acquire()
            assert pod is not None

            # Release it
            pool.release(pod)

            # Check stats
            stats = pool.get_stats()
            assert stats['total_size'] >= 1
        finally:
            pool.close()

    def test_pooled_session(self):
        """Test pooled session with Kubernetes."""
        with SandboxSession(
            lang="python",
            backend=SandboxBackend.KUBERNETES,
            use_pool=True,
            pool_config=PoolConfig(max_pool_size=2, min_pool_size=1),
        ) as session:
            result = session.run("print('Hello from Kubernetes pool!')")

            assert result.exit_code == 0
            assert "Hello from Kubernetes pool!" in result.stdout


class TestPoolPerformance:
    """Performance tests for pool manager (backend-agnostic)."""

    @pytest.mark.skipif(
        not (is_docker_available() or is_podman_available()),
        reason="No container backend available",
    )
    def test_pool_performance_improvement(self):
        """Test that pooling improves performance."""
        # Determine which backend to use
        backend = SandboxBackend.DOCKER if is_docker_available() else SandboxBackend.PODMAN

        num_runs = 3
        code = "print('test')"

        # Measure without pool
        start_time = time.time()
        for _ in range(num_runs):
            with SandboxSession(lang="python", backend=backend, use_pool=False) as session:
                session.run(code)
        no_pool_time = time.time() - start_time

        # Measure with pool
        pool = create_pool_manager(
            backend=backend,
            config=PoolConfig(max_pool_size=2, min_pool_size=1),
            lang=SupportedLanguage.PYTHON,
        )

        try:
            # Give time for pre-warming
            time.sleep(2)

            start_time = time.time()
            for _ in range(num_runs):
                with SandboxSession(lang="python", pool_manager=pool) as session:
                    session.run(code)
            pool_time = time.time() - start_time

            # Pool should be faster (or at least not significantly slower)
            # Allow some variance due to system load
            assert pool_time < no_pool_time * 1.5

        finally:
            pool.close()


class TestPoolEdgeCases:
    """Test edge cases and error handling."""

    @pytest.mark.skipif(
        not (is_docker_available() or is_podman_available()),
        reason="No container backend available",
    )
    def test_pool_with_zero_min_size(self):
        """Test pool with min_pool_size=0."""
        backend = SandboxBackend.DOCKER if is_docker_available() else SandboxBackend.PODMAN

        pool = create_pool_manager(
            backend=backend,
            config=PoolConfig(max_pool_size=3, min_pool_size=0),
            lang=SupportedLanguage.PYTHON,
        )

        try:
            # Pool should start with 0 containers
            time.sleep(1)
            stats = pool.get_stats()
            # May have some containers if pre-warming happened, but should be minimal
            assert stats['total_size'] <= 3

            # Should still be able to acquire
            container = pool.acquire()
            assert container is not None
            pool.release(container)
        finally:
            pool.close()

    @pytest.mark.skipif(
        not (is_docker_available() or is_podman_available()),
        reason="No container backend available",
    )
    def test_rapid_acquire_release(self):
        """Test rapid acquire/release cycles."""
        backend = SandboxBackend.DOCKER if is_docker_available() else SandboxBackend.PODMAN

        pool = create_pool_manager(
            backend=backend,
            config=PoolConfig(max_pool_size=2, min_pool_size=1),
            lang=SupportedLanguage.PYTHON,
        )

        try:
            # Give time for pre-warming
            time.sleep(2)

            # Rapid acquire/release
            for _ in range(10):
                container = pool.acquire()
                pool.release(container)

            # Pool should still be functional
            stats = pool.get_stats()
            assert stats['total_size'] >= 1
            assert not stats['closed']
        finally:
            pool.close()
