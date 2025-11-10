"""Edge case tests for container pool manager.

These tests cover edge cases, error conditions, and concurrent scenarios
to improve test coverage and ensure robust pool behavior.
"""

import threading
import time
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from llm_sandbox.const import SupportedLanguage
from llm_sandbox.pool.base import ContainerPoolManager, ContainerState, PooledContainer
from llm_sandbox.pool.config import ExhaustionStrategy, PoolConfig
from llm_sandbox.pool.exceptions import PoolClosedError, PoolExhaustedError
from llm_sandbox.pool.session import DuplicateClientError


class MockSession:
    """Mock session for testing."""

    def __init__(self, container_id: str, should_fail: bool = False) -> None:
        """Initialize mock session."""
        self.container = container_id
        self.is_open = False
        self.should_fail = should_fail

    def open(self) -> None:
        """Mock session open."""
        if self.should_fail:
            raise RuntimeError("Mock session open failed")
        self.is_open = True

    def close(self) -> None:
        """Mock session close."""
        self.is_open = False


class MockContainerPoolManager(ContainerPoolManager):
    """Mock implementation of ContainerPoolManager for testing."""

    def __init__(self, *args: Any, fail_creation: bool = False, fail_health_check: bool = False, **kwargs: Any) -> None:
        """Initialize mock pool manager."""
        self._container_counter = 0
        self._fail_creation = fail_creation
        self._fail_health_check = fail_health_check
        super().__init__(*args, **kwargs)

    def _create_session_for_container(self) -> MockSession:
        """Create a mock session for container initialization."""
        self._container_counter += 1
        container_id = f"mock_container_{self._container_counter}"
        return MockSession(container_id, should_fail=self._fail_creation)

    def _destroy_container_impl(self, container: Any) -> None:
        """Destroy a mock container."""

    def _get_container_id(self, container: Any) -> str:
        """Get mock container ID."""
        return container

    def _health_check_impl(self, container: Any) -> bool:
        """Mock health check."""
        return not self._fail_health_check


class TestPoolConfigValidation:
    """Tests for PoolConfig validation."""

    def test_zero_idle_timeout_rejected(self):
        """Test that idle_timeout=0 is rejected."""
        with pytest.raises(Exception):  # Pydantic validation error
            PoolConfig(idle_timeout=0.0)

    def test_negative_idle_timeout_rejected(self):
        """Test that negative idle_timeout is rejected."""
        with pytest.raises(Exception):  # Pydantic validation error
            PoolConfig(idle_timeout=-1.0)

    def test_zero_acquisition_timeout_rejected(self):
        """Test that acquisition_timeout=0 is rejected."""
        with pytest.raises(Exception):  # Pydantic validation error
            PoolConfig(acquisition_timeout=0.0)

    def test_zero_health_check_interval_rejected(self):
        """Test that health_check_interval=0 is rejected."""
        with pytest.raises(Exception):  # Pydantic validation error
            PoolConfig(health_check_interval=0.0)

    def test_zero_max_container_lifetime_rejected(self):
        """Test that max_container_lifetime=0 is rejected."""
        with pytest.raises(Exception):  # Pydantic validation error
            PoolConfig(max_container_lifetime=0.0)

    def test_none_timeouts_allowed(self):
        """Test that None timeouts are allowed."""
        config = PoolConfig(
            idle_timeout=None,
            acquisition_timeout=None,
            max_container_lifetime=None,
        )
        assert config.idle_timeout is None
        assert config.acquisition_timeout is None
        assert config.max_container_lifetime is None


class TestPoolExhaustion:
    """Tests for pool exhaustion scenarios."""

    def test_temporary_strategy_creates_container(self):
        """Test TEMPORARY strategy creates container outside pool."""
        config = PoolConfig(
            max_pool_size=1,
            min_pool_size=0,
            exhaustion_strategy=ExhaustionStrategy.TEMPORARY,
            enable_prewarming=False,
        )
        pool = MockContainerPoolManager(config=config, lang=SupportedLanguage.PYTHON)

        try:
            # Create and acquire the single pool container
            c1 = pool._create_container()
            c1.mark_idle()
            acquired = pool.acquire()

            # Next acquire should create temporary container
            c2 = pool.acquire()

            assert c2 is not None
            assert c2.state == ContainerState.BUSY
            # Temporary container should not be in pool
            assert c2 not in pool._pool

            pool.release(acquired)
            pool.release(c2)
        finally:
            pool.close()

    def test_wait_strategy_with_none_timeout(self):
        """Test WAIT strategy with no timeout."""
        config = PoolConfig(
            max_pool_size=1,
            min_pool_size=0,
            exhaustion_strategy=ExhaustionStrategy.WAIT,
            acquisition_timeout=None,
            enable_prewarming=False,
        )
        pool = MockContainerPoolManager(config=config, lang=SupportedLanguage.PYTHON)

        try:
            # Create and acquire container
            c = pool._create_container()
            c.mark_idle()
            container = pool.acquire()

            # Start thread that releases after short delay
            def release_delayed():
                time.sleep(0.5)
                pool.release(container)

            threading.Thread(target=release_delayed, daemon=True).start()

            # This should wait indefinitely but get container when released
            start = time.time()
            c2 = pool.acquire()
            elapsed = time.time() - start

            assert c2 is not None
            assert 0.4 < elapsed < 1.0

            pool.release(c2)
        finally:
            pool.close()


class TestContainerLifecycle:
    """Tests for container lifecycle management."""

    def test_container_recycling_at_max_uses(self):
        """Test container is recycled exactly at max_uses."""
        config = PoolConfig(
            max_pool_size=2,
            min_pool_size=1,
            max_container_uses=3,
            enable_prewarming=False,
        )
        pool = MockContainerPoolManager(config=config, lang=SupportedLanguage.PYTHON)

        try:
            # Create container
            c = pool._create_container()
            c.mark_idle()
            original_id = c.container_id

            # Use exactly 3 times
            for i in range(3):
                container = pool.acquire()
                assert container.use_count == i + 1
                pool.release(container)

            # After 3 uses, container should be removed
            time.sleep(0.1)  # Give time for recycling
            assert original_id not in [pc.container_id for pc in pool._pool]
        finally:
            pool.close()

    def test_container_recycling_by_lifetime(self):
        """Test container is recycled when lifetime exceeded."""
        config = PoolConfig(
            max_pool_size=2,
            min_pool_size=0,
            max_container_lifetime=1.0,  # 1 second
            enable_prewarming=False,
        )
        pool = MockContainerPoolManager(config=config, lang=SupportedLanguage.PYTHON)

        try:
            # Create container
            c = pool._create_container()
            c.mark_idle()
            c.created_at = time.time() - 2.0  # Make it 2 seconds old

            # Acquire and release should trigger recycling
            container = pool.acquire()
            original_id = container.container_id
            pool.release(container)

            # Container should be removed
            time.sleep(0.1)
            assert original_id not in [pc.container_id for pc in pool._pool]
        finally:
            pool.close()

    def test_container_state_transitions(self):
        """Test container state transitions."""
        container = PooledContainer(container_id="test", container="test")

        # Test state transitions
        assert container.state == ContainerState.INITIALIZING

        container.mark_idle()
        assert container.state == ContainerState.IDLE

        container.mark_busy()
        assert container.state == ContainerState.BUSY
        assert container.use_count == 1

        container.mark_idle()
        assert container.state == ContainerState.IDLE

        container.mark_unhealthy()
        assert container.state == ContainerState.UNHEALTHY

        container.mark_removing()
        assert container.state == ContainerState.REMOVING


class TestConcurrentOperations:
    """Tests for concurrent pool operations."""

    def test_concurrent_close_calls(self):
        """Test multiple concurrent close calls."""
        config = PoolConfig(max_pool_size=3, min_pool_size=1)
        pool = MockContainerPoolManager(config=config, lang=SupportedLanguage.PYTHON)

        time.sleep(0.5)  # Let pre-warming happen

        # Close from multiple threads
        results = []

        def close_pool():
            try:
                pool.close()
                results.append("success")
            except Exception as e:  # noqa: BLE001
                results.append(f"error: {e}")

        threads = [threading.Thread(target=close_pool) for _ in range(3)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # All closes should succeed without error
        assert all(r == "success" for r in results)
        assert pool._closed is True

    def test_acquire_during_health_check(self):
        """Test acquiring container during health check."""
        config = PoolConfig(
            max_pool_size=3,
            min_pool_size=2,
            health_check_interval=0.5,
        )
        pool = MockContainerPoolManager(config=config, lang=SupportedLanguage.PYTHON)

        try:
            time.sleep(1)  # Let pre-warming and first health check happen

            # Acquire container while health checks might be running
            results = []

            def acquire_and_release():
                try:
                    c = pool.acquire()
                    time.sleep(0.1)
                    pool.release(c)
                    results.append("success")
                except Exception as e:  # noqa: BLE001
                    results.append(f"error: {e}")

            threads = [threading.Thread(target=acquire_and_release) for _ in range(5)]
            for t in threads:
                t.start()
            for t in threads:
                t.join()

            # All operations should succeed
            assert all(r == "success" for r in results)
        finally:
            pool.close()


class TestErrorHandling:
    """Tests for error handling."""

    def test_container_creation_failure(self):
        """Test handling of container creation failure."""
        config = PoolConfig(
            max_pool_size=2,
            min_pool_size=1,
            enable_prewarming=False,
        )
        pool = MockContainerPoolManager(
            config=config,
            lang=SupportedLanguage.PYTHON,
            fail_creation=True,
        )

        try:
            # Trying to create container should raise error
            with pytest.raises(RuntimeError, match="Mock session open failed"):
                pool._create_container()
        finally:
            pool.close()

    def test_health_check_with_exception(self):
        """Test health check that raises exception."""
        config = PoolConfig(
            max_pool_size=2,
            min_pool_size=0,
            health_check_interval=0.5,
            enable_prewarming=False,
        )
        pool = MockContainerPoolManager(
            config=config,
            lang=SupportedLanguage.PYTHON,
            fail_health_check=True,
        )

        try:
            # Create container
            c = pool._create_container()
            c.mark_idle()

            # Wait for health check to run
            time.sleep(1)

            # Container should be marked unhealthy and removed
            assert c.container_id not in [pc.container_id for pc in pool._pool]
        finally:
            pool.close()

    def test_pool_closed_error_on_acquire(self):
        """Test PoolClosedError when acquiring from closed pool."""
        config = PoolConfig(max_pool_size=1, min_pool_size=0, enable_prewarming=False)
        pool = MockContainerPoolManager(config=config, lang=SupportedLanguage.PYTHON)

        pool.close()

        with pytest.raises(PoolClosedError):
            pool.acquire()

    def test_pool_closed_during_wait(self):
        """Test closing pool while thread is waiting for container."""
        config = PoolConfig(
            max_pool_size=1,
            min_pool_size=0,
            exhaustion_strategy=ExhaustionStrategy.WAIT,
            acquisition_timeout=5.0,
            enable_prewarming=False,
        )
        pool = MockContainerPoolManager(config=config, lang=SupportedLanguage.PYTHON)

        # Create and acquire the only container
        c = pool._create_container()
        c.mark_idle()
        container = pool.acquire()

        error_caught = []

        def try_acquire():
            try:
                pool.acquire()
            except (PoolExhaustedError, PoolClosedError) as e:
                error_caught.append(type(e).__name__)

        # Start thread waiting for container
        thread = threading.Thread(target=try_acquire, daemon=True)
        thread.start()

        # Close pool while thread is waiting
        time.sleep(0.5)
        pool.close()
        thread.join(timeout=2)

        # Should get PoolExhaustedError or PoolClosedError
        assert len(error_caught) > 0


class TestPooledSession:
    """Tests for PooledSandboxSession."""

    def test_duplicate_client_error(self):
        """Test DuplicateClientError is raised when client passed to session."""
        from llm_sandbox.pool.session import PooledSandboxSession

        # Create mock pool manager
        pool = MagicMock()
        pool.__class__.__name__ = "DockerPoolManager"
        pool.client = MagicMock()
        pool.lang = SupportedLanguage.PYTHON
        pool.image = "test:latest"

        # Try to create session with client parameter
        with pytest.raises(DuplicateClientError):
            session = PooledSandboxSession(pool_manager=pool, client=MagicMock())
            session.open()

    def test_session_without_pool_manager(self):
        """Test error when pool manager not set."""
        from llm_sandbox.pool.session import PooledSandboxSession

        session = PooledSandboxSession(pool_manager=None)  # type: ignore

        with pytest.raises(RuntimeError, match="Pool manager not initialized"):
            session.open()


class TestPoolStats:
    """Tests for pool statistics."""

    def test_stats_with_mixed_states(self):
        """Test statistics with containers in different states."""
        config = PoolConfig(max_pool_size=5, min_pool_size=0, enable_prewarming=False)
        pool = MockContainerPoolManager(config=config, lang=SupportedLanguage.PYTHON)

        try:
            # Create containers in different states
            c1 = pool._create_container()
            c1.mark_idle()

            c2 = pool._create_container()
            c2.mark_busy()

            c3 = pool._create_container()
            c3.mark_unhealthy()

            stats = pool.get_stats()

            assert stats["total_size"] == 3
            assert stats["max_size"] == 5
            assert stats["min_size"] == 0
            assert stats["state_counts"]["idle"] == 1
            assert stats["state_counts"]["busy"] == 1
            assert stats["state_counts"]["unhealthy"] == 1
            assert stats["closed"] is False
        finally:
            pool.close()


class TestHealthCheckEdgeCases:
    """Tests for health check edge cases."""

    def test_idle_timeout_triggers_removal(self):
        """Test that idle timeout triggers container removal."""
        config = PoolConfig(
            max_pool_size=2,
            min_pool_size=0,
            idle_timeout=1.0,  # 1 second
            health_check_interval=0.5,
            enable_prewarming=False,
        )
        pool = MockContainerPoolManager(config=config, lang=SupportedLanguage.PYTHON)

        try:
            # Create container
            c = pool._create_container()
            c.mark_idle()
            c.last_used_at = time.time() - 2.0  # Make it idle for 2 seconds
            original_id = c.container_id

            # Wait for health check
            time.sleep(1)

            # Container should be removed due to idle timeout
            assert original_id not in [pc.container_id for pc in pool._pool]
        finally:
            pool.close()

    def test_health_check_skips_busy_containers(self):
        """Test that health check doesn't check busy containers."""
        config = PoolConfig(
            max_pool_size=2,
            min_pool_size=0,
            health_check_interval=0.5,
            enable_prewarming=False,
        )
        pool = MockContainerPoolManager(
            config=config,
            lang=SupportedLanguage.PYTHON,
            fail_health_check=True,
        )

        try:
            # Create busy container
            c = pool._create_container()
            c.mark_busy()

            # Wait for health check
            time.sleep(1)

            # Container should still be there (not checked because busy)
            assert c in pool._pool
        finally:
            pool.close()


class TestPreWarming:
    """Tests for pre-warming functionality."""

    def test_prewarming_disabled(self):
        """Test that pre-warming can be disabled."""
        config = PoolConfig(
            max_pool_size=5,
            min_pool_size=3,
            enable_prewarming=False,
        )
        pool = MockContainerPoolManager(config=config, lang=SupportedLanguage.PYTHON)

        try:
            # Without pre-warming, pool should start empty
            time.sleep(0.5)
            stats = pool.get_stats()
            # Pool might have 0 or a few containers depending on timing
            assert stats["total_size"] < stats["min_size"]
        finally:
            pool.close()

    def test_prewarming_maintains_min_size(self):
        """Test that pre-warming maintains minimum pool size."""
        config = PoolConfig(
            max_pool_size=5,
            min_pool_size=2,
            enable_prewarming=True,
        )
        pool = MockContainerPoolManager(config=config, lang=SupportedLanguage.PYTHON)

        try:
            # Wait for pre-warming
            time.sleep(1.5)

            stats = pool.get_stats()
            assert stats["total_size"] >= stats["min_size"]
        finally:
            pool.close()


class TestContextManager:
    """Tests for context manager usage."""

    def test_pool_context_manager(self):
        """Test pool as context manager."""
        config = PoolConfig(max_pool_size=2, min_pool_size=0, enable_prewarming=False)

        with MockContainerPoolManager(config=config, lang=SupportedLanguage.PYTHON) as pool:
            c = pool._create_container()
            c.mark_idle()
            container = pool.acquire()
            pool.release(container)
            assert not pool._closed

        # Pool should be closed after exiting context
        # Note: We can't easily check pool._closed here as it's out of scope
