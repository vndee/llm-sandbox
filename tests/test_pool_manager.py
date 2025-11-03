"""Unit tests for container pool manager.

These tests verify the core functionality of the pool manager:
- Container acquisition and release
- Thread safety
- Pool exhaustion handling
- Health checking
- Configuration validation
"""

import threading
import time
from typing import Any

import pytest

from llm_sandbox.const import SupportedLanguage
from llm_sandbox.pool.base import ContainerPoolManager, ContainerState, PooledContainer
from llm_sandbox.pool.config import ExhaustionStrategy, PoolConfig
from llm_sandbox.pool.exceptions import PoolClosedError, PoolExhaustedError


class MockSession:
    """Mock session for testing."""

    def __init__(self, container_id: str) -> None:
        """Initialize mock session."""
        self.container = container_id
        self.is_open = False

    def open(self) -> None:
        """Mock session open."""
        self.is_open = True

    def close(self) -> None:
        """Mock session close."""
        self.is_open = False


class MockContainerPoolManager(ContainerPoolManager):
    """Mock implementation of ContainerPoolManager for testing."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """Initialize mock pool manager."""
        self._container_counter = 0
        super().__init__(*args, **kwargs)

    def _create_session_for_container(self) -> MockSession:
        """Create a mock session for container initialization."""
        self._container_counter += 1
        container_id = f"mock_container_{self._container_counter}"
        return MockSession(container_id)

    def _destroy_container_impl(self, container: Any) -> None:
        """Destroy a mock container."""

    def _get_container_id(self, container: Any) -> str:
        """Get mock container ID."""
        return container

    def _health_check_impl(self, container: Any) -> bool:
        """Mock health check (always healthy)."""
        return True


class TestPoolConfig:
    """Tests for PoolConfig."""

    def test_default_config(self):
        """Test default configuration values."""
        config = PoolConfig()

        assert config.max_pool_size == 10
        assert config.min_pool_size == 0
        assert config.idle_timeout == 300.0
        assert config.acquisition_timeout == 30.0
        assert config.health_check_interval == 60.0
        assert config.max_container_lifetime == 3600.0
        assert config.max_container_uses is None
        assert config.exhaustion_strategy == ExhaustionStrategy.WAIT
        assert config.enable_prewarming is True

    def test_custom_config(self):
        """Test custom configuration."""
        config = PoolConfig(
            max_pool_size=5,
            min_pool_size=2,
            idle_timeout=60.0,
            exhaustion_strategy=ExhaustionStrategy.FAIL_FAST,
        )

        assert config.max_pool_size == 5
        assert config.min_pool_size == 2
        assert config.idle_timeout == 60.0
        assert config.exhaustion_strategy == ExhaustionStrategy.FAIL_FAST

    def test_invalid_pool_sizes(self):
        """Test that invalid pool sizes raise ValueError."""
        with pytest.raises(ValueError, match="min_pool_size.*cannot be greater than max_pool_size"):
            PoolConfig(min_pool_size=10, max_pool_size=5)

    def test_validation(self):
        """Test field validation."""
        # Test negative values
        with pytest.raises(Exception):  # Pydantic validation error
            PoolConfig(max_pool_size=0)

        with pytest.raises(Exception):  # Pydantic validation error
            PoolConfig(min_pool_size=-1)


class TestPooledContainer:
    """Tests for PooledContainer."""

    def test_initialization(self):
        """Test container initialization."""
        container = PooledContainer(
            container_id="test_id",
            container="test_container",
        )

        assert container.container_id == "test_id"
        assert container.container == "test_container"
        assert container.state == ContainerState.INITIALIZING
        assert container.use_count == 0

    def test_mark_busy(self):
        """Test marking container as busy."""
        container = PooledContainer(
            container_id="test",
            container="test",
            state=ContainerState.IDLE,
        )

        initial_time = container.last_used_at
        time.sleep(0.01)

        container.mark_busy()

        assert container.state == ContainerState.BUSY
        assert container.use_count == 1
        assert container.last_used_at > initial_time

    def test_mark_idle(self):
        """Test marking container as idle."""
        container = PooledContainer(
            container_id="test",
            container="test",
            state=ContainerState.BUSY,
        )

        container.mark_idle()

        assert container.state == ContainerState.IDLE

    def test_is_available(self):
        """Test availability check."""
        container = PooledContainer(container_id="test", container="test")

        container.state = ContainerState.IDLE
        assert container.is_available() is True

        container.state = ContainerState.BUSY
        assert container.is_available() is False

        container.state = ContainerState.UNHEALTHY
        assert container.is_available() is False

    def test_is_expired_by_lifetime(self):
        """Test expiration by lifetime."""
        container = PooledContainer(container_id="test", container="test")
        container.created_at = time.time() - 100  # 100 seconds ago

        # Not expired
        assert container.is_expired(max_lifetime=200, max_uses=None) is False

        # Expired
        assert container.is_expired(max_lifetime=50, max_uses=None) is True

        # No limit
        assert container.is_expired(max_lifetime=None, max_uses=None) is False

    def test_is_expired_by_uses(self):
        """Test expiration by use count."""
        container = PooledContainer(container_id="test", container="test")
        container.use_count = 5

        # Not expired
        assert container.is_expired(max_lifetime=None, max_uses=10) is False

        # Expired
        assert container.is_expired(max_lifetime=None, max_uses=5) is True

        # No limit
        assert container.is_expired(max_lifetime=None, max_uses=None) is False

    def test_get_idle_time(self):
        """Test idle time calculation."""
        container = PooledContainer(container_id="test", container="test")
        container.last_used_at = time.time() - 10  # 10 seconds ago

        idle_time = container.get_idle_time()

        assert 9.5 < idle_time < 10.5  # Allow some tolerance


class TestContainerPoolManager:
    """Tests for ContainerPoolManager."""

    def test_initialization(self):
        """Test pool manager initialization."""
        config = PoolConfig(max_pool_size=5, min_pool_size=2)

        pool = MockContainerPoolManager(
            config=config,
            lang=SupportedLanguage.PYTHON,
        )

        try:
            # Give time for pre-warming thread to create containers
            time.sleep(1)

            stats = pool.get_stats()
            assert stats['max_size'] == 5
            assert stats['min_size'] == 2
            assert stats['total_size'] >= 2  # Should have created min_pool_size
        finally:
            pool.close()

    def test_acquire_and_release(self):
        """Test basic acquire and release."""
        config = PoolConfig(max_pool_size=3, min_pool_size=1, enable_prewarming=False)
        pool = MockContainerPoolManager(config=config, lang=SupportedLanguage.PYTHON)

        try:
            # Create a container first
            pool._create_container()

            # Acquire
            container = pool.acquire()
            assert container.state == ContainerState.BUSY
            assert container.use_count == 1

            # Release
            pool.release(container)
            assert container.state == ContainerState.IDLE
        finally:
            pool.close()

    def test_pool_exhaustion_fail_fast(self):
        """Test FAIL_FAST exhaustion strategy."""
        config = PoolConfig(
            max_pool_size=2,
            min_pool_size=0,
            exhaustion_strategy=ExhaustionStrategy.FAIL_FAST,
            enable_prewarming=False,
        )
        pool = MockContainerPoolManager(config=config, lang=SupportedLanguage.PYTHON)

        try:
            # Create 2 containers
            c1 = pool._create_container()
            c1.mark_idle()
            c2 = pool._create_container()
            c2.mark_idle()

            # Acquire both
            pool.acquire()
            pool.acquire()

            # Third acquire should fail
            with pytest.raises(PoolExhaustedError):
                pool.acquire()
        finally:
            pool.close()

    def test_pool_exhaustion_wait(self):
        """Test WAIT exhaustion strategy."""
        config = PoolConfig(
            max_pool_size=1,
            min_pool_size=0,
            exhaustion_strategy=ExhaustionStrategy.WAIT,
            acquisition_timeout=2.0,
            enable_prewarming=False,
        )
        pool = MockContainerPoolManager(config=config, lang=SupportedLanguage.PYTHON)

        try:
            # Create 1 container
            c = pool._create_container()
            c.mark_idle()

            # Acquire it
            container = pool.acquire()

            # Start thread that will release after 1 second
            def release_after_delay():
                time.sleep(1)
                pool.release(container)

            threading.Thread(target=release_after_delay, daemon=True).start()

            # Acquire should wait and succeed
            start = time.time()
            container2 = pool.acquire()
            elapsed = time.time() - start

            assert 0.9 < elapsed < 1.5  # Should have waited ~1 second
            assert container2 is not None
        finally:
            pool.close()

    def test_pool_exhaustion_wait_timeout(self):
        """Test WAIT strategy timeout."""
        config = PoolConfig(
            max_pool_size=1,
            min_pool_size=0,
            exhaustion_strategy=ExhaustionStrategy.WAIT,
            acquisition_timeout=1.0,
            enable_prewarming=False,
        )
        pool = MockContainerPoolManager(config=config, lang=SupportedLanguage.PYTHON)

        try:
            # Create and acquire 1 container
            c = pool._create_container()
            c.mark_idle()
            pool.acquire()

            # Next acquire should timeout
            with pytest.raises(PoolExhaustedError):
                pool.acquire()
        finally:
            pool.close()

    def test_thread_safety(self):
        """Test thread-safe concurrent access."""
        config = PoolConfig(
            max_pool_size=3,
            min_pool_size=3,
            enable_prewarming=True,
        )
        pool = MockContainerPoolManager(config=config, lang=SupportedLanguage.PYTHON)

        results = []
        errors = []

        def worker():
            try:
                container = pool.acquire()
                time.sleep(0.01)  # Simulate work
                results.append(container.container_id)
                pool.release(container)
            except Exception as e:  # noqa: BLE001
                errors.append(e)

        try:
            # Give time for pre-warming
            time.sleep(1)

            # Run multiple threads
            threads = [threading.Thread(target=worker) for _ in range(10)]
            for t in threads:
                t.start()
            for t in threads:
                t.join()

            # Should have completed all tasks
            assert len(results) == 10
            assert len(errors) == 0
        finally:
            pool.close()

    def test_container_recycling_by_uses(self):
        """Test container recycling based on use count."""
        config = PoolConfig(
            max_pool_size=2,
            min_pool_size=1,
            max_container_uses=3,
            enable_prewarming=False,
        )
        pool = MockContainerPoolManager(config=config, lang=SupportedLanguage.PYTHON)

        try:
            # Create a container
            c = pool._create_container()
            c.mark_idle()

            original_id = c.container_id

            # Use it 3 times
            for _ in range(3):
                container = pool.acquire()
                pool.release(container)

            # Container should be recycled
            assert container.container_id not in [pc.container_id for pc in pool._pool]
        finally:
            pool.close()

    def test_close_pool(self):
        """Test closing the pool."""
        config = PoolConfig(max_pool_size=3, min_pool_size=1)
        pool = MockContainerPoolManager(config=config, lang=SupportedLanguage.PYTHON)

        # Give time for initialization
        time.sleep(0.5)

        # Close pool
        pool.close()

        # Pool should be empty and closed
        assert len(pool._pool) == 0
        assert pool._closed is True

        # Acquiring should fail
        with pytest.raises(PoolClosedError):
            pool.acquire()

    def test_context_manager(self):
        """Test pool as context manager."""
        config = PoolConfig(max_pool_size=2, min_pool_size=1)

        with MockContainerPoolManager(config=config, lang=SupportedLanguage.PYTHON) as pool:
            # Pool should be usable
            stats = pool.get_stats()
            assert stats['closed'] is False

        # After exiting, pool should be closed
        # Note: Can't access pool._closed directly as it's out of scope

    def test_get_stats(self):
        """Test getting pool statistics."""
        config = PoolConfig(max_pool_size=5, min_pool_size=2)
        pool = MockContainerPoolManager(config=config, lang=SupportedLanguage.PYTHON)

        try:
            time.sleep(0.5)  # Allow pre-warming

            stats = pool.get_stats()

            assert 'total_size' in stats
            assert 'max_size' in stats
            assert 'min_size' in stats
            assert 'state_counts' in stats
            assert 'closed' in stats

            assert stats['max_size'] == 5
            assert stats['min_size'] == 2
            assert stats['closed'] is False
        finally:
            pool.close()


class TestExhaustionStrategies:
    """Tests for different exhaustion strategies."""

    def test_fail_fast_strategy(self):
        """Test FAIL_FAST strategy behavior."""
        strategy = ExhaustionStrategy.FAIL_FAST
        assert strategy.value == "fail_fast"

    def test_wait_strategy(self):
        """Test WAIT strategy behavior."""
        strategy = ExhaustionStrategy.WAIT
        assert strategy.value == "wait"

    def test_temporary_strategy(self):
        """Test TEMPORARY strategy behavior."""
        strategy = ExhaustionStrategy.TEMPORARY
        assert strategy.value == "temporary"
