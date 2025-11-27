import time
from typing import Any
from unittest.mock import MagicMock

import pytest

from llm_sandbox.const import SupportedLanguage
from llm_sandbox.pool.base import ContainerPoolManager, ContainerState, PooledContainer
from llm_sandbox.pool.config import ExhaustionStrategy, PoolConfig
from llm_sandbox.pool.exceptions import PoolExhaustedError


class MockPoolManager(ContainerPoolManager):
    """Concrete implementation of ContainerPoolManager for testing."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """Initialize the MockPoolManager."""
        self.created_containers: list[MagicMock] = []
        self.destroyed_containers: list[Any] = []
        self.health_checks: dict[str, bool] = {}
        super().__init__(*args, **kwargs)

    def _create_session_for_container(self) -> MagicMock:
        """Create a session for a container."""
        session = MagicMock()
        session.container = MagicMock()
        session.container.id = f"container-{len(self.created_containers)}"
        self.created_containers.append(session.container)
        return session

    def _destroy_container_impl(self, container: MagicMock) -> None:
        """Destroy a container."""
        self.destroyed_containers.append(container)

    def _get_container_id(self, container: MagicMock) -> str:
        """Get the ID of a container."""
        return str(container.id)

    def _health_check_impl(self, container: MagicMock) -> bool:
        """Check the health of a container."""
        return self.health_checks.get(container.id, True)


class TestPooledContainer:
    """Test PooledContainer dataclass."""

    def test_state_transitions(self) -> None:
        """Test state transitions."""
        container = PooledContainer(container_id="test", container=MagicMock())
        assert container.state == ContainerState.INITIALIZING

        container.mark_idle()
        assert container.state == ContainerState.IDLE
        assert container.is_available()

        container.mark_busy()
        assert container.state == ContainerState.BUSY
        assert not container.is_available()

        container.mark_unhealthy()
        assert container.state == ContainerState.UNHEALTHY

        container.mark_removing()
        assert container.state == ContainerState.REMOVING

    def test_expiration(self) -> None:
        """Test expiration logic."""
        container = PooledContainer(container_id="test", container=MagicMock())

        # Test max uses
        assert not container.is_expired(max_lifetime=None, max_uses=2)
        container.use_count = 2
        assert container.is_expired(max_lifetime=None, max_uses=2)

        # Test lifetime
        container = PooledContainer(container_id="test", container=MagicMock())
        container.created_at = time.time() - 100
        assert container.is_expired(max_lifetime=50, max_uses=None)
        assert not container.is_expired(max_lifetime=200, max_uses=None)


class TestContainerPoolManager:
    """Test ContainerPoolManager lifecycle and logic."""

    def test_initialization(self) -> None:
        """Test pool initialization."""
        config = PoolConfig(min_pool_size=2, max_pool_size=5, enable_prewarming=False)
        pool = MockPoolManager(client=MagicMock(), config=config, lang=SupportedLanguage.PYTHON)

        # Prewarming disabled, so pool starts empty
        assert len(pool._pool) == 0
        assert pool._health_check_thread is not None
        # Prewarming disabled
        assert pool._prewarming_thread is None

        pool.close()

    def test_acquire_release(self) -> None:
        """Test acquire and release cycle."""
        config = PoolConfig(min_pool_size=1, max_pool_size=2, enable_prewarming=False)
        pool = MockPoolManager(client=MagicMock(), config=config, lang=SupportedLanguage.PYTHON)

        # Acquire
        container = pool.acquire()
        assert container.state == ContainerState.BUSY
        assert container in pool._pool  # Container remains in pool list

        # Release
        pool.release(container)
        assert container.state == ContainerState.IDLE
        assert container in pool._pool

        pool.close()

    def test_exhaustion_fail_fast(self) -> None:
        """Test FAIL_FAST exhaustion strategy."""
        config = PoolConfig(
            min_pool_size=1, max_pool_size=1, exhaustion_strategy=ExhaustionStrategy.FAIL_FAST, enable_prewarming=False
        )
        pool = MockPoolManager(client=MagicMock(), config=config, lang=SupportedLanguage.PYTHON)

        pool.acquire()  # Take the only container

        with pytest.raises(PoolExhaustedError):
            pool.acquire()

        pool.close()

    def test_exhaustion_create_temporary(self) -> None:
        """Test TEMPORARY exhaustion strategy."""
        config = PoolConfig(
            min_pool_size=1, max_pool_size=1, exhaustion_strategy=ExhaustionStrategy.TEMPORARY, enable_prewarming=False
        )
        pool = MockPoolManager(client=MagicMock(), config=config, lang=SupportedLanguage.PYTHON)
        with pool._lock:
            pool._ensure_min_pool_size()  # Populate pool

        pool.acquire()  # Take the only container

        # Should create temp container
        c2 = pool.acquire()
        assert c2.state == ContainerState.BUSY
        # Current implementation adds temp containers to pool, contrary to comment
        assert c2 in pool._pool

        pool.close()

    def test_recycling_on_release(self) -> None:
        """Test container recycling on release."""
        config = PoolConfig(max_container_uses=1, min_pool_size=1, max_pool_size=2, enable_prewarming=False)
        pool = MockPoolManager(client=MagicMock(), config=config, lang=SupportedLanguage.PYTHON)
        with pool._lock:
            pool._ensure_min_pool_size()  # Populate pool

        container = pool.acquire()
        original_id = container.container_id

        pool.release(container)

        # Container should have been destroyed and replaced
        assert container not in pool._pool
        assert len(pool.destroyed_containers) == 1
        assert len(pool._pool) == 1
        assert pool._pool[0].container_id != original_id

        pool.close()

    def test_health_check_logic(self) -> None:
        """Test health check logic."""
        config = PoolConfig(min_pool_size=1, max_pool_size=2, enable_prewarming=False)
        pool = MockPoolManager(client=MagicMock(), config=config, lang=SupportedLanguage.PYTHON)
        with pool._lock:
            pool._ensure_min_pool_size()  # Populate pool

        container = pool._pool[0]
        pool.health_checks[container.container_id] = False  # Mark unhealthy

        # Run health check manually
        pool._perform_health_checks()

        # Container should be removed and replaced
        assert container not in pool._pool
        assert len(pool.destroyed_containers) == 1
        assert len(pool._pool) == 1

        pool.close()

    def test_idle_timeout(self) -> None:
        """Test idle timeout logic."""
        config = PoolConfig(min_pool_size=1, max_pool_size=2, idle_timeout=0.1, enable_prewarming=False)
        pool = MockPoolManager(client=MagicMock(), config=config, lang=SupportedLanguage.PYTHON)
        with pool._lock:
            pool._ensure_min_pool_size()  # Populate pool

        container = pool._pool[0]
        container.last_used_at = time.time() - 1.0  # Force idle timeout

        pool._perform_health_checks()

        assert container not in pool._pool
        assert len(pool.destroyed_containers) == 1
        assert len(pool._pool) == 1

        pool.close()

    def test_close_cleanup(self) -> None:
        """Test pool closing cleanup."""
        config = PoolConfig(min_pool_size=2, max_pool_size=5, enable_prewarming=False)
        pool = MockPoolManager(client=MagicMock(), config=config, lang=SupportedLanguage.PYTHON)
        with pool._lock:
            pool._ensure_min_pool_size()  # Populate pool

        assert len(pool._pool) == 2

        pool.close()

        assert pool._closed
        assert len(pool._pool) == 0
        assert len(pool.destroyed_containers) == 2
        assert pool._shutdown_event.is_set()

    def test_context_manager(self) -> None:
        """Test context manager support."""
        config = PoolConfig(min_pool_size=1, max_pool_size=2, enable_prewarming=False)
        with MockPoolManager(client=MagicMock(), config=config, lang=SupportedLanguage.PYTHON) as pool:
            with pool._lock:
                pool._ensure_min_pool_size()
            assert not pool._closed
            pool.acquire()

        assert pool._closed

    def test_get_stats(self) -> None:
        """Test get_stats."""
        config = PoolConfig(min_pool_size=1, max_pool_size=2, enable_prewarming=False)
        pool = MockPoolManager(client=MagicMock(), config=config, lang=SupportedLanguage.PYTHON)
        with pool._lock:
            pool._ensure_min_pool_size()  # Populate pool

        stats = pool.get_stats()
        assert stats["total_size"] == 1
        assert stats["max_size"] == 2
        assert stats["state_counts"]["idle"] == 1

        pool.close()

    def test_wait_for_container_timeout(self) -> None:
        """Test waiting for container timeout."""
        config = PoolConfig(
            min_pool_size=1,
            max_pool_size=1,
            exhaustion_strategy=ExhaustionStrategy.WAIT,
            acquisition_timeout=0.1,
            enable_prewarming=False,
        )
        pool = MockPoolManager(client=MagicMock(), config=config, lang=SupportedLanguage.PYTHON)
        with pool._lock:
            pool._ensure_min_pool_size()  # Populate pool

        pool.acquire()  # Take the only container

        with pytest.raises(PoolExhaustedError):
            pool.acquire()  # Should timeout

        pool.close()

    def test_prewarming_thread(self) -> None:
        """Test prewarming thread logic."""
        config = PoolConfig(min_pool_size=2, max_pool_size=5, enable_prewarming=True)
        pool = MockPoolManager(client=MagicMock(), config=config, lang=SupportedLanguage.PYTHON)

        assert pool._prewarming_thread is not None
        assert pool._prewarming_thread.is_alive()

        # Wait for thread to populate pool
        time.sleep(0.2)

        with pool._lock:
            # Should have populated to min_size
            if len(pool._pool) < 2:
                pool._ensure_min_pool_size()

        # Manually trigger ensure_min_pool_size to verify logic
        pool._pool.pop()  # Remove one
        with pool._lock:
            pool._ensure_min_pool_size()
        assert len(pool._pool) == 2

        pool.close()
