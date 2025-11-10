"""Base classes for container pool management."""

import logging
import threading
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from types import TracebackType
from typing import Any

from llm_sandbox.const import DefaultImage, SupportedLanguage
from llm_sandbox.pool.config import ExhaustionStrategy, PoolConfig
from llm_sandbox.pool.exceptions import PoolClosedError, PoolExhaustedError


def resolve_default_image(lang: SupportedLanguage, image: str | None, dockerfile: str | None = None) -> str | None:
    """Resolve default image for a language if not explicitly provided.

    Args:
        lang: Programming language
        image: Explicitly provided image (None to use default)
        dockerfile: Dockerfile path (if provided, no default image needed)

    Returns:
        Image name or None if dockerfile is provided

    """
    if image:
        return image
    if dockerfile:
        return None
    return DefaultImage.__dict__[lang.upper()]


class ContainerState(str, Enum):
    """State of a pooled container.

    State transitions:
    INITIALIZING -> IDLE ----> BUSY --> IDLE
                     ↓          ↓
                 UNHEALTHY -> REMOVING
    """

    INITIALIZING = "initializing"
    IDLE = "idle"
    BUSY = "busy"
    UNHEALTHY = "unhealthy"
    REMOVING = "removing"


@dataclass
class PooledContainer:
    """Wrapper for a container in the pool with state tracking.

    This class tracks the state and usage statistics of a container
    in the pool to enable lifecycle management and health checking.
    """

    container_id: str
    """Unique identifier for the container/pod"""

    container: Any
    """The actual container object (backend-specific)"""

    state: ContainerState = ContainerState.INITIALIZING
    """Current state of the container"""

    created_at: float = field(default_factory=time.time)
    """Timestamp when the container was created"""

    last_used_at: float = field(default_factory=time.time)
    """Timestamp when the container was last used"""

    use_count: int = 0
    """Number of times this container has been used"""

    def mark_busy(self) -> None:
        """Mark container as busy (in use)."""
        self.state = ContainerState.BUSY
        self.last_used_at = time.time()
        self.use_count += 1

    def mark_idle(self) -> None:
        """Mark container as idle (available for use)."""
        self.state = ContainerState.IDLE
        self.last_used_at = time.time()

    def mark_unhealthy(self) -> None:
        """Mark container as unhealthy (needs removal)."""
        self.state = ContainerState.UNHEALTHY

    def mark_removing(self) -> None:
        """Mark container as being removed."""
        self.state = ContainerState.REMOVING

    def is_available(self) -> bool:
        """Check if container is available for use."""
        return self.state == ContainerState.IDLE

    def is_expired(self, max_lifetime: float | None, max_uses: int | None) -> bool:
        """Check if container has exceeded its lifetime or use count.

        Args:
            max_lifetime: Maximum lifetime in seconds (None for no limit)
            max_uses: Maximum number of uses (None for no limit)

        Returns:
            True if container should be recycled, False otherwise

        """
        if max_lifetime is not None:
            age = time.time() - self.created_at
            if age > max_lifetime:
                return True

        return max_uses is not None and self.use_count >= max_uses

    def get_idle_time(self) -> float:
        """Get time since container was last used (in seconds)."""
        return time.time() - self.last_used_at


class ContainerPoolManager(ABC):
    """Abstract base class for container pool management.

    This class provides the core functionality for managing a pool of
    containers with thread-safe acquisition/release, health checking,
    and automatic recycling.

    Thread Safety:
        All public methods (acquire, release, close, get_stats) are thread-safe
        and can be called concurrently from multiple threads. The pool uses a
        reentrant lock (RLock) and condition variable for synchronization.

        Background threads for health checking and pre-warming run independently
        and coordinate with the main pool operations through the same lock.

    Subclasses must implement backend-specific operations:
    - _create_session_for_container()
    - _destroy_container_impl()
    - _get_container_id()
    - _health_check_impl()

    Example:
        ```python
        from llm_sandbox.pool import create_pool_manager, PoolConfig
        from llm_sandbox.const import SandboxBackend, SupportedLanguage

        # Create pool manager
        pool = create_pool_manager(
            backend=SandboxBackend.DOCKER,
            config=PoolConfig(max_pool_size=10, min_pool_size=3),
            lang=SupportedLanguage.PYTHON,
        )

        # Use pool in multiple threads
        with pool:
            container = pool.acquire()
            try:
                # Use container...
                pass
            finally:
                pool.release(container)
        ```
    """

    def __init__(
        self,
        client: Any,
        config: PoolConfig,
        lang: SupportedLanguage,
        image: str | None = None,
        **session_kwargs: Any,
    ) -> None:
        """Initialize the container pool manager.

        Args:
            client: Client to use for container creation
            config: Pool configuration
            lang: Programming language for containers
            image: Container image to use
            **session_kwargs: Additional keyword arguments for session creation

        """
        self.client = client
        self.config = config
        self.lang = lang
        self.image = image
        self.session_kwargs = session_kwargs

        # Pool state
        self._pool: list[PooledContainer] = []
        self._lock = threading.RLock()  # Reentrant lock for nested locking
        self._condition = threading.Condition(self._lock)
        self._closed = False

        # Background threads
        self._health_check_thread: threading.Thread | None = None
        self._prewarming_thread: threading.Thread | None = None
        self._shutdown_event = threading.Event()

        # Logging
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")

        # Initialize pool
        self._initialize_pool()

    def _initialize_pool(self) -> None:
        """Initialize the pool with pre-warmed containers."""
        self.logger.info(
            "Initializing pool with min_size=%d, max_size=%d",
            self.config.min_pool_size,
            self.config.max_pool_size,
        )

        # Start background threads
        if self.config.health_check_interval > 0:
            self._health_check_thread = threading.Thread(
                target=self._health_check_loop,
                daemon=True,
                name="pool-health-check",
            )
            self._health_check_thread.start()

        if self.config.enable_prewarming and self.config.min_pool_size > 0:
            self._prewarming_thread = threading.Thread(
                target=self._prewarming_loop,
                daemon=True,
                name="pool-prewarming",
            )
            self._prewarming_thread.start()

    def acquire(self) -> PooledContainer:
        """Acquire a container from the pool.

        This method attempts to get an available container from the pool.
        If no containers are available, behavior depends on the exhaustion strategy.

        Returns:
            A pooled container ready for use

        Raises:
            PoolExhaustedError: If pool is exhausted and strategy is FAIL_FAST or WAIT timeout exceeded
            PoolClosedError: If pool has been closed

        """
        with self._condition:
            if self._closed:
                raise PoolClosedError

            # Try to get an idle container
            container = self._get_idle_container()
            if container:
                container.mark_busy()
                self.logger.debug("Acquired container %s from pool", container.container_id)
                return container

            # No idle containers available - handle based on strategy
            return self._handle_exhaustion()

    def release(self, container: PooledContainer) -> None:
        """Release a container back to the pool.

        Args:
            container: The container to release

        """
        with self._condition:
            if self._closed:
                # Pool is closed, destroy the container
                self._destroy_container(container)
                return

            # Check if container should be recycled
            if container.is_expired(
                self.config.max_container_lifetime,
                self.config.max_container_uses,
            ):
                self.logger.info(
                    "Container %s expired (age=%.1fs, uses=%d), recycling",
                    container.container_id,
                    time.time() - container.created_at,
                    container.use_count,
                )
                self._destroy_container(container)
                self._ensure_min_pool_size()
            else:
                # Return to pool
                container.mark_idle()
                self.logger.debug("Released container %s to pool", container.container_id)
                self._condition.notify()  # Notify waiting threads

    def close(self) -> None:
        """Close the pool and cleanup all resources.

        This method:
        1. Stops accepting new requests
        2. Waits for busy containers to be released
        3. Destroys all containers
        4. Stops background threads
        """
        with self._condition:
            if self._closed:
                return

            self.logger.info("Closing container pool")
            self._closed = True

            # Signal shutdown to background threads
            self._shutdown_event.set()

            # Destroy all containers
            for container in self._pool[:]:  # Copy list to avoid modification during iteration
                self._destroy_container(container)

            self._pool.clear()

        # Wait for background threads to finish
        if self._health_check_thread and self._health_check_thread.is_alive():
            self._health_check_thread.join(timeout=5)

        if self._prewarming_thread and self._prewarming_thread.is_alive():
            self._prewarming_thread.join(timeout=5)

        self.logger.info("Container pool closed")

    def get_stats(self) -> dict[str, Any]:
        """Get pool statistics.

        Returns:
            Dictionary with pool statistics including size, state counts, etc.

        """
        with self._lock:
            state_counts = dict.fromkeys(ContainerState, 0)
            for container in self._pool:
                state_counts[container.state] += 1

            return {
                "total_size": len(self._pool),
                "max_size": self.config.max_pool_size,
                "min_size": self.config.min_pool_size,
                "state_counts": {state.value: count for state, count in state_counts.items()},
                "closed": self._closed,
            }

    def _get_idle_container(self) -> PooledContainer | None:
        """Get an idle container from the pool (internal, must hold lock).

        Returns:
            An idle container or None if none available

        """
        for container in self._pool:
            if container.is_available():
                return container

        return None

    def _handle_exhaustion(self) -> PooledContainer:
        """Handle pool exhaustion based on configured strategy (internal, must hold lock).

        Returns:
            A container to use

        Raises:
            PoolExhaustedError: If unable to provide a container

        """
        strategy = self.config.exhaustion_strategy

        if strategy == ExhaustionStrategy.FAIL_FAST:
            raise PoolExhaustedError(self.config.max_pool_size)

        if strategy == ExhaustionStrategy.WAIT:
            return self._wait_for_container()

        if strategy == ExhaustionStrategy.TEMPORARY:
            return self._create_temporary_container()

        raise PoolExhaustedError(self.config.max_pool_size)

    def _wait_for_container(self) -> PooledContainer:
        """Wait for a container to become available (internal, must hold lock).

        Returns:
            An available container

        Raises:
            PoolExhaustedError: If timeout is exceeded

        """
        timeout = self.config.acquisition_timeout
        self.logger.debug("Waiting for container (timeout=%s)", timeout)

        # Wait for notification
        if self._condition.wait(timeout=timeout):
            # Got notified, try to get container again
            container = self._get_idle_container()
            if container:
                container.mark_busy()
                return container

        # Timeout or spurious wakeup without available container
        raise PoolExhaustedError(self.config.max_pool_size, timeout)

    def _create_temporary_container(self) -> PooledContainer:
        """Create a temporary container outside the pool (internal, must hold lock).

        Returns:
            A temporary container

        """
        self.logger.info("Creating temporary container outside pool")
        container = self._create_container()
        container.mark_busy()
        # Note: Temporary containers are not added to the pool
        return container

    def _ensure_min_pool_size(self) -> None:
        """Ensure pool has at least min_pool_size containers (internal, must hold lock)."""
        current_size = len(self._pool)
        if current_size < self.config.min_pool_size:
            needed = self.config.min_pool_size - current_size
            self.logger.debug("Creating %d containers to maintain min_pool_size", needed)
            for _ in range(needed):
                if len(self._pool) < self.config.max_pool_size:
                    container = self._create_container()
                    container.mark_idle()

    def _create_container(self) -> PooledContainer:
        """Create a new container with full environment setup (internal, must hold lock).

        This creates a container using the standard session initialization logic,
        which handles image preparation, container creation, and environment setup
        (venv creation, pip upgrades, go mod init, library installation, etc.) automatically.

        Returns:
            PooledContainer: A new pooled container ready for use, added to the pool

        Raises:
            Exception: If container creation or initialization fails

        """
            A new pooled container ready for use

        """
        # Use backend-specific session creation logic
        session = self._create_session_for_container()

        try:
            # Open session - this creates container and sets up environment
            session.open()

            # Extract the initialized container
            container_obj = session.container
            container_id = self._get_container_id(container_obj)

            # Create pooled container wrapper
            container = PooledContainer(
                container_id=container_id,
                container=container_obj,
                state=ContainerState.IDLE,  # Ready to use
            )

            # Detach container from session so it won't be destroyed on session close
            session.container = None

        except Exception:
            self.logger.exception("Failed to create container")
            raise
        else:
            self._pool.append(container)
            self.logger.info("Created and initialized container %s", container.container_id)
            # Wake up any waiters since a new idle container is now available
            self._condition.notify()

            return container
        finally:
            # Close session (won't destroy container since we detached it)
            try:
                session.close()
            except Exception:
                self.logger.exception("Error closing temporary session")

    def _destroy_container(self, container: PooledContainer) -> None:
        """Destroy a container (internal, must hold lock).

        Args:
            container: Container to destroy

        """
        container.mark_removing()
        try:
            self._destroy_container_impl(container.container)
            self.logger.info("Destroyed container %s", container.container_id)
        except Exception:
            self.logger.exception("Failed to destroy container %s", container.container_id)
        finally:
            if container in self._pool:
                self._pool.remove(container)

    def _health_check_loop(self) -> None:
        """Background thread that performs periodic health checks."""
        self.logger.info("Starting health check loop (interval=%ds)", self.config.health_check_interval)

        while not self._shutdown_event.is_set():
            try:
                self._perform_health_checks()
            except Exception:
                self.logger.exception("Error in health check loop")

            self._shutdown_event.wait(timeout=self.config.health_check_interval)

        self.logger.info("Health check loop stopped")

    def _perform_health_checks(self) -> None:
        """Perform health checks on idle containers."""
        # First, collect containers to check (with short lock)
        containers_to_check = []
        with self._lock:
            if self._closed:
                return
            
            # Copy idle containers for checking
            containers_to_check = [
                container for container in self._pool 
                if container.state == ContainerState.IDLE
            ]
        
        # Perform health checks without holding lock
        unhealthy = []
        for container in containers_to_check:
            # Check idle timeout
            if self.config.idle_timeout and container.get_idle_time() > self.config.idle_timeout:
                self.logger.info(
                    "Container %s exceeded idle timeout (%.1fs), recycling",
                    container.container_id,
                    container.get_idle_time(),
                )
                unhealthy.append(container)
                continue

            # Perform health check
            if not self._health_check_impl(container.container):
                self.logger.warning("Container %s failed health check", container.container_id)
                unhealthy.append(container)
        
        # Remove unhealthy containers (with lock)
        if unhealthy:
            with self._lock:
                for container in unhealthy:
                    container.mark_unhealthy()
                    self._destroy_container(container)
                
                # Ensure minimum pool size
                self._ensure_min_pool_size()

    def _prewarming_loop(self) -> None:
        """Background thread that maintains minimum pool size."""
        self.logger.info("Starting pre-warming loop (min_size=%d)", self.config.min_pool_size)

        # Initial warm-up
        with self._lock:
            self._ensure_min_pool_size()

        # Periodic check to maintain minimum
        while not self._shutdown_event.is_set():
            try:
                with self._lock:
                    if not self._closed:
                        self._ensure_min_pool_size()
            except Exception:
                self.logger.exception("Error in pre-warming loop")

            self._shutdown_event.wait(timeout=10)  # Check every 10 seconds

        self.logger.info("Pre-warming loop stopped")

    @abstractmethod
    def _create_session_for_container(self) -> Any:
        """Create a session for initializing a new container (backend-specific).

        This should return a session instance that, when opened, will create
        and set up a container with all necessary environment initialization.

        Returns:
            Session instance (not yet opened)

        """
        raise NotImplementedError

    @abstractmethod
    def _destroy_container_impl(self, container: Any) -> None:
        """Destroy a container (backend-specific implementation).

        Args:
            container: Container to destroy

        """
        raise NotImplementedError

    @abstractmethod
    def _get_container_id(self, container: Any) -> str:
        """Get container ID (backend-specific implementation).

        Args:
            container: Container object

        Returns:
            Container ID string

        """
        raise NotImplementedError

    @abstractmethod
    def _health_check_impl(self, container: Any) -> bool:
        """Perform health check on container (backend-specific implementation).

        Args:
            container: Container to check

        Returns:
            True if healthy, False otherwise

        """
        raise NotImplementedError

    def __enter__(self) -> "ContainerPoolManager":
        """Enter context manager."""
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        """Exit context manager and cleanup."""
        self.close()
