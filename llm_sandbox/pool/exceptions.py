"""Exceptions for container pool management."""


class PoolExhaustedError(Exception):
    """Raised when all containers in the pool are busy and no strategy can provide one.

    This exception is raised when:
    - All containers in the pool are currently in use
    - The exhaustion strategy is FAIL_FAST
    - The exhaustion strategy is WAIT and timeout is exceeded
    """

    def __init__(self, pool_size: int, timeout: float | None = None) -> None:
        """Initialize PoolExhaustedError.

        Args:
            pool_size: Maximum size of the pool
            timeout: Timeout value if waiting was attempted

        """
        if timeout is not None:
            message = f"All {pool_size} containers in pool are busy and timeout of {timeout}s exceeded"
        else:
            message = f"All {pool_size} containers in pool are busy"
        super().__init__(message)
        self.pool_size = pool_size
        self.timeout = timeout


class PoolClosedError(Exception):
    """Raised when attempting to use a closed pool.

    This exception is raised when operations are attempted on a pool
    that has been closed/shut down.
    """

    def __init__(self) -> None:
        """Initialize PoolClosedError."""
        super().__init__("Container pool has been closed")


class PoolHealthCheckError(Exception):
    """Raised when a container health check fails.

    This exception is raised when a container in the pool fails
    its health check and needs to be removed/replaced.
    """

    def __init__(self, container_id: str, reason: str) -> None:
        """Initialize PoolHealthCheckError.

        Args:
            container_id: ID of the unhealthy container
            reason: Reason for health check failure

        """
        super().__init__(f"Health check failed for container {container_id}: {reason}")
        self.container_id = container_id
        self.reason = reason
