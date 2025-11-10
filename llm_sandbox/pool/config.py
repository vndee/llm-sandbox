"""Configuration for container pool management."""

from enum import Enum
from typing import Any

from pydantic import BaseModel, Field, field_validator


class ExhaustionStrategy(str, Enum):
    """Strategy for handling pool exhaustion when all containers are busy.

    FAIL_FAST: Immediately raise PoolExhaustedError
    WAIT: Wait for a container to become available (with optional timeout)
    TEMPORARY: Create a temporary container outside the pool
    """

    FAIL_FAST = "fail_fast"
    WAIT = "wait"
    TEMPORARY = "temporary"


class PoolConfig(BaseModel):
    """Configuration for container pool management.

    This configuration controls the behavior of the container pool,
    including size limits, timeouts, health checks, and exhaustion handling.
    """

    # Pool size configuration
    max_pool_size: int = Field(
        default=10,
        ge=1,
        description="Maximum number of containers in the pool",
    )

    min_pool_size: int = Field(
        default=0,
        ge=0,
        description="Minimum number of pre-warmed containers to maintain in the pool",
    )

    # Timeout configuration
    idle_timeout: float | None = Field(
        default=300.0,
        gt=0,
        description="Seconds before an idle container is recycled (None for no timeout)",
    )

    acquisition_timeout: float | None = Field(
        default=30.0,
        gt=0,
        description="Seconds to wait for a container when pool is exhausted (None for no timeout, \
                        only applies to WAIT strategy)",
    )

    # Health and lifecycle configuration
    health_check_interval: float = Field(
        default=60.0,
        gt=0,
        description="Seconds between health checks for idle containers",
    )

    max_container_lifetime: float | None = Field(
        default=3600.0,
        gt=0,
        description="Maximum lifetime of a container in seconds before recycling (None for no limit)",
    )

    max_container_uses: int | None = Field(
        default=None,
        ge=1,
        description="Maximum number of times a container can be used before recycling (None for no limit)",
    )

    # Exhaustion handling
    exhaustion_strategy: ExhaustionStrategy = Field(
        default=ExhaustionStrategy.WAIT,
        description="Strategy to use when all containers are busy",
    )

    # Pre-warming configuration
    enable_prewarming: bool = Field(
        default=True,
        description="Whether to automatically create min_pool_size containers on startup",
    )

    @field_validator("min_pool_size")
    @classmethod
    def validate_min_pool_size(cls, v: int) -> int:
        """Validate that min_pool_size <= max_pool_size."""
        # Note: max_pool_size might not be set yet during validation
        # We'll do a final check in model_validator
        return v

    def model_post_init(self, __context: Any, /) -> None:
        """Validate configuration after all fields are set.

        Args:
            __context: Validation context

        """
        if self.min_pool_size > self.max_pool_size:
            msg = f"min_pool_size ({self.min_pool_size}) cannot be greater than max_pool_size ({self.max_pool_size})"
            raise ValueError(msg)
