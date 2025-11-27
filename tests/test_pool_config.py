"""Tests for pool configuration module."""

import pytest
from pydantic import ValidationError

from llm_sandbox.pool.config import ExhaustionStrategy, PoolConfig


class TestExhaustionStrategy:
    """Test ExhaustionStrategy enum."""

    def test_enum_values(self) -> None:
        """Test all enum values are available."""
        assert ExhaustionStrategy.FAIL_FAST == "fail_fast"
        assert ExhaustionStrategy.WAIT == "wait"
        assert ExhaustionStrategy.TEMPORARY == "temporary"

    def test_enum_membership(self) -> None:
        """Test enum membership."""
        assert "fail_fast" in [e.value for e in ExhaustionStrategy]
        assert "wait" in [e.value for e in ExhaustionStrategy]
        assert "temporary" in [e.value for e in ExhaustionStrategy]

    def test_enum_iteration(self) -> None:
        """Test iterating over enum."""
        strategies = list(ExhaustionStrategy)
        assert len(strategies) == 3
        assert ExhaustionStrategy.FAIL_FAST in strategies
        assert ExhaustionStrategy.WAIT in strategies
        assert ExhaustionStrategy.TEMPORARY in strategies


class TestPoolConfig:
    """Test PoolConfig configuration class."""

    def test_default_values(self) -> None:
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

    def test_custom_pool_sizes(self) -> None:
        """Test custom pool size configuration."""
        config = PoolConfig(max_pool_size=20, min_pool_size=5)
        assert config.max_pool_size == 20
        assert config.min_pool_size == 5

    def test_custom_timeouts(self) -> None:
        """Test custom timeout configuration."""
        config = PoolConfig(
            idle_timeout=600.0,
            acquisition_timeout=60.0,
            health_check_interval=30.0,
        )
        assert config.idle_timeout == 600.0
        assert config.acquisition_timeout == 60.0
        assert config.health_check_interval == 30.0

    def test_custom_lifecycle_settings(self) -> None:
        """Test custom lifecycle settings."""
        config = PoolConfig(max_container_lifetime=1800.0, max_container_uses=100)
        assert config.max_container_lifetime == 1800.0
        assert config.max_container_uses == 100

    def test_none_timeouts(self) -> None:
        """Test None values for optional timeout fields."""
        config = PoolConfig(
            idle_timeout=None,
            acquisition_timeout=None,
            max_container_lifetime=None,
        )
        assert config.idle_timeout is None
        assert config.acquisition_timeout is None
        assert config.max_container_lifetime is None

    def test_exhaustion_strategy_fail_fast(self) -> None:
        """Test FAIL_FAST exhaustion strategy."""
        config = PoolConfig(exhaustion_strategy=ExhaustionStrategy.FAIL_FAST)
        assert config.exhaustion_strategy == ExhaustionStrategy.FAIL_FAST

    def test_exhaustion_strategy_temporary(self) -> None:
        """Test TEMPORARY exhaustion strategy."""
        config = PoolConfig(exhaustion_strategy=ExhaustionStrategy.TEMPORARY)
        assert config.exhaustion_strategy == ExhaustionStrategy.TEMPORARY

    def test_disable_prewarming(self) -> None:
        """Test disabling pre-warming."""
        config = PoolConfig(enable_prewarming=False)
        assert config.enable_prewarming is False

    def test_min_pool_size_greater_than_max_raises_error(self) -> None:
        """Test validation error when min_pool_size > max_pool_size."""
        with pytest.raises(ValueError, match="min_pool_size .* cannot be greater than max_pool_size"):
            PoolConfig(min_pool_size=10, max_pool_size=5)

    def test_min_pool_size_equal_to_max_is_valid(self) -> None:
        """Test that min_pool_size == max_pool_size is valid."""
        config = PoolConfig(min_pool_size=5, max_pool_size=5)
        assert config.min_pool_size == 5
        assert config.max_pool_size == 5

    def test_max_pool_size_minimum_validation(self) -> None:
        """Test max_pool_size must be >= 1."""
        with pytest.raises(ValidationError):
            PoolConfig(max_pool_size=0)

        with pytest.raises(ValidationError):
            PoolConfig(max_pool_size=-1)

    def test_min_pool_size_minimum_validation(self) -> None:
        """Test min_pool_size must be >= 0."""
        config = PoolConfig(min_pool_size=0)
        assert config.min_pool_size == 0

        with pytest.raises(ValidationError):
            PoolConfig(min_pool_size=-1)

    def test_idle_timeout_positive_validation(self) -> None:
        """Test idle_timeout must be positive when not None."""
        with pytest.raises(ValidationError):
            PoolConfig(idle_timeout=0)

        with pytest.raises(ValidationError):
            PoolConfig(idle_timeout=-1)

    def test_acquisition_timeout_positive_validation(self) -> None:
        """Test acquisition_timeout must be positive when not None."""
        with pytest.raises(ValidationError):
            PoolConfig(acquisition_timeout=0)

        with pytest.raises(ValidationError):
            PoolConfig(acquisition_timeout=-10.5)

    def test_health_check_interval_positive_validation(self) -> None:
        """Test health_check_interval must be positive."""
        with pytest.raises(ValidationError):
            PoolConfig(health_check_interval=0)

        with pytest.raises(ValidationError):
            PoolConfig(health_check_interval=-5)

    def test_max_container_lifetime_positive_validation(self) -> None:
        """Test max_container_lifetime must be positive when not None."""
        with pytest.raises(ValidationError):
            PoolConfig(max_container_lifetime=0)

        with pytest.raises(ValidationError):
            PoolConfig(max_container_lifetime=-100)

    def test_max_container_uses_minimum_validation(self) -> None:
        """Test max_container_uses must be >= 1 when not None."""
        config = PoolConfig(max_container_uses=1)
        assert config.max_container_uses == 1

        with pytest.raises(ValidationError):
            PoolConfig(max_container_uses=0)

        with pytest.raises(ValidationError):
            PoolConfig(max_container_uses=-5)

    def test_config_serialization(self) -> None:
        """Test config can be serialized to dict."""
        config = PoolConfig(max_pool_size=15, min_pool_size=3)
        config_dict = config.model_dump()
        assert config_dict["max_pool_size"] == 15
        assert config_dict["min_pool_size"] == 3

    def test_config_from_dict(self) -> None:
        """Test config can be created from dict."""
        config_dict: dict[str, int | str] = {
            "max_pool_size": 20,
            "min_pool_size": 5,
            "exhaustion_strategy": "fail_fast",
        }
        config = PoolConfig.model_validate(config_dict)
        assert config.max_pool_size == 20
        assert config.min_pool_size == 5
        assert config.exhaustion_strategy == ExhaustionStrategy.FAIL_FAST
