"""Tests for pool exception classes."""

import pytest

from llm_sandbox.pool.exceptions import PoolClosedError, PoolExhaustedError, PoolHealthCheckError


class TestPoolExhaustedError:
    """Test PoolExhaustedError exception."""

    def test_init_without_timeout(self) -> None:
        """Test PoolExhaustedError initialization without timeout."""
        error = PoolExhaustedError(pool_size=10)
        assert "All 10 containers in pool are busy" in str(error)
        assert error.pool_size == 10
        assert error.timeout is None

    def test_init_with_timeout(self) -> None:
        """Test PoolExhaustedError initialization with timeout."""
        error = PoolExhaustedError(pool_size=5, timeout=30.0)
        assert "All 5 containers in pool are busy" in str(error)
        assert "timeout of 30.0s exceeded" in str(error)
        assert error.pool_size == 5
        assert error.timeout == 30.0

    def test_inheritance(self) -> None:
        """Test PoolExhaustedError inherits from Exception."""
        error = PoolExhaustedError(pool_size=10)
        assert isinstance(error, Exception)

    def test_attributes_accessible(self) -> None:
        """Test error attributes are accessible."""
        error = PoolExhaustedError(pool_size=15, timeout=60.0)
        assert error.pool_size == 15
        assert error.timeout == 60.0

    def test_different_pool_sizes(self) -> None:
        """Test error message with different pool sizes."""
        error1 = PoolExhaustedError(pool_size=1)
        assert "1 containers" in str(error1) or "1 container" in str(error1)

        error2 = PoolExhaustedError(pool_size=100)
        assert "100 containers" in str(error2)

    def test_can_be_raised(self) -> None:
        """Test exception can be raised and caught."""
        with pytest.raises(PoolExhaustedError) as exc_info:
            raise PoolExhaustedError(pool_size=10, timeout=30.0)

        assert exc_info.value.pool_size == 10
        assert exc_info.value.timeout == 30.0


class TestPoolClosedError:
    """Test PoolClosedError exception."""

    def test_init(self) -> None:
        """Test PoolClosedError initialization."""
        error = PoolClosedError()
        assert "Container pool has been closed" in str(error)

    def test_inheritance(self) -> None:
        """Test PoolClosedError inherits from Exception."""
        error = PoolClosedError()
        assert isinstance(error, Exception)

    def test_error_message_fixed(self) -> None:
        """Test error message is consistent."""
        error1 = PoolClosedError()
        error2 = PoolClosedError()
        assert str(error1) == str(error2)

    def test_can_be_raised(self) -> None:
        """Test exception can be raised and caught."""
        with pytest.raises(PoolClosedError) as exc_info:
            raise PoolClosedError

        assert "Container pool has been closed" in str(exc_info.value)


class TestPoolHealthCheckError:
    """Test PoolHealthCheckError exception."""

    def test_init(self) -> None:
        """Test PoolHealthCheckError initialization."""
        error = PoolHealthCheckError(container_id="container123", reason="Container not responding")
        assert "Health check failed for container container123" in str(error)
        assert "Container not responding" in str(error)
        assert error.container_id == "container123"
        assert error.reason == "Container not responding"

    def test_inheritance(self) -> None:
        """Test PoolHealthCheckError inherits from Exception."""
        error = PoolHealthCheckError(container_id="test", reason="test reason")
        assert isinstance(error, Exception)

    def test_attributes_accessible(self) -> None:
        """Test error attributes are accessible."""
        error = PoolHealthCheckError(container_id="abc-def-123", reason="Pod not found")
        assert error.container_id == "abc-def-123"
        assert error.reason == "Pod not found"

    def test_different_container_ids(self) -> None:
        """Test error with different container IDs."""
        error1 = PoolHealthCheckError(container_id="short", reason="test")
        assert "short" in str(error1)

        error2 = PoolHealthCheckError(
            container_id="very-long-container-id-12345678",
            reason="timeout",
        )
        assert "very-long-container-id-12345678" in str(error2)

    def test_different_reasons(self) -> None:
        """Test error with different failure reasons."""
        reasons = [
            "Container stopped unexpectedly",
            "Health check timeout",
            "Exec command failed",
            "Container not found",
        ]

        for reason in reasons:
            error = PoolHealthCheckError(container_id="test", reason=reason)
            assert reason in str(error)

    def test_can_be_raised(self) -> None:
        """Test exception can be raised and caught."""
        with pytest.raises(PoolHealthCheckError) as exc_info:
            raise PoolHealthCheckError(container_id="test123", reason="Failed health check")

        assert exc_info.value.container_id == "test123"
        assert exc_info.value.reason == "Failed health check"


class TestExceptionIntegration:
    """Test exception integration scenarios."""

    def test_all_exceptions_are_catchable_as_base_exception(self) -> None:
        """Test all pool exceptions can be caught as Exception."""
        exceptions = [
            PoolExhaustedError(pool_size=10),
            PoolClosedError(),
            PoolHealthCheckError(container_id="test", reason="test"),
        ]

        for exc in exceptions:
            assert isinstance(exc, Exception)

    def test_exceptions_have_unique_types(self) -> None:
        """Test each exception has a unique type."""
        exc1 = PoolExhaustedError(pool_size=10)
        exc2 = PoolClosedError()
        exc3 = PoolHealthCheckError(container_id="test", reason="test")

        assert type(exc1) is not type(exc2)
        assert type(exc2) is not type(exc3)
        assert type(exc1) is not type(exc3)

    def test_exceptions_can_be_caught_separately(self) -> None:
        """Test exceptions can be caught with specific handlers."""

        def raise_exhausted() -> None:
            raise PoolExhaustedError(pool_size=5)

        def raise_closed() -> None:
            raise PoolClosedError

        def raise_health_check() -> None:
            raise PoolHealthCheckError(container_id="test", reason="failed")

        with pytest.raises(PoolExhaustedError):
            raise_exhausted()

        with pytest.raises(PoolClosedError):
            raise_closed()

        with pytest.raises(PoolHealthCheckError):
            raise_health_check()
