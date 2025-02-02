"""Tests for resource monitoring functionality."""

import pytest
from datetime import datetime, timedelta
from unittest.mock import MagicMock

from llm_sandbox.monitoring import ResourceMonitor, ResourceLimits, ResourceUsage
from llm_sandbox.exceptions import ResourceError


@pytest.fixture
def resource_limits():
    return ResourceLimits(
        max_cpu_percent=50.0,
        max_memory_bytes=256 * 1024 * 1024,  # 256MB
        max_execution_time=10,
        max_network_bytes=5 * 1024 * 1024,  # 5MB
    )


@pytest.fixture
def mock_container():
    return MagicMock()


@pytest.fixture
def monitor(mock_container, resource_limits):
    return ResourceMonitor(mock_container, resource_limits)


def test_start_monitoring(monitor):
    monitor.start()
    assert monitor.start_time is not None
    assert len(monitor.usage_history) == 0


def test_check_limits_cpu_exceeded(monitor):
    usage = ResourceUsage(
        timestamp=datetime.now(),
        cpu_percent=75.0,  # Exceeds 50% limit
        memory_bytes=100 * 1024 * 1024,
        memory_percent=20.0,
        network_rx_bytes=1000000,
        network_tx_bytes=1000000,
    )

    with pytest.raises(ResourceError) as exc_info:
        monitor.check_limits(usage)
    assert "CPU usage exceeded" in str(exc_info.value)


def test_check_limits_memory_exceeded(monitor):
    usage = ResourceUsage(
        timestamp=datetime.now(),
        cpu_percent=25.0,
        memory_bytes=300 * 1024 * 1024,  # Exceeds 256MB limit
        memory_percent=60.0,
        network_rx_bytes=1000000,
        network_tx_bytes=1000000,
    )

    with pytest.raises(ResourceError) as exc_info:
        monitor.check_limits(usage)
    assert "Memory usage exceeded" in str(exc_info.value)


def test_check_limits_network_exceeded(monitor):
    usage = ResourceUsage(
        timestamp=datetime.now(),
        cpu_percent=25.0,
        memory_bytes=100 * 1024 * 1024,
        memory_percent=20.0,
        network_rx_bytes=3 * 1024 * 1024,
        network_tx_bytes=3 * 1024 * 1024,  # Total 6MB exceeds 5MB limit
    )

    with pytest.raises(ResourceError) as exc_info:
        monitor.check_limits(usage)
    assert "Network usage exceeded" in str(exc_info.value)


def test_check_limits_time_exceeded(monitor):
    monitor.start()
    monitor.start_time = datetime.now() - timedelta(seconds=15)  # Exceeds 10s limit

    usage = ResourceUsage(
        timestamp=datetime.now(),
        cpu_percent=25.0,
        memory_bytes=100 * 1024 * 1024,
        memory_percent=20.0,
        network_rx_bytes=1000000,
        network_tx_bytes=1000000,
    )

    with pytest.raises(ResourceError) as exc_info:
        monitor.check_limits(usage)
    assert "Execution time exceeded" in str(exc_info.value)


def test_update_usage(monitor, mock_container):
    mock_stats = {
        "cpu_stats": {
            "cpu_usage": {"total_usage": 100000},
            "system_cpu_usage": 1000000,
        },
        "precpu_stats": {
            "cpu_usage": {"total_usage": 90000},
            "system_cpu_usage": 900000,
        },
        "memory_stats": {"usage": 100 * 1024 * 1024, "limit": 512 * 1024 * 1024},
        "networks": {"eth0": {"rx_bytes": 1000000, "tx_bytes": 500000}},
    }

    mock_container.stats.return_value = mock_stats

    usage = monitor.update()
    assert isinstance(usage, ResourceUsage)
    assert len(monitor.usage_history) == 1


def test_get_summary_empty(monitor):
    summary = monitor.get_summary()
    assert summary == {}


def test_get_summary_with_data(monitor):
    monitor.start()

    # Add some usage data
    usage1 = ResourceUsage(
        timestamp=datetime.now(),
        cpu_percent=25.0,
        memory_bytes=100 * 1024 * 1024,
        memory_percent=20.0,
        network_rx_bytes=1000000,
        network_tx_bytes=500000,
    )

    usage2 = ResourceUsage(
        timestamp=datetime.now(),
        cpu_percent=35.0,
        memory_bytes=150 * 1024 * 1024,
        memory_percent=30.0,
        network_rx_bytes=2000000,
        network_tx_bytes=1000000,
    )

    monitor.usage_history.extend([usage1, usage2])

    summary = monitor.get_summary()

    assert "start_time" in summary
    assert "end_time" in summary
    assert "duration_seconds" in summary
    assert "cpu_percent" in summary
    assert "memory_mb" in summary
    assert summary["samples_count"] == 2

    cpu_stats = summary["cpu_percent"]
    assert cpu_stats["min"] == 25.0
    assert cpu_stats["max"] == 35.0
    assert cpu_stats["avg"] == 30.0


@pytest.mark.parametrize(
    "usage_data",
    [
        {
            "cpu_percent": 25.0,
            "memory_bytes": 100 * 1024 * 1024,
            "network_bytes": (1000000, 500000),
        },
        {
            "cpu_percent": 45.0,
            "memory_bytes": 200 * 1024 * 1024,
            "network_bytes": (2000000, 1000000),
        },
    ],
)
def test_resource_usage_tracking(monitor, usage_data):
    monitor.start()

    usage = ResourceUsage(
        timestamp=datetime.now(),
        cpu_percent=usage_data["cpu_percent"],
        memory_bytes=usage_data["memory_bytes"],
        memory_percent=(usage_data["memory_bytes"] / (512 * 1024 * 1024)) * 100,
        network_rx_bytes=usage_data["network_bytes"][0],
        network_tx_bytes=usage_data["network_bytes"][1],
    )

    monitor.usage_history.append(usage)
    summary = monitor.get_summary()

    assert summary["cpu_percent"]["max"] == usage_data["cpu_percent"]
    assert summary["memory_mb"]["max"] == usage_data["memory_bytes"] / (1024 * 1024)
