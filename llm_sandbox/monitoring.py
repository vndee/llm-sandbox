"""Resource monitoring functionality for LLM Sandbox."""

import time
from typing import Optional, Dict, List
from dataclasses import dataclass
from datetime import datetime
from .exceptions import ResourceError

@dataclass
class ResourceUsage:
    """Represents resource usage at a point in time."""
    timestamp: datetime
    cpu_percent: float
    memory_bytes: int
    memory_percent: float
    network_rx_bytes: int
    network_tx_bytes: int

@dataclass
class ResourceLimits:
    """Resource limits for container execution."""
    max_cpu_percent: float = 100.0
    max_memory_bytes: int = 512 * 1024 * 1024  # 512MB
    max_execution_time: int = 30  # seconds
    max_network_bytes: int = 10 * 1024 * 1024  # 10MB

class ResourceMonitor:
    """Monitor and control resource usage of containers."""
    
    def __init__(self, container, limits: Optional[ResourceLimits] = None):
        self.container = container
        self.limits = limits or ResourceLimits()
        self.start_time: Optional[datetime] = None
        self.usage_history: List[ResourceUsage] = []
        self._previous_cpu_stats: Optional[Dict] = None
        
    def start(self):
        """Start monitoring resources."""
        self.start_time = datetime.now()
        self._previous_cpu_stats = None
        self.usage_history.clear()
        
    def check_limits(self, usage: ResourceUsage):
        """Check if resource usage exceeds limits."""
        if usage.cpu_percent > self.limits.max_cpu_percent:
            raise ResourceError(f"CPU usage exceeded: {usage.cpu_percent}%")
            
        if usage.memory_bytes > self.limits.max_memory_bytes:
            raise ResourceError(
                f"Memory usage exceeded: {usage.memory_bytes / 1024 / 1024:.1f}MB"
            )
            
        execution_time = (datetime.now() - self.start_time).total_seconds()
        if execution_time > self.limits.max_execution_time:
            raise ResourceError(f"Execution time exceeded: {execution_time:.1f}s")
            
        total_network = usage.network_rx_bytes + usage.network_tx_bytes
        if total_network > self.limits.max_network_bytes:
            raise ResourceError(
                f"Network usage exceeded: {total_network / 1024 / 1024:.1f}MB"
            )
    
    def update(self) -> ResourceUsage:
        """Update resource usage statistics."""
        stats = self.container.stats(stream=False)
        
        # Calculate CPU usage percentage
        cpu_stats = stats['cpu_stats']
        precpu_stats = stats['precpu_stats']
        
        cpu_delta = cpu_stats['cpu_usage']['total_usage'] - \
                   precpu_stats['cpu_usage']['total_usage']
        system_delta = cpu_stats['system_cpu_usage'] - \
                      precpu_stats['system_cpu_usage']
                      
        if system_delta > 0:
            cpu_percent = (cpu_delta / system_delta) * 100.0
        else:
            cpu_percent = 0.0
            
        # Memory usage
        memory_stats = stats['memory_stats']
        memory_bytes = memory_stats.get('usage', 0)
        memory_percent = (memory_bytes / memory_stats.get('limit', 1)) * 100.0
        
        # Network usage
        networks = stats.get('networks', {})
        rx_bytes = sum(net.get('rx_bytes', 0) for net in networks.values())
        tx_bytes = sum(net.get('tx_bytes', 0) for net in networks.values())
        
        usage = ResourceUsage(
            timestamp=datetime.now(),
            cpu_percent=cpu_percent,
            memory_bytes=memory_bytes,
            memory_percent=memory_percent,
            network_rx_bytes=rx_bytes,
            network_tx_bytes=tx_bytes
        )
        
        self.usage_history.append(usage)
        self.check_limits(usage)
        
        return usage
        
    def get_summary(self) -> Dict:
        """Get summary of resource usage."""
        if not self.usage_history:
            return {}
            
        cpu_usage = [u.cpu_percent for u in self.usage_history]
        memory_usage = [u.memory_bytes for u in self.usage_history]
        
        return {
            'start_time': self.start_time,
            'end_time': datetime.now(),
            'duration_seconds': (datetime.now() - self.start_time).total_seconds(),
            'cpu_percent': {
                'min': min(cpu_usage),
                'max': max(cpu_usage),
                'avg': sum(cpu_usage) / len(cpu_usage)
            },
            'memory_mb': {
                'min': min(memory_usage) / 1024 / 1024,
                'max': max(memory_usage) / 1024 / 1024,
                'avg': sum(memory_usage) / len(memory_usage) / 1024 / 1024
            },
            'samples_count': len(self.usage_history)
        } 