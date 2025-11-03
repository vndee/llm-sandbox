"""Container pool management for LLM Sandbox.

This module provides container pooling functionality to improve performance
by reusing pre-warmed containers instead of creating new ones for each execution.

The pool manager supports:
- Thread-safe container acquisition and release
- Multiple backends (Docker, Kubernetes, Podman)
- Configurable pool size and behavior
- Automatic health checking and container recycling
- Pre-warming of containers with environment setup
"""

from llm_sandbox.pool.base import ContainerPoolManager, ContainerState, PooledContainer
from llm_sandbox.pool.config import ExhaustionStrategy, PoolConfig
from llm_sandbox.pool.exceptions import PoolClosedError, PoolExhaustedError, PoolHealthCheckError
from llm_sandbox.pool.factory import create_pool_manager

__all__ = [
    "ContainerPoolManager",
    "ContainerState",
    "ExhaustionStrategy",
    "PoolClosedError",
    "PoolConfig",
    "PoolExhaustedError",
    "PoolHealthCheckError",
    "PooledContainer",
    "create_pool_manager",
]
