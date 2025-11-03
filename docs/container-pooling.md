# Container Pooling

Container pooling is a performance optimization feature that dramatically improves execution speed by reusing pre-warmed containers instead of creating new ones for each code execution. This is particularly beneficial for applications that execute code frequently or handle concurrent requests.

!!! success "Performance Improvement"
    Container pooling can improve execution performance by **up to 10x** by eliminating container creation overhead and reusing pre-warmed environments.

## Overview

### What is Container Pooling?

Container pooling maintains a pool of ready-to-use containers that can be quickly assigned to execute code and then returned to the pool for reuse. Instead of creating a new container for each execution (which involves image pulling, container startup, and environment setup), pooling allows you to:

1. **Pre-create** a set of containers during initialization
2. **Pre-warm** containers with language environments and dependencies
3. **Reuse** containers across multiple executions
4. **Manage** container lifecycle automatically with health checks
5. **Scale** by configuring pool size based on your workload

### Key Benefits

| Feature | Benefit |
|---------|---------|
| **� Faster Execution** | Eliminate container creation overhead (3-5 seconds per execution) |
| **=% Pre-warmed Environments** | Dependencies pre-installed, no wait time |
| **= Thread-Safe** | Safely handle concurrent requests |
| **{ Resource Efficient** | Automatic container recycling and health management |
| **=� Observable** | Real-time statistics and monitoring |
| **<� Flexible** | Configurable pool size, timeouts, and behaviors |

### When to Use Container Pooling

** Recommended for:**

- High-frequency code execution (multiple executions per minute)
- Concurrent request handling (web applications, APIs)
- Production deployments requiring low latency
- Batch processing with repeated executions
- Applications with predictable resource requirements

**L Not necessary for:**

- One-off script execution
- Extremely low-frequency usage (once per hour or less)
- Development and testing with frequent environment changes
- Scenarios requiring different container configurations per execution

## Quick Start

### Basic Usage

The simplest way to use container pooling is with the `use_pool` parameter:

```python
from llm_sandbox import SandboxSession
from llm_sandbox.pool import PoolConfig

# Configure the pool
pool_config = PoolConfig(
    max_pool_size=10,          # Maximum 10 containers
    min_pool_size=3,           # Keep at least 3 warm containers
    enable_prewarming=True,    # Pre-warm containers
)

# Create a pooled session
with SandboxSession(
    lang="python",
    use_pool=True,
    pool_config=pool_config,
) as session:
    result = session.run("print('Hello from pooled container!')")
    print(result.stdout)
    # Container automatically returned to pool on exit
```

### Shared Pool Manager

For better resource efficiency, share a single pool across multiple sessions:

```python
from llm_sandbox import SandboxSession
from llm_sandbox.pool import create_pool_manager, PoolConfig

# Create a shared pool manager
# Note: Libraries are installed during session initialization
pool = create_pool_manager(
    backend="docker",
    config=PoolConfig(
        max_pool_size=10,
        min_pool_size=3,
    ),
    lang="python",
    libraries=["numpy", "pandas"],  # Libraries installed in all pooled containers
)

try:
    # Use the pool in multiple sessions
    with SandboxSession(lang="python", pool_manager=pool) as session1:
        result1 = session1.run("import pandas as pd; print(pd.__version__)")
        print(f"Session 1: {result1.stdout}")

    with SandboxSession(lang="python", pool_manager=pool) as session2:
        result2 = session2.run("import numpy as np; print(np.__version__)")
        print(f"Session 2: {result2.stdout}")

finally:
    # Clean up pool when done
    pool.close()
```

## Configuration

### Pool Configuration Options

The `PoolConfig` class provides comprehensive configuration options:

```python
from llm_sandbox.pool import PoolConfig, ExhaustionStrategy

config = PoolConfig(
    # Pool size limits
    max_pool_size=10,                      # Maximum containers in pool
    min_pool_size=2,                       # Minimum warm containers to maintain

    # Timeout configuration
    idle_timeout=300.0,                    # Recycle idle containers after 5 minutes
    acquisition_timeout=30.0,              # Wait time when acquiring from exhausted pool

    # Health and lifecycle management
    health_check_interval=60.0,            # Health check frequency (seconds)
    max_container_lifetime=3600.0,         # Maximum container lifetime (1 hour)
    max_container_uses=100,                # Maximum uses before recycling

    # Pool exhaustion behavior
    exhaustion_strategy=ExhaustionStrategy.WAIT,  # WAIT, FAIL_FAST, or TEMPORARY

    # Pre-warming
    enable_prewarming=True,                # Automatically create min_pool_size containers on startup
)
```

### Configuration Parameters

#### Pool Size Limits

**`max_pool_size`** (int, default: 10)
:   Maximum number of containers in the pool. The pool will never exceed this limit.

    ```python
    PoolConfig(max_pool_size=20)  # Allow up to 20 containers
    ```

**`min_pool_size`** (int, default: 0)
:   Minimum number of pre-warmed containers to maintain. A background thread ensures the pool always has at least this many containers ready.

    ```python
    PoolConfig(
        max_pool_size=10,
        min_pool_size=3,  # Always keep 3 containers warm
    )
    ```

!!! warning "Pool Size Constraint"
    `min_pool_size` must be less than or equal to `max_pool_size`. A validation error will be raised if this constraint is violated.

#### Timeout Configuration

**`idle_timeout`** (float | None, default: 300.0)
:   Time in seconds before an idle container is recycled. Set to `None` to disable idle timeout.

    ```python
    PoolConfig(idle_timeout=600.0)  # Recycle after 10 minutes idle
    ```

**`acquisition_timeout`** (float | None, default: 30.0)
:   Maximum time to wait when acquiring a container from an exhausted pool (only applies with `WAIT` strategy). Set to `None` for no timeout.

    ```python
    PoolConfig(
        exhaustion_strategy=ExhaustionStrategy.WAIT,
        acquisition_timeout=60.0,  # Wait up to 60 seconds
    )
    ```

#### Health and Lifecycle

**`health_check_interval`** (float, default: 60.0)
:   Interval in seconds between health checks for idle containers. Health checks verify containers are responsive and properly functioning.

    ```python
    PoolConfig(health_check_interval=30.0)  # Check every 30 seconds
    ```

**`max_container_lifetime`** (float | None, default: 3600.0)
:   Maximum lifetime of a container in seconds before it's recycled, regardless of health. Prevents issues from long-running containers. Set to `None` for no limit.

    ```python
    PoolConfig(max_container_lifetime=7200.0)  # 2 hours maximum
    ```

**`max_container_uses`** (int | None, default: None)
:   Maximum number of times a container can be used before recycling. Useful for preventing resource leaks from repeated use. Set to `None` for no limit.

    ```python
    PoolConfig(max_container_uses=50)  # Recycle after 50 uses
    ```

#### Exhaustion Strategy

**`exhaustion_strategy`** (ExhaustionStrategy, default: WAIT)
:   Defines behavior when all containers in the pool are busy. Three strategies available:

    - **`ExhaustionStrategy.WAIT`**: Wait for a container to become available
    - **`ExhaustionStrategy.FAIL_FAST`**: Immediately raise `PoolExhaustedError`
    - **`ExhaustionStrategy.TEMPORARY`**: Create a temporary container outside the pool

    See [Pool Exhaustion Strategies](#pool-exhaustion-strategies) for details.

#### Pre-warming

**`enable_prewarming`** (bool, default: True)
:   Enable automatic container creation during pool initialization. When enabled, the pool will create `min_pool_size` containers on startup, making them immediately available for use.

    ```python
    PoolConfig(
        enable_prewarming=True,
        min_pool_size=3,  # Create 3 containers on startup
    )
    ```

!!! tip "Installing Libraries in Pooled Containers"
    To pre-install libraries in all pooled containers, pass the `libraries` parameter when creating the pool manager:

    ```python
    pool = create_pool_manager(
        backend="docker",
        config=PoolConfig(max_pool_size=10),
        lang="python",
        libraries=["requests", "numpy", "pandas"],  # Installed in all containers
    )
    ```

## Pool Exhaustion Strategies

When all containers in the pool are busy, the pool must decide how to handle new acquisition requests. LLM Sandbox provides three strategies:

### WAIT Strategy (Default)

Wait for a container to become available, with optional timeout.

```python
from llm_sandbox.pool import PoolConfig, ExhaustionStrategy

config = PoolConfig(
    max_pool_size=5,
    exhaustion_strategy=ExhaustionStrategy.WAIT,
    acquisition_timeout=30.0,  # Wait up to 30 seconds
)
```

**When to use:**

- Production applications with predictable load
- When slight delays are acceptable
- Web applications with request queuing

**Behavior:**

- Blocks until a container becomes available
- Raises `PoolExhaustedError` if timeout is exceeded
- Thread-safe with proper notification handling

### FAIL_FAST Strategy

Immediately raise an error when pool is exhausted.

```python
config = PoolConfig(
    max_pool_size=5,
    exhaustion_strategy=ExhaustionStrategy.FAIL_FAST,
)
```

**When to use:**

- When you want explicit control over exhaustion handling
- Real-time systems that can't tolerate delays
- When you want to implement custom retry logic

**Behavior:**

- Raises `PoolExhaustedError` immediately
- No waiting or blocking
- Caller must handle the exception

**Example with error handling:**

```python
from llm_sandbox import SandboxSession
from llm_sandbox.pool import PoolExhaustedError

try:
    with SandboxSession(lang="python", pool_manager=pool) as session:
        result = session.run("print('Hello')")
except PoolExhaustedError as e:
    print(f"Pool exhausted: {e}")
    # Implement custom logic: retry later, use fallback, etc.
```

### TEMPORARY Strategy

Create a temporary container outside the pool when exhausted.

```python
config = PoolConfig(
    max_pool_size=5,
    exhaustion_strategy=ExhaustionStrategy.TEMPORARY,
)
```

**When to use:**

- Handling occasional traffic spikes
- When you can't afford to fail or wait
- Development and testing environments

**Behavior:**

- Creates a new container not managed by the pool
- Container is destroyed after use (not returned to pool)
- No limits on temporary containers (use with caution)

!!! warning "Resource Management"
    Temporary containers bypass pool limits and can lead to resource exhaustion if used extensively. Monitor system resources when using this strategy.

## Concurrent Execution

Container pools are designed for thread-safe concurrent execution. Multiple threads can safely acquire and release containers simultaneously.

### Thread-Safe Pool Access

```python
import threading
from llm_sandbox import SandboxSession
from llm_sandbox.pool import create_pool_manager, PoolConfig

# Create shared pool
pool = create_pool_manager(
    backend="docker",
    config=PoolConfig(max_pool_size=5),
    lang="python",
)

def run_code(task_id: int):
    """Execute code in a thread-safe manner."""
    with SandboxSession(lang="python", pool_manager=pool) as session:
        result = session.run(f'print("Task {task_id} completed")')
        return result.stdout

try:
    # Create multiple threads
    threads = [
        threading.Thread(target=run_code, args=(i,))
        for i in range(20)
    ]

    # Start all threads
    for thread in threads:
        thread.start()

    # Wait for completion
    for thread in threads:
        thread.join()

finally:
    pool.close()
```

### Concurrent Execution with ThreadPoolExecutor

For better control and result collection, use `concurrent.futures`:

```python
from concurrent.futures import ThreadPoolExecutor, as_completed
from llm_sandbox import SandboxSession
from llm_sandbox.pool import create_pool_manager, PoolConfig

pool = create_pool_manager(
    backend="docker",
    config=PoolConfig(max_pool_size=5, min_pool_size=2),
    lang="python",
)

def execute_task(task_id: int, code: str):
    """Execute a task and return results."""
    with SandboxSession(lang="python", pool_manager=pool) as session:
        result = session.run(code)
        return {
            "task_id": task_id,
            "output": result.stdout,
            "success": result.exit_code == 0,
        }

try:
    tasks = [
        (i, f'print("Processing item {i}")')
        for i in range(20)
    ]

    # Execute concurrently
    with ThreadPoolExecutor(max_workers=10) as executor:
        # Submit all tasks
        futures = [
            executor.submit(execute_task, task_id, code)
            for task_id, code in tasks
        ]

        # Collect results as they complete
        for future in as_completed(futures):
            result = future.result()
            print(f"Task {result['task_id']}: {result['output'].strip()}")

finally:
    pool.close()
```

### Performance Comparison

```python
import time
from llm_sandbox import SandboxSession
from llm_sandbox.pool import create_pool_manager, PoolConfig

def benchmark_without_pool(num_tasks: int):
    """Benchmark without pooling."""
    start = time.time()

    for i in range(num_tasks):
        with SandboxSession(lang="python", use_pool=False) as session:
            session.run("print('test')")

    return time.time() - start

def benchmark_with_pool(num_tasks: int):
    """Benchmark with pooling."""
    pool = create_pool_manager(
        backend="docker",
        config=PoolConfig(max_pool_size=3, min_pool_size=2),
        lang="python",
    )

    try:
        start = time.time()

        for i in range(num_tasks):
            with SandboxSession(lang="python", pool_manager=pool) as session:
                session.run("print('test')")

        return time.time() - start
    finally:
        pool.close()

# Run benchmarks
num_tasks = 10
no_pool_time = benchmark_without_pool(num_tasks)
pool_time = benchmark_with_pool(num_tasks)

print(f"Without pool: {no_pool_time:.2f}s")
print(f"With pool: {pool_time:.2f}s")
print(f"Speedup: {no_pool_time / pool_time:.2f}x faster")
```

## Monitoring and Statistics

Monitor pool health and performance using the `get_stats()` method.

### Real-Time Statistics

```python
from llm_sandbox.pool import create_pool_manager, PoolConfig

pool = create_pool_manager(
    backend="docker",
    config=PoolConfig(max_pool_size=10, min_pool_size=3),
    lang="python",
)

# Get current statistics
stats = pool.get_stats()

print(f"Total containers: {stats['total_size']}/{stats['max_size']}")
print(f"Minimum pool size: {stats['min_size']}")
print(f"\nContainer states:")
for state, count in stats['state_counts'].items():
    if count > 0:
        print(f"  {state}: {count}")
print(f"\nPool status: {'Closed' if stats['closed'] else 'Active'}")

pool.close()
```

**Output example:**

```
Total containers: 5/10
Minimum pool size: 3

Container states:
  idle: 4
  busy: 1

Pool status: Active
```

### Statistics Dictionary

The `get_stats()` method returns a dictionary with the following keys:

| Key | Type | Description |
|-----|------|-------------|
| `total_size` | int | Current number of containers in pool |
| `max_size` | int | Maximum pool size (from config) |
| `min_size` | int | Minimum pool size (from config) |
| `state_counts` | dict | Count of containers in each state |
| `closed` | bool | Whether pool is closed |

**Container States:**

- `initializing`: Container is being created
- `idle`: Container is available for use
- `busy`: Container is currently in use
- `unhealthy`: Container failed health check
- `removing`: Container is being removed

### Continuous Monitoring

For production deployments, implement continuous monitoring:

```python
import threading
import time
from llm_sandbox.pool import create_pool_manager, PoolConfig

pool = create_pool_manager(
    backend="docker",
    config=PoolConfig(max_pool_size=10, min_pool_size=3),
    lang="python",
)

def monitor_pool(interval: int = 10):
    """Monitor pool statistics continuously."""
    while not stop_event.is_set():
        stats = pool.get_stats()

        # Log statistics
        print(f"[{time.strftime('%H:%M:%S')}] "
              f"Total: {stats['total_size']}, "
              f"Idle: {stats['state_counts'].get('idle', 0)}, "
              f"Busy: {stats['state_counts'].get('busy', 0)}")

        # Alert if pool is nearly exhausted
        busy_ratio = stats['state_counts'].get('busy', 0) / stats['total_size']
        if busy_ratio > 0.8:
            print(f"�  Warning: Pool is {busy_ratio*100:.0f}% utilized")

        time.sleep(interval)

# Start monitoring thread
stop_event = threading.Event()
monitor_thread = threading.Thread(target=monitor_pool, daemon=True)
monitor_thread.start()

try:
    # Use pool for your workload
    # ...
    pass
finally:
    stop_event.set()
    monitor_thread.join(timeout=5)
    pool.close()
```

## Health Management

Container pools automatically manage container health through periodic checks and lifecycle policies.

### Health Check Mechanism

The pool performs regular health checks on idle containers:

1. **Responsiveness Check**: Executes a simple command to verify the container responds
2. **Container Status**: Checks if the container is running
3. **Automatic Removal**: Unhealthy containers are automatically removed
4. **Replacement**: Removed containers are replaced to maintain `min_pool_size`

```python
from llm_sandbox.pool import PoolConfig

config = PoolConfig(
    max_pool_size=5,
    min_pool_size=2,
    health_check_interval=30.0,  # Check every 30 seconds
)
```

### Container Lifecycle Policies

Containers are recycled based on multiple criteria to prevent resource leaks and ensure reliability:

#### 1. Idle Timeout

Recycle containers that have been idle too long:

```python
config = PoolConfig(
    idle_timeout=300.0,  # Recycle after 5 minutes idle
)
```

#### 2. Maximum Lifetime

Recycle containers after a maximum lifetime:

```python
config = PoolConfig(
    max_container_lifetime=3600.0,  # Recycle after 1 hour
)
```

#### 3. Maximum Uses

Recycle containers after a number of uses:

```python
config = PoolConfig(
    max_container_uses=100,  # Recycle after 100 executions
)
```

### Custom Health Check Intervals

Adjust health check frequency based on your requirements:

```python
from llm_sandbox.pool import PoolConfig

# Frequent health checks (resource intensive but quick detection)
config_aggressive = PoolConfig(
    health_check_interval=10.0,  # Every 10 seconds
)

# Relaxed health checks (resource efficient)
config_relaxed = PoolConfig(
    health_check_interval=300.0,  # Every 5 minutes
)
```

!!! tip "Health Check Tuning"
    - **Development**: Use longer intervals (60-300 seconds) to reduce overhead
    - **Production**: Use shorter intervals (10-30 seconds) for faster failure detection
    - **Critical systems**: Combine with container lifetime limits for guaranteed freshness

## Backend-Specific Features

### Docker

Docker pools support all standard pool features with additional optimizations:

```python
from llm_sandbox.pool import create_pool_manager, PoolConfig

pool = create_pool_manager(
    backend="docker",
    config=PoolConfig(max_pool_size=10),
    lang="python",
    # Docker-specific options
    runtime_configs={
        "mem_limit": "512m",
        "cpu_period": 100000,
        "cpu_quota": 50000,
    },
)
```

### Kubernetes

Kubernetes pools manage pods instead of containers:

```python
from llm_sandbox.pool import create_pool_manager, PoolConfig
from kubernetes import client, config

# Load kubeconfig
config.load_kube_config()
k8s_client = client.CoreV1Api()

pool = create_pool_manager(
    backend="kubernetes",
    config=PoolConfig(
        max_pool_size=10,
        min_pool_size=2,
        health_check_interval=60.0,  # Pods are slower to start
    ),
    lang="python",
    client=k8s_client,
    namespace="llm-sandbox",
)
```

!!! note "Kubernetes Considerations"
    - Pods take longer to start than Docker containers (30-60 seconds)
    - Use higher `min_pool_size` to ensure availability
    - Consider pod resource limits and node capacity
    - Health checks work with pod status and exec commands

### Podman

Podman pools use the same API as Docker:

```python
from llm_sandbox.pool import create_pool_manager, PoolConfig
from podman import PodmanClient

# Create Podman client
podman_client = PodmanClient()

pool = create_pool_manager(
    backend="podman",
    config=PoolConfig(max_pool_size=10),
    lang="python",
    client=podman_client,
)
```

## Best Practices

### 1. Right-Size Your Pool

Choose pool sizes based on your workload:

**Low Concurrency (1-5 concurrent requests):**
```python
PoolConfig(
    max_pool_size=5,
    min_pool_size=2,
)
```

**Medium Concurrency (5-20 concurrent requests):**
```python
PoolConfig(
    max_pool_size=20,
    min_pool_size=5,
)
```

**High Concurrency (20+ concurrent requests):**
```python
PoolConfig(
    max_pool_size=50,
    min_pool_size=10,
)
```

### 2. Pre-install Common Dependencies

Maximize performance by pre-installing frequently used libraries when creating the pool manager:

```python
pool = create_pool_manager(
    backend="docker",
    config=PoolConfig(
        max_pool_size=10,
        enable_prewarming=True,
    ),
    lang="python",
    libraries=[
        # Data science stack
        "numpy",
        "pandas",
        "matplotlib",
        "scikit-learn",
        # Web and utilities
        "requests",
        "beautifulsoup4",
    ],
)
```

### 3. Set Appropriate Timeouts

Balance responsiveness with resource usage:

```python
PoolConfig(
    idle_timeout=300.0,           # 5 min - recycle idle containers
    acquisition_timeout=30.0,      # 30 sec - fail fast on exhaustion
    max_container_lifetime=3600.0, # 1 hour - prevent resource leaks
)
```

### 4. Choose the Right Exhaustion Strategy

Match strategy to your use case:

- **Web APIs**: Use `WAIT` with reasonable timeout
- **Batch processing**: Use `FAIL_FAST` with retry logic
- **Development**: Use `TEMPORARY` for flexibility

### 5. Monitor Pool Health

Implement monitoring for production deployments:

```python
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def log_pool_stats(pool):
    """Log pool statistics."""
    stats = pool.get_stats()
    logger.info(
        f"Pool stats - Total: {stats['total_size']}, "
        f"Idle: {stats['state_counts'].get('idle', 0)}, "
        f"Busy: {stats['state_counts'].get('busy', 0)}"
    )
```

### 6. Graceful Shutdown

Always close pools properly to clean up resources:

```python
pool = create_pool_manager(
    backend="docker",
    config=PoolConfig(max_pool_size=10),
    lang="python",
)

try:
    # Use pool
    pass
finally:
    # Ensure cleanup happens
    pool.close()
```

Or use as context manager:

```python
with create_pool_manager(
    backend="docker",
    config=PoolConfig(max_pool_size=10),
    lang="python",
) as pool:
    # Use pool
    # Automatically closed on exit
    pass
```

### 7. Language-Specific Optimizations

Different languages have different startup costs and optimization strategies:

**Python:**
```python
# Pre-install heavy packages when creating pool
pool = create_pool_manager(
    backend="docker",
    config=PoolConfig(
        max_container_lifetime=3600.0,  # Longer lifetime (venv is cached)
    ),
    lang="python",
    libraries=["numpy", "pandas"],  # Pre-install heavy packages
)
```

**JavaScript:**
```python
# Pre-install npm packages when creating pool
pool = create_pool_manager(
    backend="docker",
    config=PoolConfig(
        max_container_lifetime=1800.0,  # Shorter lifetime (node_modules)
    ),
    lang="javascript",
    libraries=["axios", "lodash"],  # Pre-install npm packages
)
```

**Go:**
```python
# Go automatically initializes go.mod during container setup
pool = create_pool_manager(
    backend="docker",
    config=PoolConfig(
        enable_prewarming=True,
        max_container_lifetime=7200.0,  # Longer lifetime (compiled binaries)
    ),
    lang="go",
)
```

## Troubleshooting

### Pool Exhaustion Issues

**Symptom:** Frequent `PoolExhaustedError` exceptions

**Solutions:**

1. Increase pool size:
   ```python
   PoolConfig(max_pool_size=20)  # Increase from default 10
   ```

2. Optimize code execution time to free containers faster

3. Switch to `WAIT` strategy with timeout:
   ```python
   PoolConfig(
       exhaustion_strategy=ExhaustionStrategy.WAIT,
       acquisition_timeout=60.0,
   )
   ```

### High Memory Usage

**Symptom:** Pool consuming excessive memory

**Solutions:**

1. Reduce pool size:
   ```python
   PoolConfig(max_pool_size=5, min_pool_size=1)
   ```

2. Set container memory limits:
   ```python
   pool = create_pool_manager(
       backend="docker",
       runtime_configs={"mem_limit": "256m"},
       ...
   )
   ```

3. Implement aggressive recycling:
   ```python
   PoolConfig(
       idle_timeout=60.0,            # Short idle timeout
       max_container_lifetime=900.0,  # 15 minute lifetime
       max_container_uses=20,         # Recycle after 20 uses
   )
   ```

### Unhealthy Containers

**Symptom:** Containers frequently marked as unhealthy

**Solutions:**

1. Increase health check interval to reduce false positives:
   ```python
   PoolConfig(health_check_interval=120.0)
   ```

2. Check container resource limits aren't too restrictive

3. Review container logs for underlying issues

### Slow Container Creation

**Symptom:** Long wait times when pool needs to create containers

**Solutions:**

1. Increase `min_pool_size` to pre-create more containers:
   ```python
   PoolConfig(max_pool_size=20, min_pool_size=10)
   ```

2. Use pre-pulled images:
   ```python
   # Pull image before creating pool
   import docker
   client = docker.from_env()
   client.images.pull("python:3.11-slim")
   ```

3. For Kubernetes, use pod anti-affinity to spread pods across nodes

## Examples

Complete working examples are available in the repository:

- **[pool_basic_demo.py](https://github.com/vndee/llm-sandbox/blob/main/examples/pool_basic_demo.py)** - Basic pool usage, configuration, and statistics
- **[pool_concurrent_demo.py](https://github.com/vndee/llm-sandbox/blob/main/examples/pool_concurrent_demo.py)** - Concurrent execution patterns and performance comparison
- **[pool_monitoring_demo.py](https://github.com/vndee/llm-sandbox/blob/main/examples/pool_monitoring_demo.py)** - Health monitoring and lifecycle management

## API Reference

For detailed API documentation, see:

- [`ContainerPoolManager`](api-reference.md#llm_sandbox.pool.base.ContainerPoolManager) - Base pool manager class
- [`PoolConfig`](api-reference.md#llm_sandbox.pool.config.PoolConfig) - Pool configuration
- [`PooledSandboxSession`](api-reference.md#llm_sandbox.pool.session.PooledSandboxSession) - Pooled session class
- [`create_pool_manager()`](api-reference.md#llm_sandbox.pool.factory.create_pool_manager) - Factory function

## Related Documentation

- [Configuration Guide](configuration.md) - General session configuration
- [Container Backends](backends.md) - Docker, Kubernetes, and Podman setup
- [Security](security.md) - Security considerations for pooled containers
- [Performance Optimization](configuration.md#performance-optimization) - Additional optimization tips
