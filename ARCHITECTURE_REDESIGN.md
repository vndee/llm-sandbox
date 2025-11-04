# Container Pool Architecture Redesign

## Summary

This document describes the architectural redesign of the container pooling feature to eliminate code duplication and improve maintainability.

## Problem Statement

The original `PooledSandboxSession` class inherited from `BaseSession` and duplicated backend-specific logic, leading to:

1. **Missing attributes**: Required attributes like `stream` were not initialized
2. **Incomplete implementations**: Output processing methods were simplified and didn't handle all cases correctly (e.g., Docker's `demux=True` returning tuples)
3. **Code duplication**: Logic for processing command output, handling streams, etc. was duplicated across backends
4. **Maintenance burden**: Changes to backend sessions didn't automatically apply to pooled sessions

### Root Cause

The pooled session tried to be a `BaseSession` subclass while also managing pooled containers, mixing two responsibilities.

## Solution: Composition Over Inheritance

### New Architecture

The redesigned architecture uses **composition** to cleanly separate concerns:

```
┌─────────────────────────────────────────────────────────────┐
│                     Pool Managers                           │
│  (DockerPoolManager, KubernetesPoolManager, PodmanPoolManager)│
│                                                             │
│  - Create and manage pool of containers                    │
│  - Use backend sessions to CREATE new containers          │
│  - Handle health checks and lifecycle management          │
└─────────────────────────────────────────────────────────────┘
                          │
                          │ provides containers to
                          ▼
┌─────────────────────────────────────────────────────────────┐
│                 PooledSandboxSession                        │
│                                                             │
│  - Acquires container from pool                            │
│  - Creates backend session connected via container_id     │
│  - Delegates ALL operations to backend session            │
│  - Returns container to pool when done                     │
└─────────────────────────────────────────────────────────────┘
                          │
                          │ delegates to
                          ▼
┌─────────────────────────────────────────────────────────────┐
│              Backend-Specific Sessions                      │
│     (SandboxDockerSession, SandboxKubernetesSession, etc.)  │
│                                                             │
│  - Handle all backend-specific operations                  │
│  - Process command output correctly                        │
│  - Support all features (stream, verbose, etc.)           │
│  - Connect to existing containers via container_id         │
└─────────────────────────────────────────────────────────────┘
```

### Key Components

#### 1. Pool Managers (Unchanged)

Pool managers remain responsible for:
- Creating containers using backend sessions
- Managing container lifecycle (health checks, recycling)
- Thread-safe acquisition and release

#### 2. PooledSandboxSession (Redesigned)

**Old approach** (Inheritance):
```python
class PooledSandboxSession(BaseSession):
    def _process_non_stream_output(self, output):
        # Duplicated logic, incomplete implementation
        if isinstance(output, bytes):
            return output.decode("utf-8"), ""
        return str(output), ""
```

**New approach** (Composition):
```python
class PooledSandboxSession:
    def __init__(self, ...):
        self._backend_session = None  # Created on open()

    def open(self):
        # Acquire container from pool
        container = self._pool_manager.acquire()

        # Create backend session connected to pooled container
        self._backend_session = SandboxDockerSession(
            container_id=container.container_id,  # Connect to existing
            skip_environment_setup=True,          # Already set up by pool
            ...
        )
        self._backend_session.open()

    def run(self, code, ...):
        # Delegate to backend session
        return self._backend_session.run(code, ...)
```

### Benefits

1. **Zero Code Duplication**
   - All backend logic stays in backend sessions
   - Pooled sessions simply delegate operations

2. **Automatic Feature Support**
   - New features in backend sessions work immediately with pooling
   - Bug fixes in backends automatically apply to pooled sessions

3. **Full Parameter Support**
   - All parameters (`stream`, `verbose`, `runtime_configs`, etc.) pass through
   - No need to maintain separate parameter lists

4. **Correct Output Handling**
   - Docker's `demux=True` tuple output handled correctly
   - Kubernetes, Podman outputs handled correctly
   - No special cases needed in pooled session

5. **Easier Maintenance**
   - Changes only need to be made in backend sessions
   - Pooled sessions stay simple and focused

## Implementation Details

### Opening a Pooled Session

```python
def open(self):
    # 1. Acquire pre-created container from pool
    self._pooled_container = self._pool_manager.acquire()

    # 2. Create backend session connected to it
    self._backend_session = self._create_backend_session(
        self._pooled_container.container_id
    )

    # 3. Open backend session (connects, doesn't create)
    self._backend_session.open()
```

### Closing a Pooled Session

```python
def close(self):
    # 1. Close backend session (disconnects, doesn't destroy)
    if self._backend_session:
        self._backend_session.close()

    # 2. Return container to pool for reuse
    if self._pooled_container:
        self._pool_manager.release(self._pooled_container)
```

### Creating Backend Sessions

For each backend, we extract parameters and create the appropriate session:

```python
def _create_backend_session(self, container_id):
    match self.backend:
        case SandboxBackend.DOCKER:
            # Extract client to avoid duplicate parameter
            session_kwargs = self._session_kwargs.copy()
            client = session_kwargs.pop("client", None) or self._pool_manager.client

            return SandboxDockerSession(
                client=client,
                container_id=container_id,      # Connect to existing
                skip_environment_setup=True,    # Pool already set up
                **session_kwargs,
            )
```

## Migration Guide

### For Users

The API is now simpler and more explicit: create a pool manager first, then pass it via `pool=`.

```python
from llm_sandbox import SandboxSession
from llm_sandbox.pool import create_pool_manager, PoolConfig

# 1) Create shared pool
pool = create_pool_manager(
    backend="docker",
    config=PoolConfig(max_pool_size=5, min_pool_size=2, enable_prewarming=True),
    lang="python",
)

# 2) Use pool in sessions
with SandboxSession(lang="python", pool=pool) as session:
    result = session.run(code)

# 3) Cleanup
pool.close()
```

### For Contributors

When adding new features:

1. **Add to backend sessions** (e.g., `SandboxDockerSession`)
2. **Automatically available in pooled sessions** - no additional work needed!

Example:
```python
# Add new method to SandboxDockerSession
def my_new_feature(self):
    ...

# Automatically available via delegation when using a pool
pool = create_pool_manager(backend="docker", lang="python")
with SandboxSession(lang="python", pool=pool) as session:
    session.my_new_feature()  # Works via __getattr__ to backend session
pool.close()
```

## Testing

### Verification

The redesigned implementation was tested with:

1. **Basic pooling**: Container acquisition, execution, and release
2. **Output handling**: Verified correct stdout/stderr processing
3. **Parameter passing**: All parameters (stream, verbose, client) work correctly
4. **Delegation**: All methods delegate correctly to backend session

### Test Results

```
✅ Container acquired from pool
✅ Code executed successfully
✅ Output processed correctly (not as tuple)
✅ Container returned to pool
✅ No code duplication
✅ Clean architecture
```

## Future Improvements

1. **Type Hints**: Add proper type hints for `_backend_session`
2. **Protocol**: Define a `BackendSession` protocol for better type safety
3. **Testing**: Add comprehensive unit tests for delegation behavior

## Conclusion

The composition-based architecture:
- ✅ Eliminates code duplication
- ✅ Simplifies maintenance
- ✅ Ensures compatibility with all backend features
- ✅ Provides a clean, understandable design

This redesign makes the codebase more maintainable and sets a good foundation for future enhancements.
