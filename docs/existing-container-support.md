# Existing Container Support

This document describes the new feature for connecting to existing containers/pods instead of creating new ones.

## Overview

LLM Sandbox now supports connecting to existing containers/pods across all backends (Docker, Kubernetes, Podman, Micromamba). This is useful for:

- **Reusing containers with complex setups**: Connect to containers that already have your specific environment configured
- **Working with long-running services**: Integrate with containers managed by external systems
- **Debugging and troubleshooting**: Connect to running containers to debug issues
- **Performance optimization**: Skip container creation and environment setup time

## Usage

### Basic Usage

```python
from llm_sandbox import SandboxSession

# Connect to existing Docker container
with SandboxSession(container_id='abc123def456', lang="python") as session:
    result = session.run("print('Hello from existing container!')")
    print(result.stdout)
```

### Docker Backend

```python
from llm_sandbox import SandboxSession

# Connect to existing Docker container
with SandboxSession(
    container_id='your-container-id',
    lang="python",
    verbose=True
) as session:
    # Run code in existing container
    result = session.run("import sys; print(sys.version)")

    # Install additional libraries
    session.install(["numpy"])

    # Execute commands
    result = session.execute_command("ls -la")

    # Copy files
    session.copy_to_runtime("local_file.py", "/container/path/file.py")
```

### Kubernetes Backend

```python
from llm_sandbox import SandboxSession, SandboxBackend

# Connect to existing Kubernetes pod
with SandboxSession(
    backend=SandboxBackend.KUBERNETES,
    container_id='my-pod-name',  # Pod name
    lang="python",
    verbose=True
) as session:
    result = session.run("print('Hello from existing pod!')")
```

### Podman Backend

```python
from llm_sandbox import SandboxSession, SandboxBackend
from podman import PodmanClient

client = PodmanClient()
with SandboxSession(
    backend=SandboxBackend.PODMAN,
    client=client,
    container_id='podman-container-id',
    lang="python"
) as session:
    result = session.run("print('Hello from existing Podman container!')")
```

## Important Notes

### Environment Setup Skipped

When using `container_id`, the sandbox **skips environment setup**. This means:

- No virtual environment creation (for Python)
- No package manager initialization
- No working directory setup
- **You must ensure the container has the proper environment and tools for your language**

### Container Management

- **Existing containers are not stopped/removed** when the session ends
- Only the connection is closed
- The container continues running after session closure
- Timeout cleanup will kill the container process but won't remove existing containers

### Error Handling

```python
from llm_sandbox import SandboxSession, ContainerError

try:
    with SandboxSession(container_id='non-existent', lang="python") as session:
        session.run("print('test')")
except ContainerError as e:
    print(f"Failed to connect: {e}")
```

## Configuration Constraints

When using `container_id`:

- Cannot use `dockerfile` parameter (validation error)
- `image` parameter is ignored
- Environment setup is skipped
- Container lifecycle management is limited

## Complete Example

```python
--8<-- "examples/existing_container_demo.py"
```

## Backend-Specific Notes

### Docker
- Uses container ID or name
- Supports both running and stopped containers (will start if stopped)
- Full Docker API compatibility

### Kubernetes
- Uses pod name as `container_id`
- Must specify correct namespace via `kube_namespace` parameter
- Pod must be in "Running" state
- Will wait briefly for "Pending" pods to start

### Podman
- Uses container ID or name
- Compatible with Docker API
- Supports both running and stopped containers

## Best Practices

1. **Ensure Environment Readiness**: Make sure containers have required interpreters/compilers
2. **Handle Connection Errors**: Always wrap in try/catch for `ContainerError`
3. **Document Dependencies**: Clearly document what your existing containers need
4. **Test Connectivity**: Verify container is accessible before production use
5. **Monitor Resources**: Existing containers may have different resource constraints
