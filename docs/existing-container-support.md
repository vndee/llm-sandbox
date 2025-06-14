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

### Micromamba Backend

```python
from llm_sandbox import SandboxSession, SandboxBackend

# Connect to existing Micromamba container
with SandboxSession(
    backend=SandboxBackend.MICROMAMBA,
    container_id='micromamba-container-id',
    environment="myenv",  # Specify environment
    lang="python"
) as session:
    result = session.run("print('Hello from existing Micromamba container!')")
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
import docker
from llm_sandbox import SandboxSession

# Create a container with custom setup (one-time)
client = docker.from_env()
container = client.containers.run(
    "ghcr.io/vndee/sandbox-python-311-bullseye",
    detach=True,
    tty=True,
    command="tail -f /dev/null"  # Keep running
)

# Install packages and setup environment
container.exec_run("pip install numpy pandas matplotlib")
container.exec_run("mkdir -p /my-workspace")

try:
    # Now use the existing container multiple times
    for i in range(3):
        with SandboxSession(
            container_id=container.id,
            lang="python",
            verbose=True
        ) as session:
            result = session.run(f"""
import numpy as np
print(f"Iteration {i+1}")
print(f"NumPy version: {np.__version__}")
data = np.random.rand(5)
print(f"Random data: {data}")
""")
            print(f"Run {i+1} output:")
            print(result.stdout)
            print("-" * 40)

finally:
    # Clean up
    container.stop()
    container.remove()
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

### Micromamba
- Inherits Docker behavior
- Commands are wrapped with `micromamba run -n <environment>`
- Specify environment via `environment` parameter

## Best Practices

1. **Ensure Environment Readiness**: Make sure containers have required interpreters/compilers
2. **Handle Connection Errors**: Always wrap in try/catch for `ContainerError`
3. **Document Dependencies**: Clearly document what your existing containers need
4. **Test Connectivity**: Verify container is accessible before production use
5. **Monitor Resources**: Existing containers may have different resource constraints

## Migration from Creating New Containers

**Before (creating new):**
```python
with SandboxSession(lang="python", image="my-image") as session:
    session.install(["numpy"])  # Environment setup
    result = session.run("import numpy; print('ready')")
```

**After (using existing):**
```python
# Container must already have numpy installed
with SandboxSession(container_id="my-container", lang="python") as session:
    result = session.run("import numpy; print('ready')")  # Direct usage
```
