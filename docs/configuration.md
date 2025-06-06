# Configuration Guide

This guide covers all configuration options available in LLM Sandbox.

## Session Configuration

### Basic Parameters

```python
from llm_sandbox import SandboxSession

session = SandboxSession(
    lang="python",              # Programming language
    verbose=True,               # Enable verbose logging
    keep_template=False,        # Keep container image after session
    workdir="/sandbox",         # Working directory in container
)
```

### Timeout Configuration

LLM Sandbox provides comprehensive timeout controls to prevent runaway code execution and manage resource usage efficiently.

#### Timeout Types

There are three types of timeouts you can configure:

| Timeout Type | Description | Default |
|--------------|-------------|---------|
| `default_timeout` | Default timeout for all operations | 30.0 seconds |
| `execution_timeout` | Timeout for code execution (per run) | Uses `default_timeout` |
| `session_timeout` | Maximum session lifetime | None (unlimited) |

#### Basic Timeout Configuration

```python
from llm_sandbox import SandboxSession

# Configure timeouts at session creation
with SandboxSession(
    lang="python",
    default_timeout=30.0,      # Default timeout for operations
    execution_timeout=60.0,    # Timeout for code execution
    session_timeout=300.0,     # Session expires after 5 minutes
    verbose=True
) as session:
    # Fast operation - should complete
    result = session.run("""
print("Hello, World!")
import time
time.sleep(2)
print("Operation completed")
    """)
```

#### Per-Execution Timeout Override

You can override the execution timeout for individual `run()` calls:

```python
with SandboxSession(lang="python", execution_timeout=10.0) as session:
    # This will use the session's execution_timeout (10 seconds)
    result1 = session.run("print('Normal execution')")

    # Override with a longer timeout for this specific execution
    result2 = session.run("""
import time
time.sleep(15)  # This needs more time
print("Long operation completed")
    """, timeout=20.0)  # Override with 20 seconds

    # Override with a shorter timeout
    try:
        session.run("""
import time
time.sleep(5)
print("This might timeout")
        """, timeout=2.0)  # Override with 2 seconds
    except SandboxTimeoutError:
        print("Operation timed out as expected")
```

#### Timeout Error Handling

```python
from llm_sandbox.exceptions import SandboxTimeoutError

def execute_with_retry(session, code, max_retries=3):
    """Execute code with automatic retry on timeout."""
    for attempt in range(max_retries):
        try:
            return session.run(code, timeout=10.0)
        except SandboxTimeoutError:
            print(f"Attempt {attempt + 1} timed out")
            if attempt == max_retries - 1:
                raise
            print("Retrying...")

# Usage example
with SandboxSession(lang="python") as session:
    try:
        result = execute_with_retry(session, """
import time
import random
time.sleep(random.uniform(5, 15))  # Variable execution time
print("Completed!")
        """)
        print(result.stdout)
    except SandboxTimeoutError:
        print("All retry attempts failed")
```

#### Backend-Specific Timeout Behavior

Different backends handle timeouts differently:

##### Docker & Podman
- Containers are forcefully killed when timeout is reached
- Container cleanup is automatic
- Resource usage monitoring during execution

##### Kubernetes
- Pods are monitored for timeout during command execution
- Timeout applies to individual command execution within the pod
- Pod lifecycle is managed independently

#### Advanced Timeout Scenarios

##### Infinite Loop Protection
```python
with SandboxSession(lang="python", execution_timeout=5.0) as session:
    try:
        session.run("""
# This infinite loop will be terminated
i = 0
while True:
    i += 1
    if i % 100000 == 0:
        print(f"Iteration: {i}")
        """)
    except SandboxTimeoutError:
        print("Infinite loop was terminated by timeout")
```

##### Resource-Intensive Operation Control
```python
with SandboxSession(lang="python") as session:
    try:
        session.run("""
# CPU-intensive operation
total = 0
for i in range(10**8):  # Large computation
    total += i * i
print(f"Result: {total}")
        """, timeout=30.0)  # Give enough time for legitimate computation
    except SandboxTimeoutError:
        print("Computation took too long and was terminated")
```

##### Session Lifetime Management
```python
import time

# Session that automatically expires
with SandboxSession(
    lang="python",
    session_timeout=60.0  # Session expires after 1 minute
) as session:

    # This will work
    session.run("print('First execution')")

    # Wait and try again
    time.sleep(30)
    session.run("print('Second execution')")

    # This might fail if session has expired
    time.sleep(40)  # Total elapsed: 70 seconds
    try:
        session.run("print('This might fail')")
    except SandboxTimeoutError:
        print("Session expired")
```

#### Best Practices

1. **Set Appropriate Timeouts**: Balance between allowing legitimate long operations and preventing runaway code
2. **Use Per-Execution Overrides**: Override timeouts for known long-running operations
3. **Implement Retry Logic**: Handle timeout errors gracefully with retry mechanisms
4. **Monitor Resource Usage**: Use timeout in combination with resource limits
5. **Log Timeout Events**: Enable verbose logging to understand timeout patterns

### Language Options

Supported languages and their identifiers:

| Language | Identifier | Default Image |
|----------|------------|---------------|
| Python | `python` | `ghcr.io/vndee/sandbox-python-311-bullseye` |
| JavaScript | `javascript` | `ghcr.io/vndee/sandbox-node-22-bullseye` |
| Java | `java` | `ghcr.io/vndee/sandbox-java-11-bullseye` |
| C++ | `cpp` | `ghcr.io/vndee/sandbox-cpp-11-bullseye` |
| Go | `go` | `ghcr.io/vndee/sandbox-go-123-bullseye` |
| Ruby | `ruby` | `ghcr.io/vndee/sandbox-ruby-302-bullseye` |

### Container Images

#### Using Default Images

```python
# Uses default Python image
with SandboxSession(lang="python") as session:
    pass
```

#### Using Custom Images

```python
# Use a specific image
with SandboxSession(
    lang="python",
    image="python:3.12-slim"
) as session:
    pass

# Use your own custom image
with SandboxSession(
    lang="python",
    image="myregistry.com/my-python:latest"
) as session:
    pass
```

#### Building from Dockerfile

```python
# Build image from Dockerfile
with SandboxSession(
    lang="python",
    dockerfile="./custom/Dockerfile"
) as session:
    pass
```

Example Dockerfile:
```dockerfile
FROM python:3.11-slim

# Install additional system packages
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Pre-install Python packages
RUN pip install numpy pandas matplotlib

WORKDIR /sandbox
```

## Runtime Configuration

**Important**: Runtime configurations work differently depending on the backend:

### Docker and Podman Backends

For Docker and Podman backends, runtime configuration options are passed as **extra arguments** to the respective Python libraries (`docker-py` and `podman-py`). These are used to configure container creation and execution parameters.

#### Docker Runtime Config
```python
# Docker-specific runtime configuration
runtime_config = {
    "privileged": False,
    "memory": "512m",
    "cpu_period": 100000,
    "cpu_quota": 50000,
    "network_mode": "bridge",
    "volumes": {"/host/path": {"bind": "/container/path", "mode": "ro"}},
    "environment": {"PYTHONPATH": "/app"},
    "working_dir": "/workspace"
}

session = SandboxSession(
    image="python:3.9",
    backend="docker",
    runtime_config=runtime_config
)
```

**Docker Documentation**: See the [Docker SDK for Python documentation](https://docker-py.readthedocs.io/) for complete API reference and available parameters.

#### Podman Runtime Config
```python
# Podman-specific runtime configuration
runtime_config = {
    "privileged": False,
    "memory": "512m",
    "cpu_shares": 512,
    "network_mode": "bridge",
    "volumes": {"/host/path": {"bind": "/container/path", "mode": "ro"}},
    "environment": {"PYTHONPATH": "/app"},
    "working_dir": "/workspace"
}

session = SandboxSession(
    image="python:3.9",
    backend="podman",
    runtime_config=runtime_config
)
```

**Podman Documentation**: See the [Podman Python SDK documentation](https://podman-py.readthedocs.io/) for complete API reference and available parameters.

### Kubernetes Backend

For Kubernetes, runtime configurations are **not supported** through the `runtime_config` parameter. Instead, users should define their requirements as **Kubernetes Pod manifests** using the `pod_manifest` parameter.

```python
# Kubernetes configuration using pod_manifest parameter
import uuid

# Generate unique pod name to avoid conflicts
unique_suffix = str(uuid.uuid4())[:8]
pod_name = f"sandbox-{unique_suffix}"

# Define custom pod manifest
pod_manifest = {
    "apiVersion": "v1",
    "kind": "Pod",
    "metadata": {
        "name": pod_name,
        "labels": {"app": "sandbox"},
    },
    "spec": {
        "containers": [
            {
                "name": "sandbox-container",
                "image": "python:3.9",
                "tty": True,
                "securityContext": {
                    "runAsUser": 1000,
                    "runAsGroup": 1000,
                },
                "resources": {
                    "limits": {
                        "memory": "512Mi",
                        "cpu": "500m"
                    },
                    "requests": {
                        "memory": "256Mi",
                        "cpu": "250m"
                    }
                }
            }
        ],
        "securityContext": {
            "runAsUser": 1000,
            "runAsGroup": 1000,
        },
    },
}

session = SandboxSession(
    backend="kubernetes",
    lang="python",
    image="python:3.9",
    pod_manifest=pod_manifest,
    workdir="/tmp/sandbox"  # Use writable directory for non-root
)
```

To configure resources, security context, volumes, and other Pod-level settings in Kubernetes, you should:

1. Create a Pod manifest file or use the Kubernetes Python client directly
2. Apply resource limits, security policies, and other configurations through Kubernetes APIs
3. Use ConfigMaps and Secrets for environment configuration

**Kubernetes Documentation**: See the [Kubernetes Python Client documentation](https://kubernetes.readthedocs.io/) and [Kubernetes API reference](https://kubernetes.io/docs/reference/) for Pod configuration options.

## Example Runtime Configurations

### Resource Limits (Docker/Podman)

```python
# Memory and CPU limits
runtime_config = {
    "memory": "1g",           # 1GB memory limit
    "cpu_period": 100000,     # CPU period in microseconds
    "cpu_quota": 50000,       # CPU quota (50% of one CPU)
    "memswap_limit": "2g"     # Memory + swap limit
}
```

### Network Configuration (Docker/Podman)

```python
# Custom network settings
runtime_config = {
    "network_mode": "host",        # Use host networking
    "ports": {"8080/tcp": 8080},   # Port mapping
    "dns": ["8.8.8.8", "8.8.4.4"] # Custom DNS servers
}
```

### Volume Mounts (Docker/Podman)

```python
# Volume mounting
runtime_config = {
    "volumes": {
        "/host/data": {"bind": "/data", "mode": "rw"},
        "/host/config": {"bind": "/config", "mode": "ro"}
    }
}
```

### Environment Variables (Docker/Podman)

```python
# Environment configuration
runtime_config = {
    "environment": {
        "PYTHONPATH": "/app:/libs",
        "DEBUG": "true",
        "API_KEY": "your-api-key"
    }
}
```

### Security Configuration (Docker/Podman)

```python
# Security settings
runtime_config = {
    "privileged": False,
    "user": "1000:1000",           # Run as specific user/group
    "cap_drop": ["ALL"],           # Drop all capabilities
    "cap_add": ["NET_ADMIN"],      # Add specific capabilities
    "security_opt": ["no-new-privileges:true"]
}
```

## Backend-Specific Documentation Links

- **Docker**: [Docker SDK for Python](https://docker-py.readthedocs.io/) - Complete API reference for container configuration
- **Podman**: [Podman Python SDK](https://podman-py.readthedocs.io/) - Complete API reference for Podman container management
- **Kubernetes**: [Kubernetes Python Client](https://kubernetes.readthedocs.io/) - Official Kubernetes API client documentation

For detailed parameter lists and advanced configuration options, please refer to the respective documentation links above.

## Security Configuration

### Security Policies

```python
from llm_sandbox.security import (
    SecurityPolicy,
    SecurityPattern,
    RestrictedModule,
    SecurityIssueSeverity
)

# Create custom security policy
policy = SecurityPolicy(
    severity_threshold=SecurityIssueSeverity.MEDIUM,
    patterns=[
        SecurityPattern(
            pattern=r"os\.system",
            description="System command execution",
            severity=SecurityIssueSeverity.HIGH
        ),
        SecurityPattern(
            pattern=r"open\s*\([^)]*['\"][wa]",
            description="File write operation",
            severity=SecurityIssueSeverity.MEDIUM
        )
    ],
    restricted_modules=[
        RestrictedModule(
            name="subprocess",
            description="Process execution",
            severity=SecurityIssueSeverity.HIGH
        )
    ]
)

with SandboxSession(lang="python", security_policy=policy) as session:
    pass
```

For more information, see the [Security Policies](security.md) page.

## Custom Client Configuration

By default, LLM Sandbox uses the standard client initialization for each backend:

- Docker: `docker.from_env()`
- Podman: `PodmanClient()`
- Kubernetes: Loads from `~/.kube/config`

However, you can provide your own client instances to connect to remote servers, custom sockets, or clusters with specific configurations.

### Docker Remote Connection

```python
import docker

# Connect to remote Docker daemon
client = docker.DockerClient(
    base_url="tcp://remote-docker-host:2376",
    tls=True,
    timeout=30
)

with SandboxSession(
    backend="docker",
    client=client,  # Use custom client instead of docker.from_env()
    lang="python"
) as session:
    pass
```

### Podman Custom Socket

```python
from podman import PodmanClient

# Connect to custom Podman socket
client = PodmanClient(
    base_url="unix:///run/user/1000/podman/podman.sock",
    timeout=60
)

with SandboxSession(
    backend="podman",
    client=client,  # Use custom client
    lang="python"
) as session:
    pass
```

### Kubernetes Remote Cluster

```python
from kubernetes import client, config

# Load config from custom file or remote cluster
config.load_kube_config(
    config_file="/path/to/custom/kubeconfig",
    context="remote-cluster-context"
)

# Or configure for remote cluster programmatically
configuration = client.Configuration()
configuration.host = "https://k8s-cluster.example.com:6443"
configuration.api_key_prefix['authorization'] = 'Bearer'
configuration.api_key['authorization'] = 'your-token-here'

k8s_client = client.CoreV1Api(client.ApiClient(configuration))

with SandboxSession(
    backend="kubernetes",
    client=k8s_client,  # Use custom configured client
    lang="python",
    kube_namespace="custom-namespace"
) as session:
    pass
```

### Docker with Custom TLS Configuration

```python
import docker
import ssl

# Docker with custom TLS/SSL settings
tls_config = docker.tls.TLSConfig(
    client_cert=('/path/to/client-cert.pem', '/path/to/client-key.pem'),
    ca_cert='/path/to/ca.pem',
    verify=True
)

client = docker.DockerClient(
    base_url="tcp://secure-docker-host:2376",
    tls=tls_config
)

with SandboxSession(backend="docker", client=client, lang="python") as session:
    pass
```

## Advanced Configuration

### Artifact Extraction

```python
from llm_sandbox import ArtifactSandboxSession

with ArtifactSandboxSession(
    lang="python",
    enable_plotting=True,
    verbose=True
) as session:
    result = session.run("""
import matplotlib.pyplot as plt
plt.plot([1, 2, 3], [1, 4, 9])
plt.show()
    """)

    # Access captured plots
    for plot in result.plots:
        print(f"Plot format: {plot.format}")
        print(f"Plot size: {len(plot.content_base64)} bytes")
```

### Custom Language Handlers

```python
from llm_sandbox.language_handlers import AbstractLanguageHandler
from llm_sandbox.language_handlers.factory import LanguageHandlerFactory

class CustomLanguageHandler(AbstractLanguageHandler):
    def __init__(self, logger=None):
        super().__init__(logger)
        self.config = LanguageConfig(
            name="custom",
            file_extension="custom",
            execution_commands=["custom-runner {file}"],
            package_manager="custom-pm install"
        )

    def get_import_patterns(self, module):
        return rf"import\s+{module}"

    # Implement other required methods...

# Register custom handler
LanguageHandlerFactory.register_handler("custom", CustomLanguageHandler)

# Use custom language
with SandboxSession(lang="custom") as session:
    pass
```

### Logging Configuration

```python
import logging

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# Create logger
logger = logging.getLogger('llm_sandbox')

# Use custom logger
with SandboxSession(
    lang="python",
    logger=logger,
    verbose=True
) as session:
    pass
```

### Session Persistence

```python
# Keep container running for reuse
with SandboxSession(
    lang="python",
    keep_template=True,
    commit_container=True
) as session:
    # Install packages
    session.install(["numpy", "pandas", "scikit-learn"])

    # Run initial setup
    session.run("""
import numpy as np
import pandas as pd
from sklearn import datasets
print("Environment ready!")
    """)

# Container state is saved and can be reused
```

## Configuration Best Practices

### 1. Use Appropriate Resource Limits

```python
# Development environment
dev_config = {
    "cpu_count": 4,
    "mem_limit": "2g",
\}

# Production environment
prod_config = {
    "cpu_count": 1,
    "mem_limit": "256m",
\    "pids_limit": 50
}
```

### 2. Layer Security Policies

```python
# Base policy
base_policy = get_security_policy("production")

# Add custom patterns
base_policy.add_pattern(SecurityPattern(
    pattern=r"requests\.get\s*\(.*internal",
    description="Internal network access",
    severity=SecurityIssueSeverity.HIGH
))
```

### 3. Use Environment-Specific Images

```python
import os

# Environment-based configuration
env = os.getenv("ENVIRONMENT", "development")

configs = {
    "development": {
        "image": "python:3.11",
        "keep_template": True,
        "verbose": True
    },
    "production": {
        "image": "python:3.11-slim",
        "keep_template": False,
        "verbose": False
    }
}

with SandboxSession(**configs[env]) as session:
    pass
```

### 4. Handle Backend Failover

```python
def create_session_with_fallback(**kwargs):
    """Create session with backend fallback"""
    backends = [
        SandboxBackend.DOCKER,
        SandboxBackend.PODMAN,
        SandboxBackend.KUBERNETES
    ]

    for backend in backends:
        try:
            return SandboxSession(backend=backend, **kwargs)
        except Exception as e:
            print(f"Backend {backend} failed: {e}")
            continue

    raise RuntimeError("No available backends")
```

## Next Steps

- Learn about [Security Policies](security.md)
- Explore [Backend Options](backends.md)
- Check out [Examples](examples.md)
- Read the [API Reference](api-reference.md)
