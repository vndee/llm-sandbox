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

### Environment Setup Control

LLM Sandbox automatically sets up language-specific environments during container initialization (e.g., creating Python virtual environments, upgrading pip, initializing Go modules). For production deployments or when using custom pre-configured images, you can skip this setup for faster startup times.

#### skip_environment_setup Parameter

The `skip_environment_setup` parameter allows you to bypass automatic environment setup:

```python
from llm_sandbox import SandboxSession, SandboxBackend

# Skip environment setup for faster container startup
with SandboxSession(
    lang="python",
    skip_environment_setup=True,  # Skip pip upgrades and venv creation
    verbose=True
) as session:
    result = session.run("print('Hello from pre-configured environment!')")
```

#### When to Use skip_environment_setup=True

**✅ Recommended for:**

- **Production deployments** where container startup time is critical
- **Custom images** with pre-installed packages and configured environments
- **CI/CD pipelines** where environment setup adds unnecessary overhead
- **Air-gapped environments** where external package repositories aren't accessible
- **Batch processing** where you want predictable, pre-configured setups

**❌ Not recommended for:**

- Development and testing with dynamic package installation
- Using base images without pre-configured language environments
- Scenarios requiring on-the-fly library installation

#### Production Deployment Examples

**Docker with custom image:**
```python
with SandboxSession(
    lang="python",
    backend=SandboxBackend.DOCKER,
    skip_environment_setup=True,
    image="my-registry.com/python-ml:latest",  # Pre-installed ML packages
) as session:
    result = session.run("import numpy as np; print(f'NumPy: {np.__version__}')")
```

**Kubernetes:**
```python
with SandboxSession(
    lang="python",
    backend=SandboxBackend.KUBERNETES,
    skip_environment_setup=True,
    image="my-registry.com/python-ml:latest",
) as session:
    result = session.run("import pandas as pd; print(f'Pandas: {pd.__version__}')")
```

**Podman for rootless containers:**
```python
with SandboxSession(
    lang="python",
    backend=SandboxBackend.PODMAN,
    skip_environment_setup=True,
    image="my-registry.com/python-secure:latest"
) as session:
    result = session.run("print('Secure environment ready!')")
```

#### Custom Image Requirements

When using `skip_environment_setup=True`, ensure your custom image includes:

**For Python:**

- Python interpreter in expected location (usually `/usr/bin/python` or `/usr/local/bin/python`)
- Required packages pre-installed (numpy, pandas, etc.)
- Proper PATH configuration

**For Other Languages:**

- Language runtime properly installed and configured
- Standard libraries and common packages available
- Appropriate environment variables set

#### Library Installation Behavior

When `skip_environment_setup=True`:

- ✅ Code execution works normally
- ❌ Dynamic library installation is disabled
- 📦 Libraries must be pre-installed in the container image

```python
# This will fail with skip_environment_setup=True
result = session.run(
    "import requests; print('OK')",
    libraries=["requests"]  # ❌ Will raise LibraryInstallationNotSupportedError
)

# Instead, use execute_command for manual installation if needed
session.execute_command("pip install requests")
result = session.run("import requests; print('OK')")  # ✅ Works
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
runtime_configs = {
    "privileged": False,
    "mem_limit": "512m",  # Memory limit (use mem_limit, not memory)
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
    runtime_configs=runtime_configs
)
```

**Docker Documentation**: See the [Docker SDK for Python documentation](https://docker-py.readthedocs.io/) for complete API reference and available parameters.

#### Podman Runtime Config
```python
# Podman-specific runtime configuration
runtime_config = {
    "privileged": False,
    "mem_limit": "512m",  # Memory limit (use mem_limit, not memory)
    "cpu_shares": 512,
    "network_mode": "bridge",
    "volumes": {"/host/path": {"bind": "/container/path", "mode": "ro"}},
    "environment": {"PYTHONPATH": "/app"},
    "working_dir": "/workspace"
}

session = SandboxSession(
    image="python:3.9",
    backend="podman",
    runtime_configs=runtime_configs
)
```

**Podman Documentation**: See the [Podman Python SDK documentation](https://podman-py.readthedocs.io/) for complete API reference and available parameters.

### Kubernetes Backend

For Kubernetes, runtime configurations are **not supported** through the `runtime_configs` parameter. Instead, users should define their requirements as **Kubernetes Pod manifests** using the `pod_manifest` parameter.

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

#### ⚠️ Critical Pod Manifest Requirements

When providing custom pod manifests, these configurations are **mandatory** for proper operation:

**Required Container Configurations:**
```python
{
    "name": "my-container",       # Can be any valid container name
    "image": "your-image:latest",
    "tty": True,                  # REQUIRED: Keeps container alive for command execution
    "securityContext": {          # REQUIRED: For proper file permissions
        "runAsUser": 0,
        "runAsGroup": 0,
    },
    # Your other settings...
}
```

**Required Pod-Level Configuration:**
```python
{
    "spec": {
        "containers": [...],
        "securityContext": {      # REQUIRED: Pod-level security context
            "runAsUser": 0,
            "runAsGroup": 0,
        },
    }
}
```

**⚠️ Common Issues:**

- **Pod exits immediately**: Missing `"tty": True` configuration
- **Permission denied errors**: Missing or incorrect `securityContext` configurations
- **Connection timeouts**: Pod may not be fully  ready - ensure proper resource limits and image availability
#### Additional Kubernetes Configuration

To configure resources, security context, volumes, and other Pod-level settings in Kubernetes, you should:

1. Create a Pod manifest file or use the Kubernetes Python client directly
2. Apply resource limits, security policies, and other configurations through Kubernetes APIs
3. Use ConfigMaps and Secrets for environment configuration

**Kubernetes Documentation**: See the [Kubernetes Python Client documentation](https://kubernetes.readthedocs.io/) and [Kubernetes API reference](https://kubernetes.io/docs/reference/) for Pod configuration options.

## Example Runtime Configurations

### Resource Limits (Docker/Podman)

```python
# Memory and CPU limits
runtime_configs = {
    "mem_limit": "1g",        # 1GB memory limit (use mem_limit, not memory)
    "cpu_period": 100000,     # CPU period in microseconds
    "cpu_quota": 50000,       # CPU quota (50% of one CPU)
    "memswap_limit": "2g"     # Memory + swap limit
}
```

### Network Configuration (Docker/Podman)

```python
# Custom network settings
runtime_configs = {
    "network_mode": "host",        # Use host networking
    "ports": {"8080/tcp": 8080},   # Port mapping
    "dns": ["8.8.8.8", "8.8.4.4"] # Custom DNS servers
}
```

### Volume Mounts (Docker/Podman)

```python
# Volume mounting
runtime_configs = {
    "volumes": {
        "/host/data": {"bind": "/data", "mode": "rw"},
        "/host/config": {"bind": "/config", "mode": "ro"}
    }
}
```

### Environment Variables (Docker/Podman)

```python
# Environment configuration
runtime_configs = {
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
runtime_configs = {
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

### Security Presets

LLM Sandbox provides predefined security configurations that combine `SecurityPolicy` (code-level analysis) with backend-specific `runtime_configs` (container-level security). These presets offer ready-to-use security configurations for common scenarios.

#### Available Presets

- **development**: Permissive configuration for local development and testing
- **production**: Strict configuration for production applications
- **strict**: Very strict configuration for untrusted code execution
- **educational**: Balanced configuration for educational platforms

#### Basic Usage

```python
from llm_sandbox import SandboxSession, get_security_preset

# Get a complete security configuration
config = get_security_preset("production", "python", "docker")

# Use it in a session
with SandboxSession(
    lang="python",
    security_policy=config.security_policy,
    runtime_configs=config.runtime_config
) as session:
    result = session.run("print('Hello, World!')")
```

#### Preset Comparison

| Preset | Severity Threshold | Network | Memory | Read-Only | User |
|--------|-------------------|---------|--------|-----------|------|
| **development** | HIGH | bridge (allowed) | 1GB | No | root |
| **production** | MEDIUM | none (disabled) | 512MB | Yes | nobody:nogroup |
| **strict** | LOW | none (disabled) | 128MB | Yes | nobody:nogroup |
| **educational** | MEDIUM | bridge (allowed) | 256MB | Yes | 1000:1000 |

#### Customizing Presets

You can start with a preset and customize it for your needs:

```python
from llm_sandbox import get_security_preset, SecurityPattern, SecurityIssueSeverity

# Start with production preset
config = get_security_preset("production", "python", "docker")

# Modify runtime configuration
config.runtime_config["mem_limit"] = "1g"  # Increase memory limit
config.runtime_config["network_mode"] = "bridge"  # Enable network

# Add custom security patterns
custom_pattern = SecurityPattern(
    pattern=r"\bpandas\.",
    description="Pandas library usage",
    severity=SecurityIssueSeverity.LOW
)
config.security_policy.add_pattern(custom_pattern)

# Use the customized configuration
with SandboxSession(
    lang="python",
    security_policy=config.security_policy,
    runtime_configs=config.runtime_config
) as session:
    pass
```

#### Backend-Specific Configurations

**Docker and Podman**: Security presets include complete runtime configurations with resource limits, network settings, user permissions, and security options.

```python
# Docker/Podman example
config = get_security_preset("strict", "python", "docker")
print(config.runtime_config)
# {
#     "mem_limit": "128m",
#     "cpu_period": 100000,
#     "cpu_quota": 50000,
#     "network_mode": "none",
#     "read_only": True,
#     "tmpfs": {"/tmp": "size=50m,noexec,nosuid,nodev"},
#     "user": "nobody:nogroup",
#     "cap_drop": ["ALL"],
#     "security_opt": ["no-new-privileges:true"]
# }
```

**Kubernetes**: Security presets only include SecurityPolicy. You must provide your own pod manifest with appropriate security context and resource limits.

```python
# Kubernetes example
config = get_security_preset("production", "python", "kubernetes")
# config.runtime_config is None for Kubernetes
# You must provide pod_manifest parameter separately
```

#### Complete Example

```python
from llm_sandbox import SandboxSession, get_security_preset

# List available presets
from llm_sandbox import list_available_presets
print("Available presets:", list_available_presets())
# ['development', 'production', 'strict', 'educational']

# Use strict preset for untrusted code
config = get_security_preset("strict", "python", "docker")

with SandboxSession(
    lang="python",
    security_policy=config.security_policy,
    runtime_configs=config.runtime_config
) as session:
    # Check code safety before execution
    code = "import os; os.system('ls')"
    is_safe, violations = session.is_safe(code)
    
    if not is_safe:
        print("Code blocked by security policy!")
        for violation in violations:
            print(f"  - {violation.description}")
    else:
        result = session.run(code)
```

For more examples, see `examples/security_presets_demo.py` in the repository.

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
