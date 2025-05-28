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

### Language Options

Supported languages and their identifiers:

| Language | Identifier | Default Image |
|----------|------------|---------------|
| Python | `python` | `python:3.11-bullseye` |
| JavaScript | `javascript` | `node:22-bullseye` |
| Java | `java` | `openjdk:11.0.12-jdk-bullseye` |
| C++ | `cpp` | `gcc:11.2.0-bullseye` |
| Go | `go` | `golang:1.23.4-bullseye` |
| Ruby | `ruby` | `ruby:3.0.2-bullseye` |

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

### Resource Limits

```python
with SandboxSession(
    lang="python",
    runtime_configs={
        # CPU limits
        "cpu_count": 2,          # Number of CPUs
        "cpu_shares": 1024,      # CPU shares (relative weight)
        "cpu_period": 100000,    # CPU CFS period in microseconds
        "cpu_quota": 50000,      # CPU CFS quota in microseconds
        
        # Memory limits
        "mem_limit": "512m",     # Memory limit (e.g., "512m", "1g")
        "memswap_limit": "1g",   # Memory + swap limit
        
        # Other limits
        "pids_limit": 100,       # Maximum number of PIDs
        "timeout": 30,           # Execution timeout in seconds
    }
) as session:
    pass
```

### User and Permissions

```python
# Run as non-root user
with SandboxSession(
    lang="python",
    runtime_configs={
        "user": "1000:1000",     # UID:GID
    },
    workdir="/tmp/sandbox"      # Use writable directory for non-root
) as session:
    pass

# Run with specific capabilities
with SandboxSession(
    lang="python",
    runtime_configs={
        "cap_add": ["SYS_PTRACE"],  # Add capabilities
        "cap_drop": ["NET_RAW"],    # Drop capabilities
    }
) as session:
    pass
```

### Environment Variables

```python
# Docker/Podman backend
with SandboxSession(
    lang="python",
    runtime_configs={
        "environment": {
            "API_KEY": "secret",
            "DEBUG": "true",
            "CUSTOM_VAR": "value"
        }
    }
) as session:
    result = session.run("import os; print(os.environ.get('API_KEY'))")

# Kubernetes backend
with SandboxSession(
    backend=SandboxBackend.KUBERNETES,
    lang="python",
    env_vars={
        "API_KEY": "secret",
        "DEBUG": "true"
    }
) as session:
    pass
```

### Volume Mounts

```python
from docker.types import Mount

# Docker/Podman mounts
with SandboxSession(
    lang="python",
    mounts=[
        Mount(
            type="bind",
            source="/host/data",
            target="/sandbox/data",
            read_only=True
        ),
        Mount(
            type="volume",
            source="myvolume",
            target="/sandbox/cache"
        )
    ]
) as session:
    pass
```

## Security Configuration

### Security Policies

```python
from llm_sandbox.security import (
    SecurityPolicy, 
    SecurityPattern, 
    DangerousModule,
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
        DangerousModule(
            name="subprocess",
            description="Process execution",
            severity=SecurityIssueSeverity.HIGH
        )
    ]
)

with SandboxSession(lang="python", security_policy=policy) as session:
    pass
```

### Using Preset Policies

```python
from llm_sandbox.security import get_security_policy

# Available presets
presets = [
    "minimal",      # Very permissive
    "development",  # Balanced for development
    "educational",  # For teaching environments  
    "production",   # Strict for production
    "strict",       # Very restrictive
    "data_science", # Optimized for data analysis
    "web_scraping"  # For web scraping tasks
]

# Use a preset
policy = get_security_policy("production")
with SandboxSession(lang="python", security_policy=policy) as session:
    pass
```

## Backend-Specific Configuration

### Docker Configuration

```python
import docker

# Use custom Docker client
client = docker.DockerClient(
    base_url="tcp://docker-host:2375",
    timeout=30
)

with SandboxSession(
    backend=SandboxBackend.DOCKER,
    client=client,
    lang="python",
    stream=True,  # Stream command output
    commit_container=True  # Save container state as new image
) as session:
    pass
```

### Kubernetes Configuration

```python
from kubernetes import client, config

# Load custom kubeconfig
config.load_kube_config(config_file="~/.kube/custom-config")
k8s_client = client.CoreV1Api()

with SandboxSession(
    backend=SandboxBackend.KUBERNETES,
    client=k8s_client,
    lang="python",
    kube_namespace="sandbox-namespace",
    pod_manifest={
        "apiVersion": "v1",
        "kind": "Pod",
        "metadata": {
            "name": "custom-sandbox",
            "labels": {"app": "sandbox"}
        },
        "spec": {
            "containers": [{
                "name": "sandbox",
                "image": "python:3.11",
                "resources": {
                    "limits": {
                        "memory": "512Mi",
                        "cpu": "500m"
                    }
                }
            }]
        }
    }
) as session:
    pass
```

### Podman Configuration

```python
from podman import PodmanClient

# Custom Podman client
client = PodmanClient(
    base_url="unix:///run/user/1000/podman/podman.sock",
    timeout=30
)

with SandboxSession(
    backend=SandboxBackend.PODMAN,
    client=client,
    lang="python",
    runtime_configs={
        "userns_mode": "keep-id",  # User namespace mode
        "security_opt": ["no-new-privileges"]
    }
) as session:
    pass
```

### Micromamba Configuration

```python
with SandboxSession(
    backend=SandboxBackend.MICROMAMBA,
    lang="python",
    image="mambaorg/micromamba:latest",
    environment="myenv"  # Conda environment name
) as session:
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
    "timeout": 60
}

# Production environment
prod_config = {
    "cpu_count": 1,
    "mem_limit": "256m",
    "timeout": 10,
    "pids_limit": 50
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

## Environment Variables Reference

LLM Sandbox respects these environment variables:

| Variable | Description | Default |
|----------|-------------|---------||
| `DOCKER_HOST` | Docker daemon socket | `unix:///var/run/docker.sock` |
| `KUBECONFIG` | Kubernetes config file | `~/.kube/config` |
| `PODMAN_SOCKET` | Podman socket path | System default |

## Next Steps

- Learn about [Security Policies](security.md)
- Explore [Backend Options](backends.md)
- Check out [Examples](examples.md)
- Read the [API Reference](api-reference.md)