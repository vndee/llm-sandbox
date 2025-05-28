# Container Backends

LLM Sandbox supports multiple container backends to suit different infrastructure needs. This guide covers each backend's features, configuration, and best practices.

## Overview

Supported backends:

| Backend | Use Case | Root Access | Orchestration | Performance |
|---------|----------|-------------|---------------|-------------|
| **Docker** | Development, single-host | Yes | Limited | High |
| **Kubernetes** | Production, scalable | Configurable | Full | High |
| **Podman** | Rootless security | No (rootless) | Limited | High |

## Docker Backend

### Overview

Docker is the default and most widely supported backend. It provides excellent performance and compatibility.

### Installation

```bash
# Install Docker
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh

# Install LLM Sandbox with Docker support
pip install 'llm-sandbox[docker]'
```

### Basic Usage

```python
from llm_sandbox import SandboxSession, SandboxBackend

# Default Docker backend
with SandboxSession(lang="python") as session:
    result = session.run("print('Hello from Docker!')")
    print(result.stdout)

# Explicit Docker backend
with SandboxSession(
    backend=SandboxBackend.DOCKER,
    lang="python"
) as session:
    pass
```

### Custom Docker Client

```python
import docker

# Connect to remote Docker daemon
client = docker.DockerClient(
    base_url='tcp://docker-host:2375',
    version='auto',
    timeout=30
)

with SandboxSession(
    backend=SandboxBackend.DOCKER,
    client=client,
    lang="python"
) as session:
    pass

# Use Docker context
client = docker.DockerClient.from_env()
```

### Docker-Specific Features

#### Container Commit

```python
# Save container state after execution
with SandboxSession(
    backend=SandboxBackend.DOCKER,
    lang="python",
    commit_container=True,
    image="my-base-image:latest"
) as session:
    # Install packages and setup environment
    session.install(["numpy", "pandas", "scikit-learn"])
    session.run("echo 'Environment configured'")
    # Container will be committed as my-base-image:latest
```

#### Volume Mounts

```python
from docker.types import Mount

with SandboxSession(
    backend=SandboxBackend.DOCKER,
    lang="python",
    mounts=[
        # Bind mount
        Mount(
            type="bind",
            source="/host/data",
            target="/container/data",
            read_only=True
        ),
        # Named volume
        Mount(
            type="volume",
            source="myvolume",
            target="/container/cache"
        ),
        # Tmpfs mount
        Mount(
            type="tmpfs",
            target="/container/tmp",
            tmpfs_size="100m"
        )
    ]
) as session:
    pass
```

#### Network Configuration

```python
# No network access
with SandboxSession(
    backend=SandboxBackend.DOCKER,
    runtime_configs={"network_mode": "none"}
) as session:
    pass

# Custom network
with SandboxSession(
    backend=SandboxBackend.DOCKER,
    runtime_configs={"network_mode": "my_isolated_network"}
) as session:
    pass

# Host network (use with caution)
with SandboxSession(
    backend=SandboxBackend.DOCKER,
    runtime_configs={"network_mode": "host"}
) as session:
    pass
```

#### Advanced Runtime Options

```python
with SandboxSession(
    backend=SandboxBackend.DOCKER,
    lang="python",
    runtime_configs={
        # Resource limits
        "cpu_count": 2,
        "cpu_shares": 1024,
        "cpu_period": 100000,
        "cpu_quota": 50000,
        "mem_limit": "512m",
        "memswap_limit": "1g",
        "pids_limit": 100,

        # Security options
        "privileged": False,
        "read_only": True,
        "cap_drop": ["ALL"],
        "cap_add": ["DAC_OVERRIDE"],
        "security_opt": ["no-new-privileges"],

        # User and group
        "user": "1000:1000",
        "userns_mode": "host",

        # Environment
        "environment": {
            "PYTHONUNBUFFERED": "1",
            "CUSTOM_VAR": "value"
        },

        # Devices
        "devices": ["/dev/sda:/dev/xvda:rwm"],

        # Logging
        "log_config": {
            "type": "json-file",
            "config": {"max-size": "10m"}
        }
    }
) as session:
    pass
```

### Docker Best Practices

1. **Use specific image tags**
   ```python
   # Good
   image="python:3.11.5-slim-bullseye"

   # Avoid
   image="python:latest"
   ```

2. **Clean up resources**
   ```python
   # Remove containers and images after use
   with SandboxSession(
       keep_template=False,  # Remove image
       runtime_configs={"auto_remove": True}  # Remove container
   ) as session:
       pass
   ```

3. **Use multi-stage builds for custom images**
   ```dockerfile
   # Dockerfile
   FROM python:3.11-slim as builder
   RUN pip install --user numpy pandas

   FROM python:3.11-slim
   COPY --from=builder /root/.local /root/.local
   ```

## Kubernetes Backend

### Overview

Kubernetes backend is ideal for production deployments, offering scalability and orchestration features.

### Installation

```bash
# Install kubectl
curl -LO "https://dl.k8s.io/release/$(curl -L -s https://dl.k8s.io/release/stable.txt)/bin/linux/amd64/kubectl"
sudo install -o root -g root -m 0755 kubectl /usr/local/bin/kubectl

# Install LLM Sandbox with Kubernetes support
pip install 'llm-sandbox[k8s]'
```

### Basic Usage

```python
from llm_sandbox import SandboxSession, SandboxBackend

with SandboxSession(
    backend=SandboxBackend.KUBERNETES,
    lang="python",
    kube_namespace="default"
) as session:
    result = session.run("print('Hello from Kubernetes!')")
    print(result.stdout)
```

### Custom Kubernetes Configuration

```python
from kubernetes import client, config

# Load custom kubeconfig
config.load_kube_config(config_file="~/.kube/custom-config")

# Or use in-cluster config
# config.load_incluster_config()

k8s_client = client.CoreV1Api()

with SandboxSession(
    backend=SandboxBackend.KUBERNETES,
    client=k8s_client,
    lang="python",
    kube_namespace="sandbox-namespace"
) as session:
    pass
```

### Custom Pod Manifests

```python
# Basic pod customization
with SandboxSession(
    backend=SandboxBackend.KUBERNETES,
    lang="python",
    pod_manifest={
        "apiVersion": "v1",
        "kind": "Pod",
        "metadata": {
            "name": "sandbox-pod",
            "namespace": "default",
            "labels": {
                "app": "llm-sandbox",
                "environment": "production"
            },
            "annotations": {
                "prometheus.io/scrape": "true"
            }
        },
        "spec": {
            "containers": [{
                "name": "sandbox",
                "image": "python:3.11-slim",
                "resources": {
                    "requests": {
                        "memory": "256Mi",
                        "cpu": "250m"
                    },
                    "limits": {
                        "memory": "512Mi",
                        "cpu": "500m"
                    }
                },
                "securityContext": {
                    "runAsNonRoot": True,
                    "runAsUser": 1000,
                    "readOnlyRootFilesystem": True,
                    "allowPrivilegeEscalation": False
                }
            }],
            "securityContext": {
                "runAsNonRoot": True,
                "fsGroup": 2000
            }
        }
    }
) as session:
    pass
```

### Advanced Kubernetes Features

#### Persistent Volumes

```python
pod_manifest = {
    "apiVersion": "v1",
    "kind": "Pod",
    "metadata": {"name": "sandbox-with-pv"},
    "spec": {
        "containers": [{
            "name": "sandbox",
            "image": "python:3.11",
            "volumeMounts": [{
                "name": "data-volume",
                "mountPath": "/data"
            }]
        }],
        "volumes": [{
            "name": "data-volume",
            "persistentVolumeClaim": {
                "claimName": "sandbox-pvc"
            }
        }]
    }
}
```

#### ConfigMaps and Secrets

```python
pod_manifest = {
    "spec": {
        "containers": [{
            "name": "sandbox",
            "image": "python:3.11",
            "env": [
                {
                    "name": "CONFIG_VALUE",
                    "valueFrom": {
                        "configMapKeyRef": {
                            "name": "app-config",
                            "key": "value"
                        }
                    }
                },
                {
                    "name": "SECRET_VALUE",
                    "valueFrom": {
                        "secretKeyRef": {
                            "name": "app-secret",
                            "key": "password"
                        }
                    }
                }
            ],
            "volumeMounts": [
                {
                    "name": "config",
                    "mountPath": "/config"
                },
                {
                    "name": "secret",
                    "mountPath": "/secrets"
                }
            ]
        }],
        "volumes": [
            {
                "name": "config",
                "configMap": {"name": "app-config"}
            },
            {
                "name": "secret",
                "secret": {"secretName": "app-secret"}
            }
        ]
    }
}
```

#### Node Affinity

```python
pod_manifest = {
    "spec": {
        "affinity": {
            "nodeAffinity": {
                "requiredDuringSchedulingIgnoredDuringExecution": {
                    "nodeSelectorTerms": [{
                        "matchExpressions": [{
                            "key": "node-type",
                            "operator": "In",
                            "values": ["sandbox"]
                        }]
                    }]
                }
            }
        },
        "tolerations": [{
            "key": "sandbox",
            "operator": "Equal",
            "value": "true",
            "effect": "NoSchedule"
        }]
    }
}
```

### Kubernetes Best Practices

1. **Resource Limits**
   ```python
   # Always set resource requests and limits
   "resources": {
       "requests": {"memory": "256Mi", "cpu": "250m"},
       "limits": {"memory": "512Mi", "cpu": "500m"}
   }
   ```

2. **Security Context**
   ```python
   # Run as non-root with restricted permissions
   "securityContext": {
       "runAsNonRoot": True,
       "runAsUser": 1000,
       "readOnlyRootFilesystem": True
   }
   ```

3. **Namespace Isolation**
   ```python
   # Use dedicated namespaces
   kube_namespace="llm-sandbox-prod"
   ```

## Podman Backend

### Overview

Podman provides rootless containers for enhanced security, making it ideal for security-conscious environments.

### Installation

```bash
# Install Podman (Ubuntu/Debian)
sudo apt-get update
sudo apt-get install -y podman

# Install LLM Sandbox with Podman support
pip install 'llm-sandbox[podman]'
```

### Basic Usage

```python
from podman import PodmanClient
from llm_sandbox import SandboxSession, SandboxBackend

# Create Podman client
client = PodmanClient(
    base_url="unix:///run/user/1000/podman/podman.sock"
)

with SandboxSession(
    backend=SandboxBackend.PODMAN,
    client=client,
    lang="python"
) as session:
    result = session.run("print('Hello from Podman!')")
    print(result.stdout)
```

### Rootless Configuration

```python
# Rootless Podman with user namespace
with SandboxSession(
    backend=SandboxBackend.PODMAN,
    client=client,
    lang="python",
    runtime_configs={
        "userns_mode": "keep-id",  # Keep user ID
        "user": "1000:1000",
        "security_opt": [
            "no-new-privileges",
            "seccomp=unconfined"  # If needed
        ]
    },
    workdir="/tmp/sandbox"  # Writable for non-root
) as session:
    pass
```

### Podman-Specific Features

#### Podman Pods

```python
# Create a pod first
client.pods.create(
    name="sandbox-pod",
    labels={"app": "llm-sandbox"}
)

# Run container in pod
with SandboxSession(
    backend=SandboxBackend.PODMAN,
    client=client,
    runtime_configs={
        "pod": "sandbox-pod"
    }
) as session:
    pass
```

#### Systemd Integration

```python
# Generate systemd unit
container = session.container
unit = container.generate_systemd(
    name="llm-sandbox",
    restart_policy="on-failure",
    time=10
)
```

### Podman Best Practices

1. **Use rootless mode**
   ```python
   # Run as regular user
   client = PodmanClient(
       base_url=f"unix:///run/user/{os.getuid()}/podman/podman.sock"
   )
   ```

2. **User namespace mapping**
   ```python
   runtime_configs={"userns_mode": "keep-id"}
   ```

### Use Case Recommendations

1. **Development**: Docker or Podman for fast iteration and easy debugging
2. **Production**: Kubernetes for scalability and enterprise features
3. **Security-Critical**: Podman for rootless containers and SELinux integration

## Multi-Backend Support

### Backend Fallback

```python
def create_session_with_fallback(**kwargs):
    """Try multiple backends in order"""
    backends = [
        (SandboxBackend.DOCKER, {}),
        (SandboxBackend.PODMAN, {"client": get_podman_client()}),
        (SandboxBackend.KUBERNETES, {"kube_namespace": "default"}),
    ]

    for backend, backend_kwargs in backends:
        try:
            return SandboxSession(
                backend=backend,
                **kwargs,
                **backend_kwargs
            )
        except Exception as e:
            print(f"Backend {backend} failed: {e}")
            continue

    raise RuntimeError("No available backends")
```

### Backend Detection

```python
import subprocess

def detect_available_backends():
    """Detect which backends are available"""
    available = []

    # Check Docker
    try:
        subprocess.run(["docker", "--version"],
                      capture_output=True, check=True)
        available.append(SandboxBackend.DOCKER)
    except:
        pass

    # Check Podman
    try:
        subprocess.run(["podman", "--version"],
                      capture_output=True, check=True)
        available.append(SandboxBackend.PODMAN)
    except:
        pass

    # Check Kubernetes
    try:
        subprocess.run(["kubectl", "version", "--client"],
                      capture_output=True, check=True)
        available.append(SandboxBackend.KUBERNETES)
    except:
        pass

    return available
```

## Troubleshooting

### Docker Issues

```bash
# Permission denied
sudo usermod -aG docker $USER
newgrp docker

# Cannot connect to daemon
sudo systemctl start docker
docker context use default
```

### Kubernetes Issues

```bash
# No access to cluster
kubectl config view
kubectl auth can-i create pods

# Pod stuck in pending
kubectl describe pod <pod-name>
kubectl get events
```

### Podman Issues

```bash
# Rootless setup
podman system migrate
podman unshare cat /etc/subuid

# Socket not found
systemctl --user start podman.socket
```

## Next Steps

- Learn about [Supported Languages](languages.md)
- Configure [Security Policies](security.md)
- Explore [Integration Options](integrations.md)
- See practical [Examples](examples.md)
