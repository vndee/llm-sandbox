# Container Backends

LLM Sandbox supports multiple container backends to suit different infrastructure needs. This guide covers each backend's features, configuration, and best practices.

## Overview

Supported backends:

| Backend | Use Case | Root Access | Orchestration | Performance |
|---------|----------|-------------|---------------|-------------|
| **Docker** | Development, single-host | Yes | Limited | High |
| **Kubernetes** | Production, scalable | Configurable | Full | High |
| **Podman** | Rootless security | No (rootless) | Limited | High |
| **Tenki** | Cloud microVMs, no local runtime | No | Managed | High |

> [!IMPORTANT]
> Working directory differs on Tenki
> Docker, Kubernetes, Podman and Micromamba all default `workdir` to `/sandbox`.
> Tenki defaults to `/home/tenki`. See [Tenki Backend](#tenki-backend).

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

#### Real-Time Output Streaming

Docker supports real-time output streaming via `on_stdout` and `on_stderr` callbacks on `run()` and `execute_command()`. When callbacks are provided, streaming mode is enabled automatically. `PYTHONUNBUFFERED=1` is injected into the container environment at creation time so Python output flushes after every write.

```python
with SandboxSession(backend=SandboxBackend.DOCKER, lang="python") as session:
    result = session.run(
        "import time\nfor i in range(3):\n    print(f'Step {i}')\n    time.sleep(1)",
        on_stdout=lambda chunk: print(f"[live] {chunk}", end=""),
    )
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
# Basic pod customization with secure read-only root filesystem
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
                },
                # IMPORTANT: When using readOnlyRootFilesystem, you MUST mount
                # writable volumes for /sandbox (workdir) and /tmp (required by pip)
                "volumeMounts": [
                    {
                        "name": "sandbox-writable",
                        "mountPath": "/sandbox"
                    },
                    {
                        "name": "tmp-writable",
                        "mountPath": "/tmp"
                    }
                ]
            }],
            "securityContext": {
                "runAsNonRoot": True,
                "fsGroup": 2000  # Ensures volumes are writable by runAsUser
            },
            # Define emptyDir volumes for writable directories
            "volumes": [
                {
                    "name": "sandbox-writable",
                    "emptyDir": {}
                },
                {
                    "name": "tmp-writable",
                    "emptyDir": {}
                }
            ]
        }
    }
) as session:
    pass
```

#### Read-Only Root Filesystem Requirements

**⚠️ Important:** When using `readOnlyRootFilesystem: True` (a security best practice), you **MUST** provide writable volumes for:

1. **`/sandbox`** (or your custom `workdir`) - Required for code execution and file operations
2. **`/tmp`** - Required by pip and other Python tools for temporary files

**Why this is needed:**
- Read-only root filesystems prevent containers from writing to any directory on the root filesystem
- This is a security best practice that prevents malicious code from modifying system files
- However, llm-sandbox needs writable directories to:
  - Create virtual environments in `/sandbox/.sandbox-venv`
  - Store pip cache in `/sandbox/.sandbox-pip-cache`
  - Allow pip to create temporary build directories in `/tmp`
  - Execute and store user code files

**Minimal working example:**

```python
pod_manifest = {
    "spec": {
        "containers": [{
            "name": "sandbox",
            "image": "python:3.11-slim",
            "securityContext": {
                "readOnlyRootFilesystem": True,
                "runAsNonRoot": True,
                "runAsUser": 1000
            },
            "volumeMounts": [
                {"name": "sandbox-writable", "mountPath": "/sandbox"},
                {"name": "tmp-writable", "mountPath": "/tmp"}
            ]
        }],
        "securityContext": {
            "fsGroup": 2000  # Critical: makes volumes writable by user 1000
        },
        "volumes": [
            {"name": "sandbox-writable", "emptyDir": {}},
            {"name": "tmp-writable", "emptyDir": {}}
        ]
    }
}
```

**Alternative: Custom workdir**

If you prefer a different working directory, configure both the `workdir` parameter and mount a volume there:

```python
with SandboxSession(
    backend=SandboxBackend.KUBERNETES,
    workdir="/workspace",  # Custom working directory
    pod_manifest={
        "spec": {
            "containers": [{
                "volumeMounts": [
                    {"name": "workspace", "mountPath": "/workspace"},
                    {"name": "tmp", "mountPath": "/tmp"}
                ]
            }],
            "volumes": [
                {"name": "workspace", "emptyDir": {}},
                {"name": "tmp", "emptyDir": {}}
            ]
        }
    }
) as session:
    pass
```

**Common errors without proper volumes:**

```
RuntimeError: Failed to create directory /sandbox: mkdir: cannot create directory '/sandbox': Read-only file system
```

```
FileNotFoundError: [Errno 2] No usable temporary directory found in ['/tmp', '/var/tmp', '/usr/tmp', '/']
```

For a complete working example, see [`examples/k8s_readonly_file_system.py`](https://github.com/vndee/llm-sandbox/blob/main/examples/k8s_readonly_file_system.py) in the repository.

#### Real-Time Output Streaming

Kubernetes supports real-time output streaming through the same `on_stdout`/`on_stderr` callback interface. The Kubernetes exec API is inherently streaming (using `_preload_content=False`), so callbacks receive chunks as they arrive. `PYTHONUNBUFFERED=1` is included in the default pod manifest's environment. Users with custom `pod_manifest` should add it manually if streaming is needed.

```python
with SandboxSession(
    backend=SandboxBackend.KUBERNETES,
    lang="python",
    kube_namespace="default",
) as session:
    result = session.run(
        "import time\nfor i in range(3):\n    print(f'Step {i}')\n    time.sleep(1)",
        on_stdout=lambda chunk: print(f"[live] {chunk}", end=""),
    )
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

2. **Security Context with Read-Only Root Filesystem**
   ```python
   # Run as non-root with restricted permissions
   # IMPORTANT: When using readOnlyRootFilesystem, you MUST also provide
   # writable volumes for /sandbox and /tmp (see example above)
   "containers": [{
       "securityContext": {
           "runAsNonRoot": True,
           "runAsUser": 1000,
           "readOnlyRootFilesystem": True,
           "allowPrivilegeEscalation": False
       },
       "volumeMounts": [
           {"name": "sandbox-writable", "mountPath": "/sandbox"},
           {"name": "tmp-writable", "mountPath": "/tmp"}
       ]
   }],
   "securityContext": {
       "fsGroup": 2000  # Ensures volumes are writable
   },
   "volumes": [
       {"name": "sandbox-writable", "emptyDir": {}},
       {"name": "tmp-writable", "emptyDir": {}}
   ]
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

#### Real-Time Output Streaming

Podman supports the same `on_stdout`/`on_stderr` streaming callbacks as Docker. Since `SandboxPodmanSession` inherits from `SandboxDockerSession`, `PYTHONUNBUFFERED=1` is injected automatically.

```python
from podman import PodmanClient
from llm_sandbox import SandboxSession, SandboxBackend

client = PodmanClient(base_url="unix:///run/user/1000/podman/podman.sock")

with SandboxSession(
    backend=SandboxBackend.PODMAN,
    client=client,
    lang="python",
) as session:
    result = session.run(
        "import time\nfor i in range(3):\n    print(f'Step {i}')\n    time.sleep(1)",
        on_stdout=lambda chunk: print(f"[live] {chunk}", end=""),
    )
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

## Tenki Backend

### Overview

Tenki runs code in cloud microVMs instead of a local container runtime. There is no daemon
to install and nothing running on your machine — sessions boot from a pre-warmed pool in a
few seconds and are billed while they run.

### Installation

```bash
pip install 'llm-sandbox[tenki]'

# Authenticate (TENKI_API_KEY also works)
export TENKI_AUTH_TOKEN="your-token"
```

The SDK reads the credential itself, so you normally pass nothing in code. To override it,
use `auth_token=...`.

### Basic Usage

```python
from llm_sandbox import SandboxSession, SandboxBackend

with SandboxSession(backend=SandboxBackend.TENKI, lang="python") as session:
    result = session.run("print('Hello from Tenki!')")
    print(result.stdout)
```

### Working Directory: `/home/tenki`, not `/sandbox`

**This is the difference most likely to break code written for another backend.** Every
other backend defaults `workdir` to `/sandbox`. Tenki defaults to `/home/tenki`, matching
the guest image's own home directory.

```python
# Works on Docker, fails on Tenki — /sandbox does not exist there
session.run("open('/sandbox/data.csv').read()")

# Portable: ask the session where it is
session.run("import os; print(os.getcwd())")
```

### Runtime Configuration

`runtime_configs` is passed straight through to the Tenki SDK's `Client.create()`, so use
**Tenki's** parameter names:

```python
with SandboxSession(
    backend=SandboxBackend.TENKI,
    lang="python",
    runtime_configs={
        "cpu_cores": 2,
        "memory_mb": 2048,      # not Docker's "mem_limit"
        "allow_outbound": False, # no network egress
        "timeout": 60,          # provisioning wait (passed to wait_ready)
        "env": {"MY_VAR": "value"},
    },
) as session:
    pass
```

`timeout` is the SDK create wait budget. llm-sandbox creates with `wait=False` so it can assign
the sandbox handle before waiting; the same `timeout` value is then passed to `wait_ready()`.
Omit it to keep the SDK default of 180 seconds.

There is no `user` option — a Tenki sandbox is single-tenant and runs everything as one
non-root user.

### Reusing a Sandbox

Pass `container_id` to attach to a sandbox that is already running. Environment setup is
skipped, and `close()` **detaches instead of terminating**, so the microVM stays up for
the next session:

```python
with SandboxSession(
    backend=SandboxBackend.TENKI,
    lang="python",
    container_id="019fa328-5b7f-7057-a3a8-3682cddb67e4",
) as session:
    session.run("print('reusing a warm sandbox')")
# Sandbox is still running here — terminate it yourself when done
```

Because setup is skipped, the sandbox must already have a Python virtualenv at
`<workdir>/.sandbox-venv`. Attaching to a bare sandbox created outside llm-sandbox will
fail to run code.

### Faster Startup

If your image already has Python ready, skip environment setup entirely:

```python
with SandboxSession(
    backend=SandboxBackend.TENKI,
    lang="python",
    image="your-image-with-python",
    skip_environment_setup=True,
) as session:
    pass
```

Note that `libraries=[...]` is unavailable in this mode — bake dependencies into the image.

### Tenki Best Practices

1. **Always use a context manager.** Sandboxes are billed while running, and `with` calls
   `close()` however the block exits.
2. **Set `allow_outbound=False`** unless the code genuinely needs network access.
3. **Prebuild images** with your dependencies rather than installing per session.
4. **Don't hardcode `/sandbox`** — see the working directory section above.

> [!WARNING]
> Cleanup is best-effort, not guaranteed
> `with` is not a promise that the sandbox was released, so a caller that cares about the
> bill should be prepared to retry or terminate the sandbox manually.

Construct the session first if you want to retry, since `as session` is never bound when
`open()` fails:

```python
session = SandboxSession(backend=SandboxBackend.TENKI, lang="python")
try:
    with session:
        session.run("print('hello')")
except ContainerError:
    if session.container:      # release failed; the microVM is still billing
        session.close()        # retry once
    raise
```

A process that exits before retrying leaves the sandbox running. Set
`idle_timeout_minutes` or `max_duration` in `runtime_configs` if you need a backstop that
does not depend on your process staying alive.

### Languages on the default guest

Tenki does not pick a per-language default image. Without `image=`, only **Python**,
**JavaScript**, and **C++** work. Other languages need a custom `image=` (or an existing
`container_id=`):

```python
SandboxSession(backend=SandboxBackend.TENKI, lang="java")  # rejected
SandboxSession(backend=SandboxBackend.TENKI, lang="java", image="your-registry/java-guest")
```

### Limitations

| Feature | Status |
|---------|--------|
| `dockerfile=` | Not supported — Tenki boots prebuilt images; use `image=` |
| `runtime_configs={"user": ...}` | Not supported — single-tenant, one non-root user |
| Default guest languages | Python, JavaScript, and C++ only; others need `image=` |
| C++ `libraries=[...]` | Not supported — `apt-get` needs root |
| Output streaming | Callbacks fire once per command, not incrementally |

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

### Tenki Issues

| Error | Cause |
|-------|-------|
| `missing auth token: set TENKI_AUTH_TOKEN or TENKI_API_KEY` | No credential found |
| `No such file or directory: /sandbox/...` | Hardcoded path; the workdir is `/home/tenki` |
| `got an unexpected keyword argument` | A Docker-style key in `runtime_configs` |
| `python: command not found` | Image ships no Python; pass a different `image=` |
| `Permission denied` on `apt-get` | The guest is non-root; bake packages into the image |

## Next Steps

- Learn about [Supported Languages](languages.md)
- Configure [Security Policies](security.md)
- Explore [Integration Options](integrations.md)
- See practical [Examples](examples.md)
