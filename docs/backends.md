# Sandbox Backends

LLM Sandbox supports container backends and local Linux and Windows sandboxes through the same session factory. This guide covers their requirements, configuration, and extension points.

## Overview

Supported backends:

| Backend | Typical use | Runtime requirement | Images | Interactive / pooling |
|---------|-------------|---------------------|--------|-----------------------|
| **Docker** | Development, single host | Docker service and Python SDK | Yes | Yes / yes |
| **Kubernetes** | Managed workers, orchestration | Cluster access and Python SDK | Yes | Yes / yes |
| **Podman** | Rootless containers | Podman service and Python SDK | Yes | Yes / yes |
| **Micromamba** | Containerized conda environments | Docker service and Python SDK | Yes | No / no |
| **Bubblewrap** | Local Linux code execution | `bwrap` 0.8.0+, user namespaces, system libseccomp | No | No / no |
| **NsJail** | Local Linux code execution | `nsjail`, user namespaces, Kafel and new mount API support | No | No / no |
| **Windows MXC** | Native Windows Python and JavaScript | Compatible Windows PSEC APIs and the native MXC helper | No | No / no |
| **WSL2 Bubblewrap** | Linux tools from Windows applications | Explicit WSL2 distribution, matching guest package, Bubblewrap and libseccomp | No | No / no |

Capabilities describe adapter features, not proof that a host can run the backend or that a particular session enables every protection. Docker remains the default; choose local backends explicitly.

## Local Linux Backends

Bubblewrap and NsJail avoid container images and a container daemon. They start each command in an isolated process tree, mount a private session workspace, and expose selected host runtime files read-only. Both share the host Linux kernel.

### Requirements

- Linux with user namespaces and permission to create them. A host, outer container, or distribution security policy can deny this even when the executable is installed.
- Bubblewrap: a working `bwrap` binary version 0.8.0 or newer with `--disable-userns` support, and system libseccomp for the default syscall filter. The filter compiler supports native x86-64 and AArch64 executables and rejects foreign syscall ABIs.
- NsJail: a build with Kafel seccomp support and the new mount API selected by `--experimental_mnt new` (available in NsJail 3.6). The host must permit the required mount syscalls; upstream's automatic detection recommends Linux 6.3 or newer. The adapter always selects the new API to enforce recursive read-only mounts and fails if it is unavailable, without falling back to legacy mounts. Native x86-64 and AArch64 are supported; the build must recognize the configured syscall names.
- Host language runtimes and compilers inside the configured read-only paths. Python defaults to `/usr/bin/python3` and needs the `venv`/`ensurepip` components when environment setup is enabled.
- `prlimit` from util-linux when any resource limit is configured.

For example, Debian/Ubuntu Bubblewrap prerequisites are:

```bash
sudo apt install bubblewrap libseccomp2 python3-venv
pip install llm-sandbox
```

Check `bwrap --version` and `bwrap --help` for the required version and `--disable-userns` option. Older distribution packages, including Ubuntu 22.04's default package, need an upgrade or an [upstream build](https://github.com/containers/bubblewrap). Unsupported launcher options fail without reducing the policy.

Install NsJail using your distribution package or the [upstream build instructions](https://github.com/google/nsjail). Neither adapter needs an optional Python container SDK. Installing prerequisites may require administrator access; ordinary sessions run as the current user.

### Basic Usage

```python
from llm_sandbox import SandboxSession

with SandboxSession(backend="bubblewrap", lang="python") as session:
    result = session.run("print('Hello from Bubblewrap!')")
    print(result.stdout)

with SandboxSession(backend="nsjail", lang="python") as session:
    result = session.run("print('Hello from NsJail!')")
    print(result.stdout)
```

These examples do not require network access. Python setup creates a private virtual environment with system site packages visible, without upgrading pip. The library does not download a Python interpreter, compiler, root filesystem, or container image.

### Shared API and Lifecycle

Both adapters implement `run()`, `execute_command()`, `install()`, `copy_to_runtime()`, and `copy_from_runtime()`. File transfers are confined to the private session workspace, normally `/sandbox` inside the sandbox, and the session's private `/tmp`. Use those APIs to supply inputs and retrieve outputs.

The workspace, installed packages, and private `/tmp` persist across commands until the session closes. The sandbox's `/tmp` is a separate session directory, not the host's `/tmp`. Each command gets a new PID namespace; background processes and interpreter state do not persist. Sessions do not expose arbitrary host writable bind mounts. `ArtifactSandboxSession(backend="bubblewrap", ...)` and the corresponding NsJail wrapper can extract artifacts when their language and plotting dependencies are available inside the sandbox.

NsJail also mounts a private 64 MiB tmpfs at `/dev/shm` for POSIX shared memory and Python multiprocessing. This storage belongs to one command and is not the host's `/dev/shm`.

Container-specific options, images, existing-container attachment, pooling, and `InteractiveSandboxSession` are unsupported for these adapters. Install host toolchains before opening a session; package operations that need to modify system runtime directories cannot work against read-only mounts. `install()` and `run(..., libraries=...)` require explicit host network access. For offline packages, copy a package source into the workspace and invoke the package manager explicitly with `execute_command()`.

### Composing Isolation Layers

The local backends combine complementary controls:

| Layer | Responsibility |
|-------|----------------|
| Namespaces and mounts | Restrict the process tree, filesystem view, and network access |
| Seccomp | Reject selected kernel operations independently of the programming language |
| Private workspace, temporary directory, and clean environment | Limit writable host data and inherited environment secrets |
| Timeouts, output caps, and optional POSIX resource limits | Bound command execution and selected per-process resources |
| Host runtime provider | Supply the interpreter, compiler, libraries, and compatible versions |

For Bubblewrap, the library compiles `SeccompPolicy` through system libseccomp and passes the filter to `bwrap --seccomp`. For NsJail, it translates the same syscall names into a Kafel policy. Default filters are denylists, not complete syscall allowlists. See [local security](security.md#local-linux-isolation-and-seccomp) for their scope and limitations, and [local configuration](configuration.md#local-linux-configuration) for overrides.

Missing prerequisites, unsupported policy elements, and launch failures are errors. The library does not retry the command unsandboxed or select a weaker backend automatically.

### Native Integration CI

The `Native Linux backends` workflow builds pinned Bubblewrap and both NsJail 3.6 and a newer pinned revision on Ubuntu 22.04. It requires both backends to create the default offline Python environment with seccomp active before running real filesystem, network, syscall, lifecycle, and multiprocessing checks. A writable bind mount beneath a read-only runtime grant checks recursive protection on both NsJail versions. Its test runner treats skipped tests as failures so an unavailable sandbox cannot produce a passing integration job.

On a host with both backends installed, run the integration gate below after preparing the nested-mount fixture as shown in `.github/workflows/native-backends.yml` and exporting `LLM_SANDBOX_TEST_READONLY_TREE` and `LLM_SANDBOX_TEST_WRITABLE_SOURCE`. The fixture requires permission to create a bind mount. Ordinary sandbox sessions do not require that setup.

```bash
uv run python .github/scripts/require_native_integration.py \
  tests/test_bubblewrap.py tests/test_nsjail.py -m integration -ra
```

## Windows Backends

The experimental Windows adapters are selected explicitly from native Windows Python. They do not need Docker, a container image, a container SDK, or a background service installed by this library. Neither supports pooling, existing-container attachment, or interactive kernels. Missing prerequisites and policy failures stop execution; there is no automatic switch between MXC, WSL, AppContainer, or an ordinary host process.

### Native Windows MXC

`backend="windows-mxc"` uses a native Rust helper pinned to [Microsoft MXC](https://github.com/microsoft/mxc/tree/6cd3d58f05d3447e67109cfb75e042803b843ca4). The helper calls the PSEC process-container APIs directly. It probes those APIs by creating and closing a container before accepting work, rather than inferring support from a Windows version string. A compatible, updated Windows installation is required; an ordinary Windows CI runner may lack PSEC support. This backend does not require Windows Sandbox, WSL, account provisioning, or elevation for each session.

The helper is bundled when building a Windows platform wheel. A source checkout or generic Python wheel needs a separately built helper, selected with an absolute `mxc_path`. See the [native helper build instructions](https://github.com/vndee/llm-sandbox/tree/main/native/windows-mxc). The Python backend never downloads native executables at runtime. The Windows build workflow produces platform wheel artifacts containing the helper; their presence in CI is not a claim that a package release has been published.

```python
from llm_sandbox import SandboxSession, WindowsMxcConfig

with SandboxSession(
    backend="windows-mxc",
    lang="python",
    sandbox_config=WindowsMxcConfig(network="none"),
    # For a source build, set the actual absolute helper path:
    # mxc_path=r"C:\tools\llm-sandbox-mxc.exe",
) as session:
    print(session.run("from pathlib import Path; Path('result.txt').write_text('done')").exit_code)
    session.copy_from_runtime("result.txt", r"C:\outputs\result.txt")
```

Create the destination directory before copying. Native sessions expose their physical private directory as `session.workspace` after `open()`. File transfers accept paths relative to it, or absolute paths inside it. This is a Windows filesystem API: Linux aliases such as `/sandbox` and `/tmp` are not runtime paths.

Initially, `run()` supports Python and JavaScript using installed Windows runtimes. Python defaults to the current interpreter; Node.js can be selected explicitly. `execute_argv()` accepts an absolute `.exe` and an argument list. `execute_command()` stages a command script and starts `cmd.exe` inside PSEC. Shell syntax is therefore Windows command syntax. Runtime directories and Windows system files are exposed read-only; the workspace is writable and persists until close. A Job Object terminates descendants when each command finishes, times out, or loses its helper.

Automatic environment creation and package installation are unsupported: provision dependencies in the selected runtime first. `install()` with packages and `run(..., libraries=...)` fail explicitly. The POSIX plotting wrapper is also unsupported; use `copy_from_runtime()` to retrieve files written by the native program. Image options, Linux seccomp and POSIX resource limits do not apply. See [Windows configuration](configuration.md#windows-configuration) and [security boundaries](security.md#windows-isolation).

### WSL2 Bubblewrap

`backend="wsl2-bubblewrap"` runs a trusted worker in a named WSL2 distribution, which delegates payload execution to the existing Linux Bubblewrap adapter. WSL1 is rejected. The adapter checks the distribution version and guest kernel, then requires the worker's protocol, package version, and security implementation fingerprint to match the Windows installation.

Prepare WSL2 and the distribution first using Microsoft's [WSL installation instructions](https://learn.microsoft.com/windows/wsl/install). Inside the distribution, install the [Bubblewrap prerequisites](#requirements), including `python3-venv`. Install the same `llm-sandbox` release/build used by the Windows client into a trusted guest Python environment, using its pure Python wheel or source distribution. A Windows platform wheel cannot be installed in Linux. For example, in the Linux shell:

```bash
python3 -m venv ~/.venvs/llm-sandbox-worker
~/.venvs/llm-sandbox-worker/bin/python -m pip install /path/to/llm_sandbox-VERSION-py3-none-any.whl
```

Replace the wheel path and `VERSION` with the actual matching build. Its Python sources must match the Windows client; the optional Windows helper does not affect this comparison. Keep this environment outside directories writable by sandboxed code. From native Windows Python:

```python
from llm_sandbox import LocalSandboxConfig, SandboxSession

with SandboxSession(
    backend="wsl2-bubblewrap",
    distribution="Ubuntu",  # Exact name from wsl.exe --list --verbose.
    guest_python="/home/alice/.venvs/llm-sandbox-worker/bin/python",  # Use your actual Linux path.
    sandbox_config=LocalSandboxConfig(network="none"),
) as session:
    print(session.run("print('Linux namespaces and seccomp from Windows')").stdout)
```

`guest_python` defaults to `/usr/bin/python3` if the matching package is installed there. It selects the trusted worker interpreter; `LocalSandboxConfig.python_executable` independently selects the payload's Linux interpreter. The host invokes the system `wsl.exe` with discrete arguments and never interpolates payload code into a Windows command line or host shell.

The worker supports Linux `run()`, commands, package installation, and artifact extraction. File-copy operations transfer bounded regular files through the control channel; directory transfers are unsupported. Workspace files and private guest `/tmp` persist during the session. Host EOF or a missed heartbeat closes the guest session and its command PID namespace, since terminating `wsl.exe` alone does not guarantee Linux descendant cleanup.

The WSL adapter requires the default seccomp restrictions; custom policies may add denied syscalls but cannot remove the baseline or disable seccomp. Read-only grants may only select from the built-in Linux runtime paths. It rejects Windows/shared mounts and interop environment variables. This intentionally limits custom toolchains outside those paths. Networking is disabled by default; `network="host"` grants the guest's full network reachability, including reachable Windows services. See [WSL configuration restrictions](configuration.md#wsl2-options) and [security boundaries](security.md#windows-isolation).

### Windows Validation

The Windows workflow builds the pinned helper and platform wheels for x86-64 and ARM64, and runs portable adapter tests plus native Windows file/process tests. If a hosted runner's capability probe succeeds, it must also pass the PSEC integration suite without skips. An unsupported runner records that limitation and validates the build only. Native PSEC and WSL2 enforcement can also be checked through the manually dispatched workflow on designated runners, which must fail when prerequisites are absent. A successful helper build or mocked policy test alone does not prove kernel enforcement.

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
        # NOTE: "read_only": True is not usable with SandboxSession. The session
        # copies your code into the container, and Docker rejects that against a
        # read-only rootfs ("container rootfs is marked read-only") -- a tmpfs on
        # the workdir does not help.
        "cap_drop": ["ALL"],
        # Keep DAC_OVERRIDE: without it the container cannot read the copied
        # source file. Everything else stays dropped.
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

## Multi-Backend Support

### Discovery and Explicit Selection

```python
from llm_sandbox import SandboxSession, get_backend, list_backends

for definition in list_backends():
    print(definition.name, definition.capabilities)

definition = get_backend("bubblewrap")
print(definition.capabilities.platforms)  # ("linux",)

with SandboxSession(backend=definition.name, lang="python") as session:
    print(session.run("print(2 + 2)").stdout)
```

`list_backends()` returns an immutable snapshot of registered definitions. `get_backend()` resolves a name without importing its backend SDK or probing the host. `BackendDefinition` and `BackendCapabilities` are immutable metadata. Names are case-insensitive.

Check the host by opening the selected backend with the intended configuration and running a representative workload. An executable's `--version` output alone does not demonstrate namespace or seccomp support. Any application-level fallback must preserve its required protections and be explicit to the caller; the session factory does not implement automatic fallback.

### Registering an Adapter

```python
from llm_sandbox import BackendCapabilities, SandboxSession, register_backend
from my_adapter import MySandboxSession  # Your BaseSession implementation

register_backend(
    "my-sandbox",
    MySandboxSession,
    capabilities=BackendCapabilities(
        platforms=("linux",),
        filesystem_isolation=True,
        network_isolation=True,
    ),
)

with SandboxSession(backend="my-sandbox", lang="python") as session:
    print(session.run("print('Custom backend')").stdout)
```

The factory receives session arguments and must return a `BaseSession` implementation. Its dependencies, host validation, policy enforcement, and lifecycle are the adapter's responsibility. Registration does not run the factory and cannot overwrite built-in or previously registered names. Register adapters once during application startup. There is no automatic third-party plugin discovery.

### Extension Roadmap

The registry is the implemented extension point. The following are future candidates, not built-in adapters:

| Candidate | Audience | Integration direction |
|-----------|----------|-----------------------|
| Landlock | Local Linux tools | Additional filesystem restrictions, with kernel ABI negotiation |
| macOS Seatbelt | Desktop agents on macOS | Native process sandbox with OS compatibility checks |
| Windows AppContainer / restricted tokens | Desktop agents on Windows | Native capabilities, ACLs, and process lifecycle controls |
| Wasmtime / WASI | Plugins with controlled APIs | Runtime isolation and explicit host capabilities |
| gVisor | Managed workers running native code | Userspace kernel boundary and runtime compatibility validation |
| MicroVMs | Managed workloads requiring a VM boundary | Guest/runtime provisioning, resource control, and lifecycle management |

Keep three decisions separate when adding adapters: the isolation mechanism, additional policy layers, and the runtime provider. For example, Bubblewrap plus seccomp is already implemented; adding Landlock would require an additional supported policy layer. Runtime packaging can evolve independently from isolation so desktop users can receive provisioned toolchains without needing Docker. Adapter metadata must describe actual features, and unsupported required protections must cause an error rather than silent degradation.

## Troubleshooting

### Local Linux Issues

- **Executable missing:** install `bubblewrap` or `nsjail` on the host and ensure it is on `PATH`.
- **Namespace permission denied:** check the host or outer container policy and distribution restrictions. The library cannot override a host denial of user namespaces.
- **Seccomp compilation fails:** check the installed libseccomp version for Bubblewrap or Kafel support for NsJail. Unknown requested syscall names are not silently ignored.
- **Python environment setup fails:** install the host Python `venv` component and ensure the configured interpreter is under a mounted read-only path.
- **Dependency download fails:** networking is disabled by default. `network="host"` permits full host networking; it is not a domain allowlist.
- **Resource limit fails:** install util-linux for `prlimit` and check that the chosen limits allow the runtime to start. JavaScript and Java may reserve much more virtual memory than their resident memory usage.

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
