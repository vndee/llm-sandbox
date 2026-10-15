# Configuration Guide

This guide covers all configuration options available in LLM Sandbox.

## Local Linux Configuration

Select `backend="bubblewrap"` or `backend="nsjail"` and pass `sandbox_config=LocalSandboxConfig(...)`. These adapters use host toolchains and a private writable workspace instead of a container image. The Python package is sufficient on the library side; install the [system prerequisites](backends.md#requirements) separately.

```python
from llm_sandbox import LocalSandboxConfig, SandboxSession

config = LocalSandboxConfig(
    network="none",
    cpu_time_limit=10,
    file_size_limit=64 * 1024 * 1024,
    open_files_limit=256,
    max_output_bytes=1024 * 1024,
)

with SandboxSession(
    backend="bubblewrap",  # Or "nsjail"
    lang="python",
    sandbox_config=config,
    execution_timeout=20.0,
) as session:
    result = session.run("print('Offline local execution')")
    print(result.stdout)
```

`LocalSandboxConfig` rejects unknown fields. Its fields are:

| Field | Default | Meaning |
|-------|---------|---------|
| `network` | `"none"` | Isolated network; `"host"` explicitly grants full host networking |
| `read_only_paths` | `/usr`, `/bin`, `/lib`, `/lib64`, `/etc/ld.so.cache`, `/etc/alternatives` | Host runtime paths exposed read-only at the same sandbox paths |
| `seccomp` | `SeccompPolicy()` | Syscall denylist; `None` explicitly disables this layer |
| `python_executable` | `"/usr/bin/python3"` | Absolute interpreter path available inside the sandbox |
| `env` | `{}` | Explicit environment additions; host environment variables are not inherited |
| `max_output_bytes` | `8388608` | Captured command output cap in bytes |
| `max_artifact_bytes` | `8388608` | Maximum regular-file payload for each extracted artifact, before archive overhead |
| `memory_limit_bytes` | `None` | Per-process virtual address-space limit in bytes |
| `cpu_time_limit` | `None` | Per-process CPU-time limit in seconds |
| `file_size_limit` | `None` | Per-file size limit in bytes |
| `open_files_limit` | `None` | Per-process open file descriptor limit; minimum 16 |

Resource limits require `prlimit` from util-linux. They are POSIX process limits inherited by descendants, not aggregate cgroup quotas. In particular, a memory limit applies to virtual address space and may need to be much larger for Java or JavaScript than expected resident memory usage. `max_artifact_bytes` is per extracted file, not a combined limit across all plots. There is no session disk quota or process-count limit in this configuration.

Local sessions default to `default_timeout=60.0`; `execution_timeout`, `session_timeout`, and per-`run()` timeouts remain available. Session-level `env_vars` entries override matching keys in `sandbox_config.env`. These values apply inside the sandbox; the trusted launcher receives a separate minimal environment. You can select a trusted launcher outside `PATH` with `bwrap_path` or `nsjail_path` on the matching session. Relative launcher paths are resolved when the session is constructed, before execution switches to the private workspace.

### Runtime Paths and Workspace

The default workspace is `/sandbox` inside the sandbox. `workdir` changes its sandbox location, not the host directory being granted. Only the library's private session workspace and temporary directory are writable on the host; copy input/output files with `copy_to_runtime()` and `copy_from_runtime()`. A separate private directory is mounted at `/tmp` and persists across commands in the session. It does not expose the host's `/tmp`.

`read_only_paths` replaces the default list when supplied. Mounts must be absolute and must not overlap the workspace. Host root, virtual filesystems, and temporary directories cannot be exposed this way. Optional missing default paths are skipped for distribution compatibility; a missing caller-specified path raises an error.

NsJail requires its new mount API to enforce read-only access recursively, including nested host mounts. Missing binary or kernel support fails explicitly. It also supplies a private 64 MiB tmpfs at `/dev/shm` for each command; shared-memory contents do not persist into the next invocation.

With `network="host"`, the adapters also expose existing resolver and TLS trust paths read-only: `/etc/resolv.conf`, `/etc/hosts`, `/etc/nsswitch.conf`, `/etc/services`, `/etc/protocols`, `/etc/ssl/certs`, `/etc/ssl/openssl.cnf`, `/etc/pki/ca-trust`, `/etc/pki/tls/certs`, `/etc/pki/tls/cert.pem`, and `/etc/pki/tls/openssl.cnf`. Resolver symlinks expose their specific resolved file, not its containing host directory.

To add a runtime installed under `/opt`, keep the defaults and explicitly expose its directory:

```python
from llm_sandbox import LocalSandboxConfig
from llm_sandbox.local.config import DEFAULT_READ_ONLY_PATHS

config = LocalSandboxConfig(
    read_only_paths=(*DEFAULT_READ_ONLY_PATHS, "/opt/my-python"),
    python_executable="/opt/my-python/bin/python3",
)
```

That directory must exist and contain the interpreter and its required files. Read-only mounts still disclose their contents to executed code. Do not expose a home directory, credential store, or service socket to make a tool work.

### Environment Setup and Packages

Local Python setup creates a virtual environment in the private workspace with `--system-site-packages`, without a pip upgrade or network download. Host-installed Python packages under mounted runtime paths are therefore visible. Install the host interpreter's `venv`/`ensurepip` support first. Other languages use the existing language handlers and require a compatible host runtime/compiler in mounted paths; the sandbox does not provision those toolchains.

`skip_environment_setup=True` skips automatic language setup and disables `install()` and `run(..., libraries=...)`, as with other backends. This mode requires a usable preconfigured runtime. A session's files and packages persist between commands, but processes and Python interpreter state do not.

Network access is disabled by default. `install()` and `run(..., libraries=...)` reject package installation unless full host networking is explicitly enabled:

```python
from llm_sandbox import LocalSandboxConfig, SandboxSession

# Grants full host networking, including reachable local services.
# This is not a package-registry or domain allowlist.
with SandboxSession(
    backend="bubblewrap",
    lang="python",
    sandbox_config=LocalSandboxConfig(network="host"),
) as session:
    session.install(["requests"])
    print(session.run("import requests; print(requests.__version__)").stdout)
```

You can instead supply offline package files in the workspace and invoke the package manager explicitly with `execute_command()`. System package installation that modifies read-only runtime paths is unsupported. Container-only configuration such as images, Dockerfiles, container attachment, and container pooling does not apply to these adapters. Interactive sessions are unsupported; the artifact wrapper works when its dependencies are installed inside the sandbox.

See [local security and seccomp](security.md#local-linux-isolation-and-seccomp) before changing the default network, mount, or syscall policy. Missing prerequisites and unsupported requested controls fail without an unsandboxed fallback.

## Windows Configuration

Select the experimental `windows-mxc` backend for native Windows tools, or `wsl2-bubblewrap` for Linux tools from a Windows client. See [backend prerequisites](backends.md#windows-backends). Both require an explicit backend choice, default to disabled networking, and reject unsupported controls instead of silently weakening isolation.

### MXC Options

```python
from llm_sandbox import SandboxSession, WindowsMxcConfig

config = WindowsMxcConfig(
    network="none",
    python_executable=r"C:\Python313\python.exe",
    max_output_bytes=1024 * 1024,
)

with SandboxSession(
    backend="windows-mxc",
    sandbox_config=config,
    execution_timeout=20.0,
) as session:
    print(session.run("print('Native Windows execution')").stdout)
```

Use paths that exist on your machine. `WindowsMxcConfig` rejects unknown fields:

| Field | Default | Meaning |
|-------|---------|---------|
| `network` | `"none"` | Denies egress, ingress and host loopback; `"host"` explicitly allows them |
| `read_only_paths` | `()` | Additional read grants beyond the Windows and selected runtime directories |
| `deny_paths` | `()` | Explicit native deny rules; a nonempty policy requires host deny-path support |
| `python_executable` | `None` | Absolute `.exe` path; defaults to the current Python interpreter |
| `javascript_executable` | `None` | Absolute Node.js `.exe` path; otherwise discovers installed `node.exe` |
| `env` | `{}` | Explicit payload environment additions; Windows bootstrap keys are reserved |
| `max_output_bytes` | `8388608` | Combined captured command output cap |
| `max_artifact_bytes` | `16777216` | Maximum size of a bounded extracted file |

Policy paths must be absolute local drive paths. Volume-root grants, UNC/device paths, alternate data streams, and ambiguous Windows names are rejected. Read-only grants still disclose their contents; granting a project or runtime directory can expose credentials stored there. Environment keys are case-insensitive on Windows. Session `env_vars` override corresponding payload values; they do not modify the trusted helper's environment.

When selecting an explicit Python executable from a virtual environment, add its base Python installation and required dependencies to `read_only_paths`. The default current interpreter automatically includes `sys.prefix` and `sys.base_prefix`; an arbitrary external interpreter's installation layout is not guessed. A runtime that cannot start with the supplied grants fails during session startup.

Pass an absolute `mxc_path` to select a trusted helper built from `native/windows-mxc`; otherwise the adapter uses `llm_sandbox/bin/llm-sandbox-mxc.exe` bundled in a Windows platform wheel. There is no runtime download. The helper is part of the trusted computing base and must not be writable by sandboxed code.

Windows file transfers reject links and pin ancestor directories while accessing files. Replacing an existing destination is not atomic: the previous regular file is removed before the new file is created, and a failed transfer can leave no destination. Use a fresh output filename when the existing file must be preserved. This also applies to the Windows destination of a WSL download.

`default_timeout` defaults to 60 seconds; `execution_timeout`, `session_timeout` and per-run overrides apply. There are no aggregate CPU, memory, process-count, or disk quotas in this initial MXC configuration. `LocalSandboxConfig`, seccomp and POSIX resource-limit settings are not accepted substitutes. Automatic package installation, container options, interactive sessions and the artifact plotting wrapper are unsupported.

### WSL2 Options

The Windows client requires `distribution="..."` and optionally `guest_python="/absolute/linux/python"` (default `/usr/bin/python3`). `sandbox_config` is a `LocalSandboxConfig`; ordinary Linux execution options are forwarded to the trusted guest worker. The guest's installed package must match the Windows client, including its security-source fingerprint.

The WSL adapter adds these restrictions:

- `read_only_paths` must be a subset of the built-in Linux defaults. Additional mounts are rejected, including `/mnt/c`, host interop sockets, and Windows/shared filesystems beneath an otherwise allowed runtime directory.
- `seccomp` must retain every syscall in the default denylist. It may add restrictions but cannot be `None`.
- `WSL_INTEROP` and `WSLENV` cannot be supplied as payload environment variables.
- `max_output_bytes` and `max_artifact_bytes` cannot exceed 8 MiB each. File transfers accept regular files only, bounded by the artifact limit.
- Runtime and workspace paths are Linux paths. Host file-copy arguments are Windows paths. No caller-specified `wsl.exe` override or shell wrapper is accepted.

`network="host"` is an explicit grant to the guest's full network, including reachable Windows and LAN services. It does not provide an outbound domain allowlist. Runtime mount validation remains active when networking is enabled; a distribution whose resolver or trust files resolve through a prohibited shared filesystem can be rejected. Package installation otherwise follows the Linux adapter's rules. Resource limits use guest Linux `prlimit`, not Windows Job Object quotas.

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

**Important**: Runtime configurations work differently depending on the backend.

> **Security Warning**: The `runtime_configs` parameter is passed directly to the container runtime without filtering. When accepting `runtime_configs` from untrusted sources, be aware that dangerous options such as `privileged: True`, `cap_add`, `network_mode: "host"`, or host volume mounts can weaken or eliminate container isolation. In applications where end users can influence container configuration, always validate and restrict the allowed keys before passing them to `runtime_configs`. See [Security Considerations for runtime_configs](#security-considerations-for-runtime_configs) below.

### Docker and Podman Backends

For Docker and Podman backends, runtime configuration options are passed as **extra arguments** to the respective Python libraries (`docker-py` and `podman-py`). These are used to configure container creation and execution parameters.

#### Docker Runtime Config
```python
# Docker-specific runtime configuration
runtime_configs = {
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
    runtime_configs=runtime_configs
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
    "memory": "1g",           # 1GB memory limit
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

!!! note "Automatic `PYTHONUNBUFFERED=1`"
    LLM Sandbox automatically injects `PYTHONUNBUFFERED=1` into the container environment for Docker, Podman, and Kubernetes (default pod manifest) backends. This ensures Python output is flushed immediately, which is required for real-time streaming callbacks to work correctly. If you explicitly set `PYTHONUNBUFFERED` in your environment configuration, your value takes precedence.

!!! warning "Security consideration for streaming callbacks"
    When using `on_stdout`/`on_stderr` callbacks, be aware that callbacks receive **all raw decoded output** from the container. If the executed code emits secrets, credentials, or PII, these will be passed to your callbacks. Avoid logging or forwarding callback data to untrusted destinations without sanitization.

### Security Configuration (Docker/Podman)

```python
# Security settings
runtime_configs = {
    "privileged": False,
    "user": "1000:1000",           # Run as specific user/group
    "cap_drop": ["ALL"],           # Drop all capabilities
    # DAC_OVERRIDE must be added back whenever you drop ALL: without it the
    # container cannot read the source file the session copies in, and every
    # run fails with "[Errno 13] Permission denied".
    "cap_add": ["DAC_OVERRIDE"],
    "security_opt": ["no-new-privileges:true"]
}
```

### Security Considerations for runtime_configs

The `runtime_configs` parameter is a **direct pass-through** to the container runtime API. This provides maximum flexibility but means **no automatic filtering is applied**. If your application allows users or external systems to influence `runtime_configs`, you must validate the input yourself.

**Dangerous options that can weaken sandbox isolation:**

| Option | Risk | Description |
|--------|------|-------------|
| `privileged: True` | **Critical** | Grants full host access, effectively disabling sandbox isolation |
| `network_mode: "host"` | **High** | Shares the host network stack, exposing host services |
| `pid_mode: "host"` | **High** | Shares the host PID namespace, allowing process visibility |
| `volumes` (host mounts) | **High** | Mounting host paths (especially `/`, `/etc`, `/var/run/docker.sock`) exposes the host filesystem |
| `cap_add` | **Medium-High** | Adding capabilities like `SYS_ADMIN`, `NET_ADMIN`, `SYS_PTRACE` can enable container escapes |
| `security_opt` | **Medium** | Disabling AppArmor/SELinux profiles removes kernel-level protections |
| `devices` | **Medium** | Exposing host devices to the container |

**Recommended: validate runtime_configs in multi-tenant applications:**

```python
# Example: allowlist for safe runtime_configs keys
ALLOWED_RUNTIME_KEYS = {
    "mem_limit", "cpu_count", "cpu_period", "cpu_quota",
    "memswap_limit", "pids_limit", "environment", "user",
    "working_dir",
}

def sanitize_runtime_configs(configs: dict) -> dict:
    """Filter runtime_configs to only allow safe options."""
    return {k: v for k, v in configs.items() if k in ALLOWED_RUNTIME_KEYS}

# Use sanitized configs
session = SandboxSession(
    lang="python",
    runtime_configs=sanitize_runtime_configs(user_provided_configs)
)
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

## Container Pooling Configuration

For high-performance applications, container pooling dramatically improves execution speed by reusing pre-warmed containers. See the [Container Pooling Guide](container-pooling.md) for comprehensive documentation.

### Basic Pool Configuration

```python
from llm_sandbox import SandboxSession
from llm_sandbox.pool import create_pool_manager, PoolConfig

# Configure pool parameters and create pool manager
pool = create_pool_manager(
    backend="docker",
    config=PoolConfig(
        max_pool_size=10,                  # Maximum containers
        min_pool_size=3,                   # Minimum warm containers
        idle_timeout=300.0,                # Recycle idle containers
        enable_prewarming=True,            # Create containers on startup
    ),
    lang="python",
    libraries=["numpy", "pandas"],         # Pre-install in all pooled containers
)

try:
    # Use pooled session
    # Libraries are installed during container initialization
    with SandboxSession(lang="python", pool=pool) as session:
        result = session.run("import pandas as pd; print(pd.__version__)")
finally:
    pool.close()
```

### Pool Manager Configuration

For shared pools across multiple sessions:

```python
from llm_sandbox.pool import create_pool_manager, PoolConfig, ExhaustionStrategy

# Create pool manager with custom configuration
pool = create_pool_manager(
    backend="docker",
    config=PoolConfig(
        # Size limits
        max_pool_size=20,
        min_pool_size=5,

        # Timeouts
        idle_timeout=300.0,
        acquisition_timeout=30.0,

        # Health and lifecycle
        health_check_interval=60.0,
        max_container_lifetime=3600.0,
        max_container_uses=100,

        # Exhaustion handling
        exhaustion_strategy=ExhaustionStrategy.WAIT,

        # Pre-warming
        enable_prewarming=True,
    ),
    lang="python",
    libraries=["requests", "numpy"],  # Pre-install in all pooled containers
)

# Use the shared pool
try:
    with SandboxSession(lang="python", pool=pool) as session:
        result = session.run("print('Hello from pool!')")
finally:
    pool.close()
```

### Pool Configuration Reference

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `max_pool_size` | int | 10 | Maximum containers in pool |
| `min_pool_size` | int | 0 | Minimum warm containers |
| `idle_timeout` | float \| None | 300.0 | Recycle idle containers (seconds) |
| `acquisition_timeout` | float \| None | 30.0 | Wait time for exhausted pool |
| `health_check_interval` | float | 60.0 | Health check frequency |
| `max_container_lifetime` | float \| None | 3600.0 | Maximum container lifetime |
| `max_container_uses` | int \| None | None | Maximum uses before recycling |
| `exhaustion_strategy` | ExhaustionStrategy | WAIT | Pool exhaustion behavior |
| `enable_prewarming` | bool | True | Create containers on startup |

!!! note "Installing Libraries"
    Libraries are pre-installed by passing the `libraries` parameter when creating the pool manager or session, not via PoolConfig.

### Exhaustion Strategies

Three strategies for handling pool exhaustion:

```python
from llm_sandbox.pool import ExhaustionStrategy

# 1. WAIT - Wait for available container (default)
PoolConfig(
    exhaustion_strategy=ExhaustionStrategy.WAIT,
    acquisition_timeout=30.0,
)

# 2. FAIL_FAST - Raise error immediately
PoolConfig(
    exhaustion_strategy=ExhaustionStrategy.FAIL_FAST,
)

# 3. TEMPORARY - Create temporary container
PoolConfig(
    exhaustion_strategy=ExhaustionStrategy.TEMPORARY,
)
```

!!! tip "Performance Tuning"
    - **High-frequency execution**: Use larger pools (max_pool_size=20+, min_pool_size=5+)
    - **Moderate usage**: Medium pools (max_pool_size=10, min_pool_size=2-3)
    - **Occasional usage**: Don't use pooling, create containers on-demand
    - **Pre-installed libraries**: Dramatically reduce execution time

See the [Container Pooling Guide](container-pooling.md) for advanced configuration, monitoring, and best practices.

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
