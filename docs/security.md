# Security Guide

LLM Sandbox combines runtime isolation with optional pre-execution code analysis. This guide explains which controls are enforced by the selected backend and which checks the caller must apply.

## Overview

Security in LLM Sandbox is implemented through multiple layers:

1. **Runtime Isolation** - Container isolation or local Linux namespaces and mounts
2. **Security Policies** - Pre-execution regex-based code analysis (this module's focus)
3. **Resource Limits** - Configure timeouts and backend-specific resource limits
4. **Network Controls** - Limit network access through the selected backend
5. **File System Restrictions** - Control which host files the sandbox can read or write
6. **Syscall Filtering** - Apply seccomp with supported Linux backends

> **Note**: Configure Docker/Podman controls through `runtime_configs`, Kubernetes controls through pod manifests and cluster policy, and local Linux controls through `LocalSandboxConfig`. See the [Configuration Guide](configuration.md).

### Security Model

LLM Sandbox follows an **advisory security model** for code analysis:

- **Runtime isolation is the primary security boundary.** Its strength and scope depend on the backend, host, exposed data, and configuration. Local namespaces and ordinary Linux containers share the host kernel.
- **Security policies are advisory.** The `SecurityPolicy` system provides pre-execution code analysis via `is_safe()`, but it is **not** automatically enforced by `run()`. You must explicitly check `is_safe()` and decide whether to proceed. This gives you full control over how violations are handled (logging, custom errors, partial execution, etc.).
- **OS and runtime controls are enforced during execution.** These are separate from source-code screening and apply to subprocesses and dynamically generated code as well.

For deployments handling untrusted code, choose the isolation boundary for the threat model, restrict data and network access, and apply appropriate resource controls. Code screening alone is not a sandbox.

## Local Linux Isolation and Seccomp

Bubblewrap and NsJail compose namespaces, restricted mounts, a clean environment, and a seccomp filter. Their default configuration disables network access and makes only the private session workspace and temporary directory writable on the host. Selected runtime paths are readable, including `/usr`; read-only access still allows the sandbox to read and copy any data in those paths. Do not add directories containing secrets to `read_only_paths`.

Each command starts in a separate PID namespace. Its background processes do not survive into the next command. Workspace files, installed packages, and private `/tmp` files do persist for the lifetime of the session. File-transfer APIs are limited to the workspace and private `/tmp` and reject unsafe paths and unsupported file types. Artifact extraction can read bounded regular files from the session's artifact directories.

`SeccompPolicy` is an OS-enforced syscall denylist. The default rejects operations such as `ptrace`, cross-process memory access, mount changes, `bpf`, kernel module management, keyring access, and `io_uring`. Ordinary file and socket operations are still allowed by the filter; namespace and mount policy control their reachable resources. This is additional hardening, not a complete syscall allowlist or a standalone security boundary.

Bubblewrap receives compiled BPF from system libseccomp. The compiler supports native x86-64 and AArch64 and rejects foreign syscall ABIs. NsJail uses its native Kafel compiler for the same denied operations, translating syscall aliases such as `umount2` to Kafel's spelling. Both filters also reject the legacy x32 syscall-number range 512–547 on x86-64, preventing the unmarked x32 encodings accepted by kernels older than 5.4. This architecture guard remains active with a custom denylist. Unknown or unsupported policy elements fail rather than being silently removed.

NsJail always selects `--experimental_mnt new` so read-only grants cover nested mounts as well as their parents. The required binary and kernel support must be present; the adapter never retries with the legacy mount implementation. Its `/dev/shm` is private to each command and supports shared memory between that command's child processes.

```python
from llm_sandbox import LocalSandboxConfig, SandboxSession, SeccompPolicy
from llm_sandbox.security.seccomp import DEFAULT_BLOCKED_SYSCALLS

# Extend the default denylist; a custom list replaces it.
config = LocalSandboxConfig(
    network="none",
    seccomp=SeccompPolicy(
        blocked_syscalls=(*DEFAULT_BLOCKED_SYSCALLS, "getppid"),
    ),
)

with SandboxSession(backend="bubblewrap", lang="python", sandbox_config=config) as session:
    print(session.run("print('Sandboxed with a custom syscall policy')").stdout)
```

Disabling the filter with `seccomp=None` is an explicit reduction in protection. Missing dependencies or policy failures never trigger an automatic retry without seccomp or outside the sandbox. `network="host"` is also an explicit grant: it enables unrestricted host networking, including reachable local services, and exposes resolver and certificate files read-only for DNS/TLS. There is no built-in domain allowlist or filtering proxy.

### Resource and Deployment Limits

- Timeouts and output caps bound command execution and captured output. Optional `prlimit` settings enforce per-process POSIX resource limits inherited by child processes.
- `memory_limit_bytes` limits virtual address space, not resident memory or total session memory. `cpu_time_limit` is CPU time per process. `file_size_limit` limits an individual file, not total disk use. These settings are not aggregate cgroup quotas and do not provide a process-count limit.
- NsJail timeout cleanup first asks its supervisor to kill the jailed process tree. If the supervisor is unresponsive and must itself be force-killed, descendant cleanup cannot be guaranteed without an external aggregate process manager such as cgroups.
- Both local adapters share the host kernel. A syscall denylist reduces exposed operations but does not turn a local sandbox into a VM or establish a strong multi-tenant hosting guarantee.
- The clean child environment excludes inherited host credentials; explicitly supplied environment values and read-only mounts remain intentional grants to sandboxed code.
- Python environments are created offline with system site packages visible from the mounted runtime. Dependencies and package install scripts execute inside the same sandbox. Host runtime installation and updates remain the operator's responsibility.

For managed hostile multi-tenant workloads, evaluate a stronger boundary such as a userspace kernel or VM alongside per-job storage and aggregate resource controls. Those backends are roadmap candidates, not currently implemented adapters.

## Windows Isolation

The experimental Windows backends use different OS boundaries. Both require an explicit backend selection and fail when the requested protection cannot be established. Capability metadata is not a live host probe.

### Native MXC/PSEC

`windows-mxc` invokes the pinned Microsoft MXC SDK's native PSEC runner directly. It does not use a generic dispatcher that could choose AppContainer, restricted tokens, or an unsandboxed fallback. An actual container-create probe checks availability, and explicit deny paths additionally require the native deny-path capability. The helper disables SDK DACL mutation; it does not provision users or change host directory ACLs to make execution succeed.

The policy grants read access to Windows and selected runtime files, plus explicit read-only paths. One private session directory is writable. File-transfer code rejects reparse points, junctions, hardlinks, device/UNC paths and alternate data streams; file operations verify handles instead of trusting a path string alone. The session root has a private DACL. These controls protect transfers and cleanup as well as execution.

`network="none"` denies egress, ingress and host loopback. `network="host"` enables all three and is a broad grant, not proxy enforcement. The helper permits registry reads needed by supported console runtimes; **this backend does not promise registry-read confidentiality**. Clipboard access, input injection and desktop/system control remain restricted. Native MXC is a process boundary sharing the Windows kernel, not a VM and not Linux seccomp.

The native launcher creates the payload suspended, assigns a kill-on-close Job Object, then resumes it. Descendants are terminated on foreground completion or helper death. The Python supervisor separately owns the helper's process tree, timeouts and output bounds. Session files persist between commands; background processes do not. The initial backend has no aggregate resource quotas.

The native helper, selected executables, runtime installations and explicit read grants are trusted inputs. They must not be writable by the sandbox. The helper receives a small bootstrap environment; only explicitly constructed payload variables reach the child. Runtime dependencies already present in read-only grants remain visible, and credentials intentionally passed in environment values are accessible to payload code.

### WSL2 Bubblewrap

`wsl2-bubblewrap` uses WSL2's Linux guest plus Bubblewrap namespaces, restricted mounts and seccomp inside that guest. Sessions share the existing distribution and guest kernel; the library does not start a separate VM per command. WSL1 is rejected. The native Windows client uses a trusted system `wsl.exe`, explicit arguments, and an isolated guest Python invocation. The worker handshake checks protocol, installed package version and the relevant security-source fingerprint.

WSL interoperability is a separate escape surface: Windows mounts and interop sockets can otherwise let Linux processes act on Windows with the user's privileges. This adapter restricts runtime grants to built-in Linux paths, rejects Windows/shared mount exposure, removes interop environment variables, and keeps the default seccomp restrictions mandatory. It hides standard `/usr/lib/wsl` integration mounts behind an empty read-only mount. It does not mount `/mnt`, `/run`, or `/init` into the payload. Granting `network="host"` still permits connections to reachable Windows services; filesystem isolation does not imply network isolation under that setting.

The worker owns each Bubblewrap process tree. Control messages are bounded and versioned; payload output is encoded as data and never interpreted as control instructions. Heartbeat expiry or transport EOF closes the guest session, because killing the Windows `wsl.exe` process alone is insufficient to prove Linux cleanup. The lease adds a bounded detection delay after an abrupt client failure. The trusted guest installation and distribution configuration remain part of the security boundary.

Portable tests and compilation validate contracts and API compatibility. Kernel filesystem/network enforcement, WSL interoperability restrictions and cancellation must also pass native integration checks on suitable Windows hosts before a deployment relies on them. See [Windows validation](backends.md#windows-validation).

## Library names: accepted formats and limitations

Whenever you pass `libraries=[...]` to `session.run()`, `session.install()`, or a pool's pre-install list, each entry is validated against a per-language allowlist before the install command is built. The install command is assembled by interpolating into a shell string (e.g. `pip install {lib}`), so a value like `foo; rm -rf /` would otherwise execute inside the container before any container limits could intervene. The validator rejects identifiers that fall outside the canonical naming spec for the target ecosystem and raises `llm_sandbox.exceptions.ValidationError`.

This is defence in depth on top of the container boundary, not a replacement for it.

### What's accepted

| Language | Accepted form |
|----------|---------------|
| `python` | PEP 508 name with optional `[extras]` and one or more PEP 440 specifiers (`==`, `>=`, `~=`, …). E.g. `requests`, `requests[security]`, `requests>=2.0,<3.0`. |
| `javascript` / `node` | npm package name, optionally `@scope/name`, optionally `@version-or-tag`. E.g. `lodash`, `@types/node`, `axios@1.7.2`. |
| `go` | Module path, optionally `@version` (semver, branch, or `vX.Y.Z-yyyymmddhhmmss-hash` pseudo-version). E.g. `github.com/spyzhov/ajson@v0.9.0`. |
| `ruby` | Gem name, optionally `:version`. E.g. `rails`, `nokogiri:1.16.0`. |
| `r` | CRAN name: must start with a letter, then letters/digits/dots only (no `_`, no `-`). E.g. `ggplot2`, `data.table`. |
| `c` / `cpp` | Debian binary package name (per Policy §5.6.7): lowercase letters, digits, `.`, `+`, `-`. E.g. `libboost-all-dev`. |
| `java` | Maven coordinates `group:artifact[:version]`. (The Java handler does not currently install at runtime, but the pattern is registered.) |

For the exact regexes, see `PACKAGE_NAME_PATTERNS` in `llm_sandbox/security/package_validators.py`.

### Known limitations

The patterns are deliberately strict — false rejections are a one-line workaround for the caller, false acceptances are an injection. Concretely, the following forms are **not** accepted:

- VCS install URLs: `git+https://github.com/x/y.git`, `https://github.com/x/y/archive/main.tar.gz`. Out of scope for now.
- Local paths: `./mypkg`, `/tmp/wheels/foo.whl`.
- Python: PEP 508 environment markers (`requests; python_version<'3.10'`) and direct URL references (`requests @ https://...`).
- Go: replace directives — `go get` doesn't accept them on the command line either.

If you need one of these, install it from inside the executed code (e.g. `subprocess.run(["pip", "install", "git+..."])`) where you control the call site, or extend the allowlist for your language.

### Adding or extending a language

Add a compiled regex to `PACKAGE_NAME_PATTERNS` in `llm_sandbox/security/package_validators.py`. Languages without an explicit entry fall back to a strict safe-charset check rather than free-pass.

## Security Policy System

### Overview

The security policy system analyzes code **before execution** using regex pattern matching to detect potentially dangerous operations. It provides:

- **Pattern-based detection** of dangerous code constructs
- **Language-specific module restriction** based on import statements
- **Severity-based filtering** with configurable thresholds
- **Comment filtering** to avoid false positives from documentation

> **Important**: Security policies are an **advisory code analysis tool**, not an automatic enforcement layer. You **must** call `session.is_safe(code)` manually before `session.run(code)` and act on the result. Calling `session.run(code)` directly will execute the code regardless of any configured security policy. See [Basic Usage](#basic-usage) for the correct pattern.

### Understanding Security Policies

```python
from llm_sandbox.security import (
    SecurityPolicy,
    SecurityPattern,
    RestrictedModule,
    SecurityIssueSeverity
)

# Create a basic security policy
policy = SecurityPolicy(
    severity_threshold=SecurityIssueSeverity.MEDIUM,
    patterns=[
        SecurityPattern(
            pattern=r"os\.system\s*\(",
            description="System command execution",
            severity=SecurityIssueSeverity.HIGH
        )
    ],
    restricted_modules=[
        RestrictedModule(
            name="os",
            description="Operating system interface",
            severity=SecurityIssueSeverity.HIGH
        )
    ]
)
```

### Severity Levels

Security issues are classified by severity with configurable blocking thresholds:

| Level | Value | Description | Example Use Case |
|-------|-------|-------------|------------------|
| `SAFE` | 0 | No security concerns | Allow everything |
| `LOW` | 1 | Minor concerns | Development environments |
| `MEDIUM` | 2 | Moderate risk | Production with controlled access |
| `HIGH` | 3 | High risk | Strict security requirements |

**Threshold Behavior**: Setting `severity_threshold=SecurityIssueSeverity.MEDIUM` will block MEDIUM, HIGH violations but allow LOW, SAFE patterns.

### Security Patterns

Define regex patterns to detect dangerous code constructs:

```python
# System command execution
SecurityPattern(
    pattern=r"\bos\.system\s*\(",
    description="System command execution",
    severity=SecurityIssueSeverity.HIGH
)

# Dynamic code evaluation
SecurityPattern(
    pattern=r"\beval\s*\(",
    description="Dynamic code evaluation",
    severity=SecurityIssueSeverity.MEDIUM
)

# File write operations
SecurityPattern(
    pattern=r"\bopen\s*\([^)]*['\"][wa]['\"][^)]*\)",
    description="File write operations",
    severity=SecurityIssueSeverity.MEDIUM
)

# Network socket creation
SecurityPattern(
    pattern=r"\bsocket\.socket\s*\(",
    description="Raw socket creation",
    severity=SecurityIssueSeverity.MEDIUM
)
```

### Restricted Modules

Block dangerous modules using **language-specific detection**. Simply specify the module name - the language handler automatically generates appropriate patterns to detect various import styles:

```python
# Block dangerous system access modules
RestrictedModule(
    name="os",
    description="Operating system interface",
    severity=SecurityIssueSeverity.HIGH
)

RestrictedModule(
    name="subprocess",
    description="Process execution",
    severity=SecurityIssueSeverity.HIGH
)

# Block networking modules
RestrictedModule(
    name="socket",
    description="Network operations",
    severity=SecurityIssueSeverity.MEDIUM
)

RestrictedModule(
    name="requests",
    description="HTTP library",
    severity=SecurityIssueSeverity.MEDIUM
)
```

**How Language-Specific Detection Works**:

When you specify a restricted module like `"os"`, the language handler automatically generates patterns to detect:

**For Python**:

- `import os`
- `import os as operating_system`
- `from os import system`
- `from os import system, environ`
- `from os import system as sys_call`

**For JavaScript**:

- `import os from 'os'`
- `const os = require('os')`
- `import { exec } from 'child_process'`

**For Other Languages**: Each language handler implements its own import detection patterns appropriate for that language's syntax.

## Container Hardening That Works

Security policies are advisory (see above). The layer the runtime actually
enforces is `runtime_configs`. This is the strongest set verified to work with
`SandboxSession`:

```python
HARDENED = {
    "network_mode": "none",                      # no egress
    "mem_limit": "512m",
    "pids_limit": 128,                           # bounds fork bombs
    "cap_drop": ["ALL"],
    "cap_add": ["DAC_OVERRIDE"],                 # see note below
    "security_opt": ["no-new-privileges:true"],
}

with SandboxSession(lang="python", runtime_configs=HARDENED) as session:
    result = session.run(code)
```

Verified inside the container: `CapEff` is `0000000000000002` (DAC_OVERRIDE
only) and outbound connections fail.

### Two caveats specific to this library

**`cap_drop: ["ALL"]` needs `DAC_OVERRIDE` added back.** The session copies your
code into the container before running it. Without `DAC_OVERRIDE` the container
cannot read that file and every run fails with `[Errno 13] Permission denied`.
Dropping everything else still removes `SYS_ADMIN`, `NET_RAW` and the rest.

**`read_only: True` does not work.** Docker rejects the code copy against a
read-only root filesystem:

```
docker.errors.APIError: 400 Client Error ...
("container rootfs is marked read-only")
```

Mounting a tmpfs on the workdir does not help — the rejection is on the rootfs
flag itself, not the destination directory. If you need an immutable filesystem,
pair the library with a runtime that provides it rather than setting this flag.

**`network_mode: "none"` disables `libraries=`.** Package installation needs the
network. Pre-bake dependencies into a custom image instead — see
[Custom Images](custom-images.md).


## Creating Security Policies

### Basic Usage

Security policies require a **manual check-then-execute** pattern. You must call `is_safe()` before `run()` — the policy is **not** automatically enforced.

```python
from llm_sandbox import SandboxSession
from llm_sandbox.security import SecurityPolicy, RestrictedModule, SecurityIssueSeverity

# Create a simple policy
policy = SecurityPolicy(
    severity_threshold=SecurityIssueSeverity.MEDIUM,
    restricted_modules=[
        RestrictedModule(
            name="os",
            description="Operating system interface",
            severity=SecurityIssueSeverity.HIGH
        ),
        RestrictedModule(
            name="subprocess",
            description="Process execution",
            severity=SecurityIssueSeverity.HIGH
        )
    ]
)

with SandboxSession(lang="python", security_policy=policy) as session:
    code = "import os\nos.system('ls')"

    # Step 1: Check if code is safe (REQUIRED — not automatic)
    is_safe, violations = session.is_safe(code)

    # Step 2: Only execute if safe
    if is_safe:
        result = session.run(code)
        print(result.stdout)
    else:
        print("Code failed security check:")
        for violation in violations:
            print(f"  - {violation.description} (Severity: {violation.severity.name})")
```

> **Warning**: `session.run(code)` does **not** enforce the security policy. It executes the code regardless of any configured policy. You **must** either call `session.is_safe(code)` first and act on the result, or use the enforced path — `session.safe_run(code)` or `session.run(code, enforce_security_policy=True)` — described below.

### Enforcing the security policy

The advisory check-then-execute pattern is easy to skip. For an enforced path that refuses to run code that violates the policy, pass `enforce_security_policy=True` to `run()`, or call the `safe_run()` wrapper. Either way the policy is evaluated **before any container interaction**, so policy-blocked code never pays the container-spinup cost.

```python
from llm_sandbox import SandboxSession
from llm_sandbox.exceptions import SecurityPolicyViolation
from llm_sandbox.security import RestrictedModule, SecurityIssueSeverity, SecurityPolicy

policy = SecurityPolicy(
    severity_threshold=SecurityIssueSeverity.HIGH,
    restricted_modules=[
        RestrictedModule(
            name="subprocess",
            description="Process execution",
            severity=SecurityIssueSeverity.HIGH,
        ),
    ],
)

with SandboxSession(lang="python", security_policy=policy) as session:
    code = "import subprocess\nsubprocess.run(['ls'])"

    try:
        # Either form works - they are equivalent.
        result = session.run(code, enforce_security_policy=True)
        # result = session.safe_run(code)
        print(result.stdout)
    except SecurityPolicyViolation as exc:
        print(f"Blocked at {exc.severity_threshold.name}:")
        for v in exc.violations:
            print(f"  - {v.description} ({v.severity.name})")
```

The exception is `llm_sandbox.exceptions.SecurityPolicyViolation` (a subclass of `SecurityViolationError`). It carries:

- `violations`: a `list[SecurityPattern]` of every matched pattern, including patterns auto-generated from `RestrictedModule` entries. Each item has `pattern`, `description`, and `severity`.
- `severity_threshold`: the `SecurityIssueSeverity` the policy was configured with at the time execution was blocked.

`enforce_security_policy` defaults to `False`, so existing callers that already use the manual `is_safe()`-then-`run()` pattern keep working unchanged.

**Recommended for production.** Wiring an LLM-driven agent into `safe_run()` (or `run(..., enforce_security_policy=True)`) means a forgotten `is_safe()` check never silently lets a bad payload through.

### Custom Security Policies

```python
# Create comprehensive custom policy
custom_policy = SecurityPolicy(
    severity_threshold=SecurityIssueSeverity.MEDIUM,
    patterns=[
        # Block cloud SDKs
        SecurityPattern(
            pattern=r"\b(boto3|google\.cloud|azure)\b",
            description="Cloud SDK usage",
            severity=SecurityIssueSeverity.HIGH
        ),
        # Block specific domains
        SecurityPattern(
            pattern=r"requests\.(get|post)\s*\(['\"].*internal\.company\.com",
            description="Internal network access",
            severity=SecurityIssueSeverity.HIGH
        ),
        # Monitor external APIs
        SecurityPattern(
            pattern=r"requests\.(get|post)\s*\(",
            description="External API call",
            severity=SecurityIssueSeverity.LOW
        )
    ],
    restricted_modules=[
        RestrictedModule(
            name="psutil",
            description="System monitoring",
            severity=SecurityIssueSeverity.MEDIUM
        ),
        RestrictedModule(
            name="ctypes",
            description="Foreign function library",
            severity=SecurityIssueSeverity.HIGH
        )
    ]
)
```

### Dynamic Policy Modification

```python
# Start with base policy and customize
policy = SecurityPolicy(
    severity_threshold=SecurityIssueSeverity.MEDIUM,
    patterns=[],
    restricted_modules=[]
)

# Add custom patterns
policy.add_pattern(SecurityPattern(
    pattern=r"\b(tensorflow|torch|keras)\b",
    description="ML framework usage",
    severity=SecurityIssueSeverity.LOW
))

# Add restricted modules (language handler will generate detection patterns)
policy.add_restricted_module(RestrictedModule(
    name="cryptography",
    description="Cryptographic operations",
    severity=SecurityIssueSeverity.MEDIUM
))
```

## Language-Specific Examples

### Python Security Patterns

```python
# Python-specific dangerous patterns
python_patterns = [
    # Dynamic imports
    SecurityPattern(r"\b__import__\s*\(", "Dynamic imports", SecurityIssueSeverity.MEDIUM),

    # Attribute manipulation
    SecurityPattern(r"\b(getattr|setattr|delattr)\s*\(", "Dynamic attributes", SecurityIssueSeverity.LOW),

    # Pickle operations (deserialization risk)
    SecurityPattern(r"\bpickle\.(loads?|load)\s*\(", "Pickle deserialization", SecurityIssueSeverity.MEDIUM),

    # Code execution
    SecurityPattern(r"\b(eval|exec|compile)\s*\(", "Code execution", SecurityIssueSeverity.HIGH)
]

# Python-specific restricted modules
python_modules = [
    RestrictedModule("os", "Operating system interface", SecurityIssueSeverity.HIGH),
    RestrictedModule("subprocess", "Process execution", SecurityIssueSeverity.HIGH),
    RestrictedModule("ctypes", "Foreign function library", SecurityIssueSeverity.HIGH),
    RestrictedModule("importlib", "Dynamic imports", SecurityIssueSeverity.MEDIUM)
]
```

### JavaScript/Node.js Security Patterns

```python
# JavaScript-specific patterns (when lang="javascript")
js_patterns = [
    # Process access
    SecurityPattern(r"process\.exit\s*\(", "Process termination", SecurityIssueSeverity.HIGH),

    # Child processes
    SecurityPattern(r"child_process", "Child process access", SecurityIssueSeverity.HIGH),

    # File system
    SecurityPattern(r"fs\.(writeFile|unlink)\s*\(", "File system operations", SecurityIssueSeverity.MEDIUM)
]

# JavaScript-specific restricted modules
js_modules = [
    RestrictedModule("fs", "File system access", SecurityIssueSeverity.MEDIUM),
    RestrictedModule("child_process", "Process execution", SecurityIssueSeverity.HIGH),
    RestrictedModule("cluster", "Process clustering", SecurityIssueSeverity.HIGH)
]
```

> **Note**: Security presets (like `get_security_policy("production")`) will be introduced in future versions and will be language-specific to provide appropriate defaults for each programming language.

## Advanced Pattern Examples

### Network Security Patterns

```python
# Monitor and control network operations
network_patterns = [
    SecurityPattern(
        pattern=r"\bsocket\.socket\s*\(",
        description="Raw socket creation",
        severity=SecurityIssueSeverity.MEDIUM
    ),
    SecurityPattern(
        pattern=r"\b\w+\.connect\s*\(",
        description="Network connections",
        severity=SecurityIssueSeverity.MEDIUM
    ),
    SecurityPattern(
        pattern=r"requests\.(get|post|put|delete)\s*\(",
        description="HTTP requests",
        severity=SecurityIssueSeverity.LOW
    )
]
```

### File System Security Patterns

```python
# File system operation patterns
file_patterns = [
    SecurityPattern(
        pattern=r"\bopen\s*\([^)]*['\"][wa]['\"][^)]*\)",
        description="File write operations",
        severity=SecurityIssueSeverity.MEDIUM
    ),
    SecurityPattern(
        pattern=r"\bos\.(remove|unlink|rmdir)\s*\(",
        description="File deletion operations",
        severity=SecurityIssueSeverity.HIGH
    ),
    SecurityPattern(
        pattern=r"\bshutil\.(rmtree|move|copy)\s*\(",
        description="File system manipulation",
        severity=SecurityIssueSeverity.MEDIUM
    )
]
```

## Security Implementation Details

### Comment Filtering

The security scanner filters comments to avoid false positives:

```python
# This comment won't trigger security alerts: import os
print("This string mentioning 'os.system' won't trigger alerts either")

# But this will be detected:
import os
os.system('whoami')
```

### Pattern Matching Process

1. **Filter Comments**: Remove comments using language-specific handlers
2. **Generate Module Patterns**: Convert restricted modules to regex patterns via language handlers
3. **Apply Patterns**: Match all patterns against filtered code
4. **Severity Check**: Apply severity threshold to determine blocking
5. **Return Results**: Report safety status and violations

### Example Workflow

```python
# Code to analyze
code = """
import os  # This imports the OS module
# os.system('commented out') - this won't be detected
os.system('whoami')  # This will be detected
"""

# Policy with os module restricted
policy = SecurityPolicy(
    severity_threshold=SecurityIssueSeverity.MEDIUM,
    restricted_modules=[
        RestrictedModule("os", "OS interface", SecurityIssueSeverity.HIGH)
    ],
    patterns=[
        SecurityPattern(r"os\.system\s*\(", "System commands", SecurityIssueSeverity.HIGH)
    ]
)

# Analysis process:
# 1. Filter comments -> "import os\nos.system('whoami')"
# 2. Language handler generates pattern for "os" import -> VIOLATION (HIGH)
# 3. Check os.system pattern -> VIOLATION (HIGH)
# 4. Result: is_safe=False, violations=[2]
```

## Best Practices

### 1. Layer Security Policies with Container Controls

```python
# Combine security policy with container restrictions
with SandboxSession(
    lang="python",
    # Code analysis layer
    security_policy=custom_policy,
    # Container restriction layer (see Configuration Guide)
    runtime_configs={
        "mem_limit": "256m",
        "cpu_count": 1,
        "timeout": 30,
        "user": "nobody:nogroup"
    }
) as session:
    # Always check before executing
    is_safe, violations = session.is_safe(user_code)
    if is_safe:
        result = session.run(user_code)
    # Container limits provide defense-in-depth even if code analysis is bypassed
```

### 2. Use Language-Appropriate Restrictions

```python
# Python-focused restrictions
python_policy = SecurityPolicy(
    severity_threshold=SecurityIssueSeverity.MEDIUM,
    restricted_modules=[
        RestrictedModule("os", "Operating system", SecurityIssueSeverity.HIGH),
        RestrictedModule("subprocess", "Process execution", SecurityIssueSeverity.HIGH),
        RestrictedModule("ctypes", "Foreign functions", SecurityIssueSeverity.HIGH),
        RestrictedModule("importlib", "Dynamic imports", SecurityIssueSeverity.MEDIUM),
        RestrictedModule("pickle", "Serialization", SecurityIssueSeverity.MEDIUM)
    ]
)

# JavaScript-focused restrictions (when available)
javascript_policy = SecurityPolicy(
    severity_threshold=SecurityIssueSeverity.MEDIUM,
    restricted_modules=[
        RestrictedModule("fs", "File system", SecurityIssueSeverity.MEDIUM),
        RestrictedModule("child_process", "Process execution", SecurityIssueSeverity.HIGH),
        RestrictedModule("cluster", "Process clustering", SecurityIssueSeverity.HIGH),
        RestrictedModule("worker_threads", "Threading", SecurityIssueSeverity.MEDIUM)
    ]
)
```

### 3. Test Security Policies

```python
def test_security_policy(policy: SecurityPolicy, lang: str = "python"):
    """Test security policy effectiveness"""
    test_cases = [
        # Should pass
        ("print('Hello')", True),
        ("x = 1 + 2", True),

        # Should fail based on restricted modules
        ("import os; os.system('ls')", False),
        ("eval('malicious_code')", False),

        # Edge cases
        ("# import os", True),  # Comment
        ("'import os in string'", True),  # String literal
    ]

    with SandboxSession(lang=lang, security_policy=policy) as session:
        for code, expected_safe in test_cases:
            is_safe, _ = session.is_safe(code)
            assert is_safe == expected_safe, f"Failed for: {code}"
```

### 4. Monitor and Log Security Events

```python
import logging
import hashlib
from datetime import datetime

class SecurityAuditor:
    def __init__(self):
        self.logger = logging.getLogger('security_audit')

    def audit_execution(self, code: str, user_id: str, is_safe: bool, violations: list):
        """Log security events for monitoring"""
        audit_entry = {
            'timestamp': datetime.utcnow().isoformat(),
            'user_id': user_id,
            'code_hash': hashlib.sha256(code.encode()).hexdigest(),
            'code_length': len(code),
            'is_safe': is_safe,
            'violations': [v.description for v in violations]
        }

        if violations:
            self.logger.warning(f"Security violations detected: {audit_entry}")
        else:
            self.logger.info(f"Safe code execution: {audit_entry}")

# Usage
auditor = SecurityAuditor()
policy = SecurityPolicy(
    severity_threshold=SecurityIssueSeverity.MEDIUM,
    restricted_modules=[
        RestrictedModule("os", "Operating system", SecurityIssueSeverity.HIGH)
    ]
)

with SandboxSession(lang="python", security_policy=policy) as session:
    is_safe, violations = session.is_safe(user_code)
    auditor.audit_execution(user_code, user_id, is_safe, violations)

    if is_safe:
        result = session.run(user_code)
```

### 5. Handle Security Violations Gracefully

```python
class SecurityViolationError(Exception):
    def __init__(self, violations: list):
        self.violations = violations
        super().__init__(f"Security policy violations: {[v.description for v in violations]}")

def safe_execute(code: str, user_id: str, lang: str = "python") -> ExecutionResult:
    """Execute code with comprehensive security handling"""

    # Input validation
    if len(code) > 50000:  # 50KB limit
        raise ValueError("Code too long")

    if '\x00' in code:
        raise ValueError("Invalid null bytes in code")

    # Security check
    policy = SecurityPolicy(
        severity_threshold=SecurityIssueSeverity.MEDIUM,
        restricted_modules=[
            RestrictedModule("os", "Operating system", SecurityIssueSeverity.HIGH),
            RestrictedModule("subprocess", "Process execution", SecurityIssueSeverity.HIGH)
        ]
    )

    with SandboxSession(lang=lang, security_policy=policy) as session:
        is_safe, violations = session.is_safe(code)

        if not is_safe:
            # Log security violation
            logging.warning(f"Security violation by user {user_id}: {[v.description for v in violations]}")
            raise SecurityViolationError(violations)

        # Execute safely
        try:
            result = session.run(code)
            logging.info(f"Successful execution by user {user_id}")
            return result
        except Exception as e:
            logging.error(f"Execution error for user {user_id}: {e}")
            raise
```

## Troubleshooting

### Common Issues

#### False Positives in Strings/Comments
```python
# Problem: Security scanner detects patterns in strings
code = 'print("Use os.system() carefully")'  # May be blocked

# Solution: The scanner automatically filters comments and should handle strings
# If issues persist, adjust patterns to be more specific
```

#### Import Detection Edge Cases
```python
# Detected patterns (via language handler):
import os                    # ✓ Detected
from os import system        # ✓ Detected
import os as operating_sys   # ✓ Detected

# May not be detected (advanced evasion):
__import__('os')             # Depends on dynamic import patterns
importlib.import_module('os') # Requires specific patterns
```

#### Performance with Large Codebases
```python
# For very large code files, consider:
# 1. Setting reasonable size limits
# 2. Using more specific patterns
# 3. Implementing caching for repeated analysis
```

### Debugging Security Policies

```python
# Enable verbose logging to understand policy behavior
import logging
logging.getLogger('llm_sandbox').setLevel(logging.DEBUG)

# Test individual patterns
pattern = SecurityPattern(r"os\.system\s*\(", "Test", SecurityIssueSeverity.HIGH)
test_code = "os.system('test')"

import re
if re.search(pattern.pattern, test_code):
    print(f"Pattern matches: {pattern.description}")
```

## API Reference

For complete API documentation, see:

- **Docker**: [docker-py documentation](https://docker-py.readthedocs.io/en/stable/)
- **Podman**: [podman-py documentation](https://podman-py.readthedocs.io/en/latest/)
- **Kubernetes**: [kubernetes-client documentation](https://kubernetes.io/docs/reference/using-api/client-libraries/)

For container-level security configuration (resource limits, network controls, file system restrictions), see the [Configuration Guide](configuration.md).
