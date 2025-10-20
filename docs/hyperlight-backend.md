# Hyperlight Backend

## Overview

The Hyperlight backend provides a lightweight alternative to container-based execution using micro virtual machines (VMs). Unlike Docker, Kubernetes, or Podman which run full container environments, Hyperlight creates minimal VMs that boot instantly without an operating system.

## What is Hyperlight?

[Hyperlight](https://github.com/hyperlight-dev/hyperlight) is a library for creating micro virtual machines - or sandboxes - specifically optimized for securely running untrusted code with minimal impact. It supports both Windows and Linux using native hypervisors:

- **Linux**: KVM or Microsoft Hypervisor (mshv)
- **Windows**: Windows Hypervisor Platform (WHP) - requires Windows 11/Server 2022+

### Key Differences from Containers

| Feature | Containers (Docker/Podman) | Hyperlight Micro VMs |
|---------|---------------------------|---------------------|
| Isolation | Process-level with namespaces | Hardware-level with hypervisor |
| Startup Time | Milliseconds | Microseconds |
| Overhead | ~100MB+ per container | ~1MB per micro VM |
| State | Persistent between runs | Stateless (destroyed after each run) |
| Library Installation | Runtime installation | Compile-time only |
| File Operations | Full support | Not supported (stateless) |
| OS/Kernel | Full OS in container | No OS/kernel |

## Requirements

### System Requirements

**Linux:**
- KVM or mshv hypervisor support
- Rust toolchain (1.88+)
- LLVM/Clang 18+

**Windows:**
- Windows 11 or Windows Server 2022+
- Windows Hypervisor Platform enabled
- Rust toolchain (1.88+)
- LLVM/Clang

### Installation

1. **Install Rust:**
```bash
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
```

2. **Install LLVM (Linux):**
```bash
wget https://apt.llvm.org/llvm.sh
chmod +x ./llvm.sh
sudo ./llvm.sh 18 clang clang-tools-extra
sudo ln -s /usr/lib/llvm-18/bin/ld.lld /usr/bin/ld.lld
sudo ln -s /usr/lib/llvm-18/bin/clang /usr/bin/clang
```

3. **Verify KVM (Linux):**
```bash
ls -l /dev/kvm
# Ensure your user has permission to access /dev/kvm
```

4. **Install llm-sandbox:**
```bash
pip install llm-sandbox
```

## Basic Usage

### Simple Example

```python
from llm_sandbox import SandboxBackend, create_session

# Create a Hyperlight session
with create_session(
    backend=SandboxBackend.HYPERLIGHT,
    lang="python",
    verbose=True
) as session:
    # Note: First run will compile guest binary (takes ~5 minutes)
    # Subsequent runs will be much faster
    print("Hyperlight session created successfully!")
```

### Using Pre-compiled Guest Binary

For faster startup times, compile the guest binary once and reuse it:

```python
from llm_sandbox import SandboxBackend, create_session

with create_session(
    backend=SandboxBackend.HYPERLIGHT,
    lang="python",
    guest_binary_path="/path/to/compiled/guest",
    verbose=True
) as session:
    # Session starts immediately without compilation
    print("Fast startup with pre-compiled guest!")
```

## Architecture

### How It Works

1. **Guest Binary Compilation** (first run only):
   - User code is wrapped in a Hyperlight guest template
   - Template is compiled to x86_64-unknown-none target
   - Results in a statically-linked bare-metal binary (~1MB)

2. **VM Creation**:
   - Hyperlight provisions memory and configures regions
   - Creates a VM with the platform hypervisor
   - Maps guest binary into VM memory
   - Configures virtual CPU registers

3. **Execution**:
   - VM starts at the guest binary's entry point
   - Guest code executes directly on virtual CPU (no OS)
   - Results are returned via host function calls
   - VM is destroyed after execution completes

### Guest Binary Structure

Hyperlight guests are special programs that:
- Have no operating system or kernel
- Are statically linked with Hyperlight guest library
- Expose functions that the host can call
- Can call limited host functions (e.g., HostPrint)
- Run directly on virtual CPU

```rust
// Example guest structure
#![no_std]
#![no_main]

// Guest code uses Hyperlight guest library
use hyperlight_guest::*;
use hyperlight_guest_bin::*;

// Guest function that host can call
fn execute_code(code: &str) -> Result<String> {
    // Execute user code and return result
    // Call host functions as needed
}
```

## Current Limitations

### What's Not Supported (Yet)

1. **Dynamic Library Installation**
   - Libraries must be compiled into guest binary
   - No runtime `pip install` or `npm install`
   - Workaround: Rebuild guest with required libraries

2. **File Operations**
   - No `copy_to_runtime()` or `copy_from_runtime()`
   - VMs are stateless and destroyed after execution
   - Workaround: Pass data via function parameters

3. **Persistent Environment**
   - Each run creates a new VM
   - No state preserved between executions
   - Workaround: Use container backends for stateful workflows

4. **Limited Language Support**
   - Currently: Python only (experimental)
   - Requires guest template for each language
   - Future: JavaScript, Java, C++, Go support planned

5. **Compilation Overhead**
   - First run: ~5 minutes to compile guest binary
   - Subsequent runs: Fast (microseconds)
   - Workaround: Pre-compile guest binaries

## Performance Characteristics

### Startup Time

- **Container backends**: 100ms - 2s
- **Hyperlight**: < 1ms (with pre-compiled guest)
- **First run**: ~5 minutes (includes compilation)

### Memory Usage

- **Container backends**: 100MB - 500MB per container
- **Hyperlight**: ~1MB per micro VM

### Execution Overhead

- **Container backends**: Minimal (native process)
- **Hyperlight**: Minimal (hardware virtualization)

## Use Cases

### Ideal For:

✅ **Functions-as-a-Service (FaaS)**
- High request rates requiring instant VM creation
- Stateless execution model fits FaaS perfectly
- Minimal memory footprint for high density

✅ **Code Sandboxing**
- Maximum isolation via hardware virtualization
- Untrusted code execution with minimal risk
- No shared kernel vulnerabilities

✅ **Microsecond-latency Requirements**
- Trading systems
- Real-time analytics
- Edge computing

### Not Ideal For:

❌ **Interactive Development**
- Use Docker/Podman for interactive workflows
- Better support for library installation
- Persistent environments across runs

❌ **Complex Workflows**
- File operations not supported
- Use containers for multi-step pipelines
- Better integration with existing tools

❌ **Standard Deployments**
- Containers are more mature and widely supported
- Better tooling and ecosystem
- More predictable behavior

## Comparison with Other Backends

```python
from llm_sandbox import SandboxBackend, create_session

# Docker - Best for development and general use
with create_session(backend=SandboxBackend.DOCKER, lang="python") as session:
    # Full container environment
    # Runtime library installation
    # File operations supported
    session.install(["numpy", "pandas"])
    session.copy_to_runtime("data.csv", "/sandbox/data.csv")
    result = session.run("import pandas as pd; df = pd.read_csv('/sandbox/data.csv')")

# Kubernetes - Best for production at scale
with create_session(backend=SandboxBackend.KUBERNETES, lang="python") as session:
    # Pod-based execution
    # Enterprise orchestration
    # High availability
    result = session.run("print('Running in Kubernetes pod')")

# Podman - Best for rootless containers
with create_session(backend=SandboxBackend.PODMAN, lang="python") as session:
    # Rootless container execution
    # Enhanced security
    # Docker-compatible
    result = session.run("print('Running in Podman container')")

# Hyperlight - Best for high-density FaaS
with create_session(backend=SandboxBackend.HYPERLIGHT, lang="python") as session:
    # Microsecond VM startup
    # Minimal memory footprint
    # Maximum isolation
    # Stateless execution
    # (Note: Current implementation is experimental)
```

## Advanced Configuration

### Custom Guest Binary

Build a custom guest binary with your own dependencies:

```rust
// In your guest Cargo.toml
[package]
name = "my-custom-guest"
version = "0.1.0"
edition = "2021"

[dependencies]
hyperlight-guest = { git = "https://github.com/hyperlight-dev/hyperlight" }
hyperlight-guest-bin = { git = "https://github.com/hyperlight-dev/hyperlight" }
hyperlight-common = { git = "https://github.com/hyperlight-dev/hyperlight" }
# Add your dependencies here

[profile.release]
panic = "abort"
```

Then use it with llm-sandbox:

```python
with create_session(
    backend=SandboxBackend.HYPERLIGHT,
    lang="python",
    guest_binary_path="/path/to/my-custom-guest",
    keep_template=True
) as session:
    # Use your custom guest
    pass
```

### Security Considerations

Hyperlight provides strong isolation:

```python
from llm_sandbox import SandboxBackend, create_session
from llm_sandbox.security import SecurityPolicy

# Security policies still apply
security_policy = SecurityPolicy(
    patterns=[],
    restricted_modules=[]
)

with create_session(
    backend=SandboxBackend.HYPERLIGHT,
    lang="python",
    security_policy=security_policy
) as session:
    # Code is checked against security policy before execution
    result = session.run("print('Secure execution')")
```

## Troubleshooting

### "Rust toolchain required" Error

```bash
# Install Rust
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh

# Verify installation
cargo --version
rustc --version
```

### "No hypervisor found" Error

**Linux:**
```bash
# Check KVM availability
ls -l /dev/kvm

# Add user to kvm group
sudo usermod -a -G kvm $USER

# Log out and back in for changes to take effect
```

**Windows:**
```powershell
# Enable Windows Hypervisor Platform
Enable-WindowsOptionalFeature -Online -FeatureName HypervisorPlatform

# Restart required
```

### Compilation Timeout

If guest binary compilation times out:

1. Increase timeout in code
2. Pre-compile guest binary separately
3. Use pre-compiled binary with `guest_binary_path`

### Performance Issues

- **First run slow**: Expected (compilation)
- **Subsequent runs slow**: Check hypervisor availability
- **High memory usage**: Check for memory leaks in guest code

## Future Enhancements

Planned improvements for the Hyperlight backend:

1. **Full Code Execution**: Currently experimental
2. **Multi-language Support**: JavaScript, Java, C++, Go
3. **Library Management**: Improved dependency handling
4. **Caching**: Guest binary compilation caching
5. **Metrics**: Performance monitoring and profiling
6. **Examples**: More comprehensive examples

## References

- [Hyperlight GitHub](https://github.com/hyperlight-dev/hyperlight)
- [Hyperlight Documentation](https://github.com/hyperlight-dev/hyperlight/tree/main/docs)
- [KVM Documentation](https://linux-kvm.org/page/Main_Page)
- [Windows Hypervisor Platform](https://docs.microsoft.com/en-us/virtualization/api/)

## Contributing

The Hyperlight backend is experimental. Contributions are welcome!

Areas needing work:
- Guest templates for additional languages
- Code execution implementation
- Binary caching mechanism
- Integration tests
- Documentation improvements

See [CONTRIBUTING.md](../CONTRIBUTING.md) for guidelines.
