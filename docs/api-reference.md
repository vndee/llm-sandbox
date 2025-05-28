# API Reference

Comprehensive API documentation for LLM Sandbox.

## Core Classes

### SandboxSession

```python
class SandboxSession(
    backend: SandboxBackend = SandboxBackend.DOCKER,
    *args,
    **kwargs
)
```

Main class for creating and managing sandbox sessions.

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `backend` | `SandboxBackend` | `DOCKER` | Container backend to use |
| `lang` | `str` | `"python"` | Programming language |
| `image` | `str \| None` | `None` | Container image name |
| `dockerfile` | `str \| None` | `None` | Path to Dockerfile |
| `keep_template` | `bool` | `False` | Keep container image after session |
| `verbose` | `bool` | `False` | Enable verbose logging |
| `workdir` | `str \| None` | `"/sandbox"` | Working directory in container |
| `security_policy` | `SecurityPolicy \| None` | `None` | Security policy to enforce |
| `logger` | `logging.Logger \| None` | `None` | Custom logger instance |
| `runtime_configs` | `dict \| None` | `None` | Runtime configuration options |
| `mounts` | `list \| None` | `None` | Volume mounts (Docker/Podman) |
| `commit_container` | `bool` | `False` | Commit container state (Docker/Podman) |
| `client` | `Any \| None` | `None` | Backend-specific client instance |

**Methods:**

#### `run(code: str, libraries: list | None = None) -> ConsoleOutput`

Execute code in the sandbox.

```python
with SandboxSession(lang="python") as session:
    result = session.run("print('Hello')", libraries=["numpy"])
    print(result.stdout)
```

#### `install(libraries: list[str] | None = None) -> None`

Install libraries in the sandbox.

```python
session.install(["pandas", "matplotlib"])
```

#### `copy_to_runtime(src: str, dest: str) -> None`

Copy file from host to sandbox.

```python
session.copy_to_runtime("local_file.txt", "/sandbox/file.txt")
```

#### `copy_from_runtime(src: str, dest: str) -> None`

Copy file from sandbox to host.

```python
session.copy_from_runtime("/sandbox/output.txt", "output.txt")
```

#### `execute_command(command: str, workdir: str | None = None) -> ConsoleOutput`

Execute shell command in sandbox.

```python
result = session.execute_command("ls -la", workdir="/sandbox")
```

#### `is_safe(code: str) -> tuple[bool, list[SecurityPattern]]`

Check if code passes security policy.

```python
is_safe, violations = session.is_safe("import os; os.system('rm -rf /')")
if not is_safe:
    for violation in violations:
        print(f"Violation: {violation.description}")
```

---

### ArtifactSandboxSession

```python
class ArtifactSandboxSession(
    backend: SandboxBackend = SandboxBackend.DOCKER,
    enable_plotting: bool = True,
    **kwargs
)
```

Extended session with artifact extraction capabilities.

**Additional Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `enable_plotting` | `bool` | `True` | Enable plot extraction |

**Methods:**

#### `run(code: str, libraries: list | None = None) -> ExecutionResult`

Execute code and extract artifacts.

```python
with ArtifactSandboxSession(lang="python") as session:
    result = session.run("""
import matplotlib.pyplot as plt
plt.plot([1, 2, 3], [1, 4, 9])
plt.show()
    """)
    
    for plot in result.plots:
        print(f"Captured plot: {plot.format}")
```

---

## Data Classes

### ConsoleOutput

```python
@dataclass(frozen=True)
class ConsoleOutput:
    exit_code: int = 0
    stderr: str = ""
    stdout: str = ""
```

Represents command execution output.

**Attributes:**

- `exit_code`: Process exit code (0 = success)
- `stdout`: Standard output content
- `stderr`: Standard error content

**Methods:**

- `success() -> bool`: Returns True if exit_code is 0

### ExecutionResult

```python
@dataclass(frozen=True)
class ExecutionResult(ConsoleOutput):
    plots: list[PlotOutput] = field(default_factory=list)
```

Extended output with captured plots.

### PlotOutput

```python
@dataclass(frozen=True)
class PlotOutput:
    format: FileType
    content_base64: str
    width: int | None = None
    height: int | None = None
    dpi: int | None = None
```

Represents a captured plot/visualization.

---

## Security Classes

### SecurityPolicy

```python
class SecurityPolicy:
    severity_threshold: SecurityIssueSeverity = SecurityIssueSeverity.SAFE
    patterns: list[SecurityPattern] | None = None
    restricted_modules: list[DangerousModule] | None = None
```

Defines security rules for code execution.

**Methods:**

#### `add_pattern(pattern: SecurityPattern) -> None`

Add a security pattern.

```python
policy.add_pattern(SecurityPattern(
    pattern=r"requests\.get\(.*\.onion",
    description="Tor network access",
    severity=SecurityIssueSeverity.HIGH
))
```

#### `add_restricted_module(module: DangerousModule) -> None`

Add a restricted module.

```python
policy.add_restricted_module(DangerousModule(
    name="cryptography",
    description="Cryptographic operations",
    severity=SecurityIssueSeverity.MEDIUM
))
```

### SecurityPattern

```python
@dataclass
class SecurityPattern:
    pattern: str
    description: str
    severity: SecurityIssueSeverity
```

Regex pattern for detecting dangerous code.

### DangerousModule

```python
@dataclass
class DangerousModule:
    name: str
    description: str
    severity: SecurityIssueSeverity
```

Module that should be restricted.

### SecurityIssueSeverity

```python
class SecurityIssueSeverity(IntEnum):
    SAFE = 0
    LOW = 1
    MEDIUM = 2
    HIGH = 3
```

Severity levels for security issues.

---

## Enumerations

### SandboxBackend

```python
class SandboxBackend(StrEnum):
    DOCKER = "docker"
    KUBERNETES = "kubernetes"
    PODMAN = "podman"
    MICROMAMBA = "micromamba"
```

Available container backends.

### SupportedLanguage

```python
class SupportedLanguage(StrEnum):
    PYTHON = "python"
    JAVA = "java"
    JAVASCRIPT = "javascript"
    CPP = "cpp"
    GO = "go"
    RUBY = "ruby"
```

Supported programming languages.

### FileType

```python
class FileType(Enum):
    PNG = "png"
    JPEG = "jpeg"
    PDF = "pdf"
    SVG = "svg"
    CSV = "csv"
    JSON = "json"
    TXT = "txt"
    HTML = "html"
```

Supported file types for artifacts.

---

## Functions

### create_session

```python
def create_session(
    backend: SandboxBackend = SandboxBackend.DOCKER,
    *args,
    **kwargs
) -> Session
```

Factory function for creating sandbox sessions.

```python
session = create_session(
    backend=SandboxBackend.KUBERNETES,
    lang="python",
    kube_namespace="sandbox"
)
```

### get_security_policy

```python
def get_security_policy(preset_name: str) -> SecurityPolicy
```

Get a preset security policy.

**Available Presets:**
- `"minimal"`: Very permissive
- `"development"`: Balanced for development
- `"educational"`: For teaching environments
- `"production"`: Strict for production
- `"strict"`: Very restrictive
- `"data_science"`: Optimized for data analysis
- `"web_scraping"`: For web scraping tasks

```python
policy = get_security_policy("production")
```

---

## Exceptions

### Base Exception

```python
class SandboxError(Exception):
    """Base exception for all sandbox errors"""
```

### Specific Exceptions

| Exception | Description |
|-----------|-------------|
| `ContainerError` | Container operation failed |
| `SecurityError` | Security policy violation |
| `ResourceError` | Resource limit exceeded |
| `ValidationError` | Input validation failed |
| `LanguageNotSupportedError` | Unsupported language |
| `ImageNotFoundError` | Container image not found |
| `NotOpenSessionError` | Session not opened |
| `LibraryInstallationNotSupportedError` | Language doesn't support library installation |
| `CommandEmptyError` | Empty command provided |
| `CommandFailedError` | Command execution failed |
| `PackageManagerError` | Package manager not found |
| `ImagePullError` | Failed to pull container image |
| `UnsupportedBackendError` | Backend not supported |
| `MissingDependencyError` | Required dependency not installed |
| `LanguageNotSupportPlotError` | Language doesn't support plot extraction |
| `InvalidRegexPatternError` | Invalid regex in security pattern |
| `SecurityViolationError` | Code violates security policy |

---

## Language Handlers

### AbstractLanguageHandler

```python
class AbstractLanguageHandler(ABC):
    config: LanguageConfig
    logger: logging.Logger
```

Base class for language-specific handlers.

**Abstract Methods:**

- `get_import_patterns(module: str) -> str`
- `get_multiline_comment_patterns() -> str`
- `get_inline_comment_patterns() -> str`

**Methods:**

- `get_execution_commands(code_file: str) -> list[str]`
- `get_library_installation_command(library: str) -> str`
- `inject_plot_detection_code(code: str) -> str`
- `extract_plots(container: ContainerProtocol, output_dir: str) -> list[PlotOutput]`
- `filter_comments(code: str) -> str`

### LanguageConfig

```python
@dataclass
class LanguageConfig:
    name: str
    file_extension: str
    execution_commands: list[str]
    package_manager: str | None
    is_support_library_installation: bool = True
    plot_detection: PlotDetectionConfig | None = None
```

Configuration for a language handler.

---

## Backend-Specific APIs

### Docker Backend

```python
class SandboxDockerSession(Session):
    client: docker.DockerClient
    container: docker.models.containers.Container
    docker_image: docker.models.images.Image
```

**Docker-Specific Parameters:**
- `stream`: Stream command output
- `commit_container`: Save container state
- `mounts`: Docker mount objects

### Kubernetes Backend

```python
class SandboxKubernetesSession(Session):
    client: kubernetes.client.CoreV1Api
    pod_name: str
    kube_namespace: str
```

**Kubernetes-Specific Parameters:**
- `kube_namespace`: Kubernetes namespace
- `pod_manifest`: Custom pod specification
- `env_vars`: Environment variables

### Podman Backend

```python
class SandboxPodmanSession(Session):
    client: podman.PodmanClient
    container: podman.domain.containers.Container
```

**Podman-Specific Parameters:**
- `stream`: Stream command output
- `commit_container`: Save container state
- `mounts`: Podman mount configurations

### Micromamba Backend

```python
class MicromambaSession(SandboxDockerSession):
    environment: str  # Conda environment name
```

**Micromamba-Specific Parameters:**
- `environment`: Conda environment to use

---

## Type Hints

### Protocol Types

```python
class ContainerProtocol(Protocol):
    """Protocol for container objects"""
    
    def execute_command(self, command: str, workdir: str | None = None) -> Any:
        ...
    
    def get_archive(self, path: str) -> tuple:
        ...
    
    def run(self, code: str, libraries: list | None = None) -> Any:
        ...
```

---

## Complete Example

```python
from llm_sandbox import (
    SandboxSession,
    SandboxBackend,
    ArtifactSandboxSession,
    get_security_policy,
    SecurityPolicy,
    SecurityPattern,
    SecurityIssueSeverity
)
import base64

# Basic usage
with SandboxSession(lang="python") as session:
    result = session.run("print('Hello, World!')")
    print(result.stdout)

# With security policy
policy = get_security_policy("production")
policy.add_pattern(SecurityPattern(
    pattern=r"requests\.get\(.*internal\.company",
    description="Internal network access",
    severity=SecurityIssueSeverity.HIGH
))

with SandboxSession(
    lang="python",
    security_policy=policy,
    runtime_configs={
        "cpu_count": 2,
        "mem_limit": "512m",
        "timeout": 30
    }
) as session:
    # Check code safety
    code = "import requests; requests.get('https://api.example.com')"
    is_safe, violations = session.is_safe(code)
    
    if is_safe:
        result = session.run(code, libraries=["requests"])
        print(result.stdout)
    else:
        print("Code failed security check")

# With artifact extraction
with ArtifactSandboxSession(
    lang="python",
    backend=SandboxBackend.DOCKER
) as session:
    result = session.run("""
import matplotlib.pyplot as plt
import numpy as np

x = np.linspace(0, 10, 100)
y = np.sin(x)
plt.plot(x, y)
plt.title('Sine Wave')
plt.show()
    """, libraries=["matplotlib", "numpy"])
    
    # Save plots
    for i, plot in enumerate(result.plots):
        with open(f"plot_{i}.{plot.format.value}", "wb") as f:
            f.write(base64.b64decode(plot.content_base64))

# Kubernetes backend
with SandboxSession(
    backend=SandboxBackend.KUBERNETES,
    lang="python",
    kube_namespace="default",
    pod_manifest={
        "spec": {
            "containers": [{
                "resources": {
                    "limits": {
                        "memory": "512Mi",
                        "cpu": "1"
                    }
                }
            }]
        }
    }
) as session:
    result = session.run("print('Running in Kubernetes!')")
    print(result.stdout)
```

---

## Version Compatibility

| LLM Sandbox Version | Python | Docker API | Kubernetes API | Podman API |
|---------------------|--------|------------|----------------|------------|
| 0.3.0 | 3.9+ | 5.0+ | 18.0+ | 4.0+ |
| 0.2.0 | 3.8+ | 4.0+ | 12.0+ | 3.0+ |
| 0.1.0 | 3.7+ | 4.0+ | 12.0+ | - |

---

## Performance Considerations

### Session Creation Overhead

| Backend | Cold Start | Warm Start | Memory Overhead |
|---------|------------|------------|------------------|
| Docker | ~1s | ~0.1s | ~10MB |
| Kubernetes | ~3-5s | ~1s | ~20MB |
| Podman | ~1s | ~0.1s | ~10MB |
| Micromamba | ~2s | ~0.5s | ~15MB |

### Best Practices

1. **Reuse Sessions**: Use `keep_template=True` for repeated executions
2. **Batch Operations**: Execute multiple code snippets in one session
3. **Resource Limits**: Always set appropriate resource limits
4. **Security Policies**: Use the strictest policy suitable for your use case
5. **Error Handling**: Always handle potential exceptions

```python
# Optimized for repeated use
with SandboxSession(
    lang="python",
    keep_template=True,
    runtime_configs={
        "cpu_count": 1,
        "mem_limit": "256m"
    }
) as session:
    # Install once
    session.install(["numpy", "pandas"])
    
    # Execute multiple times
    for code in code_snippets:
        try:
            result = session.run(code)
            process_result(result)
        except SandboxError as e:
            handle_error(e)
```