# API Reference

Comprehensive API documentation for LLM Sandbox.

## Core Classes

### SandboxSession

::: llm_sandbox.SandboxSession

### ArtifactSandboxSession

::: llm_sandbox.session.ArtifactSandboxSession

---

## Data Classes

### ConsoleOutput

::: llm_sandbox.data.ConsoleOutput

### ExecutionResult

::: llm_sandbox.data.ExecutionResult

### PlotOutput

::: llm_sandbox.data.PlotOutput

---

## Security Classes

### SecurityPolicy

::: llm_sandbox.security.SecurityPolicy

### SecurityPattern

::: llm_sandbox.security.SecurityPattern

### RestrictedModule

::: llm_sandbox.security.RestrictedModule

### SecurityIssueSeverity

::: llm_sandbox.security.SecurityIssueSeverity

---

## Enumerations

### SandboxBackend

::: llm_sandbox.const.SandboxBackend

### SupportedLanguage

::: llm_sandbox.const.SupportedLanguage

### FileType

::: llm_sandbox.data.FileType

---

## Functions

### create_session

::: llm_sandbox.create_session

---

## Exceptions

The library defines a base exception `llm_sandbox.exceptions.SandboxError` and various specific exceptions that inherit from it. Please refer to the `llm_sandbox.exceptions` module for a complete list.

### SandboxTimeoutError

::: llm_sandbox.exceptions.SandboxTimeoutError

Common exceptions include:
- `ContainerError`
- `SecurityError`
- `ResourceError`
- `ValidationError`
- `LanguageNotSupportedError`
- `ImageNotFoundError`
- `SandboxTimeoutError` - Raised when operations exceed configured timeout limits

---

## Language Handlers

### AbstractLanguageHandler

::: llm_sandbox.language_handlers.base.AbstractLanguageHandler

### LanguageConfig

::: llm_sandbox.language_handlers.LanguageConfig

---

## Backend-Specific APIs

### Docker Backend

::: llm_sandbox.docker.SandboxDockerSession

### Kubernetes Backend

::: llm_sandbox.kubernetes.SandboxKubernetesSession

### Podman Backend

::: llm_sandbox.podman.SandboxPodmanSession

### Micromamba Backend

::: llm_sandbox.micromamba.MicromambaSession

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
from llm_sandbox.exceptions import SandboxTimeoutError
import base64

# Basic usage
with SandboxSession(lang="python") as session:
    result = session.run("print('Hello, World!')")
    print(result.stdout)

# With timeout configuration
with SandboxSession(
    lang="python",
    execution_timeout=30.0,  # 30 seconds for code execution
    session_timeout=300.0,   # 5 minutes session lifetime
    default_timeout=10.0     # Default timeout for operations
) as session:
    try:
        # This will use the execution_timeout (30s)
        result = session.run("print('Normal execution')")

        # Override timeout for specific execution
        result = session.run("""
import time
time.sleep(5)
print('Long operation completed')
        """, timeout=15.0)  # Override with 15 seconds

    except SandboxTimeoutError as e:
        print(f"Operation timed out: {e}")

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
