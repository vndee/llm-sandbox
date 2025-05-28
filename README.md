# LLM Sandbox

[![Release](https://img.shields.io/github/v/release/vndee/llm-sandbox)](https://img.shields.io/github/v/release/vndee/llm-sandbox)
[![Build status](https://img.shields.io/github/actions/workflow/status/vndee/llm-sandbox/main.yml?branch=main)](https://github.com/vndee/llm-sandbox/actions/workflows/main.yml?query=branch%3Amain)
[![codecov](https://codecov.io/gh/vndee/llm-sandbox/branch/main/graph/badge.svg)](https://codecov.io/gh/vndee/llm-sandbox)
[![License](https://img.shields.io/github/license/vndee/llm-sandbox)](https://img.shields.io/github/license/vndee/llm-sandbox)
[![PyPI](https://img.shields.io/pypi/v/llm-sandbox)](https://pypi.org/project/llm-sandbox/)
[![Python Version](https://img.shields.io/pypi/pyversions/llm-sandbox)](https://pypi.org/project/llm-sandbox/)
[![PyPI Downloads](https://static.pepy.tech/badge/llm-sandbox)](https://pypi.org/project/llm-sandbox/)

**LLM Sandbox** is a lightweight and portable sandbox environment designed to run Large Language Model (LLM) generated code in a safe and isolated mode. It provides a secure execution environment for AI-generated code while offering flexibility in container backends and comprehensive language support.

## 🚀 Key Features

### 🛡️ Security First
- **Isolated Execution**: Code runs in isolated containers with no access to host system
- **Security Policies**: Define custom security policies to control code execution
- **Resource Limits**: Set CPU, memory, and execution time limits
- **Network Isolation**: Control network access for sandboxed code

### 🏗️ Flexible Container Backends
- **Docker**: Most popular and widely supported option
- **Kubernetes**: Enterprise-grade orchestration for scalable deployments
- **Podman**: Rootless containers for enhanced security

### 🌐 Multi-Language Support
Execute code in multiple programming languages with automatic dependency management:
- **Python** - Full ecosystem support with pip packages
- **JavaScript/Node.js** - npm package installation
- **Java** - Maven and Gradle dependency management
- **C++** - Compilation and execution
- **Go** - Module support and compilation

### 🔌 LLM Framework Integration
Seamlessly integrate with popular LLM frameworks:
- **LangChain** - Custom tools and agents
- **LangGraph** - Graph-based workflows
- **LlamaIndex** - Data-augmented applications

### 📊 Advanced Features
- **Artifact Extraction**: Automatically capture plots and visualizations
- **Library Management**: Install dependencies on-the-fly
- **File Operations**: Copy files to/from sandbox environments
- **Custom Images**: Use your own container images

## 📦 Installation

### Basic Installation
```bash
pip install llm-sandbox
```

### With Specific Backend Support
```bash
# For Docker support (most common)
pip install 'llm-sandbox[docker]'

# For Kubernetes support
pip install 'llm-sandbox[k8s]'

# For Podman support
pip install 'llm-sandbox[podman]'

# All backends
pip install 'llm-sandbox[docker,k8s,podman]'
```

### Development Installation
```bash
git clone https://github.com/vndee/llm-sandbox.git
cd llm-sandbox
pip install -e '.[dev]'
```

## 🏃‍♂️ Quick Start

### Basic Usage

```python
from llm_sandbox import SandboxSession

# Create and use a sandbox session
with SandboxSession(lang="python") as session:
    result = session.run("""
print("Hello from LLM Sandbox!")
print("I'm running in a secure container.")
    """)
    print(result.stdout)
```

### Installing Libraries

```python
from llm_sandbox import SandboxSession

with SandboxSession(lang="python") as session:
    result = session.run("""
import numpy as np

# Create an array
arr = np.array([1, 2, 3, 4, 5])
print(f"Array: {arr}")
print(f"Mean: {np.mean(arr)}")
    """, libraries=["numpy"])

    print(result.stdout)
```

### Multi-Language Support

#### JavaScript
```python
with SandboxSession(lang="javascript") as session:
    result = session.run("""
const greeting = "Hello from Node.js!";
console.log(greeting);

const axios = require('axios');
console.log("Axios loaded successfully!");
    """, libraries=["axios"])
```

#### Java
```python
with SandboxSession(lang="java") as session:
    result = session.run("""
public class HelloWorld {
    public static void main(String[] args) {
        System.out.println("Hello from Java!");
    }
}
    """)
```

#### C++
```python
with SandboxSession(lang="cpp") as session:
    result = session.run("""
#include <iostream>

int main() {
    std::cout << "Hello from C++!" << std::endl;
    return 0;
}
    """)
```

#### Go
```python
with SandboxSession(lang="go") as session:
    result = session.run("""
package main
import "fmt"

func main() {
    fmt.Println("Hello from Go!")
}
    """)
```

### Capturing Plots and Visualizations

```python
with SandboxSession(lang="python") as session:
    result = session.run("""
import matplotlib.pyplot as plt
import numpy as np

x = np.linspace(0, 10, 100)
y = np.sin(x)

plt.figure(figsize=(10, 6))
plt.plot(x, y)
plt.title("Sine Wave")
plt.xlabel("x")
plt.ylabel("sin(x)")
plt.grid(True)
plt.savefig("sine_wave.png", dpi=150, bbox_inches="tight")
plt.show()
    """, libraries=["matplotlib", "numpy"])

    # Extract the generated plot
    artifacts = session.get_artifacts()
    print(f"Generated {len(artifacts)} artifacts")
```

## 🔧 Configuration

### Backend Selection

```python
from llm_sandbox import SandboxSession, SandboxBackend

# Use specific backend
with SandboxSession(
    backend=SandboxBackend.DOCKER,  # or KUBERNETES, PODMAN
    lang="python"
) as session:
    pass
```

### Security and Resource Limits

```python
with SandboxSession(
    lang="python",
    timeout=30,  # 30 second timeout
    memory_limit="512m",  # 512MB memory limit
    cpu_limit="1.0",  # 1 CPU core
    network_access=False  # Disable network
) as session:
    pass
```

### Custom Container Images

```python
with SandboxSession(
    lang="python",
    image="my-custom-python:latest"
) as session:
    pass
```

## 🤖 LLM Framework Integration

### LangChain Tool

```python
from langchain.tools import BaseTool
from llm_sandbox import SandboxSession

class PythonSandboxTool(BaseTool):
    name = "python_sandbox"
    description = "Execute Python code in a secure sandbox"

    def _run(self, code: str) -> str:
        with SandboxSession(lang="python") as session:
            result = session.run(code)
            return result.stdout if result.exit_code == 0 else result.stderr
```

### Use with OpenAI Functions

```python
import openai
from llm_sandbox import SandboxSession

def execute_code(code: str, language: str = "python") -> str:
    """Execute code in a secure sandbox environment."""
    with SandboxSession(lang=language) as session:
        result = session.run(code)
        return result.stdout if result.exit_code == 0 else result.stderr

# Register as OpenAI function
functions = [
    {
        "name": "execute_code",
        "description": "Execute code in a secure sandbox",
        "parameters": {
            "type": "object",
            "properties": {
                "code": {"type": "string", "description": "Code to execute"},
                "language": {"type": "string", "enum": ["python", "javascript", "java", "cpp", "go"]}
            },
            "required": ["code"]
        }
    }
]
```

## 🏗️ Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   LLM Client    │───▶│  LLM Sandbox    │───▶│ Container       │
│                 │    │                 │    │ Backend         │
│ • OpenAI        │    │ • Security      │    │                 │
│ • Anthropic     │    │ • Isolation     │    │ • Docker        │
│ • Local LLMs    │    │ • Multi-lang    │    │ • Kubernetes    │
│ • LangChain     │    │ • Artifacts     │    │ • Podman        │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

## 📚 Documentation

- **[Full Documentation](https://vndee.github.io/llm-sandbox/)** - Complete documentation
- **[Getting Started](https://vndee.github.io/llm-sandbox/getting-started/)** - Installation and basic usage
- **[Configuration](https://vndee.github.io/llm-sandbox/configuration/)** - Detailed configuration options
- **[Security](https://vndee.github.io/llm-sandbox/security/)** - Security policies and best practices
- **[Backends](https://vndee.github.io/llm-sandbox/backends/)** - Container backend details
- **[Languages](https://vndee.github.io/llm-sandbox/languages/)** - Supported programming languages
- **[Integrations](https://vndee.github.io/llm-sandbox/integrations/)** - LLM framework integrations
- **[API Reference](https://vndee.github.io/llm-sandbox/api-reference/)** - Complete API documentation
- **[Examples](https://vndee.github.io/llm-sandbox/examples/)** - Real-world usage examples

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guide](https://vndee.github.io/llm-sandbox/contributing/) for details.

### Development Setup

```bash
# Clone the repository
git clone https://github.com/vndee/llm-sandbox.git
cd llm-sandbox

# Install in development mode
make install

# Run pre-commit hooks
uv run pre-commit run -a

# Run tests
make test
```

## 🔐 Security

LLM Sandbox is designed with security as a top priority:

- **Container Isolation**: All code runs in isolated containers
- **Resource Limits**: Configurable CPU, memory, and time limits
- **Network Controls**: Disable or restrict network access
- **File System**: Read-only file systems and controlled mounts
- **Security Policies**: Custom policies for code validation

For detailed security information, see our [Security Guide](https://vndee.github.io/llm-sandbox/security/).

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🌟 Star History

If you find LLM Sandbox useful, please consider giving it a star on GitHub!

## 📞 Support & Community

- **GitHub Issues**: [Report bugs or request features](https://github.com/vndee/llm-sandbox/issues)
- **GitHub Discussions**: [Join the community](https://github.com/vndee/llm-sandbox/discussions)
- **PyPI**: [pypi.org/project/llm-sandbox](https://pypi.org/project/llm-sandbox/)
- **Documentation**: [vndee.github.io/llm-sandbox](https://vndee.github.io/llm-sandbox/)

---

Made with ❤️ for the AI developer community
