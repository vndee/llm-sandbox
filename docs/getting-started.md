# Getting Started

## Installation

You can install `llm-sandbox` using pip:

```bash
pip install llm-sandbox
```

Or using uv (recommended):

```bash
uv pip install llm-sandbox
```

## Quick Start

Here's a simple example of how to use llm-sandbox:

```python
from llm_sandbox import Sandbox

# Create a sandbox environment
sandbox = Sandbox()

# Run some code in the sandbox
result = sandbox.run("print('Hello from sandbox!')")
print(result)
```

## Basic Concepts

The llm-sandbox package provides a secure environment for running LLM-generated code. Here are the key concepts:

1. **Sandbox**: The main container that isolates code execution
2. **Security**: Built-in security measures and limitations
3. **Resource Management**: CPU and memory limits

## Configuration

You can configure the sandbox using environment variables or a configuration file. See the [Configuration Guide](configuration.md) for more details.
