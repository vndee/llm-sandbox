# Interactive Sessions

Interactive sessions provide a persistent Python execution environment that maintains state across multiple code executions, similar to Jupyter notebooks or Python REPL environments.

## Overview

Unlike standard `SandboxSession` which creates a fresh execution context for each `run()` call, `InteractiveSandboxSession` maintains a persistent IPython interpreter inside the sandbox. This allows you to:

- **Preserve state** across multiple code executions
- **Define functions and classes** that remain available
- **Import modules once** and reuse them
- **Use IPython magic commands** like `%who`, `%pwd`, `!shell commands`
- **Build complex workflows** step-by-step

This is ideal for notebook-style workflows, AI agent interactions, exploratory programming, and multi-step data analysis.

## Quick Start

Here's a simple example showing state persistence:

```python
from llm_sandbox import InteractiveSandboxSession

with InteractiveSandboxSession(lang="python") as session:
    # First execution: define a variable
    session.run("value = 21 * 2")

    # Second execution: use the variable
    result = session.run("print(f'Result: {value}')")
    print(result.stdout)  # Output: Result: 42
```

The variable `value` persists between the two `run()` calls, demonstrating the stateful nature of interactive sessions.

## Basic Usage

### Creating an Interactive Session

```python
from llm_sandbox import InteractiveSandboxSession

with InteractiveSandboxSession(
    lang="python",
    kernel_type="ipython",
    history_size=200,
    timeout=300,
) as session:
    # Your code here
    pass
```

### Running Code Cells

Each call to `run()` behaves like a notebook cell:

```python
from llm_sandbox import InteractiveSandboxSession

with InteractiveSandboxSession(lang="python") as session:
    # Cell 1: Import libraries
    session.run("import pandas as pd")
    session.run("import numpy as np")

    # Cell 2: Create data
    session.run("""
data = pd.DataFrame({
    'x': np.random.randn(100),
    'y': np.random.randn(100)
})
""")

    # Cell 3: Analyze data
    result = session.run("""
print(f"Shape: {data.shape}")
print(f"Mean of x: {data['x'].mean():.2f}")
print(f"Mean of y: {data['y'].mean():.2f}")
""")

    print(result.stdout)
```

### Installing Libraries

You can install libraries dynamically within the session:

```python
from llm_sandbox import InteractiveSandboxSession

with InteractiveSandboxSession(lang="python") as session:
    # Install a library in the first run
    result = session.run("""
import matplotlib.pyplot as plt
import numpy as np

x = np.linspace(0, 10, 100)
y = np.sin(x)

plt.plot(x, y)
plt.title("Sine Wave")
print("Plot created")
""", libraries=["matplotlib", "numpy"])

    print(result.stdout)
```

## Configuration

### InteractiveSettings

Configure the interactive session behavior using `InteractiveSettings`:

```python
from llm_sandbox import InteractiveSandboxSession, InteractiveSettings, KernelType

settings = InteractiveSettings(
    kernel_type=KernelType.IPYTHON,  # Only IPYTHON supported currently
    max_memory="2GB",                # Memory limit for the session
    history_size=500,                # Number of cached execution entries
    timeout=600,                     # Per-cell timeout in seconds
    poll_interval=0.1                # Polling interval for results (seconds)
)

with InteractiveSandboxSession(
    lang="python",
    interactive_settings=settings
) as session:
    # Your code here
    pass
```

### Configuration Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `kernel_type` | `KernelType` | `IPYTHON` | Type of kernel (currently only IPython) |
| `max_memory` | `str` | `"1GB"` | Memory limit for the session |
| `history_size` | `int` | `1000` | Number of execution results to cache |
| `timeout` | `int` | `300` | Per-cell timeout in seconds |
| `poll_interval` | `float` | `0.1` | Polling interval for results (seconds) |

## IPython Magic Commands

Interactive sessions support IPython magic commands:

```python
from llm_sandbox import InteractiveSandboxSession

with InteractiveSandboxSession(lang="python") as session:
    # List variables
    result = session.run("%who")
    print(result.stdout)

    # Check current directory
    result = session.run("%pwd")
    print(result.stdout)

    # Execute shell commands
    result = session.run("!ls -la /sandbox")
    print(result.stdout)

    # Time code execution
    result = session.run("""
%%timeit
sum(range(1000))
""")
    print(result.stdout)
```

## Advanced Use Cases

### Multi-Step Data Analysis

Build complex analysis pipelines step-by-step:

```python
from llm_sandbox import InteractiveSandboxSession

with InteractiveSandboxSession(lang="python") as session:
    # Step 1: Load and prepare data
    session.run("""
import pandas as pd
import numpy as np

# Create sample dataset
data = pd.DataFrame({
    'date': pd.date_range('2024-01-01', periods=100),
    'value': np.random.randn(100).cumsum() + 100,
    'category': np.random.choice(['A', 'B', 'C'], 100)
})
""", libraries=["pandas", "numpy"])

    # Step 2: Clean data
    session.run("""
# Remove outliers
q1 = data['value'].quantile(0.25)
q3 = data['value'].quantile(0.75)
iqr = q3 - q1
data_clean = data[
    (data['value'] >= q1 - 1.5*iqr) &
    (data['value'] <= q3 + 1.5*iqr)
]
""")

    # Step 3: Aggregate results
    result = session.run("""
summary = data_clean.groupby('category')['value'].agg(['mean', 'std', 'count'])
print("Summary by Category:")
print(summary)
""")

    print(result.stdout)
```

### Function Definitions

Define reusable functions within the session:

```python
from llm_sandbox import InteractiveSandboxSession

with InteractiveSandboxSession(lang="python") as session:
    # Define utility functions
    session.run("""
def fibonacci(n):
    if n <= 1:
        return n
    return fibonacci(n-1) + fibonacci(n-2)

def factorial(n):
    if n <= 1:
        return 1
    return n * factorial(n-1)
""")

    # Use the functions
    result = session.run("""
print(f"Fibonacci(10): {fibonacci(10)}")
print(f"Factorial(5): {factorial(5)}")
""")

    print(result.stdout)
```

### Class Definitions

Create and use classes across multiple executions:

```python
from llm_sandbox import InteractiveSandboxSession

with InteractiveSandboxSession(lang="python") as session:
    # Define a class
    session.run("""
class DataProcessor:
    def __init__(self):
        self.data = []

    def add(self, value):
        self.data.append(value)

    def get_stats(self):
        return {
            'count': len(self.data),
            'sum': sum(self.data),
            'mean': sum(self.data) / len(self.data) if self.data else 0
        }

processor = DataProcessor()
""")

    # Use the class instance
    session.run("""
for i in range(1, 11):
    processor.add(i * 2)
""")

    result = session.run("""
stats = processor.get_stats()
print(f"Count: {stats['count']}")
print(f"Sum: {stats['sum']}")
print(f"Mean: {stats['mean']}")
""")

    print(result.stdout)
```

### AI Agent REPL Interaction

Interactive sessions are perfect for LLM agent interactions:

```python
from llm_sandbox import InteractiveSandboxSession

def ai_agent_workflow():
    with InteractiveSandboxSession(lang="python") as session:
        # Agent explores the environment
        result = session.run("%pwd")
        current_dir = result.stdout.strip()

        # Agent creates data based on exploration
        session.run(f"""
import os
print(f"Working in: {current_dir}")

# Create some data
data = [1, 2, 3, 4, 5]
print(f"Created data: {{data}}")
""")

        # Agent performs analysis
        result = session.run("""
# Analyze the data
total = sum(data)
avg = total / len(data)
print(f"Total: {total}")
print(f"Average: {avg}")
""")

        print(result.stdout)

ai_agent_workflow()
```

## Error Handling

Handle errors gracefully in interactive sessions:

```python
from llm_sandbox import InteractiveSandboxSession

with InteractiveSandboxSession(lang="python") as session:
    # Set up some state
    session.run("x = 10")

    # Try to run code that might fail
    result = session.run("y = x / 0")  # Division by zero

    if result.exit_code != 0:
        print(f"Error occurred: {result.stderr}")
        # State is preserved even after errors
        result = session.run("print(f'x is still: {x}')")
        print(result.stdout)
```

## Best Practices

### When to Use Interactive Sessions

**Use Interactive Sessions when:**

- Building multi-step workflows where each step depends on previous results
- Creating AI agents that need to explore and learn iteratively
- Implementing notebook-style interactions with LLMs
- Prototyping and exploratory data analysis
- Defining reusable functions and classes

**Use Standard SandboxSession when:**

- Each code execution is independent
- You need Kubernetes or Podman backend support
- You want to execute non-Python code
- You need a fresh, clean execution environment each time

### Resource Management

```python
from llm_sandbox import InteractiveSandboxSession, InteractiveSettings

# Set appropriate limits for your use case
settings = InteractiveSettings(
    max_memory="4GB",      # Adjust based on your data size
    timeout=600,           # Longer timeout for complex operations
    history_size=100       # Keep history manageable
)

with InteractiveSandboxSession(
    lang="python",
    interactive_settings=settings
) as session:
    # Your code here
    pass
```

### Managing Session State

```python
from llm_sandbox import InteractiveSandboxSession

with InteractiveSandboxSession(lang="python") as session:
    # Check what's in memory
    result = session.run("%who")
    print("Current variables:", result.stdout)

    # Clear specific variables if needed
    session.run("del some_large_variable")

    # Reset namespace if needed (use sparingly)
    session.run("%reset -f")
```

## Current Limitations

### Backend Support

- **Docker only**: Currently, interactive sessions only support the Docker backend
- **Kubernetes**: Not yet supported
- **Podman**: Not yet supported

### Language Support

- **Python only**: Currently, only Python with IPython kernel is supported
- Other languages may be added in future releases

### Kernel Types

- **IPython only**: Currently, only `KernelType.IPYTHON` is available
- Additional kernel types may be added in future releases

## Architecture

Interactive sessions use a file-based communication protocol:

1. **Runner Script**: A Python script is uploaded to the container that starts an IPython interpreter
2. **Command Queue**: Commands are written as JSON files to `/sandbox/.interactive/commands/`
3. **IPython Execution**: The runner reads commands and executes them in the persistent IPython context
4. **Result Queue**: Results are written as JSON to `/sandbox/.interactive/results/`
5. **Host Polling**: The host polls for results with configurable timeout and interval

This architecture ensures:
- **Isolation**: Code runs in a sandboxed container
- **Persistence**: The IPython kernel maintains state between executions
- **Reliability**: Timeout handling prevents hanging executions
- **Simplicity**: No complex networking or IPC required

## Examples

For complete working examples, see:

- [Interactive Session Demo](https://github.com/vndee/llm-sandbox/blob/main/examples/interactive_session_demo.py)
- [Examples page](examples.md)

## Troubleshooting

### Session Timeout

If executions are timing out:

```python
from llm_sandbox import InteractiveSandboxSession, InteractiveSettings

settings = InteractiveSettings(
    timeout=900,  # Increase timeout to 15 minutes
)

with InteractiveSandboxSession(
    lang="python",
    interactive_settings=settings
) as session:
    # Your long-running code here
    pass
```

### Memory Issues

If running out of memory:

```python
settings = InteractiveSettings(
    max_memory="8GB",  # Increase memory limit
)

with InteractiveSandboxSession(
    lang="python",
    interactive_settings=settings
) as session:
    # Your memory-intensive code here
    pass
```

### State Contamination

If you need to reset the session state:

```python
with InteractiveSandboxSession(lang="python") as session:
    # Do some work
    session.run("x = 100")

    # Reset if needed
    session.run("%reset -f")

    # Now x is no longer defined
    result = session.run("print(x)")
    # Will show NameError
```

## Next Steps

- Explore [Examples](examples.md) for more use cases
- Check the [API Reference](api-reference.md) for detailed documentation
- Learn about [Security](security.md) best practices
- Integrate with [LLM Frameworks](integrations.md)
