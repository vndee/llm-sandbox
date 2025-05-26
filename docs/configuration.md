# Configuration Guide

## Overview

The llm-sandbox package can be configured through various methods:

1. Environment variables
2. Configuration files
3. Programmatic configuration

## Environment Variables

The following environment variables are supported:

- `LLM_SANDBOX_MAX_MEMORY`: Maximum memory limit (in MB)
- `LLM_SANDBOX_MAX_CPU`: Maximum CPU usage (in %)
- `LLM_SANDBOX_TIMEOUT`: Maximum execution time (in seconds)
- `LLM_SANDBOX_NETWORK`: Enable/disable network access

Example:

```bash
export LLM_SANDBOX_MAX_MEMORY=512
export LLM_SANDBOX_TIMEOUT=30
```

## Configuration File

You can create a `sandbox_config.yaml` file in your project root:

```yaml
memory_limit: 512  # MB
cpu_limit: 50      # %
timeout: 30        # seconds
network: false     # disable network access
```

## Programmatic Configuration

You can also configure the sandbox programmatically:

```python
from llm_sandbox import Sandbox, SandboxConfig

config = SandboxConfig(
    memory_limit=512,
    cpu_limit=50,
    timeout=30,
    network=False
)

sandbox = Sandbox(config=config)
```

## Security Settings

### Default Security Policies

The sandbox implements several security measures by default:

1. File system isolation
2. Network access restrictions
3. System call filtering
4. Resource limitations

### Customizing Security Policies

You can customize security policies through configuration:

```python
from llm_sandbox import Sandbox, SecurityPolicy

policy = SecurityPolicy(
    allowed_modules=['math', 'json'],
    blocked_syscalls=['socket', 'fork'],
    read_only=True
)

sandbox = Sandbox(security_policy=policy)
```
