# Security Guide

LLM Sandbox provides comprehensive security features to safely execute untrusted code. This guide covers security policies, best practices, and implementation details.

## Overview

Security in LLM Sandbox is implemented at multiple levels:

1. **Container Isolation** - Code runs in isolated containers
2. **Security Policies** - Pre-execution code analysis
3. **Resource Limits** - Prevent resource exhaustion
4. **Network Controls** - Limit network access
5. **File System Restrictions** - Control file access

## Security Policies

### Understanding Security Policies

Security policies analyze code before execution to detect potentially dangerous patterns:

```python
from llm_sandbox.security import (
    SecurityPolicy,
    SecurityPattern,
    DangerousModule,
    SecurityIssueSeverity
)

# Create a security policy
policy = SecurityPolicy(
    severity_threshold=SecurityIssueSeverity.MEDIUM,
    patterns=[...],         # Regex patterns to detect
    restricted_modules=[...]  # Modules to block
)
```

### Severity Levels

Security issues are classified by severity:

| Level | Value | Description |
|-------|-------|-------------|
| `SAFE` | 0 | No security concerns |
| `LOW` | 1 | Minor concerns, usually allowed |
| `MEDIUM` | 2 | Moderate risk, context-dependent |
| `HIGH` | 3 | High risk, usually blocked |

### Security Patterns

Define patterns to detect dangerous code:

```python
# Detect system command execution
SecurityPattern(
    pattern=r"os\.system\s*\(",
    description="System command execution",
    severity=SecurityIssueSeverity.HIGH
)

# Detect file write operations
SecurityPattern(
    pattern=r"open\s*\([^)]*['\"][wa]['\"][^)]*\)",
    description="File write operation",
    severity=SecurityIssueSeverity.MEDIUM
)

# Detect network operations
SecurityPattern(
    pattern=r"socket\.socket\s*\(",
    description="Raw socket creation",
    severity=SecurityIssueSeverity.MEDIUM
)
```

### Restricted Modules

Block dangerous modules:

```python
# High-risk modules
DangerousModule(
    name="os",
    description="Operating system interface",
    severity=SecurityIssueSeverity.HIGH
)

DangerousModule(
    name="subprocess",
    description="Process execution",
    severity=SecurityIssueSeverity.HIGH
)

# Medium-risk modules
DangerousModule(
    name="socket",
    description="Network operations",
    severity=SecurityIssueSeverity.MEDIUM
)
```

## Preset Security Policies

LLM Sandbox provides preset policies for common use cases:

### Minimal Policy

Very permissive, only blocks the most dangerous operations:

```python
from llm_sandbox.security import get_security_policy

policy = get_security_policy("minimal")
# Blocks: os.system with rm -rf /, format commands
# Allows: Most operations
```

### Development Policy

Balanced for development environments:

```python
policy = get_security_policy("development")
# Blocks: Direct system commands, dangerous deletions
# Allows: File operations, network requests
```

### Production Policy

Strict policy for production use:

```python
policy = get_security_policy("production")
# Blocks: System commands, subprocess, file writes, environment access
# Allows: Computation, read operations
```

### Educational Policy

Designed for teaching environments:

```python
policy = get_security_policy("educational")
# Blocks: System damage, file deletion
# Allows: Learning operations, warns about eval/exec
```

### Data Science Policy

Optimized for data analysis:

```python
policy = get_security_policy("data_science")
# Blocks: System commands, dangerous operations
# Allows: Data processing, visualization, HTTP requests
```

### Strict Policy

Most restrictive policy:

```python
policy = get_security_policy("strict")
# Blocks: Almost all system interaction
# Allows: Pure computation only
```

## Using Security Policies

### Basic Usage

```python
from llm_sandbox import SandboxSession
from llm_sandbox.security import get_security_policy

# Use a preset policy
policy = get_security_policy("production")

with SandboxSession(lang="python", security_policy=policy) as session:
    # Check if code is safe
    code = "print('Hello, World!')"
    is_safe, violations = session.is_safe(code)
    
    if is_safe:
        result = session.run(code)
        print(result.stdout)
    else:
        print("Code failed security check:")
        for violation in violations:
            print(f"  - {violation.description} (Severity: {violation.severity.name})")
```

### Custom Security Policies

```python
# Create custom policy
custom_policy = SecurityPolicy(
    severity_threshold=SecurityIssueSeverity.LOW,
    patterns=[
        # Block cloud SDK usage
        SecurityPattern(
            pattern=r"boto3|google\.cloud|azure",
            description="Cloud SDK usage",
            severity=SecurityIssueSeverity.HIGH
        ),
        # Block specific domains
        SecurityPattern(
            pattern=r"requests\.get\s*\(['\"].*internal\.company\.com",
            description="Internal network access",
            severity=SecurityIssueSeverity.HIGH
        ),
        # Warn about external APIs
        SecurityPattern(
            pattern=r"requests\.(get|post)\s*\(",
            description="External API call",
            severity=SecurityIssueSeverity.LOW
        )
    ],
    restricted_modules=[
        DangerousModule(
            name="psutil",
            description="System monitoring",
            severity=SecurityIssueSeverity.MEDIUM
        )
    ]
)
```

### Dynamic Policy Modification

```python
# Start with base policy
policy = get_security_policy("production")

# Add custom patterns
policy.add_pattern(SecurityPattern(
    pattern=r"tensorflow|torch|keras",
    description="ML framework usage",
    severity=SecurityIssueSeverity.LOW
))

# Add restricted modules
policy.add_restricted_module(DangerousModule(
    name="cryptography",
    description="Cryptographic operations",
    severity=SecurityIssueSeverity.MEDIUM
))
```

## Security Best Practices

### 1. Defense in Depth

Combine multiple security layers:

```python
with SandboxSession(
    lang="python",
    # Layer 1: Security policy
    security_policy=get_security_policy("production"),
    # Layer 2: Resource limits
    runtime_configs={
        "cpu_count": 1,
        "mem_limit": "256m",
        "timeout": 10,
        "pids_limit": 50
    },
    # Layer 3: User isolation
    runtime_configs={"user": "nobody:nogroup"},
    # Layer 4: Read-only file system
    mounts=[
        Mount(type="bind", source="/data", target="/data", read_only=True)
    ]
) as session:
    pass
```

### 2. Validate All Inputs

```python
def safe_execute(code: str, max_length: int = 10000):
    """Execute code with input validation"""
    
    # Check code length
    if len(code) > max_length:
        raise ValueError("Code too long")
    
    # Check for null bytes
    if '\x00' in code:
        raise ValueError("Invalid characters in code")
    
    # Use strict policy
    policy = get_security_policy("strict")
    
    with SandboxSession(lang="python", security_policy=policy) as session:
        is_safe, violations = session.is_safe(code)
        if not is_safe:
            raise SecurityError(f"Code failed security check: {violations}")
        
        return session.run(code)
```

### 3. Monitor and Log

```python
import logging
from datetime import datetime

class SecurityLogger:
    def __init__(self):
        self.logger = logging.getLogger('security')
    
    def log_execution(self, code: str, user: str, result: Any):
        """Log code execution for audit"""
        self.logger.info({
            'timestamp': datetime.utcnow().isoformat(),
            'user': user,
            'code_hash': hashlib.sha256(code.encode()).hexdigest(),
            'code_length': len(code),
            'exit_code': result.exit_code,
            'execution_time': result.execution_time
        })

# Use with monitoring
logger = SecurityLogger()

with SandboxSession(lang="python") as session:
    result = session.run(code)
    logger.log_execution(code, user_id, result)
```

### 4. Escape Output

```python
import html
import json

def safe_display_output(result):
    """Safely display execution results"""
    return {
        'stdout': html.escape(result.stdout),
        'stderr': html.escape(result.stderr),
        'exit_code': result.exit_code,
        'safe_json': json.dumps(result.stdout, ensure_ascii=True)
    }
```

### 5. Rate Limiting

```python
from collections import defaultdict
from time import time

class RateLimiter:
    def __init__(self, max_requests: int = 10, window: int = 60):
        self.max_requests = max_requests
        self.window = window
        self.requests = defaultdict(list)
    
    def allow_request(self, user_id: str) -> bool:
        now = time()
        # Clean old requests
        self.requests[user_id] = [
            t for t in self.requests[user_id] 
            if now - t < self.window
        ]
        
        # Check limit
        if len(self.requests[user_id]) >= self.max_requests:
            return False
        
        self.requests[user_id].append(now)
        return True

# Use rate limiting
rate_limiter = RateLimiter(max_requests=5, window=60)

if rate_limiter.allow_request(user_id):
    with SandboxSession(lang="python") as session:
        result = session.run(code)
else:
    raise Exception("Rate limit exceeded")
```

## Container Security

### Network Isolation

```python
# Disable network access (Docker/Podman)
with SandboxSession(
    lang="python",
    runtime_configs={
        "network_mode": "none"
    }
) as session:
    # Code cannot access network
    pass

# Custom network (Docker)
with SandboxSession(
    lang="python",
    runtime_configs={
        "network_mode": "custom_isolated_network"
    }
) as session:
    pass
```

### File System Security

```python
# Read-only root filesystem
with SandboxSession(
    lang="python",
    runtime_configs={
        "read_only": True,
        "tmpfs": {
            "/tmp": "size=50m,mode=1777",
            "/run": "size=10m,mode=0755"
        }
    }
) as session:
    pass
```

### Capability Management

```python
# Drop all capabilities except specific ones
with SandboxSession(
    lang="python",
    runtime_configs={
        "cap_drop": ["ALL"],
        "cap_add": ["DAC_OVERRIDE"]  # Only if needed
    }
) as session:
    pass
```

## Security Patterns Reference

### Common Dangerous Patterns

```python
DANGEROUS_PATTERNS = [
    # Command execution
    (r"os\.system\s*\(", "System command execution"),
    (r"subprocess\.(run|call|Popen|check_output)\s*\(", "Subprocess execution"),
    (r"eval\s*\(", "Dynamic code evaluation"),
    (r"exec\s*\(", "Dynamic code execution"),
    (r"__import__\s*\(", "Dynamic module import"),
    
    # File operations
    (r"open\s*\([^)]*['\"][wa]['\"][^)]*\)", "File write"),
    (r"os\.(remove|unlink|rmdir)\s*\(", "File deletion"),
    (r"shutil\.rmtree\s*\(", "Directory deletion"),
    
    # Network operations
    (r"socket\.socket\s*\(", "Socket creation"),
    (r"urllib\.request\.urlopen\s*\(", "URL access"),
    
    # System information
    (r"os\.environ", "Environment access"),
    (r"platform\.(system|release|version)\s*\(", "System info"),
    
    # Dangerous builtins
    (r"compile\s*\(", "Code compilation"),
    (r"globals\s*\(\)", "Global scope access"),
    (r"locals\s*\(\)", "Local scope access"),
]
```

### Language-Specific Patterns

#### Python
```python
# Pickle deserialization (arbitrary code execution)
(r"pickle\.loads?\s*\(", "Unsafe deserialization")

# Code introspection
(r"inspect\.(getfile|getsource|getmodule)\s*\(", "Code introspection")

# AST manipulation
(r"ast\.(parse|compile)\s*\(", "AST manipulation")
```

#### JavaScript
```python
# Eval and Function constructor
(r"eval\s*\(", "JavaScript eval")
(r"new\s+Function\s*\(", "Function constructor")

# Process access (Node.js)
(r"process\.exit\s*\(", "Process termination")
(r"child_process", "Child process access")
```

#### Java
```python
# Reflection
(r"Class\.forName\s*\(", "Java reflection")
(r"Runtime\.getRuntime\(\)\.exec\s*\(", "Runtime execution")

# System access
(r"System\.exit\s*\(", "System exit")
```

## Troubleshooting Security Issues

### Common Issues

1. **False Positives**
   ```python
   # Code contains pattern in string/comment
   code = 'print("Use os.system to run commands")'  # Blocked!
   
   # Solution: Adjust pattern or use exceptions
   ```

2. **Import Detection**
   ```python
   # Various import styles
   import os                    # Detected
   from os import system        # Detected
   __import__('os')            # Detected
   importlib.import_module('os')  # May not be detected
   ```

3. **Obfuscation Attempts**
   ```python
   # Beware of obfuscation
   getattr(__builtins__, 'eval')  # May bypass simple patterns
   ```

### Security Testing

```python
def test_security_policy(policy: SecurityPolicy):
    """Test security policy effectiveness"""
    
    test_cases = [
        # Should pass
        ("print('Hello')", True),
        ("x = 1 + 2", True),
        
        # Should fail
        ("import os; os.system('ls')", False),
        ("eval('1+1')", False),
        
        # Edge cases
        ("# import os", True),  # Comment
        ("'import os'", True),  # String
    ]
    
    for code, expected_safe in test_cases:
        with SandboxSession(lang="python", security_policy=policy) as session:
            is_safe, _ = session.is_safe(code)
            assert is_safe == expected_safe, f"Failed for: {code}"
```

## Security Checklist

- [ ] Use appropriate security policy for your use case
- [ ] Set resource limits (CPU, memory, timeout)
- [ ] Run containers as non-root user
- [ ] Disable network access if not needed
- [ ] Use read-only file systems where possible
- [ ] Validate and sanitize all inputs
- [ ] Log all code executions for audit
- [ ] Implement rate limiting
- [ ] Escape output before display
- [ ] Regularly update container images
- [ ] Monitor for security advisories

## Next Steps

- Explore [Backend Options](backends.md) for platform-specific security
- Learn about [Configuration](configuration.md) options
- See [Examples](examples.md) of secure implementations
- Check the [API Reference](api-reference.md) for detailed documentation