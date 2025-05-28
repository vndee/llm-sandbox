# Security Guide

LLM Sandbox provides comprehensive security features to safely execute untrusted code. This guide covers the security policy system for pre-execution code analysis and best practices.

## Overview

Security in LLM Sandbox is implemented through multiple layers:

1. **Container Isolation** - Code runs in isolated containers (configured via `runtime_config` or pod manifests)
2. **Security Policies** - Pre-execution regex-based code analysis (this module's focus)
3. **Resource Limits** - Prevent resource exhaustion (configured via container runtime)
4. **Network Controls** - Limit network access (configured via container runtime)
5. **File System Restrictions** - Control file access (configured via container runtime)

> **Note**: Container-level security (resource limits, network controls, file system restrictions) is configured through `runtime_config` for Docker/Podman or pod manifests for Kubernetes as described in the [Configuration Guide](configuration.md). This module focuses on the **Security Policy** system for code analysis.

## Security Policy System

### Overview

The security policy system analyzes code **before execution** using regex pattern matching to detect potentially dangerous operations. It provides:

- **Pattern-based detection** of dangerous code constructs
- **Language-specific module restriction** based on import statements
- **Severity-based filtering** with configurable thresholds
- **Comment filtering** to avoid false positives from documentation

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

## Creating Security Policies

### Basic Usage

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
    # Check if code is safe before execution
    code = "import os\nos.system('ls')"
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
    runtime_config={
        "mem_limit": "256m",
        "cpu_count": 1,
        "timeout": 30,
        "user": "nobody:nogroup"
    }
) as session:
    # Double protection: policy blocks + container limits
    pass
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
