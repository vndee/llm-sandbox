# Security Policy Guide for LLM Sandbox

This guide provides comprehensive documentation on how to use the security features in LLM Sandbox to protect against malicious or dangerous code execution.

## Overview

The LLM Sandbox security system provides multi-layered protection through:

- **Pattern-based Detection**: Regex patterns to identify dangerous code constructs
- **Module Blacklisting**: Blocking dangerous Python modules and imports
- **Severity Levels**: Graduated response based on risk assessment
- **Dynamic Policies**: Runtime modification of security rules

## Core Components

### SecurityIssueSeverity

Defines the severity levels for security issues:

```python
from llm_sandbox.security import SecurityIssueSeverity

# Available severity levels
SecurityIssueSeverity.SAFE    # 0 - No restrictions
SecurityIssueSeverity.LOW     # 1 - Minor security concerns
SecurityIssueSeverity.MEDIUM  # 2 - Moderate security risks
SecurityIssueSeverity.HIGH    # 3 - High security risks
```

### SecurityPattern

Defines regex patterns to detect dangerous code:

```python
from llm_sandbox.security import SecurityPattern, SecurityIssueSeverity

# Create a pattern to detect os.system calls
pattern = SecurityPattern(
    pattern=r"\bos\.system\s*\(",
    description="Dangerous system command execution",
    severity=SecurityIssueSeverity.HIGH,
)
```

### DangerousModule

Defines modules that should be blocked:

```python
from llm_sandbox.security import DangerousModule, SecurityIssueSeverity

# Block the os module
module = DangerousModule(
    name="os",
    description="Operating system interface - can execute system commands",
    severity=SecurityIssueSeverity.HIGH,
)
```

### SecurityPolicy

Combines patterns and modules into a comprehensive security policy:

```python
from llm_sandbox.security import SecurityPolicy, SecurityIssueSeverity

policy = SecurityPolicy(
    severity_threshold=SecurityIssueSeverity.MEDIUM,  # Block MEDIUM and HIGH severity issues
    patterns=[pattern],  # List of SecurityPattern objects
    restricted_modules=[module],  # List of DangerousModule objects
)
```

## Usage Examples

### Basic Security Policy

```python
from llm_sandbox import SandboxSession
from llm_sandbox.security import (
    SecurityIssueSeverity,
    SecurityPattern,
    DangerousModule,
    SecurityPolicy,
)

# Create security patterns
patterns = [
    SecurityPattern(
        pattern=r"\bos\.system\s*\(",
        description="System command execution",
        severity=SecurityIssueSeverity.HIGH,
    ),
    SecurityPattern(
        pattern=r"\beval\s*\(",
        description="Dynamic code evaluation",
        severity=SecurityIssueSeverity.MEDIUM,
    ),
]

# Define dangerous modules
restricted_modules = [
    DangerousModule(
        name="subprocess",
        description="Subprocess execution",
        severity=SecurityIssueSeverity.HIGH,
    ),
    DangerousModule(
        name="socket",
        description="Network operations",
        severity=SecurityIssueSeverity.MEDIUM,
    ),
]

# Create security policy
policy = SecurityPolicy(
    severity_threshold=SecurityIssueSeverity.MEDIUM,
    patterns=patterns,
    restricted_modules=restricted_modules,
)

# Use with sandbox session
with SandboxSession(lang="python", security_policy=policy) as session:
    # Check if code is safe before execution
    code = "import os\nos.system('ls')"
    is_safe, violations = session.is_safe(code)

    if is_safe:
        result = session.run(code)
        print(result)
    else:
        print(f"Code blocked due to {len(violations)} security violations:")
        for violation in violations:
            print(f"  - {violation.description}")
```

### Dynamic Policy Modification

```python
# Start with a basic policy
policy = SecurityPolicy(
    severity_threshold=SecurityIssueSeverity.MEDIUM,
    patterns=[],
    restricted_modules=[],
)

# Add patterns dynamically
policy.add_pattern(SecurityPattern(
    pattern=r"\bopen\s*\([^)]*['\"]w['\"][^)]*\)",
    description="File writing operations",
    severity=SecurityIssueSeverity.MEDIUM,
))

# Add dangerous modules dynamically
policy.add_restricted_module(DangerousModule(
    name="pickle",
    description="Potentially unsafe serialization",
    severity=SecurityIssueSeverity.MEDIUM,
))
```

### Severity Level Configuration

```python
# Strict policy - blocks even low-severity issues
strict_policy = SecurityPolicy(
    severity_threshold=SecurityIssueSeverity.LOW,
    patterns=patterns,
    restricted_modules=restricted_modules,
)

# Permissive policy - only blocks high-severity issues
permissive_policy = SecurityPolicy(
    severity_threshold=SecurityIssueSeverity.HIGH,
    patterns=patterns,
    restricted_modules=restricted_modules,
)

# No restrictions - allows everything
open_policy = SecurityPolicy(
    severity_threshold=SecurityIssueSeverity.SAFE,
    patterns=patterns,
    restricted_modules=restricted_modules,
)
```

## Common Security Patterns

### System Command Execution

```python
patterns = [
    # Direct system calls
    SecurityPattern(
        pattern=r"\bos\.system\s*\(",
        description="os.system() calls",
        severity=SecurityIssueSeverity.HIGH,
    ),

    # Subprocess execution
    SecurityPattern(
        pattern=r"\bsubprocess\.(run|call|Popen|check_output)\s*\(",
        description="Subprocess execution",
        severity=SecurityIssueSeverity.HIGH,
    ),

    # Shell command execution
    SecurityPattern(
        pattern=r"\bos\.(popen|spawn)\w*\s*\(",
        description="Shell command execution",
        severity=SecurityIssueSeverity.HIGH,
    ),
]
```

### Dynamic Code Execution

```python
patterns = [
    # eval() function
    SecurityPattern(
        pattern=r"\beval\s*\(",
        description="Dynamic code evaluation",
        severity=SecurityIssueSeverity.MEDIUM,
    ),

    # exec() function
    SecurityPattern(
        pattern=r"\bexec\s*\(",
        description="Dynamic code execution",
        severity=SecurityIssueSeverity.MEDIUM,
    ),

    # compile() function
    SecurityPattern(
        pattern=r"\bcompile\s*\(",
        description="Code compilation",
        severity=SecurityIssueSeverity.LOW,
    ),
]
```

### File System Operations

```python
patterns = [
    # File writing
    SecurityPattern(
        pattern=r"\bopen\s*\([^)]*['\"][wa]['\"][^)]*\)",
        description="File write operations",
        severity=SecurityIssueSeverity.MEDIUM,
    ),

    # File deletion
    SecurityPattern(
        pattern=r"\bos\.(remove|unlink|rmdir)\s*\(",
        description="File deletion operations",
        severity=SecurityIssueSeverity.HIGH,
    ),

    # Directory operations
    SecurityPattern(
        pattern=r"\bshutil\.(rmtree|move|copytree)\s*\(",
        description="Directory operations",
        severity=SecurityIssueSeverity.MEDIUM,
    ),
]
```

### Network Operations

```python
patterns = [
    # Raw socket creation
    SecurityPattern(
        pattern=r"\bsocket\.socket\s*\(",
        description="Raw socket creation",
        severity=SecurityIssueSeverity.MEDIUM,
    ),

    # Server socket binding
    SecurityPattern(
        pattern=r"\b\w+\.bind\s*\(",
        description="Socket binding (potential server creation)",
        severity=SecurityIssueSeverity.MEDIUM,
    ),
]
```

## Dangerous Modules

### High-Risk Modules

```python
high_risk_modules = [
    DangerousModule(
        name="os",
        description="Operating system interface - can execute commands",
        severity=SecurityIssueSeverity.HIGH,
    ),
    DangerousModule(
        name="subprocess",
        description="Subprocess management - can execute programs",
        severity=SecurityIssueSeverity.HIGH,
    ),
    DangerousModule(
        name="ctypes",
        description="Foreign function library - can access system calls",
        severity=SecurityIssueSeverity.HIGH,
    ),
]
```

### Medium-Risk Modules

```python
medium_risk_modules = [
    DangerousModule(
        name="socket",
        description="Network socket operations",
        severity=SecurityIssueSeverity.MEDIUM,
    ),
    DangerousModule(
        name="multiprocessing",
        description="Process-based parallelism",
        severity=SecurityIssueSeverity.MEDIUM,
    ),
    DangerousModule(
        name="threading",
        description="Thread-based parallelism",
        severity=SecurityIssueSeverity.MEDIUM,
    ),
]
```

### Low-Risk Modules

```python
low_risk_modules = [
    DangerousModule(
        name="urllib",
        description="URL handling library",
        severity=SecurityIssueSeverity.LOW,
    ),
    DangerousModule(
        name="requests",
        description="HTTP requests library",
        severity=SecurityIssueSeverity.LOW,
    ),
    DangerousModule(
        name="ftplib",
        description="FTP protocol client",
        severity=SecurityIssueSeverity.LOW,
    ),
]
```

## Best Practices

### 1. Layered Security

Combine multiple security mechanisms:

```python
# Combine patterns and module blocking
policy = SecurityPolicy(
    severity_threshold=SecurityIssueSeverity.MEDIUM,
    patterns=system_command_patterns + file_operation_patterns,
    restricted_modules=high_risk_modules + medium_risk_modules,
)
```

### 2. Environment-Specific Policies

```python
# Development environment - more permissive
dev_policy = SecurityPolicy(
    severity_threshold=SecurityIssueSeverity.HIGH,
    patterns=critical_patterns_only,
    restricted_modules=high_risk_modules,
)

# Production environment - strict
prod_policy = SecurityPolicy(
    severity_threshold=SecurityIssueSeverity.LOW,
    patterns=all_security_patterns,
    restricted_modules=all_restricted_modules,
)
```

### 3. Regular Pattern Updates

```python
# Keep patterns updated with new threats
def update_security_patterns(policy: SecurityPolicy):
    # Add new patterns for emerging threats
    policy.add_pattern(SecurityPattern(
        pattern=r"\b__import__\s*\([^)]*shell=True",
        description="Dynamic import with shell execution",
        severity=SecurityIssueSeverity.HIGH,
    ))

    return policy
```

### 4. Monitoring and Logging

```python
with SandboxSession(lang="python", security_policy=policy, verbose=True) as session:
    is_safe, violations = session.is_safe(code)

    if not is_safe:
        # Log security violations
        logger.warning(f"Security policy blocked code execution:")
        for violation in violations:
            logger.warning(f"  - {violation.description} (Severity: {violation.severity})")
```

## Advanced Features

### Custom Pattern Validators

```python
from llm_sandbox.security import SecurityPattern
from llm_sandbox.exceptions import InvalidRegexPatternError

# Custom validation in patterns
try:
    complex_pattern = SecurityPattern(
        pattern=r"(?i)\b(rm|del|delete)\s+.*(\*|\.\.)",
        description="Dangerous file deletion patterns",
        severity=SecurityIssueSeverity.HIGH,
    )
except InvalidRegexPatternError as e:
    print(f"Invalid regex pattern: {e}")
```

### Policy Inheritance

```python
# Base policy
base_policy = SecurityPolicy(
    severity_threshold=SecurityIssueSeverity.MEDIUM,
    patterns=base_patterns,
    restricted_modules=base_modules,
)

# Extended policy
extended_policy = SecurityPolicy(
    severity_threshold=base_policy.severity_threshold,
    patterns=base_policy.patterns + additional_patterns,
    restricted_modules=base_policy.restricted_modules + additional_modules,
)
```

## Testing Your Security Policies

Use the provided test examples to validate your security policies:

```bash
# Run security policy examples
python examples/security_policy_examples.py

# Run integration tests
python examples/security_integration_tests.py

# Run performance benchmarks
python examples/security_benchmarks.py

# Run unit tests
python -m pytest tests/test_security_features.py -v
```

## Performance Considerations

1. **Pattern Complexity**: Simple patterns are faster than complex regex
2. **Policy Size**: Larger policies take more time to evaluate
3. **Code Size**: Larger code samples take more time to analyze
4. **Caching**: Consider caching pattern compilation results

## Troubleshooting

### Common Issues

1. **Invalid Regex Patterns**:
   ```
   InvalidRegexPatternError: Invalid regex pattern '[unclosed'
   ```
   Solution: Use proper regex escaping and validate patterns

2. **False Positives**:
   - Patterns matching comments or strings
   - Solution: Use more specific patterns with proper boundaries

3. **Performance Issues**:
   - Large policies with many patterns
   - Solution: Optimize patterns and consider severity-based filtering

### Debugging

```python
# Enable verbose logging
with SandboxSession(lang="python", security_policy=policy, verbose=True) as session:
    is_safe, violations = session.is_safe(code)

    # Print detailed violation information
    for violation in violations:
        print(f"Pattern: {violation.pattern}")
        print(f"Description: {violation.description}")
        print(f"Severity: {violation.severity}")
```

## Contributing

To contribute new security patterns or improvements:

1. Add patterns to the appropriate category
2. Include comprehensive tests
3. Document the security risk and rationale
4. Test for false positives
5. Benchmark performance impact

For more information, see the [contributing guide](../CONTRIBUTING.md).
