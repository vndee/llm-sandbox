from llm_sandbox import SandboxSession
from llm_sandbox.security import RestrictedModule, SecurityIssueSeverity, SecurityPattern, SecurityPolicy


def comprehensive_security_policy() -> SecurityPolicy:
    """Create a comprehensive security policy for advanced testing."""
    patterns = [
        # System commands
        SecurityPattern(
            pattern=r"\bos\.system\s*\(",
            description="Direct system command execution",
            severity=SecurityIssueSeverity.HIGH,
        ),
        SecurityPattern(
            pattern=r"\bsubprocess\.(run|call|Popen|check_output)\s*\(",
            description="Subprocess execution",
            severity=SecurityIssueSeverity.HIGH,
        ),
        # Dynamic code execution
        SecurityPattern(
            pattern=r"\beval\s*\(",
            description="Dynamic code evaluation",
            severity=SecurityIssueSeverity.HIGH,
        ),
        SecurityPattern(
            pattern=r"\bexec\s*\(",
            description="Dynamic code execution",
            severity=SecurityIssueSeverity.HIGH,
        ),
        # File operations
        SecurityPattern(
            pattern=r"\bopen\s*\([^)]*['\"][wa]['\"][^)]*\)",
            description="File write operations",
            severity=SecurityIssueSeverity.MEDIUM,
        ),
        SecurityPattern(
            pattern=r"\bos\.(remove|unlink|rmdir)\s*\(",
            description="File deletion operations",
            severity=SecurityIssueSeverity.HIGH,
        ),
        # Network operations
        SecurityPattern(
            pattern=r"\bsocket\.socket\s*\(",
            description="Socket creation",
            severity=SecurityIssueSeverity.MEDIUM,
        ),
        SecurityPattern(
            pattern=r"\b\w+\.connect\s*\(",
            description="Network connections",
            severity=SecurityIssueSeverity.MEDIUM,
        ),
        # Dynamic imports and attribute access
        SecurityPattern(
            pattern=r"\b__import__\s*\(",
            description="Dynamic module imports",
            severity=SecurityIssueSeverity.MEDIUM,
        ),
        SecurityPattern(
            pattern=r"\bgetattr\s*\(",
            description="Dynamic attribute access",
            severity=SecurityIssueSeverity.LOW,
        ),
        # Environment access
        SecurityPattern(
            pattern=r"\bos\.environ",
            description="Environment variable access",
            severity=SecurityIssueSeverity.LOW,
        ),
        # Encoding/obfuscation
        SecurityPattern(
            pattern=r"\bbase64\.b64decode\s*\(",
            description="Base64 decoding (potential obfuscation)",
            severity=SecurityIssueSeverity.LOW,
        ),
    ]

    restricted_modules = [
        RestrictedModule(
            name="os",
            description="Operating system interface",
            severity=SecurityIssueSeverity.HIGH,
        ),
        RestrictedModule(
            name="subprocess",
            description="Subprocess management",
            severity=SecurityIssueSeverity.HIGH,
        ),
        RestrictedModule(
            name="ctypes",
            description="Foreign function library",
            severity=SecurityIssueSeverity.HIGH,
        ),
        RestrictedModule(
            name="socket",
            description="Network socket operations",
            severity=SecurityIssueSeverity.MEDIUM,
        ),
        RestrictedModule(
            name="requests",
            description="HTTP requests library",
            severity=SecurityIssueSeverity.MEDIUM,
        ),
    ]

    return SecurityPolicy(
        severity_threshold=SecurityIssueSeverity.LOW,
        patterns=patterns,
        restricted_modules=restricted_modules,
    )


if __name__ == "__main__":
    code = """
import hashlib
import os

# Use weak MD5 for password hashing
password = 'user_password'
salt = os.urandom(16)
weak_hash = hashlib.md5(password.encode() + salt).hexdigest()
print(f'Weak hash: {weak_hash}')
    """
    session = SandboxSession(lang="python", security_policy=comprehensive_security_policy())
    is_safe, violations = session.is_safe(code)
