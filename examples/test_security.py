from llm_sandbox import SandboxSession
from llm_sandbox.security import DangerousModule, SecurityIssueSeverity, SecurityPattern, SecurityPolicy


def comprehensive_security_policy() -> SecurityPolicy:
    """Create a comprehensive security policy for testing."""
    patterns = [
        # System operations
        SecurityPattern(
            pattern=r"\bos\.system\s*\(",
            description="System command execution",
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
            severity=SecurityIssueSeverity.MEDIUM,
        ),
        SecurityPattern(
            pattern=r"\bexec\s*\(",
            description="Dynamic code execution",
            severity=SecurityIssueSeverity.MEDIUM,
        ),
        # File operations
        SecurityPattern(
            pattern=r"\bopen\s*\([^)]*['\"][wa]['\"][^)]*\)",
            description="File write operations",
            severity=SecurityIssueSeverity.MEDIUM,
        ),
        # Network operations
        SecurityPattern(
            pattern=r"\bsocket\.socket\s*\(",
            description="Raw socket creation",
            severity=SecurityIssueSeverity.LOW,
        ),
    ]

    dangerous_modules = [
        DangerousModule(
            name="os",
            description="Operating system interface",
            severity=SecurityIssueSeverity.HIGH,
        ),
        DangerousModule(
            name="subprocess",
            description="Subprocess management",
            severity=SecurityIssueSeverity.HIGH,
        ),
        DangerousModule(
            name="socket",
            description="Network socket operations",
            severity=SecurityIssueSeverity.MEDIUM,
        ),
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
    ]

    return SecurityPolicy(
        safety_level=SecurityIssueSeverity.MEDIUM,
        patterns=patterns,
        dangerous_modules=dangerous_modules,
    )


if __name__ == "__main__":
    code = """
        import os
        os.system('echo Hello from system')
        """
    session = SandboxSession(lang="python", security_policy=comprehensive_security_policy())
    is_safe, violations = session.is_safe(code)
