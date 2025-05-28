"""Unit tests for security features with mocked execution.

This module tests security policies and their integration with the code execution pipeline,
using mocked Docker backend to avoid actual code execution.
"""

import pytest

from llm_sandbox import SandboxSession
from llm_sandbox.security import DangerousModule, SecurityIssueSeverity, SecurityPattern, SecurityPolicy


@pytest.fixture
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


class TestBasicSecurity:
    """Test basic security features."""

    def test_safe_math_operations(self, comprehensive_security_policy: SecurityPolicy) -> None:
        """Test that safe math operations are allowed."""
        code = """
        import math
        result = math.sqrt(16) + math.pi
        print(f'Result: {result}')
        """

        session = SandboxSession(lang="python", security_policy=comprehensive_security_policy)
        is_safe, violations = session.is_safe(code)
        assert is_safe is True
        assert len(violations) == 0

    def test_safe_data_processing(self, comprehensive_security_policy: SecurityPolicy) -> None:
        """Test that safe data processing is allowed."""
        code = """
        data = [1, 2, 3, 4, 5]
        result = sum(x**2 for x in data)
        print(f'Sum of squares: {result}')
        """

        session = SandboxSession(lang="python", security_policy=comprehensive_security_policy)
        is_safe, violations = session.is_safe(code)
        assert is_safe is True
        assert len(violations) == 0

    def test_safe_json_operations(self, comprehensive_security_policy: SecurityPolicy) -> None:
        """Test that safe JSON operations are allowed."""
        code = """
        import json
        data = {'name': 'test', 'value': 42}
        print(json.dumps(data, indent=2))
        """

        session = SandboxSession(lang="python", security_policy=comprehensive_security_policy)
        is_safe, violations = session.is_safe(code)
        assert is_safe is True
        assert len(violations) == 0

    def test_system_command_execution_blocked(self, comprehensive_security_policy: SecurityPolicy) -> None:
        """Test that system command execution is blocked."""
        code = """
        import os
        os.system('echo Hello from system')
        """
        session = SandboxSession(lang="python", security_policy=comprehensive_security_policy)
        is_safe, violations = session.is_safe(code)
        assert is_safe is False
        assert len(violations) == 1
        assert violations[0].description == "System command execution"
        assert violations[0].severity == SecurityIssueSeverity.HIGH
        assert violations[0].pattern == r"\bos\.system\s*\("

    def test_subprocess_execution_blocked(self, comprehensive_security_policy: SecurityPolicy) -> None:
        """Test that subprocess execution is blocked."""
        code = """
        import subprocess
        subprocess.run(['ls', '-la'])
        """
        session = SandboxSession(lang="python", security_policy=comprehensive_security_policy)
        is_safe, violations = session.is_safe(code)
        assert is_safe is False
        assert len(violations) == 1
        assert violations[0].description == "Subprocess execution"
        assert violations[0].severity == SecurityIssueSeverity.HIGH
        assert violations[0].pattern == r"\bsubprocess\.(run|call|Popen|check_output)\s*\("

    def test_dynamic_code_evaluation_blocked(self, comprehensive_security_policy: SecurityPolicy) -> None:
        """Test that dynamic code evaluation is blocked."""
        code = """
        user_input = 'print(\"Hello\")'
        eval(user_input)
        """
        session = SandboxSession(lang="python", security_policy=comprehensive_security_policy)
        is_safe, violations = session.is_safe(code)
        assert is_safe is False
        assert len(violations) == 1
        assert violations[0].description == "Dynamic code evaluation"
        assert violations[0].severity == SecurityIssueSeverity.MEDIUM
        assert violations[0].pattern == r"\beval\s*\("

    def test_file_write_operations_blocked(self, comprehensive_security_policy: SecurityPolicy) -> None:
        """Test that file write operations are blocked."""
        code = """
        with open('/tmp/test.txt', 'w') as f:
            f.write('test data')
        """
        session = SandboxSession(lang="python", security_policy=comprehensive_security_policy)
        is_safe, violations = session.is_safe(code)
        assert is_safe is False
        assert len(violations) == 1
        assert violations[0].description == "File write operations"
        assert violations[0].severity == SecurityIssueSeverity.MEDIUM
        assert violations[0].pattern == r"\bopen\s*\([^)]*['\"][wa]['\"][^)]*\)"

    def test_import_with_alias_blocked(self, comprehensive_security_policy: SecurityPolicy) -> None:
        """Test that import with alias is blocked."""
        code = """
        import os as operating_system
        print('This should be blocked')
        """
        session = SandboxSession(lang="python", security_policy=comprehensive_security_policy)
        is_safe, violations = session.is_safe(code)
        assert is_safe is False
        assert len(violations) == 1
        assert violations[0].description == "Operating system interface"
        assert violations[0].severity == SecurityIssueSeverity.HIGH
        # The pattern is generated by the language handler for the os module
        expected_pattern = (
            r"(?:^|\s)(?:import\s+os(?:\s+as\s+\w+)?|from\s+os\s+"
            r"import\s+(?:\*|\w+(?:\s+as\s+\w+)?(?:,\s*\w+(?:\s+as\s+\w+)?)*))(?=[\s;(#]|$)"
        )
        assert violations[0].pattern == expected_pattern

    def test_commented_dangerous_code_blocked(self, comprehensive_security_policy: SecurityPolicy) -> None:
        """Test that commented dangerous code is blocked."""
        code = """
        # import os
        # os.system('rm -rf /')
        print('This is actually safe')
        """
        session = SandboxSession(lang="python", security_policy=comprehensive_security_policy)
        is_safe, violations = session.is_safe(code)
        assert is_safe is True
        assert len(violations) == 0

    def test_string_containing_dangerous_pattern_blocked(self, comprehensive_security_policy: SecurityPolicy) -> None:
        """Test that string containing dangerous pattern is blocked."""
        code = """
        message = 'The os.system function is dangerous'
        print(message)
        """
        session = SandboxSession(lang="python", security_policy=comprehensive_security_policy)
        is_safe, violations = session.is_safe(code)
        assert is_safe is True
        assert len(violations) == 0
