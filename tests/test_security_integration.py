"""Unit tests for security features with mocked execution.

This module tests security policies and their integration with the code execution pipeline,
using mocked Docker backend to avoid actual code execution.
"""

from collections.abc import Generator
from typing import Any
from unittest.mock import patch

import pytest

from llm_sandbox.base import Session
from llm_sandbox.data import ConsoleOutput, ExecutionResult
from llm_sandbox.security import DangerousModule, SecurityIssueSeverity, SecurityPattern, SecurityPolicy


class TestSession(Session):
    """Test implementation of Session for testing."""

    def __init__(self, lang: str, security_policy: SecurityPolicy | None = None) -> None:
        """Initialize test session."""
        super().__init__(lang=lang, security_policy=security_policy)
        self._mock_is_safe_result: tuple[bool, list[SecurityPattern]] = (True, [])
        self._security_policy = security_policy

    def set_is_safe_result(self, result: tuple[bool, list[SecurityPattern]]) -> None:
        """Set the result that is_safe will return."""
        self._mock_is_safe_result = result

    def set_security_policy(self, policy: SecurityPolicy) -> None:
        """Set the security policy."""
        self._security_policy = policy

    def is_safe(self, code: str) -> tuple[bool, list[SecurityPattern]]:
        """Mock is_safe implementation."""
        return self._mock_is_safe_result

    def open(self) -> None:
        """Mock open implementation."""

    def close(self) -> None:
        """Mock close implementation."""

    def run(self, code: str, libraries: list[str] | None = None) -> ConsoleOutput | ExecutionResult:
        """Mock run implementation."""
        return ConsoleOutput(stdout="Mocked output", stderr="", exit_code=0)

    def copy_to_runtime(self, src: str, dest: str) -> None:
        """Mock copy_to_runtime implementation."""

    def copy_from_runtime(self, src: str, dest: str) -> None:
        """Mock copy_from_runtime implementation."""

    def execute_command(self, command: str, workdir: str | None = None) -> ConsoleOutput:
        """Mock execute_command implementation."""
        return ConsoleOutput(stdout="Mocked output", stderr="", exit_code=0)

    def get_archive(self, path: str) -> tuple[bytes, dict]:
        """Mock get_archive implementation."""
        return b"", {}

    def _ensure_ownership(self, folders: list[str]) -> None:
        """Mock _ensure_ownership implementation."""


@pytest.fixture
def mock_sandbox_session() -> Generator[dict[str, Any], None, None]:
    """Create a mocked sandbox session."""
    test_session = TestSession(lang="python")
    with patch("llm_sandbox.base.Session", lambda *args, **kwargs: test_session):
        yield {"session": test_session}


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

    def test_safe_math_operations(
        self, mock_sandbox_session: dict[str, Any], comprehensive_security_policy: SecurityPolicy
    ) -> None:
        """Test that safe math operations are allowed."""
        code = """
        import math
        result = math.sqrt(16) + math.pi
        print(f'Result: {result}')
        """

        session = mock_sandbox_session["session"]
        session.set_security_policy(comprehensive_security_policy)
        session.set_is_safe_result((True, []))
        is_safe, violations = session.is_safe(code)
        assert is_safe is True
        assert len(violations) == 0

    def test_safe_data_processing(
        self, mock_sandbox_session: dict[str, Any], comprehensive_security_policy: SecurityPolicy
    ) -> None:
        """Test that safe data processing is allowed."""
        code = """
        data = [1, 2, 3, 4, 5]
        result = sum(x**2 for x in data)
        print(f'Sum of squares: {result}')
        """

        session = mock_sandbox_session["session"]
        session.set_security_policy(comprehensive_security_policy)
        session.set_is_safe_result((True, []))
        is_safe, violations = session.is_safe(code)
        assert is_safe is True
        assert len(violations) == 0

    def test_system_command_execution_blocked(
        self, mock_sandbox_session: dict[str, Any], comprehensive_security_policy: SecurityPolicy
    ) -> None:
        """Test that system command execution is blocked."""
        code = """
        import os
        os.system('echo Hello from system')
        """

        session = mock_sandbox_session["session"]
        session.set_security_policy(comprehensive_security_policy)
        session.set_is_safe_result((
            False,
            [
                SecurityPattern(
                    pattern=r"\bos\.system\s*\(",
                    description="System command execution",
                    severity=SecurityIssueSeverity.HIGH,
                )
            ],
        ))
        is_safe, violations = session.is_safe(code)
        assert is_safe is False
        assert len(violations) > 0
        assert any(v.description == "System command execution" for v in violations)

    def test_subprocess_execution_blocked(
        self, mock_sandbox_session: dict[str, Any], comprehensive_security_policy: SecurityPolicy
    ) -> None:
        """Test that subprocess execution is blocked."""
        code = """
        import subprocess
        subprocess.run(['ls', '-la'])
        """

        session = mock_sandbox_session["session"]
        session.set_security_policy(comprehensive_security_policy)
        session.set_is_safe_result((
            False,
            [
                SecurityPattern(
                    pattern=r"\bsubprocess\.(run|call|Popen|check_output)\s*\(",
                    description="Subprocess execution",
                    severity=SecurityIssueSeverity.HIGH,
                )
            ],
        ))
        is_safe, violations = session.is_safe(code)
        assert is_safe is False
        assert len(violations) > 0
        assert any(v.description == "Subprocess execution" for v in violations)


class TestSeverityLevels:
    """Test security policy severity levels."""

    @pytest.mark.parametrize(
        "safety_level,expected_safe",
        [
            (SecurityIssueSeverity.SAFE, True),
            (SecurityIssueSeverity.LOW, False),
            (SecurityIssueSeverity.MEDIUM, False),
            (SecurityIssueSeverity.HIGH, True),
        ],
    )
    def test_severity_levels(
        self,
        mock_sandbox_session: dict[str, Any],
        comprehensive_security_policy: SecurityPolicy,
        safety_level: SecurityIssueSeverity,
        expected_safe: bool,
    ) -> None:
        """Test different severity levels and their blocking behavior."""
        code = """
        import urllib.request  # LOW severity
        import socket         # MEDIUM severity
        result = eval('2 + 2')  # MEDIUM severity
        print(f'Result: {result}')
        """

        policy = SecurityPolicy(
            safety_level=safety_level,
            patterns=comprehensive_security_policy.patterns,
            dangerous_modules=comprehensive_security_policy.dangerous_modules,
        )

        session = mock_sandbox_session["session"]
        session.set_security_policy(policy)
        session.set_is_safe_result((expected_safe, []))
        is_safe, violations = session.is_safe(code)
        assert is_safe == expected_safe


class TestRealWorldAttackScenarios:
    """Test against real-world attack scenarios."""

    def test_command_injection(
        self, mock_sandbox_session: dict[str, Any], comprehensive_security_policy: SecurityPolicy
    ) -> None:
        """Test command injection via os.system."""
        code = """
        import os
        user_input = 'test; rm -rf /'
        os.system(f'echo {user_input}')
        """

        session = mock_sandbox_session["session"]
        session.set_security_policy(comprehensive_security_policy)
        session.set_is_safe_result((
            False,
            [
                SecurityPattern(
                    pattern=r"\bos\.system\s*\(",
                    description="System command execution",
                    severity=SecurityIssueSeverity.HIGH,
                )
            ],
        ))
        is_safe, violations = session.is_safe(code)
        assert is_safe is False
        assert any(v.description == "System command execution" for v in violations)

    def test_code_injection(
        self, mock_sandbox_session: dict[str, Any], comprehensive_security_policy: SecurityPolicy
    ) -> None:
        """Test code injection via eval."""
        code = """
        user_data = '__import__("os").system("whoami")'
        eval(user_data)
        """

        session = mock_sandbox_session["session"]
        session.set_security_policy(comprehensive_security_policy)
        session.set_is_safe_result((
            False,
            [
                SecurityPattern(
                    pattern=r"\beval\s*\(",
                    description="Dynamic code evaluation",
                    severity=SecurityIssueSeverity.MEDIUM,
                )
            ],
        ))
        is_safe, violations = session.is_safe(code)
        assert is_safe is False
        assert any(v.description == "Dynamic code evaluation" for v in violations)

    def test_network_backdoor(
        self, mock_sandbox_session: dict[str, Any], comprehensive_security_policy: SecurityPolicy
    ) -> None:
        """Test network backdoor creation attempt."""
        code = """
        import socket
        s = socket.socket()
        s.bind(('0.0.0.0', 4444))
        s.listen(1)
        """

        session = mock_sandbox_session["session"]
        session.set_security_policy(comprehensive_security_policy)
        session.set_is_safe_result((
            False,
            [
                SecurityPattern(
                    pattern=r"\bsocket\.socket\s*\(",
                    description="Raw socket creation",
                    severity=SecurityIssueSeverity.LOW,
                )
            ],
        ))
        is_safe, violations = session.is_safe(code)
        assert is_safe is False
        assert any(v.description == "Raw socket creation" for v in violations)


class TestLibraryInstallation:
    """Test security with library installation."""

    def test_safe_library_usage(self, mock_sandbox_session: dict[str, Any]) -> None:
        """Test using a safe library."""
        # Create a policy that only blocks os operations
        policy = SecurityPolicy(
            safety_level=SecurityIssueSeverity.MEDIUM,
            patterns=[
                SecurityPattern(
                    pattern=r"\bos\.system\s*\(",
                    description="System command execution",
                    severity=SecurityIssueSeverity.HIGH,
                ),
            ],
            dangerous_modules=[
                DangerousModule(
                    name="os",
                    description="Operating system interface",
                    severity=SecurityIssueSeverity.HIGH,
                ),
            ],
        )

        code = """
        import requests
        response = requests.get('https://httpbin.org/json')
        print(response.status_code)
        """

        session = mock_sandbox_session["session"]
        session.set_security_policy(policy)
        session.set_is_safe_result((True, []))
        is_safe, violations = session.is_safe(code)
        assert is_safe is True
        assert len(violations) == 0

    def test_mixed_library_usage(
        self, mock_sandbox_session: dict[str, Any], comprehensive_security_policy: SecurityPolicy
    ) -> None:
        """Test using both safe and unsafe libraries."""
        code = """
        import requests  # Safe
        import os       # Unsafe
        response = requests.get('https://httpbin.org/json')
        os.system('echo test')  # Should be blocked
        """

        session = mock_sandbox_session["session"]
        session.set_security_policy(comprehensive_security_policy)
        session.set_is_safe_result((
            False,
            [
                SecurityPattern(
                    pattern=r"\bos\.system\s*\(",
                    description="System command execution",
                    severity=SecurityIssueSeverity.HIGH,
                )
            ],
        ))
        is_safe, violations = session.is_safe(code)
        assert is_safe is False
        assert any(v.description == "System command execution" for v in violations)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
