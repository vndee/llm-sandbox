# ruff: noqa: E501, SLF001, ARG002, PLR2004, FBT003
"""Unit tests for security features with mocked execution.

This module tests security policies and their integration with the code execution pipeline,
using mocked Docker backend to avoid actual code execution.
"""

import pytest

from llm_sandbox import SandboxSession
from llm_sandbox.security import RestrictedModule, SecurityIssueSeverity, SecurityPattern, SecurityPolicy

# Apply mock_docker_backend fixture to all tests in this module
pytestmark = pytest.mark.usefixtures("mock_docker_backend")


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
        # File reading operations
        SecurityPattern(
            pattern=r"\bopen\s*\(.*['\"]r['\"]",  # or a more robust regex for file reading
            description="File reading operations",
            severity=SecurityIssueSeverity.MEDIUM,
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
            name="socket",
            description="Network socket operations",
            severity=SecurityIssueSeverity.MEDIUM,
        ),
        RestrictedModule(
            name="urllib",
            description="URL handling library",
            severity=SecurityIssueSeverity.LOW,
        ),
        RestrictedModule(
            name="requests",
            description="HTTP requests library",
            severity=SecurityIssueSeverity.LOW,
        ),
    ]

    return SecurityPolicy(
        severity_threshold=SecurityIssueSeverity.MEDIUM,
        patterns=patterns,
        restricted_modules=restricted_modules,
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

    def test_dangerous_code_blocked(self, comprehensive_security_policy: SecurityPolicy) -> None:
        """Test that dangerous code is blocked."""
        code = """
        import os
        os.system('rm -rf /')
        """
        session = SandboxSession(lang="python", security_policy=comprehensive_security_policy)
        is_safe, violations = session.is_safe(code)
        assert is_safe is False
        assert len(violations) == 1
        assert violations[0].description == "System command execution"
        assert violations[0].severity == SecurityIssueSeverity.HIGH
        assert violations[0].pattern == r"\bos\.system\s*\("


@pytest.fixture
def strict_security_policy() -> SecurityPolicy:
    """Create a strict security policy for testing."""
    patterns = [
        # File system operations
        SecurityPattern(
            pattern=r"\bopen\s*\(.*['\"]w['\"].*\)",
            description="File writing operations",
            severity=SecurityIssueSeverity.HIGH,
        ),
        SecurityPattern(
            pattern=r"\bos\.(remove|unlink|rmdir)\s*\(",
            description="File deletion operations",
            severity=SecurityIssueSeverity.HIGH,
        ),
        # System operations
        SecurityPattern(
            pattern=r"\bos\.environ",
            description="Environment variable access",
            severity=SecurityIssueSeverity.MEDIUM,
        ),
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
            name="socket",
            description="Network socket operations",
            severity=SecurityIssueSeverity.MEDIUM,
        ),
        RestrictedModule(
            name="urllib",
            description="URL handling library",
            severity=SecurityIssueSeverity.LOW,
        ),
    ]

    return SecurityPolicy(
        severity_threshold=SecurityIssueSeverity.LOW,  # Very strict - block even low-severity issues
        patterns=patterns,
        restricted_modules=restricted_modules,
    )


@pytest.fixture
def permissive_security_policy() -> SecurityPolicy:
    """Create a permissive security policy for testing."""
    patterns = [
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
    ]

    return SecurityPolicy(
        severity_threshold=SecurityIssueSeverity.HIGH,  # Only block high-severity issues
        patterns=patterns,
        restricted_modules=restricted_modules,
    )


class TestSeverityLevels:
    """Test different severity levels and their blocking behavior."""

    def test_safe_level_allows_everything(self, comprehensive_security_policy: SecurityPolicy) -> None:
        """Test that SAFE level allows everything."""
        code = """
        import urllib.request
        import socket
        result = eval('2 + 2')
        print(f'Result: {result}')
        """

        policy = SecurityPolicy(
            severity_threshold=SecurityIssueSeverity.SAFE,
            patterns=comprehensive_security_policy.patterns,
            restricted_modules=comprehensive_security_policy.restricted_modules,
        )

        session = SandboxSession(lang="python", security_policy=policy)
        is_safe, violations = session.is_safe(code)
        assert is_safe is True
        assert len(violations or []) == 3

    def test_low_level_blocks_low_and_above(self, comprehensive_security_policy: SecurityPolicy) -> None:
        """Test that LOW level blocks LOW severity and above."""
        code = """
        import urllib.request
        import socket
        result = eval('2 + 2')
        print(f'Result: {result}')
        """

        policy = SecurityPolicy(
            severity_threshold=SecurityIssueSeverity.LOW,
            patterns=comprehensive_security_policy.patterns,
            restricted_modules=comprehensive_security_policy.restricted_modules,
        )

        session = SandboxSession(lang="python", security_policy=policy)
        is_safe, violations = session.is_safe(code)
        assert is_safe is False
        assert len(violations) > 0
        # Should have violations for urllib (LOW) and socket (MEDIUM)

    def test_medium_level_blocks_medium_and_above(self, comprehensive_security_policy: SecurityPolicy) -> None:
        """Test that MEDIUM level blocks MEDIUM severity and above."""
        code = """
        import urllib.request
        import socket
        result = eval('2 + 2')
        print(f'Result: {result}')
        """

        policy = SecurityPolicy(
            severity_threshold=SecurityIssueSeverity.MEDIUM,
            patterns=comprehensive_security_policy.patterns,
            restricted_modules=comprehensive_security_policy.restricted_modules,
        )

        session = SandboxSession(lang="python", security_policy=policy)
        is_safe, violations = session.is_safe(code)
        assert is_safe is False
        assert len(violations) > 0
        # Should have violations for socket (MEDIUM) and eval (MEDIUM), but not urllib (LOW)

    def test_high_level_blocks_only_high(self, comprehensive_security_policy: SecurityPolicy) -> None:
        """Test that HIGH level only blocks HIGH severity violations."""
        code = """
        import urllib.request
        import socket
        result = eval('2 + 2')
        print(f'Result: {result}')
        """

        policy = SecurityPolicy(
            severity_threshold=SecurityIssueSeverity.HIGH,
            patterns=comprehensive_security_policy.patterns,
            restricted_modules=comprehensive_security_policy.restricted_modules,
        )

        session = SandboxSession(lang="python", security_policy=policy)
        is_safe, violations = session.is_safe(code)
        assert is_safe is True  # No HIGH severity violations in this code
        assert len(violations or []) == 3

    def test_high_severity_violation_blocked_at_all_levels(self, comprehensive_security_policy: SecurityPolicy) -> None:
        """Test that HIGH severity violations are blocked at all levels except SAFE."""
        code = """
        import os
        os.system('echo test')
        """

        for level in [SecurityIssueSeverity.LOW, SecurityIssueSeverity.MEDIUM, SecurityIssueSeverity.HIGH]:
            policy = SecurityPolicy(
                severity_threshold=level,
                patterns=comprehensive_security_policy.patterns,
                restricted_modules=comprehensive_security_policy.restricted_modules,
            )

            session = SandboxSession(lang="python", security_policy=policy)
            is_safe, violations = session.is_safe(code)
            assert is_safe is False
            assert len(violations) > 0


class TestLibraryInstallation:
    """Test security with library installation scenarios."""

    def test_safe_library_usage_requests(self) -> None:
        """Test that safe library usage (requests) is allowed."""
        code = """
        import requests
        response = requests.get('https://httpbin.org/json')
        print(response.status_code)
        """

        # Create a policy that allows requests but blocks os
        patterns = [
            SecurityPattern(
                pattern=r"\bos\.system\s*\(",
                description="System command execution",
                severity=SecurityIssueSeverity.HIGH,
            ),
        ]

        restricted_modules = [
            RestrictedModule(
                name="os",
                description="Operating system interface",
                severity=SecurityIssueSeverity.HIGH,
            ),
        ]

        policy = SecurityPolicy(
            severity_threshold=SecurityIssueSeverity.MEDIUM,
            patterns=patterns,
            restricted_modules=restricted_modules,
        )

        session = SandboxSession(lang="python", security_policy=policy)
        is_safe, violations = session.is_safe(code)
        assert is_safe is True
        assert len(violations) == 0

    def test_safe_library_usage_numpy(self) -> None:
        """Test that safe library usage (numpy) is allowed."""
        code = """
        import numpy as np
        arr = np.array([1, 2, 3, 4, 5])
        print(f'Mean: {np.mean(arr)}')
        """

        patterns = [
            SecurityPattern(
                pattern=r"\bos\.system\s*\(",
                description="System command execution",
                severity=SecurityIssueSeverity.HIGH,
            ),
        ]

        restricted_modules = [
            RestrictedModule(
                name="os",
                description="Operating system interface",
                severity=SecurityIssueSeverity.HIGH,
            ),
        ]

        policy = SecurityPolicy(
            severity_threshold=SecurityIssueSeverity.MEDIUM,
            patterns=patterns,
            restricted_modules=restricted_modules,
        )

        session = SandboxSession(lang="python", security_policy=policy)
        is_safe, violations = session.is_safe(code)
        assert is_safe is True
        assert len(violations) == 0

    def test_safe_library_usage_pandas(self) -> None:
        """Test that safe library usage (pandas) is allowed."""
        code = """
        import pandas as pd
        df = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]})
        print(df.describe())
        """

        patterns = [
            SecurityPattern(
                pattern=r"\bos\.system\s*\(",
                description="System command execution",
                severity=SecurityIssueSeverity.HIGH,
            ),
        ]

        restricted_modules = [
            RestrictedModule(
                name="os",
                description="Operating system interface",
                severity=SecurityIssueSeverity.HIGH,
            ),
        ]

        policy = SecurityPolicy(
            severity_threshold=SecurityIssueSeverity.MEDIUM,
            patterns=patterns,
            restricted_modules=restricted_modules,
        )

        session = SandboxSession(lang="python", security_policy=policy)
        is_safe, violations = session.is_safe(code)
        assert is_safe is True
        assert len(violations) == 0

    def test_dangerous_library_usage_blocked(self) -> None:
        """Test that dangerous library usage is blocked even with safe libraries allowed."""
        code = """
        import requests
        import os
        response = requests.get('https://httpbin.org/json')
        os.system('rm -rf /')
        """

        patterns = [
            SecurityPattern(
                pattern=r"\bos\.system\s*\(",
                description="System command execution",
                severity=SecurityIssueSeverity.HIGH,
            ),
        ]

        restricted_modules = [
            RestrictedModule(
                name="os",
                description="Operating system interface",
                severity=SecurityIssueSeverity.HIGH,
            ),
        ]

        policy = SecurityPolicy(
            severity_threshold=SecurityIssueSeverity.MEDIUM,
            patterns=patterns,
            restricted_modules=restricted_modules,
        )

        session = SandboxSession(lang="python", security_policy=policy)
        is_safe, violations = session.is_safe(code)
        assert is_safe is False
        assert len(violations) >= 1


class TestRealWorldAttackScenarios:
    """Test against real-world attack scenarios."""

    def test_command_injection_via_os_system(self, comprehensive_security_policy: SecurityPolicy) -> None:
        """Test that command injection via os.system is blocked."""
        code = """
        import os
        user_input = 'test; rm -rf /'
        os.system(f'echo {user_input}')
        """

        session = SandboxSession(lang="python", security_policy=comprehensive_security_policy)
        is_safe, violations = session.is_safe(code)
        assert is_safe is False
        assert len(violations) >= 1
        assert any(v.description == "System command execution" for v in violations)

    def test_code_injection_via_eval(self, comprehensive_security_policy: SecurityPolicy) -> None:
        """Test that code injection via eval is blocked."""
        code = """
        user_data = '__import__("os").system("whoami")'
        eval(user_data)
        """

        session = SandboxSession(lang="python", security_policy=comprehensive_security_policy)
        is_safe, violations = session.is_safe(code)
        assert is_safe is False
        assert len(violations) >= 1
        assert any(v.description == "Dynamic code evaluation" for v in violations)

    def test_file_system_traversal_read(self, comprehensive_security_policy: SecurityPolicy) -> None:
        """Test file system traversal for reading (might be allowed depending on policy)."""
        code = """
        import os
        filename = '../../../etc/passwd'
        with open(filename, 'r') as f:
            print(f.read())
        """

        session = SandboxSession(lang="python", security_policy=comprehensive_security_policy)
        is_safe, violations = session.is_safe(code)
        assert is_safe is False
        assert len(violations) >= 1
        assert any(v.description == "File reading operations" for v in violations)

    def test_network_backdoor_creation(self, comprehensive_security_policy: SecurityPolicy) -> None:
        """Test that network backdoor creation is blocked."""
        code = """
        import socket
        s = socket.socket()
        s.bind(('0.0.0.0', 4444))
        s.listen(1)
        """

        session = SandboxSession(lang="python", security_policy=comprehensive_security_policy)
        is_safe, violations = session.is_safe(code)
        assert is_safe is False
        assert len(violations) >= 1
        assert any(v.description == "Raw socket creation" for v in violations)

    def test_environment_variable_extraction(self, strict_security_policy: SecurityPolicy) -> None:
        """Test that environment variable extraction is blocked."""
        code = """
        import os
        secret = os.environ.get('SECRET_KEY', 'not_found')
        print(f'Secret: {secret}')
        """

        session = SandboxSession(lang="python", security_policy=strict_security_policy)
        is_safe, violations = session.is_safe(code)
        assert is_safe is False
        assert len(violations) >= 1
        assert any(v.description == "Environment variable access" for v in violations)

    def test_process_information_gathering(self, comprehensive_security_policy: SecurityPolicy) -> None:
        """Test that process information gathering is blocked."""
        code = """
        import subprocess
        result = subprocess.run(['ps', 'aux'], capture_output=True, text=True)
        print(result.stdout)
        """

        session = SandboxSession(lang="python", security_policy=comprehensive_security_policy)
        is_safe, violations = session.is_safe(code)
        assert is_safe is False
        assert len(violations) >= 1
        assert any(v.description == "Subprocess execution" for v in violations)

    def test_data_science_safe_scenario(self, comprehensive_security_policy: SecurityPolicy) -> None:
        """Test that safe data science code is allowed."""
        code = """
        import pandas as pd
        import numpy as np
        import matplotlib.pyplot as plt

        data = pd.DataFrame({'x': [1, 2, 3], 'y': [4, 5, 6]})
        print(data.describe())
        """

        session = SandboxSession(lang="python", security_policy=comprehensive_security_policy)
        is_safe, violations = session.is_safe(code)
        assert is_safe is True
        assert len(violations) == 0

    def test_web_scraping_suspicious_scenario(self, strict_security_policy: SecurityPolicy) -> None:
        """Test that suspicious web scraping code is blocked."""
        code = """
        import requests
        import os

        url = 'https://example.com'
        response = requests.get(url)
        os.environ['SECRET_KEY'] = response.text
        """

        session = SandboxSession(lang="python", security_policy=strict_security_policy)
        is_safe, violations = session.is_safe(code)
        assert is_safe is False
        assert len(violations) >= 1

    def test_file_processing_dangerous_scenario(self, comprehensive_security_policy: SecurityPolicy) -> None:
        """Test that dangerous file processing is blocked."""
        code = """
        import os
        import shutil

        for root, dirs, files in os.walk('/home'):
            for file in files:
                if file.endswith('.txt'):
                    os.remove(os.path.join(root, file))
        """

        session = SandboxSession(lang="python", security_policy=comprehensive_security_policy)
        is_safe, violations = session.is_safe(code)
        assert is_safe is False
        assert len(violations) >= 1

    def test_machine_learning_safe_scenario(self, comprehensive_security_policy: SecurityPolicy) -> None:
        """Test that safe machine learning code is allowed."""
        code = """
        import sklearn
        from sklearn.linear_model import LinearRegression
        import joblib

        model = LinearRegression()
        # Training code would go here
        joblib.dump(model, 'model.pkl')
        """

        session = SandboxSession(lang="python", security_policy=comprehensive_security_policy)
        is_safe, violations = session.is_safe(code)
        assert is_safe is True
        assert len(violations) == 0
