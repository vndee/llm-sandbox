"""Integration tests for security features with actual code execution.

This module demonstrates how security policies integrate with the actual
code execution pipeline, showing both blocking and allowing behaviors.
"""

import logging
from typing import Tuple, List

from llm_sandbox import SandboxSession
from llm_sandbox.security import (
    SecurityIssueSeverity,
    SecurityPattern,
    DangerousModule,
    SecurityPolicy,
)
from llm_sandbox.data import ConsoleOutput, ExecutionResult

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class SecurityTestRunner:
    """Test runner for security integration tests."""
    
    def __init__(self):
        self.test_results = []
    
    def run_security_test(
        self,
        test_name: str,
        code: str,
        security_policy: SecurityPolicy,
        expected_safe: bool,
        libraries: List[str] = None
    ) -> Tuple[bool, str]:
        """Run a single security test.
        
        Args:
            test_name: Name of the test
            code: Code to test
            security_policy: Security policy to apply
            expected_safe: Whether the code is expected to be safe
            libraries: Optional libraries to install
            
        Returns:
            Tuple of (test_passed, result_message)
        """
        try:
            with SandboxSession(
                lang="python",
                security_policy=security_policy,
                verbose=False
            ) as session:
                # Check if code passes security policy
                is_safe, violations = session.is_safe(code)
                
                result_msg = f"Test: {test_name}\n"
                result_msg += f"  Expected Safe: {expected_safe}, Actual Safe: {is_safe}\n"
                
                if violations:
                    result_msg += f"  Violations ({len(violations)}):  \n"
                    for violation in violations:
                        result_msg += f"    - {violation.description} (Severity: {violation.severity.name})\n"
                
                # Test passed if expectation matches reality
                test_passed = (is_safe == expected_safe)
                
                # If code is safe, try to actually run it
                if is_safe:
                    try:
                        output = session.run(code, libraries=libraries)
                        if isinstance(output, (ConsoleOutput, ExecutionResult)):
                            result_msg += f"  Execution Result: Exit Code {output.exit_code}\n"
                            if output.stdout:
                                result_msg += f"  Output: {output.stdout[:100]}...\n"
                        else:
                            result_msg += f"  Execution successful\n"
                    except Exception as e:
                        result_msg += f"  Execution failed: {str(e)}\n"
                        test_passed = False
                
                self.test_results.append((test_name, test_passed, result_msg))
                return test_passed, result_msg
                
        except Exception as e:
            error_msg = f"Test: {test_name} - ERROR: {str(e)}"
            self.test_results.append((test_name, False, error_msg))
            return False, error_msg
    
    def print_summary(self):
        """Print test summary."""
        passed = sum(1 for _, result, _ in self.test_results if result)
        total = len(self.test_results)
        
        logger.info(f"\n{'='*50}")
        logger.info(f"SECURITY INTEGRATION TEST SUMMARY")
        logger.info(f"{'='*50}")
        logger.info(f"Tests Passed: {passed}/{total}")
        logger.info(f"Success Rate: {(passed/total*100):.1f}%")
        
        # Show failed tests
        failed_tests = [(name, msg) for name, result, msg in self.test_results if not result]
        if failed_tests:
            logger.info(f"\nFailed Tests:")
            for name, msg in failed_tests:
                logger.info(f"❌ {name}")


def create_comprehensive_security_policy() -> SecurityPolicy:
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


def run_basic_security_tests() -> None:
    """Run basic security integration tests."""
    logger.info("Running Basic Security Integration Tests")
    
    runner = SecurityTestRunner()
    policy = create_comprehensive_security_policy()
    
    # Test cases: (name, code, expected_safe, libraries)
    test_cases = [
        # Safe code that should execute
        (
            "Safe Math Operations",
            "import math\nresult = math.sqrt(16) + math.pi\nprint(f'Result: {result}')",
            True,
            None
        ),
        (
            "Safe Data Processing",
            "data = [1, 2, 3, 4, 5]\nresult = sum(x**2 for x in data)\nprint(f'Sum of squares: {result}')",
            True,
            None
        ),
        (
            "Safe JSON Operations",
            "import json\ndata = {'name': 'test', 'value': 42}\nprint(json.dumps(data, indent=2))",
            True,
            None
        ),
        
        # Dangerous code that should be blocked
        (
            "System Command Execution",
            "import os\nos.system('echo Hello from system')",
            False,
            None
        ),
        (
            "Subprocess Execution",
            "import subprocess\nsubprocess.run(['ls', '-la'])",
            False,
            None
        ),
        (
            "Dynamic Code Evaluation",
            "user_input = 'print(\"Hello\")'\neval(user_input)",
            False,
            None
        ),
        (
            "File Write Operations",
            "with open('/tmp/test.txt', 'w') as f:\n    f.write('test data')",
            False,
            None
        ),
        
        # Edge cases
        (
            "Import with Alias",
            "import os as operating_system\nprint('This should be blocked')",
            False,
            None
        ),
        (
            "Commented Dangerous Code",
            "# import os\n# os.system('rm -rf /')\nprint('This is actually safe')",
            True,
            None
        ),
        (
            "String Containing Dangerous Pattern",
            "message = 'The os.system function is dangerous'\nprint(message)",
            True,
            None
        ),
    ]
    
    # Run all test cases
    for test_name, code, expected_safe, libraries in test_cases:
        success, message = runner.run_security_test(
            test_name, code, policy, expected_safe, libraries
        )
        status = "✅ PASS" if success else "❌ FAIL"
        logger.info(f"{status} - {test_name}")
        if not success:
            logger.info(f"  Details: {message}")
    
    runner.print_summary()


def run_severity_level_tests() -> None:
    """Test different severity levels and their blocking behavior."""
    logger.info("\nRunning Severity Level Tests")
    
    runner = SecurityTestRunner()
    
    # Test code with mixed severity violations
    test_code = """
import urllib.request  # LOW severity
import socket         # MEDIUM severity  
result = eval('2 + 2')  # MEDIUM severity
print(f'Result: {result}')
"""
    
    # Test with different safety levels
    severity_tests = [
        (SecurityIssueSeverity.SAFE, True),    # Should allow everything
        (SecurityIssueSeverity.LOW, False),    # Should block LOW and above
        (SecurityIssueSeverity.MEDIUM, False), # Should block MEDIUM and above
        (SecurityIssueSeverity.HIGH, True),    # Should allow (no HIGH severity violations)
    ]
    
    base_policy = create_comprehensive_security_policy()
    
    for safety_level, expected_safe in severity_tests:
        policy = SecurityPolicy(
            safety_level=safety_level,
            patterns=base_policy.patterns,
            dangerous_modules=base_policy.dangerous_modules,
        )
        
        success, message = runner.run_security_test(
            f"Severity Level {safety_level.name}",
            test_code,
            policy,
            expected_safe
        )
        
        status = "✅ PASS" if success else "❌ FAIL"
        logger.info(f"{status} - Safety Level: {safety_level.name}")
        if not success:
            logger.info(f"  Expected: {expected_safe}, Got: {not expected_safe}")
    
    runner.print_summary()


def run_library_installation_tests() -> None:
    """Test security with library installation."""
    logger.info("\nRunning Library Installation Security Tests")
    
    runner = SecurityTestRunner()
    
    # Create a policy that allows requests but blocks os
    patterns = [
        SecurityPattern(
            pattern=r"\bos\.system\s*\(",
            description="System command execution",
            severity=SecurityIssueSeverity.HIGH,
        ),
    ]
    
    dangerous_modules = [
        DangerousModule(
            name="os",
            description="Operating system interface",
            severity=SecurityIssueSeverity.HIGH,
        ),
        # Note: requests is not in dangerous_modules, so it should be allowed
    ]
    
    policy = SecurityPolicy(
        safety_level=SecurityIssueSeverity.MEDIUM,
        patterns=patterns,
        dangerous_modules=dangerous_modules,
    )
    
    test_cases = [
        (
            "Safe Library Usage",
            "import requests\nresponse = requests.get('https://httpbin.org/json')\nprint(response.status_code)",
            True,
            ["requests"]
        ),
        (
            "Numpy Math Operations",
            "import numpy as np\narr = np.array([1, 2, 3, 4, 5])\nprint(f'Mean: {np.mean(arr)}')",
            True,
            ["numpy"]
        ),
        (
            "Pandas Data Analysis",
            "import pandas as pd\ndf = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]})\nprint(df.describe())",
            True,
            ["pandas"]
        ),
    ]
    
    for test_name, code, expected_safe, libraries in test_cases:
        success, message = runner.run_security_test(
            test_name, code, policy, expected_safe, libraries
        )
        status = "✅ PASS" if success else "❌ FAIL"
        logger.info(f"{status} - {test_name}")
        if not success:
            logger.info(f"  Details: {message}")
    
    runner.print_summary()


def run_real_world_attack_scenarios() -> None:
    """Test against real-world attack scenarios."""
    logger.info("\nRunning Real-World Attack Scenario Tests")
    
    runner = SecurityTestRunner()
    policy = create_comprehensive_security_policy()
    
    attack_scenarios = [
        (
            "Command Injection via os.system",
            "import os\nuser_input = 'test; rm -rf /'\nos.system(f'echo {user_input}')",
            False
        ),
        (
            "Code Injection via eval",
            "user_data = '__import__(\"os\").system(\"whoami\")'\neval(user_data)",
            False
        ),
        (
            "File System Traversal",
            "import os\nfilename = '../../../etc/passwd'\nwith open(filename, 'r') as f:\n    print(f.read())",
            True  # This should be allowed as it's just reading (depending on policy)
        ),
        (
            "Network Backdoor Creation",
            "import socket\ns = socket.socket()\ns.bind(('0.0.0.0', 4444))\ns.listen(1)",
            False
        ),
        (
            "Environment Variable Extraction",
            "import os\nsecret = os.environ.get('SECRET_KEY', 'not_found')\nprint(f'Secret: {secret}')",
            False  # If we have patterns for os.environ
        ),
        (
            "Process Information Gathering",
            "import subprocess\nresult = subprocess.run(['ps', 'aux'], capture_output=True, text=True)\nprint(result.stdout)",
            False
        ),
    ]
    
    for test_name, code, expected_safe in attack_scenarios:
        success, message = runner.run_security_test(
            test_name, code, policy, expected_safe
        )
        status = "✅ PASS" if success else "❌ FAIL"
        logger.info(f"{status} - {test_name}")
        if not success:
            logger.info(f"  Expected blocked: {not expected_safe}, Actually blocked: {expected_safe}")
    
    runner.print_summary()


if __name__ == "__main__":
    logger.info("LLM Sandbox Security Integration Tests")
    logger.info("====================================")
    
    try:
        run_basic_security_tests()
        run_severity_level_tests() 
        run_library_installation_tests()
        run_real_world_attack_scenarios()
        
        logger.info("\n🎉 All security integration tests completed!")
        
    except Exception as e:
        logger.error(f"Error during integration testing: {e}")
        raise
