"""Comprehensive examples demonstrating the new security policy features in LLM Sandbox.

This module shows how to create and use various security policies to protect against
malicious or dangerous code execution in the sandbox environment.
"""

import logging

from llm_sandbox import SandboxSession
from llm_sandbox.security import RestrictedModule, SecurityIssueSeverity, SecurityPattern, SecurityPolicy

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def create_basic_security_policy() -> SecurityPolicy:
    """Create a basic security policy for demonstration.

    Returns:
        SecurityPolicy: A basic security policy with common dangerous patterns.

    """
    patterns = [
        SecurityPattern(
            pattern=r"\bos\.system\s*\(",
            description="Dangerous system command execution",
            severity=SecurityIssueSeverity.HIGH,
        ),
        SecurityPattern(
            pattern=r"\bsubprocess\.(run|call|Popen)\s*\(",
            description="Subprocess execution detected",
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
        SecurityPattern(
            pattern=r"\b__import__\s*\(",
            description="Dynamic module import",
            severity=SecurityIssueSeverity.LOW,
        ),
    ]

    restricted_modules = [
        RestrictedModule(
            name="socket",
            description="Network socket operations",
            severity=SecurityIssueSeverity.MEDIUM,
        ),
        RestrictedModule(
            name="urllib",
            description="URL handling and HTTP requests",
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


def create_strict_security_policy() -> SecurityPolicy:
    """Create a strict security policy for high-security environments.

    Returns:
        SecurityPolicy: A strict security policy with comprehensive protections.

    """
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
            pattern=r"\bgetattr\s*\(",
            description="Dynamic attribute access",
            severity=SecurityIssueSeverity.LOW,
        ),
        SecurityPattern(
            pattern=r"\bsetattr\s*\(",
            description="Dynamic attribute modification",
            severity=SecurityIssueSeverity.MEDIUM,
        ),
        # Cryptographic operations
        SecurityPattern(
            pattern=r"\bhashlib\.(md5|sha1)\s*\(",
            description="Weak cryptographic hash functions",
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
            name="sys",
            description="System-specific parameters and functions",
            severity=SecurityIssueSeverity.MEDIUM,
        ),
        RestrictedModule(
            name="subprocess",
            description="Subprocess management",
            severity=SecurityIssueSeverity.HIGH,
        ),
        RestrictedModule(
            name="multiprocessing",
            description="Process-based parallelism",
            severity=SecurityIssueSeverity.MEDIUM,
        ),
        RestrictedModule(
            name="threading",
            description="Thread-based parallelism",
            severity=SecurityIssueSeverity.LOW,
        ),
        RestrictedModule(
            name="ctypes",
            description="Foreign function library",
            severity=SecurityIssueSeverity.HIGH,
        ),
    ]

    return SecurityPolicy(
        severity_threshold=SecurityIssueSeverity.LOW,  # Very strict - block even low-severity issues
        patterns=patterns,
        restricted_modules=restricted_modules,
    )


def create_permissive_security_policy() -> SecurityPolicy:
    """Create a permissive security policy for development environments.

    Returns:
        SecurityPolicy: A permissive security policy that only blocks high-severity issues.

    """
    patterns = [
        SecurityPattern(
            pattern=r"\brm\s+-rf\s+/",
            description="Dangerous recursive file deletion",
            severity=SecurityIssueSeverity.HIGH,
        ),
        SecurityPattern(
            pattern=r"\bformat\s*\(.*\{.*\}.*\)",
            description="Potential format string vulnerability",
            severity=SecurityIssueSeverity.MEDIUM,
        ),
    ]

    restricted_modules = [
        RestrictedModule(
            name="shutil",
            description="High-level file operations (only blocked for dangerous operations)",
            severity=SecurityIssueSeverity.HIGH,
        ),
    ]

    return SecurityPolicy(
        severity_threshold=SecurityIssueSeverity.HIGH,  # Only block high-severity issues
        patterns=patterns,
        restricted_modules=restricted_modules,
    )


def test_safe_code() -> None:
    """Test safe code that should pass all security policies."""
    logger.info("=== Testing Safe Code ===")

    safe_codes = [
        "print('Hello, World!')",
        "import math\nprint(math.sqrt(16))",
        "data = [1, 2, 3, 4, 5]\nprint(sum(data))",
        "import json\ndata = {'key': 'value'}\nprint(json.dumps(data))",
    ]

    policies = {
        "Basic": create_basic_security_policy(),
        "Strict": create_strict_security_policy(),
        "Permissive": create_permissive_security_policy(),
    }

    for policy_name, policy in policies.items():
        logger.info("\nTesting with %s Policy:", policy_name)

        with SandboxSession(lang="python", security_policy=policy, verbose=False) as session:
            for i, code in enumerate(safe_codes, 1):
                is_safe, violations = session.is_safe(code)
                logger.info("  Safe Code %s: %s", i, "✅ SAFE" if is_safe else "❌ BLOCKED")
                if violations:
                    for violation in violations:
                        logger.info("    - %s (Severity: %s)", violation.description, violation.severity.name)


def test_dangerous_code() -> None:
    """Test dangerous code that should be blocked by security policies."""
    logger.info("\n=== Testing Dangerous Code ===")

    dangerous_codes = [
        "import os\nos.system('rm -rf /')",  # System command execution
        "import subprocess\nsubprocess.run(['ls', '-la'])",  # Subprocess execution
        "eval('print(\"Hello\")')",  # Dynamic evaluation
        "exec('x = 1')",  # Dynamic execution
        "import socket\ns = socket.socket()",  # Network operations
        "open('/etc/passwd', 'w').write('hacked')",  # File writing
        "import os\nos.remove('/important/file')",  # File deletion
        "import ctypes\nctypes.cdll.LoadLibrary('libc.so.6')",  # Foreign function calls
    ]

    policies = {
        "Basic": create_basic_security_policy(),
        "Strict": create_strict_security_policy(),
        "Permissive": create_permissive_security_policy(),
    }

    for policy_name, policy in policies.items():
        logger.info("\nTesting with %s Policy:", policy_name)

        with SandboxSession(lang="python", security_policy=policy, verbose=False) as session:
            for i, code in enumerate(dangerous_codes, 1):
                is_safe, violations = session.is_safe(code)
                logger.info("  Dangerous Code %s: %s", i, "❌ BLOCKED" if not is_safe else "⚠️  ALLOWED")
                if violations:
                    for violation in violations:
                        logger.info("    - %s (Severity: %s)", violation.description, violation.severity.name)


def test_edge_cases() -> None:
    """Test edge cases and complex scenarios."""
    logger.info("\n=== Testing Edge Cases ===")

    edge_cases = [
        # Comments should not trigger security alerts
        "# import os\nprint('This is safe')",
        # String literals should not trigger alerts
        "print('The import os statement is dangerous')",
        # Multi-line imports
        "from urllib.parse import (\n    quote,\n    unquote\n)\nprint(quote('hello world'))",
        # Complex import aliases
        "import os as operating_system\nprint('This should be blocked')",
        # Nested function calls
        "def safe_function():\n    return 'safe'\nprint(safe_function())",
    ]

    policy = create_basic_security_policy()

    with SandboxSession(lang="python", security_policy=policy, verbose=False) as session:
        for i, code in enumerate(edge_cases, 1):
            is_safe, violations = session.is_safe(code)
            logger.info("  Edge Case %s: %s", i, "✅ SAFE" if is_safe else "❌ BLOCKED")
            logger.info("    Code: %s...", code[:50])
            if violations:
                for violation in violations:
                    logger.info("    - %s (Severity: %s)", violation.description, violation.severity.name)


def test_dynamic_policy_modification() -> None:
    """Test dynamic modification of security policies."""
    logger.info("\n=== Testing Dynamic Policy Modification ===")

    # Start with a basic policy
    policy = create_basic_security_policy()

    # Add a new pattern dynamically
    new_pattern = SecurityPattern(
        pattern=r"\bprint\s*\(.*['\"]secret['\"].*\)",
        description="Potential secret disclosure",
        severity=SecurityIssueSeverity.MEDIUM,
    )
    policy.add_pattern(new_pattern)

    # Add a new dangerous module
    new_module = RestrictedModule(
        name="pickle",
        description="Potentially unsafe serialization",
        severity=SecurityIssueSeverity.MEDIUM,
    )
    policy.add_restricted_module(new_module)

    test_codes = [
        "print('This is normal')",
        "print('This contains secret information')",
        "import pickle\ndata = pickle.loads(b'malicious_data')",
    ]

    with SandboxSession(lang="python", security_policy=policy, verbose=False) as session:
        for i, code in enumerate(test_codes, 1):
            is_safe, violations = session.is_safe(code)
            logger.info("  Test Code %s: %s", i, "✅ SAFE" if is_safe else "❌ BLOCKED")
            if violations:
                for violation in violations:
                    logger.info("    - %s (Severity: %s)", violation.description, violation.severity.name)


def test_severity_levels() -> None:
    """Test different severity levels and their effects."""
    logger.info("\n=== Testing Severity Levels ===")

    test_code = "import socket\neval('print(1)')\nos.system('echo test')"

    # Test with different safety levels
    severity_thresholds = [
        SecurityIssueSeverity.SAFE,
        SecurityIssueSeverity.LOW,
        SecurityIssueSeverity.MEDIUM,
        SecurityIssueSeverity.HIGH,
    ]

    base_policy = create_basic_security_policy()

    for level in severity_thresholds:
        logger.info("\nTesting with safety level: %s", level.name)
        policy = SecurityPolicy(
            severity_threshold=level,
            patterns=base_policy.patterns,
            restricted_modules=base_policy.restricted_modules,
        )

        with SandboxSession(lang="python", security_policy=policy, verbose=False) as session:
            is_safe, violations = session.is_safe(test_code)
            logger.info("  Result: %s", "✅ SAFE" if is_safe else "❌ BLOCKED")
            logger.info("  Violations found: %s", len(violations))
            for violation in violations:
                logger.info("    - %s (Severity: %s)", violation.description, violation.severity.name)


def test_real_world_scenarios() -> None:
    """Test real-world scenarios that might be encountered."""
    logger.info("\n=== Testing Real-World Scenarios ===")

    scenarios = {
        "Data Science - Safe": """
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

data = pd.DataFrame({'x': [1, 2, 3], 'y': [4, 5, 6]})
print(data.describe())
""",
        "Web Scraping - Suspicious": """
import requests
import os

url = 'https://example.com'
response = requests.get(url)
os.environ['SECRET_KEY'] = response.text
""",
        "File Processing - Dangerous": """
import os
import shutil

for root, dirs, files in os.walk('/home'):
    for file in files:
        if file.endswith('.txt'):
            os.remove(os.path.join(root, file))
""",
        "Machine Learning - Safe": """
import sklearn
from sklearn.linear_model import LinearRegression
import joblib

model = LinearRegression()
# Training code would go here
joblib.dump(model, 'model.pkl')
""",
    }

    policy = create_basic_security_policy()

    with SandboxSession(lang="python", security_policy=policy, verbose=False) as session:
        for scenario_name, code in scenarios.items():
            is_safe, violations = session.is_safe(code)
            logger.info("\n  Scenario: %s", scenario_name)
            logger.info("  Result: %s", "✅ SAFE" if is_safe else "❌ BLOCKED")
            if violations:
                logger.info("  Violations (%s):", len(violations))
                for violation in violations:
                    logger.info("    - %s (Severity: %s)", violation.description, violation.severity.name)


if __name__ == "__main__":
    logger.info("LLM Sandbox Security Policy Examples")
    logger.info("====================================")

    try:
        test_safe_code()
        test_dangerous_code()
        test_edge_cases()
        test_dynamic_policy_modification()
        test_severity_levels()
        test_real_world_scenarios()

        logger.info("\n🎉 All security policy tests completed successfully!")

    except Exception:
        logger.exception("Error during testing")
        raise
