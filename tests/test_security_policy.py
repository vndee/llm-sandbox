# ruff: noqa: PLR2004

"""Unit tests for the security features in LLM Sandbox.

This module contains comprehensive unit tests for the security policy system,
including pattern matching, dangerous module detection, and policy enforcement.
"""

import re

import pytest

from llm_sandbox.exceptions import InvalidRegexPatternError
from llm_sandbox.language_handlers.python_handler import PythonHandler
from llm_sandbox.security import DangerousModule, SecurityIssueSeverity, SecurityPattern, SecurityPolicy


class TestSecurityIssueSeverity:
    """Test cases for SecurityIssueSeverity enum."""

    def test_severity_levels(self) -> None:
        """Test that severity levels have correct values."""
        assert SecurityIssueSeverity.SAFE == 0
        assert SecurityIssueSeverity.LOW == 1
        assert SecurityIssueSeverity.MEDIUM == 2
        assert SecurityIssueSeverity.HIGH == 3

    def test_severity_comparison(self) -> None:
        """Test severity level comparisons."""
        assert SecurityIssueSeverity.SAFE < SecurityIssueSeverity.LOW
        assert SecurityIssueSeverity.LOW < SecurityIssueSeverity.MEDIUM
        assert SecurityIssueSeverity.MEDIUM < SecurityIssueSeverity.HIGH
        assert SecurityIssueSeverity.HIGH >= SecurityIssueSeverity.MEDIUM


class TestSecurityPattern:
    """Test cases for SecurityPattern class."""

    def test_valid_pattern_creation(self) -> None:
        """Test creating a security pattern with valid regex."""
        pattern = SecurityPattern(
            pattern=r"\bos\.system\s*\(",
            description="System command execution",
            severity=SecurityIssueSeverity.HIGH,
        )

        assert pattern.pattern == r"\bos\.system\s*\("
        assert pattern.description == "System command execution"
        assert pattern.severity == SecurityIssueSeverity.HIGH

    def test_invalid_regex_pattern(self) -> None:
        """Test that invalid regex patterns raise an error."""
        with pytest.raises(InvalidRegexPatternError):
            SecurityPattern(
                pattern="[invalid regex",  # Missing closing bracket
                description="Invalid pattern",
                severity=SecurityIssueSeverity.MEDIUM,
            )

    def test_pattern_validation(self) -> None:
        """Test that pattern validation works correctly."""
        # Valid patterns should not raise errors
        valid_patterns = [
            r"\bimport\s+os\b",
            r"\beval\s*\(",
            r"\bsubprocess\.(run|call)\s*\(",
            r"(?:import|from)\s+requests\b",
        ]

        for pattern_str in valid_patterns:
            pattern = SecurityPattern(
                pattern=pattern_str,
                description="Test pattern",
                severity=SecurityIssueSeverity.LOW,
            )
            assert pattern.pattern == pattern_str

    def test_complex_regex_patterns(self) -> None:
        """Test complex regex patterns."""
        complex_pattern = SecurityPattern(
            pattern=r"(?:^|\s)(?:import\s+os(?:\s+as\s+\w+)?|from\s+os\s+import\s+(?:\*|\w+(?:\s+as\s+\w+)?(?:,\s*\w+(?:\s+as\s+\w+)?)*))(?=[\s;(#]|$)",
            description="Complex OS import pattern",
            severity=SecurityIssueSeverity.HIGH,
        )

        # Test that the pattern can be compiled
        compiled_pattern = re.compile(complex_pattern.pattern)
        assert compiled_pattern is not None


class TestDangerousModule:
    """Test cases for DangerousModule class."""

    def test_dangerous_module_creation(self) -> None:
        """Test creating a dangerous module."""
        module = DangerousModule(
            name="os",
            description="Operating system interface",
            severity=SecurityIssueSeverity.HIGH,
        )

        assert module.name == "os"
        assert module.description == "Operating system interface"
        assert module.severity == SecurityIssueSeverity.HIGH

    def test_module_with_different_severities(self) -> None:
        """Test modules with different severity levels."""
        modules = [
            DangerousModule(name="requests", description="HTTP library", severity=SecurityIssueSeverity.LOW),
            DangerousModule(name="socket", description="Network operations", severity=SecurityIssueSeverity.MEDIUM),
            DangerousModule(name="subprocess", description="Process execution", severity=SecurityIssueSeverity.HIGH),
        ]

        assert modules[0].severity == SecurityIssueSeverity.LOW
        assert modules[1].severity == SecurityIssueSeverity.MEDIUM
        assert modules[2].severity == SecurityIssueSeverity.HIGH


class TestSecurityPolicy:
    """Test cases for SecurityPolicy class."""

    def test_empty_policy_creation(self) -> None:
        """Test creating an empty security policy."""
        policy = SecurityPolicy(
            safety_level=SecurityIssueSeverity.MEDIUM,
            patterns=[],
            dangerous_modules=[],
        )

        assert policy.safety_level == SecurityIssueSeverity.MEDIUM
        assert len(policy.patterns) == 0
        assert len(policy.dangerous_modules) == 0

    def test_policy_with_patterns_and_modules(self) -> None:
        """Test creating a policy with patterns and modules."""
        patterns = [
            SecurityPattern(
                pattern=r"\beval\s*\(",
                description="Dynamic evaluation",
                severity=SecurityIssueSeverity.MEDIUM,
            )
        ]

        modules = [
            DangerousModule(
                name="os",
                description="OS interface",
                severity=SecurityIssueSeverity.HIGH,
            )
        ]

        policy = SecurityPolicy(
            safety_level=SecurityIssueSeverity.LOW,
            patterns=patterns,
            dangerous_modules=modules,
        )

        assert policy.safety_level == SecurityIssueSeverity.LOW
        assert len(policy.patterns) == 1
        assert len(policy.dangerous_modules) == 1
        assert policy.patterns[0].description == "Dynamic evaluation"
        assert policy.dangerous_modules[0].name == "os"

    def test_add_pattern_to_policy(self) -> None:
        """Test adding patterns to a policy dynamically."""
        policy = SecurityPolicy(
            safety_level=SecurityIssueSeverity.MEDIUM,
            patterns=[],
            dangerous_modules=[],
        )

        new_pattern = SecurityPattern(
            pattern=r"\bexec\s*\(",
            description="Dynamic execution",
            severity=SecurityIssueSeverity.HIGH,
        )

        policy.add_pattern(new_pattern)

        assert len(policy.patterns) == 1
        assert policy.patterns[0].pattern == r"\bexec\s*\("

    def test_add_dangerous_module_to_policy(self) -> None:
        """Test adding dangerous modules to a policy dynamically."""
        policy = SecurityPolicy(
            safety_level=SecurityIssueSeverity.MEDIUM,
            patterns=[],
            dangerous_modules=[],
        )

        new_module = DangerousModule(
            name="subprocess",
            description="Process management",
            severity=SecurityIssueSeverity.HIGH,
        )

        policy.add_dangerous_module(new_module)

        assert len(policy.dangerous_modules) == 1
        assert policy.dangerous_modules[0].name == "subprocess"

    def test_default_safety_level(self) -> None:
        """Test that default safety level is SAFE."""
        policy = SecurityPolicy(
            patterns=[],
            dangerous_modules=[],
        )

        assert policy.safety_level == SecurityIssueSeverity.SAFE


class TestSecurityPolicyIntegration:
    """Integration tests for security policy components."""

    @pytest.fixture
    def sample_policy(self) -> SecurityPolicy:
        """Create a sample security policy for testing."""
        patterns = [
            SecurityPattern(
                pattern=r"\bos\.system\s*\(",
                description="System command execution",
                severity=SecurityIssueSeverity.HIGH,
            ),
            SecurityPattern(
                pattern=r"\beval\s*\(",
                description="Dynamic evaluation",
                severity=SecurityIssueSeverity.MEDIUM,
            ),
            SecurityPattern(
                pattern=r"\b__import__\s*\(",
                description="Dynamic import",
                severity=SecurityIssueSeverity.LOW,
            ),
        ]

        modules = [
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
                name="requests",
                description="HTTP requests",
                severity=SecurityIssueSeverity.LOW,
            ),
        ]

        return SecurityPolicy(
            safety_level=SecurityIssueSeverity.MEDIUM,
            patterns=patterns,
            dangerous_modules=modules,
        )

    def test_policy_serialization_compatibility(self, sample_policy: SecurityPolicy) -> None:
        """Test that policies can be serialized and deserialized."""
        # Test that the policy can be converted to dict (for JSON serialization)
        policy_dict = sample_policy.model_dump()

        assert "safety_level" in policy_dict
        assert "patterns" in policy_dict
        assert "dangerous_modules" in policy_dict
        assert len(policy_dict["patterns"]) == 3
        assert len(policy_dict["dangerous_modules"]) == 3

        # Test reconstruction from dict
        reconstructed_policy = SecurityPolicy.model_validate(policy_dict)

        assert reconstructed_policy.safety_level == sample_policy.safety_level
        assert len(reconstructed_policy.patterns) == len(sample_policy.patterns)
        assert len(reconstructed_policy.dangerous_modules) == len(sample_policy.dangerous_modules)

    def test_policy_modification_chain(self, sample_policy: SecurityPolicy) -> None:
        """Test chaining policy modifications."""
        original_pattern_count = len(sample_policy.patterns)
        original_module_count = len(sample_policy.dangerous_modules)

        # Add new pattern
        new_pattern = SecurityPattern(
            pattern=r"\bexec\s*\(",
            description="Execute function",
            severity=SecurityIssueSeverity.MEDIUM,
        )
        sample_policy.add_pattern(new_pattern)

        # Add new module
        new_module = DangerousModule(
            name="socket",
            description="Network sockets",
            severity=SecurityIssueSeverity.MEDIUM,
        )
        sample_policy.add_dangerous_module(new_module)

        assert len(sample_policy.patterns) == original_pattern_count + 1
        assert len(sample_policy.dangerous_modules) == original_module_count + 1

        # Verify the additions
        assert any(p.pattern == r"\bexec\s*\(" for p in sample_policy.patterns)
        assert any(m.name == "socket" for m in sample_policy.dangerous_modules)

    def test_severity_filtering(self, sample_policy: SecurityPolicy) -> None:
        """Test filtering patterns by severity level."""
        high_severity_patterns = [p for p in sample_policy.patterns if p.severity >= SecurityIssueSeverity.HIGH]

        medium_and_above_patterns = [p for p in sample_policy.patterns if p.severity >= SecurityIssueSeverity.MEDIUM]

        all_patterns = sample_policy.patterns

        assert len(high_severity_patterns) == 1  # Only os.system
        assert len(medium_and_above_patterns) == 2  # os.system and eval
        assert len(all_patterns) == 3  # All patterns

    def test_module_severity_filtering(self, sample_policy: SecurityPolicy) -> None:
        """Test filtering modules by severity level."""
        high_severity_modules = [m for m in sample_policy.dangerous_modules if m.severity >= SecurityIssueSeverity.HIGH]

        assert len(high_severity_modules) == 2  # os and subprocess
        assert all(m.name in ["os", "subprocess"] for m in high_severity_modules)


class TestPatternMatching:
    """Test regex pattern matching functionality."""

    @pytest.fixture
    def python_handler(self) -> PythonHandler:
        """Create a Python handler for testing."""
        return PythonHandler()

    def test_os_system_pattern_matching(self, python_handler: PythonHandler) -> None:
        """Test os.system pattern matching."""
        pattern = SecurityPattern(
            pattern=r"\bos\.system\s*\(",
            description="System command execution",
            severity=SecurityIssueSeverity.HIGH,
        )

        # Test cases that should match
        positive_cases = [
            "import os\nos.system('ls')",
            "os.system ( 'echo hello' )",
            "result = os.system('pwd')",
            "    os.system('whoami')",  # With indentation
        ]

        # Test cases that should not match
        negative_cases = [
            "# os.system('commented')",
            "print('os.system is dangerous')",
            "osystem('not a match')",
            "my_os.system('different object')",
        ]

        compiled_pattern = re.compile(pattern.pattern)

        for case in positive_cases:
            filtered_code = python_handler.filter_comments(case)
            assert compiled_pattern.search(filtered_code) is not None, f"Should match: {case}"

        for case in negative_cases:
            filtered_code = python_handler.filter_comments(case)
            assert compiled_pattern.search(filtered_code) is None, f"Should not match: {case}"

    def test_import_pattern_matching(self, python_handler: PythonHandler) -> None:
        """Test import statement pattern matching."""
        # This would be generated by the language handler
        import_pattern = (
            r"(?:^|\s)(?:import\s+os(?:\s+as\s+\w+)?|"
            r"from\s+os\s+import\s+(?:\*|\w+(?:\s+as\s+\w+)?(?:,\s*\w+(?:\s+as\s+\w+)?)*))(?=[\s;(#]|$)"
        )

        pattern = SecurityPattern(
            pattern=import_pattern,
            description="OS module import",
            severity=SecurityIssueSeverity.HIGH,
        )

        # Test cases that should match
        positive_cases = [
            "import os",
            "import os as operating_system",
            "from os import system",
            "from os import system, environ",
            "from os import system as sys_call",
            "    import os  # with indentation",
        ]

        # Test cases that should not match
        negative_cases = [
            "# import os",
            "print('import os')",
            "importos",  # No space
            "import oss",  # Different module
            "from oss import something",
        ]

        compiled_pattern = re.compile(pattern.pattern)

        for case in positive_cases:
            filtered_code = python_handler.filter_comments(case)
            assert compiled_pattern.search(filtered_code) is not None, f"Should match: {case}"

        for case in negative_cases:
            filtered_code = python_handler.filter_comments(case)
            assert compiled_pattern.search(filtered_code) is None, f"Should not match: {case}"
