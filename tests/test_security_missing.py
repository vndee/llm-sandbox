# ruff: noqa: PLR2004, ARG002, SLF001
"""Additional tests for llm_sandbox.security module to improve coverage."""

import re

import pytest

from llm_sandbox.exceptions import InvalidRegexPatternError
from llm_sandbox.security import RestrictedModule, SecurityIssueSeverity, SecurityPattern, SecurityPolicy


class TestSecurityPatternEdgeCases:
    """Test SecurityPattern with edge cases and error conditions."""

    def test_security_pattern_with_invalid_regex(self) -> None:
        """Test SecurityPattern with invalid regex pattern."""
        with pytest.raises(InvalidRegexPatternError):
            SecurityPattern(
                pattern="[invalid regex",  # Missing closing bracket
                description="Invalid regex pattern",
                severity=SecurityIssueSeverity.HIGH,
            )

    def test_security_pattern_with_complex_regex(self) -> None:
        """Test SecurityPattern with complex but valid regex patterns."""
        complex_patterns = [
            r"^(?=.*[a-z])(?=.*[A-Z])(?=.*\d)[a-zA-Z\d]{8,}$",  # Password pattern
            r"\b(?:\d{1,3}\.){3}\d{1,3}\b",  # IP address pattern
            r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}",  # Email pattern
            r"(?:file|https?|ftp)://[^\s/$.?#].[^\s]*",  # URL pattern
        ]

        for pattern in complex_patterns:
            security_pattern = SecurityPattern(
                pattern=pattern, description=f"Security pattern for {pattern}", severity=SecurityIssueSeverity.MEDIUM
            )
            assert security_pattern.pattern == pattern
            # Verify it's a valid regex by compiling it
            re.compile(security_pattern.pattern)

    def test_security_pattern_with_various_severities(self) -> None:
        """Test SecurityPattern with all severity levels."""
        severities = [
            SecurityIssueSeverity.SAFE,
            SecurityIssueSeverity.LOW,
            SecurityIssueSeverity.MEDIUM,
            SecurityIssueSeverity.HIGH,
        ]

        for severity in severities:
            pattern = SecurityPattern(
                pattern=r"test_pattern", description=f"Test pattern with {severity.name} severity", severity=severity
            )
            assert pattern.severity == severity

    def test_security_pattern_validation_with_regex_error(self) -> None:
        """Test that regex compilation errors are properly caught and converted."""
        invalid_patterns = [
            "[unclosed bracket",
            "*invalid quantifier",
            "(?P<invalid group name",
            "(?P<duplicate>test)(?P<duplicate>test)",
        ]

        for invalid_pattern in invalid_patterns:
            with pytest.raises(InvalidRegexPatternError) as exc_info:
                SecurityPattern(
                    pattern=invalid_pattern, description="Test invalid pattern", severity=SecurityIssueSeverity.HIGH
                )

            # Check that the original regex error is preserved
            assert invalid_pattern in str(exc_info.value)


class TestSecurityPolicyNoneListHandling:
    """Test SecurityPolicy methods when lists are None (covering missing lines)."""

    def test_add_pattern_when_patterns_is_none(self) -> None:
        """Test add_pattern method when patterns list is None."""
        # Create policy with no patterns (None)
        policy = SecurityPolicy(severity_threshold=SecurityIssueSeverity.MEDIUM)
        assert policy.patterns is None

        # Add a pattern when patterns is None (this should cover line 59)
        pattern = SecurityPattern(
            pattern=r"dangerous_function\(", description="Dangerous function call", severity=SecurityIssueSeverity.HIGH
        )
        policy.add_pattern(pattern)

        # Verify that patterns list was created and pattern was added
        assert policy.patterns is not None
        assert len(policy.patterns) == 1
        assert policy.patterns[0] == pattern

    def test_add_pattern_when_patterns_exists(self) -> None:
        """Test add_pattern method when patterns list already exists."""
        # Create policy with existing patterns
        existing_pattern = SecurityPattern(
            pattern=r"existing_pattern", description="Existing pattern", severity=SecurityIssueSeverity.LOW
        )
        policy = SecurityPolicy(severity_threshold=SecurityIssueSeverity.MEDIUM, patterns=[existing_pattern])
        assert policy.patterns is not None
        assert len(policy.patterns) == 1

        # Add another pattern
        new_pattern = SecurityPattern(
            pattern=r"new_pattern", description="New pattern", severity=SecurityIssueSeverity.HIGH
        )
        policy.add_pattern(new_pattern)

        # Verify both patterns exist
        assert policy.patterns is not None
        assert len(policy.patterns) == 2
        assert existing_pattern in policy.patterns
        assert new_pattern in policy.patterns

    def test_add_restricted_module_when_modules_is_none(self) -> None:
        """Test add_restricted_module method when restricted_modules list is None."""
        # Create policy with no restricted modules (None)
        policy = SecurityPolicy(severity_threshold=SecurityIssueSeverity.MEDIUM)
        assert policy.restricted_modules is None

        # Add a module when restricted_modules is None (this should cover line 65)
        module = RestrictedModule(
            name="os", description="Operating system interface", severity=SecurityIssueSeverity.HIGH
        )
        policy.add_restricted_module(module)

        # Verify that restricted_modules list was created and module was added
        assert policy.restricted_modules is not None
        assert len(policy.restricted_modules) == 1
        assert policy.restricted_modules[0] == module

    def test_add_restricted_module_when_modules_exists(self) -> None:
        """Test add_restricted_module method when restricted_modules list already exists."""
        # Create policy with existing modules
        existing_module = RestrictedModule(
            name="subprocess", description="Subprocess execution", severity=SecurityIssueSeverity.HIGH
        )
        policy = SecurityPolicy(severity_threshold=SecurityIssueSeverity.MEDIUM, restricted_modules=[existing_module])
        assert policy.restricted_modules is not None
        assert len(policy.restricted_modules) == 1

        # Add another module
        new_module = RestrictedModule(
            name="socket", description="Network socket interface", severity=SecurityIssueSeverity.MEDIUM
        )
        policy.add_restricted_module(new_module)

        # Verify both modules exist
        assert policy.restricted_modules is not None
        assert len(policy.restricted_modules) == 2
        assert existing_module in policy.restricted_modules
        assert new_module in policy.restricted_modules

    def test_multiple_add_operations_on_none_lists(self) -> None:
        """Test multiple add operations starting from None lists."""
        policy = SecurityPolicy(severity_threshold=SecurityIssueSeverity.LOW)
        assert policy.patterns is None
        assert policy.restricted_modules is None

        # Add multiple patterns
        patterns = [
            SecurityPattern(pattern=r"pattern1", description="Pattern 1", severity=SecurityIssueSeverity.LOW),
            SecurityPattern(pattern=r"pattern2", description="Pattern 2", severity=SecurityIssueSeverity.MEDIUM),
            SecurityPattern(pattern=r"pattern3", description="Pattern 3", severity=SecurityIssueSeverity.HIGH),
        ]

        for pattern in patterns:
            policy.add_pattern(pattern)

        assert policy.patterns is not None
        assert len(policy.patterns) == 3

        # Add multiple modules
        modules = [
            RestrictedModule(name="module1", description="Module 1", severity=SecurityIssueSeverity.LOW),
            RestrictedModule(name="module2", description="Module 2", severity=SecurityIssueSeverity.MEDIUM),
        ]

        for module in modules:
            policy.add_restricted_module(module)

        assert policy.restricted_modules is not None
        assert len(policy.restricted_modules) == 2


class TestSecurityPolicyComplex:
    """Test complex SecurityPolicy scenarios."""

    def test_security_policy_with_all_none_defaults(self) -> None:
        """Test SecurityPolicy with all default None values."""
        policy = SecurityPolicy()

        assert policy.severity_threshold == SecurityIssueSeverity.SAFE
        assert policy.patterns is None
        assert policy.restricted_modules is None

        # Test that we can still add items to None lists
        pattern = SecurityPattern(pattern=r"test", description="Test pattern", severity=SecurityIssueSeverity.LOW)
        policy.add_pattern(pattern)
        assert policy.patterns == [pattern]

        module = RestrictedModule(name="test_module", description="Test module", severity=SecurityIssueSeverity.LOW)
        policy.add_restricted_module(module)
        assert policy.restricted_modules == [module]

    def test_security_policy_immutability_after_pydantic_validation(self) -> None:
        """Test that SecurityPolicy maintains proper state after Pydantic operations."""
        # Create policy and verify it can be serialized/deserialized
        original_policy = SecurityPolicy(
            severity_threshold=SecurityIssueSeverity.HIGH,
            patterns=[
                SecurityPattern(
                    pattern=r"dangerous_call\(",
                    description="Dangerous function call",
                    severity=SecurityIssueSeverity.HIGH,
                )
            ],
            restricted_modules=[
                RestrictedModule(name="os", description="OS module", severity=SecurityIssueSeverity.HIGH)
            ],
        )

        # Convert to dict and back
        policy_dict = original_policy.model_dump()
        reconstructed_policy = SecurityPolicy.model_validate(policy_dict)

        # Verify reconstruction worked
        assert reconstructed_policy.severity_threshold == original_policy.severity_threshold
        assert reconstructed_policy.patterns is not None
        assert original_policy.patterns is not None
        assert len(reconstructed_policy.patterns) == len(original_policy.patterns)
        assert reconstructed_policy.restricted_modules is not None
        assert original_policy.restricted_modules is not None
        assert len(reconstructed_policy.restricted_modules) == len(original_policy.restricted_modules)

        # Test adding to reconstructed policy
        new_pattern = SecurityPattern(
            pattern=r"new_pattern", description="New pattern", severity=SecurityIssueSeverity.MEDIUM
        )
        reconstructed_policy.add_pattern(new_pattern)
        assert reconstructed_policy.patterns is not None
        assert len(reconstructed_policy.patterns) == 2


class TestSecurityModelsValidation:
    """Test validation edge cases for security models."""

    def test_restricted_module_all_severities(self) -> None:
        """Test RestrictedModule with all severity levels."""
        severities = list(SecurityIssueSeverity)

        for severity in severities:
            module = RestrictedModule(
                name=f"module_{severity.name.lower()}",
                description=f"Module with {severity.name} severity",
                severity=severity,
            )
            assert module.severity == severity
            assert module.name == f"module_{severity.name.lower()}"

    def test_security_pattern_empty_description(self) -> None:
        """Test SecurityPattern with empty description."""
        pattern = SecurityPattern(
            pattern=r"test",
            description="",  # Empty description should be allowed
            severity=SecurityIssueSeverity.LOW,
        )
        assert pattern.description == ""
        assert pattern.pattern == "test"

    def test_restricted_module_special_characters_in_name(self) -> None:
        """Test RestrictedModule with special characters in module name."""
        special_names = [
            "os.path",
            "xml.etree.ElementTree",
            "__builtin__",
            "_private_module",
            "module-with-dashes",
        ]

        for name in special_names:
            module = RestrictedModule(name=name, description=f"Module {name}", severity=SecurityIssueSeverity.MEDIUM)
            assert module.name == name

    def test_security_policy_with_mixed_severities(self) -> None:
        """Test SecurityPolicy with patterns and modules of different severities."""
        policy = SecurityPolicy(severity_threshold=SecurityIssueSeverity.MEDIUM)

        # Add patterns with different severities
        severities = [SecurityIssueSeverity.LOW, SecurityIssueSeverity.MEDIUM, SecurityIssueSeverity.HIGH]

        for i, severity in enumerate(severities):
            pattern = SecurityPattern(pattern=f"pattern_{i}", description=f"Pattern {i}", severity=severity)
            policy.add_pattern(pattern)

            module = RestrictedModule(name=f"module_{i}", description=f"Module {i}", severity=severity)
            policy.add_restricted_module(module)

        assert policy.patterns is not None
        assert policy.restricted_modules is not None
        assert len(policy.patterns) == 3
        assert len(policy.restricted_modules) == 3

        # Verify all severities are represented
        pattern_severities = {p.severity for p in policy.patterns}
        module_severities = {m.severity for m in policy.restricted_modules}

        assert pattern_severities == set(severities)
        assert module_severities == set(severities)
