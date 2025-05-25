"""Security scanning functionality for LLM Sandbox."""

import re
from dataclasses import dataclass
from enum import StrEnum
from re import Pattern

from .exceptions import SecurityError


class SecurityIssueSeverity(StrEnum):
    """Severity of a security issue."""

    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


@dataclass
class SecurityIssue:
    """Represents a security issue found in code."""

    pattern: str
    description: str
    severity: SecurityIssueSeverity
    line_number: int


class SecurityRuleset:
    """Ruleset for security scanning.

    This can be load from the default ruleset or a custom ruleset
    which is defined by the user.
    """

    def __init__(self, file_path: str | None = None) -> None:
        """Initialize the security ruleset."""
        if not file_path:
            self._set_default_ruleset()
        else:
            self._load_ruleset(file_path)

    def _set_default_ruleset(self) -> None:
        """Set the default ruleset."""

    def _load_ruleset(self, file_path: str) -> None:
        """Load the ruleset from a file."""


class SecurityScanner:
    """Scanner for detecting potential security issues in code."""

    def __init__(self) -> None:
        """Initialize the security scanner."""
        self.patterns: dict[str, Pattern] = {  # type: ignore[annotation-unchecked]
            "system_calls": re.compile(r"os\.system|subprocess\."),
            "code_execution": re.compile(r"eval\(|exec\(|compile\("),
            "file_operations": re.compile(r"open\(|file\(|read|write"),
            "network_access": re.compile(r"socket\.|urllib|requests\."),
            "shell_injection": re.compile(r"shell=True|commands\."),
            "dangerous_imports": re.compile(r"import\s+(os|subprocess|sys|shutil)"),
        }

        self.severity_map = {
            "system_calls": "high",
            "code_execution": "high",
            "file_operations": "medium",
            "network_access": "medium",
            "shell_injection": "high",
            "dangerous_imports": "medium",
        }

        self.descriptions = {
            "system_calls": "Direct system command execution detected",
            "code_execution": "Dynamic code execution functionality detected",
            "file_operations": "File system operations detected",
            "network_access": "Network access functionality detected",
            "shell_injection": "Potential shell injection vulnerability",
            "dangerous_imports": "Import of potentially dangerous modules",
        }

    def scan_code(self, code: str, strict: bool = True) -> list[SecurityIssue]:
        """Scan code for security issues.

        Args:
            code: The code to scan
            strict: If True, raise SecurityError for high severity issues

        Returns:
            List of SecurityIssue objects

        Raises:
            SecurityError: If strict=True and high severity issues are found

        """
        issues = []
        lines = code.split("\n")

        for line_num, line in enumerate(lines, 1):
            for pattern_name, pattern in self.patterns.items():
                if pattern.search(line):
                    issue = SecurityIssue(
                        pattern=pattern_name,
                        description=self.descriptions[pattern_name],
                        severity=self.severity_map[pattern_name],
                        line_number=line_num,
                    )
                    issues.append(issue)

        if strict and any(issue.severity == "high" for issue in issues):
            high_severity_issues = [i for i in issues if i.severity == "high"]
            raise SecurityError(
                "High severity security issues found:\n"
                + "\n".join(
                    f"Line {i.line_number}: {i.description}"
                    for i in high_severity_issues
                )
            )

        return issues

    def is_safe(self, code: str) -> bool:
        """Quick check if code is safe (no security issues)."""
        try:
            return len(self.scan_code(code, strict=True)) == 0
        except SecurityError:
            return False
