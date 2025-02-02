"""Security scanning functionality for LLM Sandbox."""

import re
from typing import List, Pattern, Dict
from dataclasses import dataclass
from .exceptions import SecurityError


@dataclass
class SecurityIssue:
    """Represents a security issue found in code."""

    pattern: str
    description: str
    severity: str  # 'high', 'medium', 'low'
    line_number: int


class SecurityScanner:
    """Scanner for detecting potential security issues in code."""

    def __init__(self):
        self.patterns: Dict[str, Pattern] = {  # type: ignore[annotation-unchecked]
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

    def scan_code(self, code: str, strict: bool = True) -> List[SecurityIssue]:
        """
        Scan code for security issues.

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
