"""Security scanning functionality for LLM Sandbox."""

from dataclasses import dataclass
from enum import StrEnum


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
