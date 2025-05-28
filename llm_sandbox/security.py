import re
from enum import IntEnum

from pydantic import BaseModel, Field, field_validator

from llm_sandbox.exceptions import InvalidRegexPatternError


class SecurityIssueSeverity(IntEnum):
    """Severity of a security issue."""

    SAFE = 0
    LOW = 1
    MEDIUM = 2
    HIGH = 3


class SecurityPattern(BaseModel):
    """A security pattern."""

    pattern: str = Field(..., description="The pattern that caused the security issue.")
    description: str = Field(..., description="The description of the security issue.")
    severity: SecurityIssueSeverity = Field(..., description="The severity of the security issue.")

    @field_validator("pattern")
    @classmethod
    def validate_pattern(cls, v: str) -> str:
        """Validate that the pattern is a valid regex pattern."""
        try:
            re.compile(v)
        except re.error as e:
            raise InvalidRegexPatternError(v) from e
        return v


class RestrictedModule(BaseModel):
    """A dangerous module."""

    name: str = Field(..., description="The name of the module.")
    description: str = Field(..., description="The description of the module.")
    severity: SecurityIssueSeverity = Field(..., description="The severity of the module.")


class SecurityPolicy(BaseModel):
    """A security policy."""

    severity_threshold: SecurityIssueSeverity = Field(
        default=SecurityIssueSeverity.SAFE,
        description="The minimum severity level at which security issues will be blocked.",
    )
    patterns: list[SecurityPattern] | None = Field(default=None, description="The security patterns in the code.")
    restricted_modules: list[RestrictedModule] | None = Field(
        default=None, description="The modules that are restricted based on their severity level."
    )

    def add_pattern(self, pattern: SecurityPattern) -> None:
        """Add a security pattern to the policy."""
        if self.patterns is None:
            self.patterns = []
        self.patterns.append(pattern)

    def add_restricted_module(self, module: RestrictedModule) -> None:
        """Add a restricted module to the policy."""
        if self.restricted_modules is None:
            self.restricted_modules = []
        self.restricted_modules.append(module)
