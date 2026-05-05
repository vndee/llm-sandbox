"""Security helpers for input validation and sandbox-internal hardening.

This package collects two related concerns:

* :mod:`llm_sandbox.security.policy` — code-scanning policy types
  (``SecurityPolicy``, ``SecurityPattern``, ``RestrictedModule``,
  ``SecurityIssueSeverity``).
* :mod:`llm_sandbox.security.package_validators` — per-language allowlists
  for the ``libraries`` argument so user-supplied package identifiers
  cannot inject shell metacharacters into install commands. See
  https://github.com/vndee/llm-sandbox/issues/162.
"""

from llm_sandbox.security.package_validators import (
    PACKAGE_NAME_PATTERNS,
    validate_package_name,
)
from llm_sandbox.security.policy import (
    RestrictedModule,
    SecurityIssueSeverity,
    SecurityPattern,
    SecurityPolicy,
)

__all__ = [
    "PACKAGE_NAME_PATTERNS",
    "RestrictedModule",
    "SecurityIssueSeverity",
    "SecurityPattern",
    "SecurityPolicy",
    "validate_package_name",
]
