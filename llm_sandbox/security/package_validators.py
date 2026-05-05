"""Per-language validators for the ``libraries`` argument of session.run.

Language handlers build install commands by interpolating user-supplied
package identifiers into shell command strings (``f"pip install {lib}"``,
``f"apt-get install {lib}"``, etc). The container boundary is the *last*
defence — a malicious value like ``foo; rm -rf /`` would still execute
inside the sandbox before any container limits could short-circuit it.

This module provides language-specific allowlist regexes that reject any
identifier outside the canonical package-naming spec for each ecosystem
*before* the value reaches the shell. The patterns deliberately err on
the side of strict — false rejection is a one-line workaround for the
caller, false acceptance is an injection.

See https://github.com/vndee/llm-sandbox/issues/162.
"""

from __future__ import annotations

import re
from typing import Final

from llm_sandbox.exceptions import ValidationError


def _compile(pattern: str) -> re.Pattern[str]:
    """Anchor a pattern to the full string to prevent partial matches."""
    if not pattern.startswith("^"):
        pattern = "^" + pattern
    if not pattern.endswith("$"):
        pattern = pattern + "$"
    return re.compile(pattern)


# Language-specific allowlists. Each pattern matches the canonical form of a
# package/library identifier in that ecosystem, plus the common shapes the
# associated install tool accepts. Anything outside these patterns is
# treated as a potential injection attempt and rejected. Per-language
# format references are documented inline alongside each entry below.
PACKAGE_NAME_PATTERNS: Final[dict[str, re.Pattern[str]]] = {
    # PEP 508 conservative subset — covers `name`, `name[extra,extra2]`, and
    # name + a single version specifier. Multiple specifiers separated by
    # commas (e.g. `requests>=2.0,<3.0`) are also accepted.
    "python": _compile(
        r"[a-zA-Z0-9][a-zA-Z0-9._-]*"
        r"(?:\[[a-zA-Z0-9_,\s.-]+\])?"
        r"(?:(?:===|==|!=|~=|>=|<=|>|<)[0-9][a-zA-Z0-9._+!*-]*"
        r"(?:,(?:===|==|!=|~=|>=|<=|>|<)[0-9][a-zA-Z0-9._+!*-]*)*)?"
    ),
    "javascript": _compile(
        r"(?:@[a-z0-9][a-z0-9._-]*\/)?[a-z0-9][a-z0-9._-]*"
        r"(?:@[a-zA-Z0-9._-]+)?"
    ),
    # Same shape as JavaScript — node and bun share npm package semantics.
    "node": _compile(
        r"(?:@[a-z0-9][a-z0-9._-]*\/)?[a-z0-9][a-z0-9._-]*"
        r"(?:@[a-zA-Z0-9._-]+)?"
    ),
    "go": _compile(
        r"[a-zA-Z0-9][a-zA-Z0-9._/-]*"
        r"(?:@[a-zA-Z0-9._+-]+)?"
    ),
    "ruby": _compile(
        r"[a-zA-Z0-9][a-zA-Z0-9._-]*"
        r"(?::[a-zA-Z0-9._-]+)?"
    ),
    "r": _compile(r"[a-zA-Z][a-zA-Z0-9.]*"),
    # C/C++ go through apt on the default base image.
    "cpp": _compile(r"[a-z0-9][a-z0-9.+-]+"),
    "c": _compile(r"[a-z0-9][a-z0-9.+-]+"),
    # Java handler does not currently install at runtime (see
    # JavaHandler.is_support_library_installation == False); accept either a
    # bare artifact name or Maven `group:artifact[:version]` coords so a
    # future implementation has a sane default either way.
    "java": _compile(
        r"[a-zA-Z0-9][a-zA-Z0-9._-]*"
        r"(?::[a-zA-Z0-9][a-zA-Z0-9._-]*(?::[a-zA-Z0-9][a-zA-Z0-9._-]*)?)?"
    ),
}


def validate_package_name(library: str, language: str) -> str:
    """Validate ``library`` against the allowlist for ``language``.

    Returns the trimmed library string on success so callers can chain
    the validation with their f-string interpolation. Raises
    :class:`llm_sandbox.exceptions.ValidationError` on rejection — both
    empty strings and identifiers carrying shell metacharacters
    (``;``, ``&``, ``|``, ``$``, backticks, newlines, …) take that path.

    Unknown languages fall back to a strict "no shell metacharacters"
    sanity check rather than free-pass — better to surface "this
    language has no validator yet" with a known-safe rejection than to
    let an unfiltered string into a shell.
    """
    if library is None or not isinstance(library, str):
        msg = f"Library identifier must be a non-empty string, got {type(library).__name__}"
        raise ValidationError(msg)

    candidate = library.strip()
    if not candidate:
        msg = "Library identifier must be a non-empty string"
        raise ValidationError(msg)

    pattern = PACKAGE_NAME_PATTERNS.get(language.lower())
    if pattern is None:
        # Strict fallback: reject anything containing shell metacharacters,
        # whitespace, or quotes. Conservative on purpose — a maintainer
        # adding a new language can add a real pattern later.
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._/+:@-]*", candidate):
            msg = (
                f"Library {library!r} does not match the safe-charset "
                f"fallback for unknown language {language!r}. Add a "
                f"validator entry to PACKAGE_NAME_PATTERNS."
            )
            raise ValidationError(msg)
        return candidate

    if not pattern.fullmatch(candidate):
        msg = (
            f"Library {library!r} is not a valid {language} package "
            f"identifier (rejected to prevent shell-injection via "
            f"install command). Allowed characters depend on the "
            f"package manager — see PACKAGE_NAME_PATTERNS."
        )
        raise ValidationError(msg)

    return candidate
