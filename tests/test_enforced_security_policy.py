# ruff: noqa: SLF001, PLR2004, D102
"""Unit tests for the enforced security-policy execution path (#160).

These tests cover ``run(..., enforce_security_policy=True)`` and the
``safe_run`` wrapper. They mock the container backend so the policy check
fires before any container interaction.
"""

from unittest.mock import MagicMock, patch

import pytest

from llm_sandbox import SandboxSession
from llm_sandbox.exceptions import SecurityPolicyViolation, SecurityViolationError
from llm_sandbox.security import (
    RestrictedModule,
    SecurityIssueSeverity,
    SecurityPattern,
    SecurityPolicy,
)

pytestmark = pytest.mark.usefixtures("mock_docker_backend")


@pytest.fixture
def policy_high_threshold() -> SecurityPolicy:
    """Policy that blocks at HIGH severity and above."""
    return SecurityPolicy(
        severity_threshold=SecurityIssueSeverity.HIGH,
        patterns=[
            SecurityPattern(
                pattern=r"\bos\.system\s*\(",
                description="System command execution",
                severity=SecurityIssueSeverity.HIGH,
            ),
            SecurityPattern(
                pattern=r"\beval\s*\(",
                description="Dynamic code evaluation",
                severity=SecurityIssueSeverity.MEDIUM,
            ),
        ],
        restricted_modules=[
            RestrictedModule(
                name="subprocess",
                description="Subprocess management",
                severity=SecurityIssueSeverity.HIGH,
            ),
        ],
    )


@pytest.fixture
def policy_low_threshold() -> SecurityPolicy:
    """Policy that blocks at LOW severity and above."""
    return SecurityPolicy(
        severity_threshold=SecurityIssueSeverity.LOW,
        patterns=[
            SecurityPattern(
                pattern=r"\bsocket\.socket\s*\(",
                description="Raw socket creation",
                severity=SecurityIssueSeverity.LOW,
            ),
        ],
    )


@pytest.fixture
def policy_medium_threshold() -> SecurityPolicy:
    """Policy that blocks at MEDIUM and above (LOW is allowed)."""
    return SecurityPolicy(
        severity_threshold=SecurityIssueSeverity.MEDIUM,
        patterns=[
            SecurityPattern(
                pattern=r"\bsocket\.socket\s*\(",
                description="Raw socket creation",
                severity=SecurityIssueSeverity.LOW,
            ),
        ],
    )


class TestBackwardsCompatibility:
    """Default behavior must not change for callers that don't opt in."""

    def test_default_false_does_not_enforce_even_with_unsafe_code(
        self, policy_high_threshold: SecurityPolicy
    ) -> None:
        """run() without the kwarg should never check the policy."""
        session = SandboxSession(lang="python", security_policy=policy_high_threshold)

        unsafe_code = "import os\nos.system('rm -rf /')"
        # Should not raise SecurityPolicyViolation. Container is mocked, so
        # the real run path will fail later for unrelated reasons - all we
        # care about is that the *policy* did not block.
        with patch.object(session, "container", MagicMock()), patch.object(
            session, "is_open", new=True
        ), patch.object(session, "_execute_with_timeout", return_value=MagicMock(exit_code=0)):
            result = session.run(unsafe_code)
            assert result.exit_code == 0


class TestEnforcedExecutionBlocksUnsafeCode:
    """``enforce_security_policy=True`` blocks code that meets the threshold."""

    def test_high_severity_pattern_raises(
        self, policy_high_threshold: SecurityPolicy
    ) -> None:
        session = SandboxSession(lang="python", security_policy=policy_high_threshold)

        unsafe_code = "import os\nos.system('echo hi')"

        # The session container is intentionally NOT mocked-as-open: the
        # policy check must run *before* any session-state checks, so
        # SecurityPolicyViolation must be raised even on a closed session.
        with pytest.raises(SecurityPolicyViolation) as exc_info:
            session.run(unsafe_code, enforce_security_policy=True)

        exc = exc_info.value
        assert isinstance(exc, SecurityViolationError)  # subclass relationship
        assert exc.severity_threshold == SecurityIssueSeverity.HIGH
        assert any(v.description == "System command execution" for v in exc.violations)

    def test_restricted_module_import_raises_with_module_listed(
        self, policy_high_threshold: SecurityPolicy
    ) -> None:
        session = SandboxSession(lang="python", security_policy=policy_high_threshold)

        unsafe_code = "import subprocess\nsubprocess.run(['ls'])"

        with pytest.raises(SecurityPolicyViolation) as exc_info:
            session.run(unsafe_code, enforce_security_policy=True)

        exc = exc_info.value
        # The restricted module's description should be carried through on
        # the auto-generated pattern.
        assert any(v.description == "Subprocess management" for v in exc.violations)
        # __str__ must mention the offending description.
        assert "Subprocess management" in str(exc)

    def test_low_threshold_blocks_low_severity(
        self, policy_low_threshold: SecurityPolicy
    ) -> None:
        session = SandboxSession(lang="python", security_policy=policy_low_threshold)

        unsafe_code = "import socket\ns = socket.socket()"

        with pytest.raises(SecurityPolicyViolation) as exc_info:
            session.run(unsafe_code, enforce_security_policy=True)

        assert exc_info.value.severity_threshold == SecurityIssueSeverity.LOW

    def test_medium_threshold_does_not_block_low_severity(
        self, policy_medium_threshold: SecurityPolicy
    ) -> None:
        """Under-threshold matches must NOT raise."""
        session = SandboxSession(lang="python", security_policy=policy_medium_threshold)

        below_threshold_code = "import socket\ns = socket.socket()"

        # Container path mocked so that, after passing the policy check, the
        # call returns a stub ConsoleOutput. The point is no exception.
        with patch.object(session, "container", MagicMock()), patch.object(
            session, "is_open", new=True
        ), patch.object(session, "_execute_with_timeout", return_value=MagicMock(exit_code=0)):
            session.run(below_threshold_code, enforce_security_policy=True)


class TestEnforcedExecutionAllowsSafeCode:
    """Safe code must run when ``enforce_security_policy=True``."""

    def test_safe_code_runs(self, policy_high_threshold: SecurityPolicy) -> None:
        session = SandboxSession(lang="python", security_policy=policy_high_threshold)

        safe_code = "print('hello world')\nx = 1 + 1"

        with patch.object(session, "container", MagicMock()), patch.object(
            session, "is_open", new=True
        ), patch.object(session, "_execute_with_timeout", return_value=MagicMock(exit_code=0)):
            result = session.run(safe_code, enforce_security_policy=True)
            assert result.exit_code == 0


class TestSafeRunWrapper:
    """``safe_run`` is the ergonomic alias for the enforced path."""

    def test_safe_run_blocks_unsafe_code(
        self, policy_high_threshold: SecurityPolicy
    ) -> None:
        session = SandboxSession(lang="python", security_policy=policy_high_threshold)

        with pytest.raises(SecurityPolicyViolation):
            session.safe_run("import os\nos.system('id')")

    def test_safe_run_runs_safe_code(
        self, policy_high_threshold: SecurityPolicy
    ) -> None:
        session = SandboxSession(lang="python", security_policy=policy_high_threshold)

        with patch.object(session, "container", MagicMock()), patch.object(
            session, "is_open", new=True
        ), patch.object(session, "_execute_with_timeout", return_value=MagicMock(exit_code=0)):
            result = session.safe_run("print(2 + 2)")
            assert result.exit_code == 0


class TestSecurityPolicyViolationStructure:
    """The exception itself carries structured details."""

    def test_violations_populated(self, policy_high_threshold: SecurityPolicy) -> None:
        session = SandboxSession(lang="python", security_policy=policy_high_threshold)

        with pytest.raises(SecurityPolicyViolation) as exc_info:
            session.run("import os\nos.system('echo hi')", enforce_security_policy=True)

        exc = exc_info.value
        assert len(exc.violations) >= 1
        for v in exc.violations:
            assert isinstance(v, SecurityPattern)

    def test_severity_threshold_populated(
        self, policy_low_threshold: SecurityPolicy
    ) -> None:
        session = SandboxSession(lang="python", security_policy=policy_low_threshold)

        with pytest.raises(SecurityPolicyViolation) as exc_info:
            session.run("import socket\ns = socket.socket()", enforce_security_policy=True)

        assert exc_info.value.severity_threshold == SecurityIssueSeverity.LOW

    def test_str_lists_offending_items(
        self, policy_high_threshold: SecurityPolicy
    ) -> None:
        session = SandboxSession(lang="python", security_policy=policy_high_threshold)

        with pytest.raises(SecurityPolicyViolation) as exc_info:
            session.run("import os\nos.system('echo hi')", enforce_security_policy=True)

        message = str(exc_info.value)
        assert "Security policy violation" in message
        assert "System command execution" in message
        assert "HIGH" in message  # threshold name appears in the header

    def test_no_policy_means_no_enforcement(self) -> None:
        """Even with enforce_security_policy=True, no policy => no raise."""
        session = SandboxSession(lang="python")  # no security_policy

        with patch.object(session, "container", MagicMock()), patch.object(
            session, "is_open", new=True
        ), patch.object(session, "_execute_with_timeout", return_value=MagicMock(exit_code=0)):
            session.run("import os\nos.system('id')", enforce_security_policy=True)


class TestPreContainerCheckTiming:
    """Policy enforcement happens *before* any container interaction."""

    def test_policy_violation_raised_before_session_open_check(
        self, policy_high_threshold: SecurityPolicy
    ) -> None:
        """Closed session + unsafe code => SecurityPolicyViolation, not NotOpenSessionError."""
        session = SandboxSession(lang="python", security_policy=policy_high_threshold)
        # Session is intentionally closed (container=None, is_open=False).
        with pytest.raises(SecurityPolicyViolation):
            session.run("import os\nos.system('id')", enforce_security_policy=True)
