"""Run the native integration suite, treating any skipped test as a CI failure."""

# ruff: noqa: INP001 -- Standalone CI entry point, not an importable package.

import logging
import sys

import pytest

logger = logging.getLogger(__name__)


class RejectSkippedTests:
    """Record skips during both collection and execution."""

    def __init__(self) -> None:
        """Start with an empty skip list."""
        self.skipped: list[str] = []

    def pytest_collectreport(self, report: pytest.CollectReport) -> None:
        """Record module-level skips that would avoid test execution."""
        if report.skipped:
            self.skipped.append(report.nodeid)

    def pytest_runtest_logreport(self, report: pytest.TestReport) -> None:
        """Record setup or test-body skips, including missing host prerequisites."""
        if report.skipped:
            self.skipped.append(report.nodeid)


def main() -> int:
    """Return pytest's result, except that skipped coverage always fails."""
    gate = RejectSkippedTests()
    status = pytest.main(sys.argv[1:], plugins=[gate])
    if gate.skipped:
        logger.error("Native integration tests must execute; skipped: %s", ", ".join(gate.skipped))
        return 1
    return int(status)


if __name__ == "__main__":
    raise SystemExit(main())
