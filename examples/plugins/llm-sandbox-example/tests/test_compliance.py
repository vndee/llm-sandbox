"""Run the llm-sandbox compliance kit against this backend.

This is the whole file. Subclass the suite, name your backend, and pytest collects every
inherited check. Copy it into your own plugin and change `backend` and `session_kwargs`.

Run it with:

    pip install -e ".[test]"
    pytest
"""

from llm_sandbox.testing import BackendComplianceTests


class TestExampleBackendCompliance(BackendComplianceTests):
    """Full conformance suite: interface, lifecycle, result types, files, timeouts."""

    backend = "example"
    session_kwargs = {"lang": "python"}  # noqa: RUF012
