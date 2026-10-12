"""Conformance test suites for backend plugins.

A plugin subclasses one of these and points it at its own backend; pytest collects the
inherited tests and runs them against the real implementation:

```python
from llm_sandbox.testing import BackendComplianceTests


class TestMyServiceCompliance(BackendComplianceTests):
    backend = "myservice"
    session_kwargs = {"lang": "python"}
```

Two entry points, because the interface half needs no infrastructure:

- `BackendInterfaceComplianceTests` -- static checks only. Registration, plugin API version,
  capability declarations, and whether the methods a declared capability implies actually
  exist. Runs anywhere, including CI without credentials.
- `BackendComplianceTests` -- the above plus lifecycle, result types, file transfer, and
  timeout behaviour, which create real sessions.

Requires pytest, which core does not depend on. Install it with the ``testing`` extra:
``pip install llm-sandbox[testing]``.
"""

from llm_sandbox.testing.compliance import BackendComplianceTests, BackendInterfaceComplianceTests

__all__ = [
    "BackendComplianceTests",
    "BackendInterfaceComplianceTests",
]
