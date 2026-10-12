"""Public API for writing a third-party backend.

Everything a backend plugin needs is importable from here or from the top-level
``llm_sandbox`` package. Plugins must not import from ``llm_sandbox.core`` or any other
private module: only the names listed in ``docs/plugins/authoring.md`` are covered by the
compatibility guarantee, and internals will be refactored without notice.

```python
from llm_sandbox.backends import (
    BackendCapability,
    SandboxBackendBase,
    SandboxBackendPlugin,
)


class MyServiceBackend(SandboxBackendPlugin):
    PLUGIN_API_VERSION = 1
    name = "myservice"
    capabilities = frozenset({BackendCapability.ARTIFACTS})

    @classmethod
    def create_session(cls, *args, **kwargs):
        from llm_sandbox_myservice.session import MyServiceSession

        return MyServiceSession(**kwargs)
```

See ``docs/plugins/authoring.md`` for the full guide and
``llm_sandbox.testing.BackendComplianceTests`` for the conformance suite.
"""

from llm_sandbox.backends.base import SandboxBackendBase
from llm_sandbox.backends.plugin import (
    ENTRY_POINT_GROUP,
    PLUGIN_API_VERSION,
    SUPPORTED_PLUGIN_API_VERSIONS,
    BackendCapability,
    BackendInfo,
    SandboxBackendPlugin,
    normalize_backend_name,
)
from llm_sandbox.core.mixins import ContainerAPI
from llm_sandbox.core.session_base import BaseSession

__all__ = [
    "ENTRY_POINT_GROUP",
    "PLUGIN_API_VERSION",
    "SUPPORTED_PLUGIN_API_VERSIONS",
    "BackendCapability",
    "BackendInfo",
    "BaseSession",
    "ContainerAPI",
    "SandboxBackendBase",
    "SandboxBackendPlugin",
    "normalize_backend_name",
]
