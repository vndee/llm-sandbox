"""Helpers for exercising the backend plugin system against real distribution metadata.

Rather than mocking `importlib.metadata`, these helpers write an actual ``.dist-info``
directory onto ``sys.path``. Entry point discovery, ``ep.dist.name``, ``ep.dist.version``,
and ``ep.load()`` then all run for real, which is the only way to be confident the registry
behaves the same on a user's machine as it does in CI.
"""

import importlib
import sys
from collections.abc import Generator
from contextlib import contextmanager
from pathlib import Path

from llm_sandbox import registry
from llm_sandbox.backends.plugin import ENTRY_POINT_GROUP

#: A plugin that satisfies the contract completely.
VALID_PLUGIN = """
from typing import Any, ClassVar

from llm_sandbox.backends import BackendCapability, SandboxBackendPlugin


class Backend(SandboxBackendPlugin):
    PLUGIN_API_VERSION: ClassVar[int] = 1
    name: ClassVar[str] = "{name}"
    capabilities: ClassVar[frozenset] = frozenset({{BackendCapability.ARTIFACTS}})

    @classmethod
    def create_session(cls, *args: Any, **kwargs: Any) -> Any:
        return {{"backend": cls.name, "kwargs": kwargs}}
"""

#: A module that raises while being imported.
EXPLODING_PLUGIN = """
raise RuntimeError("plugin import blew up")
"""

#: A plugin that forgets to declare PLUGIN_API_VERSION.
NO_VERSION_PLUGIN = """
from typing import Any, ClassVar

from llm_sandbox.backends import SandboxBackendPlugin


class Backend(SandboxBackendPlugin):
    name: ClassVar[str] = "{name}"

    @classmethod
    def create_session(cls, *args: Any, **kwargs: Any) -> Any:
        return object()
"""

#: A plugin built against a plugin API version core does not accept.
FUTURE_VERSION_PLUGIN = """
from typing import Any, ClassVar

from llm_sandbox.backends import SandboxBackendPlugin


class Backend(SandboxBackendPlugin):
    PLUGIN_API_VERSION: ClassVar[int] = 99
    name: ClassVar[str] = "{name}"

    @classmethod
    def create_session(cls, *args: Any, **kwargs: Any) -> Any:
        return object()
"""

#: An entry point resolving to something that is not a plugin class at all.
NOT_A_PLUGIN = """
Backend = "this is a string, not a SandboxBackendPlugin subclass"
"""

#: A plugin that never implements create_session, so it stays abstract.
ABSTRACT_PLUGIN = """
from typing import ClassVar

from llm_sandbox.backends import SandboxBackendPlugin


class Backend(SandboxBackendPlugin):
    PLUGIN_API_VERSION: ClassVar[int] = 1
    name: ClassVar[str] = "{name}"
"""

#: A plugin that never declares `name`, which `require()` and the optional factories read.
NO_NAME_PLUGIN = """
from typing import Any, ClassVar

from llm_sandbox.backends import SandboxBackendPlugin


class Backend(SandboxBackendPlugin):
    PLUGIN_API_VERSION: ClassVar[int] = 1

    @classmethod
    def create_session(cls, *args: Any, **kwargs: Any) -> Any:
        return object()
"""

#: A plugin that calls sys.exit() while being imported, as one might on missing config.
SYSTEM_EXIT_PLUGIN = """
import sys

sys.exit("plugin decided to exit during import")
"""

#: A plugin that declares POOLING and returns a working pool manager.
POOLING_PLUGIN = """
from typing import Any, ClassVar

from llm_sandbox.backends import BackendCapability, SandboxBackendPlugin


class FakePoolManager:
    def __init__(self, **kwargs: Any) -> None:
        self.kwargs = kwargs
        self.lang = kwargs.get("lang", "python")
        self.image = None
        self.client = object()


class Backend(SandboxBackendPlugin):
    PLUGIN_API_VERSION: ClassVar[int] = 1
    name: ClassVar[str] = "{name}"
    capabilities: ClassVar[frozenset] = frozenset({{BackendCapability.POOLING}})

    @classmethod
    def create_session(cls, *args: Any, **kwargs: Any) -> Any:
        return {{"backend": cls.name}}

    @classmethod
    def create_pool_manager(cls, **kwargs: Any) -> Any:
        return FakePoolManager(**kwargs)
"""

#: A plugin whose declared name disagrees with its entry point name.
MISMATCHED_NAME_PLUGIN = """
from typing import Any, ClassVar

from llm_sandbox.backends import SandboxBackendPlugin


class Backend(SandboxBackendPlugin):
    PLUGIN_API_VERSION: ClassVar[int] = 1
    name: ClassVar[str] = "something-else"

    @classmethod
    def create_session(cls, *args: Any, **kwargs: Any) -> Any:
        return object()
"""


def write_distribution(
    root: Path,
    *,
    distribution: str,
    version: str,
    module: str,
    source: str,
    entry_point_name: str,
    attribute: str = "Backend",
    group: str = ENTRY_POINT_GROUP,
) -> Path:
    """Write an importable module plus the ``.dist-info`` that registers its entry point.

    Args:
        root (Path): Directory to write into. Must be added to ``sys.path`` to take effect.
        distribution (str): Distribution name, as it appears in ``pip list``.
        version (str): Distribution version.
        module (str): Importable module name to create.
        source (str): Module source. ``{name}`` is formatted with ``entry_point_name``.
        entry_point_name (str): The name users would pass to ``backend=``.
        attribute (str): Attribute within the module the entry point points at.
        group (str): Entry point group.

    Returns:
        Path: The root directory, for convenience.

    """
    root.mkdir(parents=True, exist_ok=True)
    (root / f"{module}.py").write_text(source.format(name=entry_point_name))

    dist_info = root / f"{distribution.replace('-', '_')}-{version}.dist-info"
    dist_info.mkdir(parents=True, exist_ok=True)
    (dist_info / "METADATA").write_text(f"Metadata-Version: 2.1\nName: {distribution}\nVersion: {version}\n")
    (dist_info / "entry_points.txt").write_text(f"[{group}]\n{entry_point_name} = {module}:{attribute}\n")
    return root


@contextmanager
def installed(*roots: Path) -> Generator[None, None, None]:
    """Put directories on ``sys.path`` so their distributions become discoverable.

    Clears the registry cache on the way in and on the way out, and drops any modules the
    plugins imported, so tests do not leak state into each other.

    Args:
        *roots: Directories containing modules and ``.dist-info`` directories.

    Yields:
        None: While the distributions are importable.

    """
    added = [str(root) for root in roots]
    modules_before = set(sys.modules)

    for path in added:
        sys.path.insert(0, path)
    importlib.invalidate_caches()
    registry.clear_cache()
    try:
        yield
    finally:
        for path in added:
            if path in sys.path:
                sys.path.remove(path)
        for name in set(sys.modules) - modules_before:
            sys.modules.pop(name, None)
        importlib.invalidate_caches()
        registry.clear_cache()
