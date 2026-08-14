"""End-to-end test of the reference plugin, through the real entry point machinery.

This is the one test that proves the whole chain works together: a distribution on
``sys.path`` registers an entry point, the registry discovers and loads it, `create_session`
routes to it, and the compliance kit passes against the result.

The reference plugin lives in ``examples/plugins/llm-sandbox-example/`` and is not installed
into the development environment, so the ``.dist-info`` that registers it is fabricated here.
Everything downstream of that -- discovery, loading, session lifecycle -- is real.
"""

import sys
from collections.abc import Generator
from pathlib import Path

import pytest

from llm_sandbox import create_session, list_backends
from llm_sandbox.backends import BackendCapability
from llm_sandbox.exceptions import BackendCapabilityError
from llm_sandbox.registry import get_backend
from llm_sandbox.testing import BackendComplianceTests
from tests.plugin_helpers import installed

EXAMPLE_PLUGIN_ROOT = Path(__file__).parent.parent / "examples" / "plugins" / "llm-sandbox-example"
EXAMPLE_PLUGIN_SRC = EXAMPLE_PLUGIN_ROOT / "src"


def _write_dist_info(target: Path) -> Path:
    """Fabricate the ``.dist-info`` pip would install for the reference plugin.

    Args:
        target (Path): Directory to write the metadata into.

    Returns:
        Path: The directory containing the metadata.

    """
    dist_info = target / "llm_sandbox_example-0.1.0.dist-info"
    dist_info.mkdir(parents=True, exist_ok=True)
    (dist_info / "METADATA").write_text(
        "Metadata-Version: 2.1\nName: llm-sandbox-example\nVersion: 0.1.0\n"
    )
    (dist_info / "entry_points.txt").write_text(
        "[llm_sandbox.backends]\nexample = llm_sandbox_example:ExampleBackend\n"
    )
    return target


@pytest.fixture
def example_plugin(tmp_path: Path) -> Generator[None, None, None]:
    """Make the reference plugin discoverable as an installed distribution."""
    metadata_root = _write_dist_info(tmp_path / "site")
    with installed(metadata_root, EXAMPLE_PLUGIN_SRC):
        yield


class TestExamplePluginRegistration:
    """The reference plugin registers and resolves like any third-party backend."""

    def test_appears_in_list_backends(self, example_plugin: None) -> None:
        """Appears in list backends."""
        info = next(item for item in list_backends() if item.name == "example")

        assert info.is_builtin is False
        assert info.distribution == "llm-sandbox-example"
        assert info.version == "0.1.0"
        assert info.entry_point == "llm_sandbox_example:ExampleBackend"

    def test_not_imported_until_requested(self, example_plugin: None) -> None:
        """Not imported until requested."""
        list_backends()
        assert "llm_sandbox_example" not in sys.modules

        get_backend("example")
        assert "llm_sandbox_example" in sys.modules

    def test_declares_artifacts_only(self, example_plugin: None) -> None:
        """Declares artifacts only."""
        provider = get_backend("example")

        assert provider.supports(BackendCapability.ARTIFACTS)
        assert not provider.supports(BackendCapability.POOLING)
        assert not provider.supports(BackendCapability.INTERACTIVE)

    def test_undeclared_capabilities_raise(self, example_plugin: None) -> None:
        """Undeclared capabilities raise."""
        provider = get_backend("example")

        with pytest.raises(BackendCapabilityError):
            provider.create_pool_manager()
        with pytest.raises(BackendCapabilityError):
            provider.create_interactive_session()

    def test_create_session_routes_to_the_plugin(self, example_plugin: None) -> None:
        """Create session routes to the plugin."""
        session = create_session(backend="example", lang="python")
        try:
            assert type(session).__name__ == "LocalSandboxSession"
        finally:
            session.close()


@pytest.mark.usefixtures("example_plugin")
class TestExamplePluginCompliance(BackendComplianceTests):
    """Run the published compliance kit against the reference plugin.

    A plugin author writes exactly this much in their own repository. If the kit is wrong,
    or the reference plugin is wrong, this fails.
    """

    backend = "example"
    session_kwargs = {"lang": "python"}  # noqa: RUF012
