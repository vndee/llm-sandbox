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

from llm_sandbox import ArtifactSandboxSession, create_session, list_backends
from llm_sandbox.backends import BackendCapability
from llm_sandbox.exceptions import BackendCapabilityError
from llm_sandbox.registry import get_backend
from llm_sandbox.testing import BackendComplianceTests
from tests.plugin_helpers import installed, write_distribution

#: A plugin that declares no capabilities at all.
NO_CAPABILITY_PLUGIN = '''
from typing import Any, ClassVar

from llm_sandbox.backends import SandboxBackendPlugin


class Backend(SandboxBackendPlugin):
    PLUGIN_API_VERSION: ClassVar[int] = 1
    name: ClassVar[str] = "{name}"

    @classmethod
    def create_session(cls, *args: Any, **kwargs: Any) -> Any:
        return object()
'''

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


class TestArtifactSessionWithAPlugin:
    """Regression: ArtifactSandboxSession is the only consumer of BackendCapability.ARTIFACTS.

    It forwards ``runtime_configs=None`` and ``workdir="/sandbox"`` unconditionally, which
    used to reach SessionConfig as a pydantic ValidationError -- not even a SandboxError, so
    `except SandboxError` downstream would not catch it.
    """

    def test_artifact_session_runs_against_a_plugin(self, example_plugin: None) -> None:
        """The plugin's declared capability actually works end to end."""
        with ArtifactSandboxSession(backend="example", lang="python") as session:
            result = session.run("print('artifacts ok')")

        assert result.exit_code == 0
        assert "artifacts ok" in result.stdout

    def test_artifact_session_maps_the_container_workdir(self, example_plugin: None) -> None:
        """A host-executing backend must not be handed core's in-container default verbatim."""
        session = ArtifactSandboxSession(backend="example", lang="python")
        with session:
            assert session._session.config.workdir != "/sandbox"

    def test_backend_without_artifacts_is_refused_before_construction(
        self, tmp_path: Path
    ) -> None:
        """A backend that cannot do artifacts fails up front, not mid-run with a live container."""
        root = write_distribution(
            tmp_path / "site",
            distribution="llm-sandbox-plain",
            version="0.1.0",
            module="plain_backend",
            source=NO_CAPABILITY_PLUGIN,
            entry_point_name="plain",
        )
        with installed(root), pytest.raises(BackendCapabilityError, match="artifacts"):
            ArtifactSandboxSession(backend="plain", lang="python")

    def test_existing_container_is_refused_when_undeclared(self, tmp_path: Path) -> None:
        """container_id= requires the EXISTING_CONTAINER capability."""
        root = write_distribution(
            tmp_path / "site",
            distribution="llm-sandbox-plain",
            version="0.1.0",
            module="plain_backend",
            source=NO_CAPABILITY_PLUGIN,
            entry_point_name="plain",
        )
        with installed(root), pytest.raises(BackendCapabilityError, match="existing_container"):
            create_session(backend="plain", container_id="abc123")


class TestExamplePluginWorkdirOwnership:
    """Mirrors the reference plugin's own tests, which this repo's CI does not collect.

    Whether a backend deletes a directory the caller supplied is exactly the irreversible
    class of bug worth guarding in CI.
    """

    def test_temporary_workdir_is_removed_on_close(self, example_plugin: None) -> None:
        """A directory the backend created is cleaned up."""
        session = create_session(backend="example", lang="python")
        workdir = Path(session.config.workdir)

        with session:
            assert workdir.exists()

        assert not workdir.exists(), "Session leaked its temporary working directory"

    def test_caller_supplied_workdir_is_left_alone(self, example_plugin: None, tmp_path: Path) -> None:
        """A directory the caller named is never deleted."""
        workdir = tmp_path / "mine"
        workdir.mkdir()

        with create_session(backend="example", lang="python", workdir=str(workdir)) as session:
            session.run("print('hello')")

        assert workdir.exists(), "Backend deleted a directory it did not create"
