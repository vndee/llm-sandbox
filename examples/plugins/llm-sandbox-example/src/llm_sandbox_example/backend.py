"""The plugin descriptor: the object the entry point points at.

Warning:
    This backend runs code **directly on the host machine**, with your user, your
    filesystem, and your network. It is a reference implementation of the plugin interface,
    not a sandbox. Do not run untrusted code with it. Use the Docker, Podman, or Kubernetes
    backends for that.

"""

from typing import TYPE_CHECKING, Any, ClassVar

from llm_sandbox.backends import BackendCapability, SandboxBackendPlugin

if TYPE_CHECKING:
    from llm_sandbox.backends import BaseSession


class ExampleBackend(SandboxBackendPlugin):
    """A minimal, complete backend plugin.

    Everything a plugin must declare is here and nothing else is required:

    - `PLUGIN_API_VERSION` -- which plugin interface this was built against
    - `name` -- the canonical backend name
    - `capabilities` -- the optional behaviour this backend actually implements
    - `create_session` -- the factory everything routes through

    `create_pool_manager` and `create_interactive_session` are not overridden, because this
    backend declares neither capability. Their inherited defaults raise a clear error.
    """

    PLUGIN_API_VERSION: ClassVar[int] = 1
    name: ClassVar[str] = "example"
    capabilities: ClassVar[frozenset[BackendCapability]] = frozenset({BackendCapability.ARTIFACTS})

    @classmethod
    def create_session(cls, *args: Any, **kwargs: Any) -> "BaseSession":  # noqa: ARG003
        """Create a local subprocess sandbox session.

        The session class is imported here rather than at module scope so that resolving the
        entry point stays cheap, and so an import error in the session module surfaces as a
        clear `llm_sandbox.BackendLoadError` naming this distribution.

        Args:
            *args: Ignored. Accepted because core forwards positional arguments for
                backwards compatibility with the built-in backends.
            **kwargs: Session arguments, forwarded verbatim.

        Returns:
            BaseSession: The session.

        """
        from llm_sandbox_example.session import LocalSandboxSession

        return LocalSandboxSession(**kwargs)
