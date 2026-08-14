"""The backend plugin contract.

Every name in this module is part of the frozen plugin API. Third-party distributions
depend on these, so they may not be removed or changed incompatibly within a plugin API
major version. See :data:`PLUGIN_API_VERSION` and ``docs/plugins/authoring.md``.
"""

import re
import unicodedata
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, ClassVar, Literal

from llm_sandbox.const import StrEnum
from llm_sandbox.exceptions import BackendCapabilityError

if TYPE_CHECKING:
    from llm_sandbox.core.session_base import BaseSession
    from llm_sandbox.pool.base import ContainerPoolManager

#: The plugin API version this release of llm-sandbox implements.
#:
#: Deliberately independent of the package version. A plugin declares the version it was
#: built against as ``PLUGIN_API_VERSION`` on its :class:`SandboxBackendPlugin` subclass.
PLUGIN_API_VERSION = 1

#: Plugin API versions this release accepts.
#:
#: A set rather than a single value so a future release can accept an old and a new version
#: at once, giving plugin authors a transition window instead of a flag day.
SUPPORTED_PLUGIN_API_VERSIONS = frozenset({1})

#: The entry point group third-party backends register under.
ENTRY_POINT_GROUP = "llm_sandbox.backends"


#: A normalised backend name must match this. Enforced on both sides -- an entry point whose
#: name does not match is refused at discovery, and a lookup that does not match cannot
#: resolve. Without it, an entry point named ``""`` (which ``entry_points.txt`` permits)
#: would answer to ``backend=""``, so a caller reading the backend out of an unset config
#: value or environment variable would silently route execution to whoever registered it.
BACKEND_NAME_PATTERN = re.compile(r"^[a-z0-9][a-z0-9_]*$")


def normalize_backend_name(name: str) -> str:
    """Normalise a backend name for lookup and collision detection.

    Backend names are case-insensitive and treat hyphens and underscores as equivalent,
    so ``"My-Service"``, ``"my_service"`` and ``"MY_SERVICE"`` all address the same backend.

    Compatibility characters are folded first (NFKC), so visually confusable forms -- a
    fullwidth capital D, for instance -- cannot masquerade as a distinct backend in a
    listing or a pasted config line.

    Args:
        name (str): The raw backend name, as typed by a user or declared by a plugin.

    Returns:
        str: The normalised name. May be empty or otherwise invalid; callers check it
            against `BACKEND_NAME_PATTERN`.

    """
    return unicodedata.normalize("NFKC", name).strip().lower().replace("-", "_")


class BackendCapability(StrEnum):
    """Optional behaviour a backend may support beyond the required interface.

    A backend declares these on :attr:`SandboxBackendPlugin.capabilities`. Core checks the
    declaration *before* constructing anything, so an unsupported capability fails cleanly
    instead of leaving a half-built session or a leaked container behind.

    New capabilities may be added within a plugin API major version. A plugin that does not
    declare a capability it has never heard of is unaffected.
    """

    ARTIFACTS = "artifacts"
    """Plot and artifact extraction, via ``get_archive()``. Required by ArtifactSandboxSession."""

    INTERACTIVE = "interactive"
    """Stateful interpreter sessions. Required by InteractiveSandboxSession."""

    POOLING = "pooling"
    """Pre-warmed container pooling, via ``create_pool_manager()``."""

    EXISTING_CONTAINER = "existing_container"
    """Attaching to an already-running container or pod via ``container_id=``."""


#: Why a backend is or is not usable, as reported by :func:`llm_sandbox.list_backends`.
#: Widening this later is source-compatible for callers; narrowing a bare ``str`` would not
#: have been, which is why it is a ``Literal`` in a frozen dataclass.
BackendStatus = Literal["ok", "shadowed", "conflict", "error"]


@dataclass(frozen=True)
class BackendInfo:
    """A backend that core can see, whether or not it has been loaded.

    Returned by :func:`llm_sandbox.list_backends`. Everything except ``capabilities`` is
    read from distribution metadata, so building this never imports plugin code.

    Attributes:
        name (str): The normalised backend name, as passed to ``backend=``.
        is_builtin (bool): True for backends shipped and CI-tested by llm-sandbox itself.
        distribution (str | None): Distribution providing the backend; None for built-ins.
        version (str | None): Version of that distribution; None for built-ins.
        entry_point (str | None): The entry point's target, e.g. ``"pkg.module:Class"``.
        capabilities (frozenset[BackendCapability] | None): Declared capabilities, or None
            when the backend has not been loaded (the default -- see ``load=`` on
            :func:`llm_sandbox.list_backends`).
        status (BackendStatus): ``"ok"``, ``"shadowed"`` (a plugin tried to take a built-in
            name and lost), ``"conflict"`` (several distributions claim the name), or
            ``"error"`` (loading was attempted and failed). A ``Literal`` rather than a bare
            ``str`` so callers switching on it get checked, and so a typo in core is caught.
        detail (str | None): Explanation for any status other than ``"ok"``.

    """

    name: str
    is_builtin: bool = False
    distribution: str | None = None
    version: str | None = None
    entry_point: str | None = None
    capabilities: frozenset[BackendCapability] | None = None
    status: "BackendStatus" = "ok"
    detail: str | None = None


class SandboxBackendPlugin(ABC):
    """Descriptor a distribution registers to provide a sandbox backend.

    Subclass this, declare the class attributes, implement :meth:`create_session`, and point
    an entry point at the subclass:

    ```toml
    [project.entry-points."llm_sandbox.backends"]
    myservice = "llm_sandbox_myservice:MyServiceBackend"
    ```

    The entry point must resolve to the **class**, not an instance. Core never instantiates
    it -- every method is a classmethod. This class is a factory and a declaration of intent,
    not the session itself, which is what keeps the frozen surface small enough to promise
    forever while leaving what happens behind ``create_session`` entirely up to you.

    Example:
        ```python
        from llm_sandbox.backends import BackendCapability, SandboxBackendPlugin

        class MyServiceBackend(SandboxBackendPlugin):
            PLUGIN_API_VERSION = 1
            name = "myservice"
            capabilities = frozenset({BackendCapability.ARTIFACTS})

            @classmethod
            def create_session(cls, **kwargs):
                from llm_sandbox_myservice.session import MyServiceSession

                return MyServiceSession(**kwargs)
        ```

    """

    PLUGIN_API_VERSION: ClassVar[int]
    """The plugin API version this backend was built against. Required.

    Without it, a plugin built for an interface core no longer has fails as an
    ``AttributeError`` somewhere deep inside a session, which tells the user nothing.
    """

    name: ClassVar[str]
    """The canonical backend name. Required.

    The entry point name is what users actually type, but it is chosen by whoever wrote the
    ``pyproject.toml`` -- which is not always the plugin author. Declaring the name here lets
    core report a mismatch instead of silently honouring whichever it saw first.
    """

    capabilities: ClassVar[frozenset[BackendCapability]] = frozenset()
    """Optional behaviour this backend supports. See :class:`BackendCapability`."""

    @classmethod
    @abstractmethod
    def create_session(cls, *args: Any, **kwargs: Any) -> "BaseSession":
        """Create a sandbox session for this backend.

        Everything ``SandboxSession(backend=...)`` does routes through here. Implementations
        receive the keyword arguments the caller passed, minus ``backend``, and should ignore
        any they do not understand rather than raising -- callers share one call site across
        backends.

        Positional arguments are forwarded for backwards compatibility with the built-in
        backends, which historically accepted them. Treat keyword arguments as the interface;
        accept ``*args`` and ignore it.

        The returned object must implement the required interface: ``open()``, ``close()``,
        ``run()``, ``execute_command()``, ``copy_to_runtime()`` and ``copy_from_runtime()``.
        Subclassing :class:`llm_sandbox.backends.SandboxBackendBase` gives you all of it
        except the parts only you can supply; see ``docs/plugins/authoring.md``.

        Args:
            *args: Positional arguments, forwarded verbatim. Usually empty.
            **kwargs: Session arguments -- commonly ``lang``, ``image``, ``verbose``,
                ``workdir``, ``security_policy``, ``execution_timeout``, ``container_id``.

        Returns:
            BaseSession: A session satisfying the required backend interface.

        """
        raise NotImplementedError

    @classmethod
    def create_pool_manager(cls, **kwargs: Any) -> "ContainerPoolManager":  # noqa: ARG003
        """Create a container pool manager for this backend.

        Optional. Override this and declare :attr:`BackendCapability.POOLING` to support
        `llm_sandbox.pool.create_pool_manager`. Pooling has its own lifecycle and genuinely
        backend-specific semantics -- a hosted service may pool server-side, or not at all --
        so the default raises rather than pretending.

        Args:
            **kwargs: Pool arguments, commonly ``client``, ``config`` and ``lang``.

        Returns:
            ContainerPoolManager: A pool manager for this backend.

        Raises:
            BackendCapabilityError: If this backend does not support pooling.

        """
        raise BackendCapabilityError(cls.name, capability=BackendCapability.POOLING.value)

    @classmethod
    def create_interactive_session(cls, **kwargs: Any) -> "BaseSession":  # noqa: ARG003
        """Create the backing session for an interactive (stateful interpreter) session.

        Optional. Override this and declare :attr:`BackendCapability.INTERACTIVE` to support
        `llm_sandbox.InteractiveSandboxSession`. This is the least portable capability: core's
        interactive layer writes a runner script into the sandbox filesystem and polls it, so
        it only works for container-shaped backends. The default raises.

        Args:
            **kwargs: Session arguments, plus ``runtime_configs``.

        Returns:
            BaseSession: A session for the interactive layer to drive.

        Raises:
            BackendCapabilityError: If this backend does not support interactive sessions.

        """
        raise BackendCapabilityError(cls.name, capability=BackendCapability.INTERACTIVE.value)

    @classmethod
    def supports(cls, capability: BackendCapability | str) -> bool:
        """Report whether this backend declares a capability.

        Args:
            capability (BackendCapability | str): The capability to check.

        Returns:
            bool: True if declared.

        """
        return capability in cls.capabilities

    @classmethod
    def require(cls, capability: BackendCapability | str) -> None:
        """Raise unless this backend declares a capability.

        Args:
            capability (BackendCapability | str): The capability to require.

        Raises:
            BackendCapabilityError: If the capability is not declared.

        """
        if not cls.supports(capability):
            raise BackendCapabilityError(cls.name, capability=str(capability))
