"""Public base class for third-party backend sessions.

`llm_sandbox.core.session_base.BaseSession` is the real backend contract, but six of its
eight abstract methods are underscore-prefixed, and a plugin cannot be asked to implement
private API. This module re-declares those six under public names and bridges them, so a
plugin implements only names that are covered by the compatibility guarantee.

The bridges are ``@final``: override the public name, never the underscore one.
"""

from abc import ABC, abstractmethod
from typing import Any, ClassVar, final, get_args

from llm_sandbox.backends.plugin import BackendCapability
from llm_sandbox.core.config import SessionConfig
from llm_sandbox.core.session_base import BaseSession
from llm_sandbox.data import StreamCallback
from llm_sandbox.exceptions import BackendCapabilityError


def _nullable_config_fields() -> frozenset[str]:
    """Names of `SessionConfig` fields that accept ``None``.

    Returns:
        frozenset[str]: The nullable field names.

    """
    # get_args handles both `typing.Optional[X]` and the `X | None` form, which is a
    # types.UnionType rather than a typing.Union on Python 3.10.
    return frozenset(
        name
        for name, field in SessionConfig.model_fields.items()
        if field.annotation is None or type(None) in get_args(field.annotation)
    )


_NULLABLE_CONFIG_FIELDS = _nullable_config_fields()


class SandboxBackendBase(BaseSession, ABC):
    """Base class for a backend session provided by a plugin.

    Subclasses get everything backend-agnostic for free -- security scanning, language
    handler selection, library installation, session timeouts, the context manager protocol,
    and the ``run()`` pipeline -- and implement only what is genuinely backend-specific.

    Required overrides:

    - `open` / `close` -- lifecycle
    - `handle_timeout` -- cancel in-flight work when execution times out
    - `ensure_directory_exists` / `ensure_ownership` -- filesystem preparation
    - `process_non_stream_output` / `process_stream_output` -- decode runtime output

    Optional, gated on a declared `llm_sandbox.backends.BackendCapability`:

    - `get_archive` -- required for ``ARTIFACTS``
    - `connect_to_existing_container` -- required for ``EXISTING_CONTAINER``

    Note:
        The inherited ``run()`` writes code to a temporary file, copies it into the sandbox
        workdir, and executes shell commands from the language handler. It assumes a POSIX
        filesystem and a shell. A backend without those must either supply a
        `llm_sandbox.core.mixins.ContainerAPI` that emulates them, or override ``run()``
        outright -- which also gives up ``install()`` and artifact extraction. See
        ``docs/plugins/authoring.md``.

    """

    backend_name: ClassVar[str] = ""
    """The backend name this session belongs to.

    Set it to match your plugin's `SandboxBackendPlugin.name` so that diagnostics and
    `llm_sandbox.exceptions.UnsupportedBackendError.backend` name the backend rather than
    the session class. Defaults to empty, in which case the class name is used.
    """

    def __init__(self, config: SessionConfig | None = None, **kwargs: Any) -> None:
        """Initialize the backend session.

        Args:
            config (SessionConfig | None): A pre-built session configuration. When omitted,
                one is built from any recognised keyword arguments (``lang``, ``image``,
                ``workdir``, ``verbose``, ``security_policy``, timeouts, and so on), which
                is usually what a `create_session` implementation wants.
            **kwargs: Session configuration values and backend-specific arguments.
                Unrecognised keys are passed through to `BaseSession` untouched.

        """
        if config is None:
            config, kwargs = self._build_config(kwargs)

        # CommandExecutionMixin reads self.stream; set it before super().__init__ so a
        # subclass can still override it afterwards.
        self.stream: bool = bool(kwargs.pop("stream", False))

        super().__init__(config=config, **kwargs)

    @staticmethod
    def _build_config(kwargs: dict[str, Any]) -> tuple[SessionConfig, dict[str, Any]]:
        """Split session configuration values out of a keyword argument mapping.

        Core forwards ``None`` for several settings whose caller-side default is ``None``
        but whose `SessionConfig` field is not nullable -- ``runtime_configs`` and
        ``workdir`` both arrive that way from `llm_sandbox.ArtifactSandboxSession` and
        `llm_sandbox.InteractiveSandboxSession`. The built-in backends each normalise these
        by hand before constructing their config; doing it here means a plugin does not have
        to know which settings need it.

        Args:
            kwargs (dict[str, Any]): The keyword arguments handed to the session.

        Returns:
            tuple[SessionConfig, dict[str, Any]]: The configuration, and the keyword
                arguments that were not consumed by it.

        """
        remaining = dict(kwargs)
        config_kwargs = {key: remaining.pop(key) for key in list(remaining) if key in SessionConfig.model_fields}
        config_kwargs = {
            key: value for key, value in config_kwargs.items() if value is not None or key in _NULLABLE_CONFIG_FIELDS
        }
        return SessionConfig(**config_kwargs), remaining

    # ------------------------------------------------------------------ #
    # Required: implement these
    # ------------------------------------------------------------------ #

    @abstractmethod
    def handle_timeout(self) -> None:
        """Cancel in-flight work after an execution timeout.

        Called from a background thread once `run` exceeds its timeout. Python cannot kill
        the thread running the code, so this is the only mechanism that actually stops
        work -- kill the container, cancel the remote job, close the connection.

        Must not raise; failures here are logged and swallowed by the caller.
        """
        raise NotImplementedError

    @abstractmethod
    def ensure_directory_exists(self, path: str) -> None:
        """Create a directory inside the sandbox, including parents.

        Called before copying a file in, so the destination's parent exists.

        Args:
            path (str): Absolute path of the directory to create.

        """
        raise NotImplementedError

    @abstractmethod
    def ensure_ownership(self, paths: list[str]) -> None:
        """Make the given paths writable by the user the sandbox executes as.

        Backends that always execute as the owning user may implement this as a no-op.

        Args:
            paths (list[str]): Absolute paths inside the sandbox.

        """
        raise NotImplementedError

    @abstractmethod
    def process_non_stream_output(self, output: Any) -> tuple[str, str]:
        """Decode a completed command's output into stdout and stderr.

        Args:
            output (Any): Whatever this backend's command execution returned.

        Returns:
            tuple[str, str]: Decoded ``(stdout, stderr)``.

        """
        raise NotImplementedError

    @abstractmethod
    def process_stream_output(
        self,
        output: Any,
        on_stdout: StreamCallback | None = None,
        on_stderr: StreamCallback | None = None,
    ) -> tuple[str, str]:
        """Consume a streaming command's output, invoking callbacks as chunks arrive.

        A backend that cannot stream may collect the full output and invoke each callback
        once at completion; the signature is required, real-time delivery is not.

        Args:
            output (Any): The stream object this backend's command execution returned.
            on_stdout (StreamCallback | None): Called with each decoded stdout chunk.
            on_stderr (StreamCallback | None): Called with each decoded stderr chunk.

        Returns:
            tuple[str, str]: The fully accumulated ``(stdout, stderr)``.

        """
        raise NotImplementedError

    # ------------------------------------------------------------------ #
    # Optional: gated on a declared capability
    # ------------------------------------------------------------------ #

    def connect_to_existing_container(self, container_id: str) -> None:  # noqa: ARG002
        """Attach to an already-running container, pod, or remote session.

        Override this and declare `BackendCapability.EXISTING_CONTAINER` to support
        ``container_id=``. When attached, core skips environment setup and leaves the
        runtime running on close.

        Args:
            container_id (str): Identifier of the runtime to attach to.

        Raises:
            BackendCapabilityError: If this backend cannot attach to existing runtimes.

        """
        raise BackendCapabilityError(
            self.backend_name or type(self).__name__,
            capability=BackendCapability.EXISTING_CONTAINER.value,
        )

    def get_archive(self, path: str) -> tuple[bytes, dict]:  # noqa: ARG002
        """Read a path out of the sandbox as an uncompressed tar archive.

        Override this and declare `BackendCapability.ARTIFACTS` to support
        `llm_sandbox.ArtifactSandboxSession`. The language handler calls it to collect
        generated plots.

        Args:
            path (str): Absolute path inside the sandbox.

        Returns:
            tuple[bytes, dict]: The tar bytes, and a stat mapping with at least a ``size``
                key. A ``size`` of 0 signals "not found" to the caller.

        Raises:
            BackendCapabilityError: If this backend does not support artifact extraction.

        """
        raise BackendCapabilityError(
            self.backend_name or type(self).__name__,
            capability=BackendCapability.ARTIFACTS.value,
        )

    # ------------------------------------------------------------------ #
    # Bridges onto BaseSession's private contract. Do not override.
    # ------------------------------------------------------------------ #

    @final
    def _handle_timeout(self) -> None:
        """Bridge to `handle_timeout`."""
        self.handle_timeout()

    @final
    def _connect_to_existing_container(self, container_id: str) -> None:
        """Bridge to `connect_to_existing_container`."""
        self.connect_to_existing_container(container_id)

    @final
    def _ensure_directory_exists(self, path: str) -> None:
        """Bridge to `ensure_directory_exists`."""
        self.ensure_directory_exists(path)

    @final
    def _ensure_ownership(self, paths: list[str]) -> None:
        """Bridge to `ensure_ownership`."""
        self.ensure_ownership(paths)

    @final
    def _process_non_stream_output(self, output: Any) -> tuple[str, str]:
        """Bridge to `process_non_stream_output`."""
        return self.process_non_stream_output(output)

    @final
    def _process_stream_output(
        self,
        output: Any,
        on_stdout: StreamCallback | None = None,
        on_stderr: StreamCallback | None = None,
    ) -> tuple[str, str]:
        """Bridge to `process_stream_output`."""
        return self.process_stream_output(output, on_stdout=on_stdout, on_stderr=on_stderr)
