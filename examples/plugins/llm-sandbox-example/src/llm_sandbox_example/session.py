"""The session: what `create_session` hands back.

Subclassing `llm_sandbox.backends.SandboxBackendBase` means implementing only the six
required hooks plus lifecycle. Everything else -- ``run()``, ``install()``, security policy
enforcement, timeouts, file transfer, the context manager protocol -- is inherited.
"""

import shutil
import tempfile
from pathlib import Path
from typing import Any, ClassVar

from llm_sandbox.backends import SandboxBackendBase
from llm_sandbox.data import StreamCallback

#: Core's default workdir. An absolute path inside a container, not on the host.
CONTAINER_DEFAULT_WORKDIR = "/sandbox"


class LocalSandboxSession(SandboxBackendBase):
    """A session that runs code in a temporary directory on the local machine.

    Warning:
        This provides **no isolation**. Code runs as you, with your filesystem and your
        network. It exists to demonstrate the plugin interface. Never use it for untrusted
        code -- that is what the Docker, Podman, and Kubernetes backends are for.

    """

    backend_name: ClassVar[str] = "example"

    def __init__(self, **kwargs: Any) -> None:
        """Initialize the session.

        Args:
            **kwargs: Session arguments. ``workdir`` is replaced with a fresh temporary
                directory unless the caller named one of their own, since core's default is a
                path inside a container. ``skip_environment_setup`` defaults to True because
                there is no container to prepare; pass False to build a virtualenv and enable
                ``install()``.

        """
        # Core's workdir default is "/sandbox" -- an absolute path inside a container, which
        # is not writable on a normal machine. `ArtifactSandboxSession` also passes it
        # explicitly rather than leaving it unset, so "absent" is not enough to detect it.
        # Any backend that executes somewhere other than a container has to map this.
        requested_workdir = kwargs.get("workdir")
        self._owns_workdir = requested_workdir in (None, CONTAINER_DEFAULT_WORKDIR)
        if self._owns_workdir:
            kwargs["workdir"] = tempfile.mkdtemp(prefix="llm-sandbox-example-")
        kwargs.setdefault("skip_environment_setup", True)

        super().__init__(**kwargs)

        # Imported lazily so the module stays importable without a runtime present -- the
        # same discipline a real backend needs for its vendor SDK.
        from llm_sandbox_example.runtime import LocalContainerAPI

        self.container_api = LocalContainerAPI()

    # ------------------------------------------------------------------ #
    # Lifecycle
    # ------------------------------------------------------------------ #

    def open(self) -> None:
        """Create the sandbox directory and prepare the environment."""
        super().open()

        self.container = self.container_api.create_container({"workdir": self.config.workdir})
        self.container_api.start_container(self.container)

        self.environment_setup()

    def close(self) -> None:
        """Kill any running command and remove the sandbox directory.

        Called on every exit path, including when the ``with`` body raises. Releasing
        resources here is the single most important thing a backend gets right.
        """
        try:
            super().close()
        finally:
            # Whatever the base teardown does, the runtime must be stopped and a directory
            # we created must be removed. A backend that leaks on the error path is the
            # failure mode that quietly costs its users money.
            try:
                if self.container is not None:
                    self.container_api.stop_container(self.container)
                    self.container = None
            finally:
                if self._owns_workdir:
                    shutil.rmtree(self.config.workdir, ignore_errors=True)

    # ------------------------------------------------------------------ #
    # Required hooks
    # ------------------------------------------------------------------ #

    def handle_timeout(self) -> None:
        """Kill in-flight commands so a timeout actually cancels the work."""
        self.container_api.terminate_running()

    def ensure_directory_exists(self, path: str) -> None:
        """Create a directory inside the sandbox.

        Args:
            path (str): Absolute path of the directory to create.

        """
        Path(path).mkdir(parents=True, exist_ok=True)

    def ensure_ownership(self, paths: list[str]) -> None:
        """No-op: everything already runs as the invoking user.

        Args:
            paths (list[str]): Ignored.

        """

    def process_non_stream_output(self, output: Any) -> tuple[str, str]:
        """Decode the ``(stdout, stderr)`` byte pair the runtime returns.

        Args:
            output (Any): A ``(stdout_bytes, stderr_bytes)`` pair.

        Returns:
            tuple[str, str]: Decoded ``(stdout, stderr)``.

        """
        stdout, stderr = output
        errors = self.config.encoding_errors
        return (
            stdout.decode("utf-8", errors=errors) if stdout else "",
            stderr.decode("utf-8", errors=errors) if stderr else "",
        )

    def process_stream_output(
        self,
        output: Any,
        on_stdout: StreamCallback | None = None,
        on_stderr: StreamCallback | None = None,
    ) -> tuple[str, str]:
        """Replay collected output through the callbacks.

        This runtime waits for the command to finish, so there is nothing to stream. The
        contract requires the signature and requires the callbacks to fire -- it does not
        require chunks to arrive in real time.

        Args:
            output (Any): A ``(stdout_bytes, stderr_bytes)`` pair.
            on_stdout (StreamCallback | None): Called once with the complete stdout.
            on_stderr (StreamCallback | None): Called once with the complete stderr.

        Returns:
            tuple[str, str]: The accumulated ``(stdout, stderr)``.

        """
        stdout, stderr = self.process_non_stream_output(output)
        if on_stdout and stdout:
            on_stdout(stdout)
        if on_stderr and stderr:
            on_stderr(stderr)
        return stdout, stderr

    # ------------------------------------------------------------------ #
    # Optional: declared as BackendCapability.ARTIFACTS
    # ------------------------------------------------------------------ #

    def get_archive(self, path: str) -> tuple[bytes, dict]:
        """Read a path out of the sandbox as a tar archive.

        Args:
            path (str): Absolute path inside the sandbox.

        Returns:
            tuple[bytes, dict]: Tar bytes and a stat mapping with a ``size`` key.

        """
        return self.container_api.copy_from_container(self.container, path)
