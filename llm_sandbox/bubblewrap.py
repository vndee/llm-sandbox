"""Daemonless Linux execution using bubblewrap and optional seccomp hardening."""

import shutil
import sys
from collections.abc import Iterator
from contextlib import ExitStack, contextmanager
from pathlib import Path
from typing import Any

from llm_sandbox.exceptions import MissingDependencyError, ValidationError
from llm_sandbox.local.launch import read_only_mounts
from llm_sandbox.local.session import LocalProcessSession
from llm_sandbox.security.seccomp import compile_seccomp


class BubblewrapSession(LocalProcessSession):
    """Run commands in fresh namespaces with a persistent private workspace.

    Requires a Linux host with unprivileged user namespaces permitted by its
    kernel and security policy. The default seccomp layer requires libseccomp.
    Every invocation uses a new PID namespace: background processes cannot
    survive the command. Workspace and private temporary files persist across invocations.
    """

    def __init__(self, *, bwrap_path: str = "bwrap", **kwargs: Any) -> None:
        """Validate the launcher without importing a container SDK."""
        if sys.platform != "linux":
            msg = "The bubblewrap backend requires Linux"
            raise ValidationError(msg)
        executable = shutil.which(bwrap_path)
        if executable is None:
            msg = "The bubblewrap backend requires 'bwrap'. Install the bubblewrap package for your Linux distribution."
            raise MissingDependencyError(msg)
        # Commands launch from the private workspace, so resolve relative PATH
        # entries and explicit launcher paths while the caller's cwd is intact.
        self.bwrap_path = str(Path(executable).resolve())
        super().__init__(**kwargs)
        # Validate before creating a workspace or executing any command.
        read_only_mounts(self.sandbox_config, self.config.workdir)

    @contextmanager
    def _launch_command(self, command: str, workdir: str) -> Iterator[tuple[list[str], tuple[int, ...]]]:
        """Construct a namespace sandbox, keeping filter descriptors alive."""
        argv = [
            self.bwrap_path,
            "--unshare-all",
            "--unshare-user",
            "--die-with-parent",
            "--new-session",
            "--disable-userns",
            "--cap-drop",
            "ALL",
            "--clearenv",
        ]
        if self.sandbox_config.network == "host":
            argv.append("--share-net")
        for source, destination in read_only_mounts(self.sandbox_config, self.config.workdir):
            argv.extend(("--ro-bind", source, destination))
        argv.extend((
            "--proc",
            "/proc",
            "--dev",
            "/dev",
            "--bind",
            str(self.tempdir),
            "/tmp",  # NOSONAR(S5443) Destination inside the sandbox, backed by self.tempdir's private directory.
            "--bind",
            str(self.workspace),
            self.config.workdir,
            "--chdir",
            workdir,
        ))
        for key, value in sorted(self._environment().items()):
            argv.extend(("--setenv", key, value))
        with ExitStack() as stack:
            pass_fds: tuple[int, ...] = ()
            if self.sandbox_config.seccomp is not None:
                descriptor = stack.enter_context(compile_seccomp(self.sandbox_config.seccomp))
                argv.extend(("--seccomp", str(descriptor)))
                pass_fds = (descriptor,)
            argv.extend(("--", "/bin/sh", "-c", command))
            yield argv, pass_fds
