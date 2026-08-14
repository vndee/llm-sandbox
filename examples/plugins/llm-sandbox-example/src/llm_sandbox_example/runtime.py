"""The runtime driver: a `ContainerAPI` backed by local subprocesses.

`ContainerAPI` is a structural protocol -- six methods that move a workload in, run it, and
move results out. Implementing it is the shortest path to a working backend, because
`llm_sandbox.backends.SandboxBackendBase` builds everything else on top: `run()`, `install()`,
security scanning, timeouts, and file transfer.

A real backend would talk to a container runtime or a remote API here. This one shells out.
"""

import io
import shutil
import subprocess
import sys
import tarfile
import threading
from pathlib import Path
from typing import Any

# Commands come from the language handler as `python <file>`. On many systems `python` is
# not on PATH, so the local runtime resolves it to the interpreter running llm-sandbox --
# the kind of environment translation a real backend does too.
_PYTHON_PREFIX = "python "


class LocalContainerAPI:
    """Run commands and move files in a local directory, via subprocesses.

    Satisfies `llm_sandbox.backends.ContainerAPI` structurally; there is no need to inherit
    from it. The "container" passed to each method is the sandbox directory path.
    """

    def __init__(self) -> None:
        """Initialize the runtime driver."""
        self._processes: set[subprocess.Popen[bytes]] = set()
        self._lock = threading.Lock()

    def create_container(self, config: Any) -> Any:
        """Create the sandbox directory.

        Args:
            config (Any): Mapping with a ``workdir`` key.

        Returns:
            Any: The sandbox directory path, used as the container handle.

        """
        workdir = Path(config["workdir"])
        workdir.mkdir(parents=True, exist_ok=True)
        return str(workdir)

    def start_container(self, container: Any) -> None:
        """Start the sandbox. Nothing to do for a directory.

        Args:
            container (Any): The sandbox directory path.

        """
        Path(container).mkdir(parents=True, exist_ok=True)

    def stop_container(self, container: Any) -> None:  # noqa: ARG002
        """Stop the sandbox by killing any command still running.

        Deliberately does **not** delete the directory. The session decides that, because
        only it knows whether the workdir was created for this session or handed in by the
        caller -- and deleting a caller's directory is not recoverable.

        Args:
            container (Any): The sandbox directory path. Unused.

        """
        self.terminate_running()

    def execute_command(self, container: Any, command: str, **kwargs: Any) -> tuple[int, Any]:
        """Run a shell command in the sandbox directory.

        Args:
            container (Any): The sandbox directory path.
            command (str): The command line to run.
            **kwargs: Accepts ``workdir``; ``stream`` is ignored because this runtime
                collects output and replays it (see `LocalSandboxSession.process_stream_output`).

        Returns:
            tuple[int, Any]: Exit code, and a ``(stdout_bytes, stderr_bytes)`` pair.

        """
        workdir = kwargs.get("workdir") or container
        Path(workdir).mkdir(parents=True, exist_ok=True)

        if command.startswith(_PYTHON_PREFIX):
            interpreter = shutil.which("python") or sys.executable
            command = f"{interpreter} {command[len(_PYTHON_PREFIX) :]}"

        # S602: running a shell command is the entire job of this backend. See the warning
        # in the module docstring of backend.py -- this example is not an isolation boundary.
        process = subprocess.Popen(  # noqa: S602
            command,
            shell=True,
            cwd=workdir,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        with self._lock:
            self._processes.add(process)
        try:
            stdout, stderr = process.communicate()
        finally:
            with self._lock:
                self._processes.discard(process)

        return process.returncode or 0, (stdout, stderr)

    def copy_to_container(self, container: Any, src: str, dest: str, **_kwargs: Any) -> None:  # noqa: ARG002
        """Copy a file or directory into the sandbox.

        Args:
            container (Any): The sandbox directory path.
            src (str): Host path to copy from.
            dest (str): Sandbox path to copy to.
            **_kwargs: Unused.

        """
        source = Path(src)
        destination = Path(dest)
        destination.parent.mkdir(parents=True, exist_ok=True)

        if source.is_dir():
            shutil.copytree(source, destination, dirs_exist_ok=True)
        else:
            shutil.copy2(source, destination)

    def copy_from_container(self, container: Any, src: str, **_kwargs: Any) -> tuple[bytes, dict]:  # noqa: ARG002
        """Read a path out of the sandbox as an uncompressed tar archive.

        Args:
            container (Any): The sandbox directory path.
            src (str): Sandbox path to read.
            **_kwargs: Unused.

        Returns:
            tuple[bytes, dict]: Tar bytes and a stat mapping. A ``size`` of 0 means the path
                does not exist, which is how callers detect a missing file.

        """
        source = Path(src)
        if not source.exists():
            return b"", {"size": 0}

        buffer = io.BytesIO()
        with tarfile.open(fileobj=buffer, mode="w") as tar:
            tar.add(source, arcname=source.name)

        payload = buffer.getvalue()
        return payload, {"name": source.name, "size": len(payload)}

    def terminate_running(self) -> None:
        """Kill every command still running.

        This is what makes timeouts real: core cannot kill the Python thread waiting on the
        command, so the backend has to kill the work itself.
        """
        with self._lock:
            processes = list(self._processes)
        for process in processes:
            if process.poll() is None:
                process.kill()
