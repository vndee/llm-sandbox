r"""Tenki Sandbox backend for LLM Sandbox.

This backend runs code in `Tenki <https://tenki.cloud/>`_ cloud microVMs instead of a
local container runtime, exposing the same session API as the Docker/Kubernetes/Podman
backends.
"""

import base64
import io
import shlex
import tarfile
import types
from pathlib import Path
from typing import Any

from tenki import Client, CommandResult, Sandbox, SessionNotFoundError, SessionTerminatedError
from tenki import SandboxError as TenkiError

from llm_sandbox.const import EncodingErrorsType, SupportedLanguage
from llm_sandbox.core.config import SessionConfig
from llm_sandbox.core.session_base import BaseSession
from llm_sandbox.data import ConsoleOutput, StreamCallback
from llm_sandbox.exceptions import CommandEmptyError, ContainerError, ExtraArgumentsError, NotOpenSessionError
from llm_sandbox.security import SecurityPolicy

# Runtimes present on Tenki's default guest image (live E2E). Other languages need image=.
_DEFAULT_GUEST_LANGUAGES = frozenset({
    SupportedLanguage.PYTHON,
    SupportedLanguage.JAVASCRIPT,
    SupportedLanguage.CPP,
})


def _exit_code_of(result: CommandResult) -> int:
    """Return a non-zero exit code whenever a command did not actually succeed.

    Returns:
        int: 0 only if the command genuinely succeeded.

    """
    if result.ok:
        return 0
    return result.exit_code or 128


def _failure_detail(result: Any) -> str:
    """Summarise why a command failed, beyond what its exit code conveys.

    Returns:
        str: A short ``[tenki: ...]`` note, or an empty string if the command was fine
            or the SDK reported no extra detail.

    """
    if getattr(result, "ok", True):
        return ""

    fields = ("signal", "reason", "errno")
    present = [f"{name}={value}" for name in fields if (value := getattr(result, name, None))]
    return f"[tenki: {', '.join(present)}]" if present else ""


def _archive_via_shell(container: Sandbox, src: str) -> tuple[bytes, dict]:
    """Archive a guest path with ``tar`` when the FS API cannot serve it."""
    if not container.shell(f"test -e {shlex.quote(src)}").ok:
        return b"", {"size": 0}

    parent = Path(src).parent.as_posix() or "/"
    name = Path(src).name
    result = container.shell(
        f"set -o pipefail; tar -cf - -C {shlex.quote(parent)} {shlex.quote(name)} | base64 | tr -d '\\n'"
    )
    if not result.ok or not result.stdout:
        return b"", {"size": 0}

    data = base64.b64decode(result.stdout)
    return data, {
        "name": name,
        "size": len(data),
        "mtime": 0,
        "mode": 0o644,
        "linkTarget": "",
    }


def _iter_remote_entries(container: Sandbox, root: str) -> list[tuple[str, Any]]:
    """List everything under a guest directory as ``(absolute_path, FileInfo)``.

    Directories are included, each before its contents. Emitting them is not cosmetic:
    the shared extractor treats a one-member tar whose member is a file as a single-file
    copy, so a directory holding exactly one file would be extracted over the
    destination directory, and a directory holding none would produce an empty archive.

    Returns:
        list[tuple[str, Any]]: Absolute guest paths paired with their FileInfo.

    """
    max_remote_walk_depth = 256

    entries: list[tuple[str, Any]] = []
    stack: list[tuple[str, int, list[Any]]] = [(root, 0, list(container.fs.list(root)))]
    visited: set[str] = {root.rstrip("/")}

    while stack:
        current, depth, children = stack[-1]
        if not children:
            stack.pop()
            continue

        entry = children.pop(0)
        full = f"{current.rstrip('/')}/{Path(entry.path).name}"
        entries.append((full, entry))

        if not entry.is_dir or entry.is_symlink or depth + 1 >= max_remote_walk_depth:
            continue

        normalized = full.rstrip("/")
        if normalized in visited:
            continue
        visited.add(normalized)
        stack.append((full, depth + 1, list(container.fs.list(full))))

    return entries


class TenkiContainerAPI:
    """Tenki implementation of the ContainerAPI protocol."""

    def __init__(self, client: Client) -> None:
        """Initialize Tenki container API.

        Args:
            client (Client): The Tenki client used to create and look up sandboxes.

        """
        self.client = client

    def create_container(self, config: Any) -> Sandbox:
        """Create a Tenki sandbox from ``client.create`` keyword arguments."""
        return self.client.create(**config)

    def start_container(self, container: Sandbox) -> None:
        """Wait until the sandbox can run commands (SDK default timeout)."""
        container.wait_ready()

    def stop_container(self, container: Sandbox) -> None:
        """Terminate the Tenki sandbox so the microVM is no longer billed.

        Raises:
            SessionNotFoundError: If the sandbox no longer exists.
            SessionTerminatedError: If the sandbox was already terminated.
            Exception: If both terminate attempts fail (last error raised).

        """
        try:
            container.terminate()
        except (SessionNotFoundError, SessionTerminatedError):
            raise
        except Exception as first_error:  # noqa: BLE001 - retry any release failure once
            try:
                self.client.get(container.id).terminate()
            except (SessionNotFoundError, SessionTerminatedError):
                return
            except Exception as retry_error:  # noqa: BLE001 - surface the last failure
                raise retry_error from first_error

    def execute_command(self, container: Sandbox, command: str, **kwargs: Any) -> tuple[int, Any]:
        """Execute a shell command in the sandbox.

        Args:
            container (Sandbox): The Tenki sandbox to run the command in.
            command (str): The shell command to run.
            **kwargs: Supports ``workdir`` and ``timeout`` (seconds, forwarded to
                ``Sandbox.shell``). ``stream`` is accepted for protocol compatibility
                but ignored, since Tenki exec is request/response.

        Returns:
            tuple[int, Any]: The exit code and the raw ``CommandResult``.

        """
        shell_kwargs: dict[str, Any] = {"cwd": kwargs.get("workdir")}
        timeout = kwargs.get("timeout")
        if timeout is not None:
            shell_kwargs["timeout"] = timeout
        result = container.shell(command, **shell_kwargs)
        return _exit_code_of(result), result

    def copy_to_container(self, container: Sandbox, src: str, dest: str, **_kwargs: Any) -> None:
        """Upload a host file or directory into the sandbox."""
        src_path = Path(src)
        if src_path.is_file():
            container.fs.upload(src, dest)
            return

        container.shell(f"mkdir -p {shlex.quote(dest)}")

        for path in src_path.rglob("*"):
            remote = f"{dest.rstrip('/')}/{path.relative_to(src_path).as_posix()}"
            if path.is_dir():
                container.shell(f"mkdir -p {shlex.quote(remote)}")
                continue
            if not path.is_file():
                continue
            container.shell(f"mkdir -p {shlex.quote(str(Path(remote).parent))}")
            container.fs.upload(str(path), remote)

    def copy_from_container(self, container: Sandbox, src: str, **_kwargs: Any) -> tuple[bytes, dict]:
        """Download a file or directory from the sandbox as a tar archive.

        Returns:
            tuple[bytes, dict]: The tar bytes and a stat dict (``size`` 0 if not found).

        """
        try:
            info = container.fs.stat(src)
        except TenkiError:
            return _archive_via_shell(container, src)

        name = Path(src).name
        if info.is_dir:
            entries = [(name, src, info)] + [
                (f"{name}/{Path(path).relative_to(src).as_posix()}", path, entry)
                for path, entry in _iter_remote_entries(container, src)
            ]
        else:
            entries = [(name, src, info)]

        tar_stream = io.BytesIO()
        total_size = 0

        with tarfile.open(fileobj=tar_stream, mode="w") as tar:
            for member_name, remote_path, file_info in entries:
                member = tarfile.TarInfo(name=member_name)
                member.mode = file_info.mode & 0o7777
                member.mtime = file_info.modified_unix_ns // 1_000_000_000

                if file_info.is_dir:
                    member.type = tarfile.DIRTYPE
                    tar.addfile(member)
                    continue

                data = container.fs.read_bytes(remote_path)
                member.size = len(data)
                tar.addfile(member, io.BytesIO(data))
                total_size += file_info.size

        archive = tar_stream.getvalue()
        return archive, {
            "name": name,
            "size": total_size or len(archive),
            "mtime": info.modified_unix_ns // 1_000_000_000,
            "mode": info.mode & 0o7777,
            "linkTarget": "",
        }


class SandboxTenkiSession(BaseSession):
    r"""Sandbox session implemented using Tenki cloud microVMs.

    Example:
        >>> with SandboxTenkiSession(lang="python") as session:
        ...     print(session.run("print('hello from Tenki')").stdout)

    """

    def __init__(
        self,  # NOSONAR
        client: Client | None = None,
        image: str | None = None,
        lang: str = SupportedLanguage.PYTHON,
        verbose: bool = False,
        stream: bool = False,
        runtime_configs: dict | None = None,
        workdir: str = "/home/tenki",
        security_policy: SecurityPolicy | None = None,
        default_timeout: float | None = None,
        execution_timeout: float | None = None,
        session_timeout: float | None = None,
        container_id: str | None = None,
        skip_environment_setup: bool = False,
        encoding_errors: EncodingErrorsType = "strict",
        auth_token: str | None = None,
        base_url: str | None = None,
        **kwargs: Any,
    ) -> None:
        r"""Initialize Tenki session.

        Args:
            client (Client | None): An existing Tenki client. If None, one is built from
                `auth_token`/`base_url`, falling back to the `TENKI_AUTH_TOKEN` or
                `TENKI_API_KEY` environment variable.
            image (str | None): The Tenki image or template to boot. If None, the Tenki
                account default is used. That default only includes Python, Node.js, and
                g++; other languages require a custom ``image`` (or an existing
                ``container_id``).
            lang (str): The language to use.
            verbose (bool): Whether to enable verbose output.
            stream (bool): Whether to stream the output. Tenki executes commands
                request/response, so callbacks receive one chunk per command.
            runtime_configs (dict | None): Extra keyword arguments forwarded to
                `Client.create` (e.g. `cpu_cores`, `memory_mb`, `allow_outbound`, `env`, `tags`).
            workdir (str): The working directory inside the sandbox.
            security_policy (SecurityPolicy | None): The security policy to use for the session.
            default_timeout (float | None): The default timeout to use.
            execution_timeout (float | None): The timeout for code execution.
            session_timeout (float | None): The timeout for the session.
            container_id (str | None): ID of an existing Tenki session to attach to. Such a
                session is detached, not terminated, on close.
            skip_environment_setup (bool): Skip language-specific environment setup.
            encoding_errors (EncodingErrorsType): Error handling for decoding command output.
            auth_token (str | None): Tenki auth token or API key. Prefer the `TENKI_AUTH_TOKEN`
                (or `TENKI_API_KEY`) environment variable over passing it here.
            base_url (str | None): Tenki API base URL. Defaults to the SDK's.
            **kwargs: Additional keyword arguments.

        Raises:
            ExtraArgumentsError: If `dockerfile` is provided; Tenki boots prebuilt images.
                Also raised when ``lang`` is not on the default guest and neither
                ``image`` nor ``container_id`` is set.

        """
        if kwargs.pop("dockerfile", None):
            msg = (
                "The Tenki backend does not build images from a Dockerfile. "
                "Pass a prebuilt Tenki image or template via 'image' instead."
            )
            raise ExtraArgumentsError(msg)

        resolved_lang = SupportedLanguage(lang.upper())
        if image is None and container_id is None and resolved_lang not in _DEFAULT_GUEST_LANGUAGES:
            supported = ", ".join(sorted(guest_lang.value for guest_lang in _DEFAULT_GUEST_LANGUAGES))
            msg = (
                f"The Tenki default guest image does not include {resolved_lang.value}. "
                f"It ships {supported} only. Pass image=... with a guest that provides "
                f"{resolved_lang.value}, or attach to an existing sandbox via container_id=...."
            )
            raise ExtraArgumentsError(msg)

        config = SessionConfig(
            image=image,
            lang=resolved_lang,
            verbose=verbose,
            workdir=workdir,
            runtime_configs=runtime_configs or {},
            security_policy=security_policy,
            default_timeout=default_timeout,
            execution_timeout=execution_timeout,
            session_timeout=session_timeout,
            container_id=container_id,
            skip_environment_setup=skip_environment_setup,
            encoding_errors=encoding_errors,
        )

        super().__init__(config=config, **kwargs)

        self._client = client
        self._auth_token = auth_token
        self._base_url = base_url
        self._owns_client = client is None
        self.stream: bool = stream

    def _get_client(self) -> Client:
        """Return the Tenki client, creating one if needed.

        Credential discovery is left to the SDK, which falls back to
        ``TENKI_AUTH_TOKEN`` then ``TENKI_API_KEY`` and raises its own
        ``MissingAuthTokenError`` naming both when neither is set.

        Returns:
            Client: The Tenki client for this session.

        """
        if self._client is None:
            self._client = Client(auth_token=self._auth_token, base_url=self._base_url)
        return self._client

    def open(self) -> None:
        r"""Open the Tenki session.

        Raises:
            ContainerError: If the session is already open, or the sandbox cannot be
                created, attached to, or prepared.

        """
        if self.container is not None:
            msg = "This session is already open; call close() before opening it again."
            raise ContainerError(msg)

        super().open()

        try:
            self.container_api = TenkiContainerAPI(self._get_client())

            if self.using_existing_container and self.config.container_id:
                self._connect_to_existing_container(self.config.container_id)
            else:
                self._create_container()

            needs_python = self.config.lang == SupportedLanguage.PYTHON and not self.using_existing_container
            if needs_python:
                self._ensure_python_interpreter()

            self.environment_setup()

            if needs_python and not self.config.skip_environment_setup:
                self._verify_python_environment()
        except BaseException as setup_error:
            try:
                self.close()
            except Exception as cleanup_error:  # noqa: BLE001 - must not displace setup_error
                self._log_cleanup_failure(
                    f"Tenki sandbox {getattr(self.container, 'id', 'unknown')} is STILL RUNNING: "
                    f"CLEANUP FAILED ({cleanup_error}). Call close() again to retry or terminate directly."
                )
                setup_error.__context__ = cleanup_error
            raise

    def _create_container(self) -> None:
        r"""Create a new Tenki sandbox and wait until it can run commands.

        Raises:
            ContainerError: If the sandbox cannot be created or never becomes ready.

        """
        create_config: dict[str, Any] = dict(self.config.runtime_configs)
        if self.config.image:
            create_config["image"] = self.config.image

        env = dict(create_config.get("env") or {})
        env.setdefault("PYTHONUNBUFFERED", "1")
        create_config["env"] = env

        create_config["wait"] = False
        provision_timeout = create_config.get("timeout")

        try:
            self.container = self.container_api.create_container(create_config)
            if provision_timeout is None:
                self.container.wait_ready()
            else:
                self.container.wait_ready(provision_timeout)
        except Exception as e:
            msg = f"Failed to create Tenki sandbox: {e}"
            self._log(msg, "error")
            raise ContainerError(msg) from e

        self._log(f"Created Tenki sandbox {self.container.id}")

    def _ensure_python_interpreter(self) -> None:
        r"""Make a bare ``python`` available in the guest before the venv is built.

        Raises:
            ContainerError: If the guest has neither ``python`` nor ``python3``.

        """
        python_bootstrap_command = (
            "if ! command -v python >/dev/null 2>&1; then "
            'if command -v python3 >/dev/null 2>&1; then ln -sf "$(command -v python3)" /usr/local/bin/python; '
            "else echo 'no python or python3 on PATH' >&2; exit 127; fi; fi; python -V"
        )
        result = self.container.shell(python_bootstrap_command, privileged=True)
        if not result.ok:
            reason = " ".join(filter(None, [self._decode(result.stderr).strip(), _failure_detail(result)]))
            msg = (
                "The Tenki sandbox image has no Python interpreter "
                f"({reason}). Pass an image that ships Python via "
                "image=..., or set skip_environment_setup=True and manage the runtime yourself."
            )
            self._log(msg, "error")
            raise ContainerError(msg)

        self._log(f"Guest interpreter: {self._decode(result.stdout).strip()}")

    def _verify_python_environment(self) -> None:
        r"""Confirm the virtualenv environment_setup was supposed to build actually exists.

        Raises:
            ContainerError: If the session's Python executable is missing.

        """
        exit_code, _ = self.container_api.execute_command(
            self.container, f"test -x {shlex.quote(self.python_executable_path)}"
        )
        if exit_code:
            msg = (
                f"Python environment setup failed: {self.python_executable_path} was not created. "
                "The guest may be missing the venv module (Debian: apt-get install python3-venv). "
                "Re-run with verbose=True to see the setup output."
            )
            self._log(msg, "error")
            raise ContainerError(msg)

    def close(self) -> None:
        r"""Close the Tenki session, releasing the sandbox.

        Raises:
            ContainerError: If the sandbox could not be terminated or detached. The
                sandbox handle and client are both kept so close() can be retried;
                until one succeeds the microVM is still running and still billed.

        """
        super().close()

        if self.container:
            try:
                if self.using_existing_container:
                    self.container.detach()
                    self._log(f"Detached from existing Tenki sandbox {self.container.id}")
                else:
                    self.container_api.stop_container(self.container)
                    self._log(f"Terminated Tenki sandbox {self.container.id}")
            except (SessionNotFoundError, SessionTerminatedError) as e:
                self._log(f"Tenki sandbox {self.container.id} was already gone ({e})")
            except Exception as e:
                msg = f"Failed to release Tenki sandbox {self.container.id}: {e}"
                self._log_cleanup_failure(msg)
                raise ContainerError(msg) from e

            self.container = None

        if self._owns_client and self._client:
            try:
                self._client.close()
            except Exception as e:  # noqa: BLE001
                self._log(f"Error closing Tenki client: {e}", "error")
            self._client = None

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: types.TracebackType | None,
    ) -> None:
        """Close the session without letting cleanup mask an error from the block."""
        if exc_type is None:
            self.close()
            return

        try:
            self.close()
        except Exception as cleanup_error:  # noqa: BLE001 - must not displace the block error
            self._log_cleanup_failure(
                f"Tenki sandbox {getattr(self.container, 'id', 'unknown')} is STILL RUNNING: "
                f"CLEANUP FAILED ({cleanup_error}). Call close() again to retry or terminate directly."
            )
            if exc_val is not None:
                exc_val.__context__ = cleanup_error

    def get_archive(self, path: str) -> tuple[bytes, dict]:
        """Get an archive of a path from the Tenki sandbox.

        Returns:
            tuple[bytes, dict]: Tar archive bytes and a stat dict.

        """
        if not self.container:
            raise NotOpenSessionError

        return self.container_api.copy_from_container(self.container, path)

    def execute_command(
        self,
        command: str,
        workdir: str | None = None,
        on_stdout: StreamCallback | None = None,
        on_stderr: StreamCallback | None = None,
    ) -> ConsoleOutput:
        """Execute a command, forwarding ``execution_timeout`` to the Tenki SDK.

        Same behaviour as the shared mixin, plus ``timeout=`` on ``Sandbox.shell`` so
        hung guest commands are killed by Tenki rather than only by the host session
        timeout handler.

        """
        if not command:
            raise CommandEmptyError

        if not self.container:
            raise NotOpenSessionError

        if self.verbose:
            self.logger.info("Executing command: %s", command)

        effective_stream = self.stream or (on_stdout is not None) or (on_stderr is not None)

        exit_code, output = self.container_api.execute_command(
            self.container,
            command,
            workdir=workdir,
            stream=effective_stream,
            timeout=self.config.get_execution_timeout(),
        )

        stdout, stderr = self._process_output(output, effective_stream, on_stdout=on_stdout, on_stderr=on_stderr)

        if self.verbose:
            if stdout:
                self.logger.info("STDOUT: %s", stdout)
            if stderr:
                self.logger.error("STDERR: %s", stderr)

        return ConsoleOutput(exit_code=exit_code or 0, stdout=stdout, stderr=stderr)

    def _handle_timeout(self) -> None:
        """Handle Tenki timeout cleanup."""
        try:
            self.close()
        except Exception as e:  # noqa: BLE001
            self._log_cleanup_failure(f"Error during timeout cleanup: {e}")

    def _log_cleanup_failure(self, message: str) -> None:
        """Emit billable-leak / cleanup-failure messages even when verbose is off."""
        self.logger.error(message)

    def _connect_to_existing_container(self, container_id: str) -> None:
        r"""Attach to an existing Tenki sandbox.

        Args:
            container_id (str): The ID of the existing Tenki session to attach to.

        Raises:
            ContainerError: If the sandbox cannot be found or is not usable.

        """
        try:
            container = self._get_client().get(container_id)

            if container.state in {"PAUSED", "PAUSING"}:
                self._log(f"Tenki sandbox {container_id} is paused, resuming...")
                container.resume()

            provision_timeout = self.config.runtime_configs.get("timeout")
            if provision_timeout is None:
                container.wait_ready()
            else:
                container.wait_ready(provision_timeout)
            self.container = container
            self._log(f"Attached to existing Tenki sandbox {container_id}")
        except Exception as e:
            msg = f"Failed to attach to Tenki sandbox {container_id}: {e}"
            self._log(msg, "error")
            raise ContainerError(msg) from e

    def _ensure_directory_exists(self, path: str) -> None:
        r"""Ensure a directory exists inside the sandbox.

        Args:
            path (str): The path to create.

        """
        exit_code, output = self.container_api.execute_command(self.container, f"mkdir -p {shlex.quote(path)}")
        if exit_code:
            stdout_output, stderr_output = self._process_non_stream_output(output)
            self._log(f"Failed to create directory {path}: {stderr_output or stdout_output}", "error")

    def _ensure_ownership(self, paths: list[str]) -> None:
        r"""No-op: a Tenki sandbox is single-tenant and runs everything as one user.

        Args:
            paths (list[str]): Unused; kept to satisfy the FileOperationsMixin hook.

        """

    def _decode(self, data: bytes | None) -> str:
        """Decode command output bytes using the session's encoding error mode.

        Returns:
            str: The decoded text, or an empty string if there was no output.

        """
        if not data:
            return ""
        return data.decode("utf-8", errors=self.config.encoding_errors)

    def _process_non_stream_output(self, output: Any) -> tuple[str, str]:
        """Process a Tenki ``CommandResult`` into ``(stdout, stderr)``.

        Returns:
            tuple[str, str]: The decoded stdout and stderr.

        """
        stdout = self._decode(getattr(output, "stdout", None))
        stderr = self._decode(getattr(output, "stderr", None))

        detail = _failure_detail(output)
        if detail:
            stderr = f"{stderr.rstrip()}\n{detail}" if stderr.strip() else detail

        return stdout, stderr

    def _process_stream_output(
        self,
        output: Any,
        on_stdout: StreamCallback | None = None,
        on_stderr: StreamCallback | None = None,
    ) -> tuple[str, str]:
        """Process Tenki output for streaming consumers.

        Args:
            output: The ``CommandResult`` returned by the Tenki SDK.
            on_stdout: Optional callback invoked with the decoded stdout.
            on_stderr: Optional callback invoked with the decoded stderr.

        Returns:
            tuple[str, str]: The decoded stdout and stderr.

        """
        stdout_output, stderr_output = self._process_non_stream_output(output)

        if stdout_output and on_stdout:
            on_stdout(stdout_output)
        if stderr_output and on_stderr:
            on_stderr(stderr_output)

        return stdout_output, stderr_output
