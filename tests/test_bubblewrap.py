"""Bubblewrap argument policies and conditional real namespace enforcement tests."""

from __future__ import annotations

import ctypes
import errno
import os
import platform
import shutil
import socket
import sys
from contextlib import contextmanager
from pathlib import Path
from tempfile import TemporaryFile
from typing import TYPE_CHECKING, Any
from unittest.mock import patch

import pytest

from llm_sandbox.bubblewrap import BubblewrapSession
from llm_sandbox.exceptions import CommandFailedError, MissingDependencyError, SecurityError, ValidationError
from llm_sandbox.local.config import LocalSandboxConfig
from llm_sandbox.security.seccomp import DEFAULT_BLOCKED_SYSCALLS, SeccompPolicy

if TYPE_CHECKING:
    from collections.abc import Iterator


@pytest.fixture
def session(tmp_path: Path) -> BubblewrapSession:
    """Build real driver arguments without requiring an installed launcher."""
    with (
        patch("llm_sandbox.bubblewrap.sys.platform", "linux"),
        patch("llm_sandbox.bubblewrap.shutil.which", return_value="/usr/bin/bwrap"),
    ):
        sandbox = BubblewrapSession(sandbox_config=LocalSandboxConfig(seccomp=None), skip_environment_setup=True)
    sandbox.workspace = tmp_path / "workspace"
    sandbox.tempdir = tmp_path / "tmp"
    return sandbox


def test_defaults_create_namespaces_and_only_bind_private_storage_writable(session: BubblewrapSession) -> None:
    """The default policy cannot expose the host filesystem or reuse its network."""
    with session._launch_command("pwd", "/sandbox") as (argv, pass_fds):
        options = argv[: argv.index("--")]
        assert "--unshare-all" in options
        assert "--unshare-user" in options
        assert "--share-net" not in options
        assert "--disable-userns" in options
        assert "--die-with-parent" in options
        assert "--new-session" in options
        assert options[options.index("--cap-drop") + 1] == "ALL"
        writable = [options[index + 1 : index + 3] for index, option in enumerate(options) if option == "--bind"]
        assert sorted(writable) == sorted([[str(session.workspace), "/sandbox"], [str(session.tempdir), "/tmp"]])
        assert all(source != "/tmp" for source, _ in writable)
        readonly = [options[index + 1 : index + 3] for index, option in enumerate(options) if option == "--ro-bind"]
        assert readonly
        assert all(source != "/" and destination != "/" for source, destination in readonly)
        assert options[options.index("--proc") + 1] == "/proc"
        assert options[options.index("--dev") + 1] == "/dev"
        assert pass_fds == ()


def test_host_network_is_only_enabled_explicitly(session: BubblewrapSession) -> None:
    """Sharing host networking must not disable the remaining namespace boundaries."""
    session.sandbox_config = LocalSandboxConfig(network="host", seccomp=None)
    with session._launch_command("true", "/sandbox") as (argv, _):
        assert argv.count("--share-net") == 1
        assert "--unshare-all" in argv
        assert "--unshare-user" in argv


def test_replacing_config_rebuilds_mounts_and_revokes_previous_grants(tmp_path: Path) -> None:
    """Revoked filesystem and network grants must not survive into later commands."""
    with (
        patch("llm_sandbox.bubblewrap.sys.platform", "linux"),
        patch("llm_sandbox.bubblewrap.shutil.which", return_value="/usr/bin/bwrap"),
        patch.object(Path, "exists", return_value=True),
        patch.object(Path, "resolve", autospec=True, side_effect=lambda path, **_: path),
        patch.object(Path, "is_file", return_value=True),
    ):
        sandbox = BubblewrapSession(
            sandbox_config=LocalSandboxConfig(network="host", read_only_paths=("/opt/sensitive",), seccomp=None)
        )
        sandbox.workspace = tmp_path / "workspace"
        sandbox.tempdir = tmp_path / "tmp"
        with sandbox._launch_command("true", "/sandbox") as (argv, _):
            assert "/opt/sensitive" in argv
            assert "/etc/resolv.conf" in argv
            assert "--share-net" in argv

        sandbox.sandbox_config = LocalSandboxConfig(read_only_paths=("/opt/public",), seccomp=None)
        with sandbox._launch_command("true", "/sandbox") as (argv, _):
            mounts = [argv[index + 1 : index + 3] for index, option in enumerate(argv) if option == "--ro-bind"]
            assert mounts == [["/opt/public", "/opt/public"]]
            assert "--share-net" not in argv

        sandbox.sandbox_config = LocalSandboxConfig(read_only_paths=(), seccomp=None)
        with sandbox._launch_command("true", "/sandbox") as (argv, _):
            assert "--ro-bind" not in argv


def test_launch_environment_excludes_host_secrets(session: BubblewrapSession, monkeypatch: pytest.MonkeyPatch) -> None:
    """Clear the inherited environment and pass only explicit variables and runtime defaults."""
    monkeypatch.setenv("BUBBLEWRAP_HOST_SECRET", "must-not-inherit")
    session._explicit_env = {"EXPLICIT": "value with spaces; echo cannot-run"}
    with session._launch_command("true", "/sandbox") as (argv, _):
        assert "--clearenv" in argv
        environment = {argv[index + 1]: argv[index + 2] for index, value in enumerate(argv) if value == "--setenv"}
        assert "BUBBLEWRAP_HOST_SECRET" not in environment
        assert "must-not-inherit" not in argv
        assert environment["HOME"] == "/sandbox/.home"
        assert environment["EXPLICIT"] == "value with spaces; echo cannot-run"


def test_command_and_workspace_cannot_inject_launcher_options(session: BubblewrapSession, tmp_path: Path) -> None:
    """Host paths and command text are separate arguments after the correct option boundaries."""
    session.workspace = tmp_path / "space --share-net"
    command = "--share-net; printf '%s' '$HOME'; echo --bind / /"
    with session._launch_command(command, "/sandbox") as (argv, _):
        boundary = argv.index("--")
        assert argv[boundary + 1 :] == ["/bin/sh", "-c", command]
        assert "--share-net" not in argv[:boundary]
        assert any(
            argv[index + 1 : index + 3] == [str(session.workspace), "/sandbox"]
            for index, option in enumerate(argv)
            if option == "--bind"
        )


def test_seccomp_descriptor_is_live_during_launch_and_closed_after_failure(session: BubblewrapSession) -> None:
    """Bubblewrap receives exactly the policy fd, whose lifetime covers child launch."""
    policy = SeccompPolicy(blocked_syscalls=("getppid",))
    session.sandbox_config = LocalSandboxConfig(seccomp=policy)
    observed: list[int] = []

    @contextmanager
    def compiled(request: SeccompPolicy) -> Iterator[int]:
        assert request is policy
        with TemporaryFile(mode="w+b") as file:
            file.write(b"policy-data")
            file.seek(0)
            observed.append(file.fileno())
            yield file.fileno()

    def fail_after_launch() -> None:
        with session._launch_command("true", "/sandbox") as (argv, pass_fds):
            descriptor = observed[0]
            assert pass_fds == (descriptor,)
            assert argv[argv.index("--seccomp") + 1] == str(descriptor)
            assert os.read(descriptor, 11) == b"policy-data"
            message = "child launch failed"
            raise RuntimeError(message)

    with patch("llm_sandbox.bubblewrap.compile_seccomp", compiled), pytest.raises(RuntimeError, match="child launch"):
        fail_after_launch()
    assert len(observed) == 1
    with pytest.raises(OSError, match="Bad file descriptor"):
        os.fstat(observed[0])


def test_requested_seccomp_failure_never_produces_launch_argv(session: BubblewrapSession) -> None:
    """A missing hardening dependency must not quietly produce a weaker sandbox."""
    session.sandbox_config = LocalSandboxConfig()
    with (
        patch("llm_sandbox.bubblewrap.compile_seccomp", side_effect=MissingDependencyError("libseccomp missing")),
        pytest.raises(MissingDependencyError, match="libseccomp missing"),
        session._launch_command("true", "/sandbox"),
    ):
        pytest.fail("A requested protection cannot be omitted")


def test_non_linux_constructor_rejects_backend() -> None:
    """An unsupported platform cannot fall back to host process execution."""
    with patch("llm_sandbox.bubblewrap.sys.platform", "darwin"), pytest.raises(ValidationError, match="requires Linux"):
        BubblewrapSession()


def test_missing_binary_constructor_rejects_backend() -> None:
    """Missing bubblewrap is an explicit dependency error."""
    with (
        patch("llm_sandbox.bubblewrap.sys.platform", "linux"),
        patch("llm_sandbox.bubblewrap.shutil.which", return_value=None),
        pytest.raises(MissingDependencyError, match="requires 'bwrap'"),
    ):
        BubblewrapSession()


@pytest.mark.skipif(sys.platform != "linux", reason="Requires executable POSIX symlinks")
@pytest.mark.parametrize("from_path", [False, True])
def test_relative_launcher_survives_private_working_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, from_path: bool
) -> None:
    """Actually execute a trusted cwd-reporting fixture, without claiming isolation."""
    native = tmp_path / "native"
    native.mkdir()
    launcher = native / "launcher"
    # This fixture ignores all supplied arguments and executes only trusted pwd.
    launcher.write_text("#!/bin/sh\npwd\n")
    launcher.chmod(0o700)
    (native / "bwrap").symlink_to("launcher")
    monkeypatch.chdir(tmp_path)
    if from_path:
        monkeypatch.setenv("PATH", "native")
    sandbox = BubblewrapSession(
        bwrap_path="bwrap" if from_path else "native/bwrap",
        sandbox_config=LocalSandboxConfig(seccomp=None),
        skip_environment_setup=True,
        default_timeout=5,
    )
    assert sandbox.bwrap_path == str(launcher.resolve())
    with sandbox:
        result = sandbox.execute_command("ignored by the trusted launcher fixture")
        assert result.exit_code == 0
        assert result.stdout.strip() == str(sandbox.workspace)
        assert sandbox.workspace.parent != tmp_path


@pytest.mark.parametrize(
    "kwargs",
    [
        {"image": "python:3"},
        {"dockerfile": "Dockerfile"},
        {"container_id": "existing"},
        {"runtime_configs": {"privileged": True}},
        {"keep_template": True},
        {"commit_container": True},
    ],
)
def test_container_options_are_rejected(kwargs: dict[str, Any]) -> None:
    """Unsupported container security settings must never be silently ignored."""
    with (
        patch("llm_sandbox.bubblewrap.sys.platform", "linux"),
        patch("llm_sandbox.bubblewrap.shutil.which", return_value="/usr/bin/bwrap"),
        pytest.raises(ValueError, match="Native sandboxes do not support"),
    ):
        BubblewrapSession(**kwargs)


@pytest.fixture(scope="module")
def real_bubblewrap() -> None:
    """Probe the default policy once and skip only known host limitations."""
    if sys.platform != "linux" or shutil.which("bwrap") is None:
        pytest.skip("Requires Linux and the bubblewrap executable")
    try:
        with BubblewrapSession(skip_environment_setup=True, default_timeout=5):
            pass
    except MissingDependencyError as exc:
        pytest.skip(f"Missing local sandbox dependency: {exc}")
    except SecurityError as exc:
        if "Seccomp supports native Linux" in str(exc) or "update libseccomp" in str(exc):
            pytest.skip(f"Unsupported native seccomp dependency: {exc}")
        raise
    except CommandFailedError as exc:
        known_host_failures = (
            "No permissions to create new namespace",
            "Creating new namespace failed: Operation not permitted",
            "Creating new namespace failed: Invalid argument",
            "Failed to write uid_map: Operation not permitted",
            "Failed to write gid_map: Operation not permitted",
            "Failed to create NETLINK_ROUTE socket: Operation not permitted",
            "--disable-userns requires",
            "Unknown option --disable-userns",
        )
        if any(message in str(exc) for message in known_host_failures):
            pytest.skip(f"Host cannot enforce the default bubblewrap policy: {exc}")
        raise


@pytest.mark.integration
@pytest.mark.usefixtures("real_bubblewrap")
def test_real_python_runs_and_private_storage_persists() -> None:
    """Code execution preserves private workspace and temporary files across fresh invocations."""
    sandbox = BubblewrapSession(skip_environment_setup=True, default_timeout=5)
    with sandbox:
        workspace = sandbox.workspace
        result = sandbox.run(
            "from pathlib import Path; Path('state').write_text('persisted'); "
            "Path('/tmp/artifact').write_text('private temporary data'); print(6 * 7)"
        )
        assert result.exit_code == 0, result.stderr
        assert result.stdout == "42\n"
        result = sandbox.run(
            "from pathlib import Path; assert Path('/tmp/artifact').read_text() == 'private temporary data'; "
            "print(Path('state').read_text())"
        )
        assert result.exit_code == 0, result.stderr
        assert result.stdout == "persisted\n"
    assert not workspace.exists()


@pytest.mark.integration
@pytest.mark.usefixtures("real_bubblewrap")
def test_real_secret_is_hidden_and_runtime_is_read_only(tmp_path: Path) -> None:
    """Filesystem boundaries must hold for actual child code, not just launcher arguments."""
    secret = tmp_path / "host-secret"
    secret.write_text("do not expose", encoding="utf-8")
    with BubblewrapSession(skip_environment_setup=True, default_timeout=5) as sandbox:
        result = sandbox.run(
            "import errno, os\n"
            f"assert not os.path.exists({str(secret)!r})\n"
            "try:\n"
            "    open('/usr/bubblewrap-test-must-not-create', 'w').close()\n"
            "except OSError as error:\n"
            "    assert error.errno in (errno.EROFS, errno.EACCES, errno.EPERM), error\n"
            "else:\n"
            "    raise AssertionError('runtime was writable')\n"
            "print('filesystem isolated')"
        )
        assert result.exit_code == 0, result.stderr
        assert result.stdout == "filesystem isolated\n"


@pytest.mark.integration
@pytest.mark.usefixtures("real_bubblewrap")
def test_real_default_network_cannot_reach_host_listener() -> None:
    """A known live host listener is inaccessible from the default network namespace."""
    with socket.socket() as listener, BubblewrapSession(skip_environment_setup=True, default_timeout=5) as sandbox:
        listener.bind(("127.0.0.1", 0))
        listener.listen(1)
        port = listener.getsockname()[1]
        result = sandbox.run(
            "import socket\n"
            "connection = socket.socket()\n"
            "connection.settimeout(1)\n"
            "try:\n"
            f"    connection.connect(('127.0.0.1', {port}))\n"
            "except OSError:\n"
            "    print('network isolated')\n"
            "else:\n"
            "    raise AssertionError('host network was reachable')\n"
            "finally:\n"
            "    connection.close()"
        )
        assert result.exit_code == 0, result.stderr
        assert result.stdout == "network isolated\n"


@pytest.mark.integration
@pytest.mark.usefixtures("real_bubblewrap")
def test_real_explicit_host_network_reaches_host_listener() -> None:
    """The explicit opt-in behaves as documented without requiring external network access."""
    with (
        socket.socket() as listener,
        BubblewrapSession(
            sandbox_config=LocalSandboxConfig(network="host"), skip_environment_setup=True, default_timeout=5
        ) as sandbox,
    ):
        listener.bind(("127.0.0.1", 0))
        listener.listen(1)
        port = listener.getsockname()[1]
        result = sandbox.run(
            f"import socket; connection = socket.create_connection(('127.0.0.1', {port}), timeout=1); "
            "connection.close(); print('explicit host network')"
        )
        assert result.exit_code == 0, result.stderr
        assert result.stdout == "explicit host network\n"


@pytest.mark.integration
@pytest.mark.usefixtures("real_bubblewrap")
def test_real_custom_seccomp_blocks_normally_available_getppid() -> None:
    """A custom deny rule is actually inherited by the untrusted Python process."""
    libc = ctypes.CDLL(None, use_errno=True)
    syscall_number = {"x86_64": 110, "amd64": 110, "aarch64": 173, "arm64": 173}[platform.machine().lower()]
    libc.syscall.restype = ctypes.c_long
    assert libc.syscall(syscall_number) > 0
    config = LocalSandboxConfig(seccomp=SeccompPolicy(blocked_syscalls=(*DEFAULT_BLOCKED_SYSCALLS, "getppid")))
    with BubblewrapSession(sandbox_config=config, skip_environment_setup=True, default_timeout=5) as sandbox:
        result = sandbox.run(
            "import ctypes\n"
            "libc = ctypes.CDLL(None, use_errno=True)\n"
            "libc.syscall.restype = ctypes.c_long\n"
            "ctypes.set_errno(0)\n"
            # libc's getppid wrapper assumes success and does not set errno.
            f"print(libc.syscall({syscall_number}), ctypes.get_errno())"
        )
        assert result.exit_code == 0, result.stderr
        assert result.stdout == f"-1 {errno.EPERM}\n"
