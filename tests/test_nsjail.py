"""NsJail driver policies must not weaken isolation through argument handling."""

from __future__ import annotations

import os
import platform
import re
import shutil
import signal
import socket
import subprocess
import sys
import time
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

from llm_sandbox.exceptions import (
    CommandFailedError,
    MissingDependencyError,
    SandboxTimeoutError,
    SecurityError,
    UnsupportedBackendError,
)
from llm_sandbox.local.config import DEFAULT_READ_ONLY_PATHS, LocalSandboxConfig
from llm_sandbox.nsjail import NsjailSession, _mount_argument, _seccomp_policy
from llm_sandbox.security.seccomp import DEFAULT_BLOCKED_SYSCALLS, SeccompPolicy


@pytest.fixture
def session(tmp_path: Path) -> NsjailSession:
    """Construct the real session without invoking an external sandbox process."""
    with (
        patch("llm_sandbox.nsjail.platform.system", return_value="Linux"),
        patch("llm_sandbox.nsjail.shutil.which", return_value="/usr/bin/nsjail"),
    ):
        sandbox = NsjailSession(sandbox_config=LocalSandboxConfig(seccomp=None))
    sandbox.workspace = tmp_path / "workspace"
    sandbox.tempdir = tmp_path / "tmp"
    return sandbox


def test_minimal_mounts_and_namespace_defaults(session: NsjailSession) -> None:
    """Expose the workspace and allowed runtimes without a host-root mount."""
    with session._launch_command("pwd", "/sandbox") as (argv, pass_fds):
        options = argv[: argv.index("--")]
        assert pass_fds == ()
        assert "--chroot" not in options
        assert "--rw" not in options
        assert "--disable_no_new_privs" not in options
        assert "--forward_signals" not in options
        assert "--keep_caps" not in options
        assert not any(option.startswith("--disable_clone_") for option in options)
        assert "--iface_no_lo" in options
        assert options[options.index("--experimental_mnt") + 1] == "new"
        writable_mounts = [options[i + 1] for i, option in enumerate(options) if option == "--bindmount"]
        assert writable_mounts == [f"{session.workspace}:/sandbox", f"{session.tempdir}:/tmp"]
        read_mounts = [options[i + 1] for i, option in enumerate(options) if option == "--bindmount_ro"]
        assert not any(mount.startswith("/:") for mount in read_mounts)
        assert "/dev/null:/dev/null" in read_mounts
        assert "/dev/urandom:/dev/urandom" in read_mounts
        assert "/dev:/dev" not in read_mounts
        assert options[options.index("--mount") + 1] == "none:/dev/shm:tmpfs:size=67108864,mode=1777"
        assert not any("/dev/shm" in mount for mount in read_mounts + writable_mounts)
        assert "--disable_proc" not in options


def test_host_network_requires_explicit_configuration(session: NsjailSession) -> None:
    """Grant only network-namespace sharing when the caller explicitly asks."""
    session.sandbox_config = LocalSandboxConfig(network="host", seccomp=None)
    with session._launch_command("true", "/sandbox") as (argv, _):
        assert "--disable_clone_newnet" in argv
        assert "--iface_no_lo" not in argv
        assert "--disable_clone_newuser" not in argv


def test_replacing_config_rebuilds_mounts_and_revokes_previous_grants(session: NsjailSession) -> None:
    """Each jail must honor only the current filesystem and network grants."""
    with (
        patch.object(Path, "exists", return_value=True),
        patch.object(Path, "resolve", autospec=True, side_effect=lambda path, **_: path),
        patch.object(Path, "is_file", return_value=True),
    ):
        session.sandbox_config = LocalSandboxConfig(network="host", read_only_paths=("/opt/sensitive",), seccomp=None)
        with session._launch_command("true", "/sandbox") as (argv, _):
            assert "/opt/sensitive:/opt/sensitive" in argv
            assert "/etc/resolv.conf:/etc/resolv.conf" in argv
            assert "--disable_clone_newnet" in argv

        session.sandbox_config = LocalSandboxConfig(read_only_paths=("/opt/public",), seccomp=None)
        with session._launch_command("true", "/sandbox") as (argv, _):
            mounts = {argv[index + 1] for index, option in enumerate(argv) if option == "--bindmount_ro"}
            assert mounts == {
                "/opt/public:/opt/public",
                "/dev/null:/dev/null",
                "/dev/zero:/dev/zero",
                "/dev/random:/dev/random",
                "/dev/urandom:/dev/urandom",
            }
            assert "--disable_clone_newnet" not in argv
            assert "--iface_no_lo" in argv


def test_unknown_network_fails_closed(session: NsjailSession) -> None:
    """Even a config created without Pydantic validation cannot broaden access."""
    session.sandbox_config = LocalSandboxConfig.model_construct(network="unexpected", seccomp=None)
    with pytest.raises(SecurityError, match="network policy"), session._launch_command("true", "/sandbox"):
        pytest.fail("Invalid network settings must never produce a launcher command")


def test_command_cannot_become_launcher_options(session: NsjailSession) -> None:
    """Treat command text as one shell argument after NsJail's option boundary."""
    command = "--disable_clone_newns; printf '%s' '$HOME'; echo --chroot /"
    with session._launch_command(command, "/sandbox") as (argv, _):
        boundary = argv.index("--")
        assert argv[boundary + 1 :] == ["/bin/sh", "-c", command]
        assert "--disable_clone_newns" not in argv[:boundary]


@pytest.mark.parametrize(
    ("source", "destination"), [("/safe:/", "/sandbox"), ("/safe", "/sandbox:/"), ("/safe\0", "/sandbox")]
)
def test_mount_separator_injection_rejected(source: str, destination: str) -> None:
    """NsJail's colon-delimited grammar must not reinterpret a literal path."""
    with pytest.raises(SecurityError, match="mount paths"):
        _mount_argument(source, destination)


def test_mount_paths_with_spaces_stay_one_argument(session: NsjailSession, tmp_path: Path) -> None:
    """Spaces in a caller-supplied host path cannot inject extra switches."""
    session.workspace = tmp_path / "with spaces --disable_clone_newns"
    with session._launch_command("true", "/sandbox") as (argv, _):
        index = argv.index("--bindmount")
        assert argv[index + 1] == f"{session.workspace}:/sandbox"
        assert "--disable_clone_newns" not in argv


def test_inherit_supervisor_limits_and_clean_environment(session: NsjailSession) -> None:
    """Keep configured prlimit values instead of applying NsJail's defaults."""
    with session._launch_command("true", "/sandbox") as (argv, _):
        assert "--disable_rlimits" in argv
        assert not any(option.startswith("--rlimit_") for option in argv)
        assert argv[argv.index("--time_limit") + 1] == "0"
        assert "--keep_env" not in argv


def test_payload_environment_is_forwarded_as_exact_jailed_arguments(session: NsjailSession) -> None:
    """Loader hooks and arbitrary values belong to the payload, never the host launcher."""
    session._explicit_env = {
        "LD_PRELOAD": "/sandbox/lib custom.so",
        "LD_LIBRARY_PATH": "/sandbox/custom",
        "EMPTY": "",
        "MESSAGE": "first=second\n--chroot=/",
    }
    with (
        patch.dict("os.environ", {"UNRELATED_HOST_SECRET": "must-not-be-forwarded"}),
        session._launch_command("true", "/sandbox") as (argv, _),
    ):
        options = argv[: argv.index("--")]
        environment = [options[i + 1] for i, option in enumerate(options) if option == "--env"]
        assert "--keep_env" not in options
        assert "--chroot" not in options
        assert "LD_PRELOAD=/sandbox/lib custom.so" in environment
        assert "LD_LIBRARY_PATH=/sandbox/custom" in environment
        assert "EMPTY=" in environment
        assert "MESSAGE=first=second\n--chroot=/" in environment
        assert "HOME=/sandbox/.home" in environment
        assert not any(value.startswith("UNRELATED_HOST_SECRET=") for value in environment)


def test_native_seccomp_blocks_x32_and_high_syscall_numbers(session: NsjailSession) -> None:
    """Named deny rules take precedence; x32 cannot bypass them via its ABI bit."""
    session.sandbox_config = LocalSandboxConfig(seccomp=SeccompPolicy(blocked_syscalls=("ptrace", "mount")))
    with (
        patch("llm_sandbox.nsjail.platform.machine", return_value="x86_64"),
        session._launch_command("true", "/sandbox") as (argv, _),
    ):
        policy = argv[argv.index("--seccomp_string") + 1]
        assert policy.index("ERRNO(1) { ptrace, mount }") < policy.index("ALLOW {")
        allowed_numbers = {int(number) for number in re.findall(r"SYSCALL\[(\d+)\]", policy)}
        assert 0 in allowed_numbers
        assert 1023 in allowed_numbers
        assert 1024 not in allowed_numbers
        assert not allowed_numbers.intersection(range(512, 548))
        assert (0x40000000 | 101) not in allowed_numbers  # x32 ptrace
        assert policy.endswith("DEFAULT ERRNO(1)")


@pytest.mark.parametrize("architecture", ["aarch64", "arm64"])
def test_x32_number_exclusion_is_specific_to_x86(architecture: str) -> None:
    """Do not apply an x86 compatibility-ABI restriction to AArch64."""
    with patch("llm_sandbox.nsjail.platform.machine", return_value=architecture):
        policy = _seccomp_policy(("ptrace",))
    allowed_numbers = {int(number) for number in re.findall(r"SYSCALL\[(\d+)\]", policy)}
    assert allowed_numbers == set(range(1024))


def test_default_seccomp_translates_libseccomp_unmount_name_for_kafel() -> None:
    """The shared default policy must use Kafel's name for the two-argument syscall."""
    with patch("llm_sandbox.nsjail.platform.machine", return_value="x86_64"):
        policy = _seccomp_policy(DEFAULT_BLOCKED_SYSCALLS)
    deny_rules = policy.split("ERRNO(1) { ", 1)[1].split(" }", 1)[0].split(", ")
    assert "umount" in deny_rules
    assert "umount2" not in deny_rules
    assert set(deny_rules) == (set(DEFAULT_BLOCKED_SYSCALLS) - {"umount2"}) | {"umount"}


@pytest.mark.parametrize("name", ["ptrace } DEFAULT ALLOW", "mount, read", "read\0", "#include /tmp/policy"])
def test_seccomp_source_injection_rejected(name: str) -> None:
    """Do not let a hand-built policy inject Kafel actions or include files."""
    with (
        patch("llm_sandbox.nsjail.platform.machine", return_value="x86_64"),
        pytest.raises(SecurityError, match="identifiers"),
    ):
        _seccomp_policy((name,))


def test_seccomp_unsupported_architecture_rejected() -> None:
    """A native-number policy cannot silently apply to an unverified ABI."""
    with (
        patch("llm_sandbox.nsjail.platform.machine", return_value="ppc64le"),
        pytest.raises(SecurityError, match="x86_64"),
    ):
        _seccomp_policy(("ptrace",))


def test_missing_executable_rejected() -> None:
    """Do not continue without the requested isolation backend."""
    with (
        patch("llm_sandbox.nsjail.platform.system", return_value="Linux"),
        patch("llm_sandbox.nsjail.shutil.which", return_value=None),
        pytest.raises(MissingDependencyError, match="install nsjail"),
    ):
        NsjailSession()


def test_unsupported_new_mount_api_fails_closed_without_retry(tmp_path: Path) -> None:
    """An older launcher rejecting the safe mount mode must not trigger a downgrade."""
    calls = tmp_path / "launcher-calls"
    launcher = tmp_path / "old-nsjail"
    launcher.write_text(
        f"#!{sys.executable}\n"
        "import pathlib, sys\n"
        f"with pathlib.Path({str(calls)!r}).open('a') as calls:\n"
        "    calls.write('invoked\\n')\n"
        "sys.stderr.write('unrecognized option: --experimental_mnt\\n')\n"
        "raise SystemExit(2)\n"
    )
    launcher.chmod(0o700)
    sandbox = NsjailSession(nsjail_path=str(launcher), skip_environment_setup=True)
    with pytest.raises(CommandFailedError, match="experimental_mnt"):
        sandbox.open()
    assert calls.read_text() == "invoked\n"
    assert not sandbox.is_open
    assert not sandbox.workspace.exists()


def test_non_linux_rejected() -> None:
    """Reject unsupported systems before trying to execute a binary."""
    with patch("llm_sandbox.nsjail.platform.system", return_value="Darwin"), pytest.raises(UnsupportedBackendError):
        NsjailSession()


def test_shutdown_gives_supervisor_time_to_reap_detached_child(tmp_path: Path) -> None:
    """A real mock supervisor must receive TERM and reap a child in another session."""
    marker = tmp_path / "reaped"
    script = (
        "import os, signal, subprocess, sys\n"
        "child = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(60)'], start_new_session=True)\n"
        "def stop(signum, frame):\n"
        "    child.kill()\n"
        "    child.wait()\n"
        "    open(sys.argv[1], 'w').write('reaped')\n"
        "    raise SystemExit(0)\n"
        "signal.signal(signal.SIGTERM, stop)\n"
        "print('ready', flush=True)\n"
        "signal.pause()\n"
    )
    with subprocess.Popen(  # noqa: S603
        [sys.executable, "-c", script, str(marker)], stdout=subprocess.PIPE, start_new_session=True
    ) as supervisor:
        assert supervisor.stdout is not None
        assert supervisor.stdout.readline() == b"ready\n"
        NsjailSession._terminate_process(supervisor)
        assert supervisor.returncode == 0
        assert marker.read_text() == "reaped"


def test_shutdown_escalates_only_after_grace_period() -> None:
    """A wedged supervisor cannot make shutdown wait indefinitely."""
    process = Mock(spec=subprocess.Popen)
    process.wait.side_effect = subprocess.TimeoutExpired("nsjail", 2)
    with patch("llm_sandbox.nsjail.LocalProcessSession._terminate_process") as hard_kill:
        NsjailSession._terminate_process(process)
    process.send_signal.assert_called_once_with(signal.SIGTERM)
    process.wait.assert_called_once_with(timeout=2)
    hard_kill.assert_called_once_with(process)


def test_shutdown_reaps_already_exited_supervisor() -> None:
    """A process exiting concurrently with cancellation is still reaped."""
    process = Mock(spec=subprocess.Popen)
    process.send_signal.side_effect = ProcessLookupError
    NsjailSession._terminate_process(process)
    process.wait.assert_called_once_with(timeout=2)


@pytest.mark.integration
@pytest.mark.skipif(platform.system() != "Linux" or shutil.which("nsjail") is None, reason="Requires Linux and nsjail")
def test_real_nsjail_filesystem_network_and_seccomp(tmp_path: Path) -> None:
    """Check enforced boundaries against the installed launcher, when present."""
    secret = tmp_path / "host-secret"
    secret.write_text("must stay outside")
    config = LocalSandboxConfig(
        seccomp=SeccompPolicy(blocked_syscalls=(*DEFAULT_BLOCKED_SYSCALLS, "getpriority")),
    )
    with socket.socket() as listener, NsjailSession(sandbox_config=config) as sandbox:
        listener.bind(("127.0.0.1", 0))
        listener.listen(1)
        port = listener.getsockname()[1]
        result = sandbox.run(
            "import ctypes, errno, os, socket\n"
            f"assert not os.path.exists({str(secret)!r})\n"
            "try:\n"
            "    open('/usr/sandbox-write-test', 'w').close()\n"
            "except OSError:\n"
            "    pass\n"
            "else:\n"
            "    raise AssertionError('runtime is writable')\n"
            "libc = ctypes.CDLL(None, use_errno=True)\n"
            "assert libc.getpriority(0, 0) == -1\n"
            "assert ctypes.get_errno() == errno.EPERM\n"
            f"if {platform.machine().lower() in {'x86_64', 'amd64'}!r}:\n"
            "    for number in (512, 547, 0x40000000 | 101):\n"
            "        ctypes.set_errno(0)\n"
            "        assert libc.syscall(number, -1, 0, 0, 0, 0, 0) == -1\n"
            "        assert ctypes.get_errno() == errno.EPERM\n"
            "sock = socket.socket()\n"
            "sock.settimeout(1)\n"
            "try:\n"
            f"    sock.connect(('127.0.0.1', {port}))\n"
            "except OSError:\n"
            "    pass\n"
            "else:\n"
            "    raise AssertionError('network is reachable')\n"
            "print('boundaries verified')\n"
        )
        assert result.exit_code == 0, result.text()
        assert "boundaries verified" in result.stdout


@pytest.mark.integration
@pytest.mark.skipif(platform.system() != "Linux" or shutil.which("nsjail") is None, reason="Requires Linux and nsjail")
def test_real_nsjail_supports_private_shared_memory_and_multiprocessing() -> None:
    """The default sandbox supports POSIX shm and semaphore-backed multiprocessing."""
    with NsjailSession() as sandbox:
        result = sandbox.run(
            "import multiprocessing\n"
            "from multiprocessing import shared_memory\n"
            "queue = multiprocessing.get_context('fork').Queue()\n"
            "def worker():\n"
            "    queue.put(42)\n"
            "process = multiprocessing.get_context('fork').Process(target=worker)\n"
            "process.start()\n"
            "assert queue.get(timeout=5) == 42\n"
            "process.join(timeout=5)\n"
            "assert process.exitcode == 0\n"
            "queue.close()\n"
            "queue.join_thread()\n"
            "memory = shared_memory.SharedMemory(create=True, size=16)\n"
            "try:\n"
            "    memory.buf[:4] = b'test'\n"
            "    assert bytes(memory.buf[:4]) == b'test'\n"
            "finally:\n"
            "    memory.close()\n"
            "    memory.unlink()\n"
            "open('/dev/shm/private-marker', 'w').write('first invocation')\n"
            "print('shared memory verified')\n"
        )
        assert result.exit_code == 0, result.text()
        assert "shared memory verified" in result.stdout
        second = sandbox.run("from pathlib import Path\nassert not Path('/dev/shm/private-marker').exists()")
        assert second.exit_code == 0, second.text()


@pytest.mark.integration
@pytest.mark.skipif(platform.system() != "Linux" or shutil.which("nsjail") is None, reason="Requires Linux and nsjail")
def test_real_nsjail_read_only_grant_includes_nested_host_mount() -> None:
    """A writable host bind mount inside an allowed runtime must become read-only."""
    tree_value = os.environ.get("LLM_SANDBOX_TEST_READONLY_TREE")
    source_value = os.environ.get("LLM_SANDBOX_TEST_WRITABLE_SOURCE")
    if not tree_value or not source_value:
        pytest.skip("Requires the controlled nested-mount fixture prepared by native-backend CI")
    tree = Path(tree_value).resolve(strict=True)
    source = Path(source_value).resolve(strict=True)
    host_marker = source / "marker"
    nested_marker = tree / "nested" / "marker"
    assert host_marker.samefile(nested_marker), "The fixture must bind the writable source into the runtime tree"
    mount_paths = {
        re.sub(r"\\([0-7]{3})", lambda match: chr(int(match.group(1), 8)), line.split()[4])
        for line in Path("/proc/self/mountinfo").read_text().splitlines()
    }
    assert str(tree / "nested") in mount_paths, "A plain directory would not test recursive read-only enforcement"
    assert os.access(host_marker, os.W_OK), "Host permissions must allow the attempted write"
    previous = host_marker.read_text()
    config = LocalSandboxConfig(read_only_paths=(*DEFAULT_READ_ONLY_PATHS, str(tree)))
    with NsjailSession(sandbox_config=config) as sandbox:
        result = sandbox.run(
            "import errno\n"
            "try:\n"
            f"    open({str(nested_marker)!r}, 'w').write('sandbox write escaped')\n"
            "except OSError as exc:\n"
            "    assert exc.errno == errno.EROFS, exc\n"
            "else:\n"
            "    raise AssertionError('nested host mount remained writable')\n"
        )
        assert result.exit_code == 0, result.text()
    assert host_marker.read_text() == previous


@pytest.mark.integration
@pytest.mark.skipif(platform.system() != "Linux" or shutil.which("nsjail") is None, reason="Requires Linux and nsjail")
def test_real_nsjail_timeout_kills_payload_with_parent_death_signal_disabled() -> None:
    """Timeout must terminate detached descendants even when prctl clears PDEATHSIG."""
    with NsjailSession(sandbox_config=LocalSandboxConfig(seccomp=None)) as sandbox:
        marker = sandbox.workspace / "escaped-timeout"
        output: list[str] = []
        command = (
            'exec /usr/bin/python3 -c "import ctypes, os, time; '
            "ctypes.CDLL(None).prctl(1, 0); "
            "pid = os.fork(); "
            "os.setsid() if pid == 0 else None; "
            "print('armed', flush=True); time.sleep(2); "
            "open('/sandbox/escaped-timeout', 'w').write('survived')\""
        )
        with pytest.raises(SandboxTimeoutError):
            sandbox._execute_with_timeout(sandbox.execute_command, command, timeout=1, on_stdout=output.append)
        assert "armed" in "".join(output)
        time.sleep(1.2)
        assert not marker.exists()
