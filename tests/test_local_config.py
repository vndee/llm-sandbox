"""Validate local policies and reject mount/resource requests that cannot be honored."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest
from pydantic import ValidationError

from llm_sandbox.exceptions import MissingDependencyError, SecurityError
from llm_sandbox.local.config import LocalSandboxConfig
from llm_sandbox.local.launch import apply_resource_limits, read_only_mounts
from llm_sandbox.security.seccomp import SeccompPolicy


def test_default_policy_disables_network_and_enables_syscall_hardening() -> None:
    """Default local execution requires both namespace and syscall protections."""
    config = LocalSandboxConfig()
    assert config.network == "none"
    assert isinstance(config.seccomp, SeccompPolicy)
    assert "/" not in config.read_only_paths
    with pytest.raises(ValidationError, match="frozen"):
        config.network = "host"


@pytest.mark.parametrize(
    "path",
    [
        "/",
        "/proc",
        "/proc/1/root",
        "/sys/kernel",
        "/dev",
        "/tmp",
        "/tmp/runtime",
        "//usr",
        "usr",
        "/usr/../home",
        "/usr\0",
    ],
)
def test_unsafe_mount_destinations_are_rejected(path: str) -> None:
    """Never expose the host root, virtual filesystems or ambiguous mount paths."""
    with pytest.raises(ValidationError):
        LocalSandboxConfig(read_only_paths=(path,))


@pytest.mark.parametrize("path", ["python3", "/usr/../bin/python3", "/usr/bin/python\0"])
def test_python_executable_rejects_relative_traversal_or_nul(path: str) -> None:
    """An interpreter path must identify a single absolute sandbox executable."""
    with pytest.raises(ValidationError, match="absolute sandbox path"):
        LocalSandboxConfig(python_executable=path)


@pytest.mark.parametrize("environment", [{"": "x"}, {"A=B": "x"}, {"A\0": "x"}, {"A": "x\0"}])
def test_environment_rejects_invalid_keys_and_nuls(environment: dict[str, str]) -> None:
    """Do not accept environment entries that the process API cannot represent."""
    with pytest.raises(ValidationError):
        LocalSandboxConfig(env=environment)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("max_output_bytes", 0),
        ("max_artifact_bytes", 0),
        ("memory_limit_bytes", -1),
        ("memory_limit_bytes", "1024"),
        ("cpu_time_limit", 0),
        ("cpu_time_limit", 1.5),
        ("file_size_limit", True),
        ("open_files_limit", 15),
    ],
)
def test_resource_limits_require_strict_positive_values(field: str, value: Any) -> None:
    """Invalid or coerced limit values cannot accidentally weaken the boundary."""
    with pytest.raises(ValidationError):
        LocalSandboxConfig(**{field: value})


def test_unknown_network_policy_is_rejected() -> None:
    """Only explicitly defined network modes are accepted."""
    with pytest.raises(ValidationError):
        LocalSandboxConfig(network="unrestricted")  # type: ignore[arg-type]


@pytest.mark.parametrize("workdir", ["/", "relative", "/sandbox/../usr", "/proc/job", "/tmp/job"])
def test_workspace_cannot_replace_virtual_or_root_filesystems(workdir: str) -> None:
    """Writable workspace placement must preserve the namespace layout."""
    config = LocalSandboxConfig(read_only_paths=())
    with pytest.raises(SecurityError):
        read_only_mounts(config, workdir)


@pytest.mark.parametrize(("source", "workdir"), [("/usr", "/usr/job"), ("/sandbox/runtime", "/sandbox")])
def test_mounts_cannot_overlap_writable_workspace(source: str, workdir: str) -> None:
    """Overlapping read and write mounts cannot make read-only grants writable."""
    config = LocalSandboxConfig(read_only_paths=(source,))
    with pytest.raises(SecurityError, match="overlaps"):
        read_only_mounts(config, workdir)


def test_missing_custom_mount_does_not_disappear_silently() -> None:
    """Caller-supplied paths are required even though some distro defaults are optional."""
    config = LocalSandboxConfig(read_only_paths=("/opt/custom-runtime",))
    with patch.object(Path, "exists", return_value=False), pytest.raises(FileNotFoundError, match="does not exist"):
        read_only_mounts(config, "/sandbox")


def test_optional_distro_mount_can_be_absent() -> None:
    """Missing lib64 on a distribution need not disable other protections."""
    with patch.object(Path, "exists", return_value=False):
        assert read_only_mounts(LocalSandboxConfig(read_only_paths=("/lib64",)), "/sandbox") == ()


@pytest.mark.parametrize("target", ["/", "/proc", "/sys/kernel", "/dev", "/tmp/runtime"])
def test_resolved_mount_source_cannot_point_to_virtual_filesystems(target: str) -> None:
    """A symlink at an otherwise allowed path cannot expose host process or device state."""
    config = LocalSandboxConfig(read_only_paths=("/opt/runtime-link",))
    with (
        patch.object(Path, "exists", return_value=True),
        patch.object(Path, "resolve", return_value=Path(target)),
        pytest.raises(SecurityError, match="resolves to a host"),
    ):
        read_only_mounts(config, "/sandbox")


def test_non_regular_mount_source_is_rejected() -> None:
    """Sockets and special files must not be passed through as runtime mounts."""
    config = LocalSandboxConfig(read_only_paths=("/opt/runtime-socket",))
    with (
        patch.object(Path, "exists", return_value=True),
        patch.object(Path, "resolve", return_value=Path("/opt/runtime-socket")),
        patch.object(Path, "is_dir", return_value=False),
        patch.object(Path, "is_file", return_value=False),
        pytest.raises(SecurityError, match="regular file or directory"),
    ):
        read_only_mounts(config, "/sandbox")


def test_host_network_adds_only_narrow_dns_and_trust_paths() -> None:
    """Network opt-in supplies DNS and trust configuration without exposing all of /etc."""
    with (
        patch.object(Path, "exists", return_value=True),
        patch.object(Path, "resolve", autospec=True, side_effect=lambda path, **_: path),
        patch.object(Path, "is_file", return_value=True),
    ):
        offline = read_only_mounts(LocalSandboxConfig(read_only_paths=()), "/sandbox")
        online = read_only_mounts(LocalSandboxConfig(network="host", read_only_paths=()), "/sandbox")
    assert offline == ()
    destinations = {destination for _, destination in online}
    assert {"/etc/resolv.conf", "/etc/hosts", "/etc/ssl/certs", "/etc/pki/tls/certs"} <= destinations
    assert "/etc" not in destinations
    assert "/etc/shadow" not in destinations
    for private_path in (Path("/etc/ssl/private"), Path("/etc/pki/tls/private")):
        assert not any(private_path.is_relative_to(destination) for destination in destinations)


def test_limits_use_exact_hard_and_soft_values_without_unit_conversion() -> None:
    """Address-space and file-size values are bytes, with matching hard and soft caps."""
    config = LocalSandboxConfig(
        memory_limit_bytes=123456789,
        cpu_time_limit=3,
        file_size_limit=7654321,
        open_files_limit=64,
    )
    original = ["/usr/bin/bwrap", "--", "/bin/true"]
    with patch("llm_sandbox.local.launch.shutil.which", return_value="/usr/bin/prlimit"):
        limited = apply_resource_limits(original, config)
    assert limited == [
        "/usr/bin/prlimit",
        "--as=123456789:123456789",
        "--cpu=3:3",
        "--fsize=7654321:7654321",
        "--nofile=64:64",
        "--",
        *original,
    ]
    assert original == ["/usr/bin/bwrap", "--", "/bin/true"]


@pytest.mark.skipif(os.name != "posix", reason="Requires executable POSIX symlinks")
@pytest.mark.parametrize("path_entry", ["bin", "."])
def test_relative_resource_helper_cannot_be_shadowed_by_workspace(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, path_entry: str
) -> None:
    """Changing to a payload-owned cwd must not replace the trusted limit helper."""
    caller = tmp_path / "caller"
    workspace = tmp_path / "workspace"
    (caller / path_entry).mkdir(parents=True, exist_ok=True)
    (workspace / path_entry).mkdir(parents=True, exist_ok=True)
    # Both fixtures ignore arguments and print markers; neither executes a workload.
    trusted = tmp_path / "trusted-prlimit"
    trusted.write_text("#!/bin/sh\nprintf 'trusted helper\\n'\n")
    trusted.chmod(0o700)
    (caller / path_entry / "prlimit").symlink_to(trusted)
    shadow = workspace / path_entry / "prlimit"
    shadow.write_text("#!/bin/sh\nprintf 'workspace helper\\n'\n")
    shadow.chmod(0o700)
    monkeypatch.chdir(caller)
    monkeypatch.setenv("PATH", path_entry)

    argv = apply_resource_limits(["/bin/true"], LocalSandboxConfig(open_files_limit=64))
    result = subprocess.run(  # noqa: S603 # Only the two trusted marker fixtures can execute.
        argv,
        cwd=workspace,
        env={"PATH": "/usr/bin:/bin"},
        shell=False,
        capture_output=True,
        text=True,
        check=True,
        timeout=5,
    )

    assert result.stdout == "trusted helper\n"
    assert argv[0] == str(trusted.resolve())


def test_requested_limits_fail_closed_without_prlimit() -> None:
    """A missing limit helper cannot silently turn restricted execution into unrestricted execution."""
    config = LocalSandboxConfig(memory_limit_bytes=1024)
    with (
        patch("llm_sandbox.local.launch.shutil.which", return_value=None),
        pytest.raises(MissingDependencyError, match="prlimit"),
    ):
        apply_resource_limits(["/usr/bin/bwrap"], config)


def test_no_limit_helper_required_when_no_limits_are_requested() -> None:
    """The optional resource layer does not add dependencies unless configured."""
    with patch("llm_sandbox.local.launch.shutil.which", return_value=None) as find_executable:
        assert apply_resource_limits(["bwrap"], LocalSandboxConfig()) == ["bwrap"]
    find_executable.assert_not_called()
