"""Verify artifact provenance, license boundaries and locked wheel installation inputs."""

import hashlib
import importlib.util
import json
import sys
import zipfile
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest


@pytest.fixture
def artifacts(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> ModuleType:
    """Load the actual build entrypoint against an isolated checkout."""
    script = Path(__file__).parents[1] / ".github/scripts/windows_helper_artifacts.py"
    spec = importlib.util.spec_from_file_location("windows_helper_artifacts", script)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    root = tmp_path / "checkout"
    root.mkdir()
    monkeypatch.setattr(module, "_ROOT", root)
    write(root / "LICENSE", "Repository MIT notice")
    write(root / "native/windows-mxc/Cargo.toml", "[package]")
    write(root / "native/windows-mxc/Cargo.lock", "Locked dependency versions")
    return module


def write(path: Path, content: str) -> Path:
    """Create a source fixture without platform newline translation."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content.encode())
    return path


def package(root: Path, source: str | None = "registry+https://example.invalid/index", **extra: Any) -> dict[str, Any]:
    """Describe one real on-disk Cargo manifest."""
    return {
        "name": root.name,
        "version": "1.0.0",
        "source": source,
        "license": "MIT",
        "manifest_path": str(write(root / "Cargo.toml", "[package]")),
        **extra,
    }


def symlink(link: Path, target: Path) -> None:
    """Exercise canonical paths where the host grants symlink creation."""
    try:
        link.symlink_to(target, target_is_directory=target.is_dir())
    except OSError as error:
        if getattr(error, "winerror", None) == 1314:
            pytest.skip("This Windows account cannot create symlinks")
        raise


def wheel_fixture(artifacts: ModuleType, platform: str = "win_amd64", **overrides: str) -> Path:
    """Create an archive and its independently built inputs for verification."""
    target = artifacts._TARGETS[platform]
    write(artifacts._ROOT / f"native/windows-mxc/target/{target}/release/llm-sandbox-mxc.exe", "helper bytes")
    write(artifacts._ROOT / "THIRD_PARTY_NOTICES.txt", "Dependency notices")
    destination: Path = artifacts._ROOT / "dist"
    destination.mkdir(exist_ok=True)
    wheel = destination / f"llm_sandbox-1.2.3-py3-none-{platform}.whl"
    with zipfile.ZipFile(wheel, "w") as archive:
        archive.writestr("llm_sandbox/bin/llm-sandbox-mxc.exe", overrides.get("binary", "helper bytes"))
        archive.writestr("llm_sandbox/bin/THIRD_PARTY_NOTICES.txt", overrides.get("notices", "Dependency notices"))
        archive.writestr(
            "llm_sandbox-1.2.3.dist-info/WHEEL",
            overrides.get("metadata", f"Root-Is-Purelib: false\nTag: py3-none-{platform}\n"),
        )
    return wheel


def test_collects_custom_cargo_and_rust_roots_without_reading_above_crates(
    artifacts: ModuleType,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Custom installation roots and nested Git crates retain all required notices."""
    cargo = tmp_path / "custom cargo cache"
    registry = cargo / "registry/src/index/crate-1.0"
    checkout = cargo / "git/checkouts/project/revision"
    packages = [
        package(registry),
        package(checkout / "nested/crate", "git+https://example.invalid/repository", license_file="../../LICENSE"),
        package(artifacts._ROOT / "native/windows-mxc", None),
    ]
    write(registry / "LICENSE", "Registry notice")
    write(checkout / "LICENSE", "Git workspace notice")
    rust = tmp_path / "custom rust toolchain"
    write(rust / "share/doc/rust/COPYRIGHT-library.html", "Rust library notice")
    commands: list[list[str]] = []

    def output(command: list[str]) -> str:
        commands.append(command)
        return json.dumps({"packages": packages}) if command[0] == "cargo" else str(rust)

    monkeypatch.setenv("CARGO_HOME", str(cargo))
    monkeypatch.setattr(artifacts, "tool_output", output)
    artifacts.collect_notices("win_arm64")
    notices = (artifacts._ROOT / "THIRD_PARTY_NOTICES.txt").read_text()
    for expected in ("Registry notice", "Git workspace notice", "Repository MIT notice", "Rust library notice"):
        assert expected in notices
    assert "--locked" in commands[0]
    assert "--offline" in commands[0]
    assert "aarch64-pc-windows-msvc" in commands[0]
    assert commands[1] == ["rustc", "+1.93.0", "--print", "sysroot"]


@pytest.mark.parametrize("absolute", [False, True])
def test_declared_license_cannot_read_outside_its_crate(
    artifacts: ModuleType,
    tmp_path: Path,
    absolute: bool,
) -> None:
    """Both absolute and parent-relative license escapes fail before copying secrets."""
    cargo = tmp_path / "cargo"
    root = cargo / "registry/src/index/crate"
    outside = write(root.parent / "private.txt", "Do not distribute")
    crate = package(root, license_file=str(outside) if absolute else "../private.txt")
    with pytest.raises(ValueError, match="escapes"):
        artifacts.license_files(crate, cargo)


def test_missing_license_does_not_search_other_cached_crates(artifacts: ModuleType, tmp_path: Path) -> None:
    """A parent cache notice must not silently stand in for an unlicensed crate."""
    cargo = tmp_path / "cargo"
    root = cargo / "registry/src/index/crate"
    crate = package(root)
    write(root.parent / "LICENSE", "Unrelated cache notice")
    with pytest.raises(ValueError, match="Missing license"):
        artifacts.license_files(crate, cargo)


def test_license_symlink_cannot_escape_its_crate(artifacts: ModuleType, tmp_path: Path) -> None:
    """A plausible license filename cannot redirect reads to a private host file."""
    cargo = tmp_path / "cargo"
    root = cargo / "registry/src/index/crate"
    crate = package(root)
    outside = write(tmp_path / "private.txt", "Do not distribute")
    symlink(root / "LICENSE", outside)
    with pytest.raises(ValueError, match="escapes"):
        artifacts.license_files(crate, cargo)


def test_manifest_cannot_use_adjacent_cache_with_same_prefix(artifacts: ModuleType, tmp_path: Path) -> None:
    """Containment compares path components, not a vulnerable string prefix."""
    cargo = tmp_path / "cargo"
    cargo.mkdir()
    crate = package(tmp_path / "cargo-backup/registry/src/index/crate")
    with pytest.raises(ValueError, match="escapes"):
        artifacts.license_files(crate, cargo)


def test_unknown_local_package_is_rejected(artifacts: ModuleType, tmp_path: Path) -> None:
    """The only local source accepted is the checked-out helper manifest."""
    crate = package(tmp_path / "foreign-local-package", None)
    with pytest.raises(ValueError, match="Unexpected local"):
        artifacts.license_files(crate, tmp_path)


@pytest.mark.parametrize("platform", ["win_amd64", "win_arm64"])
def test_verified_bundle_records_exact_input_and_output_hashes(artifacts: ModuleType, platform: str) -> None:
    """Receipts and checksums describe the actual helper, lockfile and wheel bytes."""
    wheel = wheel_fixture(artifacts, platform)
    artifacts.verify_wheel(platform, "a" * 40)
    receipt = json.loads((wheel.parent / "BUILD_RECEIPT.json").read_text())
    assert receipt["source_commit"] == "a" * 40
    assert receipt["platform"] == platform
    assert receipt["helper_sha256"] == hashlib.sha256(b"helper bytes").hexdigest()
    assert receipt["cargo_lock_sha256"] == hashlib.sha256(b"Locked dependency versions").hexdigest()
    assert receipt["wheel_sha256"] == hashlib.sha256(wheel.read_bytes()).hexdigest()
    requirements = (wheel.parent / "wheel-requirements.txt").read_text()
    assert requirements == f"llm-sandbox==1.2.3 --hash=sha256:{receipt['wheel_sha256']}\n"
    for line in (wheel.parent / "SHA256SUMS").read_text().splitlines():
        expected, name = line.split("  ")
        assert expected == hashlib.sha256((wheel.parent / name).read_bytes()).hexdigest()


@pytest.mark.parametrize(
    ("override", "message"),
    [
        ({"binary": "different helper"}, "Wheel helper differs"),
        ({"notices": "missing copyright"}, "Wheel dependency notices differ"),
        ({"metadata": "Root-Is-Purelib: true\nTag: py3-none-any\n"}, "incorrect installation metadata"),
    ],
)
def test_mismatched_artifacts_never_receive_a_receipt(
    artifacts: ModuleType,
    override: dict[str, str],
    message: str,
) -> None:
    """Successful ZIP creation alone does not establish native wheel provenance."""
    wheel_fixture(artifacts, **override)
    with pytest.raises(ValueError, match=message):
        artifacts.verify_wheel("win_amd64", "a" * 40)
    assert not (artifacts._ROOT / "dist/BUILD_RECEIPT.json").exists()


def test_symlinked_output_cannot_overwrite_host_file(artifacts: ModuleType, tmp_path: Path) -> None:
    """Fixed output names also reject an existing link outside the checkout."""
    wheel_fixture(artifacts)
    outside = write(tmp_path / "private.txt", "Original host contents")
    symlink(artifacts._ROOT / "dist/BUILD_RECEIPT.json", outside)
    with pytest.raises(ValueError, match="escapes"):
        artifacts.verify_wheel("win_amd64", "a" * 40)
    assert outside.read_text() == "Original host contents"


def test_multiple_wheels_cannot_create_an_ambiguous_install_manifest(artifacts: ModuleType) -> None:
    """The installer must not select a stale wheel from a previous build."""
    wheel_fixture(artifacts, "win_amd64")
    wheel_fixture(artifacts, "win_arm64")
    with pytest.raises(ValueError, match="exactly one"):
        artifacts.write_install_requirements()


@pytest.mark.parametrize("commit", ["main", "abc123", "../receipt", "a" * 40 + "\n"])
def test_receipt_requires_exact_source_commit(artifacts: ModuleType, commit: str) -> None:
    """A branch name or malformed value cannot be mistaken for reproducible provenance."""
    with pytest.raises(ValueError, match="full lowercase Git commit"):
        artifacts.verify_wheel("win_amd64", commit)


@pytest.mark.parametrize("arguments", [["notices", "../../private"], ["verify", "win_amd64", "a" * 40, "outside"]])
def test_cli_has_no_arbitrary_file_path_inputs(
    artifacts: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
    arguments: list[str],
) -> None:
    """Path-like architectures and obsolete destination arguments fail in the parser."""
    monkeypatch.setattr(sys, "argv", ["windows_helper_artifacts.py", *arguments])
    with pytest.raises(SystemExit) as error:
        artifacts.main()
    assert error.value.code == 2
