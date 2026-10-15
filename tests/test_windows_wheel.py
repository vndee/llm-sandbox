"""Build real wheel archives to verify opt-in Windows helper distribution."""

import os
import shutil
import struct
import subprocess
import sys
import tarfile
import zipfile
from pathlib import Path

import pytest

from hatch_build import windows_helper_platform


def pe_executable(machine: int = 0x8664, *, characteristics: int = 0x0002, magic: int = 0x20B) -> bytes:
    """Create a non-runnable PE header fixture; tests never execute these bytes."""
    header = bytearray(64)
    header[:2] = b"MZ"
    struct.pack_into("<I", header, 60, 64)
    optional = bytearray(112)
    struct.pack_into("<H", optional, 0, magic)
    coff = struct.pack("<HHIIIHH", machine, 1, 0, 0, 0, len(optional), characteristics)
    return bytes(header) + b"PE\0\0" + coff + optional


@pytest.fixture
def project(tmp_path: Path) -> Path:
    """Create a small distributable project using the production build hook."""
    source = tmp_path / "project"
    source.mkdir()
    package = source / "llm_sandbox"
    package.mkdir()
    (package / "__init__.py").write_text('"""Wheel fixture."""\n', encoding="utf-8")
    shutil.copyfile(Path(__file__).parents[1] / "hatch_build.py", source / "hatch_build.py")
    (source / "pyproject.toml").write_text(
        '[project]\nname = "llm-sandbox"\nversion = "0.0.0"\nrequires-python = ">=3.10"\n'
        '[build-system]\nrequires = ["hatchling"]\nbuild-backend = "hatchling.build"\n'
        '[tool.hatch.build.targets.wheel]\npackages = ["llm_sandbox"]\n'
        'exclude = ["/llm_sandbox/bin/*.exe"]\n'
        '[tool.hatch.build.targets.wheel.hooks.custom]\npath = "hatch_build.py"\n',
        encoding="utf-8",
    )
    return source


def build_wheel(project: Path, binary: Path | None) -> Path:
    """Build through Hatch's actual frontend without installing dependencies."""
    environment = os.environ.copy()
    environment.pop("LLM_SANDBOX_MXC_BINARY", None)
    environment.pop("LLM_SANDBOX_MXC_NOTICES", None)
    if binary is not None:
        environment["LLM_SANDBOX_MXC_BINARY"] = str(binary)
        notices = project.parent / "THIRD_PARTY_NOTICES.txt"
        notices.write_bytes(b"Example dependency license notice\n")
        environment["LLM_SANDBOX_MXC_NOTICES"] = str(notices)
    result = subprocess.run(  # noqa: S603 # Fixed build tool and a test-generated project.
        [sys.executable, "-m", "hatchling", "build", "-t", "wheel"],
        cwd=project,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
        timeout=30,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    return next((project / "dist").glob("*.whl"))


@pytest.mark.parametrize(("machine", "platform"), [(0x8664, "win_amd64"), (0xAA64, "win_arm64")])
def test_platform_wheel_contains_exact_prebuilt_helper(project: Path, machine: int, platform: str) -> None:
    """The PE machine, including when cross-packaging, determines the wheel tag."""
    binary = project.parent / "helper.exe"
    payload = pe_executable(machine)
    binary.write_bytes(payload)
    wheel = build_wheel(project, binary)
    assert wheel.name.endswith(f"-py3-none-{platform}.whl")
    with zipfile.ZipFile(wheel) as archive:
        assert archive.read("llm_sandbox/bin/llm-sandbox-mxc.exe") == payload
        assert archive.read("llm_sandbox/bin/THIRD_PARTY_NOTICES.txt") == b"Example dependency license notice\n"
        metadata = archive.read("llm_sandbox-0.0.0.dist-info/WHEEL").decode()
        assert "Root-Is-Purelib: false" in metadata
        assert f"Tag: py3-none-{platform}" in metadata
        assert archive.read("llm_sandbox/__init__.py")


def test_generic_wheel_stays_pure_and_omits_leftover_helpers(project: Path) -> None:
    """Ordinary builds remain portable even if a developer has a local executable."""
    binaries = project / "llm_sandbox" / "bin"
    binaries.mkdir()
    (binaries / "llm-sandbox-mxc.exe").write_bytes(pe_executable())
    wheel = build_wheel(project, None)
    assert wheel.name.endswith("-py3-none-any.whl")
    with zipfile.ZipFile(wheel) as archive:
        assert not any(name.endswith(".exe") for name in archive.namelist())
        assert "Root-Is-Purelib: true" in archive.read("llm_sandbox-0.0.0.dist-info/WHEEL").decode()


@pytest.mark.parametrize(
    "payload",
    [
        b"not an executable",
        b"MZ" + bytes(62),
        pe_executable(0x014C),
        pe_executable(characteristics=0x2002),
        pe_executable(characteristics=0),
        pe_executable(magic=0x010B),
        pe_executable()[:-1],
    ],
)
def test_invalid_or_unsupported_helpers_are_rejected(tmp_path: Path, payload: bytes) -> None:
    """Malformed headers, DLLs and 32-bit images cannot produce misleading wheels."""
    binary = tmp_path / "helper.exe"
    binary.write_bytes(payload)
    with pytest.raises(ValueError, match="MXC helper"):
        windows_helper_platform(binary)


def test_missing_helper_is_rejected(tmp_path: Path) -> None:
    """An explicitly requested helper must never disappear from the distribution silently."""
    with pytest.raises(FileNotFoundError):
        windows_helper_platform(tmp_path / "missing.exe")


def test_native_wheel_requires_dependency_notices(project: Path) -> None:
    """A builder cannot accidentally distribute the linked SDK without license notices."""
    binary = project.parent / "helper.exe"
    binary.write_bytes(pe_executable())
    environment = os.environ.copy()
    environment["LLM_SANDBOX_MXC_BINARY"] = str(binary)
    environment.pop("LLM_SANDBOX_MXC_NOTICES", None)
    result = subprocess.run(  # noqa: S603 # Fixed build tool and a test-generated project.
        [sys.executable, "-m", "hatchling", "build", "-t", "wheel"],
        cwd=project,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
        timeout=30,
    )
    assert result.returncode != 0
    assert "LLM_SANDBOX_MXC_NOTICES" in result.stderr
    assert not list((project / "dist").glob("*.whl"))


def test_source_distribution_never_requires_a_native_binary(project: Path) -> None:
    """The wheel-only hook must not burden source builders or read a stale helper path."""
    environment = os.environ.copy()
    environment["LLM_SANDBOX_MXC_BINARY"] = str(project / "missing.exe")
    result = subprocess.run(  # noqa: S603 # Fixed build tool and a test-generated project.
        [sys.executable, "-m", "hatchling", "build", "-t", "sdist"],
        cwd=project,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
        timeout=30,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    with tarfile.open(next((project / "dist").glob("*.tar.gz"))) as archive:
        names = archive.getnames()
        assert "llm_sandbox-0.0.0/hatch_build.py" in names
        assert not any(name.endswith(".exe") for name in names)
