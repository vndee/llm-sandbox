"""Opt-in packaging of a prebuilt, architecture-checked Windows MXC helper."""

import os
import struct
from pathlib import Path
from typing import Any

from hatchling.builders.hooks.plugin.interface import BuildHookInterface

_PE_SIGNATURE = b"PE\0\0"
_PE32_PLUS_MAGIC = 0x20B
_EXECUTABLE_IMAGE = 0x0002
_DLL_IMAGE = 0x2000
_PLATFORMS = {0x8664: "win_amd64", 0xAA64: "win_arm64"}
_DOS_HEADER_SIZE = 64
_COFF_HEADER_SIZE = 20
_OPTIONAL_HEADER_MIN_SIZE = 112


def windows_helper_platform(binary: Path) -> str:
    """Validate a Windows PE32+ executable and return its supported wheel platform.

    This validates the binary's format and machine type, not its provenance.
    Release builders must supply the helper built from the reviewed source.
    """
    with binary.open("rb") as source:
        header = source.read(_DOS_HEADER_SIZE)
        if len(header) != _DOS_HEADER_SIZE or header[:2] != b"MZ":
            msg = "MXC helper must be a Windows PE executable"
            raise ValueError(msg)
        offset = struct.unpack_from("<I", header, 60)[0]
        if offset < _DOS_HEADER_SIZE or offset > binary.stat().st_size - len(_PE_SIGNATURE) - _COFF_HEADER_SIZE:
            msg = "MXC helper has an invalid PE header offset"
            raise ValueError(msg)
        source.seek(offset)
        if source.read(len(_PE_SIGNATURE)) != _PE_SIGNATURE:
            msg = "MXC helper has an invalid PE signature"
            raise ValueError(msg)
        coff_header = source.read(_COFF_HEADER_SIZE)
        machine, sections, _, _, _, optional_size, characteristics = struct.unpack("<HHIIIHH", coff_header)
        platform = _PLATFORMS.get(machine)
        if platform is None:
            msg = f"MXC helper architecture {machine:#x} is unsupported; build AMD64 or ARM64"
            raise ValueError(msg)
        if not sections or not characteristics & _EXECUTABLE_IMAGE or characteristics & _DLL_IMAGE:
            msg = "MXC helper must be an executable image, not a DLL or object file"
            raise ValueError(msg)
        optional_header = source.read(optional_size)
        if (
            optional_size < _OPTIONAL_HEADER_MIN_SIZE
            or len(optional_header) != optional_size
            or struct.unpack_from("<H", optional_header)[0] != _PE32_PLUS_MAGIC
        ):
            msg = "MXC helper must have a complete 64-bit PE32+ optional header"
            raise ValueError(msg)
        return platform


class CustomBuildHook(BuildHookInterface):
    """Bundle only an explicitly supplied helper, without running or downloading it."""

    def initialize(self, version: str, build_data: dict[str, Any]) -> None:  # noqa: ARG002
        """Give helper-containing wheels the platform of the actual PE binary."""
        value = os.environ.get("LLM_SANDBOX_MXC_BINARY")
        if value is None:
            return
        if not value.strip():
            msg = "LLM_SANDBOX_MXC_BINARY must name a prebuilt helper executable"
            raise ValueError(msg)
        binary = Path(value).expanduser().resolve(strict=True)
        platform = windows_helper_platform(binary)
        notices_value = os.environ.get("LLM_SANDBOX_MXC_NOTICES", "")
        if not notices_value.strip():
            msg = "Bundling MXC requires LLM_SANDBOX_MXC_NOTICES with its dependency license notices"
            raise ValueError(msg)
        notices = Path(notices_value).expanduser().resolve(strict=True)
        if not notices.is_file() or not notices.read_text(encoding="utf-8").strip():
            msg = "MXC dependency license notices must be a nonempty UTF-8 text file"
            raise ValueError(msg)
        build_data["pure_python"] = False
        build_data["tag"] = f"py3-none-{platform}"
        build_data["force_include"][str(binary)] = "llm_sandbox/bin/llm-sandbox-mxc.exe"
        build_data["force_include"][str(notices)] = "llm_sandbox/bin/THIRD_PARTY_NOTICES.txt"
