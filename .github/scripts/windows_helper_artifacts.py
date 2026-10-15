"""Collect pinned dependency notices and verify actual Windows wheel artifacts."""

# ruff: noqa: INP001 -- Standalone CI entry point.

import argparse
import hashlib
import json
import shutil
import zipfile
from pathlib import Path
from typing import Any

_ROOT = Path(__file__).resolve().parents[2]
_LICENSE_NAMES = ("LICENSE", "LICENCE", "COPYING", "COPYRIGHT", "NOTICE")
_SUPPLEMENTAL = {
    ("flatbuffers", "25.12.19"): "flatbuffers-25.12.19.txt",
    ("tracelogging", "1.2.3"): "tracelogging-1.2.3.txt",
    ("tracelogging_macros", "1.2.3"): "tracelogging-1.2.3.txt",
}


def license_files(package: dict[str, Any]) -> list[Path]:
    """Find crate notices, allowing pinned supplements for incomplete archives."""
    manifest = Path(package["manifest_path"])
    root = manifest.parent
    files: list[Path] = []
    if package.get("license_file"):
        files.append(root / package["license_file"])
    for directory in (root, *list(root.parents)[:5]):
        found = [
            entry for entry in directory.iterdir() if entry.is_file() and entry.name.upper().startswith(_LICENSE_NAMES)
        ]
        if found:
            files.extend(sorted(found))
            break
    supplement = _SUPPLEMENTAL.get((package["name"], package["version"]))
    if supplement:
        files.append(_ROOT / ".github" / "licenses" / supplement)
        # Preserve the crate's own copyright header alongside the full license.
        files.append(root / "src" / "lib.rs")
    if not files:
        msg = f"Missing license text for {package['name']} {package['version']}"
        raise ValueError(msg)
    return list(dict.fromkeys(files))


def collect_notices(metadata_path: Path, output: Path, rust_sysroot: Path) -> None:
    """Write license text from the exact Cargo-resolved sources without fetching."""
    metadata = json.loads(metadata_path.read_text(encoding="utf-8-sig"))
    sections = ["Windows MXC helper dependency notices. Includes build-time dependencies.\n"]
    rust_notices = rust_sysroot / "share/doc/rust/COPYRIGHT-library.html"
    sections.append("\nRust standard library notices (upstream HTML):\n" + rust_notices.read_text(encoding="utf-8"))
    for package in sorted(metadata["packages"], key=lambda item: (item["name"], item["version"])):
        sections.append(
            f"\n{'=' * 72}\n{package['name']} {package['version']}\n"
            f"License: {package.get('license')}\nSource: {package.get('source') or 'this repository'}\n"
        )
        for license_file in license_files(package):
            content = license_file.read_text(encoding="utf-8")
            if license_file.name == "lib.rs":
                # Only copyright/license comments precede the first Rust item.
                content = content.split("#!", 1)[0].split("//!", 1)[0]
            sections.append(f"\n--- {license_file.name} ---\n{content}\n")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("".join(sections), encoding="utf-8")


def digest(path: Path) -> str:
    """Return a SHA-256 digest for a distribution input or output."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify_wheel(binary: Path, notices: Path, platform: str, commit: str, destination: Path) -> None:
    """Validate wheel contents, then write checksums and a build receipt."""
    wheels = list(destination.glob("*.whl"))
    if len(wheels) != 1 or not wheels[0].name.endswith(f"-py3-none-{platform}.whl"):
        msg = f"Expected exactly one py3-none-{platform} wheel: {wheels}"
        raise ValueError(msg)
    wheel = wheels[0]
    with zipfile.ZipFile(wheel) as archive:
        if archive.read("llm_sandbox/bin/llm-sandbox-mxc.exe") != binary.read_bytes():
            msg = "Wheel helper differs from the built executable"
            raise ValueError(msg)
        if archive.read("llm_sandbox/bin/THIRD_PARTY_NOTICES.txt") != notices.read_bytes():
            msg = "Wheel dependency notices differ from the collected licenses"
            raise ValueError(msg)
        metadata_name = next(name for name in archive.namelist() if name.endswith(".dist-info/WHEEL"))
        metadata = archive.read(metadata_name).decode()
        if "Root-Is-Purelib: false" not in metadata or f"Tag: py3-none-{platform}" not in metadata:
            msg = "Native helper wheel has incorrect installation metadata"
            raise ValueError(msg)
    shutil.copyfile(_ROOT / "LICENSE", destination / "LICENSE")
    if notices.resolve() != (destination / "THIRD_PARTY_NOTICES.txt").resolve():
        shutil.copyfile(notices, destination / "THIRD_PARTY_NOTICES.txt")
    receipt = {
        "source_commit": commit,
        "platform": platform,
        "helper_sha256": digest(binary),
        "wheel": wheel.name,
        "wheel_sha256": digest(wheel),
        "cargo_lock_sha256": digest(_ROOT / "native/windows-mxc/Cargo.lock"),
        "native_enforcement": "Not established by building; see the Windows native enforcement workflow.",
    }
    (destination / "BUILD_RECEIPT.json").write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    checksums = "".join(
        f"{digest(path)}  {path.name}\n"
        for path in sorted(destination.iterdir())
        if path.is_file() and path.name != "SHA256SUMS"
    )
    (destination / "SHA256SUMS").write_text(checksums, encoding="utf-8")


def main() -> None:
    """Run one build-only packaging operation."""
    parser = argparse.ArgumentParser(description=__doc__)
    subcommands = parser.add_subparsers(dest="operation", required=True)
    notices_parser = subcommands.add_parser("notices")
    notices_parser.add_argument("metadata", type=Path)
    notices_parser.add_argument("output", type=Path)
    notices_parser.add_argument("rust_sysroot", type=Path)
    verify_parser = subcommands.add_parser("verify")
    verify_parser.add_argument("binary", type=Path)
    verify_parser.add_argument("notices", type=Path)
    verify_parser.add_argument("platform", choices=("win_amd64", "win_arm64"))
    verify_parser.add_argument("commit")
    verify_parser.add_argument("destination", type=Path)
    args = parser.parse_args()
    if args.operation == "notices":
        collect_notices(args.metadata, args.output, args.rust_sysroot)
    else:
        verify_wheel(args.binary, args.notices, args.platform, args.commit, args.destination)


if __name__ == "__main__":
    main()
