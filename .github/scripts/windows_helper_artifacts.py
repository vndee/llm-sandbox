"""Collect pinned dependency notices and verify actual Windows wheel artifacts.

The CLI accepts operations, supported architectures and a commit ID, never paths.
Artifact locations are fixed within this checkout. License sources come from
locked Cargo metadata and the pinned Rust toolchain, with canonical containment
checks before reading them. Custom Cargo/Rust installation directories still work.
"""

# ruff: noqa: INP001 -- Standalone CI entry point.

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess  # nosec B404 # Pinned build tools use explicit argv and shell=False.
import zipfile
from pathlib import Path
from typing import Any

_ROOT = Path(__file__).resolve().parents[2]
_TARGETS = {"win_amd64": "x86_64-pc-windows-msvc", "win_arm64": "aarch64-pc-windows-msvc"}
_MINIMUM_CACHE_MANIFEST_PARTS = 3  # Cache owner, crate version or Git revision, and manifest.
_NOTICES = "THIRD_PARTY_NOTICES.txt"
_LICENSE_NAMES = ("LICENSE", "LICENCE", "COPYING", "COPYRIGHT", "NOTICE")
_SUPPLEMENTAL = {
    ("flatbuffers", "25.12.19"): "flatbuffers-25.12.19.txt",
    ("tracelogging", "1.2.3"): "tracelogging-1.2.3.txt",
    ("tracelogging_macros", "1.2.3"): "tracelogging-1.2.3.txt",
}


def within(path: Path, root: Path) -> Path:
    """Reject traversal and symlinks outside an explicitly allowed source root."""
    resolved = path.resolve()
    if not resolved.is_relative_to(root.resolve()):
        msg = f"Build path escapes its allowed root: {path}"
        raise ValueError(msg)
    return resolved


def repository_path(relative: str) -> Path:
    """Resolve a fixed artifact location without following links out of the checkout."""
    return within(_ROOT / relative, _ROOT)


def crate_scope(package: dict[str, Any], cargo_home: Path) -> tuple[Path, Path]:
    """Constrain registry crates and Git packages to their resolved source archive."""
    manifest = Path(package["manifest_path"]).resolve(strict=True)
    source = package.get("source")
    if source is None:
        if manifest != repository_path("native/windows-mxc/Cargo.toml"):
            msg = "Unexpected local Cargo package"
            raise ValueError(msg)
        return manifest.parent, _ROOT
    if source.startswith("registry+"):
        cache = within(cargo_home / "registry/src", cargo_home)
    elif source.startswith("git+"):
        cache = within(cargo_home / "git/checkouts", cargo_home)
    else:
        msg = f"Unsupported Cargo source: {source}"
        raise ValueError(msg)
    manifest = within(manifest, cache)
    relative = manifest.relative_to(cache)
    if len(relative.parts) < _MINIMUM_CACHE_MANIFEST_PARTS or manifest.name != "Cargo.toml":
        msg = f"Unexpected Cargo cache layout: {manifest}"
        raise ValueError(msg)
    scope = cache / relative.parts[0] / relative.parts[1]
    return manifest.parent, scope


def license_files(package: dict[str, Any], cargo_home: Path) -> list[Path]:
    """Find crate notices, allowing pinned supplements for incomplete archives."""
    root, scope = crate_scope(package, cargo_home)
    files: list[Path] = []
    if package.get("license_file"):
        files.append(within(root / package["license_file"], scope))
    for directory in (root, *root.parents):
        if not directory.is_relative_to(scope):
            break
        found = [
            within(entry, scope)
            for entry in directory.iterdir()
            if entry.is_file() and entry.name.upper().startswith(_LICENSE_NAMES)
        ]
        if found:
            files.extend(sorted(found))
            break
    supplement = _SUPPLEMENTAL.get((package["name"], package["version"]))
    if supplement:
        files.append(repository_path(f".github/licenses/{supplement}"))
        # Preserve the crate's own copyright header alongside the full license.
        files.append(within(root / "src/lib.rs", scope))
    if not files:
        msg = f"Missing license text for {package['name']} {package['version']}"
        raise ValueError(msg)
    return list(dict.fromkeys(files))


def tool_output(command: list[str]) -> str:
    """Read a fixed tool invocation; no CLI argument becomes a shell command."""
    # Callers use pinned tools and allowlisted target constants, never a command from the CLI.
    return subprocess.run(  # noqa: S603 # nosec B603
        command, cwd=_ROOT, shell=False, check=True, capture_output=True, text=True, encoding="utf-8", timeout=120
    ).stdout.strip()


def collect_notices(platform: str) -> None:
    """Write license text from the exact Cargo-resolved sources without fetching."""
    target = _TARGETS[platform]
    metadata = json.loads(
        tool_output([
            "cargo",
            "+1.93.0",
            "metadata",
            "--locked",
            "--offline",
            "--format-version",
            "1",
            "--filter-platform",
            target,
            "--manifest-path",
            "native/windows-mxc/Cargo.toml",
        ])
    )
    cargo_home = Path(os.environ.get("CARGO_HOME", Path.home() / ".cargo")).resolve(strict=True)
    rust_sysroot = Path(tool_output(["rustc", "+1.93.0", "--print", "sysroot"])).resolve(strict=True)
    sections = ["Windows MXC helper dependency notices. Includes build-time dependencies.\n"]
    rust_notices = within(rust_sysroot / "share/doc/rust/COPYRIGHT-library.html", rust_sysroot)
    sections.append("\nRust standard library notices (upstream HTML):\n" + rust_notices.read_text(encoding="utf-8"))
    for package in sorted(metadata["packages"], key=lambda item: (item["name"], item["version"])):
        sections.append(
            f"\n{'=' * 72}\n{package['name']} {package['version']}\n"
            f"License: {package.get('license')}\nSource: {package.get('source') or 'this repository'}\n"
        )
        for license_file in license_files(package, cargo_home):
            content = license_file.read_text(encoding="utf-8")
            if license_file.name == "lib.rs":
                # Only copyright/license comments precede the first Rust item.
                content = content.split("#!", 1)[0].split("//!", 1)[0]
            sections.append(f"\n--- {license_file.name} ---\n{content}\n")
    with repository_path(_NOTICES).open("w", encoding="utf-8") as output:
        output.write("".join(sections))


def digest(path: Path) -> str:
    """Return a SHA-256 digest for a distribution input or output."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def wheel_path() -> Path:
    """Require a single wheel with a filename safe for a requirements entry."""
    wheels = list(repository_path("dist").glob("*.whl"))
    if len(wheels) != 1 or not re.fullmatch(
        r"llm_sandbox-[A-Za-z0-9_.!+]+-py3-none-(?:any|win_amd64|win_arm64)\.whl", wheels[0].name
    ):
        msg = f"Expected exactly one llm-sandbox wheel: {wheels}"
        raise ValueError(msg)
    return within(wheels[0], repository_path("dist"))


def write_install_requirements() -> None:
    """Pin the local wheel's version and bytes for an offline hash-checked install."""
    wheel = wheel_path()
    version = wheel.name.split("-")[1]
    repository_path("dist/wheel-requirements.txt").write_text(
        f"llm-sandbox=={version} --hash=sha256:{digest(wheel)}\n", encoding="utf-8"
    )


def verify_wheel(platform: str, commit: str) -> None:
    """Validate wheel contents, then write checksums and a build receipt."""
    target = _TARGETS[platform]
    if not re.fullmatch(r"[0-9a-f]{40}", commit):
        msg = "Expected a full lowercase Git commit ID"
        raise ValueError(msg)
    binary = repository_path(f"native/windows-mxc/target/{target}/release/llm-sandbox-mxc.exe")
    notices = repository_path(_NOTICES)
    destination = repository_path("dist")
    wheel = wheel_path()
    if not wheel.name.endswith(f"-py3-none-{platform}.whl"):
        msg = f"Expected a py3-none-{platform} wheel: {wheel}"
        raise ValueError(msg)
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
    shutil.copyfile(repository_path("LICENSE"), repository_path("dist/LICENSE"))
    shutil.copyfile(notices, repository_path(f"dist/{_NOTICES}"))
    receipt = {
        "source_commit": commit,
        "platform": platform,
        "helper_sha256": digest(binary),
        "wheel": wheel.name,
        "wheel_sha256": digest(wheel),
        "cargo_lock_sha256": digest(repository_path("native/windows-mxc/Cargo.lock")),
        "native_enforcement": "Not established by building; see the Windows native enforcement workflow.",
    }
    with repository_path("dist/BUILD_RECEIPT.json").open("w", encoding="utf-8") as output:
        json.dump(receipt, output, indent=2)
        output.write("\n")
    write_install_requirements()
    checksums = "".join(
        f"{digest(within(path, destination))}  {path.name}\n"
        for path in sorted(destination.iterdir())
        if path.is_file() and path.name != "SHA256SUMS"
    )
    repository_path("dist/SHA256SUMS").write_text(checksums, encoding="utf-8")


def main() -> None:
    """Run one build-only packaging operation."""
    parser = argparse.ArgumentParser(description=__doc__)
    subcommands = parser.add_subparsers(dest="operation", required=True)
    notices_parser = subcommands.add_parser("notices")
    notices_parser.add_argument("platform", choices=tuple(_TARGETS))
    verify_parser = subcommands.add_parser("verify")
    verify_parser.add_argument("platform", choices=tuple(_TARGETS))
    verify_parser.add_argument("commit")
    subcommands.add_parser("install-requirements")
    args = parser.parse_args()
    if args.operation == "notices":
        collect_notices(args.platform)
    elif args.operation == "verify":
        verify_wheel(args.platform, args.commit)
    else:
        write_install_requirements()


if __name__ == "__main__":
    main()
