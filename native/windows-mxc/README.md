# Native Windows MXC helper

`llm-sandbox-mxc.exe` is the native launcher for the Python Windows MXC
backend. It directly invokes Microsoft MXC's `BaseContainerRunner` after an
actual PSEC create/close probe succeeds. It never invokes the backend
dispatcher, AppContainer, the transitional SBOX API, host ACL provisioning,
or an unsandboxed command as a fallback.

The SDK is pinned to Microsoft MXC revision
[`6cd3d58f05d3447e67109cfb75e042803b843ca4`](https://github.com/microsoft/mxc/tree/6cd3d58f05d3447e67109cfb75e042803b843ca4).
Its API probe determines availability; a Windows version string alone does
not establish support. Some Windows hosts, including CI hosts, cannot run PSEC.

## Build and validation

On Windows with the Rust MSVC toolchain and Visual Studio C++ build tools:

```powershell
cd native/windows-mxc
$env:RUSTFLAGS = "-C target-feature=+crt-static"
cargo test --locked --target x86_64-pc-windows-msvc
cargo build --release --locked --target x86_64-pc-windows-msvc
.\target\x86_64-pc-windows-msvc\release\llm-sandbox-mxc.exe --probe
```

`rust-toolchain.toml` selects Rust 1.93.0, matching the pinned SDK's toolchain.
For ARM64 Windows, replace the target with `aarch64-pc-windows-msvc`. Both
architectures use the same source. It uses the OS's PSEC APIs through `processmodel.dll`;
no separately downloaded MXC DLL, administrator setup, or Windows Sandbox VM
is required. The release command above and the Windows wheel CI statically
link the CRT, avoiding a separate Visual C++ runtime installation. Keep the
explicit `--target` when setting this flag so it does not apply to host-side
procedural macro crates. Builds without `+crt-static` retain normal MSVC
runtime deployment requirements.

Portable protocol, filesystem-policy and quoting tests run on Linux as well.
Windows API type checking can be done without a Windows linker:

```sh
rustup target add x86_64-pc-windows-msvc aarch64-pc-windows-msvc
cargo check --locked --target x86_64-pc-windows-msvc
cargo check --locked --target aarch64-pc-windows-msvc
```

Cross-checking or successfully building the executable does **not** verify
kernel enforcement. Filesystem, networking, argument fidelity, descendant
cleanup and helper termination must also be exercised on a PSEC-capable
Windows machine before a release is described as verified there.

## Internal protocol, version 1

`--probe` always exits zero after returning one JSON object on stdout:

```json
{"protocol":1,"available":true,"deny_paths":true}
```

An unsupported machine returns `available: false`, `deny_paths: false` and a
`reason` string. Non-Windows builds have the same probe contract and refuse
execution. `deny_paths` reports the separate native deny-path support bit.

`--run` reads exactly one UTF-8 JSON object from stdin, followed by EOF. Input
is limited to 1 MiB, and unknown fields or trailing JSON are rejected:

```json
{
  "protocol": 1,
  "argv": ["C:\\Python\\python.exe", "C:\\private\\job\\main.py"],
  "cwd": "C:\\private\\job",
  "read_only_paths": ["C:\\Windows", "C:\\Python"],
  "writable_paths": ["C:\\private\\job"],
  "deny_paths": [],
  "network": "none",
  "env": {
    "SystemRoot": "C:\\Windows",
    "SystemDrive": "C:",
    "TEMP": "C:\\private\\job",
    "TMP": "C:\\private\\job"
  }
}
```

The example paths must exist and match the host's runtime installation.
The trusted Python parent creates the private workspace and determines grants;
this executable is not a privileged broker for requests from other users.
It checks that exactly one non-root directory is writable and that `cwd` is
inside it. Workspace/cwd ancestors must not be reparse points. Ordinary
absolute DOS paths are required; UNC, device paths, ADS, reserved names,
ambiguous components and relative paths are rejected. SDK normalization
resolves conflicting grants for aliases of the same existing filesystem object.
PSEC enforces junction-resolved targets at access time.

`argv[0]` must be an absolute `.exe` path. The launcher applies Windows
CreateProcess argument quoting directly; no host shell evaluates the request.
For shell workloads, the Python parent writes a script in the workspace and
requests `[absolute_cmd_exe, "/d", "/c", generated_script_basename]` with that
directory as `cwd`. A generated safe basename avoids cmd.exe reparsing special
characters in the absolute workspace path. Any shell is
itself launched inside PSEC. Arbitrary `cmd.exe /c` command text is not an
argv-safe interface and should not be sent as such.

`env` must be nonempty, with unique case-insensitive names and no NULs. Only
these variables become the child's explicit environment; no helper environment
is inherited. The parent supplies trusted Windows/runtime variables and points
HOME, USERPROFILE, APPDATA, TEMP and TMP to workspace-owned directories as
needed. The helper's own environment remains separate and trusted: SDK host
discovery must not use environment values supplied by sandboxed code.
Request environment entries are passed only to the child's explicit environment
block; they never modify the helper's environment. The pinned SDK loads PSEC
APIs from System32, and this helper disables capture paths that consult the
host's LOCALAPPDATA.

Network mode `none` denies egress, ingress and host loopback. Explicit `host`
allows all three; it is not proxy-only or domain-filtered networking. Native
denies are required whenever `deny_paths` is nonempty; missing support fails
before launch. The fixed `registryRead` capability permits registry reads;
this helper does not promise registry-read isolation. Clipboard access, input
injection and desktop/system control remain restricted while permitting Win32k
startup requirements for console applications.

Offline mode additionally carries explicit IPv4 and IPv6 deny-all egress rules;
host mode requests explicit host-loopback access. Both shapes require PSEC in
the pinned SDK. If PSEC cannot satisfy the policy, SDK validation rejects it
instead of selecting transitional SBOX.

The child gets EOF on stdin. Its stdout and stderr are forwarded concurrently
as bytes without decoration, flushing every chunk even without a newline.
Normal completion returns its real Windows exit
code (including the full 32-bit code). Helper validation/launch/transport errors
write a diagnostic prefixed `llm-sandbox-mxc:` to stderr and exit 125; a workload
can itself return 125, so the number alone is not a structured error channel.
SDK debug buffers are never emitted.

The SDK creates the child suspended, assigns a non-inheritable Job Object with
`JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE`, then resumes it. Foreground completion
terminates remaining descendants before closing PSEC or joining output readers.
Killing the helper also closes its job. The Python supervisor additionally
owns the helper's lifecycle; no cancellation command is sent over stdin.

## Distribution and licenses

This helper and Microsoft MXC are MIT-licensed. Microsoft MXC's license is in
[`LICENSE`](https://github.com/microsoft/mxc/blob/6cd3d58f05d3447e67109cfb75e042803b843ca4/LICENSE).
`Cargo.lock` pins the complete dependency graph. When distributing compiled
binaries, preserve the project's license and required copyright/license notices
for the pinned SDK and transitive Rust dependencies; their individual licenses
are declared in the downloaded crates. No upstream source is vendored here.
