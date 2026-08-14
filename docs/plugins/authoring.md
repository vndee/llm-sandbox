# Writing a backend plugin

LLM Sandbox ships with Docker, Podman, Kubernetes, and Micromamba backends. A **backend
plugin** adds another one from a separate package, without a fork and without a pull request
to this project.

```bash
pip install llm-sandbox-<service>
```

```python
from llm_sandbox import SandboxSession

with SandboxSession(backend="<service>", lang="python") as session:
    print(session.run("print('hello')").stdout)
```

You do not need approval to publish one. LLM Sandbox is MIT-licensed and the entry-point
interface documented here is public. [INTEGRATIONS.md][integrations] explains which backends
belong in core and which belong in a plugin; this page explains how to build one.

[integrations]: https://github.com/vndee/llm-sandbox/blob/main/INTEGRATIONS.md

!!! danger "Installing a plugin means running someone else's code"

    A backend is the component that executes untrusted code. It is the most
    security-sensitive part of this system.

    Installing a backend plugin means trusting that package's authors — with your
    credentials, your infrastructure, and the code you send to be executed. It does not mean
    trusting the LLM Sandbox maintainers. A listing in the [community plugin index](index.md)
    records that a plugin exists and is community-maintained. It implies no review, no
    endorsement, and no guarantee.

    LLM Sandbox does not import a plugin until you ask for its backend by name, which limits
    what runs at import time. That is not a sandbox: by the time a package is installed, its
    build already ran on your machine. Evaluate a community backend the way you would
    evaluate any dependency that handles your secrets.

## The shape of a plugin

Three pieces, in increasing size:

| Layer | What it is | Why |
|---|---|---|
| **Descriptor** | A `SandboxBackendPlugin` subclass | What the entry point points at. Declares who you are and hands back sessions. |
| **Session** | A `SandboxBackendBase` subclass | What users hold. Inherits `run()`, `install()`, security scanning, timeouts. |
| **Runtime driver** | A `ContainerAPI` implementation | Creates somewhere to run code, runs it, moves files in and out. |

The descriptor is deliberately tiny. It is the only part frozen as public contract, which is
what lets LLM Sandbox refactor its internals without breaking you.

A complete, runnable version of everything below lives in
[`examples/plugins/llm-sandbox-example/`][example]. Copy it rather than retyping this page.

[example]: https://github.com/vndee/llm-sandbox/tree/main/examples/plugins/llm-sandbox-example

## 1. Set up the project

```text
llm-sandbox-myservice/
├── pyproject.toml
├── README.md
├── src/
│   └── llm_sandbox_myservice/
│       ├── __init__.py
│       ├── backend.py       # the descriptor
│       ├── session.py       # the session
│       └── runtime.py       # the runtime driver
└── tests/
    ├── test_compliance.py
    └── test_backend.py
```

```toml title="pyproject.toml"
[project]
name = "llm-sandbox-myservice"
version = "0.1.0"
description = "MyService backend for llm-sandbox"
readme = "README.md"
requires-python = ">=3.10"
license = { text = "MIT" }

# Depend on llm-sandbox. Never vendor or fork it.
dependencies = [
    "llm-sandbox>=0.4.0,<1.0.0",
    "myservice-sdk>=2.0",
]

[project.optional-dependencies]
test = ["llm-sandbox[testing]>=0.4.0,<1.0.0", "pytest>=7.2.0"]

# This is the entire registration mechanism.
[project.entry-points."llm_sandbox.backends"]
myservice = "llm_sandbox_myservice:MyServiceBackend"

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.hatch.build.targets.wheel]
packages = ["src/llm_sandbox_myservice"]
```

The entry point **key** (`myservice`) is what users pass to `backend=`. The **value**
resolves to your descriptor class. There is no registration call and nothing to import at
start-up — LLM Sandbox reads packaging metadata and imports your module only when someone
asks for your backend.

Names are normalised before lookup: case is ignored and hyphens and underscores are
equivalent, so `myservice`, `MyService`, `my-service`, and `my_service` all reach the same
backend.

## 2. Write the descriptor

```python title="src/llm_sandbox_myservice/backend.py"
from typing import TYPE_CHECKING, Any, ClassVar

from llm_sandbox.backends import BackendCapability, SandboxBackendPlugin

if TYPE_CHECKING:
    from llm_sandbox.backends import BaseSession


class MyServiceBackend(SandboxBackendPlugin):
    """Executes code on MyService."""

    PLUGIN_API_VERSION: ClassVar[int] = 1
    name: ClassVar[str] = "myservice"
    capabilities: ClassVar[frozenset[BackendCapability]] = frozenset({
        BackendCapability.ARTIFACTS,
        BackendCapability.EXISTING_CONTAINER,
    })

    @classmethod
    def create_session(cls, *args: Any, **kwargs: Any) -> "BaseSession":
        # Import inside the method: resolving the entry point stays cheap, and an import
        # error surfaces as a clear BackendLoadError naming your distribution.
        from llm_sandbox_myservice.session import MyServiceSession

        return MyServiceSession(**kwargs)
```

```python title="src/llm_sandbox_myservice/__init__.py"
from llm_sandbox_myservice.backend import MyServiceBackend

__all__ = ["MyServiceBackend"]
```

The entry point must resolve to the **class**, not an instance. LLM Sandbox never
instantiates it — every method is a classmethod.

### What each declaration is for

**`PLUGIN_API_VERSION`** — required. Which version of this interface you built against.
Without it, a plugin built against an interface core no longer provides fails with an
`AttributeError` somewhere deep inside a session, which tells the user nothing. See
[Version compatibility](#version-compatibility).

**`name`** — required. Your canonical backend name. The entry-point key is what users type,
but it is chosen by whoever wrote the `pyproject.toml`, which is not always you. Declaring it
here lets core warn about a mismatch instead of silently honouring one of them.

**`capabilities`** — optional, defaults to empty. Declare only what you actually implement.
Core checks the declaration *before* constructing anything, so an unsupported capability
fails cleanly instead of leaving a half-built session or a leaked container behind. All four
are enforced: `ArtifactSandboxSession` requires `ARTIFACTS`, `container_id=` requires
`EXISTING_CONTAINER`, `create_pool_manager()` requires `POOLING`, and
`InteractiveSandboxSession` requires `INTERACTIVE`.

| Capability | Unlocks | You must implement |
|---|---|---|
| `ARTIFACTS` | `ArtifactSandboxSession`, plot capture | `get_archive()` on the session |
| `EXISTING_CONTAINER` | `container_id=` | `connect_to_existing_container()` |
| `POOLING` | `create_pool_manager()` | `create_pool_manager()` on the descriptor, **and** `EXISTING_CONTAINER` |
| `INTERACTIVE` | `InteractiveSandboxSession` | `create_interactive_session()` on the descriptor |

Declaring a capability you do not have is worse than not declaring it. Leave
`create_pool_manager` and `create_interactive_session` alone if you do not support them —
the inherited defaults raise a clear `BackendCapabilityError`.

!!! note "`POOLING` implies `EXISTING_CONTAINER`"

    A pooled session attaches to a container the pool already created, so a backend that
    declares `POOLING` must also declare `EXISTING_CONTAINER` and implement
    `connect_to_existing_container()`. Core checks this before building a pooled session.

!!! note "`INTERACTIVE` is the hard one"

    Interactive sessions write a runner script into the sandbox filesystem and poll it, and
    reach into the backing session's container handle. In practice only container-shaped
    backends can support it. Shipping without it is normal.

## 3. Write the session

Subclass `SandboxBackendBase` and you inherit `run()`, `install()`, security policy
enforcement, session timeouts, file transfer, and the context manager protocol. You implement
lifecycle plus five hooks.

```python title="src/llm_sandbox_myservice/session.py"
from typing import Any

from llm_sandbox.backends import SandboxBackendBase
from llm_sandbox import StreamCallback
from llm_sandbox.exceptions import ContainerError


class MyServiceSession(SandboxBackendBase):
    """A session backed by a MyService sandbox."""

    backend_name = "myservice"  # so errors name the backend, not this class

    def __init__(self, api_key: str | None = None, **kwargs: Any) -> None:
        super().__init__(**kwargs)

        from myservice_sdk import Client

        self.client = Client(api_key=api_key)
        self.container_api = MyServiceRuntime(self.client)

    # -- lifecycle ---------------------------------------------------------

    def open(self) -> None:
        super().open()  # starts the session timer, marks the session open

        if self.using_existing_container and self.config.container_id:
            self.connect_to_existing_container(self.config.container_id)
        else:
            self.container = self.container_api.create_container({
                "image": self.config.image or "myservice/python:3.11",
                "workdir": self.config.workdir,
            })
            self.container_api.start_container(self.container)

        self.environment_setup()

    def close(self) -> None:
        super().close()  # stops the timer, marks the session closed

        if self.container is not None and not self.using_existing_container:
            try:
                self.container_api.stop_container(self.container)
            except Exception as exc:
                raise ContainerError(str(exc)) from exc
            finally:
                self.container = None

    # -- required hooks ----------------------------------------------------

    def handle_timeout(self) -> None:
        """Cancel in-flight work. This is what makes a timeout real."""
        if self.container is not None:
            self.client.cancel(self.container)

    def ensure_directory_exists(self, path: str) -> None:
        exit_code, output = self.container_api.execute_command(self.container, f"mkdir -p {path}")
        if exit_code != 0:
            # Do not swallow this: the copy that follows will fail confusingly instead.
            _, stderr = self.process_non_stream_output(output)
            raise ContainerError(f"Could not create {path}: {stderr}")

    def ensure_ownership(self, paths: list[str]) -> None:
        """No-op when the sandbox always runs as the owning user."""

    def process_non_stream_output(self, output: Any) -> tuple[str, str]:
        stdout, stderr = output
        errors = self.config.encoding_errors
        return (
            stdout.decode("utf-8", errors=errors) if stdout else "",
            stderr.decode("utf-8", errors=errors) if stderr else "",
        )

    def process_stream_output(
        self,
        output: Any,
        on_stdout: StreamCallback | None = None,
        on_stderr: StreamCallback | None = None,
    ) -> tuple[str, str]:
        stdout, stderr = "", ""
        for channel, chunk in output:
            text = chunk.decode("utf-8", errors=self.config.encoding_errors)
            if channel == "stdout":
                stdout += text
                if on_stdout:
                    on_stdout(text)
            else:
                stderr += text
                if on_stderr:
                    on_stderr(text)
        return stdout, stderr

    # -- optional, because ARTIFACTS is declared ---------------------------

    def get_archive(self, path: str) -> tuple[bytes, dict]:
        return self.container_api.copy_from_container(self.container, path)

    # -- optional, because EXISTING_CONTAINER is declared ------------------

    def connect_to_existing_container(self, container_id: str) -> None:
        self.container = self.client.attach(container_id)
```

### The required interface

Every compliant backend supports these. They are what the public session API promises users
regardless of which backend they picked:

| Method | Contract |
|---|---|
| `open()` / `close()` | `close()` releases every resource and is safe to call twice, and after a failed `open()`. |
| `run(code, libraries=None, timeout=None, on_stdout=None, on_stderr=None)` | Returns `ConsoleOutput`. Propagates the real exit code. Raises `SandboxTimeoutError` on timeout. |
| `execute_command(command, workdir=None, ...)` | Returns `ConsoleOutput`. Raises `CommandEmptyError` on empty input, `NotOpenSessionError` if not open. |
| `copy_to_runtime(src, dest)` / `copy_from_runtime(src, dest)` | Raises `FileNotFoundError` for a missing source; rejects path traversal in `dest`. |

`run`, `execute_command`, `copy_to_runtime`, and `copy_from_runtime` are all inherited from
`SandboxBackendBase` — you get them by implementing the runtime driver in the next step.

!!! warning "Streaming callbacks are required; real-time delivery is not"

    `on_stdout` and `on_stderr` must be accepted and must fire. A backend that cannot stream
    may collect the complete output and invoke each callback once at the end.

!!! warning "`workdir` defaults to a path inside a container"

    Core's default `workdir` is `/sandbox`, and `ArtifactSandboxSession` passes it
    *explicitly* rather than leaving it unset — so "the caller did not choose one" is not
    something you can detect by absence alone.

    If your backend executes anywhere other than a container filesystem, map it:

    ```python
    CONTAINER_DEFAULT_WORKDIR = "/sandbox"

    def __init__(self, **kwargs):
        requested = kwargs.get("workdir")
        self._owns_workdir = requested in (None, CONTAINER_DEFAULT_WORKDIR)
        if self._owns_workdir:
            kwargs["workdir"] = tempfile.mkdtemp()
        super().__init__(**kwargs)
    ```

    And only delete a directory you created. Deleting one the caller named is not
    recoverable.

!!! warning "The inherited `run()` assumes a filesystem and a shell"

    `SandboxBackendBase.run()` writes the code to a temporary file, copies it into the
    workdir, and executes shell commands from the language handler.

    If your service is a `POST /execute {code}` endpoint with no filesystem, you must either
    emulate one in your runtime driver, or override `run()` outright — which also gives up
    `install()` and artifact extraction. Emulating is usually less work than it sounds and
    keeps every other feature working.

## 4. Write the runtime driver

`ContainerAPI` is a structural protocol — six methods. You do not need to inherit from it;
matching the shape is enough.

```python title="src/llm_sandbox_myservice/runtime.py"
import io
import tarfile
from typing import Any


class MyServiceRuntime:
    """Moves work to MyService and results back."""

    def __init__(self, client: Any) -> None:
        self.client = client

    def create_container(self, config: Any) -> Any:
        return self.client.create_sandbox(image=config["image"], workdir=config["workdir"])

    def start_container(self, container: Any) -> None:
        self.client.start(container)

    def stop_container(self, container: Any) -> None:
        self.client.destroy(container)

    def execute_command(self, container: Any, command: str, **kwargs: Any) -> tuple[int, Any]:
        result = self.client.exec(container, command, cwd=kwargs.get("workdir"))
        return result.exit_code, (result.stdout_bytes, result.stderr_bytes)

    def copy_to_container(self, container: Any, src: str, dest: str, **kwargs: Any) -> None:
        with open(src, "rb") as handle:
            self.client.upload(container, dest, handle.read())

    def copy_from_container(self, container: Any, src: str, **kwargs: Any) -> tuple[bytes, dict]:
        payload = self.client.download(container, src)
        if payload is None:
            return b"", {"size": 0}  # size 0 is how callers detect "not found"

        buffer = io.BytesIO()
        with tarfile.open(fileobj=buffer, mode="w") as tar:
            info = tarfile.TarInfo(name=src.rsplit("/", 1)[-1])
            info.size = len(payload)
            tar.addfile(info, io.BytesIO(payload))

        data = buffer.getvalue()
        return data, {"size": len(data)}
```

Two contracts worth stating explicitly:

- `execute_command` returns `(exit_code, output)`, where `output` is whatever your
  `process_non_stream_output` / `process_stream_output` hooks know how to decode.
- `copy_from_container` returns **uncompressed tar bytes** plus a stat mapping. A `size` of
  `0` is how callers detect a missing path — do not raise for that case.

## 5. Run the compliance kit

The compliance kit is how you check your backend satisfies the interface without anyone
having to review your code.

```bash
pip install -e ".[test]"
```

```python title="tests/test_compliance.py"
from llm_sandbox.testing import BackendComplianceTests


class TestMyServiceCompliance(BackendComplianceTests):
    backend = "myservice"
    session_kwargs = {"lang": "python", "api_key": "test-key"}
```

```bash
pytest
```

That is the whole file. pytest collects every inherited check:

- **Interface completeness** — registration, plugin API version, name agreement, capability
  declarations, and that undeclared optional factories raise.
- **Lifecycle** — the context manager runs code; `run()` before `open()` raises; `close()` is
  idempotent; closing an unopened session is safe; **and an exception inside the `with` body
  still closes the session.** That last one is the check most worth having: a backend that
  leaks a container when the body raises passes every happy-path test and quietly costs its
  users money.
- **Result types** — `run()` returns a `ConsoleOutput` with correctly typed fields, and a
  failing program's exit code survives instead of being normalised to 0 or 1.
- **File transfer** — a file round-trips unchanged; a missing source raises `FileNotFoundError`.
- **Timeouts** — long-running code raises `SandboxTimeoutError`, and teardown after a timeout
  does not raise on top of it.
- **Declared capabilities** — if you declare `ARTIFACTS`, `get_archive()` really returns tar
  bytes for a file that exists.

Customise the code snippets if your backend's default language is not Python:

```python
class TestMyServiceCompliance(BackendComplianceTests):
    backend = "myservice"
    hello_code = "console.log('llm-sandbox-compliance')"
    hello_expected = "llm-sandbox-compliance"
    failing_code = "process.exit(3)"
    slow_code = "setTimeout(() => {}, 30000)"
    timeout_seconds = 1.0
```

If your CI cannot reach the real service, run the static half there and the full suite in a
job that has credentials:

```python
from llm_sandbox.testing import BackendInterfaceComplianceTests


class TestMyServiceInterface(BackendInterfaceComplianceTests):
    backend = "myservice"
```

Then add your own tests for the things only you can check — that your service is actually
being called, that your options are honoured, that your errors map onto LLM Sandbox's.

## Version compatibility

LLM Sandbox exposes a plugin API version independent of its package version:

```python
from llm_sandbox.backends import PLUGIN_API_VERSION, SUPPORTED_PLUGIN_API_VERSIONS

PLUGIN_API_VERSION           # 1  -- what this release implements
SUPPORTED_PLUGIN_API_VERSIONS  # frozenset({1}) -- what this release accepts
```

Declare the version you built against. On load, a mismatch produces an error naming your
distribution, the version you declared, and what that release accepts — instead of an
obscure failure later.

**Within a plugin API major version, LLM Sandbox guarantees:**

- No new *required* member on `SandboxBackendPlugin`.
- No incompatible signature change to a required method.
- No change to the meaning or type of a required return value.
- The entry-point group name does not change.
- Every name in [Public API surface](#public-api-surface) keeps working from its documented
  import path.
- New capabilities may be added. Not declaring one is always safe.
- New *optional* hooks may be added, always with a working default.

**These bump the major version:**

- Removing or renaming a required member.
- Changing a required signature or return type incompatibly.
- Changing the entry-point group.
- Changing name-normalisation or collision rules such that an existing plugin resolves
  differently.

A future release can accept more than one version at once, so a bump comes with a transition
window rather than a flag day.

## Public API surface

**Covered by the compatibility guarantee.** Import only from these paths:

| Import from | Names |
|---|---|
| `llm_sandbox.backends` | `SandboxBackendPlugin`, `SandboxBackendBase`, `BackendCapability`, `BackendInfo`, `BaseSession`, `ContainerAPI`, `PLUGIN_API_VERSION`, `SUPPORTED_PLUGIN_API_VERSIONS`, `ENTRY_POINT_GROUP`, `normalize_backend_name` |
| `llm_sandbox` | `SandboxSession`, `create_session`, `ArtifactSandboxSession`, `InteractiveSandboxSession`, `list_backends`, `SessionConfig`, `ConsoleOutput`, `ExecutionResult`, `PlotOutput`, `FileType`, `StreamCallback`, `SupportedLanguage`, `SandboxBackend`, `DefaultImage`, `SecurityPolicy`, `SecurityPattern`, `SecurityIssueSeverity`, `KernelType`, `BackendShadowWarning`, `BackendNameMismatchWarning` |
| `llm_sandbox.registry` | `get_backend`, `list_backends`, `clear_cache`, `BackendShadowWarning`, `BackendNameMismatchWarning` |
| `llm_sandbox.exceptions` | every exception in the module |
| `llm_sandbox.pool` | `ContainerPoolManager`, `PoolConfig`, `ExhaustionStrategy`, `PooledContainer`, `ContainerState`, and the pool exceptions |
| `llm_sandbox.testing` | `BackendComplianceTests`, `BackendInterfaceComplianceTests` |

**Not covered. Do not import these** — they will change without a major version bump:

- Anything under `llm_sandbox.core`. `BaseSession` and `ContainerAPI` are re-exported from
  `llm_sandbox.backends` precisely so you never have to.
- `BaseSession` **as a base class**. It is exported so you can annotate against it, and
  `create_session` is typed as returning it. Do not subclass it directly: its abstract
  method set may grow within a plugin API major version, and `SandboxBackendBase` exists
  to absorb that for you. Subclass `SandboxBackendBase`.
- Underscore-prefixed names in `llm_sandbox.registry`. The four listed above are stable;
  nothing else in that module is.
- The language handler classes in `llm_sandbox.language_handlers`.
- The underscore-prefixed methods on `BaseSession`. `SandboxBackendBase` gives you public
  names for all of them; the bridges are `@final`, so override the public name.
- The concrete built-in session classes (`SandboxDockerSession` and friends). Subclassing
  them couples you to internals.

Exceptions you are expected to raise, all importable from `llm_sandbox` or
`llm_sandbox.exceptions`:

| Exception | Raise it when |
|---|---|
| `ContainerError` | The runtime failed to create, start, attach to, or destroy a sandbox |
| `NotOpenSessionError` | An operation needs an open session and there is not one |
| `SandboxTimeoutError` | Execution exceeded its timeout |
| `CommandFailedError` | A command that had to succeed did not |
| `MissingDependencyError` | Your SDK is not installed |
| `LibraryInstallationNotSupportedError` | Your backend cannot install libraries |
| `SecurityViolationError` | A security policy was violated |
| `ImagePullError` / `ImageNotFoundError` | An image could not be fetched or found |
| `BackendCapabilityError` | Something asked for a capability you do not declare |

## Conventions

**Package naming.** `llm-sandbox-<service>`, with the import name
`llm_sandbox_<service>`. Please do not publish an `llm-sandbox-*` package that is a fork or a
vendored copy rather than a dependent — it confuses users about what they are installing.

**Entry-point naming.** The entry-point key should be the service name, lowercase, and should
match your `name` attribute. Prefer the shortest unambiguous form: `myservice`, not
`llm-sandbox-myservice`. Users type it on every call.

**Depend on `llm-sandbox`; do not vendor it.** Vendoring means your users get two copies of
the session machinery and no upstream fixes.

**Declare a compatible version range.** Lower bound is the release that introduced the plugin
API you target; upper bound is the next major:

```toml
dependencies = ["llm-sandbox>=0.4.0,<1.0.0"]
```

Do not pin exactly (`==0.4.1`) — it makes your plugin uninstallable alongside anything else.
Do not leave it open-ended (`llm-sandbox`) — a future major will break you silently.

**Semver your own package.** Bump the major when you change your own options or behaviour in
a way that breaks users. Your release cadence is yours; that is the point of the plugin path.

**What belongs where:**

| Core handles | You handle |
|---|---|
| Security policy scanning | Talking to your service |
| Language handler selection, execution commands | Authentication and credentials |
| Session and execution timeout policy | Cancelling work when a timeout fires |
| The `run()` pipeline, `install()` | Runtime lifecycle |
| File transfer safety (traversal, symlinks) | Moving the bytes |
| Artifact detection and extraction logic | `get_archive()` |

Do not reimplement the left column. If something there does not fit your backend,
[open an issue](https://github.com/vndee/llm-sandbox/issues) — that is a gap in this
interface and worth fixing for everyone.

## Publish

```bash
python -m build
twine upload dist/*
```

Then [open an issue](https://github.com/vndee/llm-sandbox/issues) to be added to the
[community plugin index](index.md). Include the package name, the service, the maintainer,
and the repository. There is no application, no review, and no fee.

If you are an employee of, contractor for, or otherwise compensated by the service your
plugin integrates, please say so — most vendor integrations are written by the vendor, and
that is expected.

## Troubleshooting

**`Unknown backend 'myservice'`** — the distribution is not installed, or the entry point is
not declared. Check `llm_sandbox.list_backends()` and the
`[project.entry-points."llm_sandbox.backends"]` table. Editable installs sometimes need a
reinstall after adding an entry point.

**`... does not declare PLUGIN_API_VERSION`** — add `PLUGIN_API_VERSION = 1` to the descriptor.

**`... targets plugin API version N`** — your plugin and the installed LLM Sandbox disagree.
Check your version range.

**`... must resolve to a subclass of SandboxBackendPlugin`** — the entry point points at an
instance, a function, or the session class. Point it at the descriptor class.

**`... is incomplete: it does not implement create_session`** — the descriptor is still
abstract.

**`Backend name 'x' is claimed by more than one installed distribution`** — two packages
registered the same name. Uninstall one. Other backends keep working.

**A `BackendShadowWarning` naming your distribution** — you registered a name a built-in
already owns. Built-ins always win; rename your entry point.

**Your plugin is not being imported** — that is by design. Discovery reads metadata only;
your module is imported when someone requests your backend.

## See also

- [Community plugin index](index.md)
- [INTEGRATIONS.md][integrations] — what belongs in core, what belongs in a plugin
- [Reference plugin][example] — everything on this page, runnable
- [Container Backends](../backends.md) — how the built-in backends work
