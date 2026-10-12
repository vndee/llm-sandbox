# llm-sandbox-example

A complete, working [llm-sandbox](https://github.com/vndee/llm-sandbox) backend plugin,
kept as small as a correct one can be. Copy it as the starting point for a real backend.

> [!WARNING]
> **This backend provides no isolation.** It runs code directly on your machine, as you,
> with your filesystem and your network. It exists to demonstrate the plugin interface end
> to end with something you can actually run. Never point it at untrusted code — that is
> what the Docker, Podman, and Kubernetes backends are for.

## Try it

```bash
pip install -e ".[test]"
```

```python
from llm_sandbox import SandboxSession

with SandboxSession(backend="example", lang="python") as session:
    print(session.run("print(6 * 7)").stdout)  # 42
```

```python
import llm_sandbox

for backend in llm_sandbox.list_backends():
    origin = "built-in" if backend.is_builtin else f"{backend.distribution} {backend.version}"
    print(f"{backend.name:<14} {origin}")
```

## What is in here

| File | What it does |
|---|---|
| `pyproject.toml` | Declares the entry point. This is the entire registration mechanism. |
| `src/llm_sandbox_example/backend.py` | The `SandboxBackendPlugin` descriptor the entry point resolves to. |
| `src/llm_sandbox_example/session.py` | The session, a `SandboxBackendBase` subclass. |
| `src/llm_sandbox_example/runtime.py` | The `ContainerAPI` that actually moves and runs things. |
| `tests/test_compliance.py` | The whole compliance kit, in five lines. |
| `tests/test_backend.py` | Tests only you can write, about your backend's own behaviour. |

## How registration works

One table in `pyproject.toml`:

```toml
[project.entry-points."llm_sandbox.backends"]
example = "llm_sandbox_example:ExampleBackend"
```

The key is the name users pass to `SandboxSession(backend=...)`. The value resolves to a
`SandboxBackendPlugin` subclass. That is all — there is no registration call, no plugin
manager, and nothing to import at start-up. llm-sandbox reads the metadata and imports your
module only when someone asks for your backend by name.

Names are normalised: `example`, `Example`, and `EXAMPLE` all resolve here, and hyphens and
underscores are interchangeable.

## The three layers

**The descriptor** (`backend.py`) declares who you are and hands back sessions. It is
deliberately tiny; keeping it that way is what lets llm-sandbox refactor its internals
without breaking you.

**The session** (`session.py`) is what users hold. Subclassing `SandboxBackendBase` means
inheriting `run()`, `install()`, security policy enforcement, timeouts, file transfer, and
the context manager protocol — you implement lifecycle plus five hooks.

**The runtime driver** (`runtime.py`) implements `ContainerAPI`: six methods that create a
place to run things, run them, and move files in and out. Implement this and the two layers
above it mostly write themselves. A real backend talks to a container runtime or a vendor
API here; this one shells out to subprocesses.

## Running the compliance kit

```bash
pytest
```

`tests/test_compliance.py` is five lines and gets you the full suite: interface
completeness, lifecycle (including cleanup when the `with` body raises), result-type
conformance, file round-tripping, and timeout cancellation.

If you cannot run live sessions in CI — no credentials, no daemon — subclass
`BackendInterfaceComplianceTests` instead for the static half, and run the full suite in a
job that does have access.

## Turning this into a real backend

1. Rename the package and the distribution. The convention is `llm-sandbox-<service>`.
2. Change `name` in `backend.py` and the entry point key in `pyproject.toml` to match.
3. Replace `runtime.py` with calls to your service.
4. Declare only the `capabilities` you actually implement. Declaring one you do not have is
   worse than not declaring it: core takes the declaration at its word and lets the call
   through, so the failure surfaces later and further from the cause.
5. Keep `PLUGIN_API_VERSION` accurate, and pin `llm-sandbox` to a compatible range.
6. Delete this warning box and write an honest one about your own isolation guarantees.

Full guide: [Writing a backend plugin](https://vndee.github.io/llm-sandbox/plugins/authoring/).
Policy on what belongs in core versus a plugin:
[INTEGRATIONS.md](https://github.com/vndee/llm-sandbox/blob/main/INTEGRATIONS.md).

## License

MIT, same as llm-sandbox.
