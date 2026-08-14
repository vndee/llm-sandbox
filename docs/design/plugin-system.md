# Backend plugin system — design note

**Status:** proposed, awaiting review
**Scope:** the mechanism `INTEGRATIONS.md` already promises — third-party backends
shipped as separate distributions, discovered through an entry point, selected with
`SandboxSession(backend="<service>")`.

`INTEGRATIONS.md` is on `main` and commits publicly to this. The mechanism does not
exist. This note proposes it. Once a third party publishes against the entry-point
contract we cannot change it without breaking them, so this note is written as if
every decision below is permanent.

---

## 1. What is actually there today

Read before proposing, because several assumptions in the surrounding discussion are
wrong.

### 1.1 `SandboxSession` is a function

```python
# llm_sandbox/session.py:751
SandboxSession = create_session
```

It is an alias for a factory function, not a class. `SandboxSession(backend=...)` is
`create_session(backend=...)`. Nothing can subclass it, and `isinstance(x, SandboxSession)`
is a `TypeError`. Plugin documentation must never imply otherwise.

### 1.2 `backend=` already accepts plain strings

`SandboxBackend` is a `str` subclass (`const.py:13`, `StrEnum`). `create_session`
dispatches with `match`/`case`, which compares by `==`, and `_check_dependency` uses
set membership, which compares by hash and `==`. Both succeed for a bare `str`:

```python
SandboxSession(backend="docker")  # works today, on main
```

This is load-bearing. Whatever we build must not regress it.

`SandboxBackend._missing_` also does case-insensitive lookup, so `SandboxBackend("Docker")`
resolves. It raises `ValueError` — not a `SandboxError` — for anything unknown. Any new
code path that calls `SandboxBackend(value)` on a user-supplied string would surface a bare
`ValueError` for plugin names. It must not.

### 1.3 There are four built-in backends, not three

```python
# const.py:61
DOCKER = "docker"
KUBERNETES = "kubernetes"
PODMAN = "podman"
MICROMAMBA = "micromamba"
```

`micromamba` is a `SandboxDockerSession` subclass that wraps commands in `micromamba run`.
It is a real, selectable backend value. Every error message and every listing helper has to
account for it. (See §8 — the example error message in the task brief omits it.)

### 1.4 Backend resolution is duplicated across three sites, inconsistently

| Site | Supports |
|---|---|
| `session.py:248` `create_session` | docker, kubernetes, podman, micromamba |
| `interactive.py:85` `_create_backend_session` | docker, podman, kubernetes |
| `pool/factory.py:97` `create_pool_manager` | docker, kubernetes, podman |

Three `match` statements, three different backend sets. `session.py` additionally owns
`_check_dependency`, which the other two lack. A registry has to unify these without
changing any of their current behaviour — including the omissions, which are observable
(asking for a micromamba pool raises `UnsupportedBackendError` today, and must continue to).

### 1.5 The real backend contract is `BaseSession`, and all of it is private

`BaseSession` (`core/session_base.py:41`) is an ABC with four abstract methods, and it
inherits four more from the mixins in `core/mixins.py`:

| Method | Declared in |
|---|---|
| `_handle_timeout` | `session_base.py:544` |
| `_connect_to_existing_container` | `session_base.py:552` |
| `open` | `session_base.py:564` |
| `close` | `session_base.py:573` |
| `_ensure_directory_exists` | `mixins.py:259` |
| `_ensure_ownership` | `mixins.py:264` |
| `_process_non_stream_output` | `mixins.py:337` |
| `_process_stream_output` | `mixins.py:342` |

Six of the eight are underscore-prefixed. A third party cannot implement a backend today
without implementing private API — exactly what a plugin system is supposed to prevent.
This is the single largest obstacle and §4 addresses it.

### 1.6 Artifact support depends on an undeclared method

`ArtifactSandboxSession.run` calls
`language_handler.run_with_artifacts(container=self._session, ...)`, and the handler calls
`container.get_archive(path)` (`language_handlers/base.py:189`). `get_archive` is defined on
`SandboxDockerSession` (`docker.py:458`) and `SandboxKubernetesSession` (`kubernetes.py:651`)
and inherited by Podman. It is declared on **no** ABC. It is a required method that exists
only by convention.

There are also two unrelated protocols both called "container":

- `ContainerAPI` (`core/mixins.py:21`) — a `runtime_checkable` Protocol for the low-level
  runtime driver (create / start / stop / exec / copy in / copy out).
- `ContainerProtocol` (`language_handlers/base.py:17`) — a `TYPE_CHECKING`-only Protocol
  describing what a *session* must offer the language handler (`execute_command`,
  `get_archive`, `run`).

They do not match, and `ArtifactSandboxSession` passes a session where the parameter is
named `container`. Any interface we publish has to be unambiguous about which is which.

### 1.7 Constraints from the existing test suite

Two tests constrain the implementation directly:

- `tests/test_const.py:84` — `assert len(list(SandboxBackend)) == 4`. Dynamically extending
  the enum breaks this, and would be wrong anyway (see §5.3).
- `tests/test_exceptions.py:193` — `assert str(error) == "Unsupported backend: invalid_backend"`.
  The `UnsupportedBackendError` message is pinned exactly. We cannot enrich it in place.

Five call sites use `pytest.raises(UnsupportedBackendError)` (`test_session.py:97`,
`test_backend.py:75`, `test_pool_factory.py:148`, `test_interactive_session.py:57` and `:404`).
Any new error must be a **subclass** so those keep passing — and so downstream
`except UnsupportedBackendError` keeps working.

### 1.8 Packaging and CI

- `requires-python = ">=3.10,<4.0"`. Verified on the repo's own 3.10.16 venv:
  `entry_points(group=...)` returns `EntryPoints`, `.select()` and `.names` exist, and
  `ep.dist.name` / `ep.dist.version` are populated. The API is safe at the floor.
  `ep.dist` is still read defensively (`getattr(ep, "dist", None)`) because it is `None`
  for hand-constructed entry points.
- The only runtime dependency is `pydantic`. `importlib.metadata` and `difflib` are stdlib,
  so the registry adds nothing.
- CI matrix is 3.10–3.13. `make check` runs pre-commit, `mypy`, and `deptry`.
  `ruff` is `select = ["ALL"]`, line length 120. Docs build with `mkdocs build -s` (strict)
  and `pymdownx.snippets: check_paths: true`.

---

## 2. The entry point

### 2.1 Group name

```
llm_sandbox.backends
```

Matches the import path of the module that defines the contract (§4), which is the
convention users expect (`pytest11` notwithstanding). Underscore, not hyphen, so it matches
the Python package name rather than the distribution name.

### 2.2 What a plugin declares

```toml
# llm-sandbox-tenki/pyproject.toml
[project]
name = "llm-sandbox-tenki"
dependencies = ["llm-sandbox>=0.4,<1.0"]

[project.entry-points."llm_sandbox.backends"]
tenki = "llm_sandbox_tenki:TenkiBackend"
```

The entry-point **name** (`tenki`) is what users pass to `backend=`. The entry-point
**value** resolves to the registered object.

### 2.3 Shape of the registered object

The entry point must resolve to a **subclass of `SandboxBackendPlugin`** — a class, never
an instance. Core never instantiates it; every method is a `classmethod`. It is a
descriptor and a factory, not the session itself.

```python
from llm_sandbox.backends import BackendCapability, SandboxBackendPlugin

class TenkiBackend(SandboxBackendPlugin):
    PLUGIN_API_VERSION = 1
    name = "tenki"
    capabilities = frozenset({BackendCapability.ARTIFACTS})

    @classmethod
    def create_session(cls, **kwargs):
        from llm_sandbox_tenki.session import TenkiSession
        return TenkiSession(**kwargs)
```

**Why a descriptor and not the session class itself.** This is the decision that determines
what we are frozen into. Registering the session class directly would be simpler to write,
but it would freeze the whole of `BaseSession` — including the six private methods in §1.5
and the container-with-a-shell assumption in §7.1 — as permanent public contract. The
descriptor freezes six names, and every one of them is a name we chose deliberately. What a
plugin does *behind* `create_session` stays the plugin's business, and `BaseSession` stays
refactorable.

**Why not a `ContainerAPI` implementation.** It is the smallest possible contract and would
give plugins the most reuse, but it forces every hosted execution service to emulate
container semantics — create a container, start it, exec into it, tar files in and out —
whether or not it has any of those concepts. It closes the door on the exact category of
backend the plugin path exists to serve.

**Rejected leniency.** Accepting either a class or an instance was considered and dropped.
Two accepted shapes means two shapes to support forever and two ways for a plugin author
to be subtly wrong. The registry validates strictly and says what it expected.

### 2.4 The plugin interface

```python
class SandboxBackendPlugin(ABC):
    """Descriptor a third-party distribution registers to provide a backend."""

    PLUGIN_API_VERSION: ClassVar[int]
    name: ClassVar[str]
    capabilities: ClassVar[frozenset[BackendCapability]] = frozenset()

    @classmethod
    @abstractmethod
    def create_session(cls, **kwargs: Any) -> BaseSession: ...

    @classmethod
    def create_pool_manager(cls, **kwargs: Any) -> ContainerPoolManager: ...

    @classmethod
    def create_interactive_session(cls, **kwargs: Any) -> BaseSession: ...
```

Six frozen names. Every one justified:

| Name | Required | Why a plugin cannot work without it |
|---|---|---|
| `PLUGIN_API_VERSION` | yes | The only thing that lets a future core tell "built for the interface you have" from "built for an interface you no longer have". Without it a v1 plugin on a v2 core fails as an `AttributeError` somewhere inside a session, which is unactionable. §6. |
| `name` | yes | The canonical backend name. The entry-point name is the *lookup* key, but it is chosen by whoever wrote the `pyproject.toml`, which may not be the plugin author (forks, vendoring). Carrying it on the object lets the registry detect and report a mismatch instead of silently honouring whichever came first. |
| `capabilities` | no (defaults empty) | `ArtifactSandboxSession`, `InteractiveSandboxSession`, and the pool factory each need to know, *before* constructing anything, whether this backend can do the thing. Discovering it by calling and catching produces half-built sessions and containers that leak. |
| `create_session` | yes | The entire point. Everything `SandboxSession(backend=...)` does routes here. |
| `create_pool_manager` | no (raises by default) | Pooling is a distinct object with its own lifecycle (`ContainerPoolManager` has four abstract methods of its own) and genuinely backend-specific semantics — a hosted service may pool server-side, or not at all. Optional, with a default that raises a clear capability error. |
| `create_interactive_session` | no (raises by default) | Same reasoning, and see §7.6: interactive sessions currently reach into backend session internals, so this is the least portable capability of the three. Optional, default raises. |

Deliberately **not** on the interface:

- **A `describe()`/metadata hook.** Distribution name and version come free from
  `ep.dist`. Anything more is documentation, and documentation belongs in the plugin's README.
- **A streaming capability flag.** `run()` takes `on_stdout`/`on_stderr` in its signature
  regardless. A backend that cannot stream invokes each callback once with the complete
  output at completion. Requiring the *signature* rather than real-time delivery keeps the
  contract smaller and keeps caller code uniform.
- **A library-installation capability flag.** Core already has
  `LibraryInstallationNotSupportedError` and already raises it for languages that cannot
  install (`session_base.py:271`). A backend that cannot install libraries raises the same
  existing exception. No new concept.
- **A `close`/teardown hook on the plugin class.** The plugin class is a namespace, not a
  resource. Lifecycle belongs to the objects it returns.

### 2.5 Capabilities

```python
class BackendCapability(StrEnum):
    ARTIFACTS = "artifacts"                      # get_archive() + plot extraction
    INTERACTIVE = "interactive"                  # InteractiveSandboxSession support
    POOLING = "pooling"                          # create_pool_manager() support
    EXISTING_CONTAINER = "existing_container"    # container_id= attach support
```

Four. New capabilities may be added within a major API version (§6) — a plugin that does
not declare a capability it has never heard of is unaffected.

### 2.6 Name normalisation

```python
def normalize_backend_name(name: str) -> str:
    return name.strip().lower().replace("-", "_")
```

`"My-Service"`, `"my_service"`, and `"MY_SERVICE"` all resolve to `my_service`. Applied to
the requested name, the entry-point name, and the plugin's declared `name` alike, so
collisions are detected on the normalised form.

### 2.7 Collision resolution

| Situation | Resolution |
|---|---|
| Plugin name normalises onto a built-in (`docker`, `kubernetes`, `podman`, `micromamba`) | **Built-in wins.** Emit `BackendShadowWarning` naming the offending distribution and version. The built-in resolves as though the plugin were not installed. |
| Two plugins claim the same normalised name | **Error naming both distributions** — but only when *that name* is requested. |
| Plugin's declared `name` disagrees with its entry-point name | Registered under the entry-point name (that is what users type), with a warning naming both. |

The "only when requested" rule matters. Raising at discovery time would let one bad pair of
plugins break `list_backends()` and every other backend's resolution — the exact failure
mode the brief forbids. Instead the conflict is *recorded* at discovery and raised at
lookup, so `list_backends()` reports the conflicted name with `status="conflict"` and
everything else keeps working.

Built-in-wins is not negotiable: `SandboxSession(backend="docker")` must mean Docker on
every machine, and an installed package must never be able to change that.

---

## 3. Lazy loading and the registry

`llm_sandbox/registry.py`. Two distinct phases, and the distinction is a security property,
not an optimisation:

**Discovery** reads distribution metadata via
`importlib.metadata.entry_points(group="llm_sandbox.backends")`. It imports nothing and
executes no third-party code. It yields names, distributions, and versions.

**Loading** calls `ep.load()`, which imports the plugin's module. This happens only when
that specific backend is requested by name.

Discovery runs on the first backend resolution of the process — **including when a built-in
will win** — and is cached from then on. An earlier draft short-circuited built-ins before
discovery to save the metadata scan; building it showed that suppresses the shadow warning
for exactly the user who most needs it, since a package registering itself as `docker` is a
supply-chain attack shape. One cached metadata scan is the right price for that signal.

Consequences:

- `import llm_sandbox` never imports a plugin, and never even scans metadata. Discovery is
  triggered by the first resolution, not by import. `tests/test_lazy_imports.py` already
  asserts the analogous property for optional backend dependencies and must keep passing.
- `list_backends()` never imports a plugin either — it reports name, distribution, and
  version from metadata alone. `list_backends(load=True)` opts into importing, to populate
  `capabilities`, and tolerates per-plugin failures by recording `status="error"` rather
  than propagating.
- Installing a plugin means its build already ran on your machine. Lazy loading limits
  *import*-time execution to explicit opt-in; it is not a sandbox. §9 says so plainly and
  the authoring guide will repeat it.

Caching is a module-level dict guarded by a `threading.Lock` — pool code creates sessions
from worker threads. `clear_cache()` is public for tests and for processes that install
plugins at runtime.

### 3.1 Failure isolation

Any failure loading one entry point raises `BackendLoadError` **for that request only** and
never propagates into discovery of other backends:

- entry point raises on import
- resolved object is not a `SandboxBackendPlugin` subclass
- `PLUGIN_API_VERSION` missing or unsupported
- `create_session` not implemented

Each produces a message naming the distribution, the entry point, and what was expected.

### 3.2 Error message for an unknown backend

This message is the entire onboarding experience for most plugin users, so it is specified
here rather than left to the implementation.

Nothing installed provides the name, and nothing is close:

```
Unknown backend 'tenki'.
Built-in backends: docker, kubernetes, micromamba, podman.
No installed package provides 'tenki'. Third-party backends ship as separate
packages — try: pip install llm-sandbox-tenki
See https://github.com/vndee/llm-sandbox/blob/main/INTEGRATIONS.md
```

A plugin *is* installed and the name is a near miss (`difflib.get_close_matches`, stdlib):

```
Unknown backend 'tenkki'.
Built-in backends: docker, kubernetes, micromamba, podman.
Installed plugin backends: tenki (llm-sandbox-tenki 0.1.0).
Did you mean 'tenki'?
See https://github.com/vndee/llm-sandbox/blob/main/INTEGRATIONS.md
```

The second form is the common case in practice — someone has the plugin and typo'd — and it
is worth the extra branch. The `pip install llm-sandbox-<name>` suggestion follows the
naming convention `INTEGRATIONS.md` already states; it is a suggestion, and the message
says "try", because the convention is not enforced.

### 3.3 Exception hierarchy

All new exceptions subclass `UnsupportedBackendError` so that the five existing
`pytest.raises(UnsupportedBackendError)` sites and any downstream `except` keep working:

```
SandboxError
└── UnsupportedBackendError          (unchanged: message pinned by tests/test_exceptions.py:193)
    ├── BackendNotFoundError          no backend by that name
    ├── BackendLoadError              entry point failed to load or is malformed
    ├── BackendNameConflictError      two distributions claim the name
    └── BackendCapabilityError        backend exists but lacks the requested capability
```

`UnsupportedBackendError` itself is left completely untouched — same constructor, same
message. The subclasses bypass its message formatting and set their own.

### 3.4 Public helper

```python
llm_sandbox.list_backends(load: bool = False) -> list[BackendInfo]
```

`BackendInfo` is a frozen dataclass: `name`, `is_builtin`, `distribution`, `version`,
`entry_point`, `capabilities` (`None` unless loaded), `status`
(`"ok" | "shadowed" | "conflict" | "error"`), `error`.

---

## 4. The public backend base

Registering a descriptor answers *how core finds the plugin*. It does not answer *how a
plugin writes a session*. For that we publish `SandboxBackendBase`.

`llm_sandbox/backends/` is a new public package containing `SandboxBackendPlugin`,
`BackendCapability`, `SandboxBackendBase`, `BackendInfo`, and `PLUGIN_API_VERSION`.

`SandboxBackendBase` subclasses `BaseSession` and re-declares the six *private* abstract
methods under public names, adapting to the private ones with `final` bridges:

```python
class SandboxBackendBase(BaseSession, ABC):
    @abstractmethod
    def handle_timeout(self) -> None: ...

    @typing.final
    def _handle_timeout(self) -> None:      # satisfies BaseSession
        self.handle_timeout()
```

...and the same for `connect_to_existing_container`, `ensure_directory_exists`,
`ensure_ownership`, `process_non_stream_output`, `process_stream_output`. `open` and `close`
are already public and pass through unchanged.

It additionally declares `get_archive` — abstract when `BackendCapability.ARTIFACTS` is
declared, otherwise raising `BackendCapabilityError` — which closes the §1.6 gap for
plugins without changing anything for the built-ins.

**Why this and not renaming the methods on `BaseSession`.** Renaming is cleaner long-term
and would leave one obvious way to write a backend. It also touches a class that
`SandboxDockerSession`, `SandboxPodmanSession`, `SandboxKubernetesSession`,
`MicromambaSession`, and `InteractiveSandboxSession` all inherit from, plus any downstream
code we cannot see. The blast radius is unknown and the payoff is aesthetic. Deferred; it
can happen later behind deprecation shims without disturbing the plugin contract, which is
precisely the property the descriptor in §2.3 buys us.

**Cost, stated honestly.** In-tree backends keep using `BaseSession` and private names;
out-of-tree backends use `SandboxBackendBase` and public names. Two idioms for the same
job. The alternative was freezing six underscore-prefixed names as permanent public API.

### 4.1 The required set

Confirmed for v1. A compliant backend implements:

| Required | Contract |
|---|---|
| `open()` / `close()` | Idempotent. `close()` must release every resource, and must be safe to call after a failed `open()`. |
| `run(code, libraries=None, timeout=None, on_stdout=None, on_stderr=None)` | Returns `ConsoleOutput`. `exit_code` is 0 on success. Raises `SandboxTimeoutError` on timeout. |
| `execute_command(command, workdir=None, on_stdout=None, on_stderr=None)` | Returns `ConsoleOutput`. Raises `CommandEmptyError` on empty input, `NotOpenSessionError` if not open. |
| `copy_to_runtime(src, dest)` / `copy_from_runtime(src, dest)` | Raises `FileNotFoundError` for a missing source; must reject path traversal in `dest`. |

Optional, capability-declared: artifacts (`get_archive`), interactive, pooling,
existing-container attach.

The line sits here because `copy_to_runtime`/`copy_from_runtime` are already public,
already documented session methods, and artifact extraction is built on file transfer. A
narrower required set would make the public session API partial in a way users cannot
predict from the backend name.

### 4.2 What plugins import, and from where

Everything a realistic plugin needs, at a stable path:

| Need | Path | Status |
|---|---|---|
| `SandboxBackendPlugin`, `BackendCapability`, `SandboxBackendBase`, `PLUGIN_API_VERSION`, `BackendInfo`, `normalize_backend_name` | `llm_sandbox.backends` | new |
| `BaseSession`, `ContainerAPI` | `llm_sandbox.backends` | re-exported from `core` |
| `list_backends`, backend exceptions and warnings | `llm_sandbox` | new |
| `BackendComplianceTests` | `llm_sandbox.testing` | new, needs the `testing` extra |
| `ConsoleOutput`, `ExecutionResult`, `PlotOutput`, `FileType`, `StreamCallback` | `llm_sandbox` | already exported |
| `SessionConfig`, `SecurityPolicy`, `SupportedLanguage`, `SandboxBackend` | `llm_sandbox` | already exported |
| `ContainerPoolManager`, `PoolConfig` | `llm_sandbox.pool` | already exported |
| Exceptions | `llm_sandbox.exceptions` | see below |

Audit result: five exceptions a plugin must be able to raise are **not** in
`llm_sandbox.__all__` — `SandboxTimeoutError`, `NotOpenSessionError`, `CommandFailedError`,
`MissingDependencyError`, `LibraryInstallationNotSupportedError`. `llm_sandbox.exceptions`
is already a public, non-underscore module referenced by the docs, so it is declared public
in full, and these five are additionally added to the top-level `__all__`. Purely additive.

`ContainerAPI` (`core/mixins.py`) **is** promoted, re-exported from `llm_sandbox.backends`.
This reverses an earlier decision in this note. Writing the reference plugin showed the
choice was not "publish `ContainerAPI` or don't" but "publish it or freeze its six-method
shape anyway, undocumented": `FileOperationsMixin` and `CommandExecutionMixin` both read
`self.container_api`, so any backend that wants the inherited file transfer and command
execution must supply an object with exactly that shape. A structural protocol the plugin
can satisfy without importing anything is still a frozen contract. Better to name it.

Nothing else under `llm_sandbox.core` is promoted.

---

## 5. Backend-specific vs session-level, and what has to move

### 5.1 Where the boundary actually falls

| Behaviour | Layer | Notes |
|---|---|---|
| Security scanning (`is_safe`, pattern checks) | session | `BaseSession`, fully backend-agnostic. Free for plugins. |
| Language handler selection, execution commands | session | Fully backend-agnostic. Free. |
| Timeout *policy* (session lifetime, `_execute_with_timeout`) | session | `TimeoutMixin`. Free. |
| Timeout *enforcement* (killing the workload) | backend | `_handle_timeout`. Required. |
| Temp-file write → copy in → run commands | session | `BaseSession.run`. Assumes a filesystem — see §7.1. |
| Library installation | session, via handler | Uses `execute_commands`, so it works for any backend with a shell. |
| Container/pod lifecycle | backend | `open`/`close`. Required. |
| Command execution and output decoding | backend | `execute_command` + the two `_process_*_output` hooks. |
| File transfer | backend | `copy_to_container`/`copy_from_container`. |
| Artifact extraction | split | Handler drives it; backend supplies `get_archive`. §1.6. |
| Pooling | backend | Separate class hierarchy, four abstract methods. |
| Interactive | backend, deeply | §7.6. |

### 5.2 Refactor required

One refactor, and it is mechanical: the three dispatch sites in §1.4 route through the
registry instead of their own `match` statements. Built-ins become internal
`SandboxBackendPlugin` subclasses in `llm_sandbox/backends/builtin.py`, each importing its
module lazily inside `create_session` so the §3 lazy-import property is preserved for
built-ins too.

Behaviour is preserved exactly, including the current inconsistencies:

- micromamba's provider declares neither `INTERACTIVE` nor `POOLING`, so
  `InteractiveSandboxSession(backend="micromamba")` and
  `create_pool_manager(backend="micromamba")` keep raising an `UnsupportedBackendError`
  subclass, as they do today.
- Kubernetes' provider keeps filtering `runtime_configs` out of interactive kwargs
  (`interactive.py:99`).

Two constraints from the existing tests pin how the refactor may be written:

- **`_check_dependency` stays in `llm_sandbox/session.py`,** and `session.py` keeps importing
  `find_spec` at module level. Six tests patch `llm_sandbox.session.find_spec`
  (`test_session.py:19,27,35,43,56,69,82`) and moving the function would break all of them.
  `create_session` calls it for built-ins before delegating, preserving current ordering
  (after the `pool=` short-circuit, before construction).
- **Built-in providers must import the session class inside the method, not at module
  scope.** Tests patch the source-module attribute
  (`@patch("llm_sandbox.docker.SandboxDockerSession")`, `test_session.py:44,57,70,83`), which
  only works with late binding — the same pattern `create_session` uses today. This is also
  what keeps `tests/test_lazy_imports.py` green.
- `_create_backend_session` in `interactive.py` must be **kept as a function**, not deleted:
  `test_interactive_session.py:404` calls it directly.

### 5.3 Breaking-change risks

**None identified as unavoidable.** Assessed:

| Risk | Assessment |
|---|---|
| `UnsupportedBackendError` message changes | **Avoided.** Untouched; new subclasses carry the rich messages. `tests/test_exceptions.py:193` stays green. |
| Extending `SandboxBackend` dynamically | **Rejected.** Breaks `tests/test_const.py:84` (`len == 4`), and is wrong regardless: the enum means "backends core ships and tests in CI", which is exactly the line `INTEGRATIONS.md` draws. Plugins are strings; built-ins are enum members that are also strings. They coexist because the enum already *is* `str`. |
| Widening `backend: SandboxBackend` to `SandboxBackend \| str` | Parameter-type widening. Every existing call stays valid; no runtime change (bare strings already worked, §1.2). |
| Registry adds import cost to `import llm_sandbox` | Discovery is lazy, triggered on first resolution, not at import. |
| Built-in resolution changes shape | Same classes, same kwargs, same lazy imports. The `match` is replaced by a dict lookup. |
| A plugin shadowing a built-in | Prevented by design (§2.7). |

The one thing to watch during implementation: `_check_dependency` currently runs *before*
session construction and after the pool short-circuit. That ordering must be preserved —
`create_session(pool=...)` never checks dependencies today.

---

## 6. Versioning

```python
PLUGIN_API_VERSION = 1                       # what core implements
SUPPORTED_PLUGIN_API_VERSIONS = frozenset({1})  # what core accepts
```

Independent of the package version. A plugin declares the version it targets; core checks
membership in `SUPPORTED_PLUGIN_API_VERSIONS` at load. The set (rather than a single int)
exists so a future core can accept `{1, 2}` through a transition instead of forcing a flag
day.

Mismatch produces a message naming the distribution, the version it declared, and what this
core accepts — never an `AttributeError` from inside a session.

**Guaranteed within a major plugin API version:**

- No new *required* member on `SandboxBackendPlugin`.
- No incompatible signature change to a required method.
- No change to the meaning or type of a required return value.
- The entry-point group name does not change.
- The import paths in §4.2 keep working.
- New capabilities may be added; not declaring one is always safe.
- New *optional* hooks may be added, always with a working default.

**Bumps the major version:**

- Removing or renaming a required member.
- Changing a required signature or return type incompatibly.
- Changing the entry-point group.
- Changing collision or normalisation rules such that an existing plugin resolves differently.

**Not covered by the guarantee:** anything under `llm_sandbox.core`, `llm_sandbox.pool`
internals, the language handler classes, and the private methods on `BaseSession`. A plugin
that reaches into those is on its own, which is the whole reason §4 exists.

---

## 7. What makes this harder than it looks

### 7.1 `BaseSession.run()` assumes a POSIX container with a shell

`run()` writes the code to a host temp file, copies it into `workdir`, asks the language
handler for shell command strings, and executes them (`session_base.py:485-530`). It assumes
a writable filesystem, file transfer, and a shell.

A hosted execution API with an `POST /execute {code}` endpoint has none of these. Such a
backend either emulates them, or overrides `run()` and loses `install()`, artifact
extraction, and interactive support along with it. The plugin descriptor in §2.3 makes this
survivable — the plugin can return whatever session it likes — but it does not make it
*easy*, and the authoring guide has to say so. This is the single biggest reason the
descriptor shape was chosen over registering a `ContainerAPI`.

### 7.2 `get_archive` is a phantom requirement

§1.6. Artifacts need it; no ABC declares it. `SandboxBackendBase` fixes it for plugins;
the built-ins keep it as an undeclared convention, which is worth cleaning up separately.

### 7.3 Two protocols named "container"

§1.6. `ContainerAPI` describes a runtime driver, `ContainerProtocol` describes a session as
seen by a language handler, they do not match, and `ArtifactSandboxSession` passes a session
into a parameter named `container`. Documentation has to disambiguate carefully or plugin
authors will implement the wrong one.

### 7.4 Three dispatch sites disagree about which backends exist

§1.4. The registry unifies them, which means it also *surfaces* the disagreement. The
inconsistencies are preserved deliberately (§5.2) — fixing them would be a behaviour change
outside this scope, and is worth a separate issue.

### 7.5 `SandboxSession` is not a class

§1.1. It cannot be subclassed or `isinstance`-checked. Documentation and type hints have to
be careful; the natural thing to write in a plugin README is wrong.

### 7.6 Interactive sessions reach into backend internals

`InteractiveSandboxSession` reads `_backend_session._session_start_time`, `.container`,
`.container_api`, and `.python_executable_path`, then bootstraps by writing a runner script
into the container and polling files on its filesystem (`interactive.py:186-215`). A plugin
can only support this by subclassing `SandboxBackendBase` and behaving like a container.
Recommendation: v1 documents `INTERACTIVE` as supported only for container-shaped backends,
and does not pretend otherwise.

### 7.7 `SessionConfig.lang` is typed as `SupportedLanguage`

A plugin cannot introduce a language. Out of scope, but plugin authors will ask.

### 7.8 Entry-point caching

`importlib.metadata` caches. A plugin `pip install`ed into a running process will not appear
until `clear_cache()`. Tests that install fixtures at runtime must call it.

### 7.9 Repo tooling will resist the reference plugin

`ruff` runs `select = ["ALL"]` across the repo, `deptry` runs in `make check`, and
`mkdocs build -s` is strict with `snippets.check_paths: true`. The reference plugin in
`examples/plugins/` depends on `llm-sandbox` and pytest, which `deptry` will flag exactly as
it already flags `examples/agent_sdks` (already handled with `extend_exclude`). Plan for
tooling configuration as part of Phase 4, not as an afterthought.

### 7.10 The compliance kit needs pytest, which core does not depend on

`from llm_sandbox.testing import BackendComplianceTests` requires pytest at import. Proposal:
`llm_sandbox/testing/` imports pytest at module level and a new `testing` extra
(`pip install llm-sandbox[testing]`) declares it. No new *required* runtime dependency —
the constraint is respected — but the module is not importable without the extra, and the
error when it is missing should say so.

### 7.11 Nav placement of this note

`docs/design/` is not in `mkdocs.yml` nav and `mkdocs build -s` is strict. Verified by
running it: mkdocs logs unlisted pages at INFO, not WARNING, and the build exits 0. Left out
of nav deliberately — it is a design record, not user documentation. (The run also emits a
`git-revision-date-localized` warning about revision timestamps because the file is not yet
committed; it comes from the plugin's own logger, not mkdocs, so `-s` does not fail on it.)

---

## 8. Where `INTEGRATIONS.md` and the implementation will differ

Per instruction, `INTEGRATIONS.md` is not edited. Discrepancies found:

1. **Accurate.** "A backend plugin is a standalone package that depends on `llm-sandbox`
   and registers itself through the backend entry-point interface" — this is exactly what
   §2 builds.
2. **Accurate.** `SandboxSession(backend="<service>")` works, and already works for strings
   today (§1.2).
3. **Accurate but currently false, by design.** "the entry-point interface is public" is
   the premise of this work; it becomes true when this lands.
4. **Incomplete, not wrong.** No mention of `PLUGIN_API_VERSION`, capability declaration, or
   the compliance kit. The authoring guide covers them; `INTEGRATIONS.md` reads as a policy
   document and does not need to.
5. **Worth noting.** It names Docker, Podman, and Kubernetes as what ships in core and does
   not mention micromamba (§1.3). Not a plugin-system inaccuracy, but the "which backends
   are built in" question now has a user-visible answer via `list_backends()` and the error
   messages, and those will say four. Flagging rather than editing.

---

## 9. Security

Stated here so it is stated once, plainly, and repeated verbatim in the authoring guide and
the plugin index:

Installing a backend plugin means running third-party code in your environment. A backend,
by definition, is the component that executes untrusted code — it is the most
security-sensitive part of this system. Choosing a community backend means trusting that
package's authors. It does not mean trusting this project: listing in the plugin index
records that a plugin exists and is community-maintained, and implies no review, no
endorsement, and no guarantee.

Lazy loading (§3) means core does not import a plugin until you ask for that backend by
name. That limits *import*-time execution. It is not a sandbox, and by the time you can ask
for the backend, the package's install hooks have already run.

---

## 10. Questions raised at review, and how they were resolved

1. **Module name.** Kept `llm_sandbox.backends`, matching the entry point group name.
2. **`list_backends()` at top level.** Kept, as specified. It is the only function other than
   `create_session` at the top level.
3. **The pre-existing defects in §7.2 and §7.4.** Filed, not fixed. This change stays purely
   additive: `SandboxBackendBase` declares `get_archive` for plugins, and the built-ins keep
   their undeclared convention; the micromamba gaps in the interactive and pool dispatch
   paths are preserved exactly rather than quietly closed.
4. **`SUPPORTED_PLUGIN_API_VERSIONS` as a set.** Kept. It costs nothing now and is what lets
   a future major give plugin authors a transition window instead of a flag day.

## 11. What changed while building it

Two decisions in this note were wrong and were reversed during implementation. Both are
corrected in place above; they are collected here so the record is honest.

- **`ContainerAPI` is now public** (§4.2). The note originally refused to promote it. Writing
  the reference plugin showed the real choice was between publishing it and freezing its
  shape anyway, undocumented, since the inherited file-transfer and command-execution
  machinery both read `self.container_api`.
- **Discovery is no longer skipped for built-ins** (§3). The note originally short-circuited
  built-in lookups before any entry point scan. A test proved that suppresses the shadow
  warning for the one user who most needs it — someone whose `docker` backend a package is
  trying to hijack.

A third correction came from the test suite rather than the design: `_check_dependency` cannot
move out of `llm_sandbox/session.py`, because six tests patch `llm_sandbox.session.find_spec`.
See §5.2.

### Found in code review, after the first implementation landed

- **There were four dispatch sites, not three** (§1.4). `PooledSandboxSession` inferred the
  backend by substring-matching the pool manager's *class name*, then dispatched through its
  own hardcoded `match`. That made `POOLING` undeliverable for any plugin, and silently
  misrouted a third-party manager whose class name happened to contain `Docker`.
  `ContainerPoolManager` now carries a `backend_name`, and the dispatch goes through the
  registry like the other three.
- **`SandboxBackendBase` rejected `None` for non-nullable config fields.** Core forwards
  `runtime_configs=None` and `workdir="/sandbox"` unconditionally from
  `ArtifactSandboxSession`; the built-ins each normalise by hand, and the new base class did
  not. The result was a pydantic `ValidationError` — not even a `SandboxError` — from the one
  consumer of the `ARTIFACTS` capability. `_build_config` now drops `None` for fields that
  do not accept it.
- **Capabilities were documented as enforced but only two of four were.** `ARTIFACTS` and
  `EXISTING_CONTAINER` are now checked before construction, as §2.4 always claimed.
- **Degenerate backend names failed open.** An entry point may legally be named `""`, and
  `str(None)` is `"none"` — so a plugin could answer to `backend=""` or `backend=None`, which
  is what a caller passes when a config value is unset. Names are now NFKC-folded and
  validated against `^[a-z0-9][a-z0-9_]*$` on both the registration and lookup sides.
- **Warning delivery could abort discovery.** `warnings.warn` inside `_discover` meant that
  under `-W error`, a single shadowing plugin broke *every* backend and re-scanned metadata
  on every call. Discovery is now pure; warnings are logged unconditionally and delivered
  best-effort after the cache is committed.
- **`SystemExit` from a plugin escaped failure isolation**, taking the host process down.
  Both isolation points now catch `BaseException`, re-raising `KeyboardInterrupt`.
- **`_get_records()` could return `None`** when `clear_cache()` landed between the
  assignment and the return.
