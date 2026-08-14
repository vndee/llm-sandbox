"""Discovery and resolution of sandbox backends.

Built-in backends resolve from a dict; third-party backends are discovered through the
``llm_sandbox.backends`` entry point group.

Discovery and loading are deliberately separate:

- **Discovery** reads distribution metadata. It imports nothing and runs no third-party code.
  `list_backends` works entirely at this level.
- **Loading** imports the plugin's module, and happens only when that backend is requested
  by name.

That split is a security property, not an optimisation: importing ``llm_sandbox`` must never
execute code belonging to every backend plugin that happens to be installed. It is not a
sandbox, though -- by the time a plugin is installed, its build already ran. See
``docs/plugins/authoring.md``.
"""

import contextlib
import inspect
import logging
import threading
import warnings
from collections import defaultdict
from dataclasses import dataclass
from importlib.metadata import EntryPoint, entry_points
from typing import Any

from llm_sandbox.backends.builtin import BUILTIN_BACKENDS
from llm_sandbox.backends.plugin import (
    BACKEND_NAME_PATTERN,
    ENTRY_POINT_GROUP,
    SUPPORTED_PLUGIN_API_VERSIONS,
    BackendCapability,
    BackendInfo,
    BackendStatus,
    SandboxBackendPlugin,
    normalize_backend_name,
)
from llm_sandbox.exceptions import (
    BackendLoadError,
    BackendNameConflictError,
    BackendNotFoundError,
)

__all__ = [
    "BackendNameMismatchWarning",
    "BackendShadowWarning",
    "clear_cache",
    "get_backend",
    "list_backends",
]

logger = logging.getLogger(__name__)

INTEGRATIONS_URL = "https://github.com/vndee/llm-sandbox/blob/main/INTEGRATIONS.md"

_DID_YOU_MEAN_CUTOFF = 0.6


class BackendShadowWarning(UserWarning):
    """Warned when an installed plugin registers a name a built-in backend already owns.

    The built-in always wins. ``SandboxSession(backend="docker")`` must mean Docker on every
    machine, and no installable package may change that.
    """


class BackendNameMismatchWarning(UserWarning):
    """Warned when a plugin's declared ``name`` differs from its entry point name.

    The entry point name wins, because that is what users type.
    """


@dataclass(frozen=True)
class _PluginRecord:
    """Bookkeeping for a discovered entry point, before it is loaded."""

    name: str
    entry_point: EntryPoint
    distribution: str | None
    version: str | None
    conflicts: tuple[str, ...] = ()


_lock = threading.Lock()
_records: dict[str, _PluginRecord] | None = None
_shadowed: tuple[BackendInfo, ...] = ()
_loaded: dict[str, type[SandboxBackendPlugin]] = {}


def clear_cache() -> None:
    """Discard cached entry point discovery and loaded plugin classes.

    Needed when plugins are installed into a running process, and by tests that register
    backends dynamically. Ordinary applications never call this.
    """
    global _records, _shadowed  # noqa: PLW0603
    with _lock:
        _records = None
        _shadowed = ()
        _loaded.clear()


def _entry_points() -> list[EntryPoint]:
    """Read this group's entry points, tolerating a broken metadata directory.

    Returns:
        list[EntryPoint]: Discovered entry points; empty if metadata could not be read.

    """
    try:
        return list(entry_points(group=ENTRY_POINT_GROUP))
    except Exception as exc:  # noqa: BLE001
        # A corrupt or unreadable distribution on sys.path must not stop built-in backends
        # from resolving. Report the cause, or the user has nothing to act on.
        logger.warning(
            "Could not read %r entry points; no plugin backends available: %s: %s",
            ENTRY_POINT_GROUP,
            type(exc).__name__,
            exc,
        )
        return []


def _discover() -> tuple[dict[str, _PluginRecord], tuple[BackendInfo, ...], list[str]]:
    """Scan entry points and build the plugin record table. Imports nothing.

    Pure with respect to module state, and emits no warnings: it returns the messages for
    the caller to deliver *after* the record table is committed. Warning delivery runs
    arbitrary user code (filters, ``showwarning``) and raises under ``-W error``, and neither
    may be able to abort discovery or leave the cache unpopulated.

    Returns:
        tuple[dict[str, _PluginRecord], tuple[BackendInfo, ...], list[str]]: Records keyed
            by normalised backend name, shadowed registrations, and pending warnings.

    """
    candidates: dict[str, list[tuple[EntryPoint, str | None, str | None]]] = defaultdict(list)
    shadowed: list[BackendInfo] = []
    pending: list[str] = []

    for entry_point in _entry_points():
        name = normalize_backend_name(entry_point.name)
        dist = getattr(entry_point, "dist", None)
        dist_name = getattr(dist, "name", None)
        dist_version = getattr(dist, "version", None)
        source = f"{dist_name or 'an unknown distribution'}{' ' + dist_version if dist_version else ''}"

        if not BACKEND_NAME_PATTERN.match(name):
            # Refused rather than registered. An empty or exotic entry point name would
            # otherwise answer to backend="" or a homoglyph, turning an unset config value
            # into a silent hand-off of code execution.
            pending.append(
                f"Plugin backend {entry_point.name!r} from {source} has an unusable name and will be "
                f"ignored. Backend names must match {BACKEND_NAME_PATTERN.pattern}."
            )
            continue

        if name in BUILTIN_BACKENDS:
            pending.append(
                f"Plugin backend {entry_point.name!r} from {source} shadows the built-in backend "
                f"{name!r} and will be ignored. Built-in backends always win."
            )
            shadowed.append(
                BackendInfo(
                    name=name,
                    distribution=dist_name,
                    version=dist_version,
                    entry_point=entry_point.value,
                    status="shadowed",
                    detail=f"Shadows the built-in {name!r} backend; the built-in is used instead.",
                )
            )
            continue

        # The same distribution can appear twice if it is installed twice on sys.path. That
        # is not a conflict between two projects, so collapse it instead of reporting one.
        duplicate_key = (dist_name, entry_point.value)
        if any((seen_dist, seen_ep.value) == duplicate_key for seen_ep, seen_dist, _ in candidates[name]):
            continue
        candidates[name].append((entry_point, dist_name, dist_version))

    records: dict[str, _PluginRecord] = {}
    for name, entries in candidates.items():
        entry_point, dist_name, dist_version = entries[0]
        # A conflict is recorded, not raised. Raising here would let one bad pair of plugins
        # break resolution for every other backend; instead it surfaces when that specific
        # name is requested.
        conflicts = tuple(sorted({d or "<unknown distribution>" for _, d, _ in entries})) if len(entries) > 1 else ()
        records[name] = _PluginRecord(
            name=name,
            entry_point=entry_point,
            distribution=dist_name,
            version=dist_version,
            conflicts=conflicts,
        )

    return records, tuple(shadowed), pending


def _get_records() -> dict[str, _PluginRecord]:
    """Return the plugin record table, discovering once and caching.

    Returns:
        dict[str, _PluginRecord]: Records keyed by normalised backend name.

    """
    global _records, _shadowed

    records = _records
    if records is not None:
        return records

    pending: list[str] = []
    with _lock:
        # Re-read under the lock, and bind to a local: a concurrent clear_cache() may set
        # the global back to None between the assignment and the return.
        records = _records
        if records is None:
            records, shadowed, pending = _discover()
            _records, _shadowed = records, shadowed

    _emit_discovery_warnings(pending)
    return records


def _emit_discovery_warnings(messages: list[str]) -> None:
    """Deliver discovery warnings outside the lock, best effort.

    Every message is logged unconditionally, because a shadowing attempt is a supply-chain
    signal and ``warnings`` may be filtered to silence, deduplicated to once per location,
    or configured to raise. The ``warnings`` call is then best effort: warning *policy* must
    never decide whether a backend resolves.

    Args:
        messages (list[str]): Warning texts collected during discovery.

    """
    for message in messages:
        logger.warning("%s", message)
        # Best effort: a filter set to "error", or a custom showwarning that raises, must not
        # decide whether a backend resolves. The logger call above is the guaranteed channel.
        with contextlib.suppress(Exception):
            warnings.warn(message, BackendShadowWarning, stacklevel=3)


def _validate(obj: Any, record: _PluginRecord) -> type[SandboxBackendPlugin]:
    """Check a loaded entry point really is a usable backend plugin.

    Args:
        obj (Any): Whatever the entry point resolved to.
        record (_PluginRecord): Bookkeeping for the entry point, used in error messages.

    Returns:
        type[SandboxBackendPlugin]: The validated plugin class.

    Raises:
        BackendLoadError: If the object is not a concrete `SandboxBackendPlugin` subclass,
            or targets a plugin API version this release does not accept.

    """
    source = f"{record.distribution or '<unknown distribution>'} (entry point {record.entry_point.value!r})"

    if not isinstance(obj, type) or not issubclass(obj, SandboxBackendPlugin):
        raise BackendLoadError(
            record.name,
            f"Backend {record.name!r} from {source} is not usable: the entry point must resolve to a "
            f"subclass of llm_sandbox.backends.SandboxBackendPlugin, but it resolved to "
            f"{type(obj).__name__}. Point the entry point at the class itself, not an instance.",
            record.distribution,
        )

    if inspect.isabstract(obj):
        missing = ", ".join(sorted(obj.__abstractmethods__))
        raise BackendLoadError(
            record.name,
            f"Backend {record.name!r} from {source} is incomplete: it does not implement {missing}.",
            record.distribution,
        )

    if not getattr(obj, "name", None):
        # `require()` and the default optional factories all read `cls.name`. Without this
        # check the failure is a bare AttributeError from deep inside a call, which is the
        # exact outcome declaring PLUGIN_API_VERSION exists to prevent.
        raise BackendLoadError(
            record.name,
            f"Backend {record.name!r} from {source} does not declare a `name` class attribute. "
            f"Add `name = {record.name!r}` to the plugin class.",
            record.distribution,
        )

    declared = getattr(obj, "PLUGIN_API_VERSION", None)
    if declared is None:
        raise BackendLoadError(
            record.name,
            f"Backend {record.name!r} from {source} does not declare PLUGIN_API_VERSION. "
            f"Add `PLUGIN_API_VERSION = {sorted(SUPPORTED_PLUGIN_API_VERSIONS)[-1]}` to the plugin class. "
            f"This release accepts plugin API version(s): "
            f"{', '.join(str(v) for v in sorted(SUPPORTED_PLUGIN_API_VERSIONS))}.",
            record.distribution,
        )

    if declared not in SUPPORTED_PLUGIN_API_VERSIONS:
        supported = ", ".join(str(v) for v in sorted(SUPPORTED_PLUGIN_API_VERSIONS))
        raise BackendLoadError(
            record.name,
            f"Backend {record.name!r} from {source} targets plugin API version {declared}, "
            f"which this release of llm-sandbox does not support (it accepts: {supported}). "
            f"Upgrade or downgrade {record.distribution or 'the plugin'}, or change the version of "
            f"llm-sandbox. See {INTEGRATIONS_URL}",
            record.distribution,
        )

    declared_name = getattr(obj, "name", None)
    if declared_name and normalize_backend_name(str(declared_name)) != record.name:
        warnings.warn(
            f"Backend plugin from {source} declares name {declared_name!r} but is registered under "
            f"entry point name {record.name!r}. The entry point name is used.",
            BackendNameMismatchWarning,
            stacklevel=2,
        )

    return obj


def _load(record: _PluginRecord) -> type[SandboxBackendPlugin]:
    """Import and validate one plugin, caching the result.

    Args:
        record (_PluginRecord): The discovered entry point to load.

    Returns:
        type[SandboxBackendPlugin]: The plugin class.

    Raises:
        BackendLoadError: If importing the entry point fails for any reason.

    """
    cached = _loaded.get(record.name)
    if cached is not None:
        return cached

    try:
        obj = record.entry_point.load()
    except (BackendLoadError, KeyboardInterrupt):
        raise
    except BaseException as exc:
        # BaseException, not Exception: a plugin doing sys.exit() on missing config raises
        # SystemExit, and failure isolation has to hold for that too.
        source = record.distribution or "<unknown distribution>"
        raise BackendLoadError(
            record.name,
            f"Backend {record.name!r} from {source} failed to load: {type(exc).__name__}: {exc}\n"
            f"This is a problem with the plugin, not with llm-sandbox. Other backends are unaffected.",
            record.distribution,
        ) from exc

    plugin = _validate(obj, record)
    _loaded[record.name] = plugin
    return plugin


def _unknown_backend_error(key: str, requested: str) -> BackendNotFoundError:
    """Build the message shown when a backend name resolves to nothing.

    For most plugin users this message is the whole onboarding experience, so it names what
    is available, what is installed, and what to install.

    Args:
        key (str): The normalised backend name.
        requested (str): The name exactly as the caller typed it.

    Returns:
        BackendNotFoundError: The error, ready to raise.

    """
    records = _get_records()
    lines = [
        f"Unknown backend {requested!r}.",
        f"Built-in backends: {', '.join(sorted(BUILTIN_BACKENDS))}.",
    ]

    if records:
        installed = ", ".join(
            f"{name} ({record.distribution or 'unknown'} {record.version or '?'})"
            for name, record in sorted(records.items())
        )
        lines.append(f"Installed plugin backends: {installed}.")

    import difflib

    close = difflib.get_close_matches(key, [*BUILTIN_BACKENDS, *records], n=1, cutoff=_DID_YOU_MEAN_CUTOFF)
    if close:
        lines.append(f"Did you mean {close[0]!r}?")
    elif BACKEND_NAME_PATTERN.match(key):
        # Only suggest a command for a name that survived validation. This line is written to
        # be pasted into a shell, so it must never carry through whatever the caller passed.
        lines.append(
            f"No installed package provides {requested!r}. Third-party backends ship as separate\n"
            f"packages — try: pip install llm-sandbox-{key.replace('_', '-')}"
        )
    else:
        lines.append(f"{requested!r} is not a usable backend name. Names must match {BACKEND_NAME_PATTERN.pattern}.")

    lines.append(f"See {INTEGRATIONS_URL}")
    return BackendNotFoundError(requested, "\n".join(lines))


def get_backend(name: str) -> type[SandboxBackendPlugin]:
    """Resolve a backend name to its provider, loading a plugin only if necessary.

    Entry point metadata is scanned once per process and cached. Plugin code is imported
    only when that plugin's backend is the one being requested.

    Args:
        name (str): A backend name. Case-insensitive; hyphens and underscores are
            equivalent. Accepts `llm_sandbox.const.SandboxBackend` members, which are
            strings.

    Returns:
        type[SandboxBackendPlugin]: The provider for that backend.

    Raises:
        BackendNotFoundError: If nothing provides the name.
        BackendNameConflictError: If several installed distributions claim the name.
        BackendLoadError: If the plugin's entry point cannot be loaded or is malformed.

    Examples:
        ```python
        from llm_sandbox.registry import get_backend

        provider = get_backend("docker")
        session = provider.create_session(lang="python")
        ```

    """
    requested = str(name)
    key = normalize_backend_name(requested)

    # Discovery runs even when a built-in will win, so that a package attempting to shadow
    # a built-in name is reported. That shape is a supply-chain attack, and a user who only
    # ever asks for "docker" is exactly the user who needs to hear about it. Metadata is
    # scanned once per process and cached; no plugin code is imported.
    records = _get_records()

    if not BACKEND_NAME_PATTERN.match(key):
        # Fail closed. `backend=None` and `backend=""` are what a caller passes when a config
        # value or environment variable is unset, and they must never be able to select
        # anything -- least of all a name an installed package chose to answer to.
        raise _unknown_backend_error(key, requested)

    builtin = BUILTIN_BACKENDS.get(key)
    if builtin is not None:
        return builtin

    record = records.get(key)
    if record is None:
        raise _unknown_backend_error(key, requested)

    if record.conflicts:
        claimants = ", ".join(sorted(record.conflicts))
        raise BackendNameConflictError(
            requested,
            f"Backend name {key!r} is claimed by more than one installed distribution: {claimants}. "
            f"Uninstall all but one of them. Other backends are unaffected.",
            record.conflicts,
        )

    return _load(record)


def list_backends(load: bool = False) -> list[BackendInfo]:
    """List every backend core can see, built-in and installed plugins alike.

    By default this reads distribution metadata only and imports no plugin code, so it is
    safe to call in any context.

    Args:
        load (bool): Import each plugin to populate `BackendInfo.capabilities`. Plugins that
            fail to load are reported with ``status="error"`` rather than raising, so one
            broken plugin never hides the rest.

    Returns:
        list[BackendInfo]: One entry per visible backend, built-ins first, then plugins,
            each alphabetically. Shadowed plugin registrations are listed after those.

    Examples:
        ```python
        import llm_sandbox

        for backend in llm_sandbox.list_backends():
            origin = "built-in" if backend.is_builtin else f"{backend.distribution} {backend.version}"
            print(f"{backend.name:<16} {origin}")
        ```

    """
    infos: list[BackendInfo] = [
        BackendInfo(
            name=name,
            is_builtin=True,
            capabilities=frozenset(provider.capabilities),
        )
        for name, provider in sorted(BUILTIN_BACKENDS.items())
    ]

    for name, record in sorted(_get_records().items()):
        capabilities: frozenset[BackendCapability] | None = None
        status: BackendStatus = "ok"
        detail: str | None = None

        if record.conflicts:
            status = "conflict"
            detail = f"Claimed by: {', '.join(sorted(record.conflicts))}"
        elif load:
            try:
                capabilities = frozenset(_load(record).capabilities)
            except KeyboardInterrupt:
                raise
            except BaseException as exc:  # noqa: BLE001
                status = "error"
                detail = str(exc)

        infos.append(
            BackendInfo(
                name=name,
                is_builtin=False,
                distribution=record.distribution,
                version=record.version,
                entry_point=record.entry_point.value,
                capabilities=capabilities,
                status=status,
                detail=detail,
            )
        )

    infos.extend(_shadowed)  # atomically swapped tuple; safe to read unlocked
    return infos
