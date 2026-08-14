"""Tests for backend discovery, resolution, and error reporting."""

import contextlib
import logging
import sys
import threading
import time
import warnings
from pathlib import Path

import pytest

from llm_sandbox import list_backends, registry
from llm_sandbox.backends.builtin import BUILTIN_BACKENDS, DockerBackend, MicromambaBackend
from llm_sandbox.backends.plugin import BackendCapability, SandboxBackendPlugin, normalize_backend_name
from llm_sandbox.exceptions import (
    BackendCapabilityError,
    BackendLoadError,
    BackendNameConflictError,
    BackendNotFoundError,
    UnsupportedBackendError,
)
from llm_sandbox.registry import BackendNameMismatchWarning, BackendShadowWarning, get_backend
from tests.plugin_helpers import (
    ABSTRACT_PLUGIN,
    EXPLODING_PLUGIN,
    FUTURE_VERSION_PLUGIN,
    MISMATCHED_NAME_PLUGIN,
    NO_NAME_PLUGIN,
    NO_VERSION_PLUGIN,
    NOT_A_PLUGIN,
    POOLING_PLUGIN,
    SYSTEM_EXIT_PLUGIN,
    VALID_PLUGIN,
    installed,
    write_distribution,
)


@pytest.fixture(autouse=True)
def _clean_registry() -> None:
    """Keep discovery state from leaking between tests."""
    registry.clear_cache()


class TestNameNormalisation:
    """Backend names are case-insensitive and hyphen/underscore agnostic."""

    @pytest.mark.parametrize(
        ("raw", "expected"),
        [
            ("docker", "docker"),
            ("Docker", "docker"),
            ("DOCKER", "docker"),
            ("my-service", "my_service"),
            ("my_service", "my_service"),
            ("MY-SERVICE", "my_service"),
            ("  spaced  ", "spaced"),
        ],
    )
    def test_normalisation(self, raw: str, expected: str) -> None:
        """Normalisation."""
        assert normalize_backend_name(raw) == expected

    def test_builtin_resolves_through_variants(self) -> None:
        """Builtin resolves through variants."""
        for variant in ("docker", "Docker", "DOCKER", " docker "):
            assert get_backend(variant) is DockerBackend


class TestBuiltins:
    """Built-in backends resolve without touching entry points at all."""

    def test_all_four_builtins_resolve(self) -> None:
        """All four builtins resolve."""
        for name in ("docker", "kubernetes", "podman", "micromamba"):
            assert issubclass(get_backend(name), SandboxBackendPlugin)

    def test_discovery_runs_once_and_is_cached(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Metadata is scanned once per process, however many backends are resolved."""
        calls: list[int] = []

        def counting() -> list[object]:
            calls.append(1)
            return []

        monkeypatch.setattr(registry, "_entry_points", counting)

        get_backend("docker")
        get_backend("podman")
        get_backend("docker")

        assert len(calls) == 1, f"Entry points scanned {len(calls)} times; expected exactly one cached scan"

    def test_micromamba_declares_no_interactive_or_pooling(self) -> None:
        """Preserves long-standing behaviour: neither has ever worked for micromamba."""
        assert not MicromambaBackend.supports(BackendCapability.INTERACTIVE)
        assert not MicromambaBackend.supports(BackendCapability.POOLING)
        assert MicromambaBackend.supports(BackendCapability.ARTIFACTS)

    def test_require_raises_backend_capability_error(self) -> None:
        """Require raises backend capability error."""
        with pytest.raises(BackendCapabilityError) as excinfo:
            MicromambaBackend.require(BackendCapability.POOLING)

        assert isinstance(excinfo.value, UnsupportedBackendError)
        assert "pooling" in str(excinfo.value)


class TestUnknownBackendMessage:
    """The unknown-backend message is most plugin users' entire onboarding experience."""

    def test_names_builtins_and_suggests_install(self) -> None:
        """Names builtins and suggests install."""
        with pytest.raises(BackendNotFoundError) as excinfo:
            get_backend("tenki")

        message = str(excinfo.value)
        assert "Unknown backend 'tenki'." in message
        assert "Built-in backends: docker, kubernetes, micromamba, podman." in message
        assert "No installed package provides 'tenki'." in message
        assert "pip install llm-sandbox-tenki" in message
        assert "INTEGRATIONS.md" in message

    def test_subclasses_unsupported_backend_error_for_compatibility(self) -> None:
        """Downstream `except UnsupportedBackendError` must keep working."""
        with pytest.raises(UnsupportedBackendError):
            get_backend("nope")

    def test_underscored_name_suggests_hyphenated_package(self) -> None:
        """Underscored name suggests hyphenated package."""
        with pytest.raises(BackendNotFoundError) as excinfo:
            get_backend("my_service")

        assert "pip install llm-sandbox-my-service" in str(excinfo.value)

    def test_typo_of_builtin_gets_did_you_mean(self) -> None:
        """Typo of builtin gets did you mean."""
        with pytest.raises(BackendNotFoundError) as excinfo:
            get_backend("dokcer")

        assert "Did you mean 'docker'?" in str(excinfo.value)

    def test_typo_of_installed_plugin_lists_it(self, tmp_path: Path) -> None:
        """Typo of installed plugin lists it."""
        root = write_distribution(
            tmp_path / "site",
            distribution="llm-sandbox-tenki",
            version="0.1.0",
            module="tenki_backend",
            source=VALID_PLUGIN,
            entry_point_name="tenki",
        )
        with installed(root), pytest.raises(BackendNotFoundError) as excinfo:
            get_backend("tenkki")

        message = str(excinfo.value)
        assert "Installed plugin backends: tenki (llm-sandbox-tenki 0.1.0)." in message
        assert "Did you mean 'tenki'?" in message


class TestDiscoveryAndLoading:
    """Discovery reads metadata; loading imports. They must stay separate."""

    def test_plugin_resolves_and_creates_session(self, tmp_path: Path) -> None:
        """Plugin resolves and creates session."""
        root = write_distribution(
            tmp_path / "site",
            distribution="llm-sandbox-tenki",
            version="1.2.3",
            module="tenki_backend",
            source=VALID_PLUGIN,
            entry_point_name="tenki",
        )
        with installed(root):
            provider = get_backend("tenki")

            assert provider.name == "tenki"
            assert provider.create_session(lang="python") == {"backend": "tenki", "kwargs": {"lang": "python"}}

    def test_plugin_resolves_through_name_variants(self, tmp_path: Path) -> None:
        """Plugin resolves through name variants."""
        root = write_distribution(
            tmp_path / "site",
            distribution="llm-sandbox-my-service",
            version="0.1.0",
            module="my_service_backend",
            source=VALID_PLUGIN,
            entry_point_name="my-service",
        )
        with installed(root):
            provider = get_backend("my-service")
            assert get_backend("my_service") is provider
            assert get_backend("MY_SERVICE") is provider

    def test_discovery_does_not_import_the_plugin(self, tmp_path: Path) -> None:
        """Listing backends must never execute third-party code."""
        root = write_distribution(
            tmp_path / "site",
            distribution="llm-sandbox-tenki",
            version="0.1.0",
            module="tenki_backend",
            source=VALID_PLUGIN,
            entry_point_name="tenki",
        )
        with installed(root):
            backends = list_backends()
            assert "tenki" in {info.name for info in backends}
            assert "tenki_backend" not in sys.modules, "list_backends() imported the plugin module"

            get_backend("tenki")
            assert "tenki_backend" in sys.modules, "get_backend() did not import the plugin"

    def test_load_is_cached(self, tmp_path: Path) -> None:
        """Load is cached."""
        root = write_distribution(
            tmp_path / "site",
            distribution="llm-sandbox-tenki",
            version="0.1.0",
            module="tenki_backend",
            source=VALID_PLUGIN,
            entry_point_name="tenki",
        )
        with installed(root):
            assert get_backend("tenki") is get_backend("tenki")

    def test_importing_llm_sandbox_does_not_load_plugins(self, tmp_path: Path) -> None:
        """The security property: installing a plugin must not get it imported on start-up."""
        root = write_distribution(
            tmp_path / "site",
            distribution="llm-sandbox-tenki",
            version="0.1.0",
            module="tenki_backend",
            source=VALID_PLUGIN,
            entry_point_name="tenki",
        )
        with installed(root):
            import llm_sandbox

            llm_sandbox.create_session  # noqa: B018
            assert "tenki_backend" not in sys.modules


class TestFailureIsolation:
    """A broken plugin fails only the request that names it."""

    @pytest.mark.parametrize(
        ("source", "expected_fragment"),
        [
            (EXPLODING_PLUGIN, "failed to load"),
            (NOT_A_PLUGIN, "must resolve to a subclass"),
            (NO_VERSION_PLUGIN, "does not declare PLUGIN_API_VERSION"),
            (FUTURE_VERSION_PLUGIN, "targets plugin API version 99"),
            (ABSTRACT_PLUGIN, "does not implement create_session"),
        ],
    )
    def test_broken_plugin_raises_backend_load_error(self, tmp_path: Path, source: str, expected_fragment: str) -> None:
        """Broken plugin raises backend load error."""
        root = write_distribution(
            tmp_path / "site",
            distribution="llm-sandbox-broken",
            version="0.1.0",
            module="broken_backend",
            source=source,
            entry_point_name="broken",
        )
        with installed(root), pytest.raises(BackendLoadError) as excinfo:
            get_backend("broken")

        message = str(excinfo.value)
        assert expected_fragment in message
        assert "llm-sandbox-broken" in message, "Error must name the offending distribution"

    def test_broken_plugin_does_not_break_other_backends(self, tmp_path: Path) -> None:
        """Broken plugin does not break other backends."""
        broken = write_distribution(
            tmp_path / "broken",
            distribution="llm-sandbox-broken",
            version="0.1.0",
            module="broken_backend",
            source=EXPLODING_PLUGIN,
            entry_point_name="broken",
        )
        good = write_distribution(
            tmp_path / "good",
            distribution="llm-sandbox-tenki",
            version="0.1.0",
            module="tenki_backend",
            source=VALID_PLUGIN,
            entry_point_name="tenki",
        )
        with installed(broken, good):
            assert get_backend("docker") is DockerBackend
            assert get_backend("tenki").name == "tenki"
            with pytest.raises(BackendLoadError):
                get_backend("broken")

    def test_broken_plugin_is_reported_not_raised_by_list_backends(self, tmp_path: Path) -> None:
        """Broken plugin is reported not raised by list backends."""
        root = write_distribution(
            tmp_path / "site",
            distribution="llm-sandbox-broken",
            version="0.1.0",
            module="broken_backend",
            source=EXPLODING_PLUGIN,
            entry_point_name="broken",
        )
        with installed(root):
            infos = {info.name: info for info in list_backends(load=True)}

        assert infos["broken"].status == "error"
        assert "failed to load" in (infos["broken"].detail or "")
        assert infos["docker"].status == "ok"

    def test_unreadable_metadata_does_not_break_builtins(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Unreadable metadata does not break builtins."""

        def explode(**_kwargs: object) -> None:
            msg = "corrupt metadata on sys.path"
            raise RuntimeError(msg)

        monkeypatch.setattr(registry, "entry_points", explode)
        registry.clear_cache()

        assert get_backend("docker") is DockerBackend
        assert [info.name for info in list_backends() if info.is_builtin] == sorted(BUILTIN_BACKENDS)


class TestCollisions:
    """Built-ins always win; two plugins claiming one name is an error naming both."""

    def test_plugin_shadowing_a_builtin_loses_and_warns(self, tmp_path: Path) -> None:
        """Plugin shadowing a builtin loses and warns."""
        root = write_distribution(
            tmp_path / "site",
            distribution="llm-sandbox-impostor",
            version="6.6.6",
            module="impostor_backend",
            source=VALID_PLUGIN,
            entry_point_name="docker",
        )
        with installed(root), pytest.warns(BackendShadowWarning, match="llm-sandbox-impostor"):
            assert get_backend("docker") is DockerBackend

    def test_shadowed_registration_is_listed(self, tmp_path: Path) -> None:
        """Shadowed registration is listed."""
        root = write_distribution(
            tmp_path / "site",
            distribution="llm-sandbox-impostor",
            version="6.6.6",
            module="impostor_backend",
            source=VALID_PLUGIN,
            entry_point_name="docker",
        )
        with installed(root), pytest.warns(BackendShadowWarning):
            shadowed = [info for info in list_backends() if info.status == "shadowed"]

        assert len(shadowed) == 1
        assert shadowed[0].distribution == "llm-sandbox-impostor"
        assert shadowed[0].name == "docker"

    def test_two_distributions_claiming_one_name_is_an_error_naming_both(self, tmp_path: Path) -> None:
        """Two distributions claiming one name is an error naming both."""
        first = write_distribution(
            tmp_path / "first",
            distribution="llm-sandbox-tenki",
            version="1.0.0",
            module="tenki_one",
            source=VALID_PLUGIN,
            entry_point_name="tenki",
        )
        second = write_distribution(
            tmp_path / "second",
            distribution="tenki-sandbox",
            version="2.0.0",
            module="tenki_two",
            source=VALID_PLUGIN,
            entry_point_name="tenki",
        )
        with installed(first, second), pytest.raises(BackendNameConflictError) as excinfo:
            get_backend("tenki")

        message = str(excinfo.value)
        assert "llm-sandbox-tenki" in message
        assert "tenki-sandbox" in message

    def test_conflict_does_not_break_other_backends(self, tmp_path: Path) -> None:
        """Conflict does not break other backends."""
        first = write_distribution(
            tmp_path / "first",
            distribution="llm-sandbox-tenki",
            version="1.0.0",
            module="tenki_one",
            source=VALID_PLUGIN,
            entry_point_name="tenki",
        )
        second = write_distribution(
            tmp_path / "second",
            distribution="tenki-sandbox",
            version="2.0.0",
            module="tenki_two",
            source=VALID_PLUGIN,
            entry_point_name="tenki",
        )
        other = write_distribution(
            tmp_path / "other",
            distribution="llm-sandbox-other",
            version="1.0.0",
            module="other_backend",
            source=VALID_PLUGIN,
            entry_point_name="other",
        )
        with installed(first, second, other):
            assert get_backend("other").name == "other"
            assert get_backend("docker") is DockerBackend

            infos = {info.name: info for info in list_backends()}
            assert infos["tenki"].status == "conflict"
            assert infos["other"].status == "ok"

    def test_declared_name_mismatch_warns_but_resolves(self, tmp_path: Path) -> None:
        """Declared name mismatch warns but resolves."""
        root = write_distribution(
            tmp_path / "site",
            distribution="llm-sandbox-confused",
            version="0.1.0",
            module="confused_backend",
            source=MISMATCHED_NAME_PLUGIN,
            entry_point_name="confused",
        )
        with installed(root), pytest.warns(BackendNameMismatchWarning, match="something-else"):
            assert get_backend("confused") is not None


class TestListBackends:
    """The public discovery helper."""

    def test_lists_all_builtins_with_capabilities(self) -> None:
        """Lists all builtins with capabilities."""
        infos = {info.name: info for info in list_backends()}

        assert set(BUILTIN_BACKENDS) <= set(infos)
        assert infos["docker"].is_builtin is True
        assert infos["docker"].distribution is None
        assert BackendCapability.POOLING in (infos["docker"].capabilities or frozenset())

    def test_reports_plugin_distribution_and_version(self, tmp_path: Path) -> None:
        """Reports plugin distribution and version."""
        root = write_distribution(
            tmp_path / "site",
            distribution="llm-sandbox-tenki",
            version="1.2.3",
            module="tenki_backend",
            source=VALID_PLUGIN,
            entry_point_name="tenki",
        )
        with installed(root):
            info = next(item for item in list_backends() if item.name == "tenki")

        assert info.is_builtin is False
        assert info.distribution == "llm-sandbox-tenki"
        assert info.version == "1.2.3"
        assert info.entry_point == "tenki_backend:Backend"
        assert info.capabilities is None, "capabilities require load=True"

    def test_load_populates_capabilities(self, tmp_path: Path) -> None:
        """Load populates capabilities."""
        root = write_distribution(
            tmp_path / "site",
            distribution="llm-sandbox-tenki",
            version="1.2.3",
            module="tenki_backend",
            source=VALID_PLUGIN,
            entry_point_name="tenki",
        )
        with installed(root):
            info = next(item for item in list_backends(load=True) if item.name == "tenki")

        assert info.capabilities == frozenset({BackendCapability.ARTIFACTS})

    def test_builtins_are_listed_first_and_sorted(self) -> None:
        """Builtins are listed first and sorted."""
        names = [info.name for info in list_backends() if info.is_builtin]
        assert names == sorted(BUILTIN_BACKENDS)


class TestCacheControl:
    """clear_cache() exists for tests and for processes that install plugins at runtime."""

    def test_clear_cache_picks_up_newly_installed_plugin(self, tmp_path: Path) -> None:
        """Clear cache picks up newly installed plugin."""
        assert "late" not in {info.name for info in list_backends()}

        root = write_distribution(
            tmp_path / "site",
            distribution="llm-sandbox-late",
            version="0.1.0",
            module="late_backend",
            source=VALID_PLUGIN,
            entry_point_name="late",
        )
        with installed(root):
            assert "late" in {info.name for info in list_backends()}

        assert "late" not in {info.name for info in list_backends()}


class TestHardenedNames:
    """Degenerate and confusable names must fail closed, not select a plugin."""

    @pytest.mark.parametrize("requested", [None, "", "   ", "  \t ", "foo; rm -rf /", "-", "_x"])
    def test_unusable_names_never_resolve(self, requested: object) -> None:
        """A name that is empty or malformed cannot select any backend.

        `backend=""` and `backend=None` are what a caller passes when a config value or
        environment variable is unset. Those must not be selectable by an installed package.
        """
        with pytest.raises(BackendNotFoundError):
            get_backend(requested)  # type: ignore[arg-type]

    def test_entry_point_with_empty_name_is_refused(self, tmp_path: Path) -> None:
        """An entry point named '' is ignored rather than answering to backend=''."""
        root = tmp_path / "site"
        root.mkdir(parents=True)
        (root / "empty_backend.py").write_text(VALID_PLUGIN.format(name="empty"))
        dist_info = root / "llm_sandbox_empty-0.1.0.dist-info"
        dist_info.mkdir()
        (dist_info / "METADATA").write_text("Metadata-Version: 2.1\nName: llm-sandbox-empty\nVersion: 0.1.0\n")
        (dist_info / "entry_points.txt").write_text("[llm_sandbox.backends]\n = empty_backend:Backend\n")

        with installed(root), pytest.warns(BackendShadowWarning, match="unusable name"):
            names = {info.name for info in list_backends()}

        assert "" not in names

        with installed(root), pytest.raises(BackendNotFoundError):
            get_backend("")

    def test_fullwidth_homoglyph_folds_onto_the_builtin(self) -> None:
        """NFKC folding means a fullwidth homoglyph cannot pose as a separate backend."""
        assert normalize_backend_name("ＤOCKER") == "docker"  # noqa: RUF001
        assert get_backend("ＤOCKER") is DockerBackend  # noqa: RUF001

    def test_shell_metacharacters_get_no_install_suggestion(self) -> None:
        """The pip line is written to be pasted into a shell, so it never echoes junk."""
        with pytest.raises(BackendNotFoundError) as excinfo:
            get_backend("foo; rm -rf /")

        message = str(excinfo.value)
        assert "pip install" not in message
        assert "not a usable backend name" in message


class TestWarningDeliveryCannotBreakDiscovery:
    """Warning policy must never decide whether a backend resolves."""

    def test_warnings_as_errors_does_not_break_resolution(self, tmp_path: Path) -> None:
        """With -W error and a shadowing plugin installed, every backend still resolves."""
        root = write_distribution(
            tmp_path / "site",
            distribution="llm-sandbox-impostor",
            version="6.6.6",
            module="impostor_backend",
            source=VALID_PLUGIN,
            entry_point_name="docker",
        )
        with installed(root), warnings.catch_warnings():
            warnings.simplefilter("error")

            assert get_backend("docker") is DockerBackend
            assert get_backend("podman").name == "podman"
            assert [info.name for info in list_backends() if info.is_builtin] == sorted(BUILTIN_BACKENDS)

    def test_discovery_is_cached_even_when_a_warning_fires(self, tmp_path: Path) -> None:
        """A shadowing plugin must not force a metadata re-scan on every call."""
        root = write_distribution(
            tmp_path / "site",
            distribution="llm-sandbox-impostor",
            version="6.6.6",
            module="impostor_backend",
            source=VALID_PLUGIN,
            entry_point_name="docker",
        )
        with installed(root), warnings.catch_warnings():
            warnings.simplefilter("error")
            with contextlib.suppress(Exception):
                get_backend("docker")

            scans = []
            original = registry._entry_points

            def counting() -> list[object]:
                scans.append(1)
                return original()

            registry._entry_points = counting  # type: ignore[assignment]
            try:
                get_backend("docker")
                get_backend("podman")
            finally:
                registry._entry_points = original  # type: ignore[assignment]

        assert scans == [], "Discovery re-scanned metadata after a warning fired"

    def test_shadow_attempt_is_logged_even_when_warnings_are_silenced(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        """A hijack attempt is a supply-chain signal, so it survives a warnings filter."""
        root = write_distribution(
            tmp_path / "site",
            distribution="llm-sandbox-impostor",
            version="6.6.6",
            module="impostor_backend",
            source=VALID_PLUGIN,
            entry_point_name="docker",
        )
        with (
            installed(root),
            warnings.catch_warnings(),
            caplog.at_level(logging.WARNING, logger="llm_sandbox.registry"),
        ):
            warnings.simplefilter("ignore")
            get_backend("docker")

        assert "llm-sandbox-impostor" in caplog.text
        assert "shadows the built-in" in caplog.text


class TestFailureIsolationIsTotal:
    """Isolation has to hold for BaseException too, not just Exception."""

    def test_plugin_calling_sys_exit_does_not_kill_the_process(self, tmp_path: Path) -> None:
        """SystemExit inherits BaseException; a plugin must not be able to exit the host."""
        root = write_distribution(
            tmp_path / "site",
            distribution="llm-sandbox-suicidal",
            version="0.1.0",
            module="suicidal_backend",
            source=SYSTEM_EXIT_PLUGIN,
            entry_point_name="suicidal",
        )
        with installed(root):
            with pytest.raises(BackendLoadError, match="failed to load"):
                get_backend("suicidal")

            infos = {info.name: info for info in list_backends(load=True)}
            assert infos["suicidal"].status == "error"
            assert infos["docker"].status == "ok"

    def test_plugin_without_a_name_is_rejected_at_load(self, tmp_path: Path) -> None:
        """`name` is read by require() and the optional factories, so absence is fatal early."""
        root = write_distribution(
            tmp_path / "site",
            distribution="llm-sandbox-nameless",
            version="0.1.0",
            module="nameless_backend",
            source=NO_NAME_PLUGIN,
            entry_point_name="nameless",
        )
        with installed(root), pytest.raises(BackendLoadError, match="does not declare a `name`"):
            get_backend("nameless")


class TestConcurrency:
    """Pool code creates sessions from worker threads, so the cache is used concurrently."""

    def test_concurrent_resolution_is_consistent(self, tmp_path: Path) -> None:
        """Many threads resolving at once agree, and discovery still runs once."""
        root = write_distribution(
            tmp_path / "site",
            distribution="llm-sandbox-tenki",
            version="1.0.0",
            module="tenki_backend",
            source=VALID_PLUGIN,
            entry_point_name="tenki",
        )
        with installed(root):
            results: list[object] = []
            errors: list[BaseException] = []

            def resolve() -> None:
                try:
                    results.extend(get_backend(name) for name in ("docker", "tenki", "podman"))
                except BaseException as exc:  # noqa: BLE001
                    errors.append(exc)

            threads = [threading.Thread(target=resolve) for _ in range(16)]
            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join()

        assert not errors, f"Concurrent resolution raised: {errors}"
        assert len(results) == 16 * 3

    def test_clear_cache_racing_resolution_never_returns_none(self, tmp_path: Path) -> None:
        """_get_records() must not hand back None when clear_cache() lands mid-call."""
        root = write_distribution(
            tmp_path / "site",
            distribution="llm-sandbox-tenki",
            version="1.0.0",
            module="tenki_backend",
            source=VALID_PLUGIN,
            entry_point_name="tenki",
        )
        errors: list[BaseException] = []
        stop = threading.Event()

        def clear() -> None:
            while not stop.is_set():
                registry.clear_cache()

        def resolve() -> None:
            try:
                while not stop.is_set():
                    get_backend("docker")
                    list_backends()
            except BaseException as exc:  # noqa: BLE001
                errors.append(exc)

        with installed(root):
            workers = [threading.Thread(target=clear) for _ in range(3)]
            workers += [threading.Thread(target=resolve) for _ in range(3)]
            for thread in workers:
                thread.start()
            time.sleep(0.75)
            stop.set()
            for thread in workers:
                thread.join()

        assert not errors, f"Racing clear_cache() with resolution raised: {errors}"


class TestPluginPooling:
    """POOLING must be reachable end to end, not just declarable."""

    def test_plugin_pool_manager_is_created_and_stamped(self, tmp_path: Path) -> None:
        """create_pool_manager routes to the plugin and stamps the resolved backend name."""
        from llm_sandbox.pool import create_pool_manager

        root = write_distribution(
            tmp_path / "site",
            distribution="llm-sandbox-pooler",
            version="0.1.0",
            module="pooler_backend",
            source=POOLING_PLUGIN,
            entry_point_name="pooler",
        )
        with installed(root):
            manager = create_pool_manager(backend="pooler", lang="python")

            assert type(manager).__name__ == "FakePoolManager"
            assert manager.backend_name == "pooler"

    def test_pooled_session_routes_back_to_the_plugin(self, tmp_path: Path) -> None:
        """A PooledSandboxSession built on a plugin pool resolves through the registry.

        Previously this raised a bare RuntimeError from _infer_backend_from_pool, which made
        the POOLING capability undeliverable for any third-party backend.
        """
        from llm_sandbox.pool import create_pool_manager
        from llm_sandbox.pool.session import PooledSandboxSession

        root = write_distribution(
            tmp_path / "site",
            distribution="llm-sandbox-pooler",
            version="0.1.0",
            module="pooler_backend",
            source=POOLING_PLUGIN,
            entry_point_name="pooler",
        )
        with installed(root):
            manager = create_pool_manager(backend="pooler", lang="python")
            session = PooledSandboxSession(pool_manager=manager)

            assert session.backend == "pooler"
            assert session._create_backend_session("container-1") == {"backend": "pooler"}

    def test_builtin_pool_managers_declare_their_own_name(self) -> None:
        """PodmanPoolManager subclasses DockerPoolManager, so it must declare its own name."""
        from llm_sandbox.pool.docker_pool import DockerPoolManager
        from llm_sandbox.pool.kubernetes_pool import KubernetesPoolManager
        from llm_sandbox.pool.podman_pool import PodmanPoolManager

        assert DockerPoolManager.backend_name == "docker"
        assert KubernetesPoolManager.backend_name == "kubernetes"
        assert PodmanPoolManager.backend_name == "podman", (
            "PodmanPoolManager inherits DockerPoolManager; without its own backend_name a "
            "pooled Podman session would be routed to the Docker backend."
        )
