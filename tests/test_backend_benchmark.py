"""Correctness tests with in-memory adapters; these are not performance tests."""

# Test names describe the assertions; fake adapter methods mirror the public API.
# ruff: noqa: D102, D107

from __future__ import annotations

import ast
import contextlib
import hashlib
import io
import json
import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest

from benchmarks import backend_benchmark as benchmark


class FakeSession:
    """Model file-copy and output contracts without running an external runtime."""

    def __init__(
        self, *, corrupt_upload: bool = False, corrupt_download: bool = False, fail_open: bool = False
    ) -> None:
        self.config = SimpleNamespace(workdir="/sandbox")
        self.files: dict[str, bytes] = {}
        self.corrupt_upload = corrupt_upload
        self.corrupt_download = corrupt_download
        self.fail_open = fail_open
        self.closed = False
        self.calls: list[str] = []

    def open(self) -> None:
        self.calls.append("open")
        if self.fail_open:
            msg = "runtime unavailable"
            raise RuntimeError(msg)

    def close(self) -> None:
        self.calls.append("close")
        self.closed = True

    def run(self, code: str, timeout: float) -> SimpleNamespace:
        self.calls.append("run")
        if code == benchmark.ENVIRONMENT_CODE:
            stdout = json.dumps({"python": "fake-python", "cwd": "/sandbox"})
        elif "hashlib.sha256" in code:
            filename = ast.literal_eval(code.split("open(")[1].split(", 'rb'")[0])
            stdout = hashlib.sha256(self.files["/sandbox/" + filename]).hexdigest()
        elif code.startswith("total = 0"):
            count = int(code.split("range(")[1].split(")")[0])
            stdout = str(sum(i * i for i in range(count)) % 1_000_000_007)
        else:
            stdout = benchmark.MARKER
        return SimpleNamespace(stdout=stdout + "\n", stderr="", exit_code=0)

    def copy_to_runtime(self, src: str, dest: str) -> None:
        self.calls.append("copy_to")
        self.files[dest] = b"corrupted" if self.corrupt_upload else Path(src).read_bytes()

    def copy_from_runtime(self, src: str, dest: str) -> None:
        self.calls.append("copy_from")
        Path(dest).write_bytes(b"corrupted" if self.corrupt_download else self.files[src])


def definition(platforms: tuple[str, ...] = ()) -> SimpleNamespace:
    """Supply the small registry interface used by the runner."""
    return SimpleNamespace(name="fake", capabilities=SimpleNamespace(platforms=platforms))


class TestBenchmark:
    """Verify error exclusion, lifecycle cleanup and independent correctness checks."""

    def test_summary_excludes_failed_trials_and_invalid_samples(self) -> None:
        trials = [
            {
                "status": "passed",
                "samples": [{"metric": "run", "valid": True, "seconds": value} for value in range(1, 21)],
            },
            {"status": "failed", "samples": [{"metric": "run", "valid": True, "seconds": 10000}]},
            {"status": "running", "samples": [{"metric": "run", "valid": True, "seconds": 20000}]},
            {"status": "passed", "samples": [{"metric": "run", "valid": False, "seconds": 30000}]},
        ]
        result = benchmark.summarize(trials)["run"]
        assert result["n"] == 20
        assert result["median_seconds"] == 10.5
        assert result["p95_seconds"] == 19

    def test_validation_is_outside_timer_and_failure_is_retained(self) -> None:
        trial = {"samples": []}

        def invalid(_: object) -> None:
            msg = "wrong result"
            raise ValueError(msg)

        with (
            patch.object(benchmark.time, "perf_counter", side_effect=[10.0, 10.2, 99.0]),
            pytest.raises(ValueError, match="wrong result"),
        ):
            benchmark.timed(trial, "run", lambda: "bad", invalid)
        assert trial["samples"][0]["seconds"] == pytest.approx(0.2)
        assert not trial["samples"][0]["valid"]
        assert trial["samples"][0]["error"]["type"] == "ValueError"

    def test_output_validation_rejects_wrong_value_and_exit_code(self) -> None:
        for stdout, exit_code in [("unexpected", 0), (benchmark.MARKER, 1)]:
            with pytest.raises(ValueError, match="Execution validation failed"):
                benchmark.validate_output(
                    SimpleNamespace(stdout=stdout, stderr="", exit_code=exit_code), benchmark.MARKER
                )

    def test_cpu_oracle_matches_workload(self) -> None:
        code, expected = benchmark.cpu_workload(113)
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            exec(code, {})  # noqa: S102 - locally generated fixed arithmetic, no external code.
        assert output.getvalue().strip() == expected

    def test_platform_skip_never_constructs(self) -> None:
        record = {}

        def fail_if_called(*_: object) -> None:
            pytest.fail("wrong-platform adapter was constructed")

        benchmark.benchmark_backend(
            definition(("imaginary-os",)), {}, benchmark.Settings(), record, factory=fail_if_called
        )
        assert record["status"] == "skipped"
        assert record["trials"] == []

    def test_preflight_excluded_and_samples_are_complete(self) -> None:
        sessions = []

        def factory(*_: object) -> FakeSession:
            session = FakeSession()
            sessions.append(session)
            return session

        record = {}
        settings = benchmark.Settings(trials=2, iterations=3, warmups=1, cpu_iterations=10)
        benchmark.benchmark_backend(definition(), {}, settings, record, factory=factory, progress=lambda _: None)
        assert record["status"] == "passed"
        assert len(sessions) == 3
        assert all(session.closed for session in sessions)
        for metric in ("construct", "open", "first_run", "close"):
            assert record["summary"][metric]["n"] == 2
        for metric in ("warm_run", "cpu_run", "copy_to_1mib", "copy_from_1mib"):
            assert record["summary"][metric]["n"] == 6
        assert record["preflight"]["runtime_environment"]["python"] == "fake-python"
        assert not record["preflight"]["measured"]

    @pytest.mark.parametrize("failure", ["corrupt_upload", "corrupt_download"])
    def test_failed_upload_or_download_excludes_trial_and_closes(self, failure: str) -> None:
        sessions = []

        def factory(*_: object) -> FakeSession:
            session = FakeSession(**{failure: bool(sessions)})
            sessions.append(session)
            return session

        record = {}
        settings = benchmark.Settings(trials=1, iterations=1, cpu_iterations=10)
        benchmark.benchmark_backend(definition(), {}, settings, record, factory=factory, progress=lambda _: None)
        assert record["status"] == "failed"
        assert record["summary"] == {}
        assert "error" in record["trials"][0]
        assert all(session.closed for session in sessions)

    def test_open_failure_is_not_timed_as_success_and_closes(self) -> None:
        session = FakeSession(fail_open=True)
        record = {}
        benchmark.benchmark_backend(
            definition(),
            {},
            benchmark.Settings(),
            record,
            factory=lambda *_: session,
            progress=lambda _: None,
        )
        assert record["status"] == "preflight_failed"
        assert record["trials"] == []
        assert session.closed
        sample = next(sample for sample in record["preflight"]["samples"] if sample["metric"] == "open")
        assert not sample["valid"]
        assert "runtime unavailable" in sample["error"]["message"]

    def test_close_failure_excludes_other_successful_measurements(self) -> None:
        sessions = []

        def factory(*_: object) -> FakeSession:
            session = FakeSession()
            if sessions:

                def fail_close() -> None:
                    msg = "cleanup failed"
                    raise RuntimeError(msg)

                session.close = fail_close
            sessions.append(session)
            return session

        record = {}
        benchmark.benchmark_backend(
            definition(),
            {},
            benchmark.Settings(trials=1, iterations=1, cpu_iterations=10),
            record,
            factory=factory,
            progress=lambda _: None,
        )
        assert record["status"] == "failed"
        assert record["summary"] == {}
        assert "cleanup failed" in record["trials"][0]["close_error"]["message"]

    def test_image_pull_invalidates_measured_open(self) -> None:
        with pytest.raises(ValueError, match="pulled/built"):
            benchmark.validate_no_image_preparation(SimpleNamespace(is_create_template=True), "docker")

    def test_list_prints_every_registered_backend(self, capsys: pytest.CaptureFixture[str]) -> None:
        from llm_sandbox.backends import list_backends

        assert benchmark.main(["--list"]) == 0
        output = capsys.readouterr().out
        for backend in list_backends():
            assert f"{backend.name}: platforms=" in output

    def test_cleanup_lookup_accepts_only_confirmed_not_found(self) -> None:
        client = SimpleNamespace(containers=SimpleNamespace(get=Mock()))
        missing = RuntimeError("container missing")
        missing.status_code = 404
        client.containers.get.side_effect = missing
        benchmark.confirm_container_removed(client, "owned-id")
        client.containers.get.side_effect = ConnectionError("daemon unavailable")
        with pytest.raises(ConnectionError, match="daemon unavailable"):
            benchmark.confirm_container_removed(client, "owned-id")

    def test_swallowed_container_cleanup_error_invalidates_trial(self) -> None:
        session = FakeSession()
        container = SimpleNamespace(id="owned-id", remove=Mock())
        missing = RuntimeError("container missing")
        missing.status_code = 404
        client = SimpleNamespace(containers=SimpleNamespace(get=Mock(side_effect=[container, missing])), close=Mock())
        session.container = container
        session.client = client
        trial = {"status": "passed", "samples": []}
        benchmark.finish_session(session, "docker", trial, lambda: None)
        assert trial["status"] == "failed"
        assert "still exists" in trial["cleanup_verification_error"]["message"]
        assert trial["forced_cleanup"]
        assert trial["cleanup_verified"]
        container.remove.assert_called_once_with(force=True)
        client.close.assert_called_once()
        assert benchmark.summarize([trial]) == {}

    def test_kubernetes_policy_applies_to_sidecars_and_init_containers(self) -> None:
        session = SimpleNamespace(pod_manifest={"spec": {"containers": [{}, {}], "initContainers": [{}]}})
        benchmark.image_policy(session, "kubernetes", preflight=False)
        for container in session.pod_manifest["spec"]["containers"] + session.pod_manifest["spec"]["initContainers"]:
            assert container["imagePullPolicy"] == "Never"
        assert session.pod_manifest["metadata"]["labels"][benchmark.POD_OWNER_LABEL] == session.benchmark_pod_owner

    def test_kubernetes_waits_for_async_removal_without_measuring_close(self) -> None:
        missing = RuntimeError("pod missing")
        missing.status = 404
        pod = SimpleNamespace(metadata=SimpleNamespace(labels={benchmark.POD_OWNER_LABEL: "owner"}, uid="pod-uid"))
        client = SimpleNamespace(read_namespaced_pod=Mock(side_effect=[pod, missing]))
        with (
            patch.object(benchmark.time, "monotonic", side_effect=[0.0, 0.0, 0.1, 0.2]),
            patch.object(benchmark.time, "sleep") as sleep,
        ):
            benchmark.wait_for_pod_removal(client, "pod-name", "namespace", "owner")
        assert client.read_namespaced_pod.call_count == 2
        sleep.assert_called_once_with(0.25)

    def test_kubernetes_removal_wait_is_bounded(self) -> None:
        pod = SimpleNamespace(metadata=SimpleNamespace(labels={benchmark.POD_OWNER_LABEL: "owner"}, uid="pod-uid"))
        client = SimpleNamespace(read_namespaced_pod=Mock(return_value=pod))
        with (
            patch.object(benchmark.time, "monotonic", side_effect=[0.0, 0.0, 60.1, 60.1]),
            patch.object(benchmark.time, "sleep"),
            pytest.raises(TimeoutError, match="60s cleanup wait"),
        ):
            benchmark.wait_for_pod_removal(client, "pod-name", "namespace", "owner")
        client.read_namespaced_pod.assert_called_once()

    def test_kubernetes_failed_open_without_handle_still_cleans_owned_pod(self) -> None:
        missing = RuntimeError("pod missing")
        missing.status = 404
        pod = SimpleNamespace(metadata=SimpleNamespace(labels={benchmark.POD_OWNER_LABEL: "owner"}, uid="pod-uid"))
        session = FakeSession()
        session.container = None
        session.pod_name = "pod-name"
        session.kube_namespace = "namespace"
        session.benchmark_pod_owner = "owner"
        session.client = SimpleNamespace(
            read_namespaced_pod=Mock(side_effect=[pod, missing]), delete_namespaced_pod=Mock()
        )
        trial = {"status": "failed", "stage": "open", "samples": []}
        with patch.object(benchmark.time, "monotonic", side_effect=[0.0, 0.0, 0.0, 0.2]):
            benchmark.finish_session(session, "kubernetes", trial, lambda: None)
        assert session.closed
        assert trial["status"] == "failed"
        assert trial["forced_cleanup"]
        assert trial["cleanup_verified"]
        assert trial["cleanup_wait_seconds"] == pytest.approx(0.2)
        session.client.delete_namespaced_pod.assert_called_once_with(
            name="pod-name",
            namespace="namespace",
            body={"gracePeriodSeconds": 0, "preconditions": {"uid": "pod-uid"}},
            _request_timeout=5.0,
        )

    def test_kubernetes_force_recovery_after_timeout_keeps_trial_invalid(self) -> None:
        pod = SimpleNamespace(metadata=SimpleNamespace(labels={benchmark.POD_OWNER_LABEL: "owner"}, uid="pod-uid"))
        session = SimpleNamespace(
            client=SimpleNamespace(
                read_namespaced_pod=Mock(return_value=pod),
                delete_namespaced_pod=Mock(),
            )
        )
        trial = {"status": "passed", "samples": []}
        with (
            patch.object(benchmark, "wait_for_pod_removal", side_effect=[TimeoutError("still exists"), None]),
            patch.object(benchmark.time, "monotonic", side_effect=[0.0, 60.2]),
        ):
            benchmark.finish_kubernetes(session, "pod-name", "namespace", "owner", trial, open_incomplete=False)
        assert trial["status"] == "failed"
        assert trial["cleanup_verified"]
        assert trial["forced_cleanup"]
        assert trial["cleanup_wait_seconds"] == pytest.approx(60.2)
        assert benchmark.summarize([trial]) == {}

    def test_kubernetes_never_force_deletes_unowned_pod(self) -> None:
        pod = SimpleNamespace(
            metadata=SimpleNamespace(labels={benchmark.POD_OWNER_LABEL: "someone-else"}, uid="other-uid")
        )
        client = SimpleNamespace(read_namespaced_pod=Mock(return_value=pod), delete_namespaced_pod=Mock())
        with pytest.raises(RuntimeError, match="ownership label differs"):
            benchmark.force_delete_owned_pod(client, "pod-name", "namespace", "owner")
        client.delete_namespaced_pod.assert_not_called()

    def test_unconfirmed_cleanup_stops_remaining_backend_trials(self) -> None:
        def run_trial(
            _name: str,
            _kwargs: dict,
            _settings: benchmark.Settings,
            trial: dict,
            *,
            preflight: bool,
            **_ignored: object,
        ) -> None:
            trial.update({"status": "passed"} if preflight else {"status": "failed", "cleanup_verified": False})

        record = {}
        with patch.object(benchmark, "run_trial", side_effect=run_trial):
            benchmark.benchmark_backend(definition(), {}, benchmark.Settings(trials=3), record, progress=lambda _: None)
        assert len(record["trials"]) == 1
        assert record["status"] == "failed"
        assert "removal could not be confirmed" in record["stopped_early"]

    def test_config_defaults_are_backend_specific_and_do_not_mutate_input(self) -> None:
        override = {"image": "prepared:version", "runtime_configs": {"environment": {"A": "B"}}}
        actual = benchmark.effective_kwargs("docker", override)
        assert actual["skip_environment_setup"]
        assert actual["keep_template"]
        assert not actual["stream"]
        actual["runtime_configs"]["environment"]["A"] = "changed"
        assert override["runtime_configs"]["environment"]["A"] == "B"
        assert "stream" not in benchmark.effective_kwargs("kubernetes", {})
        assert "image" not in benchmark.effective_kwargs("bubblewrap", {})
        with pytest.raises(ValueError, match="keep_template must be true"):
            benchmark.effective_kwargs("docker", {"keep_template": False})

    def test_incremental_result_is_valid_json(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            destination = Path(directory) / "results" / "test.json"
            benchmark.save_json(destination, {"status": "running"})
            benchmark.save_json(destination, {"status": "finished"})
            assert json.loads(destination.read_text()) == {"status": "finished"}
            assert not destination.with_suffix(".json.tmp").exists()
