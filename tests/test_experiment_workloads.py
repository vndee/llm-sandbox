"""Validate expanded benchmark oracles and failures without external runtimes."""

# Fake methods mirror the adapter API; test names describe their assertions.
# ruff: noqa: D102, D107

from __future__ import annotations

import ast
import contextlib
import hashlib
import io
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest

from benchmarks import backend_benchmark as benchmark
from benchmarks import experiment_workloads as workloads


def local_output(code: str) -> SimpleNamespace:
    """Execute only generated fixed benchmark code as a lightweight oracle check."""
    output = io.StringIO()
    with contextlib.redirect_stdout(output):
        exec(code, {})  # noqa: S102 - fixed generated workload; no untrusted source.
    return SimpleNamespace(stdout=output.getvalue(), stderr="", exit_code=0)


def resource_output(result: object) -> SimpleNamespace:
    """Construct valid instrumented output for malformed-output and timing tests."""
    return SimpleNamespace(
        stdout=json.dumps({
            "result": result,
            "resources": {"payload_cpu_seconds": 0.01, "payload_wall_seconds": 0.02, "payload_peak_rss_kib": 8192},
            "resource_scope": workloads.RESOURCE_SCOPE,
        }),
        stderr="",
        exit_code=0,
    )


class FakeSession:
    """Model transfers in memory and run only the locally generated task code."""

    def __init__(
        self, *, corrupt_upload: bool = False, corrupt_download: bool = False, fail_open: bool = False
    ) -> None:
        self.config = SimpleNamespace(workdir="/sandbox")
        self.files: dict[str, bytes] = {}
        self.corrupt_upload = corrupt_upload
        self.corrupt_download = corrupt_download
        self.fail_open = fail_open
        self.omit_download = False
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
        if "hashlib.sha256" in code:
            filename = ast.literal_eval(code.split("open(")[1].split(", 'rb'")[0])
            digest = hashlib.sha256(self.files["/sandbox/" + filename]).hexdigest()
            return SimpleNamespace(stdout=digest, stderr="", exit_code=0)
        return local_output(code)

    def copy_to_runtime(self, src: str, dest: str) -> None:
        self.calls.append("copy_to")
        self.files[dest] = b"corrupted" if self.corrupt_upload else Path(src).read_bytes()

    def copy_from_runtime(self, src: str, dest: str) -> None:
        self.calls.append("copy_from")
        if not self.omit_download:
            Path(dest).write_bytes(b"corrupted" if self.corrupt_download else self.files[src])


class TestExpandedWorkloads:
    """Keep correctness, resource scope and lifecycle failures inspectable."""

    @pytest.mark.parametrize("records", [1, 2, 19, 20, 21, 79, 10_000])
    def test_json_arithmetic_oracle_matches_executed_workload(self, records: int) -> None:
        code, expected = workloads.json_workload(records)
        output = local_output(code)
        resources = workloads.validate_instrumented_output(output, expected)
        assert set(resources) == set(workloads.RESOURCE_FIELDS)
        assert resources["payload_peak_rss_kib"] > 0
        assert expected["selected"] == records - (records + 3) // 4
        assert sum(group["count"] for group in expected["groups"]) == expected["selected"]

    def test_json_oracle_known_aggregate_and_partial_cycle(self) -> None:
        assert workloads.json_oracle(6) == {
            "records": 6,
            "selected": 4,
            "groups": [
                {"region": "region-0", "count": 1, "amount_cents": 185},
                {"region": "region-1", "count": 1, "amount_cents": 117},
                {"region": "region-2", "count": 1, "amount_cents": 134},
                {"region": "region-3", "count": 1, "amount_cents": 151},
            ],
        }
        expected = workloads.json_oracle(10_000)
        assert expected["selected"] == 7500
        assert sum(group["amount_cents"] for group in expected["groups"]) == 638_250_000

    @pytest.mark.parametrize("records", [0, -1, True, 1.2])
    def test_json_rejects_invalid_counts(self, records: int) -> None:
        with pytest.raises(ValueError, match="positive integer"):
            workloads.json_workload(records)

    def test_marker_instrumentation_does_not_change_expected_result(self) -> None:
        code, validate = workloads.get_workload("marker")
        assert validate(local_output(code))["payload_wall_seconds"] >= 0
        with pytest.raises(ValueError, match="Unknown instrumented workload"):
            workloads.get_workload("unknown")

    @pytest.mark.parametrize("invalid", [-1, float("nan"), float("inf"), True, "1", None])
    @pytest.mark.parametrize("field", workloads.RESOURCE_FIELDS)
    def test_resource_validator_rejects_invalid_measurements(self, field: str, invalid: object) -> None:
        output = resource_output("expected")
        document = json.loads(output.stdout)
        document["resources"][field] = invalid
        output.stdout = json.dumps(document)
        with pytest.raises(ValueError, match="Invalid payload resource"):
            workloads.validate_instrumented_output(output, "expected")

    def test_resource_validator_rejects_zero_rss_missing_scope_and_wrong_result(self) -> None:
        output = resource_output("expected")
        document = json.loads(output.stdout)
        document["resources"]["payload_peak_rss_kib"] = 0
        output.stdout = json.dumps(document)
        with pytest.raises(ValueError, match="Invalid payload resource"):
            workloads.validate_instrumented_output(output, "expected")
        document.pop("resource_scope")
        output.stdout = json.dumps(document)
        with pytest.raises(ValueError, match="invalid resource scope"):
            workloads.validate_instrumented_output(output, "expected")
        with pytest.raises(ValueError, match="result mismatch"):
            workloads.validate_instrumented_output(resource_output(result=True), 1)

    @pytest.mark.parametrize("stdout", ["not json", "[]", "{}"])
    def test_resource_validator_rejects_malformed_document(self, stdout: str) -> None:
        output = resource_output("expected")
        output.stdout = stdout
        with pytest.raises(ValueError, match="not valid JSON|must be an object"):
            workloads.validate_instrumented_output(output, "expected")

    def test_nonzero_exit_rejected_even_with_correct_output(self) -> None:
        output = resource_output("expected")
        output.exit_code = 1
        with pytest.raises(ValueError, match="exit_code=1"):
            workloads.validate_instrumented_output(output, "expected")

    def test_execute_resources_separate_from_api_timer(self) -> None:
        trial = {"samples": [], "status": "passed"}
        session = SimpleNamespace(run=Mock(return_value=resource_output(benchmark.MARKER)))
        with patch.object(benchmark.time, "perf_counter", side_effect=[10.0, 10.125]):
            workloads.execute_workload(session, trial, benchmark.Settings(), "marker")
        sample = trial["samples"][0]
        assert sample["seconds"] == 0.125
        assert sample["valid"]
        assert sample["payload_cpu_seconds"] == 0.01
        assert trial["resources"][0]["payload_cpu_seconds"] == 0.01
        assert benchmark.summarize([trial])["marker_resource_run"]["median_seconds"] == 0.125

    def test_failed_execution_has_no_valid_resource_sample(self) -> None:
        trial = {"samples": []}
        session = SimpleNamespace(run=Mock(return_value=resource_output("wrong")))
        with pytest.raises(ValueError, match="result mismatch"):
            workloads.execute_workload(session, trial, benchmark.Settings(), "marker")
        assert not trial["samples"][0]["valid"]
        assert "resources" not in trial

    @pytest.mark.parametrize("failure", ["corrupt_upload", "corrupt_download"])
    def test_each_transfer_direction_is_independently_checked(self, tmp_path: Path, failure: str) -> None:
        session = FakeSession(**{failure: True})
        trial = {"samples": []}
        with pytest.raises(ValueError, match="validation failed|SHA-256 mismatch"):
            workloads.transfer_workload(session, trial, tmp_path, 0, 5, payload_bytes=513)
        assert not trial["samples"][-1]["valid"]
        if failure == "corrupt_upload":
            assert "copy_from" not in session.calls
        else:
            assert trial["samples"][0]["valid"]

    def test_reused_sequence_never_accepts_a_stale_download(self, tmp_path: Path) -> None:
        session = FakeSession()
        trial = {"samples": []}
        workloads.transfer_workload(session, trial, tmp_path, 0, 5, payload_bytes=513)
        session.omit_download = True
        with pytest.raises(FileNotFoundError):
            workloads.transfer_workload(session, trial, tmp_path, 0, 5, payload_bytes=513)
        assert len(session.files) == 2
        assert not trial["samples"][-1]["valid"]

    @pytest.mark.parametrize("payload_bytes", [0, -1, True, 1.1])
    def test_transfer_rejects_invalid_sizes(self, tmp_path: Path, payload_bytes: int) -> None:
        with pytest.raises(ValueError, match="positive integer"):
            workloads.transfer_workload(FakeSession(), {"samples": []}, tmp_path, 0, 5, payload_bytes=payload_bytes)

    def test_trial_separates_warmups_and_always_closes(self) -> None:
        session = FakeSession()
        trial = {"samples": []}
        workloads.run_workload_trial(
            "fake",
            {},
            benchmark.Settings(iterations=3, warmups=1),
            trial,
            preflight=False,
            factory=lambda *_: session,
            checkpoint=lambda: None,
            records=79,
            payload_bytes=513,
            transfer_iterations=2,
        )
        assert session.closed
        assert trial["status"] == "passed"
        assert len(trial["resources"]) == 3
        summary = benchmark.summarize([trial])
        assert summary["json_run"]["n"] == 3
        assert summary["copy_to_513bytes"]["n"] == 2
        assert summary["copy_from_513bytes"]["n"] == 2
        assert len(trial["workload_warmup"]["resources"]) == 1
        assert not trial["workload_warmup"]["measured"]

    @pytest.mark.parametrize("failure", ["fail_open", "corrupt_upload", "corrupt_download"])
    def test_trial_failure_excludes_samples_and_closes(self, failure: str) -> None:
        session = FakeSession(**{failure: True})
        trial = {"samples": []}
        workloads.run_workload_trial(
            "fake",
            {},
            benchmark.Settings(iterations=1, warmups=0),
            trial,
            preflight=False,
            factory=lambda *_: session,
            checkpoint=lambda: None,
            records=6,
            payload_bytes=513,
            transfer_iterations=1,
        )
        assert session.closed
        assert trial["status"] == "failed"
        assert benchmark.summarize([trial]) == {}
