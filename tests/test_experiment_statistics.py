"""Validate experimental units, failure exclusion and deterministic inference."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

import pytest

from benchmarks import experiment_statistics as stats

if TYPE_CHECKING:
    from pathlib import Path


def trial(round_id: int, values: list[float], *, status: str = "passed") -> dict[str, Any]:
    """Create one independent session containing correlated measurements."""
    return {
        "round": round_id,
        "status": status,
        "samples": [{"metric": "warm_run", "valid": True, "seconds": value} for value in values],
    }


def document(backends: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    """Wrap session lists in the expanded runner schema."""
    return {
        "schema_version": 2,
        "profile": "cached",
        "backends": {name: {"status": "passed", "trials": records} for name, records in backends.items()},
    }


def batch(index: int, values: list[float], *, status: str = "passed") -> dict[str, Any]:
    """Represent two concurrent workers sharing a single independent batch."""
    return {
        "index": index,
        "workers": 2,
        "workload": "marker",
        "status": status,
        "expected_calls": len(values) * 2,
        "validated_calls": len(values) * 2,
        "throughput_validated_calls_per_second": 10 + index,
        "execution_seconds": 1.0,
        "initialization_seconds": 0.1,
        "teardown_seconds": 0.2,
        "workers_results": [trial(index, values), trial(index, values)],
    }


def test_bootstrap_is_deterministic_and_handles_constant_values() -> None:
    """A fixed seed reproduces inference; a constant distribution has a point interval."""
    kwargs = {"resamples": 1000, "seed": 17, "key": "warm_run"}
    result = stats.bootstrap_interval([1, 3, 8, 10, 12], **kwargs)
    assert result == stats.bootstrap_interval([1, 3, 8, 10, 12], **kwargs)
    assert stats.bootstrap_interval([4] * 30, **kwargs) == {"lower": 4, "upper": 4}
    assert stats.percentile([0, 10], 0.025) == pytest.approx(0.25)
    assert stats.percentile([0, 10], 0.975) == pytest.approx(9.75)


def test_no_interval_for_single_session_and_small_n_flagged() -> None:
    """Repeated calls do not create independent replication."""
    single = stats.summarize_units([[1] * 1000], unit="session", resamples=50)
    assert single["n_units"] == 1
    assert single["n_samples"] == 1000
    assert single["ci95"] is None
    assert single["uncertainty"].startswith("unavailable")
    five = stats.summarize_units([[1], [2], [3], [4], [5]], unit="batch", resamples=50)
    assert five["ci95"] is not None
    assert five["uncertainty"].startswith("exploratory")


def test_median_of_session_medians_does_not_weight_sessions_by_call_count() -> None:
    """Unbalanced call counts must not turn one session into 100 independent units."""
    result = stats.summarize_units([[1] * 100, [10], [20]], unit="session", resamples=500, seed=3)
    assert result["n_units"] == 3
    assert result["n_samples"] == 102
    assert result["estimate"] == 10
    assert result["pooled_descriptive"]["median"] == 1
    assert result["ci95"] == stats.bootstrap_interval([1, 10, 20], resamples=500, seed=3, key="")


def test_failed_sessions_excluded_but_failure_counts_preserved() -> None:
    """Good samples from failed sessions remain excluded, and errors remain countable."""
    records = [trial(0, [1, 3]), trial(1, [10000], status="failed"), trial(2, [20000], status="running")]
    records[0]["samples"].extend([
        {"metric": "warm_run", "valid": False, "seconds": 999},
        {"metric": "warm_run", "valid": True, "seconds": float("nan")},
        {"metric": "warm_run", "valid": True, "seconds": -1},
        {"metric": "warm_run", "valid": True, "seconds": True},
    ])
    result = stats.summarize_document(document({"bubblewrap": records}), resamples=50)["backends"]["bubblewrap"]
    latency = result["latency_seconds"]["warm_run"]
    assert latency["estimate"] == 2
    assert latency["n_samples"] == 2
    assert result["session_counts"] == {
        "total": 3,
        "passed": 1,
        "failed": 1,
        "other": 1,
        "by_status": {"failed": 1, "passed": 1, "running": 1},
        "samples_total": 8,
        "samples_validated": 7,
        "samples_invalid_or_unvalidated": 1,
    }


def test_paired_bootstrap_resamples_matching_rounds_together() -> None:
    """A constant paired multiplicative effect remains exact after resampling."""
    numerator = {0: 2.0, 1: 200.0, 2: 4.0, 3: 40.0}
    denominator = {0: 1.0, 1: 100.0, 2: 2.0, 3: 20.0, 4: 999.0}
    comparison = stats.paired_comparison(numerator, denominator, resamples=1000, seed=1, key="pair")
    assert comparison["estimate"] == 2
    assert comparison["ci95"] == {"lower": 2, "upper": 2}
    assert comparison["n_pairs"] == 4
    assert comparison["unmatched_rounds"] == 1
    assert comparison["ci95_includes_one"] is False


def test_paired_comparison_reports_overlap_and_excludes_failed_rounds() -> None:
    """Missing and failed trials cannot be paired by list position."""
    source = document({
        "bubblewrap": [trial(0, [1]), trial(1, [2]), trial(2, [3])],
        "nsjail": [trial(0, [1]), trial(1, [2], status="failed"), trial(2, [3])],
    })
    comparison = stats.summarize_document(source, resamples=100)["paired_comparisons"][0]
    assert comparison["rounds"] == [0, 2]
    assert comparison["n_pairs"] == 2
    assert comparison["unmatched_rounds"] == 1
    assert comparison["ci95_includes_one"] is True
    with pytest.raises(ValueError, match="Duplicate passed trial"):
        stats.trial_medians_by_round([trial(0, [1]), trial(0, [2])])


def test_nonpositive_pairs_and_missing_rounds_cannot_produce_invalid_ratios() -> None:
    """Zero-duration readings and absent round IDs never create fake comparisons."""
    result = stats.paired_comparison({0: 0, 1: 1}, {0: 1, 1: 0}, resamples=10, seed=0, key="zeros")
    assert result["nonpositive_pairs_excluded"] == 2
    assert result["estimate"] is None
    assert result["ci95"] is None
    assert stats.trial_medians_by_round([{"status": "passed", "samples": trial(0, [1])["samples"]}]) == {}


def test_concurrency_bootstrap_unit_is_batch_and_failures_remain_visible() -> None:
    """Correlated workers and calls must not inflate the replication count."""
    batches = [batch(index, [float(index + 1)] * 10) for index in range(5)]
    batches.append(batch(5, [999.0] * 100, status="failed"))
    results = stats.summarize_batches(batches, resamples=500, seed=7, key="docker")
    result = results[0]
    assert result["batch_counts"]["total"] == 6
    assert result["batch_counts"]["failed"] == 1
    assert result["worker_counts"]["total"] == 12
    assert result["accepted_batches"] == 5
    latency = result["latency_seconds"]["warm_run"]
    assert latency["n_units"] == 5
    assert latency["n_samples"] == 100
    assert latency["estimate"] == 3
    assert latency["uncertainty"].startswith("exploratory")
    throughput = result["throughput_validated_calls_per_second"]
    assert throughput["estimate"] == 12
    assert throughput["n_units"] == 5
    assert result["validated_calls_all_batches"] == 300


def test_bad_worker_invalidates_entire_batch() -> None:
    """A nominal batch pass cannot hide one failed worker."""
    source = batch(0, [1, 2])
    source["workers_results"][0]["status"] = "failed"
    result = stats.summarize_batches([source], resamples=10, seed=1, key="failure")[0]
    assert result["accepted_batches"] == 0
    assert result["latency_seconds"] == {}
    assert result["worker_counts"]["failed"] == 1
    assert result["throughput_validated_calls_per_second"]["estimate"] is None


def test_nominal_pass_with_missing_worker_or_call_is_excluded() -> None:
    """Status alone cannot admit an incomplete concurrency experiment."""
    missing_worker = batch(0, [1, 2])
    missing_worker["workers_results"].pop()
    missing_call = batch(1, [1, 2])
    missing_call["validated_calls"] -= 1
    assert not stats.batch_passed(missing_worker)
    assert not stats.batch_passed(missing_call)


def test_payload_resources_kept_distinct_from_backend_resource_usage() -> None:
    """Payload telemetry preserves process-level labels and session grouping."""
    source = document({"nsjail": [trial(0, [0.1, 0.2])]})
    samples = source["backends"]["nsjail"]["trials"][0]["samples"]
    for sample in samples:
        sample.update({"payload_cpu_seconds": 0.04, "payload_wall_seconds": 0.05, "payload_peak_rss_kib": 12000})
    summary = stats.summarize_document(source, resamples=10)
    resources = summary["backends"]["nsjail"]["payload_resources"]
    assert resources["payload_cpu_seconds"]["warm_run"]["estimate"] == 0.04
    assert resources["payload_peak_rss_kib"]["warm_run"]["n_units"] == 1
    assert "not whole-sandbox" in summary["methodology"]["resource_scope"]


def test_cli_keeps_profiles_separate_and_writes_inspectable_csv(tmp_path: Path) -> None:
    """Multiple inputs retain their original experimental boundaries and provenance."""
    first = tmp_path / "cached.json"
    second = tmp_path / "setup.json"
    first.write_text(json.dumps(document({"docker": [trial(0, [1])]})))
    setup = document({"docker": [trial(0, [100])]})
    setup["profile"] = "setup"
    second.write_text(json.dumps(setup))
    output = tmp_path / "summaries" / "summary.json"
    csv_path = tmp_path / "summaries" / "summary.csv"
    assert (
        stats.main([str(first), str(second), "--output", str(output), "--csv", str(csv_path), "--resamples", "10"]) == 0
    )
    summaries = json.loads(output.read_text())["experiments"]
    assert [summary["profile"] for summary in summaries] == ["cached", "setup"]
    assert [summary["backends"]["docker"]["latency_seconds"]["warm_run"]["estimate"] for summary in summaries] == [
        1,
        100,
    ]
    assert "independent_unit,n_units,n_samples" in csv_path.read_text()
    assert str(first) in csv_path.read_text()


def test_invalid_resampling_count_is_rejected() -> None:
    """Reject invalid settings before writing partially meaningful output."""
    with pytest.raises(ValueError, match="must be positive"):
        stats.bootstrap_interval([1, 2], resamples=0)
    with pytest.raises(ValueError, match="must be positive"):
        stats.summarize_document(document({}), resamples=0)
