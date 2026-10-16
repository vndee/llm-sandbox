"""Summarize randomized backend experiments with session/batch bootstrap intervals.

Run after measurement; this module intentionally imports no runtime SDKs. Calls
within a session and workers within a concurrency batch are correlated, so only
whole sessions or batches are resampled. Separate input files stay separate.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
import math
import random
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Sequence

RESOURCE_FIELDS = ("payload_cpu_seconds", "payload_wall_seconds", "payload_peak_rss_kib")
MIN_DESCRIPTIVE_UNITS = 10
CONFIDENCE_LEVEL = 0.95


def finite_nonnegative(value: Any) -> bool:
    """Exclude booleans, nonnumeric values, infinities and negative durations."""
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value) and value >= 0


def percentile(values: Sequence[float], fraction: float) -> float:
    """Return the linearly interpolated percentile (Hyndman-Fan type 7)."""
    if not values or not 0 <= fraction <= 1:
        msg = "Percentile requires nonempty values and a fraction between zero and one"
        raise ValueError(msg)
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    lower = math.floor(position)
    upper = math.ceil(position)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def seeded_random(seed: int, key: str) -> random.Random:
    """Give each result a stable stream independent of dictionary iteration order."""
    digest = hashlib.sha256(f"{seed}:{key}".encode()).digest()
    return random.Random(int.from_bytes(digest[:16], "big"))  # noqa: S311 - statistical resampling only.


def uncertainty_label(count: int) -> str:
    """Flag unavailable and small-sample intervals rather than imply precision."""
    if count < 2:  # noqa: PLR2004 - two independent units are the minimum for resampling.
        return "unavailable: fewer than two independent units"
    if count < MIN_DESCRIPTIVE_UNITS:
        return "exploratory: fewer than ten independent units"
    return "bootstrap: conditional on this host, workload and measurement period"


def bootstrap_interval(
    values: Sequence[float], *, resamples: int = 10_000, seed: int = 20261016, key: str = "median"
) -> dict[str, float] | None:
    """Bootstrap the median using independent units, never individual pooled calls."""
    if resamples < 1:
        msg = "Bootstrap resamples must be positive"
        raise ValueError(msg)
    if len(values) < 2:  # noqa: PLR2004 - a single unit cannot estimate between-unit variability.
        return None
    rng = seeded_random(seed, key)
    estimates = [statistics.median(rng.choices(values, k=len(values))) for _ in range(resamples)]
    return {"lower": percentile(estimates, 0.025), "upper": percentile(estimates, 0.975)}


def summarize_units(
    units: Sequence[Sequence[float]], *, unit: str, resamples: int = 10_000, seed: int = 20261016, key: str = ""
) -> dict[str, Any]:
    """Estimate the median of unit medians and keep pooled statistics descriptive."""
    nonempty = [list(values) for values in units if values]
    medians = [statistics.median(values) for values in nonempty]
    pooled = sorted(itertools.chain.from_iterable(nonempty))
    return {
        "independent_unit": unit,
        "estimand": f"median of per-{unit} medians",
        "n_units": len(medians),
        "n_samples": len(pooled),
        "estimate": statistics.median(medians) if medians else None,
        "ci95": bootstrap_interval(medians, resamples=resamples, seed=seed, key=key),
        "uncertainty": uncertainty_label(len(medians)),
        "pooled_descriptive": {
            "median": statistics.median(pooled) if pooled else None,
            "p95_nearest_rank": pooled[math.ceil(0.95 * len(pooled)) - 1] if pooled else None,
            "minimum": pooled[0] if pooled else None,
            "maximum": pooled[-1] if pooled else None,
        },
    }


def record_counts(records: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Retain all record statuses, including unfinished measurements."""
    statuses = Counter(str(record.get("status", "missing")) for record in records)
    samples = [sample for record in records for sample in record.get("samples", [])]
    return {
        "total": len(records),
        "passed": statuses.get("passed", 0),
        "failed": statuses.get("failed", 0),
        "other": sum(count for status, count in statuses.items() if status not in {"passed", "failed"}),
        "by_status": dict(sorted(statuses.items())),
        "samples_total": len(samples),
        "samples_validated": sum(sample.get("valid") is True for sample in samples),
        "samples_invalid_or_unvalidated": sum(sample.get("valid") is not True for sample in samples),
    }


def sample_values(samples: Sequence[dict[str, Any]], field: str = "seconds") -> dict[str, list[float]]:
    """Group validated finite values by workload; malformed samples are excluded."""
    grouped: dict[str, list[float]] = defaultdict(list)
    for sample in samples:
        metric = sample.get("metric")
        value = sample.get(field)
        if sample.get("valid") is True and isinstance(metric, str) and finite_nonnegative(value):
            grouped[metric].append(float(value))
    return dict(grouped)


def summarize_samples(
    sample_groups: Sequence[Sequence[dict[str, Any]]],
    *,
    unit: str,
    field: str = "seconds",
    resamples: int,
    seed: int,
    key: str,
) -> dict[str, Any]:
    """Aggregate each independent group before constructing intervals."""
    by_metric: dict[str, list[list[float]]] = defaultdict(list)
    for samples in sample_groups:
        for metric, values in sample_values(samples, field).items():
            by_metric[metric].append(values)
    return {
        metric: summarize_units(units, unit=unit, resamples=resamples, seed=seed, key=f"{key}:{field}:{metric}")
        for metric, units in sorted(by_metric.items())
    }


def resource_summary(
    groups: Sequence[Sequence[dict[str, Any]]], *, unit: str, resamples: int, seed: int, key: str
) -> dict[str, Any]:
    """Label process-level workload telemetry separately from backend resources."""
    result = {
        field: summarize_samples(groups, unit=unit, field=field, resamples=resamples, seed=seed, key=key)
        for field in RESOURCE_FIELDS
    }
    return {field: metrics for field, metrics in result.items() if metrics}


def batch_passed(batch: dict[str, Any]) -> bool:
    """Require the entire batch and every worker to pass before reporting latency."""
    workers = batch.get("workers_results", [])
    expected_calls = batch.get("expected_calls", 0)
    return (
        batch.get("status") == "passed"
        and bool(workers)
        and len(workers) == batch.get("workers")
        and finite_nonnegative(expected_calls)
        and expected_calls > 0
        and batch.get("validated_calls") == expected_calls
        and all(worker.get("status") == "passed" for worker in workers)
    )


def summarize_batches(
    batches: Sequence[dict[str, Any]], *, resamples: int, seed: int, key: str
) -> list[dict[str, Any]]:
    """Use independent batch medians for latency and batch values for throughput."""
    grouped: dict[tuple[int, str], list[dict[str, Any]]] = defaultdict(list)
    for batch in batches:
        grouped[(int(batch["workers"]), str(batch["workload"]))].append(batch)
    output = []
    for (workers, workload), records in sorted(grouped.items()):
        accepted = [batch for batch in records if batch_passed(batch)]
        all_workers = [worker for batch in records for worker in batch.get("workers_results", [])]
        samples = [
            [sample for worker in batch.get("workers_results", []) for sample in worker.get("samples", [])]
            for batch in accepted
        ]
        group_key = f"{key}:{workers}:{workload}"
        throughput = [
            [float(batch["throughput_validated_calls_per_second"])]
            for batch in accepted
            if finite_nonnegative(batch.get("throughput_validated_calls_per_second"))
        ]
        timing = {
            field: summarize_units(
                [[float(batch[field])] for batch in accepted if finite_nonnegative(batch.get(field))],
                unit="batch",
                resamples=resamples,
                seed=seed,
                key=f"{group_key}:{field}",
            )
            for field in ("execution_seconds", "initialization_seconds", "teardown_seconds")
        }
        output.append({
            "workers": workers,
            "workload": workload,
            "batch_counts": record_counts(records),
            "accepted_batches": len(accepted),
            "worker_counts": record_counts(all_workers),
            "validated_calls_all_batches": sum(batch.get("validated_calls", 0) for batch in records),
            "expected_calls_all_batches": sum(batch.get("expected_calls", 0) for batch in records),
            "throughput_validated_calls_per_second": summarize_units(
                throughput, unit="batch", resamples=resamples, seed=seed, key=f"{group_key}:throughput"
            ),
            "latency_seconds": summarize_samples(samples, unit="batch", resamples=resamples, seed=seed, key=group_key),
            "batch_timing_seconds": timing,
            "payload_resources": resource_summary(samples, unit="batch", resamples=resamples, seed=seed, key=group_key),
        })
    return output


def trial_medians_by_round(trials: Sequence[dict[str, Any]]) -> dict[str, dict[int, float]]:
    """Index complete session medians by explicit randomized round, rejecting duplicates."""
    result: dict[str, dict[int, float]] = defaultdict(dict)
    for trial in trials:
        if trial.get("status") != "passed" or not isinstance(trial.get("round"), int):
            continue
        round_id = trial["round"]
        for metric, values in sample_values(trial.get("samples", [])).items():
            if round_id in result[metric]:
                msg = f"Duplicate passed trial for round {round_id}, metric {metric!r}"
                raise ValueError(msg)
            result[metric][round_id] = statistics.median(values)
    return dict(result)


def paired_comparison(
    numerator: dict[int, float], denominator: dict[int, float], *, resamples: int, seed: int, key: str
) -> dict[str, Any]:
    """Bootstrap the ratio of median latencies, resampling matched rounds together."""
    common = sorted(set(numerator) & set(denominator))
    rounds = [index for index in common if numerator[index] > 0 and denominator[index] > 0]
    pairs = [(numerator[index], denominator[index]) for index in rounds]
    estimate = (
        statistics.median([pair[0] for pair in pairs]) / statistics.median([pair[1] for pair in pairs])
        if pairs
        else None
    )
    interval = None
    if len(pairs) >= 2:  # noqa: PLR2004 - two paired rounds are the minimum for resampling.
        rng = seeded_random(seed, key)
        estimates = []
        for _ in range(resamples):
            selected = rng.choices(pairs, k=len(pairs))
            estimates.append(
                statistics.median([pair[0] for pair in selected]) / statistics.median([pair[1] for pair in selected])
            )
        interval = {"lower": percentile(estimates, 0.025), "upper": percentile(estimates, 0.975)}
    return {
        "estimand": "ratio of medians of session medians over matched randomized rounds",
        "independent_unit": "matched round",
        "n_pairs": len(pairs),
        "rounds": rounds,
        "unmatched_rounds": len(set(numerator) ^ set(denominator)),
        "nonpositive_pairs_excluded": len(common) - len(rounds),
        "estimate": estimate,
        "ci95": interval,
        "ci95_includes_one": interval["lower"] <= 1 <= interval["upper"] if interval else None,
        "uncertainty": uncertainty_label(len(pairs)),
        "interpretation": "ratio below one means numerator backend has lower latency",
    }


def summarize_document(document: dict[str, Any], *, resamples: int = 10_000, seed: int = 20261016) -> dict[str, Any]:
    """Summarize one experimental profile without mixing files, profiles or workloads."""
    if resamples < 1:
        msg = "Bootstrap resamples must be positive"
        raise ValueError(msg)
    backends = {}
    round_medians = {}
    for name, record in sorted(document["backends"].items()):
        trials = record.get("trials", [])
        passed = [trial for trial in trials if trial.get("status") == "passed"]
        samples = [trial.get("samples", []) for trial in passed]
        backends[name] = {
            "status": record.get("status"),
            "reason": record.get("reason"),
            "preflight_status": record.get("preflight", {}).get("status"),
            "session_counts": record_counts(trials),
            "latency_seconds": summarize_samples(samples, unit="session", resamples=resamples, seed=seed, key=name),
            "payload_resources": resource_summary(samples, unit="session", resamples=resamples, seed=seed, key=name),
            "concurrency": summarize_batches(record.get("batches", []), resamples=resamples, seed=seed, key=name),
        }
        round_medians[name] = trial_medians_by_round(trials)
    comparisons = []
    for numerator, denominator in itertools.combinations(sorted(round_medians), 2):
        common_metrics = sorted(set(round_medians[numerator]) & set(round_medians[denominator]))
        for metric in common_metrics:
            comparison = paired_comparison(
                round_medians[numerator][metric],
                round_medians[denominator][metric],
                resamples=resamples,
                seed=seed,
                key=f"pair:{numerator}:{denominator}:{metric}",
            )
            comparisons.append({"numerator": numerator, "denominator": denominator, "metric": metric, **comparison})
    return {
        "profile": document.get("profile"),
        "metadata": document.get("metadata"),
        "methodology": {
            "confidence_level": CONFIDENCE_LEVEL,
            "bootstrap_resamples": resamples,
            "bootstrap_seed": seed,
            "interval_method": "percentile bootstrap with linearly interpolated percentiles",
            "latency_estimand": "median of per-session medians; concurrency uses per-batch medians",
            "selection": "complete passed sessions/batches only; all status and failure counts retained",
            "pooled_statistics": "descriptive only; p95 uses nearest rank and has no confidence interval",
            "small_sample_policy": "one unit: no interval; fewer than ten units: exploratory",
            "pairing": "explicit matching randomized round IDs within one profile/input file",
            "multiplicity": "intervals are pointwise, not adjusted for multiple comparisons",
            "resource_scope": "payload process CPU, wall time and peak RSS; not whole-sandbox or host resource usage",
            "scope": "conditional on this host, configuration and measurement period; not cross-host replication",
        },
        "backends": backends,
        "paired_comparisons": comparisons,
    }


def csv_rows(experiment: dict[str, Any]) -> list[dict[str, Any]]:
    """Flatten scalar summaries with explicit units and bootstrap grouping."""
    rows = []
    for name, backend in experiment["backends"].items():
        groups = [("session", None, None, "seconds", backend["latency_seconds"])]
        groups.extend(
            ("payload", None, None, field, metrics) for field, metrics in backend["payload_resources"].items()
        )
        for concurrency in backend["concurrency"]:
            workers, workload = concurrency["workers"], concurrency["workload"]
            groups.extend([
                ("concurrency", workers, workload, "seconds", concurrency["latency_seconds"]),
                ("batch_timing", workers, workload, "seconds", concurrency["batch_timing_seconds"]),
                (
                    "throughput",
                    workers,
                    workload,
                    "calls/second",
                    {"throughput": concurrency["throughput_validated_calls_per_second"]},
                ),
            ])
            groups.extend(
                ("concurrent_payload", workers, workload, field, metrics)
                for field, metrics in concurrency["payload_resources"].items()
            )
        for kind, workers, workload, units, metrics in groups:
            for metric, summary in metrics.items():
                interval = summary["ci95"] or {}
                rows.append({
                    "source": experiment.get("source"),
                    "profile": experiment.get("profile"),
                    "backend": name,
                    "kind": kind,
                    "workers": workers,
                    "workload": workload,
                    "metric": metric,
                    "units": units,
                    "independent_unit": summary["independent_unit"],
                    "n_units": summary["n_units"],
                    "n_samples": summary["n_samples"],
                    "estimate": summary["estimate"],
                    "ci95_lower": interval.get("lower"),
                    "ci95_upper": interval.get("upper"),
                    "pooled_median": summary["pooled_descriptive"]["median"],
                    "pooled_p95_nearest_rank": summary["pooled_descriptive"]["p95_nearest_rank"],
                    "uncertainty": summary["uncertainty"],
                })
    return rows


def main(argv: Sequence[str] | None = None) -> int:
    """Write JSON and optional CSV summaries after all timed experiments finish."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", type=Path, nargs="+")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--csv", type=Path)
    parser.add_argument("--resamples", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=20261016)
    args = parser.parse_args(argv)
    if args.resamples < 1:
        parser.error("--resamples must be positive")
    experiments = []
    for path in args.inputs:
        summary = summarize_document(json.loads(path.read_text()), resamples=args.resamples, seed=args.seed)
        summary["source"] = str(path)
        experiments.append(summary)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps({"schema_version": 1, "experiments": experiments}, indent=2, allow_nan=False) + "\n"
    )
    if args.csv:
        rows = [row for experiment in experiments for row in csv_rows(experiment)]
        args.csv.parent.mkdir(parents=True, exist_ok=True)
        with args.csv.open("w", newline="") as handle:
            if rows:
                writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
