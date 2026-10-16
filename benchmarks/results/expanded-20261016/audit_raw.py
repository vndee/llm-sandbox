"""Independently audit the completed expanded experiment; never run backends.

Run only after all measurement has finished. This script imports no benchmark
modules and changes no source or measurement files. Its output is separate audit
evidence. Native cleanup is evidenced by successful close calls; only Docker has
an independently recorded runtime-removal confirmation in this result schema.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import itertools
import json
import math
import random
import statistics
from collections import Counter, defaultdict
from pathlib import Path


EXPECTED_BACKENDS = {"docker", "bubblewrap", "nsjail"}
EXPECTED_REVISION = "d79a0e2038ba90f267a3f5109beea9810dde292a"
PROFILES = {
    "baseline": {"rounds": 30, "iterations": 10, "seed": 20261016, "sessions": 90},
    "setup": {"rounds": 10, "iterations": 1, "seed": 20261017, "sessions": 30},
    "workload": {"rounds": 10, "iterations": 10, "seed": 20261018, "sessions": 30},
    "concurrency": {"iterations": 10, "seed": 20261019, "sessions": 630},
}
SOURCE_KEYS = {
    "backend_benchmark.py": "harness_sha256",
    "experiment_suite.py": "suite_sha256",
    "experiment_workloads.py": "workloads_sha256",
}
RESOURCE_FIELDS = ("payload_cpu_seconds", "payload_wall_seconds", "payload_peak_rss_kib")
RESOURCE_SCOPE = "payload Python process only; peak RSS includes interpreter; excludes backend/daemon/host"


def numeric(value, *, positive=False):
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(value)
        and (value > 0 if positive else value >= 0)
    )


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


class Audit:
    def __init__(self):
        self.checks = 0
        self.errors = []
        self.docker_confirmations = 0
        self.native_successful_closes = 0

    def check(self, condition, message):
        self.checks += 1
        if not condition:
            self.errors.append(message)

    def close(self, trial, backend, context):
        self.check(trial.get("cleanup_verified") is not False, f"{context}: unverified cleanup failure")
        self.check(not any("error" in key for key in trial), f"{context}: retained error field")
        closes = [s for s in trial.get("samples", []) if s.get("metric") == "close"]
        self.check(len(closes) == 1 and closes[0].get("valid") is True, f"{context}: close not successful")
        if backend == "docker":
            verified = trial.get("cleanup_verified") is True
            self.check(verified, f"{context}: Docker removal not positively confirmed")
            self.docker_confirmations += int(verified)
        else:
            self.native_successful_closes += int(len(closes) == 1 and closes[0].get("valid") is True)

    def samples(self, trial, expected_metrics, context):
        samples = trial.get("samples", [])
        self.check(Counter(s.get("metric") for s in samples) == expected_metrics, f"{context}: metric counts")
        self.check(trial.get("measured") is True, f"{context}: not marked measured")
        self.check(trial.get("status") == "passed", f"{context}: trial not passed")
        resources = []
        for index, sample in enumerate(samples):
            location = f"{context}: sample {index}"
            self.check(sample.get("valid") is True, f"{location}: not validated")
            self.check(numeric(sample.get("seconds"), positive=True), f"{location}: invalid API duration")
            self.check("error" not in sample, f"{location}: retained sample error")
            if sample.get("metric") in {"json_run", "marker_resource_run"}:
                for field in RESOURCE_FIELDS:
                    self.check(
                        numeric(sample.get(field), positive=field == "payload_peak_rss_kib"),
                        f"{location}: invalid {field}",
                    )
                resources.append({"metric": sample["metric"], **{k: sample.get(k) for k in RESOURCE_FIELDS}})
        self.check(trial.get("resources", []) == resources, f"{context}: duplicate resource records disagree")
        for field in ("workload_warmup", "transfer_warmup"):
            if field in trial:
                warmup = trial[field]
                self.check(warmup.get("measured") is False, f"{context}: warmup not explicitly excluded")
                self.check(all(s.get("valid") is True for s in warmup.get("samples", [])), f"{context}: warmup failure")
        return len(samples)


def expected_order(document):
    metadata = document["metadata"]
    names = document["preflight_order"]
    rng = random.Random(metadata["seed"])
    schedule = []
    if document["profile"] == "concurrency":
        for index in range(metadata["batches"]):
            jobs = [
                {"index": index, "backend": name, "workers": workers, "workload": workload}
                for name, workers, workload in itertools.product(names, metadata["levels"], metadata["workloads"])
            ]
            rng.shuffle(jobs)
            schedule.extend(jobs)
    else:
        for round_index in range(metadata["settings"]["trials"]):
            names_in_round = names.copy()
            rng.shuffle(names_in_round)
            schedule.extend({"round": round_index, "backend": name} for name in names_in_round)
    return schedule


def audit_summary(audit, record, context):
    values = defaultdict(list)
    for trial in record["trials"]:
        for sample in trial["samples"]:
            if sample.get("valid") is True:
                values[sample["metric"]].append(sample["seconds"])
    expected = {}
    for metric, samples in values.items():
        ordered = sorted(samples)
        expected[metric] = {
            "n": len(samples),
            "median_seconds": statistics.median(samples),
            "p95_seconds": ordered[math.ceil(0.95 * len(samples)) - 1],
            "min_seconds": ordered[0],
            "max_seconds": ordered[-1],
        }
    audit.check(record.get("summary") == expected, f"{context}: saved summary differs from raw measured samples")


def audit_batch(audit, batch, backend, iterations, context):
    workers = batch["workers_results"]
    count = batch["workers"]
    metric = "json_run" if batch["workload"] == "json" else "marker_resource_run"
    expected_metrics = Counter({"construct": 1, "open": 1, "close": 1, metric: iterations})
    audit.check(
        batch.get("status") == "passed" and batch.get("measured") is True, f"{context}: not measured passed batch"
    )
    audit.check(len(workers) == count, f"{context}: worker count mismatch")
    audit.check({worker["worker"] for worker in workers} == set(range(count)), f"{context}: worker IDs")
    audit.check(batch.get("expected_calls") == count * iterations, f"{context}: expected calls mismatch")
    validated_calls = sum(s.get("valid") is True and s.get("metric") == metric for w in workers for s in w["samples"])
    audit.check(
        validated_calls == batch.get("validated_calls") == count * iterations, f"{context}: validated calls mismatch"
    )
    audit.check(not any("error" in key for key in batch), f"{context}: retained batch error")
    audit.check(batch.get("cleanup_verified") is not False, f"{context}: batch cleanup failure")
    for field in ("total_seconds", "initialization_seconds", "execution_seconds", "teardown_seconds"):
        audit.check(numeric(batch.get(field), positive=True), f"{context}: nonpositive {field}")
    window = batch["execution_seconds"]
    execution_end = max(w["execution_finished_at"] for w in workers)
    execution_start = execution_end - window
    total_start = execution_start - batch["initialization_seconds"]
    total_end = execution_end + batch["teardown_seconds"]
    audit.check(
        math.isclose(
            batch["total_seconds"],
            sum(batch[field] for field in ("initialization_seconds", "execution_seconds", "teardown_seconds")),
            rel_tol=1e-12,
            abs_tol=1e-8,
        ),
        f"{context}: phases do not sum to total time",
    )
    audit.check(
        numeric(window, positive=True)
        and math.isclose(batch["throughput_validated_calls_per_second"], validated_calls / window, rel_tol=1e-12),
        f"{context}: throughput does not equal validated calls / execution window",
    )
    sample_count = 0
    for worker in workers:
        location = f"{context}: worker {worker['worker']}"
        sample_count += audit.samples(worker, expected_metrics, location)
        audit.close(worker, backend, location)
        audit.check(
            total_start <= worker["ready_at"] <= execution_start + 1e-8,
            f"{location}: ready after shared execution starts",
        )
        audit.check(
            execution_start <= worker["execution_finished_at"] <= execution_end,
            f"{location}: execution outside shared window",
        )
        audit.check(
            execution_end <= worker["finished_at"] <= total_end + 1e-8,
            f"{location}: lifecycle finished outside teardown window",
        )
        api_time = sum(s["seconds"] for s in worker["samples"] if s["metric"] == metric)
        audit.check(api_time <= window + 1e-8, f"{location}: summed API durations exceed execution window")
    return {
        "worker_sessions": count,
        "validated_calls": validated_calls,
        "samples": sample_count,
        "start": total_start,
        "end": total_end,
    }


def check_frozen_barriers(audit, source):
    """Verify the frozen control-flow evidence unavailable as individual raw timestamps."""
    module = ast.parse(source.read_text())
    batch = next(
        node for node in module.body if isinstance(node, ast.FunctionDef) and node.name == "run_concurrent_batch"
    )
    worker = next(node for node in batch.body if isinstance(node, ast.FunctionDef) and node.name == "worker")
    lifecycle = next(node for node in worker.body if isinstance(node, ast.Try))
    ready_waits = [
        node for node in ast.walk(lifecycle) if isinstance(node, ast.Call) and ast.unparse(node.func) == "gate.wait"
    ]
    finish_waits = [
        node
        for node in ast.walk(lifecycle)
        if isinstance(node, ast.Call) and ast.unparse(node.func) == "finish_gate.wait"
    ]
    closes = [
        node
        for node in ast.walk(lifecycle)
        if isinstance(node, ast.Call) and ast.unparse(node.func) == "base.finish_session"
    ]
    ready_assignments = [
        node
        for node in ast.walk(lifecycle)
        if isinstance(node, ast.Assign) and any(ast.unparse(target) == "trial['ready_at']" for target in node.targets)
    ]
    finished_assignments = [
        node
        for node in ast.walk(lifecycle)
        if isinstance(node, ast.Assign)
        and any(ast.unparse(target) == "trial['execution_finished_at']" for target in node.targets)
    ]
    audit.check(
        len(ready_waits) == len(finish_waits) == len(closes) == 1,
        "frozen suite: unique readiness/finish barriers and cleanup",
    )
    audit.check(
        bool(ready_assignments) and bool(finished_assignments), "frozen suite: readiness/execution timestamps present"
    )
    audit.check(
        ready_assignments[0].lineno
        < ready_waits[0].lineno
        < finished_assignments[0].lineno
        < finish_waits[0].lineno
        < closes[0].lineno,
        "frozen suite: barriers/timestamps/cleanup out of order",
    )
    final_calls = {
        ast.unparse(node.func) for body in lifecycle.finalbody for node in ast.walk(body) if isinstance(node, ast.Call)
    }
    audit.check(
        {"finish_gate.wait", "base.finish_session"} <= final_calls,
        "frozen suite: finish barrier and cleanup not both in finally",
    )


def run_audit(results, repository):
    audit = Audit()
    result = {
        "results_directory": str(results),
        "profiles": {},
        "source_sha256": {},
        "limitations": [
            "Raw timestamps do not record individual teardown-start instants; exclusion is supported by frozen completion-barrier control flow plus shared-window arithmetic.",
            "Docker deletion has explicit cleanup_verified records. Native cleanup has successful close samples, with no separate directory-deletion assertion in this schema.",
            "This audits retained validation evidence; payload outputs/files were validated at measurement time and are not re-executed here.",
        ],
    }
    frozen_hashes = {}
    for filename in (*SOURCE_KEYS, "experiment_statistics.py"):
        frozen_hashes[filename] = digest(results / "source" / filename)
        audit.check(
            frozen_hashes[filename] == digest(repository / "benchmarks" / filename),
            f"source {filename}: live file differs from frozen source",
        )
    result["source_sha256"] = frozen_hashes
    check_frozen_barriers(audit, results / "source" / "experiment_suite.py")
    total_sessions = total_samples = 0
    for profile, expected in PROFILES.items():
        path = results / f"{profile}.json"
        document = json.loads(path.read_text())
        metadata = document["metadata"]
        settings = metadata["settings"]
        audit.check(
            document.get("profile") == profile and document.get("schema_version") == 2, f"{profile}: identity/schema"
        )
        audit.check(metadata.get("finished_at_utc") is not None, f"{profile}: not finished")
        audit.check(
            not document.get("interrupted") and not document.get("stopped_early"), f"{profile}: interrupted/stopped"
        )
        audit.check(set(document["backends"]) == EXPECTED_BACKENDS, f"{profile}: backend roster mismatch")
        audit.check(
            metadata.get("revision") == EXPECTED_REVISION and metadata.get("branch") == "main",
            f"{profile}: source revision",
        )
        audit.check(
            metadata.get("seed") == expected["seed"] and settings["iterations"] == expected["iterations"],
            f"{profile}: expected seed/iterations",
        )
        audit.check("micromamba" in metadata.get("excluded_backends", []), f"{profile}: Micromamba exclusion")
        audit.check(settings["warmups"] == 2, f"{profile}: warmup count")
        for filename, key in SOURCE_KEYS.items():
            audit.check(metadata.get(key) == frozen_hashes[filename], f"{profile}: {key} differs from frozen source")
        order = expected_order(document)
        audit.check(
            document["planned_order"] == document["executed_order"] == order,
            f"{profile}: randomized schedule mismatch/incomplete execution",
        )
        sessions = samples = batches = calls = 0
        batch_windows = {}
        per_backend = {}
        for backend, record in document["backends"].items():
            context = f"{profile}/{backend}"
            audit.check(
                record.get("status") == "passed" and record.get("failed_trials") == 0, f"{context}: status/failures"
            )
            preflight = record["preflight"]
            audit.check(
                preflight.get("status") == "passed" and preflight.get("measured") is False,
                f"{context}: preflight failed or counted",
            )
            audit.close(preflight, backend, f"{context}/preflight")
            audit.check(
                record["effective_config"]["skip_environment_setup"] is (profile != "setup"),
                f"{context}: setup profile configuration",
            )
            if profile == "concurrency":
                audit.check(
                    metadata["batches"] == 5
                    and metadata["levels"] == [1, 4, 16]
                    and metadata["workloads"] == ["marker", "json"],
                    f"{context}: concurrency design",
                )
                records = record["batches"]
                audit.check(len(records) == record["successful_trials"] == 30, f"{context}: expected 30 batches")
                audit.check(record["trials"] == [], f"{context}: unexpected measured sequential trials")
                coverage = Counter((batch["index"], batch["workers"], batch["workload"]) for batch in records)
                audit.check(
                    coverage == Counter(itertools.product(range(5), [1, 4, 16], ["marker", "json"])),
                    f"{context}: batch factorial coverage",
                )
                backend_sessions = backend_calls = 0
                for batch in records:
                    location = f"{context}/{batch['index']}/{batch['workers']}/{batch['workload']}"
                    counts = audit_batch(audit, batch, backend, settings["iterations"], location)
                    sessions += counts["worker_sessions"]
                    backend_sessions += counts["worker_sessions"]
                    calls += counts["validated_calls"]
                    backend_calls += counts["validated_calls"]
                    samples += counts["samples"]
                    batches += 1
                    batch_windows[(backend, batch["index"], batch["workers"], batch["workload"])] = counts
                per_backend[backend] = {
                    "batches": len(records),
                    "worker_sessions": backend_sessions,
                    "validated_calls": backend_calls,
                }
            else:
                audit.check(settings["trials"] == expected["rounds"], f"{context}: rounds setting")
                records = record["trials"]
                audit.check(
                    len(records) == record["successful_trials"] == expected["rounds"], f"{context}: session count"
                )
                audit.check(
                    [trial["round"] for trial in records] == list(range(expected["rounds"])),
                    f"{context}: paired randomized round IDs",
                )
                audit.check(record["batches"] == [], f"{context}: unexpected concurrent batches")
                metrics = Counter({"construct": 1, "open": 1, "first_run": 1, "close": 1})
                if profile == "workload":
                    metrics.update({"json_run": 10, "copy_to_16mib": 3, "copy_from_16mib": 3})
                    audit.check(
                        metadata["workload_definitions"]["json_records"] == 10000
                        and metadata["workload_definitions"]["large_transfer_bytes"] == 16777216,
                        f"{context}: workload definition",
                    )
                else:
                    metrics.update({
                        name: settings["iterations"]
                        for name in ("warm_run", "cpu_run", "copy_to_1mib", "copy_from_1mib")
                    })
                for trial in records:
                    location = f"{context}/round-{trial['round']}"
                    samples += audit.samples(trial, metrics, location)
                    audit.close(trial, backend, location)
                    sessions += 1
                audit_summary(audit, record, context)
                per_backend[backend] = {"sessions": len(records)}
        if profile == "concurrency":
            audit.check(
                (batches, sessions, calls) == (90, 630, 6300), "concurrency: aggregate expected batches/workers/calls"
            )
            previous_end = None
            for job in order:
                window = batch_windows[(job["backend"], job["index"], job["workers"], job["workload"])]
                if previous_end is not None:
                    audit.check(
                        window["start"] >= previous_end - 1e-8, "concurrency: consecutive backend batches overlap"
                    )
                previous_end = window["end"]
        audit.check(sessions == expected["sessions"], f"{profile}: total measured sessions")
        result["profiles"][profile] = {
            "raw_sha256": digest(path),
            "sessions": sessions,
            "validated_samples_including_lifecycle": samples,
            "batches": batches,
            "validated_concurrent_calls": calls,
            "backends": per_backend,
        }
        total_sessions += sessions
        total_samples += samples
    audit.check(total_sessions == 780, "all profiles: expected 780 measured sessions")
    audit.check(total_samples == 12990, "all profiles: expected 12990 validated measured API samples")
    result.update(
        status="passed" if not audit.errors else "failed",
        checks=audit.checks,
        errors=audit.errors,
        total_measured_sessions=total_sessions,
        total_validated_samples_including_lifecycle=total_samples,
        cleanup={
            "docker_positive_removal_confirmations_including_preflights": audit.docker_confirmations,
            "native_successful_closes_including_preflights": audit.native_successful_closes,
        },
    )
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    artifact_directory = Path(__file__).resolve().parent
    parser.add_argument("--repository", type=Path, default=artifact_directory.parents[2])
    parser.add_argument("--results", type=Path, default=artifact_directory)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    if args.output is None:
        args.output = args.results / "raw-audit.json"
    result = run_audit(args.results, args.repository)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps(result, indent=2, allow_nan=False))
    return int(result["status"] != "passed")


if __name__ == "__main__":
    raise SystemExit(main())
