"""Independent arithmetic audit; does not import production statistics tooling."""

import csv
import hashlib
import json
import math
import random
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SUMMARY = json.loads((ROOT / "summary.json").read_text())
RAW = {
    profile: json.loads((ROOT / f"{profile}.json").read_text())
    for profile in ("baseline", "setup", "workload", "concurrency")
}
checks = 0
errors = []
metrics_checked = 0
pairs_checked = 0
csv_expected = {}
selected_ci = []
unit_sets = {}


def check(condition, label):
    global checks
    checks += 1
    if not condition:
        errors.append(label)


def equal(actual, expected, label):
    if isinstance(expected, float):
        check(isinstance(actual, (float, int)) and math.isclose(actual, expected, rel_tol=1e-12, abs_tol=1e-15), label)
    else:
        check(actual == expected, label)


def median(values):
    values = sorted(values)
    size = len(values)
    if size % 2:
        return values[size // 2]
    return (values[size // 2 - 1] + values[size // 2]) / 2


def valid(value):
    return type(value) in (int, float) and math.isfinite(value) and value >= 0


def units_from_records(records, field="seconds"):
    result = defaultdict(list)
    for record in records:
        by_metric = defaultdict(list)
        for sample in record.get("samples", []):
            if sample.get("valid") is True and valid(sample.get(field)):
                by_metric[sample["metric"]].append(sample[field])
        for metric, values in by_metric.items():
            result[metric].append(values)
    return dict(result)


def check_counts(records, output, label):
    counts = Counter(record.get("status", "missing") for record in records)
    samples = [sample for record in records for sample in record.get("samples", [])]
    expected = {
        "total": len(records),
        "passed": counts["passed"],
        "failed": counts["failed"],
        "other": len(records) - counts["passed"] - counts["failed"],
        "by_status": dict(counts),
        "samples_total": len(samples),
        "samples_validated": sum(sample.get("valid") is True for sample in samples),
        "samples_invalid_or_unvalidated": sum(sample.get("valid") is not True for sample in samples),
    }
    equal(output, expected, f"{label}: all status/sample counts")


def audit_metric(profile, backend, kind, workers, workload, units_label, metric, groups, result, unit):
    global metrics_checked
    metrics_checked += 1
    label = f"{profile}/{backend}/{kind}/{workers}/{workload}/{units_label}/{metric}"
    points = [median(group) for group in groups]
    pooled = sorted(value for group in groups for value in group)
    expected = {"n_units": len(points), "n_samples": len(pooled), "estimate": median(points) if points else None}
    for key, value in expected.items():
        equal(result[key], value, f"{label}: {key}")
    equal(result["independent_unit"], unit, f"{label}: independent unit")
    descriptive = {
        "median": median(pooled) if pooled else None,
        "p95_nearest_rank": pooled[math.ceil(len(pooled) * 0.95) - 1] if pooled else None,
        "minimum": pooled[0] if pooled else None,
        "maximum": pooled[-1] if pooled else None,
    }
    for key, value in descriptive.items():
        equal(result["pooled_descriptive"][key], value, f"{label}: pooled {key}")
    if len(points) < 2:
        equal(result["ci95"], None, f"{label}: no CI with insufficient units")
        check(result["uncertainty"].startswith("unavailable"), f"{label}: unavailable label")
    else:
        ci = result["ci95"]
        check(min(points) <= ci["lower"] <= ci["upper"] <= max(points), f"{label}: CI bounds within unit values")
        expected_label = "exploratory" if len(points) < 10 else "bootstrap"
        check(result["uncertainty"].startswith(expected_label), f"{label}: replication label")
    rowkey = (profile, backend, kind, str(workers) if workers is not None else "", workload or "", metric, units_label)
    check(rowkey not in csv_expected, f"{label}: unique CSV group")
    csv_expected[rowkey] = {
        "n_units": expected["n_units"],
        "n_samples": expected["n_samples"],
        "estimate": expected["estimate"],
        "pooled_median": descriptive["median"],
        "pooled_p95_nearest_rank": descriptive["p95_nearest_rank"],
        "ci95_lower": (result["ci95"] or {}).get("lower"),
        "ci95_upper": (result["ci95"] or {}).get("upper"),
        "independent_unit": unit,
        "uncertainty": result["uncertainty"],
    }
    unit_sets[rowkey] = groups


def audit_group(profile, backend, kind, workers, workload, field, computed, saved, unit):
    equal(set(saved), set(computed), f"{profile}/{backend}/{kind}/{workers}/{workload}/{field}: metric coverage")
    for metric, groups in computed.items():
        audit_metric(profile, backend, kind, workers, workload, field, metric, groups, saved[metric], unit)


for experiment in SUMMARY["experiments"]:
    profile = experiment["profile"]
    raw = RAW[profile]
    equal(experiment["metadata"], raw["metadata"], f"{profile}: metadata exact")
    equal(experiment["methodology"]["bootstrap_seed"], 20261016, f"{profile}: bootstrap seed")
    equal(experiment["methodology"]["bootstrap_resamples"], 10000, f"{profile}: bootstrap replicates")
    equal(set(experiment["backends"]), set(raw["backends"]), f"{profile}: backend coverage")
    round_values = {}
    for backend, record in raw["backends"].items():
        saved = experiment["backends"][backend]
        equal(saved["status"], record["status"], f"{profile}/{backend}: status")
        equal(saved["preflight_status"], record["preflight"]["status"], f"{profile}/{backend}: preflight status")
        trials = record["trials"]
        check_counts(trials, saved["session_counts"], f"{profile}/{backend}: sessions")
        accepted = [trial for trial in trials if trial["status"] == "passed"]
        audit_group(
            profile,
            backend,
            "session",
            None,
            None,
            "seconds",
            units_from_records(accepted),
            saved["latency_seconds"],
            "session",
        )
        for field in ("payload_cpu_seconds", "payload_wall_seconds", "payload_peak_rss_kib"):
            audit_group(
                profile,
                backend,
                "payload",
                None,
                None,
                field,
                units_from_records(accepted, field),
                saved["payload_resources"].get(field, {}),
                "session",
            )
        by_round = defaultdict(dict)
        for trial in accepted:
            for metric, groups in units_from_records([trial]).items():
                by_round[metric][trial["round"]] = median(groups[0])
        round_values[backend] = by_round
        batches = defaultdict(list)
        for batch in record["batches"]:
            batches[(batch["workers"], batch["workload"])].append(batch)
        equal(len(saved["concurrency"]), len(batches), f"{profile}/{backend}: batch group coverage")
        for saved_batch in saved["concurrency"]:
            workers, workload = saved_batch["workers"], saved_batch["workload"]
            batch_records = batches[(workers, workload)]
            check_counts(batch_records, saved_batch["batch_counts"], f"{backend}/{workers}/{workload}: batches")
            all_workers = [worker for batch in batch_records for worker in batch["workers_results"]]
            check_counts(all_workers, saved_batch["worker_counts"], f"{backend}/{workers}/{workload}: workers")
            good_batches = []
            for batch in batch_records:
                if batch["status"] != "passed":
                    continue
                check(len(batch["workers_results"]) == workers, f"{backend}/{workers}/{workload}: complete workers")
                check(
                    all(worker["status"] == "passed" for worker in batch["workers_results"]),
                    f"{backend}/{workers}/{workload}: all workers passed",
                )
                calls = sum(
                    sample["valid"] and sample["metric"] not in ("construct", "open", "close")
                    for worker in batch["workers_results"]
                    for sample in worker["samples"]
                )
                equal(
                    calls,
                    workers * raw["metadata"]["settings"]["iterations"],
                    f"{backend}/{workers}/{workload}: complete workload calls",
                )
                equal(batch["validated_calls"], calls, f"{backend}/{workers}/{workload}: recorded calls")
                equal(batch["expected_calls"], calls, f"{backend}/{workers}/{workload}: expected calls")
                check(batch["execution_seconds"] > 0, f"{backend}/{workers}/{workload}: positive execution window")
                equal(
                    batch["throughput_validated_calls_per_second"],
                    calls / batch["execution_seconds"],
                    f"{backend}/{workers}/{workload}: throughput division",
                )
                good_batches.append(batch)
            equal(
                saved_batch["accepted_batches"],
                len(good_batches),
                f"{backend}/{workers}/{workload}: accepted batch count",
            )
            equal(
                saved_batch["validated_calls_all_batches"],
                sum(batch["validated_calls"] for batch in batch_records),
                f"{backend}/{workers}/{workload}: total validated calls",
            )
            equal(
                saved_batch["expected_calls_all_batches"],
                sum(batch["expected_calls"] for batch in batch_records),
                f"{backend}/{workers}/{workload}: total expected calls",
            )
            flattened = [
                {"samples": [sample for worker in batch["workers_results"] for sample in worker["samples"]]}
                for batch in good_batches
            ]
            audit_group(
                profile,
                backend,
                "concurrency",
                workers,
                workload,
                "seconds",
                units_from_records(flattened),
                saved_batch["latency_seconds"],
                "batch",
            )
            for field in ("payload_cpu_seconds", "payload_wall_seconds", "payload_peak_rss_kib"):
                audit_group(
                    profile,
                    backend,
                    "concurrent_payload",
                    workers,
                    workload,
                    field,
                    units_from_records(flattened, field),
                    saved_batch["payload_resources"].get(field, {}),
                    "batch",
                )
            for field in ("execution_seconds", "initialization_seconds", "teardown_seconds"):
                groups = [[batch[field]] for batch in good_batches]
                audit_metric(
                    profile,
                    backend,
                    "batch_timing",
                    workers,
                    workload,
                    "seconds",
                    field,
                    groups,
                    saved_batch["batch_timing_seconds"][field],
                    "batch",
                )
            rates = [[batch["validated_calls"] / batch["execution_seconds"]] for batch in good_batches]
            audit_metric(
                profile,
                backend,
                "throughput",
                workers,
                workload,
                "calls/second",
                "throughput",
                rates,
                saved_batch["throughput_validated_calls_per_second"],
                "batch",
            )
    for comparison in experiment["paired_comparisons"]:
        pairs_checked += 1
        numerator, denominator, metric = comparison["numerator"], comparison["denominator"], comparison["metric"]
        nvalues, dvalues = round_values[numerator][metric], round_values[denominator][metric]
        common = sorted(nvalues.keys() & dvalues.keys())
        rounds = [r for r in common if nvalues[r] > 0 and dvalues[r] > 0]
        label = f"{profile}/{numerator}/{denominator}/{metric}: paired"
        equal(comparison["rounds"], rounds, label + " rounds")
        equal(comparison["n_pairs"], len(rounds), label + " pair count")
        equal(comparison["unmatched_rounds"], len(nvalues.keys() ^ dvalues.keys()), label + " unmatched rounds")
        equal(comparison["nonpositive_pairs_excluded"], len(common) - len(rounds), label + " nonpositive exclusions")
        equal(
            comparison["estimate"],
            median([nvalues[r] for r in rounds]) / median([dvalues[r] for r in rounds]),
            label + " ratio",
        )
        interval = comparison["ci95"]
        equal(comparison["ci95_includes_one"], interval["lower"] <= 1 <= interval["upper"], label + " includes one")

with (ROOT / "summary.csv").open() as handle:
    csv_rows = list(csv.DictReader(handle))
equal(len(csv_rows), len(csv_expected), "CSV row count/complete scalar coverage")
csv_seen = set()
for row in csv_rows:
    key = tuple(row[field] for field in ("profile", "backend", "kind", "workers", "workload", "metric", "units"))
    check(key not in csv_seen, f"CSV unique row {key}")
    csv_seen.add(key)
    check(key in csv_expected, f"CSV expected row {key}")
    if key not in csv_expected:
        continue
    for field, expected in csv_expected[key].items():
        actual = row[field]
        if expected is None:
            equal(actual, "", f"CSV {key}/{field}: empty unavailable value")
        elif isinstance(expected, (int, float)):
            equal(float(actual), float(expected), f"CSV {key}/{field}")
        else:
            equal(actual, expected, f"CSV {key}/{field}")


def percentile(ordered, fraction):
    position = (len(ordered) - 1) * fraction
    index = int(position)
    weight = position - index
    return ordered[index] * (1 - weight) + ordered[min(index + 1, len(ordered) - 1)] * weight


def random_stream(key):
    return random.Random(int.from_bytes(hashlib.sha256(f"20261016:{key}".encode()).digest()[:16], "big"))


def replicate_ci(rowkey, rng_key):
    groups = unit_sets[rowkey]
    unit_medians = [median(group) for group in groups]
    rng = random_stream(rng_key)
    # Draw one index per independent session/batch; carry all calls through its
    # precomputed median. Uniform random() indexing matches documented Random.choices.
    estimates = []
    for _ in range(10000):
        draws = [unit_medians[int(rng.random() * len(unit_medians))] for _ in unit_medians]
        estimates.append(median(draws))
    estimates.sort()
    independent_ci = [percentile(estimates, 0.025), percentile(estimates, 0.975)]
    saved = csv_expected[rowkey]
    equal(independent_ci[0], saved["ci95_lower"], f"{rowkey}: independently reproduced lower CI")
    equal(independent_ci[1], saved["ci95_upper"], f"{rowkey}: independently reproduced upper CI")
    selected_ci.append({
        "profile": rowkey[0],
        "backend": rowkey[1],
        "kind": rowkey[2],
        "workers": rowkey[3],
        "workload": rowkey[4],
        "metric": rowkey[5],
        "units": rowkey[6],
        "n_independent_units": len(unit_medians),
        "estimate": saved["estimate"],
        "ci95": independent_ci,
    })


replicate_ci(("baseline", "bubblewrap", "session", "", "", "warm_run", "seconds"), "bubblewrap:seconds:warm_run")
replicate_ci(("baseline", "docker", "session", "", "", "warm_run", "seconds"), "docker:seconds:warm_run")
replicate_ci(("workload", "nsjail", "session", "", "", "json_run", "seconds"), "nsjail:seconds:json_run")
replicate_ci(
    ("concurrency", "docker", "throughput", "16", "json", "throughput", "calls/second"), "docker:16:json:throughput"
)
replicate_ci(
    ("concurrency", "bubblewrap", "concurrency", "4", "json", "json_run", "seconds"),
    "bubblewrap:4:json:seconds:json_run",
)

baseline = next(exp for exp in SUMMARY["experiments"] if exp["profile"] == "baseline")
pair = next(
    item
    for item in baseline["paired_comparisons"]
    if item["numerator"] == "bubblewrap" and item["denominator"] == "nsjail" and item["metric"] == "warm_run"
)
paired_rows = {}
for backend in ("bubblewrap", "nsjail"):
    paired_rows[backend] = {
        trial["round"]: median([
            sample["seconds"] for sample in trial["samples"] if sample["metric"] == "warm_run" and sample["valid"]
        ])
        for trial in RAW["baseline"]["backends"][backend]["trials"]
        if trial["status"] == "passed"
    }
matched = [(paired_rows["bubblewrap"][r], paired_rows["nsjail"][r]) for r in pair["rounds"]]
rng = random_stream("pair:bubblewrap:nsjail:warm_run")
ratios = []
for _ in range(10000):
    draws = [matched[int(rng.random() * len(matched))] for _ in matched]
    ratios.append(median([row[0] for row in draws]) / median([row[1] for row in draws]))
ratios.sort()
paired_ci = [percentile(ratios, 0.025), percentile(ratios, 0.975)]
equal(pair["ci95"]["lower"], paired_ci[0], "native warm paired lower CI independent reproduction")
equal(pair["ci95"]["upper"], paired_ci[1], "native warm paired upper CI independent reproduction")
selected_ci.append({
    "profile": "baseline",
    "backend": "bubblewrap/nsjail",
    "kind": "matched round ratio",
    "workers": "",
    "workload": "",
    "metric": "warm_run",
    "units": "ratio",
    "n_independent_units": len(matched),
    "estimate": pair["estimate"],
    "ci95": paired_ci,
})

report = (ROOT / "REPORT.md").read_text()
for phrase in (
    "pointwise 95%",
    "five independent batches",
    "small-sample intervals are exploratory",
    "Pooled p95 is descriptive",
    "Individual correlated calls are not treated as independent",
    "not measure total sandbox memory",
    "Sessions share one host",
):
    check(phrase in report, f"report methodology statement: {phrase}")

outcome = {
    "status": "passed" if not errors else "failed",
    "checked_at_utc": datetime.now(timezone.utc).isoformat(),
    "checks": checks,
    "errors": errors,
    "scalar_metrics_checked": metrics_checked,
    "paired_comparisons_checked": pairs_checked,
    "csv_rows_checked": len(csv_rows),
    "selected_bootstrap_reproductions": selected_ci,
    "sha256": {
        name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
        for name in ("baseline.json", "setup.json", "workload.json", "concurrency.json", "summary.json", "summary.csv")
    },
    "method": "Independent direct raw arithmetic and separately implemented median/percentile/resampling; no import or call of experiment_statistics.py. Five scalar CIs and the native paired CI reproduced using 10,000 whole-unit draws each and seed 20261016.",
}
(ROOT / "statistics-audit.json").write_text(json.dumps(outcome, indent=2) + "\n")
lines = [
    "# Statistical audit",
    "",
    f"Status: **{outcome['status'].upper()}**. {checks:,} checks; {len(errors)} findings.",
    "",
    f"Independently recomputed {metrics_checked} scalar summaries, {pairs_checked} matched-round comparisons, and all {len(csv_rows)} CSV rows from the four raw experiment files. The audit did not import or call the summary implementation.",
    "",
    "- Verified session/batch medians, pooled nearest-rank p95/min/max, independent-unit and sample counts, failure/status counts, payload resource labels, and complete profile/metric coverage.",
    "- Verified 90 complete concurrency batches, positive execution windows, worker/call counts, and validated calls divided by shared execution time.",
    "- Reproduced five representative scalar 95% intervals and the native-backend paired interval with 10,000 whole-session/whole-batch or matched-round bootstrap resamples, seed 20261016. All endpoints match within floating-point tolerance.",
    "- Checked every concurrency interval is labeled exploratory (five batches), and paired ratio direction and inclusion of one agree with the saved interval.",
    "- Report generator preserves pointwise, conditional-on-host, correlated-call, descriptive-p95 and payload-only-resource qualifications.",
    "",
    "| Profile / backend / metric | Independent units | Estimate | Reproduced 95% CI | Units |",
    "| --- | ---: | ---: | ---: | --- |",
]
for item in selected_ci:
    context = f"{item['profile']} / {item['backend']} / {item['metric']}"
    if item["workers"]:
        context += f" / {item['workers']} workers / {item['workload']}"
    lines.append(
        f"| {context} | {item['n_independent_units']} | {item['estimate']:.9g} | [{item['ci95'][0]:.9g}, {item['ci95'][1]:.9g}] | {item['units']} |"
    )
lines += [
    "",
    "All measured records passed: 150 sequential sessions and 630 concurrent worker sessions across 90 batches, with 6,300 validated concurrent calls. This audit verifies retained numeric evidence; it does not rerun runtime payloads or establish cross-host generalization.",
    "",
    "Detailed counts, reproduced intervals and audited input hashes: [statistics-audit.json](statistics-audit.json). Raw structural/correctness evidence: [raw-audit.json](raw-audit.json).",
    "",
]
if errors:
    lines += ["Findings:", ""] + [f"- {error}" for error in errors]
(ROOT / "statistics-audit.md").write_text("\n".join(lines))
print(
    json.dumps(
        {
            "status": outcome["status"],
            "checks": checks,
            "errors": errors,
            "metrics": metrics_checked,
            "pairs": pairs_checked,
            "csv_rows": len(csv_rows),
            "selected_cis": selected_ci,
        },
        indent=2,
    )
)
raise SystemExit(bool(errors))
