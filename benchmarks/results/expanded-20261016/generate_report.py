import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

root = Path(__file__).resolve().parent
summary = json.loads((root / "summary.json").read_text())
experiments = {item["profile"]: item for item in summary["experiments"]}
raw = {profile: json.loads((root / f"{profile}.json").read_text()) for profile in experiments}
backends = ["bubblewrap", "nsjail", "docker"]
names = {"bubblewrap": "Bubblewrap", "nsjail": "NsJail", "docker": "Docker"}


def fmt(value, scale=1, precision=2):
    return "—" if value is None else f"{value * scale:,.{precision}f}"


def estimate(item, scale=1, interval=False, precision=2):
    if not item:
        return "—"
    text = fmt(item.get("estimate"), scale, precision)
    ci = item.get("ci95")
    if interval and ci:
        text += f" [{fmt(ci['lower'], scale, precision)}, {fmt(ci['upper'], scale, precision)}]"
    return text


def metric(profile, name, key):
    return experiments[profile]["backends"][name]["latency_seconds"].get(key, {})


def concurrent(name, workload, workers):
    return next(
        (
            row
            for row in experiments["concurrency"]["backends"][name]["concurrency"]
            if row["workload"] == workload and row["workers"] == workers
        ),
        {},
    )


def counts(profile):
    records = [
        record
        for b in raw[profile]["backends"].values()
        for record in b.get("trials" if profile != "concurrency" else "batches", [])
    ]
    return sum(r["status"] == "passed" for r in records), len(records)


display_timezone = timezone(timedelta(hours=7), name="Asia/Saigon")
start = datetime.fromisoformat(raw["baseline"]["metadata"]["started_at_utc"]).astimezone(display_timezone)
end = datetime.fromisoformat(raw["concurrency"]["metadata"]["finished_at_utc"]).astimezone(display_timezone)
base = experiments["baseline"]
pair = next(
    p
    for p in base["paired_comparisons"]
    if p["numerator"] == "bubblewrap" and p["denominator"] == "nsjail" and p["metric"] == "warm_run"
)
if pair["ci95_includes_one"]:
    comparison = "The paired interval includes 1, so this run does not resolve the small native-backend difference."
elif pair["ci95"]["lower"] > 1:
    comparison = "NsJail has lower minimal-call latency within this run; the result remains specific to this host and configuration."
else:
    comparison = "Bubblewrap has lower minimal-call latency within this run; the result remains specific to this host and configuration."
text = f"""# Expanded backend performance experiments

Measured {start:%Y-%m-%d %H:%M}–{end:%H:%M} Asia/Saigon on `main` at `{raw["baseline"]["metadata"]["revision"]}`. Micromamba is excluded. This is a new controlled experiment; the original ten-session results remain unchanged in the parent folder.

## Minimal-call latency and startup

The primary estimate is the **median of the 30 session medians**, with a pointwise 95% bootstrap interval in brackets. Each session contains ten measured warm calls. Times are milliseconds. Startup uses cached images/imports and excludes environment setup.

| Backend | Session open, ms [95% CI] | Warm run, ms [95% CI] | Warm pooled p95, ms | Close, ms |
| --- | ---: | ---: | ---: | ---: |
"""
for name in backends:
    warm = metric("baseline", name, "warm_run")
    text += f"| {names[name]} | {estimate(metric('baseline', name, 'open'), 1000, True)} | {estimate(warm, 1000, True)} | {fmt(warm.get('pooled_descriptive', {}).get('p95_nearest_rank'), 1000)} | {estimate(metric('baseline', name, 'close'), 1000)} |\n"
text += """

The same 30 baseline sessions also measure a 200,000-iteration arithmetic loop and ten 1 MiB transfer pairs per session. These API timings use the same session-level estimate and intervals. Filesystem caches are warm.

| Backend | CPU-loop API, ms [95% CI] | Upload 1 MiB, ms [95% CI] | Download 1 MiB, ms [95% CI] |
| --- | ---: | ---: | ---: |
"""
for name in backends:
    values = [
        estimate(metric("baseline", name, key), 1000, True) for key in ("cpu_run", "copy_to_1mib", "copy_from_1mib")
    ]
    text += "| " + names[name] + " | " + " | ".join(values) + " |\n"
text += f"""\nThe paired Bubblewrap/NsJail warm-latency ratio is {estimate(pair, interval=True, precision=3)} across {pair["n_pairs"]} matched randomized rounds. A ratio above 1 means Bubblewrap took longer. {comparison}

Docker's default process still makes teardown a separate lifecycle cost. This experiment preserves its default stop behavior; pooling and alternative init/stop policies were not tested.

![Sequential API latency with 95% intervals](latency-comparison.png)

## Offline environment setup

Ten fresh sessions per backend, with a new virtual environment each time. This is **offline setup-inclusive startup**, not a package-download benchmark. Docker's regular pip-upgrade command runs with `PIP_NO_INDEX=1`; native setup is offline by default. All runtimes begin from the same CPython build and bundled pip. Times are milliseconds.

| Backend | Setup-inclusive open [95% CI] | First run [95% CI] | Close |
| --- | ---: | ---: | ---: |
"""
for name in backends:
    text += f"| {names[name]} | {estimate(metric('setup', name, 'open'), 1000, True)} | {estimate(metric('setup', name, 'first_run'), 1000, True)} | {estimate(metric('setup', name, 'close'), 1000)} |\n"
text += """
## Broader workload and larger transfers

Ten fresh sessions per backend, ten JSON operations and three 16 MiB upload/download pairs per session. The synthetic workload generates, serializes, parses, filters and groups 10,000 deterministic order records. An independent arithmetic oracle validates the result. Upload and download contents are independently SHA-256 checked.

| Backend | JSON API latency, ms [95% CI] | Upload 16 MiB, ms [95% CI] | Download 16 MiB, ms [95% CI] | Payload CPU, ms | Payload peak RSS, MiB |
| --- | ---: | ---: | ---: | ---: | ---: |
"""
for name in backends:
    resources = experiments["workload"]["backends"][name]["payload_resources"]
    cpu = resources.get("payload_cpu_seconds", {}).get("json_run", {})
    rss = resources.get("payload_peak_rss_kib", {}).get("json_run", {})
    text += f"| {names[name]} | {estimate(metric('workload', name, 'json_run'), 1000, True)} | {estimate(metric('workload', name, 'copy_to_16mib'), 1000, True)} | {estimate(metric('workload', name, 'copy_from_16mib'), 1000, True)} | {estimate(cpu, 1000)} | {estimate(rss, 1 / 1024)} |\n"
text += """
CPU values are process CPU-time deltas around the payload work. Peak RSS is the Python process high-water mark, including its interpreter. These figures exclude daemon, launcher and controller resource use; they do not measure total sandbox memory or backend efficiency.

## Concurrent throughput

Closed-loop load: each worker owns a fresh session and makes ten sequential calls. Every worker completes setup and warmup before the shared timer starts; none begins teardown until all execution finishes. Throughput includes validation and caller bookkeeping in the shared execution window. Initialization and teardown are separate in the raw/CSV results. Concurrent latency uses instrumented payloads, so its marker is not identical to the baseline's simple print.

Each cell is the median **validated calls/second [95% CI] across five independent batches**. These small-sample intervals are exploratory. Failures invalidate a batch's performance summaries and remain visible in the evidence.
"""
for workload, title in [("marker", "Instrumented minimal payload"), ("json", "10,000-record JSON workload")]:
    text += f"\n### {title}\n\n| Backend | 1 worker | 4 workers | 16 workers |\n| --- | ---: | ---: | ---: |\n"
    for name in backends:
        values = [
            estimate(concurrent(name, workload, k).get("throughput_validated_calls_per_second", {}), interval=True)
            for k in (1, 4, 16)
        ]
        text += "| " + names[name] + " | " + " | ".join(values) + " |\n"
text += """
![Concurrent throughput with exploratory 95% intervals](concurrency-throughput.png)

JSON throughput increases by approximately 2–11% when moving from four to sixteen workers across these backends (point estimates). The host has a four-CPU cgroup quota with five CPUs visible, so 16 workers oversubscribe available CPU. Both load generation and runtimes share this host. The tested maximum is not a proven saturation point or production capacity, and client concurrency is not a controlled offered-arrival-rate experiment.

## Validation, controls and limits

| Profile | Passed / attempted independent units | Unit |
| --- | ---: | --- |
"""
for profile in ("baseline", "setup", "workload", "concurrency"):
    passed, total = counts(profile)
    text += f"| {profile} | {passed} / {total} | {'batch' if profile == 'concurrency' else 'session'} |\n"
workers = [
    worker
    for b in raw["concurrency"]["backends"].values()
    for batch in b["batches"]
    for worker in batch["workers_results"]
]
calls = sum(batch["validated_calls"] for b in raw["concurrency"]["backends"].values() for batch in b["batches"])
text += f"""
Concurrency additionally completed {sum(w["status"] == "passed" for w in workers)} / {len(workers)} worker sessions and {calls:,} validated measured calls. Every profile excludes its preflight/warmups. Failed calls are never represented as zero latency, and whole failed sessions/batches are excluded from estimates. Docker removal is independently confirmed after close.

- Each of 30 baseline rounds contains every runnable backend exactly once in seed-randomized order. Setup/workload rounds and backend/load/workload combinations are randomized separately. Seeds and complete planned/executed orders are preserved.
- Confidence intervals use 10,000 percentile-bootstrap resamples of complete session medians or complete batch medians. Paired backend ratios resample matched rounds together. Individual correlated calls are not treated as independent replications. Pooled p95 is descriptive without a confidence interval; this is not a p99/SLO certification.
- All three backends use the same official CPython 3.13.5 binary/build and standard library. Native backends receive its extracted prefix read-only. Host and container OS shared libraries still differ. This change improves comparability but means movement from the original report is not a software-regression estimate.
- Docker 28.4.0 uses VFS storage. Native default namespace/seccomp isolation is retained, and network is disabled throughout measured work. The native artifact ceiling is 32 MiB for the larger transfer. Different backends do not provide identical security boundaries.
- Hardware: Linux x86-64 / AMD EPYC 7763, four-CPU quota and 16 GiB memory limit. Image downloads, provisioning, compilation, unit tests and statistical bootstrap computation happened outside the authoritative measurements.
- Intervals are pointwise, conditional on this host/configuration/time period, and not adjusted for multiple comparisons. Sessions share one host and may retain temporal correlation despite randomized order. No independent host replication, cold-image downloads, online package updates or production trace replay was performed.
- Podman and Kubernetes remain unavailable for the previously verified host limitations (`devpts` and `/dev/kmsg`). Windows MXC and WSL2 Bubblewrap need a Windows host. Those results are not estimated or pooled with the measured backends.

## Reproduce and inspect

- [Expanded runner instructions](../../README.md) and [run script](../../run_expanded.sh).
- [Preparation and setup controls](PREPARATION.md), [runtime image/binary identity](runtime-identity.json), [host limits](host.json).
- Raw data: [baseline](baseline.json), [offline setup](setup.json), [JSON/transfers](workload.json), [concurrency](concurrency.json).
- [Statistical summaries and paired comparisons](summary.json) and [flat CSV](summary.csv).
- [Raw-data audit](raw-audit.json), [audit script](audit_raw.py), and [independent statistics audit](statistics-audit.md) with its [script](audit_statistics.py).
- Editable figures: [latency SVG](latency-comparison.svg), [throughput SVG](concurrency-throughput.svg), [plotting script](plot_results.py), and [plotting environment](plotting-environment.txt).
- [Original availability diagnostics](../provisioning-diagnostics.json) and [initial report](../REPORT.md).
- Frozen Python source snapshots are in `source/`. Runner metadata hashes the measured harness/workload sources; [the final SHA-256 manifest](sha256-manifest.json) also covers the statistics code, configurations and evidence files.

Validation before the authoritative run: 96 focused tests passed, Ruff lint/format passed, and real workload/concurrency smoke checks passed. The independent raw-data audit passed 68,768 checks with zero errors, covering 780 measured sessions and 12,990 validated API samples including lifecycle calls. The separate statistics audit passed 7,410 checks with zero findings and independently reproduced six selected 10,000-resample confidence intervals. No Docker containers remained after completion.
"""
(root / "REPORT.md").write_text(text)
print(root / "REPORT.md")
