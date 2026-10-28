# Backend benchmark

Latest results: [expanded experiments on main, 2026-10-16](results/expanded-20261016/REPORT.md), with randomized sessions, setup, JSON/transfers and concurrency.

Run from the repository checkout with the optional SDKs for the adapters you want
to exercise installed. All registered adapters are discovered dynamically, including
Docker, Kubernetes, Podman, Micromamba, Bubblewrap, NsJail, Windows MXC and WSL2
Bubblewrap. Unsupported operating systems are explicitly skipped. Missing SDKs,
daemons, binaries, permissions, clusters and bad configuration are reported as
preflight failures; there is no fallback to another adapter or to host execution.

```bash
python -m benchmarks.backend_benchmark --list
python -m benchmarks.backend_benchmark --backends all \
  --config benchmarks/config.json --trials 5 --iterations 5 --warmups 1 \
  --output benchmarks/results/backend-benchmark.json
python -m pytest tests/test_backend_benchmark.py
```

`--backends` also accepts individual names, separated by spaces or commas. The
runner executes one session at a time. Do not run other benchmarks concurrently.
Results are written atomically after every measured operation and trial; interruption
preserves completed work. Exit code 0 means some valid measurements were collected,
1 means none were, and 130 means interrupted. Check each backend's status even when
the command exits 0.

## Expanded randomized suite

`experiment_suite.py` adds four separate experiments. The checked-in
`run_expanded.sh` runs Docker, Bubblewrap and NsJail sequentially across these
profiles, writing separate JSON files under `results/expanded-20261016`:

| Profile | Sampling per backend | Measured work per fresh session or worker | Ordering seed |
| --- | --- | --- | --- |
| `baseline` | 30 fresh sessions | 10 marker calls, 10 CPU calls, 10 upload/download pairs of 1 MiB | 20261016 |
| `setup` | 10 fresh sessions, environment setup enabled | 1 marker call, 1 CPU call, 1 upload/download pair of 1 MiB | 20261017 |
| `workload` | 10 fresh sessions | 10 JSON tasks, 3 upload/download pairs of 16 MiB | 20261018 |
| `concurrency` | 5 independent batches for each combination of 1, 4 or 16 workers and marker or JSON workload | 10 calls per worker; every worker owns a fresh session and client | 20261019 |

Each profile completes all backend preflights before measurement. Sequential
profiles shuffle backend order independently within every round. Concurrency
shuffles the backend/worker-count/workload combinations within each of five batch
rounds. Seeds and complete planned/executed orders are saved. Backends and batches
do not overlap. Two unmeasured workload warmups precede measurement in each
session; sequential profiles additionally warm one transfer pair. Baseline and
setup also record a separate first marker call and lifecycle timings.

The JSON task generates, serializes and parses 10,000 synthetic order records,
then filters and groups them; an independent arithmetic oracle checks the answer.
Larger transfers use SHA-256 checks at both endpoints. These are synthetic tasks
with warm filesystem caches, not production traces or cold storage tests.

```bash
bash benchmarks/run_expanded.sh
```

**The script overwrites its four result files. Copy an existing results directory
before rerunning.** Its configurations contain host-specific paths; inspect them
before use on another machine. To run one profile with different settings:

```bash
.venv/bin/python -m benchmarks.experiment_suite --profile baseline \
  --backends docker bubblewrap nsjail \
  --config benchmarks/results/expanded-20261016/config-common.json \
  --rounds 30 --iterations 10 --warmups 2 --seed 20261016 \
  --output benchmarks/results/baseline-new.json
```

The suite defaults to excluding Micromamba. It can discover all registered backends
with `--backends all`, retaining platform skips and preflight failures. The supplied
script selects the three previously runnable backends. Podman, Kubernetes and
Windows availability remains based on the earlier host diagnostics: Podman failed
mounting `devpts`, Kubernetes lacked `/dev/kmsg`, and Windows adapters require
Windows. The expanded run provisions no new host and makes no new claim that those
backends are supported here.

### Runtime and setup controls

`config-common.json` uses `python:3.13.5-slim-bookworm` for Docker and its extracted
`/usr/local` prefix at `/workspace/scratch/benchmark-python-3135/local` for both
native adapters. `runtime-identity.json` records the image digest and Python/libpython
hashes. This shares the CPython binary, build and standard library; native OS
shared libraries still come from the host. Native read-only mounts expose that
prefix, and the artifact limit is 32 MiB to permit 16 MiB transfers. Cached images
and runtimes are used throughout; image fetching and preparation are excluded.

`setup` forces `skip_environment_setup=false` and uses
`config-setup-offline.json`. Docker has networking disabled and sets
`PIP_NO_INDEX=1` and `PIP_DISABLE_PIP_VERSION_CHECK=1`; its setup still runs the
library's virtualenv and pip commands without downloading a newer pip. Native
adapters create their virtualenvs offline. No additional libraries are requested.
This measures each adapter's offline setup path, which differs between adapters;
it does not measure online package installation. Keep it separate from baseline
results, where setup is skipped.

### Concurrent timing and resource scope

Concurrency is a **closed-loop** test: each worker submits its next call only
after the previous call and validation finish. It measures worker-count scaling,
not behavior under a fixed incoming request rate. A start barrier waits for every
session to open and warm up. Batch throughput is validated calls divided by the
shared execution window, ending when the last worker finishes its measured calls.
That window includes host-side validation and loop overhead; individual API
latencies exclude validation. A finish barrier prevents any session teardown from
overlapping a peer's measured execution. Initialization and teardown are reported
separately. Readiness waits are bounded; SDK lifecycle calls retain their own
timeout behavior.

Any worker failure invalidates the entire batch for latency and throughput
summaries, while raw successful-call counts and error details remain available.
Workers attempt cleanup, and unconfirmed runtime removal stops the remaining
suite. Workers never write shared checkpoints; the coordinator persists each
finished batch atomically.

JSON tasks and concurrent marker calls also report payload-process CPU time and
peak RSS. CPU is the delta around the payload body, excluding interpreter startup;
peak RSS is the Python process lifetime high-water mark, including its interpreter,
in Linux KiB. These are not total backend, daemon or host CPU/memory measurements.
The concurrent marker includes JSON/resource instrumentation, so its latency is a
separate workload from the baseline's plain print marker.

### Summaries and uncertainty

Run statistics only after all timed profiles finish, avoiding contention:

```bash
.venv/bin/python -m benchmarks.experiment_statistics \
  benchmarks/results/expanded-20261016/baseline.json \
  benchmarks/results/expanded-20261016/setup.json \
  benchmarks/results/expanded-20261016/workload.json \
  benchmarks/results/expanded-20261016/concurrency.json \
  --resamples 10000 --seed 20261016 \
  --output benchmarks/results/expanded-20261016/summary.json \
  --csv benchmarks/results/expanded-20261016/summary.csv
```

The latency estimate is the median of per-session medians; concurrent latency uses
the median of per-batch medians. Throughput estimates are medians of batch
throughputs. The 95% percentile bootstrap resamples whole sessions or whole batches,
not individual calls or workers. Sequential backend ratios jointly resample
matched randomized rounds. Pooled median/p95 remain descriptive and can differ
from the session-based estimate. Only complete successful sessions or batches
contribute to estimates; failures and unavailable backends remain in the output.

There are only five independent batches per concurrency setting, so their
intervals are explicitly exploratory. More calls inside a batch do not increase
that independent sample count. All intervals are conditional on this host,
configuration and measurement period; they do not establish a universal ranking.
The result directory also preserves host/runtime provenance, configuration and
frozen harness sources for inspection.

## Configuration

The optional JSON configuration maps backend names to constructor arguments. For
example (replace image names, sockets, distribution and executable paths with real
ones on the benchmark host):

```json
{
  "docker": {
    "image": "benchmark-python:3.13",
    "client_kwargs": {"base_url": "unix:///var/run/docker.sock", "timeout": 60}
  },
  "podman": {
    "image": "localhost/benchmark-python:3.13",
    "client_kwargs": {"base_url": "unix:///run/podman/podman.sock", "timeout": 60}
  },
  "micromamba": {
    "image": "benchmark-micromamba-python:3.13",
    "environment": "base"
  },
  "bubblewrap": {
    "sandbox_config": {"python_executable": "/usr/bin/python3"}
  },
  "nsjail": {"nsjail_path": "/usr/local/bin/nsjail"},
  "windows-mxc": {"mxc_path": "C:\\tools\\llm-sandbox-mxc.exe"},
  "wsl2-bubblewrap": {"distribution": "Ubuntu", "guest_python": "/usr/bin/python3"}
}
```

`client_kwargs` creates the relevant Docker or Podman SDK client. `sandbox_config`
objects become `LocalSandboxConfig` or `WindowsMxcConfig`; `security_policy` objects
become `SecurityPolicy`. Other arguments go to the backend unchanged. Configuration
is saved in the result, so do not put secrets in the JSON. Use the SDK's normal
external credential configuration. Kubernetes uses the current kubeconfig context.

The primary profile uses Python, `skip_environment_setup=true`, `verbose=false`,
and `stream=false` where supported. Container image retention is enabled. By
default Docker, Podman and Kubernetes use `python:3.11-slim`; Micromamba uses
`mambaorg/micromamba:2.3.2`. **The base Micromamba image may lack Python: prepare a
Micromamba image containing Python, then override `image`.** Use identical Python
versions across adapters where practical, and prefer image digests or immutable
tags. Runtime Python, platform, executable and container image IDs are captured in
preflight. Existing containers/pools and Dockerfile builds are rejected because
they change the startup measurement. To explore setup-inclusive startup, set
`skip_environment_setup=false` explicitly and label that result as a separate
profile; it may include network/package installation, and native adapters may
implement setup differently.

## What is measured

Each backend first completes one unmeasured full preflight session. It may download
images and validate runtime readiness. The preflight is saved separately and never
aggregated. Images are retained. Kubernetes uses `IfNotPresent` in preflight and
`Never` in measured sessions; preload every eligible node or constrain scheduling
if benchmarking a multi-node cluster. A Docker/Podman/Micromamba measured open that
pulls or builds an image is detected and its entire trial excluded.

Every trial creates a fresh session and records elapsed wall time with
`time.perf_counter()` around these public API calls:

| Metric | Operation |
| --- | --- |
| `construct` | Create SDK client if configured and construct the session |
| `open` | Open the session, using a cached image; native adapters may launch a readiness probe |
| `first_run` | First `run()` of a Python print marker in that session |
| `warm_run` | Repeated `run()` of the same minimal print marker |
| `cpu_run` | Python loop: 200,000 modular square accumulations by default |
| `copy_to_1mib` | Upload one deterministic 1 MiB file through `copy_to_runtime` |
| `copy_from_1mib` | Download that file through `copy_from_runtime` |
| `close` | Close the session; container removal verification and SDK transport cleanup occur afterward, outside this timer |

`--trials` controls fresh session count. `--iterations` controls repeated marker,
CPU and transfer measurements per session. `--warmups` controls unmeasured marker
and CPU repetitions after `first_run`. One additional unmeasured transfer pair
warms file-operation paths in every session, including when `--warmups=0`.
`--cpu-iterations` controls loop length. `--timeout` applies to execution calls;
SDK/adapter lifecycle and transfer timeouts must be configured separately where
supported. The runner does not forcibly kill stuck SDK calls.

Marker output and exit status are checked. The CPU result is checked against an
independent closed-form formula. Upload contents are checked by SHA-256 inside the
runtime, and downloaded contents are checked on the host. Validation, payload-file
creation, result persistence and progress logging are outside the timed calls.
These checks still affect subsequent cache state. This is a warm filesystem/cache
benchmark, not a disk cold-cache or network throughput benchmark. A fresh download
destination prevents stale files from satisfying validation. File paths use each
session's workdir (including the real workspace path on native Windows).

Raw samples, error type/message/traceback and lifecycle stage are retained. Only
**complete successful measured trials**, including successful cleanup, contribute
to summaries. An individual valid call in a failed trial is excluded too. A cleanup
exception is recorded. For Docker, Podman and Micromamba, an untimed API lookup
requires a confirmed 404 after close: lingering containers invalidate the trial
and receive best-effort force removal. If removal cannot be confirmed, later trials
for that backend stop to avoid contention. Kubernetes close measures the deletion
request; an untimed poll then waits up to 60 seconds for a confirmed 404, with
bounded API reads. `cleanup_wait_seconds` records that additional wait. Pods receive
a unique benchmark ownership label before creation. If open fails before the
adapter assigns its pod handle, cleanup still uses the captured pod name and
namespace. Failed close or removal verification triggers force deletion only of a
pod bearing that label, with a UID precondition, followed by another bounded wait.
Failed verification invalidates the trial even if recovery succeeds; unconfirmed
removal stops subsequent trials. No failed
call is counted as zero latency. The median is Python's
`statistics.median`; p95 is the nearest-rank value at sorted index
`ceil(0.95 * n) - 1`, without interpolation. Each metric includes its sample count,
minimum and maximum. For small counts p95 is effectively the maximum and is
exploratory, not a reliable tail-latency estimate. Repeated samples within a session
are correlated; they are not independent session replications.

These results compare end-to-end library API latency for a specific host, runtime,
Python version and isolation policy. `run()` includes code staging, interpreter
startup and backend transport, not just Python execution. Native adapters create a
new isolated process for each command while containers keep a running container.
Default security guarantees and CPU/memory constraints differ. The original
`backend_benchmark` runner uses fixed registry order, so host drift can bias
comparisons; the expanded suite above randomizes order within rounds. Source revision, dirty status, host environment,
dependencies, workload settings and effective backend configuration accompany each
run. Do not interpret one host's measurements as universal backend rankings.
