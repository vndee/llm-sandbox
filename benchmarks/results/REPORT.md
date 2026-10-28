# Backend performance on latest main

Measured 2026-10-16 14:07–14:11 Asia/Saigon, from `main` at `d79a0e2038ba90f267a3f5109beea9810dde292a`. The remote was fetched before checkout. Library code was unchanged; the benchmark harness and results are new workspace files. Micromamba was excluded at the user's request.

## Results

All 30 measured sessions passed output, checksum and cleanup checks. Each runnable backend has 10 fresh-session samples and 100 samples per repeated workload. Values below are **median milliseconds**; lower is faster.

| Backend | Open fresh session | First run | Warm run | CPU workload | Close |
| --- | ---: | ---: | ---: | ---: | ---: |
| Bubblewrap | 9.08 | 20.18 | 20.54 | 44.79 | 1.19 |
| NsJail | 8.92 | 19.45 | 19.40 | 43.56 | 1.05 |
| Docker | 1,467.39 | 124.27 | 114.57 | 154.17 | 10,176.90 |

Native Bubblewrap and NsJail have substantially lower API overhead on this host. Relative to Docker, their median warm-run latencies are 5.6× and 5.9× lower, respectively. These are host-specific comparisons with the configurations below, not equivalence of security boundaries or a universal ranking.

Docker close consistently incurs approximately 10 seconds with this image and default container process. The adapter calls `container.stop()` without an explicit timeout; the result is consistent with Docker's default graceful-stop wait for this process. This lifecycle cost is separate from the much smaller warm execution cost. Keeping sessions alive can amortize startup and teardown; pooling was not measured.

| Backend | Warm run p95 (ms) | Upload 1 MiB median (ms) | Download 1 MiB median (ms) |
| --- | ---: | ---: | ---: |
| Bubblewrap | 25.53 | 0.84 | 1.06 |
| NsJail | 23.64 | 0.85 | 1.03 |
| Docker | 131.04 | 54.89 | 5.31 |

## Coverage

| Backend | Outcome | Evidence / constraint |
| --- | --- | --- |
| Docker | Measured | Docker 28.4.0, cached Python 3.13.5 image, VFS storage |
| Bubblewrap | Measured | Bubblewrap 0.12.0, system Python 3.13.5, default seccomp and no network |
| NsJail | Measured | CI-pinned source `4ff54a6e`, Kafel `76d0f41`, system Python 3.13.5, default seccomp and no network |
| Podman | Unavailable on this host | 5.4.2/crun 1.21: privileged UID/GID mapping helpers absent; isolated single-ID attempt failed mounting `devpts` |
| Kubernetes | Unavailable on this host | kind 0.30 / Kubernetes v1.34 node started, but kubelet failed because `/dev/kmsg` is absent |
| Windows MXC | Platform skipped | Requires native Windows/PSEC APIs; this host is Linux |
| WSL2 Bubblewrap | Platform skipped | Requires a Windows client and explicit WSL2 distribution |
| Micromamba | Excluded | User requested exclusion as a Docker variant |

Podman and Kubernetes were provisioned and probed before timing, then cleaned up. Their final runner preflights therefore report absent service/configuration. The underlying runtime failures are retained in [provisioning diagnostics](provisioning-diagnostics.json), [Podman output](podman-failure.txt) and [kubelet output](kubernetes-kubelet-failure.txt). No timings are substituted for unavailable backends.

## Method

- Sequential run: Docker, Bubblewrap, NsJail, then unavailable/platform checks. No provisioning or other benchmark ran concurrently.
- Ten fresh sessions per backend; two unmeasured print/CPU warmups and one transfer warmup in each; ten measured repetitions per workload per session. One complete preflight per backend is excluded from aggregates.
- Workload: Python print marker; 200,000 modular square accumulations with a closed-form expected result; deterministic 1 MiB upload/download verified by SHA-256.
- `skip_environment_setup=True`, no package installation, no streaming. Images are cached before timing. This measures fresh sessions with warm images/imports/filesystem caches, not first-install or cold-image startup.
- `run()` timings include source staging, interpreter launch, isolation/transport and execution. Native backends launch a fresh isolated process for each call. Container services retain a running container across calls in a session.
- API calls use `perf_counter`; validation, JSON persistence and progress output are outside the timer. Entire failed trials are excluded. Docker removal is confirmed independently after close.
- Medians and nearest-rank p95 are computed from raw samples. Lifecycle metrics have n=10, so their p95 is the maximum; repeated samples have n=100 and are correlated within sessions. These are exploratory latency summaries, not service-level tail guarantees. No concurrency/saturation or memory-efficiency measurements were made.
- Linux x86-64, AMD EPYC 7763, five CPUs visible with a four-CPU cgroup quota, 16 GiB memory limit. This is a containerized cloud host. Docker uses VFS; container creation performance will differ with other storage drivers.
- Every payload uses Python 3.13.5, but Docker uses the official GCC 12.2 build and native backends use the Debian GCC 14.2 build. CPU-workload differences are therefore not solely sandbox overhead. Docker has network disabled; native default isolation is retained. No equivalent security guarantee is implied.
- Backend order was fixed; host drift may bias close comparisons. There was no independent reverse-order replication. The original JSON ordering label is imprecise; `selected_backend_order` records the actual CLI order.

## Reproduction and evidence

Install the package and optional backend SDKs in a virtual environment, prepare the runtime prerequisites, and run from the repository root:

```bash
.venv/bin/python -m benchmarks.backend_benchmark \
  --backends docker bubblewrap nsjail podman kubernetes windows-mxc wsl2-bubblewrap \
  --config benchmarks/results/config-linux.json \
  --output benchmarks/results/linux-main-d79a0e2.json \
  --trials 10 --iterations 10 --warmups 2
```

The saved config contains local paths; adapt `nsjail_path` and socket paths on another host. The native commands require a host that permits user/mount namespaces; this run used the execution environment outside the tool's filesystem sandbox while retaining each backend's normal isolation. Default native virtualenv setup was not benchmarked because the system Python lacks ensurepip.

- [Benchmark CLI and methodology](../README.md)
- [Raw samples, effective configuration and provenance](linux-main-d79a0e2.json)
- [CSV summary](summary.csv); `session_ready` and `first_result` are derived by summing the relevant API durations within each trial before aggregating.
- [Exact measured harness snapshot](harness-used.py.txt), matching the recorded SHA-256. The current runner clarifies only the ordering metadata label; timing code is unchanged.
- [Exact run configuration](config-linux.json)
- [Host constraints](host.json) and [Docker version/image digests](docker-environment.json)
- [Pinned NsJail build recipe and checksums](nsjail-provisioning.txt)

The runner records its SHA-256, source commit, package versions and interpreter identity. Validation passed: 23 benchmark harness tests, 53 backend-registry tests, Ruff lint and formatting. Real measurements independently validated every result; Windows runtime behavior was not exercised.
