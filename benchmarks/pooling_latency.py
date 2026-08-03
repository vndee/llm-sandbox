# ruff: noqa: T201, INP001
"""Measure the latency benefit of container pooling.

The paper claims container pooling "removes creation latency from agent loops
executing code many times per task". This script substantiates that claim by
timing the same workload with and without a pool.

Both arms execute an identical trivial snippet, so what is measured is session
overhead -- container creation, startup, and teardown -- rather than the cost of
the code itself. The cold arm creates and destroys a container per execution,
which is what a naive `docker run` wrapper does. The pooled arm acquires a
pre-warmed container from the pool and returns it afterwards.

Usage:
    python benchmarks/pooling_latency.py --iterations 20 --backend docker

Requires the chosen backend to be running locally, and the language image to be
present (the script pulls it during warmup so image download is not timed).
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

from llm_sandbox import SandboxBackend, SandboxSession
from llm_sandbox.pool import PoolConfig, create_pool_manager

# Trivial by design: we are timing session overhead, not user code.
WORKLOAD = "print(sum(range(100)))"


@dataclass
class Stats:
    """Summary statistics for one arm of the benchmark, in milliseconds."""

    arm: str
    iterations: int
    mean_ms: float
    median_ms: float
    p95_ms: float
    min_ms: float
    max_ms: float

    @classmethod
    def from_samples(cls, arm: str, samples_ms: list[float]) -> Stats:
        """Summarise raw millisecond samples for one arm.

        Raises:
            ValueError: If no samples were collected.

        """
        if not samples_ms:
            msg = "cannot summarise an empty sample list"
            raise ValueError(msg)
        ordered = sorted(samples_ms)
        # Nearest-rank p95: ceil(0.95 * n), 1-indexed. round() would pick rank 10
        # of 11, which is the 91st percentile, understating the tail.
        p95_index = min(len(ordered) - 1, math.ceil(0.95 * len(ordered)) - 1)
        return cls(
            arm=arm,
            iterations=len(ordered),
            mean_ms=statistics.fmean(ordered),
            median_ms=statistics.median(ordered),
            p95_ms=ordered[p95_index],
            min_ms=ordered[0],
            max_ms=ordered[-1],
        )


def _positive_int(value: str) -> int:
    """Parse a CLI integer that must be at least 1."""
    parsed = int(value)
    if parsed < 1:
        msg = f"must be >= 1, got {parsed}"
        raise argparse.ArgumentTypeError(msg)
    return parsed


def _time_cold(backend: SandboxBackend, lang: str, iterations: int) -> list[float]:
    """Time sessions that create and destroy a container per execution."""
    samples: list[float] = []
    for _ in range(iterations):
        start = time.perf_counter()
        with SandboxSession(lang=lang, backend=backend, verbose=False) as session:
            session.run(WORKLOAD)
        samples.append((time.perf_counter() - start) * 1000)
    return samples


def _time_pooled(backend: SandboxBackend, lang: str, iterations: int) -> list[float]:
    """Time sessions that borrow a pre-warmed container from a pool."""
    config = PoolConfig(max_pool_size=4, min_pool_size=2)
    pool = create_pool_manager(backend=backend, config=config, lang=lang)
    samples: list[float] = []
    with pool:
        # Exclude pool warm-up from the measurement: the first acquisition may
        # still create a container, which is the cost pooling exists to amortise.
        with SandboxSession(lang=lang, backend=backend, pool=pool, verbose=False) as session:
            session.run(WORKLOAD)

        for _ in range(iterations):
            start = time.perf_counter()
            with SandboxSession(lang=lang, backend=backend, pool=pool, verbose=False) as session:
                session.run(WORKLOAD)
            samples.append((time.perf_counter() - start) * 1000)
    return samples


def main() -> int:
    """Run both arms, print a summary table, and optionally write JSON."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--iterations",
        type=_positive_int,
        default=20,
        help="timed executions per arm (must be >= 1)",
    )
    parser.add_argument("--backend", default="docker", choices=["docker", "podman"])
    parser.add_argument("--lang", default="python")
    parser.add_argument("--json", type=Path, default=None, help="write results to this path")
    args = parser.parse_args()

    backend = SandboxBackend(args.backend)

    print(f"warmup ({args.backend}/{args.lang}) -- pulling image if absent, not timed")
    with SandboxSession(lang=args.lang, backend=backend, verbose=False) as session:
        session.run(WORKLOAD)

    print(f"cold arm: {args.iterations} iterations")
    cold = Stats.from_samples("cold", _time_cold(backend, args.lang, args.iterations))

    print(f"pooled arm: {args.iterations} iterations")
    pooled = Stats.from_samples("pooled", _time_pooled(backend, args.lang, args.iterations))

    speedup = cold.median_ms / pooled.median_ms if pooled.median_ms else float("nan")

    print()
    print(f"{'arm':<8} {'median':>10} {'mean':>10} {'p95':>10}")
    for s in (cold, pooled):
        print(f"{s.arm:<8} {s.median_ms:>9.1f}ms {s.mean_ms:>9.1f}ms {s.p95_ms:>9.1f}ms")
    print()
    print(f"median speedup from pooling: {speedup:.1f}x")

    if args.json:
        payload = {
            "backend": args.backend,
            "lang": args.lang,
            "workload": WORKLOAD,
            "cold": asdict(cold),
            "pooled": asdict(pooled),
            "median_speedup": speedup,
        }
        args.json.write_text(json.dumps(payload, indent=2) + "\n")
        print(f"wrote {args.json}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
