"""Randomized fresh-session and closed-loop concurrency backend experiments.

Use ``python -m benchmarks.experiment_suite --help``. Profiles are separate
experiments and output files: baseline, setup, workload and concurrency.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import random
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any

from benchmarks import backend_benchmark as base

if TYPE_CHECKING:
    from collections.abc import Callable


def sequential_schedule(names: list[str], rounds: int, seed: int) -> list[dict[str, Any]]:
    """Shuffle backend order independently within each complete round."""
    generator = random.Random(seed)  # noqa: S311 - reproducible experiment ordering, not security.
    schedule = []
    for round_index in range(rounds):
        order = names.copy()
        generator.shuffle(order)
        schedule.extend({"round": round_index, "backend": name} for name in order)
    return schedule


def concurrency_schedule(
    names: list[str], levels: list[int], workloads: list[str], batches: int, seed: int
) -> list[dict[str, Any]]:
    """Balance every round, randomizing backend/worker-count/workload jobs."""
    generator = random.Random(seed)  # noqa: S311
    schedule = []
    for index in range(batches):
        jobs = [
            {"index": index, "backend": name, "workers": level, "workload": workload}
            for name in names
            for level in levels
            for workload in workloads
        ]
        generator.shuffle(jobs)
        schedule.extend(jobs)
    return schedule


def prepare_backends(
    definitions: dict[str, Any],
    configs: dict[str, Any],
    settings: base.Settings,
    document: dict[str, Any],
    *,
    factory: Callable[..., Any],
    trial_runner: Callable[..., None],
    checkpoint: Callable[[], None],
    progress: Callable[[str], None],
) -> list[str]:
    """Complete all availability checks before any measured experiment starts."""
    ready = []
    document["preflight_order"] = list(definitions)
    for name, definition in definitions.items():
        record: dict[str, Any] = {
            "status": "preparing",
            "capabilities": asdict(definition.capabilities),
            "requested_config": configs.get(name, {}),
            "trials": [],
            "batches": [],
            "summary": {},
        }
        document["backends"][name] = record
        allowed = definition.capabilities.platforms
        if allowed and sys.platform not in allowed:
            record.update(status="skipped", reason=f"Requires {list(allowed)}; host is {sys.platform}")
            checkpoint()
            continue
        try:
            kwargs = base.effective_kwargs(name, configs.get(name, {}))
            if document["profile"] == "setup":
                kwargs["skip_environment_setup"] = False
            record["effective_config"] = kwargs
        except Exception as exc:  # noqa: BLE001 - retain unavailable backends independently.
            record.update(status="configuration_failed", error=base.error_record(exc))
            checkpoint()
            continue
        trial: dict[str, Any] = {"status": "running", "samples": [], "measured": False}
        record["preflight"] = trial
        progress(f"{name}: preflight")
        checkpoint()
        trial_runner(name, kwargs, settings, trial, preflight=True, factory=factory, checkpoint=checkpoint)
        record["status"] = "ready" if trial["status"] == "passed" else "preflight_failed"
        if trial.get("cleanup_verified") is False:
            document["stopped_early"] = f"{name}: preflight runtime removal could not be confirmed"
            checkpoint()
            return []
        if trial["status"] == "passed":
            ready.append(name)
        checkpoint()
    return ready


def execute_operation(session: Any, trial: dict[str, Any], settings: base.Settings, workload: str) -> None:
    """Run one instrumented, validated workload; resource scope is payload only."""
    from benchmarks.experiment_workloads import execute_workload

    execute_workload(session, trial, settings, workload)


def run_concurrent_batch(  # noqa: PLR0915 - one lifecycle makes barrier/cleanup ordering explicit.
    name: str,
    kwargs: dict[str, Any],
    settings: base.Settings,
    batch: dict[str, Any],
    *,
    factory: Callable[..., Any] = base.make_session,
    operation: Callable[..., None] = execute_operation,
    readiness_timeout: float = 120.0,
) -> None:
    """Synchronize fresh workers after warmup and time a shared execution window.

    Each worker owns its session/client. A failed worker aborts the readiness
    barrier; all workers attempt cleanup. Lifecycle SDK calls retain their own
    timeout behavior; readiness and execution calls have explicit bounds.
    """
    worker_count = batch["workers"]
    workload = batch["workload"]
    batch.update(status="running", workers_results=[], expected_calls=worker_count * settings.iterations)
    started = time.perf_counter()
    shared: dict[str, float] = {}

    def start_clock() -> None:
        shared["started"] = time.perf_counter()

    gate = threading.Barrier(worker_count + 1, action=start_clock, timeout=readiness_timeout)
    finish_gate = threading.Barrier(worker_count, timeout=settings.iterations * settings.timeout + readiness_timeout)

    def worker(index: int) -> dict[str, Any]:
        trial: dict[str, Any] = {"worker": index, "status": "running", "samples": [], "measured": True}
        session = None
        try:
            trial["stage"] = "construct"
            session = base.timed(trial, "construct", lambda: factory(name, copy.deepcopy(kwargs)))
            base.image_policy(session, name, preflight=False)
            trial["stage"] = "open"
            base.timed(trial, "open", session.open, lambda _: base.validate_no_image_preparation(session, name))
            trial["stage"] = "warmup"
            warmup: dict[str, Any] = {"samples": [], "measured": False}
            for _ in range(settings.warmups):
                operation(session, warmup, settings, workload)
            trial["stage"] = "ready"
            trial["ready_at"] = time.perf_counter()
            gate.wait()
            trial["stage"] = "execute"
            for iteration in range(settings.iterations):
                trial["iteration"] = iteration
                operation(session, trial, settings, workload)
            trial["execution_finished_at"] = time.perf_counter()
            trial["status"] = "passed"
        except Exception as exc:  # noqa: BLE001 - preserve failures and release every waiting worker.
            trial["status"] = "failed"
            trial["error"] = base.error_record(exc)
            if "started" in shared:
                trial["execution_finished_at"] = time.perf_counter()
            if "started" not in shared:
                gate.abort()
        finally:
            # Teardown must not contend with a peer still executing measured calls.
            if "started" in shared:
                trial.setdefault("execution_finished_at", time.perf_counter())
                try:
                    finish_gate.wait()
                except threading.BrokenBarrierError as exc:
                    trial["status"] = "failed"
                    trial["finish_barrier_error"] = base.error_record(exc)
            if session is not None:
                base.finish_session(session, name, trial, lambda: None)
            trial["finished_at"] = time.perf_counter()
        return trial

    with ThreadPoolExecutor(max_workers=worker_count, thread_name_prefix=f"benchmark-{name}") as executor:
        futures = [executor.submit(worker, index) for index in range(worker_count)]
        try:
            gate.wait()
        except threading.BrokenBarrierError as exc:
            batch["readiness_error"] = base.error_record(exc)
        batch["workers_results"] = [future.result() for future in futures]
    finished = time.perf_counter()
    batch["total_seconds"] = finished - started
    batch["validated_calls"] = sum(
        sample["valid"] and sample["metric"] not in {"construct", "open", "close"}
        for trial in batch["workers_results"]
        for sample in trial["samples"]
    )
    batch["status"] = (
        "passed"
        if all(trial["status"] == "passed" for trial in batch["workers_results"])
        and batch["validated_calls"] == batch["expected_calls"]
        else "failed"
    )
    if any(trial.get("cleanup_verified") is False for trial in batch["workers_results"]):
        batch["cleanup_verified"] = False
    if "started" in shared:
        execution_end = max(trial["execution_finished_at"] for trial in batch["workers_results"])
        batch["initialization_seconds"] = shared["started"] - started
        batch["execution_seconds"] = execution_end - shared["started"]
        batch["teardown_seconds"] = finished - execution_end
        if batch["status"] == "passed":
            batch["throughput_validated_calls_per_second"] = batch["validated_calls"] / batch["execution_seconds"]
    else:
        batch["initialization_and_cleanup_seconds"] = finished - started


def finish_records(document: dict[str, Any], expected_trials: int) -> None:
    """Summarize only completed valid trials and record failures independently."""
    for record in document["backends"].values():
        if record["status"] not in {"ready", "running"}:
            continue
        trials = record["batches"] if document["profile"] == "concurrency" else record["trials"]
        passed = sum(trial["status"] == "passed" for trial in trials)
        record["successful_trials"] = passed
        record["failed_trials"] = sum(trial["status"] == "failed" for trial in trials)
        record["status"] = "passed" if passed == expected_trials else "partial" if passed else "failed"
        if document["profile"] != "concurrency":
            record["summary"] = base.summarize(record["trials"])


def run_suite(
    definitions: dict[str, Any],
    configs: dict[str, Any],
    settings: base.Settings,
    document: dict[str, Any],
    *,
    seed: int,
    levels: list[int],
    workloads: list[str],
    batches: int,
    factory: Callable[..., Any] = base.make_session,
    trial_runner: Callable[..., None] = base.run_trial,
    batch_runner: Callable[..., None] = run_concurrent_batch,
    checkpoint: Callable[[], None] = lambda: None,
    progress: Callable[[str], None] = print,
) -> None:
    """Preflight all backends, then execute one isolated scheduled job at a time."""
    document.setdefault("backends", {})
    document.update(planned_order=[], executed_order=[])
    ready = prepare_backends(
        definitions,
        configs,
        settings,
        document,
        factory=factory,
        trial_runner=trial_runner,
        checkpoint=checkpoint,
        progress=progress,
    )
    concurrent = document["profile"] == "concurrency"
    schedule = (
        concurrency_schedule(ready, levels, workloads, batches, seed)
        if concurrent
        else sequential_schedule(ready, settings.trials, seed)
    )
    document["planned_order"] = schedule
    checkpoint()
    for job in schedule:
        name = job["backend"]
        record = document["backends"][name]
        record["status"] = "running"
        trial: dict[str, Any] = {**job, "status": "running", "samples": [], "measured": True}
        document["executed_order"].append(dict(job))
        if concurrent:
            record["batches"].append(trial)
            progress(f"{name}: {job['workload']}, workers={job['workers']}, batch {job['index'] + 1}/{batches}")
            checkpoint()
            batch_runner(name, record["effective_config"], settings, trial, factory=factory)
        else:
            trial["index"] = len(record["trials"])
            record["trials"].append(trial)
            progress(f"{name}: {document['profile']} round {job['round'] + 1}/{settings.trials}")
            checkpoint()
            trial_runner(
                name,
                record["effective_config"],
                settings,
                trial,
                preflight=False,
                factory=factory,
                checkpoint=checkpoint,
            )
            record["summary"] = base.summarize(record["trials"])
        checkpoint()
        if trial.get("cleanup_verified") is False:
            document["stopped_early"] = f"{name}: runtime removal could not be confirmed; stopped to avoid contention"
            break
    expected_trials = batches * len(levels) * len(workloads) if concurrent else settings.trials
    finish_records(document, expected_trials)
    checkpoint()


def parse_args(argv: list[str] | None) -> argparse.Namespace:
    """Parse explicit profile settings so separate experiments stay reproducible."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=["baseline", "setup", "workload", "concurrency"], default="baseline")
    parser.add_argument("--backends", nargs="+", default=["all"])
    parser.add_argument("--exclude", nargs="*", default=["micromamba"])
    parser.add_argument("--config", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--rounds", type=base.positive_int, default=30)
    parser.add_argument("--iterations", type=base.positive_int, default=10)
    parser.add_argument("--warmups", type=int, default=2)
    parser.add_argument("--cpu-iterations", type=base.positive_int, default=200_000)
    parser.add_argument("--timeout", type=float, default=60.0)
    parser.add_argument("--seed", type=int, default=20261016)
    parser.add_argument("--levels", nargs="+", type=base.positive_int, default=[1, 4, 16])
    parser.add_argument("--batches", type=base.positive_int, default=5)
    parser.add_argument("--workloads", nargs="+", choices=["marker", "json"], default=["marker", "json"])
    parser.add_argument("--transfer-iterations", type=base.positive_int, default=3)
    args = parser.parse_args(argv)
    if args.warmups < 0 or not math.isfinite(args.timeout) or args.timeout <= 0:
        parser.error("warmups must be nonnegative and timeout must be finite and positive")
    return args


def main(argv: list[str] | None = None) -> int:
    """Write atomic checkpoints and retain provenance for every profile."""
    from llm_sandbox.backends import list_backends

    args = parse_args(argv)
    definitions = {definition.name: definition for definition in list_backends()}
    selected = list(dict.fromkeys(name for group in args.backends for name in group.split(",")))
    if selected == ["all"]:
        selected = list(definitions)
    unknown = set(selected + args.exclude) - definitions.keys()
    if unknown:
        msg = f"Unknown backends: {sorted(unknown)}"
        raise SystemExit(msg)
    selected = [name for name in selected if name not in args.exclude]
    configs = json.loads(args.config.read_text(encoding="utf-8")) if args.config else {}
    if not isinstance(configs, dict) or any(not isinstance(value, dict) for value in configs.values()):
        msg = "config must map backend names to constructor-kwargs objects"
        raise SystemExit(msg)
    if set(configs) - definitions.keys():
        msg = f"Unknown backend configuration keys: {sorted(set(configs) - definitions.keys())}"
        raise SystemExit(msg)
    settings = base.Settings(args.rounds, args.iterations, args.warmups, args.cpu_iterations, args.timeout)
    document: dict[str, Any] = {"schema_version": 2, "profile": args.profile, "metadata": base.metadata(settings)}
    document["metadata"].update(
        profile=args.profile,
        seed=args.seed,
        excluded_backends=args.exclude,
        levels=args.levels,
        batches=args.batches,
        workloads=args.workloads,
        transfer_iterations=args.transfer_iterations,
        suite_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        ordering="All preflights, then seeded shuffle within each round; planned and executed orders recorded.",
        concurrency_clock=(
            "Closed-loop workers. Shared window starts after all opens/warmups and includes validation; "
            "ends at last worker execution completion. Per-call API durations exclude validation. "
            "Init/teardown separate; no overlapping backend batches. Payload resources exclude runtime/host overhead."
        ),
    )
    from benchmarks import experiment_workloads

    document["metadata"]["workload_definitions"] = {
        "json_records": experiment_workloads.DEFAULT_RECORDS,
        "large_transfer_bytes": experiment_workloads.DEFAULT_PAYLOAD_BYTES,
        "resource_scope": experiment_workloads.RESOURCE_SCOPE,
        "marker_instrumentation": "baseline/setup plain marker; concurrency instrumented marker",
    }
    if args.profile == "workload":
        payload = bytes(range(256)) * (experiment_workloads.DEFAULT_PAYLOAD_BYTES // 256)
        document["metadata"]["payload_bytes"] = len(payload)
        document["metadata"]["payload_sha256"] = hashlib.sha256(payload).hexdigest()
    elif args.profile == "concurrency":
        document["metadata"]["payload_bytes"] = None
        document["metadata"]["payload_sha256"] = None
    document["metadata"]["workloads_sha256"] = hashlib.sha256(
        Path(experiment_workloads.__file__).read_bytes()
    ).hexdigest()

    def checkpoint() -> None:
        base.save_json(args.output, document)

    trial_runner = base.run_trial
    if args.profile == "workload":
        from functools import partial

        trial_runner = partial(experiment_workloads.run_workload_trial, transfer_iterations=args.transfer_iterations)
    try:
        run_suite(
            {name: definitions[name] for name in selected},
            configs,
            settings,
            document,
            seed=args.seed,
            levels=list(dict.fromkeys(args.levels)),
            workloads=list(dict.fromkeys(args.workloads)),
            batches=args.batches,
            trial_runner=trial_runner,
            checkpoint=checkpoint,
            progress=lambda message: print(message, flush=True),  # noqa: T201
        )
    except KeyboardInterrupt:
        document["interrupted"] = True
        checkpoint()
        return 130
    document["metadata"]["finished_at_utc"] = datetime.now(timezone.utc).isoformat()  # noqa: UP017
    checkpoint()
    print(f"Results: {args.output.resolve()}")  # noqa: T201
    return 0 if any(record.get("successful_trials", 0) for record in document["backends"].values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
