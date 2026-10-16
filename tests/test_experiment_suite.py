"""Check experiment balance, independent sessions and concurrent failure handling."""

# Fake session methods mirror the public adapter interface.
# ruff: noqa: D102, D107, ARG001

from __future__ import annotations

import threading
import time
from collections import Counter
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any
from unittest.mock import patch

import pytest

from benchmarks import backend_benchmark as base
from benchmarks import experiment_suite as suite


@dataclass
class Capabilities:
    """Minimal serializable backend capabilities."""

    platforms: tuple[str, ...] = ()


def definitions(*names: str) -> dict[str, Any]:
    """Build independent fake registry entries."""
    return {name: SimpleNamespace(name=name, capabilities=Capabilities()) for name in names}


def fake_trial(
    name: str,
    kwargs: dict[str, Any],
    settings: base.Settings,
    trial: dict[str, Any],
    **options: Any,
) -> None:
    """Use clearly identified, valid timing samples without any runtime."""
    trial.update(status="passed", samples=[{"metric": "warm_run", "valid": True, "seconds": 1.0}])


def run_fake_suite(
    registry: dict[str, Any], document: dict[str, Any], runner: Any = fake_trial, rounds: int = 3
) -> None:
    """Invoke only the schedule/checkpoint code, avoiding external adapters."""
    suite.run_suite(
        registry,
        {},
        base.Settings(trials=rounds),
        document,
        seed=17,
        levels=[1, 4, 16],
        workloads=["marker", "json"],
        batches=2,
        trial_runner=runner,
        progress=lambda _: None,
    )


def test_randomized_rounds_are_balanced_reproducible_and_seeded() -> None:
    """Every round includes all backends once and fixed seeds reproduce order."""
    names = ["docker", "bubblewrap", "nsjail"]
    schedule = suite.sequential_schedule(names, 30, 17)
    assert schedule == suite.sequential_schedule(names, 30, 17)
    assert schedule != suite.sequential_schedule(names, 30, 18)
    assert len(schedule) == 90
    for index in range(30):
        assert Counter(job["backend"] for job in schedule if job["round"] == index) == Counter(names)
    assert len({tuple(job["backend"] for job in schedule[index : index + 3]) for index in range(0, 90, 3)}) > 1


def test_concurrency_schedule_balances_all_combinations_per_batch() -> None:
    """Backend, workload and worker counts form a full balanced design."""
    schedule = suite.concurrency_schedule(["a", "b"], [1, 4, 16], ["marker", "json"], 5, 17)
    assert len(schedule) == 60
    assert schedule == suite.concurrency_schedule(["a", "b"], [1, 4, 16], ["marker", "json"], 5, 17)
    counts = Counter((job["backend"], job["workers"], job["workload"]) for job in schedule)
    assert set(counts.values()) == {5}
    assert len(counts) == 12


def test_all_preflights_precede_measurement_and_failed_preflight_is_excluded() -> None:
    """Preparation never overlaps measurements and unavailable backends stay visible."""
    calls = []

    def runner(
        name: str, kwargs: dict[str, Any], settings: base.Settings, trial: dict[str, Any], **options: Any
    ) -> None:
        calls.append((name, options["preflight"]))
        fake_trial(name, kwargs, settings, trial, **options)
        if name == "unavailable":
            trial["status"] = "failed"

    document = {"profile": "baseline"}
    run_fake_suite(definitions("a", "unavailable", "b"), document, runner)
    assert calls[:3] == [("a", True), ("unavailable", True), ("b", True)]
    assert all(not preflight for _, preflight in calls[3:])
    assert document["planned_order"] == document["executed_order"]
    assert len(document["planned_order"]) == 6
    assert document["backends"]["unavailable"]["status"] == "preflight_failed"
    assert document["backends"]["unavailable"]["summary"] == {}
    assert document["backends"]["a"]["successful_trials"] == 3
    assert document["backends"]["b"]["summary"]["warm_run"]["n"] == 3


def test_cleanup_failure_stops_global_schedule_and_retains_planned_order() -> None:
    """A leftover runtime prevents subsequent backends from being benchmarked."""

    def runner(
        name: str, kwargs: dict[str, Any], settings: base.Settings, trial: dict[str, Any], **options: Any
    ) -> None:
        fake_trial(name, kwargs, settings, trial, **options)
        if not options["preflight"]:
            trial.update(status="failed", cleanup_verified=False)

    document = {"profile": "baseline"}
    run_fake_suite(definitions("a", "b"), document, runner)
    assert len(document["planned_order"]) == 6
    assert len(document["executed_order"]) == 1
    assert "removal could not be confirmed" in document["stopped_early"]
    assert all(record["summary"] == {} for record in document["backends"].values())


def test_cleanup_failure_in_preflight_prevents_all_measurement() -> None:
    """Do not begin another preparation after a cleanup failure either."""

    def runner(
        name: str, kwargs: dict[str, Any], settings: base.Settings, trial: dict[str, Any], **options: Any
    ) -> None:
        fake_trial(name, kwargs, settings, trial, **options)
        trial.update(status="failed", cleanup_verified=False)

    document = {"profile": "baseline"}
    run_fake_suite(definitions("a", "b"), document, runner)
    assert document["executed_order"] == []
    assert "b" not in document["backends"]
    assert "preflight" in document["stopped_early"]


def test_setup_forces_environment_setup_enabled() -> None:
    """Both preflight and measured setup sessions use the setup-inclusive path."""
    seen = []

    def runner(
        name: str, kwargs: dict[str, Any], settings: base.Settings, trial: dict[str, Any], **options: Any
    ) -> None:
        seen.append(kwargs["skip_environment_setup"])
        fake_trial(name, kwargs, settings, trial, **options)

    document = {"profile": "setup"}
    run_fake_suite(definitions("a"), document, runner)
    assert seen == [False] * 4


def test_failed_trial_excluded_without_discarding_successful_other_rounds() -> None:
    """A complete failed trial contributes no samples even if one call succeeded."""

    def runner(
        name: str, kwargs: dict[str, Any], settings: base.Settings, trial: dict[str, Any], **options: Any
    ) -> None:
        fake_trial(name, kwargs, settings, trial, **options)
        if trial.get("round") == 1:
            trial["status"] = "failed"
            trial["samples"][0]["seconds"] = 9999.0

    document = {"profile": "baseline"}
    run_fake_suite(definitions("a"), document, runner)
    record = document["backends"]["a"]
    assert record["status"] == "partial"
    assert record["summary"]["warm_run"]["n"] == 2
    assert record["summary"]["warm_run"]["median_seconds"] == 1.0


class ConcurrentSession:
    """A fresh fake session with externally inspectable lifecycle events."""

    def __init__(self, index: int, *, fail_open: bool = False) -> None:
        self.index = index
        self.fail_open = fail_open
        self.opened = False
        self.closed = False

    def open(self) -> None:
        if self.fail_open:
            msg = "injected open failure"
            raise RuntimeError(msg)
        self.opened = True

    def close(self) -> None:
        self.closed = True


def test_concurrent_workers_overlap_and_cleanup_waits_for_last_execution() -> None:
    """Separate sessions really run together and no teardown contaminates timing."""
    sessions: list[ConcurrentSession] = []
    lock = threading.Lock()
    overlap = threading.Barrier(4, timeout=2)
    completed: set[int] = set()
    warmups: set[int] = set()

    def factory(name: str, kwargs: dict[str, Any]) -> ConcurrentSession:
        with lock:
            session = ConcurrentSession(len(sessions))
            sessions.append(session)
        return session

    def operation(session: ConcurrentSession, trial: dict[str, Any], settings: base.Settings, workload: str) -> None:
        if not trial["measured"]:
            warmups.add(session.index)
            return
        assert len(warmups) == 4

        def call() -> None:
            overlap.wait()
            time.sleep(0.005 if session.index else 0.02)
            assert not any(item.closed for item in sessions)

        base.timed(trial, "marker_resource_run", call)
        if trial["iteration"] == settings.iterations - 1:
            with lock:
                completed.add(session.index)

    original_close = ConcurrentSession.close

    def close(session: ConcurrentSession) -> None:
        assert len(completed) == 4
        original_close(session)

    batch = {"workers": 4, "workload": "marker"}
    with patch.object(ConcurrentSession, "close", close):
        suite.run_concurrent_batch(
            "fake", {}, base.Settings(iterations=2, warmups=1), batch, factory=factory, operation=operation
        )
    assert batch["status"] == "passed"
    assert batch["validated_calls"] == 8
    assert batch["throughput_validated_calls_per_second"] == pytest.approx(8 / batch["execution_seconds"])
    assert batch["initialization_seconds"] > 0
    assert batch["teardown_seconds"] > 0
    assert len(sessions) == 4
    assert all(session.closed for session in sessions)


def test_open_failure_breaks_gate_and_cleans_every_constructed_worker() -> None:
    """A peer failing before readiness releases all waiters without deadlock."""
    sessions = []
    lock = threading.Lock()

    def factory(name: str, kwargs: dict[str, Any]) -> ConcurrentSession:
        with lock:
            session = ConcurrentSession(len(sessions), fail_open=not sessions)
            sessions.append(session)
        return session

    batch = {"workers": 4, "workload": "marker"}
    suite.run_concurrent_batch(
        "fake",
        {},
        base.Settings(iterations=2, warmups=0),
        batch,
        factory=factory,
        operation=lambda *_: pytest.fail("measured operation ran after readiness failure"),
        readiness_timeout=0.1,
    )
    assert batch["status"] == "failed"
    assert batch["validated_calls"] == 0
    assert "execution_seconds" not in batch
    assert "throughput_validated_calls_per_second" not in batch
    assert len(sessions) == 4
    assert all(session.closed for session in sessions)
    assert any("injected open failure" in trial["error"]["message"] for trial in batch["workers_results"])


def test_execution_failure_retains_partial_counts_but_excludes_throughput() -> None:
    """One invalid result invalidates the batch; other workers still finish safely."""
    sessions = []
    lock = threading.Lock()

    def factory(name: str, kwargs: dict[str, Any]) -> ConcurrentSession:
        with lock:
            session = ConcurrentSession(len(sessions))
            sessions.append(session)
        return session

    def operation(session: ConcurrentSession, trial: dict[str, Any], settings: base.Settings, workload: str) -> None:
        def validate(_: Any) -> None:
            if session.index == 0:
                msg = "wrong answer"
                raise ValueError(msg)

        base.timed(trial, "json_run", lambda: None, validate)

    batch = {"workers": 4, "workload": "json"}
    suite.run_concurrent_batch(
        "fake", {}, base.Settings(iterations=2, warmups=0), batch, factory=factory, operation=operation
    )
    assert batch["status"] == "failed"
    assert batch["validated_calls"] == 6
    assert batch["expected_calls"] == 8
    assert "throughput_validated_calls_per_second" not in batch
    assert all(session.closed for session in sessions)
    failed = [trial for trial in batch["workers_results"] if trial["status"] == "failed"]
    assert len(failed) == 1
    assert not failed[0]["samples"][2]["valid"]


def test_readiness_timeout_is_bounded_and_waiters_cleanup() -> None:
    """Slow open eventually cleans up, while the coordination gate has a bound."""

    class SlowSession(ConcurrentSession):
        def open(self) -> None:
            time.sleep(0.03)
            self.opened = True

    sessions = [SlowSession(0)]
    batch = {"workers": 1, "workload": "marker"}
    suite.run_concurrent_batch(
        "fake",
        {},
        base.Settings(iterations=1, warmups=0),
        batch,
        factory=lambda *_: sessions[0],
        operation=lambda *_: pytest.fail("timed-out worker executed payload"),
        readiness_timeout=0.005,
    )
    assert batch["status"] == "failed"
    assert sessions[0].closed
    assert "readiness_error" in batch
