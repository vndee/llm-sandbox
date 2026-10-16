"""Correctness-checked synthetic workloads for the expanded Linux experiments.

The JSON task generates, serializes and parses deterministic order records, then
filters and groups them. It is a representative synthetic stdlib task, not a
production trace. Resource values describe only the Python payload process:
CPU is a delta around its work, and peak RSS is the process lifetime high-water
mark (including its interpreter). Neither includes runtime daemons or host APIs.
"""

from __future__ import annotations

import hashlib
import json
import math
import tempfile
import uuid
from pathlib import Path
from typing import TYPE_CHECKING, Any

from benchmarks import backend_benchmark as benchmark

if TYPE_CHECKING:
    from collections.abc import Callable

DEFAULT_RECORDS = 10_000
DEFAULT_PAYLOAD_BYTES = 16 * 1024 * 1024
RESOURCE_FIELDS = ("payload_cpu_seconds", "payload_wall_seconds", "payload_peak_rss_kib")
RESOURCE_SCOPE = "payload Python process only; peak RSS includes interpreter; excludes backend/daemon/host"


def instrumented_code(body: str) -> str:
    """Wrap fixed Python assigning ``result`` with Linux process instrumentation."""
    return (
        "import json, resource, sys, time\n"
        "if not sys.platform.startswith('linux'):\n"
        "    raise RuntimeError('Payload ru_maxrss instrumentation requires Linux KiB units')\n"
        "_usage_before = resource.getrusage(resource.RUSAGE_SELF)\n"
        "_cpu_before = _usage_before.ru_utime + _usage_before.ru_stime\n"
        "_wall_before = time.perf_counter()\n"
        f"{body}\n"
        "_wall_elapsed = time.perf_counter() - _wall_before\n"
        "_usage_after = resource.getrusage(resource.RUSAGE_SELF)\n"
        "print(json.dumps({'result': result, 'resources': {\n"
        "    'payload_cpu_seconds': _usage_after.ru_utime + _usage_after.ru_stime - _cpu_before,\n"
        "    'payload_wall_seconds': _wall_elapsed,\n"
        "    'payload_peak_rss_kib': _usage_after.ru_maxrss,\n"
        f"}}, 'resource_scope': {RESOURCE_SCOPE!r}}}, sort_keys=True))"
    )


def marker_workload() -> tuple[str, str]:
    """Return a separate instrumented marker, leaving the baseline marker intact."""
    return instrumented_code(f"result = {benchmark.MARKER!r}"), benchmark.MARKER


def json_oracle(records: int) -> dict[str, Any]:
    """Compute expected aggregates by arithmetic progressions, without parsing rows."""
    if isinstance(records, bool) or not isinstance(records, int) or records <= 0:
        msg = "records must be a positive integer"
        raise ValueError(msg)
    # Region has period 5, active status period 4. Each residue modulo 20
    # forms an arithmetic progression with a closed-form count and sum.
    groups = {f"region-{region}": {"count": 0, "amount_cents": 0} for region in range(5)}
    for residue in range(20):
        if residue % 4 == 0 or residue >= records:
            continue
        count = (records - 1 - residue) // 20 + 1
        id_sum = count * residue + 20 * count * (count - 1) // 2
        group = groups[f"region-{residue % 5}"]
        group["count"] += count
        group["amount_cents"] += 100 * count + 17 * id_sum
    return {
        "records": records,
        "selected": sum(group["count"] for group in groups.values()),
        "groups": [dict(region=region, **group) for region, group in sorted(groups.items()) if group["count"]],
    }


def json_workload(records: int = DEFAULT_RECORDS) -> tuple[str, dict[str, Any]]:
    """Generate, serialize, parse, filter and aggregate a synthetic order dataset."""
    expected = json_oracle(records)
    body = (
        f"rows = [{{'id': i, 'region': 'region-' + str(i % 5), 'active': i % 4 != 0,\n"
        f"         'amount_cents': 100 + 17 * i, 'product': 'product-' + str(i % 23)}}\n"
        f"        for i in range({records})]\n"
        "encoded = json.dumps(rows, separators=(',', ':'))\n"
        "decoded = json.loads(encoded)\n"
        "groups = {}\n"
        "selected = 0\n"
        "for row in decoded:\n"
        "    if not row['active']:\n"
        "        continue\n"
        "    selected += 1\n"
        "    group = groups.setdefault(row['region'], {'count': 0, 'amount_cents': 0})\n"
        "    group['count'] += 1\n"
        "    group['amount_cents'] += row['amount_cents']\n"
        "result = {'records': len(decoded), 'selected': selected,\n"
        "          'groups': [dict(region=region, **groups[region]) for region in sorted(groups)]}"
    )
    return instrumented_code(body), expected


def validate_instrumented_output(output: Any, expected: Any) -> dict[str, float]:
    """Reject wrong results and malformed resources after the host API timer stops."""
    if output.exit_code != 0:
        msg = f"Instrumented execution failed: exit_code={output.exit_code}, stderr={output.stderr[:2048]!r}"
        raise ValueError(msg)
    try:
        document = json.loads(output.stdout)
    except (ValueError, TypeError) as exc:
        msg = f"Instrumented output is not valid JSON: {output.stdout[:2048]!r}"
        raise ValueError(msg) from exc
    if not isinstance(document, dict) or "result" not in document:
        msg = "Instrumented output must be an object containing result"
        raise ValueError(msg)
    # JSON comparison also distinguishes booleans from expected integer values.
    if json.dumps(document["result"], sort_keys=True) != json.dumps(expected, sort_keys=True):
        msg = f"Instrumented result mismatch: expected={expected!r}, actual={document['result']!r}"
        raise ValueError(msg)
    resources = document.get("resources")
    if not isinstance(resources, dict) or document.get("resource_scope") != RESOURCE_SCOPE:
        msg = "Instrumented output has missing resources or an invalid resource scope"
        raise ValueError(msg)
    validated = {}
    for field in RESOURCE_FIELDS:
        value = resources.get(field)
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
            or value < 0
            or (field == "payload_peak_rss_kib" and value == 0)
        ):
            msg = f"Invalid payload resource {field}: {value!r}"
            raise ValueError(msg)
        validated[field] = float(value)
    return validated


def get_workload(name: str, records: int = DEFAULT_RECORDS) -> tuple[str, Callable[[Any], dict[str, float]]]:
    """Return generated code and its independent result/resource validator."""
    if name == "marker":
        code, expected = marker_workload()
    elif name == "json":
        code, expected = json_workload(records)
    else:
        msg = f"Unknown instrumented workload: {name}"
        raise ValueError(msg)
    return code, lambda output: validate_instrumented_output(output, expected)


def execute_workload(
    session: Any,
    trial: dict[str, Any],
    settings: benchmark.Settings,
    workload: str = "json",
    checkpoint: Callable[[], None] = lambda: None,
    *,
    records: int = DEFAULT_RECORDS,
) -> None:
    """Record one API duration and separate validated payload-process resources."""
    code, validate = get_workload(workload, records)
    metric = "json_run" if workload == "json" else "marker_resource_run"

    def check(output: Any) -> None:
        resources = validate(output)
        trial["samples"][-1].update(resources)
        trial.setdefault("resources", []).append({"metric": metric, **resources})

    benchmark.timed(trial, metric, lambda: session.run(code, timeout=settings.timeout), check, checkpoint)


def transfer_workload(
    session: Any,
    trial: dict[str, Any],
    temporary: Path,
    sequence: int,
    timeout: float,
    checkpoint: Callable[[], None] = lambda: None,
    *,
    payload_bytes: int = DEFAULT_PAYLOAD_BYTES,
) -> None:
    """Measure fresh configurable transfers, independently checking both endpoints."""
    if isinstance(payload_bytes, bool) or not isinstance(payload_bytes, int) or payload_bytes <= 0:
        msg = "payload_bytes must be a positive integer"
        raise ValueError(msg)
    pattern = bytes(range(256))
    payload = pattern * (payload_bytes // len(pattern)) + pattern[: payload_bytes % len(pattern)]
    expected_digest = hashlib.sha256(payload).hexdigest()
    identifier = f"{sequence}-{uuid.uuid4().hex}"
    basename = f"benchmark-large-payload-{identifier}.bin"
    source = temporary / f"source-{identifier}.bin"
    source.write_bytes(payload)
    output_directory = temporary / f"download-{identifier}"
    output_directory.mkdir()
    destination = output_directory / basename
    remote = f"{session.config.workdir.rstrip('/')}/{basename}"
    digest_code = f"import hashlib\nprint(hashlib.sha256(open({basename!r}, 'rb').read()).hexdigest())"
    label = "16mib" if payload_bytes == DEFAULT_PAYLOAD_BYTES else f"{payload_bytes}bytes"

    def validate_upload(_: Any) -> None:
        benchmark.validate_output(session.run(digest_code, timeout=timeout), expected_digest)

    def validate_download(_: Any) -> None:
        # Every call has a new directory: a missing write cannot pass by reading
        # a previous successful download, even when a caller reuses sequence.
        digest = hashlib.sha256(destination.read_bytes()).hexdigest()
        if digest != expected_digest:
            msg = f"Downloaded payload SHA-256 mismatch: {digest}"
            raise ValueError(msg)

    benchmark.timed(
        trial, f"copy_to_{label}", lambda: session.copy_to_runtime(str(source), remote), validate_upload, checkpoint
    )
    benchmark.timed(
        trial,
        f"copy_from_{label}",
        lambda: session.copy_from_runtime(remote, str(destination)),
        validate_download,
        checkpoint,
    )


def run_workload_trial(
    name: str,
    kwargs: dict[str, Any],
    settings: benchmark.Settings,
    trial: dict[str, Any],
    *,
    preflight: bool,
    factory: Callable[[str, dict[str, Any]], Any],
    checkpoint: Callable[[], None],
    transfer_iterations: int = 3,
    payload_bytes: int = DEFAULT_PAYLOAD_BYTES,
    records: int = DEFAULT_RECORDS,
) -> None:
    """Run JSON and larger transfers in fresh sessions with verified cleanup."""
    session = None
    trial["resource_scope"] = RESOURCE_SCOPE
    try:
        trial["stage"] = "construct"
        session = benchmark.timed(trial, "construct", lambda: factory(name, kwargs), checkpoint=checkpoint)
        benchmark.image_policy(session, name, preflight=preflight)
        trial["stage"] = "open"
        benchmark.timed(
            trial,
            "open",
            session.open,
            None if preflight else lambda _: benchmark.validate_no_image_preparation(session, name),
            checkpoint,
        )
        trial["stage"] = "first_run"
        benchmark.timed(
            trial,
            "first_run",
            lambda: session.run(f"print({benchmark.MARKER!r})", timeout=settings.timeout),
            lambda output: benchmark.validate_output(output, benchmark.MARKER),
            checkpoint,
        )
        if preflight:
            trial["stage"] = "environment"
            trial["runtime_environment"] = benchmark.runtime_environment(session, settings.timeout)
            if name in benchmark.CONTAINER_BACKENDS:
                trial["image_id"] = getattr(getattr(session, "docker_image", None), "id", None)
        trial["stage"] = "warmup"
        warmup: dict[str, Any] = {"samples": [], "measured": False}
        trial["workload_warmup"] = warmup
        for _ in range(settings.warmups):
            execute_workload(session, warmup, settings, records=records)
        with tempfile.TemporaryDirectory(prefix="llm-sandbox-experiment-") as directory:
            temporary = Path(directory).resolve()
            transfer_workload(session, warmup, temporary, -1, settings.timeout, payload_bytes=payload_bytes)
            trial["stage"] = "workloads"
            for iteration in range(1 if preflight else settings.iterations):
                trial["iteration"] = iteration
                execute_workload(session, trial, settings, checkpoint=checkpoint, records=records)
            trial["stage"] = "transfers"
            for iteration in range(1 if preflight else transfer_iterations):
                trial["transfer_iteration"] = iteration
                transfer_workload(
                    session, trial, temporary, iteration, settings.timeout, checkpoint, payload_bytes=payload_bytes
                )
        trial["status"] = "passed"
    except Exception as exc:  # noqa: BLE001 - preserve individual failed trials and always clean up.
        trial["status"] = "failed"
        trial["error"] = benchmark.error_record(exc)
    finally:
        if session is not None:
            benchmark.finish_session(session, name, trial, checkpoint)
        checkpoint()
