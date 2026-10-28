"""Export expanded benchmark figures after all timed experiments finish.

Run with a Python environment containing matplotlib. Reads summary.json and
host.json beside this script; writes PNG and SVG into the same directory.
"""

# ruff: noqa: INP001 - this is a standalone result artifact, not a package.

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt

RESULTS = Path(__file__).resolve().parent
BACKENDS = ("docker", "bubblewrap", "nsjail")
LABELS = {"docker": "Docker", "bubblewrap": "Bubblewrap", "nsjail": "NsJail"}
COLORS = {"docker": "#2563EB", "bubblewrap": "#0F766E", "nsjail": "#D97706"}
MARKERS = {"docker": "o", "bubblewrap": "s", "nsjail": "^"}
LEVELS = (1, 4, 16)


def estimate_interval(summary: dict[str, Any], scale: float = 1.0) -> tuple[float, float, float]:
    """Read valid estimates and intervals without substituting absent data."""
    interval = summary.get("ci95")
    if not interval or summary.get("estimate") is None:
        msg = "Every plotted estimate must have a complete 95% interval"
        raise ValueError(msg)
    estimate, lower, upper = (
        float(value) * scale for value in (summary["estimate"], interval["lower"], interval["upper"])
    )
    if not all(math.isfinite(value) for value in (estimate, lower, upper)) or not 0 <= lower <= estimate <= upper:
        msg = f"Invalid plotted estimate/interval: {(estimate, lower, upper)}"
        raise ValueError(msg)
    return estimate, lower, upper


def quota_label() -> str:
    """Report the cgroup CPU quota, rather than host logical CPU count."""
    host = json.loads((RESULTS / "host.json").read_text(encoding="utf-8"))
    quota, period = host["cpu.max"].split()
    return f"Same host · {float(quota) / float(period):g}-CPU quota"


def count_label(counts: list[int], unit: str) -> str:
    """Show accepted independent sample counts, including any missing batches."""
    distinct = sorted(set(counts))
    count = str(distinct[0]) if len(distinct) == 1 else f"{distinct[0]}-{distinct[-1]}"
    return f"n = {count} {unit}"


def style_axis(axis: Any) -> None:
    """Use zero-based linear scales, restrained grids and readable labels."""
    axis.set_facecolor("#F8FAFC")
    axis.grid(axis="y", color="#CBD5E1", linewidth=0.7, alpha=0.65)
    axis.set_axisbelow(True)
    axis.spines[["top", "right"]].set_visible(False)
    for side in ("left", "bottom"):
        axis.spines[side].set_color("#94A3B8")
    axis.tick_params(colors="#334155", labelsize=10)


def save_figure(figure: Any, name: str) -> None:
    """Keep raster and editable vector exports of each figure."""
    for suffix in ("png", "svg"):
        figure.savefig(RESULTS / f"{name}.{suffix}", dpi=180, facecolor="white")
    plt.close(figure)


def plot_latency(profiles: dict[str, Any]) -> None:
    """Compare API latency in separate workload panels on a common linear axis."""
    selections = [
        ("baseline", "warm_run", "Warm marker"),
        ("workload", "json_run", "JSON processing · 10,000 records"),
    ]
    data = [
        [profiles[profile]["backends"][name]["latency_seconds"][metric] for name in BACKENDS]
        for profile, metric, _ in selections
    ]
    values = [[estimate_interval(summary, 1000.0) for summary in panel] for panel in data]
    upper_limit = max(upper for panel in values for _, _, upper in panel) * 1.25
    figure, axes = plt.subplots(1, 2, figsize=(12.4, 5.7), sharey=True)
    figure.subplots_adjust(left=0.075, right=0.98, bottom=0.24, top=0.75, wspace=0.13)
    figure.suptitle("Sequential API latency", y=0.96, fontsize=19, fontweight="bold", color="#0F172A")
    figure.text(
        0.5,
        0.885,
        f"Median of session medians · pointwise 95% bootstrap intervals · {quota_label()}",
        ha="center",
        fontsize=10,
        color="#475569",
    )
    for axis, selection, panel_data, panel_values in zip(axes, selections, data, values, strict=True):
        style_axis(axis)
        for index, (name, value) in enumerate(zip(BACKENDS, panel_values, strict=True)):
            estimate, lower, upper = value
            axis.bar(index, estimate, width=0.55, color=COLORS[name], alpha=0.9)
            axis.errorbar(
                index,
                estimate,
                yerr=[[estimate - lower], [upper - estimate]],
                fmt="none",
                ecolor="#0F172A",
                capsize=5,
                elinewidth=1.4,
            )
            axis.text(index, upper + upper_limit * 0.035, f"{estimate:.1f}", ha="center", fontsize=11, color="#0F172A")
        counts = count_label([summary["n_units"] for summary in panel_data], "fresh sessions / backend")
        axis.set_title(f"{selection[2]}\n{counts}", fontsize=11, pad=12, color="#0F172A")
        axis.set_xticks(range(len(BACKENDS)), [LABELS[name] for name in BACKENDS])
        axis.set_ylim(0, upper_limit)
    axes[0].set_ylabel("API latency (ms) · lower is faster", fontsize=11, color="#334155")
    figure.text(
        0.075,
        0.12,
        "Cached runtimes; environment setup skipped. API timers exclude correctness validation.",
        fontsize=10,
        color="#475569",
    )
    figure.text(
        0.075,
        0.075,
        "JSON is a synthetic generate/serialize/parse/filter/group task. Intervals describe this host and run only.",
        fontsize=9,
        color="#64748B",
    )
    save_figure(figure, "latency-comparison")


def plot_throughput(profile: dict[str, Any]) -> None:
    """Compare validated closed-loop throughput with whole-batch intervals."""
    data = {
        (name, workload): {
            entry["workers"]: entry["throughput_validated_calls_per_second"]
            for entry in profile["backends"][name]["concurrency"]
            if entry["workload"] == workload
        }
        for name in BACKENDS
        for workload in ("marker", "json")
    }
    intervals = {key: [estimate_interval(by_level[level]) for level in LEVELS] for key, by_level in data.items()}
    counts = [by_level[level]["n_units"] for by_level in data.values() for level in LEVELS]
    sampling = count_label(counts, "batches per backend / worker count / workload")
    upper_limit = max(upper for panel in intervals.values() for _, _, upper in panel) * 1.16
    figure, axes = plt.subplots(1, 2, figsize=(12.4, 6.1), sharey=True)
    figure.subplots_adjust(left=0.075, right=0.98, bottom=0.25, top=0.69, wspace=0.13)
    figure.suptitle("Closed-loop concurrency throughput", y=0.965, fontsize=19, fontweight="bold", color="#0F172A")
    figure.text(
        0.5,
        0.895,
        f"Exploratory · {sampling} · {quota_label()}",
        ha="center",
        fontsize=10,
        color="#475569",
    )
    for axis, workload, title in zip(axes, ("marker", "json"), ("Instrumented marker", "JSON processing"), strict=True):
        style_axis(axis)
        for name in BACKENDS:
            values = intervals[(name, workload)]
            axis.errorbar(
                LEVELS,
                [estimate for estimate, _, _ in values],
                yerr=[
                    [estimate - lower for estimate, lower, _ in values],
                    [upper - estimate for estimate, _, upper in values],
                ],
                color=COLORS[name],
                marker=MARKERS[name],
                markersize=6,
                linewidth=1.8,
                capsize=4,
                elinewidth=1.3,
                label=LABELS[name],
            )
        axis.set_title(title, fontsize=12, pad=12, color="#0F172A")
        axis.set_xticks(LEVELS)
        axis.set_xlim(0, 17)
        axis.set_ylim(0, upper_limit)
        axis.set_xlabel("Concurrent workers (linear scale)", fontsize=11, labelpad=9, color="#334155")
    axes[0].set_ylabel("Validated calls / second · higher is faster", fontsize=11, color="#334155")
    handles, labels = axes[0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.845), ncol=3, frameon=False, fontsize=11)
    figure.text(
        0.075,
        0.13,
        "Median batch throughput with pointwise 95% bootstrap intervals; "
        "five batches provide limited uncertainty evidence.",
        fontsize=9.5,
        color="#475569",
    )
    figure.text(
        0.075,
        0.085,
        "Shared execution window includes validation; initialization and teardown excluded. "
        "Ten calls per fresh worker.",
        fontsize=9.5,
        color="#475569",
    )
    figure.text(
        0.075,
        0.04,
        "This tests worker-count scaling, not a fixed arrival rate. "
        "The instrumented marker differs from the baseline marker.",
        fontsize=9,
        color="#64748B",
    )
    save_figure(figure, "concurrency-throughput")


def main() -> None:
    """Read complete profile summaries and export both figures without recomputing statistics."""
    plt.rcParams.update({"font.family": "DejaVu Sans", "svg.fonttype": "none", "axes.unicode_minus": False})
    document = json.loads((RESULTS / "summary.json").read_text(encoding="utf-8"))
    profiles = {}
    for experiment in document["experiments"]:
        if experiment["profile"] in profiles:
            msg = f"Duplicate profile in summary: {experiment['profile']}"
            raise ValueError(msg)
        profiles[experiment["profile"]] = experiment
    plot_latency(profiles)
    plot_throughput(profiles["concurrency"])


if __name__ == "__main__":
    main()
