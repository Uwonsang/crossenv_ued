"""Plot mean value-network stiffness against the number of environments.

Each W&B run is first averaged over all of its logged measurements.  Runs with
the same algorithm and NUM_ENVS (for example, different seeds) are then given
equal weight when computing the plotted point.
"""
from __future__ import annotations

import argparse
import csv
import math
import re
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import wandb


DEFAULT_ENTITY = "overcooked_ai"
DEFAULT_PROJECT = "cec_stiffness_100m"
DEFAULT_TARGET_TIMESTEPS = 300_000_000
DEFAULT_OUTPUT_DIR = Path(__file__).parent.parent / "stiffness_results" / "stiffness_num_envs"

METRICS = {
    "same_layout": "stiffness/paper_value_same_layout",
    "different_layout": "stiffness/paper_value_different_layout",
    "off_diagonal": "stiffness/paper_value_off_diagonal",
    "layout_gap": "stiffness/paper_value_layout_gap",
}
PANEL_TITLES = {
    "same_layout": "Same-layout",
    "different_layout": "Different-layout",
    "off_diagonal": "Off-diagonal",
    "layout_gap": "Layout gap",
}
COLORS = {"CEC": "#4c9a3a", "CEC_IDAAC": "#377eb8"}
MARKERS = {"CEC": "o", "CEC_IDAAC": "o"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--entity", default=DEFAULT_ENTITY)
    parser.add_argument("--project", default=DEFAULT_PROJECT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--target-timesteps", type=int, default=DEFAULT_TARGET_TIMESTEPS
    )
    parser.add_argument(
        "--min-progress",
        type=float,
        default=0.95,
        help="Minimum target-step fraction required for inclusion.",
    )
    parser.add_argument(
        "--algorithms",
        nargs="*",
        default=None,
        help="Only plot these canonical labels (CEC and/or CEC_IDAAC).",
    )
    parser.add_argument(
        "--num-envs",
        nargs="*",
        type=int,
        default=None,
        help="Only include these NUM_ENVS values.",
    )
    parser.add_argument(
        "--include-running",
        action="store_true",
        help="Include runs whose W&B state is running (disabled by default).",
    )
    parser.add_argument(
        "--history-samples",
        type=int,
        default=10_000,
        help="Maximum history rows fetched per run (default: 10000).",
    )
    return parser.parse_args()


def canonical_algorithm(run) -> str:
    candidates = [
        run.config.get("model_name"),
        run.config.get("MODEL_NAME"),
        run.config.get("algorithm"),
        run.name,
    ]
    text = " ".join(str(value) for value in candidates if value).upper()
    if "IDAAC" in text or "IDDAC" in text:
        return "CEC_IDAAC"
    if "CEC" in text or "IPPO" in text:
        return "CEC"
    # Keep an unknown algorithm visible instead of silently discarding data.
    name = str(run.name).split("_")[0]
    return re.sub(r"[^A-Za-z0-9_-]+", "_", name) or "unknown"


def finite_float(value):
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def fetch_run_means(args: argparse.Namespace):
    api = wandb.Api()
    runs = list(api.runs(f"{args.entity}/{args.project}"))
    rows = []
    keys = list(METRICS.values())

    for run in runs:
        if not args.include_running and run.state == "running":
            print(f"Skipping running run: {run.id} ({run.name})")
            continue
        total_timesteps = run.config.get(
            "TOTAL_TIMESTEPS", run.config.get("total_timesteps")
        )
        try:
            total_timesteps = int(total_timesteps)
        except (TypeError, ValueError):
            continue
        if total_timesteps != args.target_timesteps:
            continue
        summary_env_step = finite_float(run.summary.get("env_step"))
        if summary_env_step is None:
            summary_env_step = finite_float(run.summary.get("_step"))
        progress = (
            summary_env_step / args.target_timesteps
            if summary_env_step is not None else 0.0
        )
        if progress < args.min_progress:
            print(
                f"Skipping incomplete run: {run.id} ({run.name}), "
                f"progress={100 * progress:.2f}%"
            )
            continue
        algorithm = canonical_algorithm(run)
        if args.algorithms and algorithm not in args.algorithms:
            continue
        num_envs = run.config.get("NUM_ENVS", run.config.get("num_envs"))
        try:
            num_envs = int(num_envs)
        except (TypeError, ValueError):
            print(f"Skipping run without integer NUM_ENVS: {run.id} ({run.name})")
            continue
        if args.num_envs and num_envs not in args.num_envs:
            continue

        values = {metric: [] for metric in METRICS}
        history = run.history(
            keys=["env_step", "_step", *keys],
            samples=args.history_samples,
            pandas=False,
        )
        for history_row in history:
            env_step = finite_float(
                history_row.get("env_step", history_row.get("_step"))
            )
            if env_step is not None and env_step > args.target_timesteps:
                continue
            for metric, wandb_key in METRICS.items():
                value = finite_float(history_row.get(wandb_key))
                if value is not None:
                    values[metric].append(value)

        for metric, samples in values.items():
            if samples:
                rows.append(
                    {
                        "run_id": run.id,
                        "run_name": run.name,
                        "run_state": run.state,
                        "algorithm": algorithm,
                        "num_envs": num_envs,
                        "configured_total_timesteps": total_timesteps,
                        "max_logged_env_step": int(summary_env_step),
                        "progress": progress,
                        "metric": metric,
                        "history_mean": float(np.mean(samples)),
                        "history_std": float(np.std(samples, ddof=1)) if len(samples) > 1 else 0.0,
                        "history_count": len(samples),
                    }
                )
        if not any(values.values()):
            print(f"No stiffness history found: {run.id} ({run.name})")

    return rows


def aggregate(run_rows):
    grouped = defaultdict(list)
    for row in run_rows:
        grouped[(row["algorithm"], row["num_envs"], row["metric"])].append(
            row["history_mean"]
        )

    rows = []
    for (algorithm, num_envs, metric), run_means in sorted(grouped.items()):
        rows.append(
            {
                "algorithm": algorithm,
                "num_envs": num_envs,
                "metric": metric,
                "mean": float(np.mean(run_means)),
                "std_across_runs": (
                    float(np.std(run_means, ddof=1)) if len(run_means) > 1 else 0.0
                ),
                "num_runs": len(run_means),
            }
        )
    return rows


def write_csv(path: Path, rows, fieldnames):
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def plot(rows, output_path: Path, target_timesteps: int):
    algorithms = sorted({row["algorithm"] for row in rows})
    num_envs_values = sorted({row["num_envs"] for row in rows})
    lookup = {(r["metric"], r["algorithm"], r["num_envs"]): r for r in rows}

    fig, axes = plt.subplots(1, len(METRICS), figsize=(13.2, 3.55), sharex=True)
    for ax, metric in zip(axes, METRICS):
        for index, algorithm in enumerate(algorithms):
            points = [lookup.get((metric, algorithm, num_envs)) for num_envs in num_envs_values]
            if not any(point is not None for point in points):
                continue
            x = num_envs_values
            y = np.asarray(
                [point["mean"] if point is not None else np.nan for point in points]
            )
            yerr = np.asarray(
                [
                    point["std_across_runs"] if point is not None else np.nan
                    for point in points
                ]
            )
            color = COLORS.get(algorithm, plt.get_cmap("tab10")(index))
            ax.plot(
                x,
                y,
                color=color,
                marker=MARKERS.get(algorithm, "o"),
                linewidth=1.8,
                markersize=5,
                label=algorithm,
            )
            if np.any(np.isfinite(yerr) & (yerr > 0)):
                ax.fill_between(
                    x,
                    y - yerr,
                    y + yerr,
                    color=color,
                    alpha=0.16,
                    linewidth=0,
                )
        ax.set_title(PANEL_TITLES[metric], fontsize=13)
        ax.set_xlabel("# environments")
        ax.set_xticks(num_envs_values)
        ax.grid(alpha=0.35)
    axes[0].set_ylabel("Mean stiffness")

    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(
            handles,
            labels,
            loc="upper center",
            bbox_to_anchor=(0.5, 1.06),
            ncol=len(labels),
            frameon=True,
        )
    fig.suptitle(
        "Value-network stiffness vs. number of environments "
        f"(0M–{target_timesteps / 1e6:.0f}M mean)",
        y=1.14,
        fontsize=14,
    )
    fig.subplots_adjust(left=0.065, right=0.99, bottom=0.20, top=0.78, wspace=0.34)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main():
    args = parse_args()
    if not 0 < args.min_progress <= 1:
        raise ValueError("--min-progress must be in (0, 1].")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    run_rows = fetch_run_means(args)
    if not run_rows:
        raise RuntimeError("No completed W&B runs with stiffness history were found.")
    aggregate_rows = aggregate(run_rows)

    suffix = f"{args.target_timesteps // 1_000_000}m"
    run_csv = args.output_dir / f"stiffness_run_means_{suffix}.csv"
    aggregate_csv = args.output_dir / f"stiffness_num_envs_aggregate_{suffix}.csv"
    figure_path = args.output_dir / f"stiffness_vs_num_envs_{suffix}.png"
    write_csv(run_csv, run_rows, list(run_rows[0]))
    write_csv(aggregate_csv, aggregate_rows, list(aggregate_rows[0]))
    plot(aggregate_rows, figure_path, args.target_timesteps)

    print(f"Runs with data: {len({row['run_id'] for row in run_rows})}")
    print(f"Saved: {run_csv}")
    print(f"Saved: {aggregate_csv}")
    print(f"Saved: {figure_path}")


if __name__ == "__main__":
    main()
