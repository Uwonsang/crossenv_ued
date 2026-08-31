"""Analyze 300M-step gradient diagnostics by algorithm and NUM_ENVS.

The script downloads four W&B metric families (policy/value, environment
cosine, effective rank, and SNR), saves the sampled history as CSV, and
summarizes the final window against NUM_ENVS. A run must reach
``--min-summary-progress`` before it contributes to the final-window summary,
so an incomplete run is retained in the run metadata without being treated as
a 300M-step result.
"""
from __future__ import annotations

import argparse
import csv
import math
import re
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import wandb


DEFAULT_ENTITY = "overcooked_ai"
DEFAULT_PROJECT = "cec_stiffness_100m"
DEFAULT_TARGET_TIMESTEPS = 300_000_000
DEFAULT_OUTPUT_DIR = (
    Path(__file__).parent.parent
    / "stiffness_results"
    / "gradient_diagnostics_num_envs"
)

METRIC_GROUPS = {
    "policy_value": (
        "policy_value/shared_cosine",
        "policy_value/shared_conflict_rate",
        "policy_value/policy_grad_norm",
        "policy_value/weighted_value_grad_norm",
    ),
    "env_gradient_snr": (
        "env_gradient_snr/policy_snr",
        "env_gradient_snr/value_snr",
        "env_gradient_snr/policy_log_snr",
        "env_gradient_snr/value_log_snr",
    ),
    "env_gradient_gsnr": (
        "env_gradient_gsnr/policy_parameterwise_gsnr_mean",
        "env_gradient_gsnr/value_parameterwise_gsnr_mean",
        "env_gradient_gsnr/policy_parameterwise_gsnr_mean_log10",
        "env_gradient_gsnr/value_parameterwise_gsnr_mean_log10",
    ),
    "env_gradient_norm": tuple(
        f"env_gradient_norm/{gradient_name}_{statistic}"
        for gradient_name in (
            "policy_norm",
            "weighted_value_norm",
        )
        for statistic in ("mean", "cv", "p10", "p90", "iqm")
    ),
    **{
        f"feature_rank_{role}": (
            f"feature_rank_{role}/effective_rank",
            f"feature_rank_{role}/between_slot_effective_rank",
        ) + tuple(
            f"feature_rank_{role}/within_slot_{statistic}"
            for statistic in ("mean", "cv", "p10", "p90", "iqm")
        )
        for role in ("shared", "policy", "value")
    },
    "env_gradient_rank": (
        "env_gradient_rank/policy_effective_rank",
        "env_gradient_rank/value_effective_rank",
        "env_gradient_rank/policy_effective_rank_ratio",
        "env_gradient_rank/value_effective_rank_ratio",
    ),
    "env_gradient_cosine": (
        "env_gradient_cosine/policy_policy_mean_cosine",
        "env_gradient_cosine/value_value_mean_cosine",
        "env_gradient_cosine/policy_value_same_env_cosine",
        "env_gradient_cosine/policy_value_cross_env_cosine",
    ),
}
ALL_METRICS = tuple(
    metric for metrics in METRIC_GROUPS.values() for metric in metrics
)
ALGORITHM_COLORS = {"CEC": "#4c9a3a", "CEC_IDAAC": "#377eb8"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--entity", default=DEFAULT_ENTITY)
    parser.add_argument("--project", default=DEFAULT_PROJECT)
    parser.add_argument(
        "--target-timesteps", type=int, default=DEFAULT_TARGET_TIMESTEPS
    )
    parser.add_argument(
        "--tail-window-steps",
        type=int,
        default=30_000_000,
        help="Final-window width used by NUM_ENVS summary plots.",
    )
    parser.add_argument(
        "--min-summary-progress",
        type=float,
        default=0.95,
        help="Minimum target-step fraction required for final-window summary.",
    )
    parser.add_argument(
        "--states",
        nargs="*",
        default=("finished", "crashed", "running"),
        help="W&B run states included while fetching histories.",
    )
    parser.add_argument("--num-envs", nargs="*", type=int, default=None)
    parser.add_argument(
        "--algorithms", nargs="*", default=("CEC", "CEC_IDAAC")
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--history-samples",
        type=int,
        default=10_000,
        help="Maximum W&B history rows fetched per run.",
    )
    return parser.parse_args()


def canonical_algorithm(run) -> str:
    candidates = (
        run.config.get("model_name"),
        run.config.get("MODEL_NAME"),
        run.config.get("algorithm"),
        run.name,
    )
    text = " ".join(str(value) for value in candidates if value).upper()
    if "IDAAC" in text or "IDDAC" in text:
        return "CEC_IDAAC"
    if "CEC" in text or "IPPO" in text:
        return "CEC"
    name = str(run.name).split("_")[0]
    return re.sub(r"[^A-Za-z0-9_-]+", "_", name) or "unknown"


def finite_float(value):
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def metric_group(metric: str) -> str:
    for group, metrics in METRIC_GROUPS.items():
        if metric in metrics:
            return group
    raise KeyError(metric)


def fetch_history(args: argparse.Namespace):
    api = wandb.Api(timeout=90)
    runs = list(api.runs(f"{args.entity}/{args.project}"))
    history_rows = []
    run_rows = []

    for run in runs:
        total_timesteps = run.config.get(
            "TOTAL_TIMESTEPS", run.config.get("total_timesteps")
        )
        try:
            total_timesteps = int(total_timesteps)
        except (TypeError, ValueError):
            continue
        if total_timesteps != args.target_timesteps or run.state not in args.states:
            continue

        algorithm = canonical_algorithm(run)
        if args.algorithms and algorithm not in args.algorithms:
            continue
        num_envs = run.config.get("NUM_ENVS", run.config.get("num_envs"))
        seed = run.config.get("SEED", run.config.get("seed"))
        try:
            num_envs, seed = int(num_envs), int(seed)
        except (TypeError, ValueError):
            print(f"Skipping run with invalid NUM_ENVS/SEED: {run.id}")
            continue
        if args.num_envs and num_envs not in args.num_envs:
            continue

        print(
            f"Fetching {run.id}: {algorithm}, NUM_ENVS={num_envs}, "
            f"seed={seed}, state={run.state}"
        )
        available_metrics = [
            metric for metric in ALL_METRICS if metric in run.summary
        ]
        max_env_step = 0
        metric_counts = defaultdict(int)
        history = run.history(
            keys=["env_step", "_step", *available_metrics],
            samples=args.history_samples,
            pandas=False,
        )
        for row in history:
            env_step = finite_float(row.get("env_step", row.get("_step")))
            if env_step is None or env_step > args.target_timesteps:
                continue
            max_env_step = max(max_env_step, int(env_step))
            for metric in available_metrics:
                value = finite_float(row.get(metric))
                if value is None:
                    continue
                metric_counts[metric] += 1
                history_rows.append(
                    {
                        "run_id": run.id,
                        "run_name": run.name,
                        "run_state": run.state,
                        "algorithm": algorithm,
                        "num_envs": num_envs,
                        "seed": seed,
                        "env_step": int(env_step),
                        "metric_group": metric_group(metric),
                        "metric": metric,
                        "value": value,
                    }
                )

        progress = max_env_step / args.target_timesteps
        run_rows.append(
            {
                "run_id": run.id,
                "run_name": run.name,
                "run_state": run.state,
                "algorithm": algorithm,
                "num_envs": num_envs,
                "seed": seed,
                "configured_total_timesteps": total_timesteps,
                "max_logged_env_step": max_env_step,
                "progress": progress,
                "summary_eligible": progress >= args.min_summary_progress,
                "metrics_found": len(metric_counts),
            }
        )
    return history_rows, run_rows


def write_csv(path: Path, rows):
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def build_window_summary(args, history_rows, run_rows, window_start):
    eligible = {
        row["run_id"] for row in run_rows if row["summary_eligible"]
    }
    samples = defaultdict(list)
    metadata = {}
    for row in history_rows:
        if (
            row["run_id"] not in eligible
            or row["env_step"] < window_start
            or row["env_step"] > args.target_timesteps
        ):
            continue
        key = (row["run_id"], row["metric"])
        samples[key].append(row["value"])
        metadata[key] = row

    per_run_rows = []
    for key, values in sorted(samples.items()):
        meta = metadata[key]
        per_run_rows.append(
            {
                "run_id": key[0],
                "algorithm": meta["algorithm"],
                "num_envs": meta["num_envs"],
                "seed": meta["seed"],
                "metric_group": meta["metric_group"],
                "metric": key[1],
                "window_start": window_start,
                "window_end": args.target_timesteps,
                "mean": float(np.mean(values)),
                "std_over_time": (
                    float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
                ),
                "num_measurements": len(values),
            }
        )

    grouped = defaultdict(list)
    for row in per_run_rows:
        key = (row["algorithm"], row["num_envs"], row["metric"])
        grouped[key].append(row["mean"])

    aggregate_rows = []
    for (algorithm, num_envs, metric), values in sorted(grouped.items()):
        aggregate_rows.append(
            {
                "algorithm": algorithm,
                "num_envs": num_envs,
                "metric_group": metric_group(metric),
                "metric": metric,
                "mean": float(np.mean(values)),
                "std_across_runs": (
                    float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
                ),
                "num_runs": len(values),
            }
        )
    return per_run_rows, aggregate_rows


def short_metric_name(metric: str) -> str:
    return metric.split("/", 1)[1].replace("_", " ")


def subplot_grid(num_panels: int):
    cols = min(2, num_panels)
    rows = int(math.ceil(num_panels / cols))
    return rows, cols


def plot_summary(args, aggregate_rows, window_start, output_suffix):
    for group, configured_metrics in METRIC_GROUPS.items():
        metrics = [
            metric for metric in configured_metrics
            if any(row["metric"] == metric for row in aggregate_rows)
        ]
        if not metrics:
            continue
        rows, cols = subplot_grid(len(metrics))
        fig, axes = plt.subplots(rows, cols, figsize=(6.2 * cols, 3.8 * rows))
        axes = np.asarray(axes, dtype=object).reshape(-1)
        for ax, metric in zip(axes, metrics):
            for algorithm in args.algorithms:
                points = [
                    row for row in aggregate_rows
                    if row["algorithm"] == algorithm and row["metric"] == metric
                ]
                if not points:
                    continue
                points.sort(key=lambda row: row["num_envs"])
                x = np.asarray([row["num_envs"] for row in points])
                y = np.asarray([row["mean"] for row in points])
                yerr = np.asarray([row["std_across_runs"] for row in points])
                color = ALGORITHM_COLORS.get(algorithm)
                ax.plot(x, y, marker="o", linewidth=1.8, color=color, label=algorithm)
                if np.any(yerr > 0):
                    ax.fill_between(x, y - yerr, y + yerr, color=color, alpha=0.14)
            ax.set_title(short_metric_name(metric))
            ax.set_xlabel("NUM_ENVS")
            ax.set_xticks(sorted({row["num_envs"] for row in aggregate_rows}))
            ax.grid(alpha=0.3)
        for ax in axes[len(metrics):]:
            ax.set_visible(False)
        legend_entries = {}
        for ax in axes[:len(metrics)]:
            handles, labels = ax.get_legend_handles_labels()
            for handle, label in zip(handles, labels):
                legend_entries.setdefault(label, handle)
        if legend_entries:
            fig.legend(
                legend_entries.values(),
                legend_entries.keys(),
                loc="upper center",
                bbox_to_anchor=(0.5, 0.955),
                ncol=len(legend_entries),
            )
        fig.suptitle(
            f"{group}: mean over "
            f"{window_start / 1e6:.0f}M–{args.target_timesteps / 1e6:.0f}M",
            fontsize=14,
            y=0.995,
        )
        fig.tight_layout(rect=(0, 0, 1, 0.91))
        output = args.output_dir / (
            f"{group}_vs_num_envs_{output_suffix}.png"
        )
        fig.savefig(output, dpi=200, bbox_inches="tight")
        plt.close(fig)


def main():
    args = parse_args()
    if not 0 < args.min_summary_progress <= 1:
        raise ValueError("--min-summary-progress must be in (0, 1].")
    if args.tail_window_steps <= 0 or args.tail_window_steps > args.target_timesteps:
        raise ValueError("--tail-window-steps must be in (0, target_timesteps].")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    history_rows, run_rows = fetch_history(args)
    if not history_rows:
        raise RuntimeError("No matching W&B gradient-diagnostic history found.")
    final_window_start = args.target_timesteps - args.tail_window_steps
    final_run_rows, final_aggregate_rows = build_window_summary(
        args, history_rows, run_rows, final_window_start
    )
    full_run_rows, full_aggregate_rows = build_window_summary(
        args, history_rows, run_rows, 0
    )

    write_csv(args.output_dir / "gradient_diagnostics_history.csv", history_rows)
    write_csv(args.output_dir / "gradient_diagnostics_runs.csv", run_rows)
    write_csv(
        args.output_dir / "gradient_diagnostics_final_window_runs.csv",
        final_run_rows,
    )
    write_csv(
        args.output_dir / "gradient_diagnostics_final_window_aggregate.csv",
        final_aggregate_rows,
    )
    write_csv(
        args.output_dir / "gradient_diagnostics_full_window_runs.csv",
        full_run_rows,
    )
    write_csv(
        args.output_dir / "gradient_diagnostics_full_window_aggregate.csv",
        full_aggregate_rows,
    )
    plot_summary(
        args, final_aggregate_rows, final_window_start, "final_window"
    )
    plot_summary(args, full_aggregate_rows, 0, "full_window")

    print(f"Runs fetched: {len(run_rows)}")
    for row in sorted(run_rows, key=lambda item: (item["algorithm"], item["num_envs"])):
        print(
            f"  {row['algorithm']:10s} NUM_ENVS={row['num_envs']:3d} "
            f"state={row['run_state']:8s} progress={100 * row['progress']:6.2f}% "
            f"summary={'yes' if row['summary_eligible'] else 'no'}"
        )
    print(f"Saved outputs to: {args.output_dir}")


if __name__ == "__main__":
    main()
