"""Plot CEC train/evaluation TD-error learning curves from cached CSV data."""
from __future__ import annotations

import argparse
import math
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

try:
    from .td_error_generalization_graph import (
        LAYOUTS,
        LAYOUT_LABELS,
        PANELS,
        read_history_csv,
    )
except ImportError:
    from td_error_generalization_graph import (
        LAYOUTS,
        LAYOUT_LABELS,
        PANELS,
        read_history_csv,
    )


REPOSITORY_ROOT = Path(__file__).resolve().parents[4]
DEFAULT_INPUT_CSV = (
    REPOSITORY_ROOT / "artifacts" / "td_error_generalization"
    / "td_error_history_3000m.csv"
)
DEFAULT_OUTPUT_DIR = REPOSITORY_ROOT / "artifacts" / "td_error_learning_curves"
COLORS = {"train": "#377eb8", "evaluation": "#e68632"}
LINESTYLES = {"train": "-", "evaluation": "--"}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-csv", type=Path, default=DEFAULT_INPUT_CSV)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--model-name", default="CEC")
    parser.add_argument(
        "--num-envs", nargs="+", type=int, default=[32, 64, 128, 256],
    )
    parser.add_argument("--num-minibatches", type=int, default=2)
    parser.add_argument("--seeds", nargs="*", type=int, default=None)
    parser.add_argument("--target-env-steps", type=int, default=3_000_000_000)
    parser.add_argument(
        "--smooth-window", type=int, default=1,
        help="Optional rolling window applied after seed aggregation.",
    )
    return parser.parse_args()


def filtered_rows(args):
    if not args.input_csv.exists():
        raise FileNotFoundError(
            f"TD-error history CSV not found: {args.input_csv}. "
            "Run td_error_generalization_graph.py once to create it."
        )
    rows = read_history_csv(args.input_csv)
    selected = [
        row for row in rows
        if row["model"] == args.model_name
        and row["num_envs"] in args.num_envs
        and row["num_minibatches"] == args.num_minibatches
        and row["env_step"] <= args.target_env_steps
        and (args.seeds is None or row["seed"] in args.seeds)
    ]
    if not selected:
        raise RuntimeError("No rows in the CSV match the requested configuration.")
    return selected


def add_macro_mean(rows):
    """Add an equal-layout mean TD-error at each run and logging step."""
    grouped = defaultdict(list)
    for row in rows:
        grouped[(row["run_id"], row["env_step"])].append(row)

    expanded = list(rows)
    for group in grouped.values():
        first = group[0]
        by_layout = {row["eval_layout"]: row for row in group}
        if not all(layout in by_layout for layout in LAYOUTS):
            continue
        mean_row = dict(first)
        mean_row["eval_layout"] = "mean"
        for key in ("train_td_error_rmse", "eval_td_error_rmse"):
            values = [by_layout[layout][key] for layout in LAYOUTS]
            finite = [value for value in values if value is not None]
            mean_row[key] = float(np.mean(finite)) if len(finite) == len(LAYOUTS) else None
        if (
            mean_row["train_td_error_rmse"] is not None
            or mean_row["eval_td_error_rmse"] is not None
        ):
            expanded.append(mean_row)
    return expanded


def aggregate_seeds(rows):
    grouped = defaultdict(list)
    for row in rows:
        for split, key in (
            ("train", "train_td_error_rmse"),
            ("evaluation", "eval_td_error_rmse"),
        ):
            value = row[key]
            if value is None:
                continue
            group_key = (
                row["num_envs"], row["eval_layout"], split, row["env_step"],
            )
            grouped[group_key].append(value)

    aggregated = []
    for (num_envs, layout, split, env_step), values in sorted(grouped.items()):
        values = np.asarray(values, dtype=float)
        std = float(values.std(ddof=1)) if len(values) > 1 else 0.0
        aggregated.append(
            {
                "num_envs": num_envs,
                "eval_layout": layout,
                "split": split,
                "env_step": env_step,
                "mean": float(values.mean()),
                "sem": std / math.sqrt(len(values)),
                "num_seeds": len(values),
            }
        )
    return aggregated


def rolling_mean(values, window):
    if window <= 1:
        return values
    result = np.empty_like(values, dtype=float)
    for index in range(len(values)):
        start = max(0, index - window + 1)
        result[index] = values[start:index + 1].mean()
    return result


def plot_num_envs(rows, args, num_envs, output_path):
    fig, axes = plt.subplots(2, 3, figsize=(11.2, 7.0), sharex=True)
    axes = axes.ravel()

    for ax, layout in zip(axes, PANELS):
        for split in ("train", "evaluation"):
            points = sorted(
                (
                    row for row in rows
                    if row["num_envs"] == num_envs
                    and row["eval_layout"] == layout
                    and row["split"] == split
                ),
                key=lambda row: row["env_step"],
            )
            if not points:
                continue
            x = np.asarray([row["env_step"] for row in points]) / 1e9
            mean = np.asarray([row["mean"] for row in points])
            sem = np.asarray([row["sem"] for row in points])
            mean = rolling_mean(mean, args.smooth_window)
            sem = rolling_mean(sem, args.smooth_window)
            ax.plot(
                x, mean, LINESTYLES[split], color=COLORS[split],
                linewidth=1.7,
            )
            ax.fill_between(
                x, mean - sem, mean + sem, color=COLORS[split], alpha=0.13,
            )
        ax.set_title(LAYOUT_LABELS[layout], fontsize=12)
        ax.grid(alpha=0.25)

    for index, ax in enumerate(axes):
        if index // 3 == 1:
            ax.set_xlabel("Environment steps (billions)")
        if index % 3 == 0:
            ax.set_ylabel("TD-error RMSE")

    handles = [
        Line2D(
            [0], [0], color=COLORS[split], linestyle=LINESTYLES[split],
            linewidth=1.8, label=split.capitalize(),
        )
        for split in ("train", "evaluation")
    ]
    fig.legend(
        handles=handles, loc="upper center", ncol=2,
        bbox_to_anchor=(0.5, 0.90), frameon=True,
    )
    fig.suptitle(
        f"{args.model_name.replace('_', '-')}: Train vs Evaluation TD-error Learning Curves\n"
        f"NUM_ENVS={num_envs}, NUM_MINIBATCHES={args.num_minibatches}",
        fontsize=14,
    )
    fig.tight_layout(rect=(0.02, 0.01, 0.99, 0.81))
    fig.savefig(output_path)
    plt.close(fig)


def main():
    args = parse_args()
    args.model_name = args.model_name.upper().replace("-", "_")
    if args.target_env_steps <= 0:
        raise ValueError("--target-env-steps must be positive.")
    if args.num_minibatches <= 0 or args.smooth_window <= 0:
        raise ValueError("minibatches and smooth window must be positive.")

    rows = add_macro_mean(filtered_rows(args))
    aggregated = aggregate_seeds(rows)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    horizon = f"{args.target_env_steps // 1_000_000}m"
    model = args.model_name.lower()
    for num_envs in args.num_envs:
        output_path = args.output_dir / (
            f"td_error_learning_curves_{model}_envs{num_envs}_{horizon}.pdf"
        )
        plot_num_envs(aggregated, args, num_envs, output_path)
        print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()
