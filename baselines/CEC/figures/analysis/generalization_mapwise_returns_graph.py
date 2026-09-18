"""Plot map-wise train/evaluation returns from an existing CSV cache."""
from __future__ import annotations

import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

plt.rcParams.update({
    "font.size": 14,
    "axes.labelsize": 15,
    "xtick.labelsize": 13,
    "ytick.labelsize": 13,
    "legend.fontsize": 13,
})


REPOSITORY_ROOT = Path(__file__).resolve().parents[4]
DEFAULT_DATA_DIR = REPOSITORY_ROOT / "artifacts" / "generalization_gap"
LAYOUTS = [
    "cramped_room_9",
    "asymm_advantages_9",
    "coord_ring_9",
    "counter_circuit_9",
    "forced_coord_9",
]
LAYOUT_LABELS = {
    "cramped_room_9": "Cramped Room",
    "asymm_advantages_9": "Asymmetric Advantages",
    "coord_ring_9": "Coordination Ring",
    "counter_circuit_9": "Counter Circuit",
    "forced_coord_9": "Forced Coordination",
}
MODEL_LABELS = {"CEC": "CEC", "CEC_IDAAC": "DCEC"}
MODEL_COLORS = {"CEC": "#377eb8", "CEC_IDAAC": "#e68632"}
SPLIT_STYLES = {"train": "-", "evaluation": "--"}
SPLIT_MARKERS = {"train": "o", "evaluation": "s"}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--input-csv", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument(
        "--aggregation", choices=("last", "final-window"), default="last",
    )
    parser.add_argument("--target-env-steps", type=int, default=3_000_000_000)
    parser.add_argument(
        "--model-names", nargs="+", default=["CEC", "CEC_IDAAC"],
    )
    parser.add_argument(
        "--presentation", choices=("separate", "combined", "both"),
        default="separate",
        help="Write one PDF per model, one comparison PDF, or both.",
    )
    parser.add_argument(
        "--num-envs", nargs="+", type=int, default=[32, 64, 128, 256],
    )
    parser.add_argument("--num-minibatches", type=int, default=2)
    parser.add_argument("--seeds", nargs="*", type=int, default=None)
    return parser.parse_args()


def resolve_input_csv(args):
    if args.input_csv is not None:
        return args.input_csv
    horizon = f"{args.target_env_steps // 1_000_000}m"
    if args.aggregation == "last":
        filename = f"generalization_gap_runs_last_{horizon}.csv"
    else:
        # The original final-window cache predates aggregation in its filename.
        filename = f"generalization_gap_runs_{horizon}.csv"
    return args.data_dir / filename


def read_rows(path, args):
    if not path.exists():
        raise FileNotFoundError(f"Generalization-gap CSV not found: {path}")
    with path.open(newline="", encoding="utf-8") as file:
        rows = list(csv.DictReader(file))
    parsed = []
    for row in rows:
        if row["eval_layout"] not in LAYOUTS:
            continue
        model = row["model"].upper().replace("-", "_")
        num_envs = int(float(row["num_envs"]))
        num_minibatches = int(float(row.get("num_minibatches", 0)))
        seed = int(float(row["seed"]))
        if model not in args.model_names or num_envs not in args.num_envs:
            continue
        if num_minibatches != args.num_minibatches:
            continue
        if args.seeds is not None and seed not in args.seeds:
            continue
        parsed.append(
            {
                "run_id": row["run_id"],
                "model": model,
                "num_envs": num_envs,
                "seed": seed,
                "eval_layout": row["eval_layout"],
                "train": float(row["train_return_mean"]),
                "evaluation": float(row["eval_return_mean"]),
            }
        )
    if not parsed:
        raise RuntimeError("No CSV rows match the requested configuration.")
    return parsed


def mean_sem(values):
    values = np.asarray(values, dtype=float)
    sem = (
        float(values.std(ddof=1) / math.sqrt(len(values)))
        if len(values) > 1 else 0.0
    )
    return float(values.mean()), sem, len(values)


def aggregate_layouts(rows):
    grouped = defaultdict(list)
    for row in rows:
        grouped[(row["model"], row["num_envs"], row["eval_layout"])].append(row)
    result = []
    for (model, num_envs, layout), group in sorted(grouped.items()):
        train_mean, train_sem, n = mean_sem([row["train"] for row in group])
        eval_mean, eval_sem, _ = mean_sem([row["evaluation"] for row in group])
        result.append(
            {
                "model": model,
                "num_envs": num_envs,
                "eval_layout": layout,
                "train_mean": train_mean,
                "train_sem": train_sem,
                "evaluation_mean": eval_mean,
                "evaluation_sem": eval_sem,
                "num_seeds": n,
            }
        )
    return result


def plot(layout_rows, args, models, output_path):
    fig = plt.figure(figsize=(15.0, 9.0))
    grid = fig.add_gridspec(2, 6)
    axes = [
        fig.add_subplot(grid[0, 0:2]),
        fig.add_subplot(grid[0, 2:4]),
        fig.add_subplot(grid[0, 4:6]),
        fig.add_subplot(grid[1, 1:3]),
        fig.add_subplot(grid[1, 3:5]),
    ]
    x_lookup = {num_envs: index for index, num_envs in enumerate(args.num_envs)}

    for ax, layout in zip(axes, LAYOUTS):
        for model in models:
            points = sorted(
                (
                    row for row in layout_rows
                    if row["model"] == model and row["eval_layout"] == layout
                ),
                key=lambda row: row["num_envs"],
            )
            for split in ("train", "evaluation"):
                x = np.asarray([x_lookup[row["num_envs"]] for row in points])
                y = np.asarray([row[f"{split}_mean"] for row in points])
                sem = np.asarray([row[f"{split}_sem"] for row in points])
                ax.errorbar(
                    x, y, yerr=sem, color=MODEL_COLORS[model],
                    linestyle=SPLIT_STYLES[split],
                    marker=SPLIT_MARKERS[split], linewidth=1.7,
                    markersize=4.8, capsize=2.5,
                )
        ax.set_title(LAYOUT_LABELS[layout], fontsize=12)
        ax.grid(alpha=0.25)

    for index, ax in enumerate(axes):
        ax.set_xticks(range(len(args.num_envs)))
        ax.set_xticklabels(args.num_envs)
        if index >= 3:
            ax.set_xlabel("Number of parallel training environments")
        if index in (0, 3):
            ax.set_ylabel("Return")

    legend_handles = []
    for model in models:
        for split in ("train", "evaluation"):
            legend_handles.append(
                Line2D(
                    [0], [0], color=MODEL_COLORS[model],
                    linestyle=SPLIT_STYLES[split], marker=SPLIT_MARKERS[split],
                    linewidth=1.7, markersize=4.8,
                    label=f"{MODEL_LABELS.get(model, model)} {split}",
                )
            )
    fig.legend(
        handles=legend_handles, loc="upper center", ncol=4,
        bbox_to_anchor=(0.5, 0.90), frameon=True,
    )
    # Figure-level titles and legends make tight_layout reserve excessive
    # vertical space. Place the panel grid explicitly for a compact layout.
    fig.subplots_adjust(
        left=0.075, right=0.985, bottom=0.095, top=0.84,
        wspace=0.42, hspace=0.38,
    )
    fig.savefig(output_path)
    plt.close(fig)


def main():
    args = parse_args()
    args.model_names = [name.upper().replace("-", "_") for name in args.model_names]
    if args.target_env_steps <= 0 or args.num_minibatches <= 0:
        raise ValueError("target steps and minibatches must be positive.")
    input_csv = resolve_input_csv(args)
    rows = read_rows(input_csv, args)
    layout_rows = aggregate_layouts(rows)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    horizon = f"{args.target_env_steps // 1_000_000}m"
    mode = args.aggregation.replace("-", "_")
    print(f"Input CSV: {input_csv}")

    if args.presentation in ("separate", "both"):
        for model in args.model_names:
            output_path = args.output_dir / (
                f"generalization_mapwise_train_eval_{model.lower()}_"
                f"{mode}_{horizon}.pdf"
            )
            plot(layout_rows, args, [model], output_path)
            print(f"Saved: {output_path}")

    if args.presentation in ("combined", "both"):
        output_path = args.output_dir / (
            f"generalization_mapwise_train_eval_{mode}_{horizon}.pdf"
        )
        plot(layout_rows, args, args.model_names, output_path)
        print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()
