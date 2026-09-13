"""Measure train-evaluation return gaps from W&B training runs.

The absolute generalization gap is defined as

    train return - evaluation return

so a positive value means performance on generated training layouts is higher
than performance on the corresponding fixed evaluation layout. Metrics are
averaged over a common final training window, then aggregated across seeds.
"""
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
import wandb

try:
    from .value_loss_generalization_graph import (
        canonical_model,
        compatible_cached_rows,
        finite_float,
        select_runs,
    )
except ImportError:
    from value_loss_generalization_graph import (
        canonical_model,
        compatible_cached_rows,
        finite_float,
        select_runs,
    )


DEFAULT_ENTITY = "overcooked_ai"
DEFAULT_PROJECT = "crossenv_ICLR"
DEFAULT_TARGET_ENV_STEPS = 3_000_000_000
REPOSITORY_ROOT = Path(__file__).resolve().parents[4]
DEFAULT_OUTPUT_DIR = REPOSITORY_ROOT / "artifacts" / "generalization_gap"

LAYOUTS = [
    "cramped_room_9",
    "asymm_advantages_9",
    "coord_ring_9",
    "counter_circuit_9",
    "forced_coord_9",
]
PANELS = ["mean", *LAYOUTS]
LAYOUT_LABELS = {
    "mean": "Mean",
    "cramped_room_9": "Cramped Room",
    "asymm_advantages_9": "Asymmetric Advantages",
    "coord_ring_9": "Coordination Ring",
    "counter_circuit_9": "Counter Circuit",
    "forced_coord_9": "Forced Coordination",
}
MODEL_LABELS = {"CEC": "CEC", "CEC_IDAAC": "CEC-IDAAC"}
MODEL_COLORS = {"CEC": "#377eb8", "CEC_IDAAC": "#e68632"}
MODEL_MARKERS = {"CEC": "o", "CEC_IDAAC": "s"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--entity", default=DEFAULT_ENTITY)
    parser.add_argument("--project", default=DEFAULT_PROJECT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--target-env-steps", type=int, default=DEFAULT_TARGET_ENV_STEPS,
        help="Compare runs at this common environment-step horizon.",
    )
    parser.add_argument(
        "--tail-fraction", type=float, default=0.10,
        help="Average train/eval returns over this final horizon fraction.",
    )
    parser.add_argument(
        "--aggregation", choices=("final-window", "last"),
        default="final-window",
        help=(
            "Use a mean over the final window, or the last valid logged value "
            "at or before --target-env-steps."
        ),
    )
    parser.add_argument(
        "--min-progress", type=float, default=0.95,
        help="Minimum fraction of target steps a run must have completed.",
    )
    parser.add_argument(
        "--model-names", nargs="+", default=["CEC", "CEC_IDAAC"],
    )
    parser.add_argument(
        "--num-envs", nargs="*", type=int, default=[32, 64, 128, 256],
    )
    parser.add_argument(
        "--num-minibatches", type=int, default=2,
        help="Only include runs whose NUM_MINIBATCHES equals this value.",
    )
    parser.add_argument("--seeds", nargs="*", type=int, default=None)
    parser.add_argument("--run-name-contains", default=None)
    parser.add_argument("--include-running", action="store_true")
    parser.add_argument("--all-matching-runs", action="store_true")
    parser.add_argument(
        "--history-samples", type=int, default=10_000,
        help="Maximum sampled history rows fetched per run.",
    )
    parser.add_argument(
        "--refresh-wandb", action="store_true",
        help="Ignore an existing compatible run CSV and fetch W&B again.",
    )
    return parser.parse_args()


def fetch_run_rows(args: argparse.Namespace):
    train_keys = {layout: f"train_returns/{layout}" for layout in LAYOUTS}
    eval_keys = {layout: f"eval/{layout}" for layout in LAYOUTS}
    history_keys = ["env_step", *train_keys.values(), *eval_keys.values()]
    window_start = args.target_env_steps * (1.0 - args.tail_fraction)
    selection_start = window_start if args.aggregation == "final-window" else 0
    run_rows = []

    for item in select_runs(args):
        run = item["run"]
        print(
            f"Fetching {run.id}: {item['model']}, NUM_ENVS={item['num_envs']}, "
            f"seed={item['seed']}"
        )
        train_values = {layout: [] for layout in LAYOUTS}
        eval_values = {layout: [] for layout in LAYOUTS}
        history = run.history(
            keys=history_keys,
            samples=args.history_samples,
            pandas=False,
        )
        for history_row in history:
            env_step = finite_float(history_row.get("env_step"))
            if env_step is None:
                continue
            if env_step > args.target_env_steps:
                continue
            if args.aggregation == "final-window" and env_step < window_start:
                continue
            for layout in LAYOUTS:
                train_value = finite_float(history_row.get(train_keys[layout]))
                eval_value = finite_float(history_row.get(eval_keys[layout]))
                if train_value is not None:
                    train_values[layout].append((env_step, train_value))
                if eval_value is not None:
                    eval_values[layout].append((env_step, eval_value))

        if args.aggregation == "last":
            train_values = {
                layout: [max(values, key=lambda pair: pair[0])[1]] if values else []
                for layout, values in train_values.items()
            }
            eval_values = {
                layout: [max(values, key=lambda pair: pair[0])[1]] if values else []
                for layout, values in eval_values.items()
            }
        else:
            train_values = {
                layout: [value for _, value in values]
                for layout, values in train_values.items()
            }
            eval_values = {
                layout: [value for _, value in values]
                for layout, values in eval_values.items()
            }

        per_layout = {}
        for layout in LAYOUTS:
            if not train_values[layout] or not eval_values[layout]:
                print(
                    f"Missing final-window pair: {run.id}, layout={layout}, "
                    f"train_n={len(train_values[layout])}, "
                    f"eval_n={len(eval_values[layout])}"
                )
                continue
            train_mean = float(np.mean(train_values[layout]))
            eval_mean = float(np.mean(eval_values[layout]))
            per_layout[layout] = (train_mean, eval_mean)
            run_rows.append(
                make_run_row(
                    args, item, run, layout, train_mean, eval_mean,
                    len(train_values[layout]), len(eval_values[layout]),
                    selection_start,
                )
            )

        # Use an equal-layout mean on both sides, avoiding dependence on how
        # frequently each generated training-layout family completed episodes.
        if len(per_layout) == len(LAYOUTS):
            train_mean = float(np.mean([value[0] for value in per_layout.values()]))
            eval_mean = float(np.mean([value[1] for value in per_layout.values()]))
            run_rows.append(
                make_run_row(
                    args, item, run, "mean", train_mean, eval_mean,
                    sum(len(values) for values in train_values.values()),
                    sum(len(values) for values in eval_values.values()),
                    selection_start,
                )
            )
    return run_rows


def make_run_row(
    args, item, run, layout, train_mean, eval_mean,
    train_count, eval_count, window_start,
):
    gap = train_mean - eval_mean
    relative_gap = gap / max(abs(train_mean), 1e-12)
    return {
        "run_id": run.id,
        "run_name": run.name,
        "created_at": str(getattr(run, "created_at", "") or ""),
        "model": canonical_model(run),
        "num_envs": item["num_envs"],
        "num_minibatches": item["num_minibatches"],
        "aggregation": args.aggregation,
        "seed": item["seed"],
        "training_layout": item["training_layout"],
        "target_env_steps": args.target_env_steps,
        "window_start_env_step": int(window_start),
        "max_logged_env_step": int(item["max_logged_step"]),
        "eval_layout": layout,
        "train_return_mean": train_mean,
        "eval_return_mean": eval_mean,
        "generalization_gap": gap,
        "relative_gap": relative_gap,
        "train_history_count": train_count,
        "eval_history_count": eval_count,
    }


def aggregate_seeds(run_rows):
    grouped = defaultdict(list)
    for row in run_rows:
        grouped[(row["model"], row["num_envs"], row["eval_layout"])].append(row)

    rows = []
    for (model, num_envs, layout), group in sorted(grouped.items()):
        train = np.asarray([row["train_return_mean"] for row in group])
        evaluation = np.asarray([row["eval_return_mean"] for row in group])
        gaps = np.asarray([row["generalization_gap"] for row in group])
        relative = np.asarray([row["relative_gap"] for row in group])
        n = len(group)
        gap_std = float(gaps.std(ddof=1)) if n > 1 else 0.0
        rows.append(
            {
                "model": model,
                "num_envs": num_envs,
                "eval_layout": layout,
                "train_return_mean": float(train.mean()),
                "eval_return_mean": float(evaluation.mean()),
                "generalization_gap_mean": float(gaps.mean()),
                "generalization_gap_std": gap_std,
                "generalization_gap_sem": gap_std / math.sqrt(n),
                "relative_gap_mean": float(relative.mean()),
                "num_seeds": n,
                "seeds": " ".join(
                    str(row["seed"])
                    for row in sorted(group, key=lambda value: value["seed"])
                ),
            }
        )
    return rows


def plot(rows, models, num_envs_values, output_path: Path, aggregation: str):
    fig, axes = plt.subplots(2, 3, figsize=(11.2, 7.0), sharex=True)
    axes = axes.ravel()
    x_lookup = {num_envs: index for index, num_envs in enumerate(num_envs_values)}

    for ax, layout in zip(axes, PANELS):
        for model in models:
            points = sorted(
                (
                    row for row in rows
                    if row["model"] == model and row["eval_layout"] == layout
                ),
                key=lambda row: row["num_envs"],
            )
            if not points:
                continue
            x = np.asarray([x_lookup[row["num_envs"]] for row in points])
            y = np.asarray([row["generalization_gap_mean"] for row in points])
            yerr = np.asarray([row["generalization_gap_sem"] for row in points])
            ax.errorbar(
                x, y, yerr=yerr,
                color=MODEL_COLORS.get(model),
                marker=MODEL_MARKERS.get(model, "o"),
                linewidth=1.8, markersize=5.5, capsize=3,
            )
        ax.axhline(0.0, color="black", linestyle="--", linewidth=1.0, alpha=0.65)
        ax.set_title(LAYOUT_LABELS[layout], fontsize=11)
        ax.set_xticks(range(len(num_envs_values)))
        ax.set_xticklabels(num_envs_values)
        ax.grid(alpha=0.25)

    for index, ax in enumerate(axes):
        if index // 3 == 1:
            ax.set_xlabel("Number of parallel training environments")
        if index % 3 == 0:
            ax.set_ylabel("Train return - Evaluation return")

    legend_handles = [
        Line2D(
            [0], [0], color=MODEL_COLORS.get(model),
            marker=MODEL_MARKERS.get(model, "o"), linewidth=1.8,
            markersize=5.5,
            label=MODEL_LABELS.get(model, model.replace("_", "-")),
        )
        for model in models
    ]
    if legend_handles:
        fig.legend(
            handles=legend_handles, loc="upper center",
            ncol=len(legend_handles), bbox_to_anchor=(0.5, 0.875),
            frameon=True,
        )
    statistic_label = (
        "Last logged metrics"
        if aggregation == "last"
        else "Final-window mean metrics"
    )
    fig.suptitle(
        f"Generalization Gap by Evaluation Layout ({statistic_label})\n"
        "Positive values indicate higher training-layout performance",
        fontsize=14,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.80))
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def write_csv(path: Path, rows):
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    args = parse_args()
    if args.target_env_steps <= 0:
        raise ValueError("--target-env-steps must be positive.")
    if not 0.0 < args.tail_fraction <= 1.0:
        raise ValueError("--tail-fraction must be in (0, 1].")
    if not 0.0 < args.min_progress <= 1.0:
        raise ValueError("--min-progress must be in (0, 1].")
    if args.history_samples <= 0:
        raise ValueError("--history-samples must be positive.")
    if args.num_minibatches <= 0:
        raise ValueError("--num-minibatches must be positive.")

    args.model_names = [name.upper().replace("-", "_") for name in args.model_names]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    horizon = f"{args.target_env_steps // 1_000_000}m"
    mode = args.aggregation.replace("-", "_")
    suffix = f"{mode}_{horizon}"
    run_csv = args.output_dir / f"generalization_gap_runs_{suffix}.csv"
    aggregate_csv = args.output_dir / f"generalization_gap_by_num_envs_{suffix}.csv"
    figure_path = args.output_dir / f"generalization_gap_{suffix}.pdf"

    run_rows = compatible_cached_rows(run_csv, args)
    if run_rows is None and args.aggregation == "final-window":
        legacy_run_csv = args.output_dir / f"generalization_gap_runs_{horizon}.csv"
        run_rows = compatible_cached_rows(legacy_run_csv, args)
    loaded_from_cache = run_rows is not None
    if run_rows is None:
        print("No compatible CSV cache; fetching W&B data.")
        run_rows = fetch_run_rows(args)
    if not run_rows:
        raise RuntimeError(
            "No matching W&B train_returns/* and eval/* data found."
        )
    aggregate_rows = aggregate_seeds(run_rows)
    if not loaded_from_cache:
        write_csv(run_csv, run_rows)
        write_csv(aggregate_csv, aggregate_rows)
    plot(
        aggregate_rows, args.model_names, args.num_envs, figure_path,
        args.aggregation,
    )

    print(f"Runs with data: {len({row['run_id'] for row in run_rows})}")
    if not loaded_from_cache:
        print(f"Saved: {run_csv}")
        print(f"Saved: {aggregate_csv}")
    print(f"Saved: {figure_path}")


if __name__ == "__main__":
    main()
