"""Compare training and evaluation TD-error RMSE from W&B runs.

Raw history is cached as CSV after the first download. The script produces
both last-value and final-window summaries by default, with one PDF per model.
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

try:
    from .value_loss_generalization_graph import finite_float, select_runs
except ImportError:
    from value_loss_generalization_graph import finite_float, select_runs


DEFAULT_ENTITY = "overcooked_ai"
DEFAULT_PROJECT = "crossenv_ICLR"
DEFAULT_TARGET_ENV_STEPS = 3_000_000_000
REPOSITORY_ROOT = Path(__file__).resolve().parents[4]
DEFAULT_OUTPUT_DIR = REPOSITORY_ROOT / "artifacts" / "td_error_generalization"

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
SPLIT_COLORS = {"train": "#377eb8", "evaluation": "#e68632"}
SPLIT_MARKERS = {"train": "o", "evaluation": "s"}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--entity", default=DEFAULT_ENTITY)
    parser.add_argument("--project", default=DEFAULT_PROJECT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--target-env-steps", type=int, default=DEFAULT_TARGET_ENV_STEPS,
    )
    parser.add_argument("--tail-fraction", type=float, default=0.10)
    parser.add_argument(
        "--aggregation", choices=("both", "last", "final-window"),
        default="both",
    )
    parser.add_argument("--min-progress", type=float, default=0.95)
    parser.add_argument(
        "--model-names", nargs="+", default=["CEC", "CEC_IDAAC"],
    )
    parser.add_argument(
        "--num-envs", nargs="+", type=int, default=[32, 64, 128, 256],
    )
    parser.add_argument("--num-minibatches", type=int, default=2)
    parser.add_argument("--seeds", nargs="*", type=int, default=None)
    parser.add_argument("--run-name-contains", default=None)
    parser.add_argument("--include-running", action="store_true")
    parser.add_argument("--all-matching-runs", action="store_true")
    parser.add_argument("--history-samples", type=int, default=10_000)
    parser.add_argument("--refresh-wandb", action="store_true")
    return parser.parse_args()


def fetch_history_rows(args):
    train_keys = {layout: f"td_error/{layout}/rmse" for layout in LAYOUTS}
    eval_keys = {
        layout: f"eval_critic/{layout}/td_error_rmse" for layout in LAYOUTS
    }
    keys = ["env_step", *train_keys.values(), *eval_keys.values()]
    rows = []

    for item in select_runs(args):
        run = item["run"]
        print(
            f"Fetching {run.id}: {item['model']}, NUM_ENVS={item['num_envs']}, "
            f"seed={item['seed']}"
        )
        history = run.history(keys=keys, samples=args.history_samples, pandas=False)
        for history_row in history:
            env_step = finite_float(history_row.get("env_step"))
            if env_step is None or env_step < 0 or env_step > args.target_env_steps:
                continue
            for layout in LAYOUTS:
                train = finite_float(history_row.get(train_keys[layout]))
                evaluation = finite_float(history_row.get(eval_keys[layout]))
                if train is None and evaluation is None:
                    continue
                rows.append(
                    {
                        "run_id": run.id,
                        "run_name": run.name,
                        "model": item["model"],
                        "num_envs": item["num_envs"],
                        "num_minibatches": item["num_minibatches"],
                        "seed": item["seed"],
                        "training_layout": item["training_layout"],
                        "env_step": int(env_step),
                        "eval_layout": layout,
                        "train_td_error_rmse": train,
                        "eval_td_error_rmse": evaluation,
                    }
                )
    return rows


def write_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def read_history_csv(path):
    with path.open(newline="", encoding="utf-8") as file:
        rows = list(csv.DictReader(file))
    for row in rows:
        for key in ("num_envs", "num_minibatches", "seed", "env_step"):
            row[key] = int(float(row[key]))
        for key in ("train_td_error_rmse", "eval_td_error_rmse"):
            row[key] = finite_float(row[key])
    return rows


def compatible_cache(path, args):
    if args.refresh_wandb or not path.exists():
        return None
    if args.run_name_contains or args.include_running or args.all_matching_runs:
        return None
    rows = read_history_csv(path)
    selected = [
        row for row in rows
        if row["model"] in args.model_names
        and row["num_envs"] in args.num_envs
        and row["num_minibatches"] == args.num_minibatches
        and row["env_step"] <= args.target_env_steps
        and (args.seeds is None or row["seed"] in args.seeds)
    ]
    available = {(row["model"], row["num_envs"]) for row in selected}
    requested = {
        (model, num_envs)
        for model in args.model_names for num_envs in args.num_envs
    }
    if not selected or not requested.issubset(available):
        return None
    if args.seeds is not None:
        available_seed_groups = {
            (row["model"], row["num_envs"], row["seed"])
            for row in selected
        }
        requested_seed_groups = {
            (model, num_envs, seed)
            for model in args.model_names
            for num_envs in args.num_envs
            for seed in args.seeds
        }
        if not requested_seed_groups.issubset(available_seed_groups):
            return None
    print(f"Using cached TD-error histories: {path}")
    return selected


def reduce_values(observations, aggregation, window_start):
    observations = [
        (step, value) for step, value in observations if value is not None
    ]
    if aggregation == "last":
        if not observations:
            return None, 0
        return max(observations, key=lambda pair: pair[0])[1], 1
    values = [value for step, value in observations if step >= window_start]
    return (float(np.mean(values)), len(values)) if values else (None, 0)


def summarize_runs(history_rows, aggregation, target_steps, tail_fraction):
    grouped = defaultdict(list)
    for row in history_rows:
        grouped[(row["run_id"], row["eval_layout"])].append(row)
    window_start = target_steps * (1.0 - tail_fraction)
    rows = []

    for (_, layout), group in grouped.items():
        first = group[0]
        train, train_count = reduce_values(
            [(row["env_step"], row["train_td_error_rmse"]) for row in group],
            aggregation, window_start,
        )
        evaluation, eval_count = reduce_values(
            [(row["env_step"], row["eval_td_error_rmse"]) for row in group],
            aggregation, window_start,
        )
        if train is None or evaluation is None:
            continue
        rows.append(
            {
                "run_id": first["run_id"],
                "run_name": first["run_name"],
                "model": first["model"],
                "num_envs": first["num_envs"],
                "num_minibatches": first["num_minibatches"],
                "seed": first["seed"],
                "training_layout": first["training_layout"],
                "aggregation": aggregation,
                "eval_layout": layout,
                "train_td_error_rmse": train,
                "eval_td_error_rmse": evaluation,
                "td_error_gap": evaluation - train,
                "train_count": train_count,
                "eval_count": eval_count,
            }
        )

    by_run = defaultdict(list)
    for row in rows:
        by_run[row["run_id"]].append(row)
    for group in by_run.values():
        layout_rows = {row["eval_layout"]: row for row in group}
        if not all(layout in layout_rows for layout in LAYOUTS):
            continue
        first = group[0]
        train = float(np.mean([
            layout_rows[layout]["train_td_error_rmse"] for layout in LAYOUTS
        ]))
        evaluation = float(np.mean([
            layout_rows[layout]["eval_td_error_rmse"] for layout in LAYOUTS
        ]))
        rows.append(
            {
                "run_id": first["run_id"],
                "run_name": first["run_name"],
                "model": first["model"],
                "num_envs": first["num_envs"],
                "num_minibatches": first["num_minibatches"],
                "seed": first["seed"],
                "training_layout": first["training_layout"],
                "aggregation": aggregation,
                "eval_layout": "mean",
                "train_td_error_rmse": train,
                "eval_td_error_rmse": evaluation,
                "td_error_gap": evaluation - train,
                "train_count": sum(row["train_count"] for row in group),
                "eval_count": sum(row["eval_count"] for row in group),
            }
        )
    return rows


def aggregate_seeds(run_rows):
    grouped = defaultdict(list)
    for row in run_rows:
        grouped[(row["model"], row["num_envs"], row["eval_layout"])].append(row)
    rows = []
    for (model, num_envs, layout), group in sorted(grouped.items()):
        train = np.asarray([row["train_td_error_rmse"] for row in group])
        evaluation = np.asarray([row["eval_td_error_rmse"] for row in group])
        gap = evaluation - train
        n = len(group)
        rows.append(
            {
                "model": model,
                "num_envs": num_envs,
                "eval_layout": layout,
                "train_td_error_rmse_mean": float(train.mean()),
                "train_td_error_rmse_sem": (
                    float(train.std(ddof=1) / math.sqrt(n)) if n > 1 else 0.0
                ),
                "eval_td_error_rmse_mean": float(evaluation.mean()),
                "eval_td_error_rmse_sem": (
                    float(evaluation.std(ddof=1) / math.sqrt(n)) if n > 1 else 0.0
                ),
                "td_error_gap_mean": float(gap.mean()),
                "num_seeds": n,
                "seeds": " ".join(str(row["seed"]) for row in sorted(group, key=lambda x: x["seed"])),
            }
        )
    return rows


def plot_model(rows, model, num_envs_values, aggregation, output_path):
    fig, axes = plt.subplots(2, 3, figsize=(11.2, 7.0), sharex=True)
    axes = axes.ravel()
    x_lookup = {num_envs: index for index, num_envs in enumerate(num_envs_values)}

    for ax, layout in zip(axes, PANELS):
        points = sorted(
            (row for row in rows if row["model"] == model and row["eval_layout"] == layout),
            key=lambda row: row["num_envs"],
        )
        for split in ("train", "evaluation"):
            column_prefix = "train" if split == "train" else "eval"
            mean_key = f"{column_prefix}_td_error_rmse_mean"
            sem_key = f"{column_prefix}_td_error_rmse_sem"
            x = np.asarray([x_lookup[row["num_envs"]] for row in points])
            y = np.asarray([row[mean_key] for row in points])
            yerr = np.asarray([row[sem_key] for row in points])
            ax.errorbar(
                x, y, yerr=yerr, color=SPLIT_COLORS[split],
                marker=SPLIT_MARKERS[split], linewidth=1.8,
                markersize=5.5, capsize=3,
            )
        ax.set_title(LAYOUT_LABELS[layout], fontsize=12)
        ax.set_xticks(range(len(num_envs_values)))
        ax.set_xticklabels(num_envs_values)
        ax.grid(alpha=0.25)

    for index, ax in enumerate(axes):
        if index // 3 == 1:
            ax.set_xlabel("Number of parallel training environments")
        if index % 3 == 0:
            ax.set_ylabel("TD-error RMSE")

    handles = [
        Line2D(
            [0], [0], color=SPLIT_COLORS[split], marker=SPLIT_MARKERS[split],
            linewidth=1.8, markersize=5.5, label=split.capitalize(),
        )
        for split in ("train", "evaluation")
    ]
    fig.legend(handles=handles, loc="upper center", ncol=2,
               bbox_to_anchor=(0.5, 0.89), frameon=True)
    statistic = "Last logged values" if aggregation == "last" else "Final-window means"
    fig.suptitle(
        f"{MODEL_LABELS.get(model, model)}: Train vs Evaluation TD-error RMSE\n"
        f"{statistic}; gap = evaluation - train",
        fontsize=14,
    )
    fig.tight_layout(rect=(0.025, 0.01, 0.99, 0.81))
    fig.savefig(output_path)
    plt.close(fig)


def main():
    args = parse_args()
    if args.target_env_steps <= 0:
        raise ValueError("--target-env-steps must be positive.")
    if not 0 < args.tail_fraction <= 1:
        raise ValueError("--tail-fraction must be in (0, 1].")
    if not 0 < args.min_progress <= 1:
        raise ValueError("--min-progress must be in (0, 1].")
    if args.num_minibatches <= 0 or args.history_samples <= 0:
        raise ValueError("minibatches and history samples must be positive.")

    args.model_names = [name.upper().replace("-", "_") for name in args.model_names]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    horizon = f"{args.target_env_steps // 1_000_000}m"
    history_csv = args.output_dir / f"td_error_history_{horizon}.csv"
    history_rows = compatible_cache(history_csv, args)
    if history_rows is None:
        print("No compatible TD-error CSV; fetching W&B histories.")
        history_rows = fetch_history_rows(args)
        if not history_rows:
            raise RuntimeError("No matching train/evaluation TD-error data found.")
        write_csv(history_csv, history_rows)
        print(f"Saved: {history_csv}")

    aggregations = (
        ("last", "final-window") if args.aggregation == "both"
        else (args.aggregation,)
    )
    for aggregation in aggregations:
        mode = aggregation.replace("-", "_")
        run_rows = summarize_runs(
            history_rows, aggregation, args.target_env_steps, args.tail_fraction,
        )
        aggregate_rows = aggregate_seeds(run_rows)
        run_csv = args.output_dir / f"td_error_runs_{mode}_{horizon}.csv"
        aggregate_csv = args.output_dir / f"td_error_by_num_envs_{mode}_{horizon}.csv"
        write_csv(run_csv, run_rows)
        write_csv(aggregate_csv, aggregate_rows)
        print(f"Saved: {run_csv}")
        print(f"Saved: {aggregate_csv}")
        for model in args.model_names:
            output_path = args.output_dir / (
                f"td_error_train_vs_eval_{model.lower()}_{mode}_{horizon}.pdf"
            )
            plot_model(
                aggregate_rows, model, args.num_envs, aggregation, output_path,
            )
            print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()
