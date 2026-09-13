"""Plot train and evaluation learning curves for CEC and CEC-IDAAC.

The first invocation downloads run histories from W&B and stores a long-form
CSV. Later invocations reuse that CSV unless --refresh-wandb is supplied.
One PDF is written per NUM_ENVS value to keep the four train/evaluation curves
in each layout panel readable.
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
import numpy as np

try:
    from .value_loss_generalization_graph import finite_float, select_runs
except ImportError:
    from value_loss_generalization_graph import finite_float, select_runs


DEFAULT_ENTITY = "overcooked_ai"
DEFAULT_PROJECT = "crossenv_ICLR"
DEFAULT_TARGET_ENV_STEPS = 3_000_000_000
REPOSITORY_ROOT = Path(__file__).resolve().parents[4]
DEFAULT_OUTPUT_DIR = REPOSITORY_ROOT / "artifacts" / "generalization_learning_curves"

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
ENV_COLORS = {
    32: "#9ecae1",
    64: "#6baed6",
    128: "#3182bd",
    256: "#08519c",
}
SPLIT_STYLES = {"train": "-", "evaluation": "--"}


def parse_args(default_model_names=None) -> argparse.Namespace:
    if default_model_names is None:
        default_model_names = ["CEC"]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--entity", default=DEFAULT_ENTITY)
    parser.add_argument("--project", default=DEFAULT_PROJECT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--target-env-steps", type=int, default=DEFAULT_TARGET_ENV_STEPS,
    )
    parser.add_argument("--min-progress", type=float, default=0.95)
    parser.add_argument(
        "--model-names", nargs="+", default=default_model_names,
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
    parser.add_argument(
        "--refresh-wandb", action="store_true",
        help="Ignore a compatible curve CSV and fetch W&B histories again.",
    )
    return parser.parse_args()


def make_row(item, env_step, layout, split, value):
    run = item["run"]
    return {
        "run_id": run.id,
        "run_name": run.name,
        "model": item["model"],
        "num_envs": item["num_envs"],
        "num_minibatches": item["num_minibatches"],
        "seed": item["seed"],
        "training_layout": item["training_layout"],
        "env_step": int(env_step),
        "eval_layout": layout,
        "split": split,
        "return": float(value),
    }


def fetch_rows(args: argparse.Namespace):
    train_keys = {layout: f"train_returns/{layout}" for layout in LAYOUTS}
    eval_keys = {layout: f"eval/{layout}" for layout in LAYOUTS}
    history_keys = ["env_step", *train_keys.values(), *eval_keys.values()]
    rows = []

    for item in select_runs(args):
        run = item["run"]
        print(
            f"Fetching {run.id}: {item['model']}, NUM_ENVS={item['num_envs']}, "
            f"seed={item['seed']}"
        )
        history = run.history(
            keys=history_keys,
            samples=args.history_samples,
            pandas=False,
        )
        for history_row in history:
            env_step = finite_float(history_row.get("env_step"))
            if env_step is None or env_step < 0 or env_step > args.target_env_steps:
                continue

            train_at_step = {}
            eval_at_step = {}
            for layout in LAYOUTS:
                train_value = finite_float(history_row.get(train_keys[layout]))
                eval_value = finite_float(history_row.get(eval_keys[layout]))
                if train_value is not None:
                    train_at_step[layout] = train_value
                    rows.append(make_row(item, env_step, layout, "train", train_value))
                if eval_value is not None:
                    eval_at_step[layout] = eval_value
                    rows.append(
                        make_row(item, env_step, layout, "evaluation", eval_value)
                    )

            # The mean panel gives every fixed layout equal weight.
            if len(train_at_step) == len(LAYOUTS):
                rows.append(
                    make_row(
                        item, env_step, "mean", "train",
                        np.mean(list(train_at_step.values())),
                    )
                )
            if len(eval_at_step) == len(LAYOUTS):
                rows.append(
                    make_row(
                        item, env_step, "mean", "evaluation",
                        np.mean(list(eval_at_step.values())),
                    )
                )
    return rows


def write_csv(path: Path, rows):
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def read_csv(path: Path):
    with path.open(newline="", encoding="utf-8") as file:
        rows = list(csv.DictReader(file))
    for row in rows:
        for key in ("num_envs", "num_minibatches", "seed", "env_step"):
            row[key] = int(float(row[key]))
        row["return"] = float(row["return"])
    return rows


def compatible_cached_rows(path: Path, args: argparse.Namespace):
    if args.refresh_wandb or not path.exists():
        return None
    if args.run_name_contains or args.include_running or args.all_matching_runs:
        return None
    rows = read_csv(path)
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
        available_seeds = {
            (row["model"], row["num_envs"], row["seed"])
            for row in selected
        }
        requested_seeds = {
            (model, num_envs, seed)
            for model in args.model_names
            for num_envs in args.num_envs
            for seed in args.seeds
        }
        if not requested_seeds.issubset(available_seeds):
            return None
    print(f"Using cached W&B histories: {path}")
    return selected


def aggregate_seeds(rows):
    grouped = defaultdict(list)
    for row in rows:
        key = (
            row["model"], row["num_envs"], row["env_step"],
            row["eval_layout"], row["split"],
        )
        grouped[key].append(row["return"])

    aggregated = []
    for (model, num_envs, env_step, layout, split), values in sorted(grouped.items()):
        values = np.asarray(values, dtype=float)
        std = float(values.std(ddof=1)) if len(values) > 1 else 0.0
        aggregated.append(
            {
                "model": model,
                "num_envs": num_envs,
                "env_step": env_step,
                "eval_layout": layout,
                "split": split,
                "return_mean": float(values.mean()),
                "return_sem": std / math.sqrt(len(values)),
                "num_seeds": len(values),
            }
        )
    return aggregated


def plot_num_envs(rows, args: argparse.Namespace, num_envs: int, output_path: Path):
    fig, axes = plt.subplots(2, 3, figsize=(11.2, 7.0), sharex=True)
    axes = axes.ravel()

    for ax, layout in zip(axes, PANELS):
        for model in args.model_names:
            for split in ("train", "evaluation"):
                points = sorted(
                    (
                        row for row in rows
                        if row["model"] == model
                        and row["num_envs"] == num_envs
                        and row["eval_layout"] == layout
                        and row["split"] == split
                    ),
                    key=lambda row: row["env_step"],
                )
                if not points:
                    continue
                x = np.asarray([row["env_step"] for row in points]) / 1e9
                y = np.asarray([row["return_mean"] for row in points])
                sem = np.asarray([row["return_sem"] for row in points])
                color = MODEL_COLORS[model]
                label = f"{MODEL_LABELS[model]} {split}"
                ax.plot(
                    x, y, SPLIT_STYLES[split], color=color,
                    linewidth=1.8, label=label,
                )
                ax.fill_between(x, y - sem, y + sem, color=color, alpha=0.12)
        ax.set_title(LAYOUT_LABELS[layout], fontsize=12)
        ax.grid(alpha=0.25)

    for index, ax in enumerate(axes):
        if index // 3 == 1:
            ax.set_xlabel("Environment steps (billions)")
        if index % 3 == 0:
            ax.set_ylabel("Return")

    handles, labels = axes[0].get_legend_handles_labels()
    unique = dict(zip(labels, handles))
    if unique:
        fig.legend(
            unique.values(), unique.keys(), loc="upper center", ncol=4,
            bbox_to_anchor=(0.5, 0.94), frameon=True,
        )
    model_title = (
        "CEC"
        if args.model_names == ["CEC"]
        else " and ".join(MODEL_LABELS.get(model, model) for model in args.model_names)
    )
    fig.suptitle(
        f"{model_title} Train and Evaluation Learning Curves "
        f"(NUM_ENVS={num_envs}, NUM_MINIBATCHES={args.num_minibatches})",
        fontsize=15, y=0.982,
    )
    # Explicit margins avoid the large title/legend gap introduced by
    # tight_layout while retaining enough room for the shared legend.
    fig.subplots_adjust(
        left=0.065, right=0.985, bottom=0.09, top=0.84,
        wspace=0.22, hspace=0.28,
    )
    fig.savefig(output_path)
    plt.close(fig)


def plot_all_num_envs(rows, args: argparse.Namespace, output_path: Path):
    """Overlay CEC curves for all requested NUM_ENVS values in one figure."""
    fig, axes = plt.subplots(2, 3, figsize=(11.8, 7.2), sharex=True)
    axes = axes.ravel()

    for ax, layout in zip(axes, PANELS):
        for num_envs in args.num_envs:
            for split in ("train", "evaluation"):
                points = sorted(
                    (
                        row for row in rows
                        if row["model"] == "CEC"
                        and row["num_envs"] == num_envs
                        and row["eval_layout"] == layout
                        and row["split"] == split
                    ),
                    key=lambda row: row["env_step"],
                )
                if not points:
                    continue
                x = np.asarray([row["env_step"] for row in points]) / 1e9
                y = np.asarray([row["return_mean"] for row in points])
                sem = np.asarray([row["return_sem"] for row in points])
                color = ENV_COLORS.get(num_envs, "#377eb8")
                ax.plot(
                    x, y, SPLIT_STYLES[split], color=color, linewidth=1.65,
                    label=f"{num_envs} {split}",
                )
                ax.fill_between(x, y - sem, y + sem, color=color, alpha=0.08)
        ax.set_title(LAYOUT_LABELS[layout], fontsize=12)
        ax.grid(alpha=0.25)

    for index, ax in enumerate(axes):
        if index // 3 == 1:
            ax.set_xlabel("Environment steps (billions)")
        if index % 3 == 0:
            ax.set_ylabel("Return")

    handles, labels = axes[0].get_legend_handles_labels()
    unique = dict(zip(labels, handles))
    if unique:
        fig.legend(
            unique.values(), unique.keys(), loc="upper center", ncol=4,
            bbox_to_anchor=(0.5, 0.945), frameon=True,
        )
    fig.suptitle(
        f"CEC Train and Evaluation Learning Curves "
        f"(NUM_MINIBATCHES={args.num_minibatches})",
        fontsize=15, y=0.982,
    )
    # The all-environment legend uses two rows, so it needs slightly more
    # headroom than the individual NUM_ENVS figures.
    fig.subplots_adjust(
        left=0.065, right=0.985, bottom=0.09, top=0.80,
        wspace=0.22, hspace=0.28,
    )
    fig.savefig(output_path)
    plt.close(fig)


def main(default_model_names=None, output_tag="cec", make_combined=True):
    args = parse_args(default_model_names)
    if args.target_env_steps <= 0:
        raise ValueError("--target-env-steps must be positive.")
    if not 0.0 < args.min_progress <= 1.0:
        raise ValueError("--min-progress must be in (0, 1].")
    if args.num_minibatches <= 0:
        raise ValueError("--num-minibatches must be positive.")
    if args.history_samples <= 0:
        raise ValueError("--history-samples must be positive.")

    args.model_names = [name.upper().replace("-", "_") for name in args.model_names]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    horizon = f"{args.target_env_steps // 1_000_000}m"
    run_csv = args.output_dir / f"generalization_learning_curves_runs_{horizon}.csv"

    rows = compatible_cached_rows(run_csv, args)
    if rows is None:
        print("No compatible curve CSV; fetching W&B histories.")
        rows = fetch_rows(args)
        if not rows:
            raise RuntimeError("No matching W&B learning-curve data found.")
        write_csv(run_csv, rows)
        print(f"Saved: {run_csv}")

    aggregated = aggregate_seeds(rows)
    for num_envs in args.num_envs:
        tag = f"_{output_tag}" if output_tag else ""
        output_path = args.output_dir / (
            f"generalization_learning_curves{tag}_envs{num_envs}_{horizon}.pdf"
        )
        plot_num_envs(aggregated, args, num_envs, output_path)
        print(f"Saved: {output_path}")

    if make_combined and "CEC" in args.model_names:
        combined_path = args.output_dir / (
            f"generalization_learning_curves_cec_all_envs_{horizon}.pdf"
        )
        plot_all_num_envs(aggregated, args, combined_path)
        print(f"Saved: {combined_path}")


if __name__ == "__main__":
    main()
