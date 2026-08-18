"""Compare eval XP learning curves grouped by ``config.model_name`` in wandb."""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import wandb


ENTITY = "overcooked_ai"
PROJECT = "crossenv_ICLR"
MODEL_NAMES = ["CEC_IDAAC", "CEC_POP", "CEC_IDAAC_POP", "CEC"]
SMOOTH_WINDOW = 1
MIN_RUN_FRACTION = 0.5

SAVE_DIR = Path(__file__).parent.parent / "results" / "eval_xp_model_graph"
MODEL_COLORS = [
    "#7f56d9", "#e66b2e", "#ef3e4a", "#df70d6", "#55a868",
    "#f2a65a", "#4c72b0", "#c44e52", "#8172b2", "#ccb974",
]


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--entity", default=ENTITY)
    parser.add_argument("--project", default=PROJECT)
    parser.add_argument("--model-names", nargs="+", default=MODEL_NAMES)
    parser.add_argument("--num-envs", type=int, nargs="+")
    parser.add_argument("--seeds", type=int, nargs="+")
    parser.add_argument("--smooth-window", type=int, default=SMOOTH_WINDOW)
    parser.add_argument("--min-run-fraction", type=float, default=MIN_RUN_FRACTION)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def fetch_model_histories(
    entity: str,
    project: str,
    model_names,
    num_envs=None,
    seeds_by_model=None,
):
    api = wandb.Api()
    filters = {"config.model_name": {"$in": list(model_names)}}
    if num_envs is not None:
        if isinstance(num_envs, (list, tuple, set)):
            filters["config.NUM_ENVS"] = {"$in": [int(n) for n in num_envs]}
        else:
            filters["config.NUM_ENVS"] = int(num_envs)
    runs = api.runs(
        f"{entity}/{project}",
        filters=filters,
    )

    histories = []
    run_counts = {model_name: 0 for model_name in model_names}
    for run in runs:
        model_name = run.config.get("model_name")
        if model_name not in run_counts:
            continue
        if seeds_by_model is not None:
            allowed_seeds = seeds_by_model.get(model_name, ())
            if int(run.config.get("SEED", -1)) not in allowed_seeds:
                continue

        history = run.history(
            keys=["_step", "env_step", "update_step", "eval_xp/mean"],
            samples=10000,
        )
        if "eval_xp/mean" not in history.columns:
            print(f"Skipping {run.id} ({model_name}): no eval_xp/mean history")
            continue
        history = history.dropna(subset=["eval_xp/mean"])
        if history.empty:
            print(f"Skipping {run.id} ({model_name}): no eval_xp/mean history")
            continue

        steps_per_update = (
            int(run.config["NUM_ENVS"]) * int(run.config["NUM_STEPS"])
        )
        has_update_step = (
            "update_step" in history.columns
            and history["update_step"].notna().any()
        )
        has_wandb_step = (
            "_step" in history.columns and history["_step"].notna().any()
        )
        has_env_step = (
            "env_step" in history.columns and history["env_step"].notna().any()
        )
        if not has_update_step and has_wandb_step:
            history["update_step"] = history["_step"]
        elif not has_update_step and has_env_step:
            history["update_step"] = history["env_step"] / steps_per_update
        elif not has_update_step:
            print(f"Skipping {run.id} ({model_name}): no step history")
            continue

        # Recompute this instead of trusting old runs' env_step field: some
        # trainers historically logged the raw update index under that name.
        history["env_step"] = history["update_step"] * steps_per_update
        history = history[["env_step", "eval_xp/mean"]].dropna()
        history["model_name"] = model_name
        history["run_id"] = run.id
        history["num_envs"] = int(run.config["NUM_ENVS"])
        history["seed"] = int(run.config["SEED"])
        histories.append(history)
        run_counts[model_name] += 1

    if not histories:
        raise RuntimeError(
            "No eval_xp/mean data found for the requested model names."
        )

    return pd.concat(histories, ignore_index=True), run_counts


def aggregate_histories(histories, model_name: str, num_envs=None):
    model_history = histories[histories["model_name"] == model_name]
    if num_envs is not None:
        model_history = model_history[model_history["num_envs"] == num_envs]
    per_run = (
        model_history.groupby(["run_id", "env_step"], as_index=False)[
            "eval_xp/mean"
        ]
        .mean()
    )
    if per_run.empty:
        return pd.DataFrame(
            columns=["env_step", "mean", "minimum", "maximum", "count"]
        )

    # Evaluation intervals can differ with NUM_ENVS. Interpolate each run onto
    # one common grid so the model average does not alternate between subsets
    # of runs at adjacent x positions.
    grid = np.linspace(per_run["env_step"].min(), per_run["env_step"].max(), 500)
    interpolated = []
    for _, run_history in per_run.groupby("run_id"):
        run_history = run_history.sort_values("env_step")
        x = run_history["env_step"].to_numpy(dtype=float)
        y = run_history["eval_xp/mean"].to_numpy(dtype=float)
        values = np.interp(grid, x, y, left=np.nan, right=np.nan)
        interpolated.append(values)

    values = pd.DataFrame(np.stack(interpolated), columns=grid)
    return pd.DataFrame(
        {
            "env_step": grid,
            "mean": values.mean(axis=0).to_numpy(),
            "minimum": values.min(axis=0).to_numpy(),
            "maximum": values.max(axis=0).to_numpy(),
            "count": values.count(axis=0).to_numpy(),
        }
    )


def plot_eval_xp_by_model(
    histories,
    model_names,
    smooth_window: int,
    min_run_fraction: float,
    out_path: Path,
    title: str = "BC Cross-Play Evaluation by Model",
    label_suffixes=None,
    num_envs_values=None,
):
    if smooth_window < 1:
        raise ValueError("smooth_window must be at least 1")
    if not 0.0 < min_run_fraction <= 1.0:
        raise ValueError("min_run_fraction must be in (0, 1]")

    fig, ax = plt.subplots(figsize=(11.5, 5.5))
    if num_envs_values:
        series = [
            (model_name, int(num_envs))
            for model_name in model_names
            for num_envs in num_envs_values
        ]
    else:
        series = [(model_name, None) for model_name in model_names]

    plotted = 0
    for index, (model_name, num_envs) in enumerate(series):
        stats = aggregate_histories(histories, model_name, num_envs)
        group_history = histories[histories["model_name"] == model_name]
        if num_envs is not None:
            group_history = group_history[group_history["num_envs"] == num_envs]
        group_run_count = group_history["run_id"].nunique()
        minimum_runs = max(
            1, int(np.ceil(group_run_count * min_run_fraction))
        )
        stats = stats[stats["count"] >= minimum_runs]
        if stats.empty:
            print(f"Skipping {model_name}: no usable runs")
            continue

        mean = stats["mean"].rolling(smooth_window, min_periods=1).mean()
        minimum = stats["minimum"].rolling(
            smooth_window, min_periods=1
        ).mean()
        maximum = stats["maximum"].rolling(
            smooth_window, min_periods=1
        ).mean()
        color = MODEL_COLORS[index % len(MODEL_COLORS)]
        suffix = "" if label_suffixes is None else label_suffixes.get(model_name, "")
        if num_envs is None:
            label = f"{model_name}{suffix} (n={group_run_count})"
        else:
            label = (
                f"NUM_ENVS: {num_envs}, model_name: {model_name}{suffix} "
                f"(n={group_run_count})"
            )
        ax.plot(
            stats["env_step"],
            mean,
            color=color,
            linewidth=2,
            label=label,
        )
        ax.fill_between(
            stats["env_step"],
            minimum,
            maximum,
            color=color,
            alpha=0.16,
            linewidth=0,
        )
        plotted += 1

    if plotted == 0:
        raise RuntimeError("No model curves could be plotted.")

    ax.set_xlabel("Environment Steps")
    ax.set_ylabel("Eval XP Return")
    ax.margins(x=0)
    ax.set_title(title, pad=54)
    ax.grid(alpha=0.3)
    ax.legend(
        fontsize=8,
        loc="lower center",
        bbox_to_anchor=(0.5, 1.01),
        ncol=3,
        columnspacing=1.0,
        handletextpad=0.5,
        frameon=False,
    )
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


def main():
    args = parse_args()
    seeds_by_model = None
    if args.seeds:
        seeds_by_model = {
            model_name: tuple(args.seeds) for model_name in args.model_names
        }
    histories, _ = fetch_model_histories(
        args.entity,
        args.project,
        args.model_names,
        num_envs=args.num_envs,
        seeds_by_model=seeds_by_model,
    )
    filename_parts = ["eval_xp_by_model"]
    title_filters = []
    if args.num_envs:
        num_envs_label = "_".join(str(value) for value in args.num_envs)
        filename_parts.append(f"num_envs{num_envs_label}")
        title_filters.append(
            f"NUM_ENVS={','.join(str(value) for value in args.num_envs)}"
        )
    if args.seeds:
        title_filters.append(
            f"SEEDS={','.join(str(seed) for seed in args.seeds)}"
        )
    filename_parts.append("env_step")
    default_filename = "_".join(filename_parts) + ".png"
    if title_filters:
        title = (
            "BC Cross-Play Evaluation by Model "
            f"({'; '.join(title_filters)})"
        )
    else:
        title = "BC Cross-Play Evaluation by Model"
    output = args.output or (SAVE_DIR / default_filename)
    plot_eval_xp_by_model(
        histories=histories,
        model_names=args.model_names,
        smooth_window=args.smooth_window,
        min_run_fraction=args.min_run_fraction,
        out_path=output,
        title=title,
        num_envs_values=args.num_envs,
    )


if __name__ == "__main__":
    main()
