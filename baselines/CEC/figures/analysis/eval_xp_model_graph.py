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
X_AXIS = "env_step"
SMOOTH_WINDOW = 1
MIN_RUN_FRACTION = 0.5

SAVE_DIR = Path(__file__).parent.parent / "generated" / "eval_xp_model_graph"
MODEL_COLORS = ["#7f56d9", "#e66b2e", "#ef3e4a", "#df70d6"]


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--entity", default=ENTITY)
    parser.add_argument("--project", default=PROJECT)
    parser.add_argument("--model-names", nargs="+", default=MODEL_NAMES)
    parser.add_argument(
        "--x-axis", default=X_AXIS, choices=["env_step", "update_step"]
    )
    parser.add_argument("--smooth-window", type=int, default=SMOOTH_WINDOW)
    parser.add_argument("--min-run-fraction", type=float, default=MIN_RUN_FRACTION)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def fetch_model_histories(entity: str, project: str, model_names):
    api = wandb.Api()
    runs = api.runs(
        f"{entity}/{project}",
        filters={"config.model_name": {"$in": list(model_names)}},
    )

    histories = []
    run_counts = {model_name: 0 for model_name in model_names}
    for run in runs:
        model_name = run.config.get("model_name")
        if model_name not in run_counts:
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
        history = history[
            ["env_step", "update_step", "eval_xp/mean"]
        ].dropna()
        history["model_name"] = model_name
        history["run_id"] = run.id
        histories.append(history)
        run_counts[model_name] += 1

    if not histories:
        raise RuntimeError(
            "No eval_xp/mean data found for the requested model names."
        )

    return pd.concat(histories, ignore_index=True), run_counts


def aggregate_histories(histories, model_name: str, x_axis: str):
    model_history = histories[histories["model_name"] == model_name]
    per_run = (
        model_history.groupby(["run_id", x_axis], as_index=False)["eval_xp/mean"]
        .mean()
    )
    if per_run.empty:
        return pd.DataFrame(columns=[x_axis, "mean", "count"])

    # Evaluation intervals can differ with NUM_ENVS. Interpolate each run onto
    # one common grid so the model average does not alternate between subsets
    # of runs at adjacent x positions.
    grid = np.linspace(per_run[x_axis].min(), per_run[x_axis].max(), 500)
    interpolated = []
    for _, run_history in per_run.groupby("run_id"):
        run_history = run_history.sort_values(x_axis)
        x = run_history[x_axis].to_numpy(dtype=float)
        y = run_history["eval_xp/mean"].to_numpy(dtype=float)
        values = np.interp(grid, x, y, left=np.nan, right=np.nan)
        interpolated.append(values)

    values = pd.DataFrame(np.stack(interpolated), columns=grid)
    return pd.DataFrame(
        {
            x_axis: grid,
            "mean": values.mean(axis=0).to_numpy(),
            "count": values.count(axis=0).to_numpy(),
        }
    )


def plot_eval_xp_by_model(
    histories,
    run_counts,
    model_names,
    x_axis: str,
    smooth_window: int,
    min_run_fraction: float,
    out_path: Path,
):
    if smooth_window < 1:
        raise ValueError("smooth_window must be at least 1")
    if not 0.0 < min_run_fraction <= 1.0:
        raise ValueError("min_run_fraction must be in (0, 1]")

    fig, ax = plt.subplots(figsize=(9, 5.5))
    plotted = 0
    for index, model_name in enumerate(model_names):
        stats = aggregate_histories(histories, model_name, x_axis)
        minimum_runs = max(
            1, int(np.ceil(run_counts[model_name] * min_run_fraction))
        )
        stats = stats[stats["count"] >= minimum_runs]
        if stats.empty:
            print(f"Skipping {model_name}: no usable runs")
            continue

        mean = stats["mean"].rolling(smooth_window, min_periods=1).mean()
        color = MODEL_COLORS[index % len(MODEL_COLORS)]
        ax.plot(
            stats[x_axis],
            mean,
            color=color,
            linewidth=2,
            label=f"{model_name} (n={run_counts[model_name]})",
        )
        plotted += 1

    if plotted == 0:
        raise RuntimeError("No model curves could be plotted.")

    ax.set_xlabel(
        "Environment Steps" if x_axis == "env_step" else "Update Step"
    )
    ax.set_ylabel("Eval XP Return")
    ax.set_title("BC Cross-Play Evaluation by Model")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=9)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


def main():
    args = parse_args()
    histories, run_counts = fetch_model_histories(
        args.entity, args.project, args.model_names
    )
    output = args.output or (
        SAVE_DIR / f"eval_xp_by_model_{args.x_axis}.png"
    )
    plot_eval_xp_by_model(
        histories=histories,
        run_counts=run_counts,
        model_names=args.model_names,
        x_axis=args.x_axis,
        smooth_window=args.smooth_window,
        min_run_fraction=args.min_run_fraction,
        out_path=output,
    )


if __name__ == "__main__":
    main()
