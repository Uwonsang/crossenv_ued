"""Plot single-seed or aggregated XP environment scaling curves."""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from eval_xp_model_graph import fetch_model_histories


ENTITY = "overcooked_ai"
PROJECT = "crossenv_ICLR"
MODEL_NAMES = ["CEC", "CEC_IDAAC_POP"]
NUM_ENVS_VALUES = [32, 64, 128, 256]
SEED = 1
OUTPUT_DIR = (
    Path(__file__).parent.parent
    / "results"
    / "eval_xp_scaling"
)
NUM_ENVS_COLORS = {
    32: "#4878d0",
    64: "#6acc64",
    128: "#d65f5f",
    256: "#956cb4",
}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--entity", default=ENTITY)
    parser.add_argument("--project", default=PROJECT)
    seed_group = parser.add_mutually_exclusive_group()
    seed_group.add_argument("--seed", type=int)
    seed_group.add_argument("--seeds", type=int, nargs="+")
    parser.add_argument("--smooth-window", type=int, default=1)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def aggregate_seeds(curve, seeds):
    per_seed = (
        curve.groupby(["seed", "env_step"], as_index=False)["eval_xp/mean"]
        .mean()
    )
    grid = np.linspace(per_seed["env_step"].min(), per_seed["env_step"].max(), 500)
    interpolated = []
    for seed in seeds:
        seed_history = per_seed[per_seed["seed"] == seed].sort_values("env_step")
        if seed_history.empty:
            continue
        interpolated.append(
            np.interp(
                grid,
                seed_history["env_step"].to_numpy(dtype=float),
                seed_history["eval_xp/mean"].to_numpy(dtype=float),
                left=np.nan,
                right=np.nan,
            )
        )

    values = pd.DataFrame(np.stack(interpolated), columns=grid)
    valid = values.count(axis=0) == len(seeds)
    values = values.loc[:, valid]
    return pd.DataFrame(
        {
            "env_step": values.columns.to_numpy(dtype=float),
            "mean": values.mean(axis=0).to_numpy(),
            "minimum": values.min(axis=0).to_numpy(),
            "maximum": values.max(axis=0).to_numpy(),
        }
    )


def plot_scaling(histories, seeds, smooth_window, output):
    if smooth_window < 1:
        raise ValueError("smooth_window must be at least 1")

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2), sharex=True, sharey=True)
    missing = []
    for ax, model_name in zip(axes, MODEL_NAMES):
        model_history = histories[histories["model_name"] == model_name]
        for num_envs in NUM_ENVS_VALUES:
            curve = model_history[model_history["num_envs"] == num_envs]
            available_seeds = set(curve["seed"].unique())
            missing_seeds = [seed for seed in seeds if seed not in available_seeds]
            if missing_seeds:
                missing.append(
                    f"{model_name}/NUM_ENVS={num_envs}/seeds={missing_seeds}"
                )
                continue

            stats = aggregate_seeds(curve, seeds)
            mean = stats["mean"].rolling(
                smooth_window, min_periods=1
            ).mean()
            ax.plot(
                stats["env_step"],
                mean,
                color=NUM_ENVS_COLORS[num_envs],
                linewidth=2,
                label=f"NUM_ENVS={num_envs}",
            )
            if len(seeds) > 1:
                minimum = stats["minimum"].rolling(
                    smooth_window, min_periods=1
                ).mean()
                maximum = stats["maximum"].rolling(
                    smooth_window, min_periods=1
                ).mean()
                ax.fill_between(
                    stats["env_step"],
                    minimum,
                    maximum,
                    color=NUM_ENVS_COLORS[num_envs],
                    alpha=0.18,
                    linewidth=0,
                )

        ax.set_title(model_name)
        ax.set_xlabel("Environment Steps")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=9)

    axes[0].set_ylabel("Eval XP Return")
    seed_label = ",".join(str(seed) for seed in seeds)
    title_suffix = "mean with min–max range" if len(seeds) > 1 else "single seed"
    fig.suptitle(
        f"XP Environment Scaling (seeds={seed_label}; {title_suffix})",
        fontsize=15,
    )
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {output}")
    if missing:
        print("Missing runs: " + ", ".join(missing))


def main():
    args = parse_args()
    seeds = tuple(args.seeds or ([args.seed] if args.seed is not None else [SEED]))
    seed_label = "_".join(f"seed{seed}" for seed in seeds)
    if len(seeds) > 1:
        seed_label += "_aggregated"
    output = args.output or (
        OUTPUT_DIR / f"xp_environment_scaling_{seed_label}.png"
    )
    histories, _ = fetch_model_histories(
        entity=args.entity,
        project=args.project,
        model_names=MODEL_NAMES,
        num_envs=NUM_ENVS_VALUES,
        seeds_by_model={name: seeds for name in MODEL_NAMES},
    )
    plot_scaling(
        histories=histories,
        seeds=seeds,
        smooth_window=args.smooth_window,
        output=output,
    )


if __name__ == "__main__":
    main()
