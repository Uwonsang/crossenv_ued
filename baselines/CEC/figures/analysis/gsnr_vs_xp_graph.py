"""Relate per-run gradient GSNR to cross-seed XP reward.

Inputs:
  * ``--gsnr-runs-csv``: ``gradient_diagnostics_full_window_runs.csv`` written by
    ``gradient_diagnostics_num_envs_graph.py`` (one training-averaged value per
    run and metric).
  * ``--xp-root`` / ``--pcg-xp-root``: output directories of
    ``bash_test_general.sh`` / ``bash_test_general_pcg.sh`` (origin/gradient),
    laid out as ``<root>/<MODEL>/envs<N>/...XP_results.csv``.

A run's XP is the mean reward of every cross-seed pairing it takes part in
(both seats, partner restricted to ``--seeds``, self-pairs excluded).
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

from gradient_diagnostics_num_envs_graph import (
    ALGORITHM_COLORS,
    algorithm_display_name,
    batch_size_label,
)

FIXED_LAYOUTS = (
    "asymm_advantages_9",
    "coord_ring_9",
    "counter_circuit_9",
    "cramped_room_9",
    "forced_coord_9",
)
NUM_ENVS_MARKERS = {32: "o", 64: "s", 128: "^", 256: "D"}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gsnr-runs-csv", type=Path, required=True)
    parser.add_argument("--xp-root", type=Path, required=True)
    parser.add_argument("--pcg-xp-root", type=Path, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", required=True)
    parser.add_argument("--algorithms", nargs="+", default=("CEC", "CEC_IDAAC"))
    parser.add_argument("--num-envs", type=int, nargs="+", default=(32, 64, 128, 256))
    parser.add_argument(
        "--metric",
        default="sample_gradient_gsnr/value_parameterwise_gsnr_mean",
    )
    parser.add_argument("--rollout-steps", type=int, default=256)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def per_seed_xp(df: pd.DataFrame, seeds) -> dict:
    """Mean cross-seed reward for each seed over pairs within ``seeds``."""
    df = df[
        df["seed_1"].isin(seeds)
        & df["seed_2"].isin(seeds)
        & (df["seed_1"] != df["seed_2"])
    ]
    return {
        seed: float(df.loc[(df["seed_1"] == seed) | (df["seed_2"] == seed), "reward"].mean())
        for seed in seeds
        if ((df["seed_1"] == seed) | (df["seed_2"] == seed)).any()
    }


def load_xp(args) -> pd.DataFrame:
    rows = []
    for algorithm in args.algorithms:
        for num_envs in args.num_envs:
            fixed_dir = args.xp_root / algorithm / f"envs{num_envs}"
            layout_xp = []
            for layout in FIXED_LAYOUTS:
                path = fixed_dir / f"{algorithm}_{layout}_XP_results.csv"
                if not path.exists():
                    print(f"Missing fixed-task XP: {path}")
                    continue
                layout_xp.append(per_seed_xp(pd.read_csv(path), args.seeds))
            pcg_path = (
                args.pcg_xp_root / algorithm / f"envs{num_envs}"
                / f"{algorithm}_envs{num_envs}_PCG_XP_results.csv"
            )
            pcg_xp = (
                per_seed_xp(pd.read_csv(pcg_path), args.seeds)
                if pcg_path.exists() else {}
            )
            if not pcg_path.exists():
                print(f"Missing PCG XP: {pcg_path}")
            for seed in args.seeds:
                fixed_values = [xp[seed] for xp in layout_xp if seed in xp]
                rows.append({
                    "algorithm": algorithm,
                    "num_envs": num_envs,
                    "seed": seed,
                    # Equal weight per fixed layout; NaN unless all 5 exist.
                    "xp_fixed": (
                        float(np.mean(fixed_values))
                        if len(fixed_values) == len(FIXED_LAYOUTS) else np.nan
                    ),
                    "xp_pcg": pcg_xp.get(seed, np.nan),
                })
    return pd.DataFrame(rows)


def correlations(df: pd.DataFrame, xp_column: str) -> dict:
    df = df.dropna(subset=["gsnr", xp_column])
    result = {"n_runs": len(df)}
    if len(df) >= 3:
        result["pooled_spearman"] = stats.spearmanr(df["gsnr"], df[xp_column])
        # Remove algorithm x batch-size effects: seed-level residuals only.
        grouped = df.groupby(["algorithm", "num_envs"])
        residual_gsnr = df["gsnr"] - grouped["gsnr"].transform("mean")
        residual_xp = df[xp_column] - grouped[xp_column].transform("mean")
        result["within_condition_pearson"] = stats.pearsonr(
            residual_gsnr, residual_xp
        )
    condition = df.groupby(["algorithm", "num_envs"])[["gsnr", xp_column]].mean()
    if len(condition) >= 3:
        result["condition_mean_spearman"] = stats.spearmanr(
            condition["gsnr"], condition[xp_column]
        )
    result["n_conditions"] = len(condition)
    return result


def format_correlations(result: dict) -> str:
    lines = [f"runs={result['n_runs']}, conditions={result['n_conditions']}"]
    for key, label in (
        ("pooled_spearman", "pooled Spearman"),
        ("condition_mean_spearman", "condition-mean Spearman"),
        ("within_condition_pearson", "within-condition Pearson"),
    ):
        if key in result:
            statistic, p_value = result[key]
            lines.append(f"{label}: {statistic:+.2f} (p={p_value:.3f})")
    return "\n".join(lines)


def plot(df: pd.DataFrame, args, results: dict):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for ax, (xp_column, title) in zip(
        axes,
        (("xp_fixed", "5 Fixed Tasks"), ("xp_pcg", "100 Procedurally Generated Tasks")),
    ):
        for algorithm in args.algorithms:
            for num_envs in args.num_envs:
                points = df[
                    (df["algorithm"] == algorithm) & (df["num_envs"] == num_envs)
                ].dropna(subset=["gsnr", xp_column])
                if points.empty:
                    continue
                ax.scatter(
                    points["gsnr"],
                    points[xp_column],
                    color=ALGORITHM_COLORS.get(algorithm),
                    marker=NUM_ENVS_MARKERS.get(num_envs, "o"),
                    s=45,
                    alpha=0.85,
                    edgecolor="white",
                    linewidth=0.6,
                )
        ax.set_title(title)
        ax.set_xlabel("Value-gradient GSNR (training mean)")
        ax.set_ylabel("XP Reward")
        ax.grid(alpha=0.3)
        # Below the axes so the summary never hides data points.
        ax.text(
            0.0, -0.2, format_correlations(results[xp_column]),
            transform=ax.transAxes, va="top", fontsize=8,
        )
    handles = [
        plt.Line2D([], [], color=ALGORITHM_COLORS.get(a), marker="o", linestyle="",
                   label=algorithm_display_name(a))
        for a in args.algorithms
    ] + [
        plt.Line2D([], [], color="gray", marker=NUM_ENVS_MARKERS[n], linestyle="",
                   label=batch_size_label(n, args.rollout_steps))
        for n in args.num_envs
    ]
    fig.legend(handles=handles, loc="upper center", ncol=len(handles),
               bbox_to_anchor=(0.5, 1.04), frameon=False)
    fig.tight_layout()
    output = args.output_dir / "gsnr_vs_xp.pdf"
    fig.savefig(output, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {output}")


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    gsnr = pd.read_csv(args.gsnr_runs_csv)
    gsnr = gsnr[gsnr["metric"] == args.metric][
        ["algorithm", "num_envs", "seed", "mean"]
    ].rename(columns={"mean": "gsnr"})
    if gsnr.duplicated(["algorithm", "num_envs", "seed"]).any():
        raise ValueError("Multiple GSNR runs per algorithm/num_envs/seed.")
    df = load_xp(args).merge(gsnr, on=["algorithm", "num_envs", "seed"], how="left")
    df.to_csv(args.output_dir / "gsnr_vs_xp_runs.csv", index=False)

    results = {column: correlations(df, column) for column in ("xp_fixed", "xp_pcg")}
    for column, result in results.items():
        print(f"== {column}\n{format_correlations(result)}")
    plot(df, args, results)


if __name__ == "__main__":
    main()
