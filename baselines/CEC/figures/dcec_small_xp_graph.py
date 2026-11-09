"""Ablation bar charts: DCEC (CEC_IDAAC) vs DCEC-small (CEC_IDAAC_small).

(a) cross-play on the 5 fixed tasks (xp_results)
(b) cross-play on 100 procedurally generated tasks (pcg_xp_results)

Same aggregation as ablation_xp_graph.py: self-play pairs excluded, each
training seed averaged over partners/trajectories/layouts, SEM across seeds.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ablation_xp_graph import _default_results_root, load_fixed, load_pcg

METHODS = [  # (result directory name, display label, color)
    ("CEC_IDAAC", "DCEC", "#0072B2"),
    ("CEC_IDAAC_small", "DCEC (small)", "#56B4E9"),
]
BATCHES = [(256, "65K")]  # CEC_IDAAC_small ckpts exist only for 256 envs


def summarize(root: Path, loader) -> pd.DataFrame:
    rows = []
    for method, label, _ in METHODS:
        for num_envs, batch in BATCHES:
            seeds = loader(root, method, num_envs)
            n = 0 if seeds is None else len(seeds)
            mean = float(seeds.mean()) if n else np.nan
            sem = float(seeds.std(ddof=1) / np.sqrt(n)) if n > 1 else 0.0
            rows.append({"method": label, "batch": batch, "mean": mean,
                         "sem": sem, "n_seeds": n})
    return pd.DataFrame(rows)


def plot_panel(ax, summary: pd.DataFrame, title: str) -> None:
    x = np.arange(len(BATCHES))
    width = 0.8 / len(METHODS)
    for m, (_, label, color) in enumerate(METHODS):
        sub = summary[summary["method"] == label].set_index("batch")
        means = [sub.loc[batch, "mean"] for _, batch in BATCHES]
        sems = [sub.loc[batch, "sem"] for _, batch in BATCHES]
        ax.bar(
            x + (m - (len(METHODS) - 1) / 2) * width, np.nan_to_num(means),
            width, yerr=np.nan_to_num(sems), capsize=3, color=color,
            edgecolor="black", linewidth=0.7, label=label,
            error_kw={"elinewidth": 1.0},
        )
    ax.set_xticks(x)
    ax.set_xticklabels([f"Batch {batch}" for _, batch in BATCHES])
    ax.set_ylabel("XP Reward")
    ax.set_title(title)
    ax.grid(axis="y", alpha=0.35)
    ax.set_axisbelow(True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", type=Path,
                        default=_default_results_root())
    parser.add_argument(
        "--output-dir", type=Path,
        default=Path(__file__).parent / "results" / "dcec_small_xp")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    fixed = summarize(args.results_root, load_fixed)
    pcg = summarize(args.results_root, load_pcg)
    pd.concat([fixed.assign(eval="fixed_5_tasks"),
               pcg.assign(eval="pcg_100_tasks")]).to_csv(
        args.output_dir / "dcec_small_xp_summary.csv", index=False)
    print(fixed.to_string(index=False))
    print(pcg.to_string(index=False))

    fig, axes = plt.subplots(1, 2, figsize=(9, 4.2))
    plot_panel(axes[0], fixed, "(a) 5 Fixed Tasks")
    plot_panel(axes[1], pcg, "(b) 100 Procedurally Generated Tasks")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=len(METHODS),
               frameon=False, bbox_to_anchor=(0.5, 1.04))
    fig.tight_layout()
    for ext in ("pdf", "png"):
        out = args.output_dir / f"dcec_small_xp_bar.{ext}"
        fig.savefig(out, bbox_inches="tight", dpi=200)
        print(f"Saved {out}")


if __name__ == "__main__":
    main()
