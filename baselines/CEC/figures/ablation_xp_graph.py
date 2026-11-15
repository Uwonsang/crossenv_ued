"""Ablation line plots: CEC / Decoupled / DAAC / DCEC across 8K-65K batch size.

(a) cross-play on the 5 fixed tasks (xp_results)
(b) cross-play on 100 procedurally generated tasks (pcg_xp_results)

Self-play pairs (seed_1 == seed_2) are excluded. Each training seed (seed_1)
is averaged over partners, trajectories and layouts; error bars are the SEM
across training seeds.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd


def _default_results_root() -> Path:
    for root in (Path("/app/nas/models/ICRL"),
                 Path("/mnt/nas/wonsang/crossenv_ued/models/ICRL")):
        if root.is_dir():
            return root
    return Path("/app/nas/models/ICRL")


# (result directory name, display label, color);
# CEC/DCEC colors match test_general_graph(_scaling).py.
METHODS = [
    ("CEC", "CEC", "#117733"),
    ("CEC_decouple", "+ Decoupled", "#009E73"),
    ("CEC_DAAC", "+ Adv.", "#CC79A7"),
    ("CEC_IDAAC", "DCEC", "#0072B2"),
]
BATCHES = [(32, "8K"), (64, "16K"), (128, "32K"), (256, "65K")]
LAYOUTS = [
    "asymm_advantages_9", "coord_ring_9", "counter_circuit_9",
    "cramped_room_9", "forced_coord_9",
]


def _seed_rewards(df: pd.DataFrame) -> pd.Series:
    df = df[df["seed_1"] != df["seed_2"]]
    return df.groupby("seed_1")["reward"].mean()


def load_fixed(root: Path, method: str, num_envs: int) -> pd.Series | None:
    per_layout = []
    for layout in LAYOUTS:
        path = (root / "xp_results" / method / f"envs{num_envs}"
                / f"{method}_{layout}_XP_results.csv")
        if not path.exists():
            print(f"Missing: {path}")
            return None
        per_layout.append(_seed_rewards(pd.read_csv(path)))
    # Average layouts within each training seed.
    return pd.concat(per_layout, axis=1).mean(axis=1)


def load_pcg(root: Path, method: str, num_envs: int) -> pd.Series | None:
    path = (root / "pcg_xp_results" / method / f"envs{num_envs}"
            / f"{method}_envs{num_envs}_PCG_XP_results.csv")
    if not path.exists():
        print(f"Missing: {path}")
        return None
    return _seed_rewards(pd.read_csv(path))


def summarize(root: Path, loader) -> pd.DataFrame:
    rows = []
    for method, label, _ in METHODS:
        for num_envs, batch in BATCHES:
            seeds = loader(root, method, num_envs)
            if seeds is None:
                mean = sem = np.nan
                n = 0
            else:
                n = len(seeds)
                mean = float(seeds.mean())
                sem = float(seeds.std(ddof=1) / np.sqrt(n)) if n > 1 else 0.0
            rows.append({"method": label, "batch": batch, "mean": mean,
                         "sem": sem, "n_seeds": n})
    return pd.DataFrame(rows)


# Same style as plot_overall_line in test_general_graph_scaling.py
SCALING_RC = {
    "font.size": 20,
    "axes.titlesize": 26,
    "axes.labelsize": 24,
    "xtick.labelsize": 22,
    "ytick.labelsize": 22,
    "legend.fontsize": 20,
}


# Display names in legends and bar ticks (cumulative ablation); data keeps METHODS labels.
DISPLAY_LABELS = {"DCEC": "+ Invariance\n(DCEC)"}


def plot_line(panels, out_path: Path) -> None:
    """panels: [(summary, title), ...] drawn side by side with one legend."""
    x = np.arange(len(BATCHES))
    with plt.rc_context(SCALING_RC):
        fig, axes = plt.subplots(1, len(panels), figsize=(18.0, 6.5))
        for ax, (summary, title) in zip(axes, panels):
            for _, label, color in METHODS:
                sub = summary[summary["method"] == label].set_index("batch")
                means = np.array([sub.loc[b, "mean"] for _, b in BATCHES])
                sems = np.array([sub.loc[b, "sem"] for _, b in BATCHES])
                ax.plot(x, means, color=color, marker="o", markersize=9,
                        linewidth=2.8)
                # Standard error across seeds as a shaded band.
                ax.fill_between(x, means - sems, means + sems, color=color,
                                alpha=0.2, linewidth=0)
            ax.set_xticks(x)
            ax.set_xticklabels([batch for _, batch in BATCHES])
            ax.set_xlabel("Batch Size")
            ax.set_ylabel("XP Reward")
            ax.set_title(title, pad=12)
            ax.grid(alpha=0.25)
            ax.set_axisbelow(True)
        fig.tight_layout()
        # One horizontal legend centered below the panels, thick line handles only.
        handles = [Line2D([], [], color=color, linewidth=6)
                   for _, _, color in METHODS]
        legend = fig.legend(
            handles,
            [DISPLAY_LABELS.get(label, label).replace("\n", " ")
             for _, label, _ in METHODS],
            loc="upper center", bbox_to_anchor=(0.5, 0.0), ncol=len(METHODS),
            frameon=True, handlelength=1.6, columnspacing=1.8,
        )
        legend.get_frame().set_edgecolor("#dddddd")
        fig.savefig(out_path, bbox_inches="tight")
        plt.close(fig)
    print(f"Saved {out_path}")


# Same style as plot_overall in test_general_graph.py / test_general_graph_pcg.py
OVERALL_RC = {
    "font.family": "DejaVu Sans",
    "font.size": 24,
    "axes.titlesize": 28,
    "axes.labelsize": 24,
    "xtick.labelsize": 24,
    "ytick.labelsize": 24,
}


def plot_bar_overall(summary: pd.DataFrame, title: str, batch: str,
                     out_path: Path, ymax: float) -> None:
    sub = summary[summary["batch"] == batch].set_index("method")
    labels = [label for _, label, _ in METHODS]
    with plt.rc_context(OVERALL_RC):
        fig, ax = plt.subplots(figsize=(10.0, 5.694))
        x = np.arange(len(METHODS))
        ax.bar(
            x, sub.loc[labels, "mean"], yerr=sub.loc[labels, "sem"],
            capsize=5, color=[color for _, _, color in METHODS],
            edgecolor="none", linewidth=0,
            alpha=0.8,
        )
        ax.set_xticks(x)
        ax.set_xticklabels([DISPLAY_LABELS.get(l, l) for l in labels],
                           rotation=0, ha="center", multialignment="center")
        ax.tick_params(axis="x", pad=12)
        ax.set_ylabel("XP Reward")
        ax.set_title(title, fontweight="normal", pad=16)
        ax.set_ylim(0, ymax)
        ax.grid(False)
        fig.tight_layout()
        fig.savefig(out_path)
        plt.close(fig)
    print(f"Saved {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", type=Path,
                        default=_default_results_root())
    parser.add_argument(
        "--output-dir", type=Path,
        default=Path(__file__).parent / "results" / "ablation_xp")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    fixed = summarize(args.results_root, load_fixed)
    pcg = summarize(args.results_root, load_pcg)
    fixed.assign(eval="fixed_5_tasks").pipe(
        lambda d: pd.concat([d, pcg.assign(eval="pcg_100_tasks")])
    ).to_csv(args.output_dir / "ablation_xp_summary.csv", index=False)
    print(fixed.to_string(index=False))
    print(pcg.to_string(index=False))

    plot_line(
        [(fixed, "5 Fixed Tasks"), (pcg, "100 Procedurally Generated Tasks")],
        args.output_dir / "ablation_xp_line.pdf",
    )

    # One bar plot per batch size; shared y-axis per evaluation for comparison.
    for summary, title, name in (
        (fixed, "(a) 5 Fixed Tasks", "fixed"),
        (pcg, "(b) 100 Procedurally Generated Tasks", "pcg"),
    ):
        ymax = float((summary["mean"] + summary["sem"]).max()) * 1.08
        for _, batch in BATCHES:
            plot_bar_overall(summary, title, batch,
                             args.output_dir / f"ablation_xp_bar_{batch}_{name}.pdf",
                             ymax)


if __name__ == "__main__":
    main()
