"""Ablation bar charts: CEC / Decoupled / DAAC / DCEC at 8K and 65K batch size.

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
import numpy as np
import pandas as pd


def _default_results_root() -> Path:
    for root in (Path("/app/nas/models/ICRL"),
                 Path("/mnt/nas/wonsang/crossenv_ued/models/ICRL")):
        if root.is_dir():
            return root
    return Path("/app/nas/models/ICRL")


METHODS = [  # (result directory name, display label, color)
    ("CEC", "CEC", "#2F9956"),
    ("CEC_decouple", "Decoupled", "#E69F00"),
    ("CEC_DAAC", "DAAC", "#CC79A7"),
    ("CEC_IDAAC", "DCEC", "#0072B2"),
]
BATCHES = [(32, "8K"), (256, "65K")]
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


def plot_panel(ax, summary: pd.DataFrame, title: str) -> None:
    x = np.arange(len(METHODS))
    width = 0.38
    for b, (_, batch) in enumerate(BATCHES):
        sub = summary[summary["batch"] == batch].set_index("method")
        means = [sub.loc[label, "mean"] for _, label, _ in METHODS]
        sems = [sub.loc[label, "sem"] for _, label, _ in METHODS]
        colors = [color for _, _, color in METHODS]
        ax.bar(
            x + (b - 0.5) * width, np.nan_to_num(means), width,
            yerr=np.nan_to_num(sems), capsize=4, color=colors,
            alpha=0.45 if b == 0 else 1.0, edgecolor="black", linewidth=0.7,
            hatch="//" if b == 0 else None,
        )
    ax.set_xticks(x)
    ax.set_xticklabels([label for _, label, _ in METHODS])
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

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
    plot_panel(axes[0], fixed, "(a) 5 Fixed Tasks")
    plot_panel(axes[1], pcg, "(b) 100 Procedurally Generated Tasks")
    handles = [
        plt.Rectangle((0, 0), 1, 1, facecolor="gray", alpha=0.45,
                      hatch="//", edgecolor="black"),
        plt.Rectangle((0, 0), 1, 1, facecolor="gray", edgecolor="black"),
    ]
    fig.legend(handles, [f"Batch {b}" for _, b in BATCHES], loc="upper center",
               ncol=2, frameon=False, bbox_to_anchor=(0.5, 1.04))
    fig.tight_layout()
    for ext in ("pdf", "png"):
        out = args.output_dir / f"ablation_xp_bar.{ext}"
        fig.savefig(out, bbox_inches="tight", dpi=200)
        print(f"Saved {out}")


if __name__ == "__main__":
    main()
