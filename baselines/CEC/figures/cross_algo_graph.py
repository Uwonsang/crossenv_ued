"""Cross-algorithm XP result heatmaps — directional (role-labeled) and symmetric versions."""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ──────────────────────────────────────────────
# Config
# ──────────────────────────────────────────────
RESULTS_DIR = Path(__file__).parent.parent / "results" / "cross_algo"
SAVE_DIR = Path(__file__).parent / "results" / "cross_algo_graph"
SAVE_DIR.mkdir(parents=True, exist_ok=True)

ALGO_RENAME = {
    "ik":          "CEC",
    "sk":          "IPPO",
    "fcp":         "FCP",
    "e3t":         "E3T",
    "ik_pop_art":  "CEC-PopArt",
}
ALGO_ORDER = ["IPPO", "E3T", "FCP", "CEC", "CEC-PopArt"]

LAYOUT_ORDER = [
    "asymm_advantages_9",
    "coord_ring_9",
    "counter_circuit_9",
    "cramped_room_9",
    "forced_coord_9",
]
LAYOUT_LABEL = {
    "asymm_advantages_9": "Asymm Advantages",
    "coord_ring_9":        "Coord Ring",
    "counter_circuit_9":   "Counter Circuit",
    "cramped_room_9":      "Cramped Room",
    "forced_coord_9":      "Forced Coord",
}


# ──────────────────────────────────────────────
# Data loading / transforms
# ──────────────────────────────────────────────
def load_pivot(csv_path: Path) -> pd.DataFrame:
    """Return (algo_1 × algo_2) mean-reward pivot, with display names, ordered."""
    df = pd.read_csv(csv_path)
    df["algo_1"] = df["algo_1"].map(ALGO_RENAME)
    df["algo_2"] = df["algo_2"].map(ALGO_RENAME)
    df = df.dropna(subset=["algo_1", "algo_2"])

    pivot = df.groupby(["algo_1", "algo_2"])["reward"].mean().unstack("algo_2")
    present = [a for a in ALGO_ORDER if a in pivot.index]
    return pivot.reindex(index=present, columns=present)


def symmetrize_pivot(pivot: pd.DataFrame) -> pd.DataFrame:
    """Average (i,j) and (j,i) to produce a symmetric matrix."""
    p = pivot.copy().astype(float)
    return (p + p.T) / 2


def normalize_pivot(pivot: pd.DataFrame) -> pd.DataFrame:
    """Divide by this pivot's own max value."""
    return pivot.copy().astype(float) / pivot.max().max()


# ──────────────────────────────────────────────
# Plotting
# ──────────────────────────────────────────────
def plot_heatmap(ax: plt.Axes, data: pd.DataFrame, title: str,
                 xlabel: str, ylabel: str, vmin=0.0, vmax=1.0):
    algos = list(data.index)
    mat = data.values.astype(float)

    im = ax.imshow(mat, cmap="viridis", vmin=vmin, vmax=vmax, aspect="auto")
    ax.set_xticks(range(len(algos)))
    ax.set_xticklabels(algos, fontsize=9)
    ax.set_yticks(range(len(algos)))
    ax.set_yticklabels(algos, fontsize=9)
    ax.set_xlabel(xlabel, fontsize=9)
    ax.set_ylabel(ylabel, fontsize=9)
    ax.set_title(title, fontsize=11, fontweight="bold")

    for i in range(len(algos)):
        for j in range(len(algos)):
            v = mat[i, j]
            if not np.isnan(v):
                color = "white" if v < (vmin + vmax) * 0.55 else "black"
                ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                        fontsize=9, color=color)
    return im


def save_figures(pivots_raw: dict, tag: str, xlabel: str, ylabel: str, suptitle: str):
    """Generate per-layout + overall figures for a given pivot set."""
    pivots_norm = {layout: normalize_pivot(p) for layout, p in pivots_raw.items()}

    # per-layout
    n = len(pivots_norm)
    fig, axes = plt.subplots(1, n, figsize=(4.5 * n, 4.5))
    if n == 1:
        axes = [axes]

    for ax, (layout, pivot) in zip(axes, pivots_norm.items()):
        im = plot_heatmap(ax, pivot, LAYOUT_LABEL[layout],
                          xlabel=xlabel, ylabel=ylabel)

    fig.subplots_adjust(right=0.88, wspace=0.4)
    cbar_ax = fig.add_axes([0.90, 0.15, 0.015, 0.7])
    cbar = fig.colorbar(im, cax=cbar_ax)
    cbar.set_label("Normalized Reward", fontsize=10)
    fig.suptitle(f"{suptitle} — per Layout", fontsize=13, y=1.01)

    out = SAVE_DIR / f"cross_algo_per_layout_{tag}.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved: {out}")
    plt.close(fig)

    # overall: mean of raw pivots, then normalize
    stacked = np.stack([p.values for p in pivots_raw.values()], axis=0)
    mean_mat = np.nanmean(stacked, axis=0)
    ref = next(iter(pivots_raw.values()))
    overall_pivot = normalize_pivot(
        pd.DataFrame(mean_mat, index=ref.index, columns=ref.columns)
    )

    fig2, ax2 = plt.subplots(figsize=(5, 4.5))
    im2 = plot_heatmap(ax2, overall_pivot, "Overall (Mean across Layouts)",
                       xlabel=xlabel, ylabel=ylabel)
    fig2.subplots_adjust(right=0.85)
    cbar_ax2 = fig2.add_axes([0.87, 0.15, 0.025, 0.7])
    cbar2 = fig2.colorbar(im2, cax=cbar_ax2)
    cbar2.set_label("Normalized Reward", fontsize=10)
    fig2.suptitle(f"{suptitle} — Overall", fontsize=13)

    out2 = SAVE_DIR / f"cross_algo_overall_{tag}.png"
    fig2.savefig(out2, dpi=150, bbox_inches="tight")
    print(f"Saved: {out2}")
    plt.close(fig2)


# ──────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────
def main():
    csv_files = {
        layout: RESULTS_DIR / f"{layout}_cross_algo_eval_onIK.csv"
        for layout in LAYOUT_ORDER
    }
    missing = [l for l, p in csv_files.items() if not p.exists()]
    if missing:
        print(f"Warning: missing CSVs for {missing}")

    pivots_raw = {}
    for layout, path in csv_files.items():
        if path.exists():
            pivots_raw[layout] = load_pivot(path)

    if not pivots_raw:
        print("No data found.")
        return

    # Version 1: directional — rows = Agent 0, columns = Agent 1
    save_figures(
        pivots_raw,
        tag="directional",
        xlabel="Algorithm 2 (Agent 1)",
        ylabel="Algorithm 1 (Agent 0)",
        suptitle="Cross-Algorithm XP (Directional)",
    )

    # Version 2: symmetric — average both role assignments
    pivots_sym = {layout: symmetrize_pivot(p) for layout, p in pivots_raw.items()}
    save_figures(
        pivots_sym,
        tag="symmetric",
        xlabel="Algorithm",
        ylabel="Algorithm",
        suptitle="Cross-Algorithm XP (Symmetric)",
    )


if __name__ == "__main__":
    main()
