"""Overall cross-algorithm heatmaps for 100 procedurally generated tasks.

Each input CSV corresponds to one checkpoint-training layout and contains
evaluations on the same 100 PCG tasks.  The overall matrix first averages all
rollouts for each ordered algorithm pair within a checkpoint cohort, then
averages the five cohort matrices with equal weight.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from cross_algo_graph import (
    CEC_IDAAC_BLUE,
    LAYOUT_ORDER,
    load_pivot,
    plot_heatmap,
    symmetrize_pivot,
)


DEFAULT_RESULTS_DIR = Path(
    "/mnt/nas/wonsang/crossenv_ued/models/ICRL/pcg_xp_results_diff_algo"
)
DEFAULT_SAVE_DIR = (
    Path(__file__).resolve().parents[3] / "artifacts" / "cross_algo_graph_pcg"
)


def overall_pivot(results_dir: Path) -> pd.DataFrame:
    paths = [
        results_dir / f"{layout}_pcg_cross_algo_results.csv"
        for layout in LAYOUT_ORDER
    ]
    missing = [path for path in paths if not path.exists()]
    if missing:
        missing_text = "\n".join(str(path) for path in missing)
        raise FileNotFoundError(f"Missing PCG result CSVs:\n{missing_text}")

    pivots = [load_pivot(path) for path in paths]
    reference = pivots[0]
    for path, pivot in zip(paths[1:], pivots[1:]):
        if not pivot.index.equals(reference.index) or not pivot.columns.equals(
            reference.columns
        ):
            raise ValueError(f"Algorithm set/order differs in {path}")

    values = np.stack([pivot.to_numpy(dtype=float) for pivot in pivots])
    return pd.DataFrame(
        np.nanmean(values, axis=0),
        index=reference.index,
        columns=reference.columns,
    )


def save_heatmap(data: pd.DataFrame, tag: str, save_dir: Path) -> None:
    values = data.to_numpy(dtype=float)
    finite = values[np.isfinite(values)]
    if not len(finite):
        raise ValueError("The PCG overall matrix has no finite rewards")

    vmin = min(0.0, float(finite.min()))
    vmax = float(finite.max())
    if np.isclose(vmin, vmax):
        vmax = vmin + 1.0

    fig, ax = plt.subplots(figsize=(14.0, 7.5))
    image = plot_heatmap(
        ax,
        data,
        "Overall",
        xlabel="Agent 1" if tag == "directional" else "Algorithm",
        ylabel="Agent 0" if tag == "directional" else "Algorithm",
        vmin=vmin,
        vmax=vmax,
    )
    # Keep the same blue scale used by the fixed-task cross-play figure.
    image.set_cmap(CEC_IDAAC_BLUE)
    colorbar = fig.colorbar(image, ax=ax, pad=0.04, fraction=0.05)
    colorbar.set_label("Mean Reward", fontsize=22)
    colorbar.ax.tick_params(labelsize=18)
    fig.tight_layout()

    for suffix in ("pdf", "png"):
        output = save_dir / f"cross_algo_pcg_overall_{tag}.{suffix}"
        fig.savefig(output, bbox_inches="tight", dpi=300)
        print(f"Saved: {output}")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    parser.add_argument("--save-dir", type=Path, default=DEFAULT_SAVE_DIR)
    args = parser.parse_args()
    args.save_dir.mkdir(parents=True, exist_ok=True)

    directional = overall_pivot(args.results_dir)
    save_heatmap(directional, "directional", args.save_dir)
    save_heatmap(symmetrize_pivot(directional), "symmetric", args.save_dir)


if __name__ == "__main__":
    main()
