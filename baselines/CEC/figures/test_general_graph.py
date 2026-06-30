from __future__ import annotations

import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import hydra
from omegaconf import OmegaConf

# ALG_ORDER = ["IPPO", "E3T", "FCP", "CEC", "CEC_POP_ART"]
ALG_ORDER = ["IPPO", "E3T", "FCP", "CEC"]

_ALG_PAT = "|".join(re.escape(a) for a in sorted(ALG_ORDER, key=len, reverse=True))
FNAME_RE = re.compile(
    rf"^(?P<alg>{_ALG_PAT})_(?P<map>.+)_9_XP_results\.csv$"
)

# IPPO, E3T, FCP, CEC — "5 Heldout Grids" style (red / purple / gold / forest green)
# ALG_COLORS = ["#d62728", "#7b126b", "#e3a21a", "#117733", "#008000"]
ALG_COLORS = ["#d62728", "#7b126b", "#e3a21a", "#117733"]

MAP_ORDER = [
    "asymm_advantages",
    "coord_ring",
    "counter_circuit",
    "cramped_room",
    "forced_coord",
]

MAP_LABEL_KO = {
    "asymm_advantages": "asymm advantages",
    "coord_ring": "coord ring",
    "counter_circuit": "counter circuit",
    "cramped_room": "cramped room",
    "forced_coord": "forced coord",
}


def _alg_color_map() -> dict[str, str]:
    return {alg: c for alg, c in zip(ALG_ORDER, ALG_COLORS)}


def load_grid(config, results_dir: Path) -> pd.DataFrame:
    rows = []
    for path in sorted(results_dir.glob("*_9_XP_results.csv")):
        m = FNAME_RE.match(path.name)
        if not m:
            continue
        alg, map_name = m.group("alg"), m.group("map")
        df = pd.read_csv(path)
        if "reward" not in df.columns:
            continue
        if config["XP_ONLY"]:
            df = df[df["seed_1"] != df["seed_2"]]
        r = df["reward"].astype(float)
        mean_r = float(r.mean())
        # 시드 쌍 단위로 묶어 SEM(standard error of mean) 계산
        if "seed_1" in df.columns and "seed_2" in df.columns:
            g = df.groupby(["seed_1", "seed_2"], as_index=False)["reward"].mean()
            n_pairs = len(g)
            std_across_pairs = float(g["reward"].std(ddof=1)) if n_pairs > 1 else 0.0
            sem_pairs = std_across_pairs / np.sqrt(n_pairs) if n_pairs > 1 else 0.0
        else:
            n_pairs = len(r)
            std_across_pairs = float(r.std(ddof=1)) if n_pairs > 1 else 0.0
            sem_pairs = std_across_pairs / np.sqrt(n_pairs) if n_pairs > 1 else 0.0
        rows.append(
            {
                "algorithm": alg,
                "map": map_name,
                "mean_reward": mean_r,
                "sem_pairs": sem_pairs,
                "n_rows": len(df),
            }
        )
    return pd.DataFrame(rows)


def plot_per_map(grid: pd.DataFrame, out_path: Path, title_prefix: str = "") -> None:
    n_maps = len(MAP_ORDER)
    ncols = 3
    nrows = int(np.ceil(n_maps / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.2 * ncols, 3.8 * nrows), squeeze=False)
    axes_flat = axes.ravel()
    colors = _alg_color_map()

    for idx, map_name in enumerate(MAP_ORDER):
        ax = axes_flat[idx]
        sub = grid[grid["map"] == map_name]
        if sub.empty:
            ax.set_visible(False)
            continue
        x = np.arange(len(ALG_ORDER))
        means = []
        errs = []
        for alg in ALG_ORDER:
            row = sub[sub["algorithm"] == alg]
            if row.empty:
                means.append(0.0)
                errs.append(0.0)
            else:
                means.append(row["mean_reward"].iloc[0])
                errs.append(row["sem_pairs"].iloc[0])
        ax.bar(
            x,
            means,
            yerr=errs,
            capsize=4,
            color=[colors[a] for a in ALG_ORDER],
            edgecolor="black",
            linewidth=0.6,
            alpha=0.9,
        )
        ax.set_xticks(x)
        ax.set_xticklabels(ALG_ORDER, rotation=15, ha="right")
        label = MAP_LABEL_KO.get(map_name, map_name)
        ax.set_title(f"{title_prefix}{label}")
        ax.set_ylabel("mean reward")
        ax.grid(axis="y", alpha=0.35)
        ax.set_axisbelow(True)

    for j in range(len(MAP_ORDER), len(axes_flat)):
        axes_flat[j].set_visible(False)

    fig.suptitle("Algorithm performance per map (test_general)", fontsize=13, y=1.02)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def plot_overall(grid: pd.DataFrame, out_path: Path) -> None:
    overall = (
        grid.groupby("algorithm", as_index=False)
        .agg(mean_reward=("mean_reward", "mean"), std_maps=("mean_reward", "std"), n_maps=("mean_reward", "count"))
        .set_index("algorithm")
        .reindex(ALG_ORDER)
    )
    means = overall["mean_reward"].values.astype(float)

    errs = (overall["std_maps"] / np.sqrt(overall["n_maps"])).values.astype(float)
    errs = np.nan_to_num(errs, nan=0.0)

    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    x = np.arange(len(ALG_ORDER))
    colors = [_alg_color_map()[a] for a in ALG_ORDER]
    ax.bar(
        x,
        means,
        yerr=errs,
        capsize=5,
        color=colors,
        edgecolor="black",
        linewidth=0.7,
        alpha=0.92,
    )
    ax.set_xticks(x)
    ax.set_xticklabels(ALG_ORDER)
    ax.set_ylabel("mean reward (average over maps)")
    ax.set_title("Overall performance — average over 5 maps (test_general)")
    ax.grid(axis="y", alpha=0.35)
    ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


@hydra.main(version_base=None, config_path="../repro_config", config_name="test_general")
def main(config):
    config = OmegaConf.to_container(config)

    results_path = Path(__file__).resolve().parent.parent / "results" / f"test_general_{config['NUM_MODELS']}" 
    out_path = Path(__file__).resolve().parent / "results" / f"test_general_graph_{config['NUM_MODELS']}"

    grid = load_grid(config, results_path)

    os.makedirs(out_path, exist_ok=True)
    plot_per_map(grid, out_path / "xp_per_map.png")
    plot_overall(grid, out_path / "xp_overall.png")

    pivot = grid.pivot_table(
        index="map", columns="algorithm", values="mean_reward", aggfunc="first"
    )
    pivot = pivot.reindex(index=MAP_ORDER, columns=ALG_ORDER)
    pivot.to_csv(out_path / "xp_per_map_table.csv", encoding="utf-8")
    overall_df = (
        grid.groupby("algorithm", as_index=False)
        .agg(mean_over_maps=("mean_reward", "mean"), std_across_maps=("mean_reward", "std"), n_maps=("mean_reward", "count"))
    )
    overall_df["sem_across_maps"] = overall_df["std_across_maps"] / np.sqrt(overall_df["n_maps"])
    overall_df = (
        overall_df
        .set_index("algorithm")
        .reindex(ALG_ORDER)
        .reset_index()
    )
    overall_df.to_csv(out_path / "xp_overall_table.csv", index=False, encoding="utf-8")

    print(f"Saved: {out_path / 'xp_per_map.png'}")
    print(f"Saved: {out_path / 'xp_overall.png'}")
    print(f"Saved: {out_path / 'xp_per_map_table.csv'}")
    print(f"Saved: {out_path / 'xp_overall_table.csv'}")


if __name__ == "__main__":
    main()
