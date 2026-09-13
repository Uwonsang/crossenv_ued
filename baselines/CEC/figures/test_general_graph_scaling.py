from __future__ import annotations

import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import hydra
from omegaconf import OmegaConf

plt.rcParams.update(
    {
        "font.size": 14,
        "axes.titlesize": 16,
        "axes.labelsize": 15,
        "xtick.labelsize": 12,
        "ytick.labelsize": 13,
        "figure.titlesize": 18,
        "legend.fontsize": 13,
    }
)

DEFAULT_XP_RESULTS_DIR = Path(
    "/app/nas/models/ICRL/xp_results"
)
DEFAULT_HUMAN_PROXY_RESULTS_DIR = Path(
    "/app/nas/models/ICRL/human_proxy_results"
)

ALG_ORDER = [
    "CEC_envs32",
    "CEC_IDAAC_envs32",
    "CEC_envs64",
    "CEC_IDAAC_envs64",
    "CEC_envs128",
    "CEC_IDAAC_envs128",
    "CEC_envs256",
    "CEC_IDAAC_envs256",
]

# graph key: (directory relative to xp_results, checkpoint filename prefix)
ALG_SOURCES = {
    "CEC_envs32": ("CEC/envs32", "CEC"),
    "CEC_IDAAC_envs32": ("CEC_IDAAC/envs32", "CEC_IDAAC"),
    "CEC_envs64": ("CEC/envs64", "CEC"),
    "CEC_IDAAC_envs64": ("CEC_IDAAC/envs64", "CEC_IDAAC"),
    "CEC_envs128": ("CEC/envs128", "CEC"),
    "CEC_IDAAC_envs128": ("CEC_IDAAC/envs128", "CEC_IDAAC"),
    "CEC_envs256": ("CEC/envs256", "CEC"),
    "CEC_IDAAC_envs256": ("CEC_IDAAC/envs256", "CEC_IDAAC"),
}

ALG_LABELS = {
    "CEC_envs32": "CEC-32",
    "CEC_IDAAC_envs32": "CEC-IDAAC-32",
    "CEC_envs64": "CEC-64",
    "CEC_IDAAC_envs64": "CEC-IDAAC-64",
    "CEC_envs128": "CEC-128",
    "CEC_IDAAC_envs128": "CEC-IDAAC-128",
    "CEC_envs256": "CEC-256",
    "CEC_IDAAC_envs256": "CEC-IDAAC-256",
}

ALG_COLORS = [
    "#8FD19E",  # CEC-32
    "#56B4E9",  # CEC-IDAAC-32
    "#5ABF75",  # CEC-64
    "#3D9BD3",  # CEC-IDAAC-64
    "#2F9956",  # CEC-128
    "#1F86C2",  # CEC-IDAAC-128
    "#117733",  # CEC-256
    "#0072B2",  # CEC-IDAAC-256
]

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
    for alg, (relative_dir, filename_prefix) in ALG_SOURCES.items():
        source_dir = results_dir / relative_dir
        filename_re = re.compile(
            rf"^{re.escape(filename_prefix)}_(?P<map>.+)_9_XP_results\.csv$"
        )
        for path in sorted(source_dir.glob("*_9_XP_results.csv")):
            match = filename_re.match(path.name)
            if not match:
                continue
            map_name = match.group("map")
            df = pd.read_csv(path)
            required_columns = {"seed_1", "seed_2", "reward"}
            if not required_columns.issubset(df.columns):
                print(f"Skipping malformed CSV: {path}")
                continue
            if config["XP_ONLY"]:
                df = df[df["seed_1"] != df["seed_2"]]
            if df.empty:
                print(f"Skipping empty XP data: {path}")
                continue
            r = df["reward"].astype(float)
            mean_r = float(r.mean())
            # 먼저 trajectory를 seed pair별로 평균 낸 뒤 pair 간 SEM 계산
            grouped = df.groupby(
                ["seed_1", "seed_2"], as_index=False
            )["reward"].mean()
            n_pairs = len(grouped)
            std_across_pairs = (
                float(grouped["reward"].std(ddof=1))
                if n_pairs > 1
                else 0.0
            )
            sem_pairs = (
                std_across_pairs / np.sqrt(n_pairs)
                if n_pairs > 1
                else 0.0
            )
            rows.append(
                {
                    "algorithm": alg,
                    "map": map_name,
                    "mean_reward": mean_r,
                    "sem_pairs": sem_pairs,
                    "n_pairs": n_pairs,
                    "n_rows": len(df),
                    "source_file": str(path),
                }
            )
    return pd.DataFrame(rows)


def load_human_proxy_grid(results_dir: Path) -> pd.DataFrame:
    rows = []
    for alg, (relative_dir, filename_prefix) in ALG_SOURCES.items():
        source_dir = results_dir / relative_dir
        filename_re = re.compile(
            rf"^{re.escape(filename_prefix)}_(?P<map>.+)_9_BC_results\.csv$"
        )
        for path in sorted(source_dir.glob("*_9_BC_results.csv")):
            match = filename_re.match(path.name)
            if not match:
                continue
            map_name = match.group("map")
            df = pd.read_csv(path)
            required_columns = {
                "model_seed",
                "bc_seed",
                "model_agent",
                "reward",
            }
            if not required_columns.issubset(df.columns):
                print(f"Skipping malformed CSV: {path}")
                continue
            if df.empty:
                print(f"Skipping empty human-proxy data: {path}")
                continue

            grouped = df.groupby(
                ["model_seed", "bc_seed", "model_agent"], as_index=False
            )["reward"].mean()
            n_pairs = len(grouped)
            std_across_pairs = (
                float(grouped["reward"].std(ddof=1))
                if n_pairs > 1
                else 0.0
            )
            sem_pairs = (
                std_across_pairs / np.sqrt(n_pairs)
                if n_pairs > 1
                else 0.0
            )
            rows.append(
                {
                    "algorithm": alg,
                    "map": map_name,
                    "mean_reward": float(df["reward"].astype(float).mean()),
                    "sem_pairs": sem_pairs,
                    "n_pairs": n_pairs,
                    "n_rows": len(df),
                    "source_file": str(path),
                }
            )
    return pd.DataFrame(rows)


def plot_per_map(
    grid: pd.DataFrame,
    out_path: Path,
    evaluation_label: str,
) -> None:
    n_maps = len(MAP_ORDER)
    ncols = 3
    nrows = int(np.ceil(n_maps / ncols))
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(6.4 * ncols, 5.0 * nrows),
        squeeze=False,
    )
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
        ax.set_xticklabels(
            [ALG_LABELS[alg] for alg in ALG_ORDER],
            rotation=30,
            ha="right",
        )
        label = MAP_LABEL_KO.get(map_name, map_name)
        ax.set_title(label)
        ax.set_ylabel("mean reward")
        ax.grid(axis="y", alpha=0.35)
        ax.set_axisbelow(True)

    for j in range(len(MAP_ORDER), len(axes_flat)):
        axes_flat[j].set_visible(False)

    fig.suptitle(
        f"{evaluation_label} performance per layout", fontsize=18, y=1.02
    )
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def plot_overall(
    grid: pd.DataFrame,
    out_path: Path,
    evaluation_label: str,
) -> None:
    overall = (
        grid.groupby("algorithm", as_index=False)
        .agg(mean_reward=("mean_reward", "mean"), std_maps=("mean_reward", "std"), n_maps=("mean_reward", "count"))
        .set_index("algorithm")
        .reindex(ALG_ORDER)
    )
    means = overall["mean_reward"].values.astype(float)

    errs = (overall["std_maps"] / np.sqrt(overall["n_maps"])).values.astype(float)
    errs = np.nan_to_num(errs, nan=0.0)

    fig, ax = plt.subplots(figsize=(14.0, 6.5))
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
    ax.set_xticklabels(
        [ALG_LABELS[alg] for alg in ALG_ORDER],
        rotation=25,
        ha="right",
    )
    ax.set_ylabel("mean reward (average over maps)")
    ax.set_title(
        f"Overall {evaluation_label.lower()} performance — "
        "average over 5 layouts"
    )
    ax.grid(axis="y", alpha=0.35)
    ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def save_graph_set(
    grid: pd.DataFrame,
    results_path: Path,
    out_path: Path,
    file_prefix: str,
    evaluation_label: str,
) -> None:
    if grid.empty:
        raise RuntimeError(
            f"No matching {evaluation_label} result CSVs found under "
            f"{results_path}"
        )

    expected = {(alg, map_name) for alg in ALG_ORDER for map_name in MAP_ORDER}
    found = set(zip(grid["algorithm"], grid["map"]))
    missing = sorted(expected - found)
    if missing:
        print(
            f"Warning: missing {len(missing)} "
            f"{evaluation_label} algorithm/layout results: {missing}"
        )

    os.makedirs(out_path, exist_ok=True)
    per_map_pdf = out_path / f"{file_prefix}_per_map.pdf"
    overall_pdf = out_path / f"{file_prefix}_overall.pdf"
    per_map_csv = out_path / f"{file_prefix}_per_map_table.csv"
    overall_csv = out_path / f"{file_prefix}_overall_table.csv"

    plot_per_map(grid, per_map_pdf, evaluation_label)
    plot_overall(grid, overall_pdf, evaluation_label)

    pivot = grid.pivot_table(
        index="map", columns="algorithm", values="mean_reward", aggfunc="first"
    )
    pivot = pivot.reindex(index=MAP_ORDER, columns=ALG_ORDER)
    pivot.to_csv(per_map_csv, encoding="utf-8")
    overall_df = (
        grid.groupby("algorithm", as_index=False)
        .agg(
            mean_over_maps=("mean_reward", "mean"),
            std_across_maps=("mean_reward", "std"),
            n_maps=("mean_reward", "count"),
        )
    )
    overall_df["sem_across_maps"] = (
        overall_df["std_across_maps"] / np.sqrt(overall_df["n_maps"])
    )
    overall_df = (
        overall_df.set_index("algorithm").reindex(ALG_ORDER).reset_index()
    )
    overall_df.to_csv(overall_csv, index=False, encoding="utf-8")

    print(f"Saved: {per_map_pdf}")
    print(f"Saved: {overall_pdf}")
    print(f"Saved: {per_map_csv}")
    print(f"Saved: {overall_csv}")


@hydra.main(version_base=None, config_path="../repro_config", config_name="test_general")
def main(config):
    config = OmegaConf.to_container(config)
    project_root = Path(__file__).resolve().parents[3]
    xp_results_path = Path(
        config.get("XP_RESULTS_DIR") or DEFAULT_XP_RESULTS_DIR
    )
    human_proxy_results_path = Path(
        config.get("HUMAN_PROXY_RESULTS_DIR")
        or DEFAULT_HUMAN_PROXY_RESULTS_DIR
    )
    xp_out_path = Path(
        config.get("GRAPH_OUTPUT_DIR")
        or project_root
        / "artifacts"
        / "test_general_graph_scaling_ICRL_xp"
    )
    human_proxy_out_path = Path(
        config.get("HUMAN_PROXY_GRAPH_OUTPUT_DIR")
        or project_root
        / "artifacts"
        / "test_general_graph_scaling_ICRL_human_proxy"
    )

    xp_only = bool(config["XP_ONLY"])
    xp_grid = load_grid(config, xp_results_path)
    save_graph_set(
        xp_grid,
        xp_results_path,
        xp_out_path,
        (
            "test_general_scaling_xp"
            if xp_only
            else "test_general_scaling_xp_with_sp"
        ),
        (
            "CEC vs CEC-IDAAC scaling cross-play"
            if xp_only
            else "CEC vs CEC-IDAAC scaling cross-play + self-play"
        ),
    )
    human_proxy_grid = load_human_proxy_grid(human_proxy_results_path)
    save_graph_set(
        human_proxy_grid,
        human_proxy_results_path,
        human_proxy_out_path,
        "test_general_scaling_human_proxy",
        "CEC vs CEC-IDAAC scaling human-proxy",
    )


if __name__ == "__main__":
    main()
