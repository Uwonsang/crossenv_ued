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
        "font.family": "DejaVu Sans",
        "font.size": 24,
        "axes.titlesize": 28,
        "axes.labelsize": 24,
        "xtick.labelsize": 24,
        "ytick.labelsize": 24,
        "figure.titlesize": 28,
        "legend.fontsize": 24,
    }
)

DEFAULT_XP_RESULTS_DIR = Path(
    "/mnt/nas/wonsang/crossenv_ued/models/ICRL/xp_results"
)
DEFAULT_HUMAN_PROXY_RESULTS_DIR = Path(
    "/app/nas/models/ICRL/human_proxy_results"
)

ALG_ORDER = [
    "IPPO",
    "E3T",
    "FCP",
    "CEC_envs64",
    "CEC_IDAAC_envs32",
    "CEC_IDAAC_envs256",
]

# graph key: (directory relative to xp_results, checkpoint filename prefix)
ALG_SOURCES = {
    "CEC_envs64": ("CEC/envs64", "CEC"),
    "CEC_IDAAC_envs32": ("CEC_IDAAC/envs32", "CEC_IDAAC"),
    "CEC_IDAAC_envs256": ("CEC_IDAAC/envs256", "CEC_IDAAC"),
    "E3T": ("E3T", "E3T"),
    "FCP": ("FCP", "FCP"),
    "IPPO": ("IPPO", "IPPO"),
}

ALG_LABELS = {
    "IPPO": "IPPO",
    "E3T": "E3T",
    "FCP": "FCP",
    "CEC_envs64": "CEC",
    "CEC_IDAAC_envs32": "DCEC (8K)",
    "CEC_IDAAC_envs256": "DCEC (65K)",
}

ALG_COLORS = [
    "#d62728",  # IPPO
    "#7b126b",  # E3T
    "#e3a21a",  # FCP
    "#117733",  # CEC (64)
    "#56B4E9",  # DCEC (8K): Okabe-Ito sky blue
    "#0072B2",  # DCEC (65K): Okabe-Ito blue
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
            # Treat seed_1 as the focal training seed.  Average over partner
            # seeds and evaluation trajectories before aggregating layouts.
            seed_rewards = (
                df.groupby("seed_1")["reward"].mean().astype(float).to_dict()
            )
            rows.append(
                {
                    "algorithm": alg,
                    "map": map_name,
                    "mean_reward": mean_r,
                    "sem_pairs": sem_pairs,
                    "n_pairs": n_pairs,
                    "n_rows": len(df),
                    "seed_rewards": seed_rewards,
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
                    "seed_rewards": (
                        df.groupby("model_seed")["reward"]
                        .mean().astype(float).to_dict()
                    ),
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
        figsize=(8.2 * ncols, 6.2 * nrows),
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
            edgecolor="none",
            linewidth=0,
            alpha=0.8,
        )
        ax.set_xticks(x)
        panel_labels = [
            ALG_LABELS[alg].replace(" ", "\n", 1)
            if ALG_LABELS[alg].startswith("DCEC ")
            else ALG_LABELS[alg]
            for alg in ALG_ORDER
        ]
        ax.set_xticklabels(panel_labels, rotation=0, ha="center")
        ax.tick_params(axis="x", pad=8)
        label = MAP_LABEL_KO.get(map_name, map_name)
        ax.set_title(label, fontweight="normal", pad=12)
        ax.set_ylabel("mean reward")
        ax.grid(False)
        ax.set_axisbelow(True)

    for j in range(len(MAP_ORDER), len(axes_flat)):
        axes_flat[j].set_visible(False)

    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def aggregate_overall_by_seed(grid: pd.DataFrame) -> pd.DataFrame:
    """Average layouts within each training seed, then summarize seeds."""
    seed_rows = []
    for row in grid.itertuples(index=False):
        for seed, reward in row.seed_rewards.items():
            seed_rows.append(
                {
                    "algorithm": row.algorithm,
                    "map": row.map,
                    "seed": int(seed),
                    "reward": float(reward),
                }
            )
    if not seed_rows:
        raise RuntimeError("No seed-level rewards are available")

    seed_frame = pd.DataFrame(seed_rows)
    seed_overall = (
        seed_frame.groupby(["algorithm", "seed"], as_index=False)["reward"]
        .mean()
    )
    overall = seed_overall.groupby("algorithm", as_index=False).agg(
        mean_reward=("reward", "mean"),
        std_seeds=("reward", "std"),
        n_seeds=("reward", "count"),
    )
    overall["sem_seeds"] = (
        overall["std_seeds"] / np.sqrt(overall["n_seeds"])
    ).fillna(0.0)
    return overall.set_index("algorithm").reindex(ALG_ORDER)


def plot_overall(
    grid: pd.DataFrame,
    out_path: Path,
    evaluation_label: str,
) -> None:
    overall = aggregate_overall_by_seed(grid)
    means = overall["mean_reward"].values.astype(float)
    errs = overall["sem_seeds"].values.astype(float)
    errs = np.nan_to_num(errs, nan=0.0)

    fig, ax = plt.subplots(figsize=(10.0, 5.694))
    x = np.arange(len(ALG_ORDER))
    colors = [_alg_color_map()[a] for a in ALG_ORDER]
    ax.bar(
        x,
        means,
        yerr=errs,
        capsize=5,
        color=colors,
        edgecolor="none",
        linewidth=0,
        alpha=0.8,
    )
    ax.set_xticks(x)
    overall_labels = [
        ALG_LABELS[alg].replace(" ", "\n", 1)
        if ALG_LABELS[alg].startswith("DCEC ")
        else ALG_LABELS[alg]
        for alg in ALG_ORDER
    ]
    ax.set_xticklabels(overall_labels, rotation=0, ha="center")
    ax.tick_params(axis="x", pad=12)
    ax.set_ylabel("XP Reward")
    ax.set_title("(a) 5 Fixed Tasks", fontweight="normal", pad=16)
    ax.grid(False)
    ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(out_path)
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
    overall_df = aggregate_overall_by_seed(grid).reset_index().rename(
        columns={"mean_reward": "mean_over_maps"}
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
    xp_only = bool(config["XP_ONLY"])
    default_xp_output_name = (
        "test_general_graph_ICRL_xp"
        if xp_only
        else "test_general_graph_ICRL_xp_with_sp"
    )
    xp_out_path = Path(
        config.get("GRAPH_OUTPUT_DIR")
        or project_root / "artifacts" / default_xp_output_name
    )
    human_proxy_out_path = Path(
        config.get("HUMAN_PROXY_GRAPH_OUTPUT_DIR")
        or project_root
        / "artifacts"
        / "test_general_graph_ICRL_human_proxy"
    )

    xp_grid = load_grid(config, xp_results_path)
    save_graph_set(
        xp_grid,
        xp_results_path,
        xp_out_path,
        "test_general_xp" if xp_only else "test_general_xp_with_sp",
        "Cross-play" if xp_only else "Cross-play + self-play",
    )
    human_proxy_grid = load_human_proxy_grid(human_proxy_results_path)
    save_graph_set(
        human_proxy_grid,
        human_proxy_results_path,
        human_proxy_out_path,
        "test_general_human_proxy",
        "Human-proxy",
    )


if __name__ == "__main__":
    main()
