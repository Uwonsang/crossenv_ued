import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


plt.rcParams.update(
    {
        "font.size": 15,
        "axes.titlesize": 18,
        "axes.labelsize": 15,
        "xtick.labelsize": 15,
        "ytick.labelsize": 15,
    }
)


ALGORITHM_GROUPS = {
    "IPPO": ("ippo_empty", "ippo_wall_a"),
    "E3T": ("e3t_empty", "e3t_wall_a"),
    "CEC": ("cec_random3",),
    "DCEC": ("dcec_random3",),
}

COLORS = {
    "IPPO": "#D62728",
    "E3T": "#7B126B",
    "CEC": "#117733",
    "DCEC": "#56B4E9",
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Plot Random3 cross-play results with SEM error bars."
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path("baselines/CEC_UED/results/random3_xp"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("baselines/CEC_UED/results/random3_xp/plots"),
    )
    return parser.parse_args()


def load_pair_values(results_dir, algorithm):
    frames = []
    for model_group in ALGORITHM_GROUPS[algorithm]:
        path = results_dir / f"{model_group}_numenv256_pairs.csv"
        if not path.exists():
            raise FileNotFoundError(f"Missing XP pair results: {path}")
        frames.append(pd.read_csv(path))

    frame = pd.concat(frames, ignore_index=True)
    frame["minmax_normalized"] = (frame["reward_mean"] + 100.0) / 300.0
    return (
        frame.groupby(["split", "seed_pair"], as_index=False)
        .agg(
            reward=("reward_mean", "mean"),
            normalized=("minmax_normalized", "mean"),
        )
    )


def build_statistics(results_dir):
    rows = []
    for algorithm in ALGORITHM_GROUPS:
        pair_values = load_pair_values(results_dir, algorithm)
        for split, split_frame in pair_values.groupby("split"):
            rows.append(
                {
                    "algorithm": algorithm,
                    "num_envs": 256,
                    "budget": "65K",
                    "split": split,
                    "reward_mean": split_frame["reward"].mean(),
                    "reward_sem": split_frame["reward"].sem(),
                    "normalized_mean": split_frame["normalized"].mean(),
                    "normalized_sem": split_frame["normalized"].sem(),
                    "num_seed_pairs": len(split_frame),
                }
            )
    return pd.DataFrame(rows)


def draw_plot(statistics, split, metric, output_dir):
    rows = [
        statistics[
            (statistics["algorithm"] == algorithm)
            & (statistics["split"] == split)
        ].iloc[0]
        for algorithm in ALGORITHM_GROUPS
    ]
    labels = [row["algorithm"] for row in rows]

    fig, axis = plt.subplots(figsize=(7.2, 4.4))
    axis.bar(
        labels,
        [row[f"{metric}_mean"] for row in rows],
        yerr=[row[f"{metric}_sem"] for row in rows],
        capsize=4,
        color=[COLORS[row["algorithm"]] for row in rows],
        edgecolor="#222222",
        linewidth=0.8,
        error_kw={"elinewidth": 1.1, "capthick": 1.1},
    )
    axis.axhline(0, color="#222222", linewidth=0.8)
    axis.set_axisbelow(True)
    axis.grid(axis="y", color="#D9D9D9", linewidth=0.8, alpha=0.75)
    axis.set_title(
        "(a) Fixed tasks"
        if split == "fixed"
        else "(b) Procedurally generated tasks",
        fontweight="bold",
    )
    axis.set_ylabel(
        "Normalized XP reward" if metric == "normalized" else "Mean XP reward"
    )
    if metric == "normalized":
        axis.set_ylim(0.0, 1.05)
    fig.tight_layout()

    stem = f"random3_65k_{split}_{metric}"
    png_path = output_dir / f"{stem}.png"
    pdf_path = output_dir / f"{stem}.pdf"
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)


def draw_combined_normalized_plot(statistics, output_dir):
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2), sharey=True)
    for axis, split, title in zip(
        axes,
        ("fixed", "procedural"),
        ("(a) Fixed tasks", "(b) Procedurally generated tasks"),
    ):
        rows = [
            statistics[
                (statistics["algorithm"] == algorithm)
                & (statistics["split"] == split)
            ].iloc[0]
            for algorithm in ALGORITHM_GROUPS
        ]
        axis.bar(
            [row["algorithm"] for row in rows],
            [row["normalized_mean"] for row in rows],
            yerr=[row["normalized_sem"] for row in rows],
            capsize=4,
            color=[COLORS[row["algorithm"]] for row in rows],
            edgecolor="#222222",
            linewidth=0.8,
            error_kw={"elinewidth": 1.1, "capthick": 1.1},
        )
        axis.set_title(title, fontweight="bold")
        axis.set_ylim(0.0, 1.05)
        axis.set_axisbelow(True)
        axis.grid(axis="y", color="#D9D9D9", linewidth=0.8, alpha=0.75)
        axis.set_xlabel("Algorithm")
    axes[0].set_ylabel("Normalized XP reward")
    fig.tight_layout()

    stem = "random3_65k_normalized"
    fig.savefig(output_dir / f"{stem}.png", dpi=300, bbox_inches="tight")
    fig.savefig(output_dir / f"{stem}.pdf", bbox_inches="tight")
    plt.close(fig)


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    statistics = build_statistics(args.results_dir)
    statistics.to_csv(args.output_dir / "random3_65k_plot_values.csv", index=False)

    for split in ("fixed", "procedural"):
        for metric in ("normalized", "reward"):
            draw_plot(statistics, split, metric, args.output_dir)
    draw_combined_normalized_plot(statistics, args.output_dir)

    print(statistics.to_string(index=False))
    print(f"Saved plots to {args.output_dir}")


if __name__ == "__main__":
    main()
