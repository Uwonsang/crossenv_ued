import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


ENV_LABELS = {
    32: "8K",
    64: "16K",
    128: "32K",
    256: "65K",
}

ALGORITHM_GROUPS = {
    "IPPO": ("ippo_empty", "ippo_wall_a"),
    "E3T": ("e3t_empty", "e3t_wall_a"),
    "CEC": ("cec",),
    "DCEC": ("idaac_cec",),
}

COLORS = {
    "IPPO": "#D62F3A",
    "E3T": "#92278F",
    "CEC": "#238B45",
    "DCEC_16K": "#67B7DF",
    "DCEC_65K": "#1F82B7",
}

PLOT_SPECS = (
    {
        "name": "fixed_env64_all_algorithms",
        "split": "fixed",
        "series": (("IPPO", 64), ("E3T", 64), ("CEC", 64), ("DCEC", 64)),
    },
    {
        "name": "fixed_env256_all_algorithms",
        "split": "fixed",
        "series": (("IPPO", 256), ("E3T", 256), ("CEC", 256), ("DCEC", 256)),
    },
    {
        "name": "procedural_env64_all_algorithms",
        "split": "procedural",
        "series": (("IPPO", 64), ("E3T", 64), ("CEC", 64), ("DCEC", 64)),
    },
    {
        "name": "procedural_env256_all_algorithms",
        "split": "procedural",
        "series": (("IPPO", 256), ("E3T", 256), ("CEC", 256), ("DCEC", 256)),
    },
    {
        "name": "fixed_selected",
        "split": "fixed",
        "series": (
            ("IPPO", 64),
            ("E3T", 64),
            ("CEC", 64),
            ("DCEC", 64),
            ("DCEC", 256),
        ),
    },
    {
        "name": "procedural_selected",
        "split": "procedural",
        "series": (
            ("IPPO", 64),
            ("E3T", 64),
            ("CEC", 64),
            ("DCEC", 64),
            ("DCEC", 256),
        ),
    },
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Plot ToyCoopNoPink cross-play rewards with SEM error bars."
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path("baselines/CEC_UED/results/procedural_xp"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("baselines/CEC_UED/results/procedural_xp/plots"),
    )
    return parser.parse_args()


def load_pair_values(results_dir, algorithm, num_envs):
    frames = []
    for model_group in ALGORITHM_GROUPS[algorithm]:
        path = results_dir / f"{model_group}_numenv{num_envs}_pairs.csv"
        if not path.exists():
            raise FileNotFoundError(f"Missing XP pair results: {path}")
        frames.append(pd.read_csv(path))

    frame = pd.concat(frames, ignore_index=True)
    # IPPO/E3T have separate empty and wall_a populations. Average the two
    # training conditions within each seed pair before computing SEM.
    return (
        frame.groupby(["split", "seed_pair"], as_index=False)
        .agg(
            reward=("reward_mean", "mean"),
            normalized=("normalized_return_mean", "mean"),
        )
    )


def build_statistics(results_dir):
    rows = []
    for algorithm in ALGORITHM_GROUPS:
        for num_envs in (64, 256):
            pair_values = load_pair_values(results_dir, algorithm, num_envs)
            for split, split_frame in pair_values.groupby("split"):
                rows.append(
                    {
                        "algorithm": algorithm,
                        "num_envs": num_envs,
                        "budget": ENV_LABELS[num_envs],
                        "split": split,
                        "reward_mean": split_frame["reward"].mean(),
                        "reward_sem": split_frame["reward"].sem(),
                        "normalized_mean": split_frame["normalized"].mean(),
                        "normalized_sem": split_frame["normalized"].sem(),
                        "num_seed_pairs": len(split_frame),
                    }
                )
    return pd.DataFrame(rows)


def bar_color(algorithm, budget):
    if algorithm == "DCEC":
        return COLORS[f"DCEC_{budget}"]
    return COLORS[algorithm]


def draw_plot(statistics, spec, metric, output_dir):
    mean_column = f"{metric}_mean"
    sem_column = f"{metric}_sem"
    rows = []
    labels = []
    colors = []

    for algorithm, num_envs in spec["series"]:
        match = statistics[
            (statistics["algorithm"] == algorithm)
            & (statistics["num_envs"] == num_envs)
            & (statistics["split"] == spec["split"])
        ]
        if len(match) != 1:
            raise ValueError(
                f"Expected one row for {algorithm}, NUM_ENVS={num_envs}, "
                f"split={spec['split']}; found {len(match)}"
            )
        row = match.iloc[0]
        rows.append(row)
        labels.append(f"{algorithm}\n({row['budget']})")
        colors.append(bar_color(algorithm, row["budget"]))

    means = [row[mean_column] for row in rows]
    sems = [row[sem_column] for row in rows]
    fig, axis = plt.subplots(figsize=(7.2, 4.4))
    axis.bar(
        labels,
        means,
        yerr=sems,
        capsize=4,
        color=colors,
        edgecolor="#222222",
        linewidth=0.8,
        error_kw={"elinewidth": 1.1, "capthick": 1.1},
    )
    axis.axhline(0, color="#222222", linewidth=0.8)
    axis.set_axisbelow(True)
    axis.grid(axis="y", color="#D9D9D9", linewidth=0.8, alpha=0.75)
    axis.set_title(
        "Fixed tasks" if spec["split"] == "fixed" else "100 procedurally generated tasks",
        fontweight="bold",
    )
    axis.set_ylabel("Normalized XP reward" if metric == "normalized" else "Mean XP reward")
    if metric == "normalized":
        axis.set_ylim(-0.55, 1.05)
    axis.tick_params(axis="x", labelsize=9)
    fig.tight_layout()

    stem = f"{spec['name']}_{metric}"
    png_path = output_dir / f"{stem}.png"
    pdf_path = output_dir / f"{stem}.pdf"
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)
    return png_path, pdf_path


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    statistics = build_statistics(args.results_dir)
    values_path = args.output_dir / "plot_values.csv"
    statistics.to_csv(values_path, index=False)

    output_paths = []
    for spec in PLOT_SPECS:
        for metric in ("normalized", "reward"):
            output_paths.extend(draw_plot(statistics, spec, metric, args.output_dir))

    print(statistics.to_string(index=False))
    print(f"Saved plot values: {values_path}")
    print(f"Saved {len(output_paths)} plot files to {args.output_dir}")


if __name__ == "__main__":
    main()
