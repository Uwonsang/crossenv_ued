import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from plot_modified_wall_xp import ENV_LABELS, bar_color


MODEL_GROUPS = {
    "IPPO": {"empty": "ippo_empty", "wall_a": "ippo_wall_a"},
    "E3T": {"empty": "e3t_empty", "wall_a": "e3t_wall_a"},
    "CEC": {"empty": "cec", "wall_a": "cec"},
    "DCEC": {"empty": "idaac_cec", "wall_a": "idaac_cec"},
}

SERIES = {
    "env64_all_algorithms": (
        ("IPPO", 64),
        ("E3T", 64),
        ("CEC", 64),
        ("DCEC", 64),
    ),
    "env256_all_algorithms": (
        ("IPPO", 256),
        ("E3T", 256),
        ("CEC", 256),
        ("DCEC", 256),
    ),
    "selected": (
        ("IPPO", 64),
        ("E3T", 64),
        ("CEC", 64),
        ("DCEC", 64),
        ("DCEC", 256),
    ),
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Plot empty and wall_a fixed-task cross-play separately."
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path("baselines/CEC_UED/results/procedural_xp"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("baselines/CEC_UED/results/procedural_xp/plots/fixed_maps"),
    )
    return parser.parse_args()


def load_seed_pairs(results_dir, algorithm, num_envs, layout):
    model_group = MODEL_GROUPS[algorithm][layout]
    path = results_dir / f"{model_group}_numenv{num_envs}_episodes.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing XP episode results: {path}")

    frame = pd.read_csv(path)
    frame = frame[
        (frame["split"] == "fixed") & (frame["eval_layout"] == layout)
    ].copy()
    frame["seed_pair"] = frame.apply(
        lambda row: f"{min(row['policy_1'], row['policy_2'])}x"
        f"{max(row['policy_1'], row['policy_2'])}",
        axis=1,
    )
    pair_values = (
        frame.groupby("seed_pair", as_index=False)
        .agg(
            reward=("reward", "mean"),
            normalized=("normalized_return", "mean"),
        )
    )
    if len(pair_values) != 15:
        raise ValueError(
            f"Expected 15 seed pairs for {algorithm}, NUM_ENVS={num_envs}, "
            f"layout={layout}; found {len(pair_values)}"
        )
    return pair_values


def build_statistics(results_dir):
    rows = []
    for layout in ("empty", "wall_a"):
        for algorithm in MODEL_GROUPS:
            for num_envs in (64, 256):
                pairs = load_seed_pairs(results_dir, algorithm, num_envs, layout)
                rows.append(
                    {
                        "layout": layout,
                        "algorithm": algorithm,
                        "num_envs": num_envs,
                        "budget": ENV_LABELS[num_envs],
                        "reward_mean": pairs["reward"].mean(),
                        "reward_sem": pairs["reward"].sem(),
                        "normalized_mean": pairs["normalized"].mean(),
                        "normalized_sem": pairs["normalized"].sem(),
                        "num_seed_pairs": len(pairs),
                    }
                )
    return pd.DataFrame(rows)


def draw_plot(statistics, layout, series_name, series, metric, output_dir):
    rows = []
    labels = []
    colors = []
    for algorithm, num_envs in series:
        match = statistics[
            (statistics["layout"] == layout)
            & (statistics["algorithm"] == algorithm)
            & (statistics["num_envs"] == num_envs)
        ]
        if len(match) != 1:
            raise ValueError(
                f"Expected one row for {layout}, {algorithm}, NUM_ENVS={num_envs}; "
                f"found {len(match)}"
            )
        row = match.iloc[0]
        rows.append(row)
        labels.append(f"{algorithm}\n({row['budget']})")
        colors.append(bar_color(algorithm, row["budget"]))

    fig, axis = plt.subplots(figsize=(7.2, 4.4))
    axis.bar(
        labels,
        [row[f"{metric}_mean"] for row in rows],
        yerr=[row[f"{metric}_sem"] for row in rows],
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
        f"Fixed task: {'Empty' if layout == 'empty' else 'Wall A'}",
        fontweight="bold",
    )
    axis.set_ylabel("Normalized XP reward" if metric == "normalized" else "Mean XP reward")
    if metric == "normalized":
        axis.set_ylim(-0.55, 1.05)
    axis.tick_params(axis="x", labelsize=9)
    fig.tight_layout()

    stem = f"fixed_{layout}_{series_name}_{metric}"
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
    values_path = args.output_dir / "fixed_map_plot_values.csv"
    statistics.to_csv(values_path, index=False)

    output_paths = []
    for layout in ("empty", "wall_a"):
        for series_name, series in SERIES.items():
            for metric in ("normalized", "reward"):
                output_paths.extend(
                    draw_plot(
                        statistics,
                        layout,
                        series_name,
                        series,
                        metric,
                        args.output_dir,
                    )
                )

    print(statistics.to_string(index=False))
    print(f"Saved plot values: {values_path}")
    print(f"Saved {len(output_paths)} plot files to {args.output_dir}")


if __name__ == "__main__":
    main()
