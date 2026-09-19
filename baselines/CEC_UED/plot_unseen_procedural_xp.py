import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from plot_modified_wall_xp import COLORS


ALGORITHM_GROUPS = {
    "IPPO": ("ippo_empty", "ippo_wall_a"),
    "E3T": ("e3t_empty", "e3t_wall_a"),
    "CEC": ("cec",),
    "DCEC": ("idaac_cec",),
}
TASK_COUNTS = (100, 200, 300, 400, 500)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Plot cumulative unseen-layout procedural XP results."
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path("baselines/CEC_UED/results/unseen_procedural_xp"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("baselines/CEC_UED/results/unseen_procedural_xp/plots"),
    )
    return parser.parse_args()


def load_pair_values(results_dir, algorithm, task_count):
    frames = []
    for model_group in ALGORITHM_GROUPS[algorithm]:
        path = results_dir / f"{model_group}_numenv256_tasks{task_count}_pairs.csv"
        if not path.exists():
            raise FileNotFoundError(f"Missing cumulative XP results: {path}")
        frames.append(pd.read_csv(path))
    frame = pd.concat(frames, ignore_index=True)
    return (
        frame.groupby("seed_pair", as_index=False)
        .agg(
            reward=("reward_mean", "mean"),
            normalized=("normalized_return_mean", "mean"),
            success_rate=("success_rate", "mean"),
        )
    )


def build_statistics(results_dir):
    rows = []
    for task_count in TASK_COUNTS:
        for algorithm in ALGORITHM_GROUPS:
            pairs = load_pair_values(results_dir, algorithm, task_count)
            rows.append(
                {
                    "task_count": task_count,
                    "algorithm": algorithm,
                    "reward_mean": pairs["reward"].mean(),
                    "reward_sem": pairs["reward"].sem(),
                    "normalized_mean": pairs["normalized"].mean(),
                    "normalized_sem": pairs["normalized"].sem(),
                    "success_rate": pairs["success_rate"].mean(),
                    "num_seed_pairs": len(pairs),
                }
            )
    return pd.DataFrame(rows)


def algorithm_color(algorithm):
    return COLORS["DCEC_65K"] if algorithm == "DCEC" else COLORS[algorithm]


def draw_bar_plot(statistics, task_count, metric, output_dir):
    algorithms = list(ALGORITHM_GROUPS)
    frame = statistics[statistics["task_count"] == task_count].set_index("algorithm")
    fig, axis = plt.subplots(figsize=(7.2, 4.4))
    axis.bar(
        algorithms,
        [frame.loc[name, f"{metric}_mean"] for name in algorithms],
        yerr=[frame.loc[name, f"{metric}_sem"] for name in algorithms],
        capsize=4,
        color=[algorithm_color(name) for name in algorithms],
        edgecolor="#222222",
        linewidth=0.8,
        error_kw={"elinewidth": 1.1, "capthick": 1.1},
    )
    axis.axhline(0, color="#222222", linewidth=0.8)
    axis.set_axisbelow(True)
    axis.grid(axis="y", color="#D9D9D9", linewidth=0.8, alpha=0.75)
    axis.set_title(f"{task_count} cumulative procedural tasks", fontweight="bold")
    axis.set_ylabel("Normalized XP reward" if metric == "normalized" else "Mean XP reward")
    if metric == "normalized":
        axis.set_ylim(-0.55, 1.05)
    fig.tight_layout()
    stem = f"procedural_tasks{task_count}_{metric}"
    paths = (output_dir / f"{stem}.png", output_dir / f"{stem}.pdf")
    fig.savefig(paths[0], dpi=300, bbox_inches="tight")
    fig.savefig(paths[1], bbox_inches="tight")
    plt.close(fig)
    return paths


def draw_line_plot(statistics, metric, output_dir):
    fig, axis = plt.subplots(figsize=(7.2, 4.4))
    for algorithm in ALGORITHM_GROUPS:
        frame = statistics[statistics["algorithm"] == algorithm].sort_values(
            "task_count"
        )
        axis.errorbar(
            frame["task_count"],
            frame[f"{metric}_mean"],
            yerr=frame[f"{metric}_sem"],
            marker="o",
            linewidth=2,
            capsize=4,
            label=algorithm,
            color=algorithm_color(algorithm),
        )
    axis.axhline(0, color="#222222", linewidth=0.8)
    axis.set_axisbelow(True)
    axis.grid(color="#D9D9D9", linewidth=0.8, alpha=0.75)
    axis.set_xticks(TASK_COUNTS)
    axis.set_xlabel("Cumulative procedural tasks")
    axis.set_ylabel("Normalized XP reward" if metric == "normalized" else "Mean XP reward")
    axis.set_title("Generalization to unseen wall layouts", fontweight="bold")
    if metric == "normalized":
        axis.set_ylim(-0.55, 1.05)
    axis.legend(frameon=False, ncol=4, loc="best")
    fig.tight_layout()
    stem = f"procedural_cumulative_{metric}"
    paths = (output_dir / f"{stem}.png", output_dir / f"{stem}.pdf")
    fig.savefig(paths[0], dpi=300, bbox_inches="tight")
    fig.savefig(paths[1], bbox_inches="tight")
    plt.close(fig)
    return paths


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    statistics = build_statistics(args.results_dir)
    values_path = args.output_dir / "plot_values.csv"
    statistics.to_csv(values_path, index=False)

    output_paths = []
    for task_count in TASK_COUNTS:
        for metric in ("normalized", "reward"):
            output_paths.extend(draw_bar_plot(statistics, task_count, metric, args.output_dir))
    for metric in ("normalized", "reward"):
        output_paths.extend(draw_line_plot(statistics, metric, args.output_dir))

    print(statistics.to_string(index=False))
    print(f"Saved plot values: {values_path}")
    print(f"Saved {len(output_paths)} plot files to {args.output_dir}")


if __name__ == "__main__":
    main()
