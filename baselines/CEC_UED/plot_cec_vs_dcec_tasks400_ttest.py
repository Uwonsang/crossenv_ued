import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import ttest_rel

from plot_modified_wall_xp import COLORS


METRICS = {
    "reward": "reward_mean",
    "normalized": "normalized_return_mean",
    "success_rate": "success_rate",
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Paired t-test of CEC and DCEC seed-pair results."
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path("baselines/CEC_UED/results/unseen_procedural_xp"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(
            "baselines/CEC_UED/results/unseen_procedural_xp/plots/tasks400_ttest"
        ),
    )
    parser.add_argument("--cec-pairs", type=Path)
    parser.add_argument("--dcec-pairs", type=Path)
    parser.add_argument("--task-count", type=int, default=400)
    parser.add_argument("--title", default="400 cumulative procedural tasks")
    parser.add_argument("--output-prefix", default="cec_vs_dcec_tasks400")
    return parser.parse_args()


def load_paired_results(results_dir, cec_pairs=None, dcec_pairs=None):
    paths = {
        "cec": cec_pairs or results_dir / "cec_numenv256_tasks400_pairs.csv",
        "dcec": dcec_pairs
        or results_dir / "idaac_cec_numenv256_tasks400_pairs.csv",
    }
    for path in paths.values():
        if not path.exists():
            raise FileNotFoundError(f"Missing seed-pair results: {path}")

    cec = pd.read_csv(paths["cec"])[
        ["seed_pair", "reward_mean", "normalized_return_mean", "success_rate"]
    ]
    dcec = pd.read_csv(paths["dcec"])[
        ["seed_pair", "reward_mean", "normalized_return_mean", "success_rate"]
    ]
    paired = cec.merge(
        dcec,
        on="seed_pair",
        how="inner",
        validate="one_to_one",
        suffixes=("_cec", "_dcec"),
    ).sort_values("seed_pair")
    if len(paired) != 15:
        raise ValueError(f"Expected 15 matched seed pairs, found {len(paired)}")

    for column in METRICS.values():
        paired[f"{column}_difference_dcec_minus_cec"] = (
            paired[f"{column}_dcec"] - paired[f"{column}_cec"]
        )
    return paired


def calculate_tests(paired, task_count):
    rows = []
    for metric, column in METRICS.items():
        cec = paired[f"{column}_cec"]
        dcec = paired[f"{column}_dcec"]
        result = ttest_rel(dcec, cec, alternative="two-sided")
        rows.append(
            {
                "task_count": task_count,
                "metric": metric,
                "test": "paired_two_sided_t_test",
                "n_seed_pairs": len(paired),
                "degrees_of_freedom": len(paired) - 1,
                "cec_mean": cec.mean(),
                "cec_sem": cec.sem(),
                "dcec_mean": dcec.mean(),
                "dcec_sem": dcec.sem(),
                "mean_difference_dcec_minus_cec": (dcec - cec).mean(),
                "t_statistic": result.statistic,
                "p_value": result.pvalue,
                "significant_at_0_05": bool(result.pvalue < 0.05),
            }
        )
    return pd.DataFrame(rows)


def p_value_label(p_value):
    if p_value < 0.001:
        return "p < 0.001"
    return f"p = {p_value:.3f}"


def draw_plot(tests, metric, output_dir, title, output_prefix):
    row = tests[tests["metric"] == metric].iloc[0]
    means = [row["cec_mean"], row["dcec_mean"]]
    sems = [row["cec_sem"], row["dcec_sem"]]

    fig, axis = plt.subplots(figsize=(5.4, 4.4))
    axis.bar(
        ["CEC\n(65K)", "DCEC\n(65K)"],
        means,
        yerr=sems,
        capsize=5,
        color=[COLORS["CEC"], COLORS["DCEC_65K"]],
        edgecolor="#222222",
        linewidth=0.8,
        error_kw={"elinewidth": 1.1, "capthick": 1.1},
    )
    axis.axhline(0, color="#222222", linewidth=0.8)
    axis.set_axisbelow(True)
    axis.grid(axis="y", color="#D9D9D9", linewidth=0.8, alpha=0.75)
    axis.set_title(title, fontweight="bold")
    axis.set_ylabel(
        "Normalized XP reward" if metric == "normalized" else "Mean XP reward"
    )
    if metric == "normalized":
        axis.set_ylim(-0.55, 1.05)

    error_tops = [mean + sem for mean, sem in zip(means, sems)]
    data_range = max(error_tops) - min(0.0, min(means))
    bracket_y = max(error_tops) + max(0.03 * data_range, 0.015)
    bracket_height = max(0.025 * data_range, 0.012)
    axis.plot(
        [0, 0, 1, 1],
        [bracket_y, bracket_y + bracket_height, bracket_y + bracket_height, bracket_y],
        color="#222222",
        linewidth=1.1,
    )
    axis.text(
        0.5,
        bracket_y + bracket_height,
        f"paired t-test, {p_value_label(row['p_value'])}",
        ha="center",
        va="bottom",
        fontsize=10,
    )
    if metric == "reward":
        axis.set_ylim(top=bracket_y + bracket_height + max(0.09 * data_range, 8.0))
    fig.tight_layout()

    stem = f"{output_prefix}_{metric}"
    png_path = output_dir / f"{stem}.png"
    pdf_path = output_dir / f"{stem}.pdf"
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)
    return png_path, pdf_path


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    paired = load_paired_results(
        args.results_dir, args.cec_pairs, args.dcec_pairs
    )
    tests = calculate_tests(paired, args.task_count)
    paired_path = args.output_dir / f"{args.output_prefix}_paired_values.csv"
    tests_path = args.output_dir / f"{args.output_prefix}_ttest.csv"
    paired.to_csv(paired_path, index=False)
    tests.to_csv(tests_path, index=False)

    output_paths = []
    for metric in ("reward", "normalized"):
        output_paths.extend(
            draw_plot(
                tests,
                metric,
                args.output_dir,
                args.title,
                args.output_prefix,
            )
        )

    print(tests.to_string(index=False))
    print(f"Saved paired values: {paired_path}")
    print(f"Saved t-tests: {tests_path}")
    print(f"Saved {len(output_paths)} plot files to {args.output_dir}")


if __name__ == "__main__":
    main()
