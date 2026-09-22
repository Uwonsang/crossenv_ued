"""Visualize layout-wise outputs from policy_value_concrete_example.py --large-scale."""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


FAMILY_ORDER = (
    "asymm_advantages",
    "coord_ring",
    "counter_circuit",
    "forced_coord",
    "cramped_room",
)
FAMILY_LABELS = {
    "asymm_advantages": "Asymmetric Advantages",
    "coord_ring": "Coordination Ring",
    "counter_circuit": "Counter Circuit",
    "forced_coord": "Forced Coordination",
    "cramped_room": "Cramped Room",
}
MODEL_LABELS = {"CEC": "CEC", "CEC_IDAAC": "DCEC"}
POLICY_COLOR = "#0072B2"
VALUE_COLOR = "#D55E00"


def load_metrics(input_dir: Path) -> pd.DataFrame:
    frames = []
    for family_dir in sorted(path for path in input_dir.iterdir() if path.is_dir()):
        metrics = family_dir / "concrete_example_metrics.csv"
        if not metrics.is_file():
            continue
        frame = pd.read_csv(metrics)
        frame["layout"] = family_dir.name
        frames.append(frame)
    if not frames:
        raise FileNotFoundError(
            f"No layout-level concrete_example_metrics.csv files found under {input_dir}"
        )
    frame = pd.concat(frames, ignore_index=True)
    frame["series"] = frame.apply(
        lambda row: (
            f"{MODEL_LABELS.get(row['model'], row['model'])} "
            f"({int(row['num_envs'])})"
        ),
        axis=1,
    )
    return frame


def ordered_layouts(frame: pd.DataFrame) -> list[str]:
    present = set(frame["layout"])
    return [name for name in FAMILY_ORDER if name in present] + sorted(
        present - set(FAMILY_ORDER)
    )


def ordered_series(frame: pd.DataFrame) -> list[str]:
    order = frame[["series", "model", "num_envs"]].drop_duplicates().copy()
    order["model_rank"] = order["model"].map({"CEC": 0, "CEC_IDAAC": 1}).fillna(2)
    return order.sort_values(["model_rank", "num_envs", "series"])["series"].tolist()


def mean_sem_by_seed(
    frame: pd.DataFrame, column: str, layouts: list[str], series: str
) -> tuple[np.ndarray, np.ndarray]:
    subset = frame[frame["series"] == series]
    per_seed = subset.groupby(["layout", "seed"], as_index=False)[column].mean()
    grouped = per_seed.groupby("layout")[column]
    mean = grouped.mean().reindex(layouts)
    count = grouped.count().reindex(layouts)
    sem = (grouped.std().reindex(layouts) / np.sqrt(count)).fillna(0)
    return mean.to_numpy(), sem.to_numpy()


def plot_layout_distances(
    frame: pd.DataFrame,
    output_dir: Path,
    suffix: str,
    policy_column: str,
    value_column: str,
    ylabel: str,
) -> None:
    layouts = ordered_layouts(frame)
    series_names = ordered_series(frame)
    figure, axes = plt.subplots(
        1, len(series_names), figsize=(6.2 * len(series_names), 4.2), squeeze=False
    )
    x = np.arange(len(layouts))
    width = .36
    for axis, series in zip(axes[0], series_names):
        policy_mean, policy_sem = mean_sem_by_seed(
            frame, policy_column, layouts, series
        )
        value_mean, value_sem = mean_sem_by_seed(frame, value_column, layouts, series)
        axis.bar(
            x - width / 2, policy_mean, width, yerr=policy_sem, capsize=3,
            color=POLICY_COLOR, edgecolor="black", linewidth=.5, label="Policy",
        )
        axis.bar(
            x + width / 2, value_mean, width, yerr=value_sem, capsize=3,
            color=VALUE_COLOR, edgecolor="black", linewidth=.5, label="Value",
        )
        axis.set_xticks(x)
        axis.set_xticklabels(
            [FAMILY_LABELS.get(layout, layout) for layout in layouts],
            rotation=25,
            ha="right",
        )
        axis.set_title(series, fontweight="bold")
        axis.set_ylabel(ylabel)
        axis.grid(axis="y", alpha=.25)
        axis.legend(frameon=False)
    figure.suptitle("Policy and value representation distance", fontweight="bold")
    figure.tight_layout()
    figure.savefig(output_dir / f"representation_distance_{suffix}.png", dpi=300,
                   bbox_inches="tight")
    plt.close(figure)


def correlation(x: np.ndarray, y: np.ndarray) -> float:
    valid = np.isfinite(x) & np.isfinite(y)
    if valid.sum() < 2 or np.std(x[valid]) == 0 or np.std(y[valid]) == 0:
        return float("nan")
    return float(np.corrcoef(x[valid], y[valid])[0, 1])


def plot_return_scatter(frame: pd.DataFrame, output_dir: Path) -> None:
    series_names = ordered_series(frame)
    figure, axes = plt.subplots(
        1, len(series_names), figsize=(5.7 * len(series_names), 4.4), squeeze=False
    )
    for axis, series in zip(axes[0], series_names):
        subset = frame[frame["series"] == series]
        x = subset["mc_return_delta_a_minus_b"].abs().to_numpy()
        policy = subset["policy_rep_zscored_euclidean_distance"].to_numpy()
        value = subset["value_rep_zscored_euclidean_distance"].to_numpy()
        axis.scatter(x, policy, s=18, alpha=.35, color=POLICY_COLOR, label="Policy")
        axis.scatter(x, value, s=18, alpha=.35, color=VALUE_COLOR, label="Value")
        axis.text(
            .98, .98,
            f"r(policy)={correlation(x, policy):.2f}\n"
            f"r(value)={correlation(x, value):.2f}",
            transform=axis.transAxes, ha="right", va="top", fontsize=9,
            bbox=dict(facecolor="white", edgecolor=".7", alpha=.85),
        )
        axis.set_title(series, fontweight="bold")
        axis.set_xlabel("Absolute MC return difference")
        axis.set_ylabel("Z-scored representation L2 distance")
        axis.grid(alpha=.25)
        axis.legend(frameon=False)
    figure.suptitle("Representation distance versus future-return difference",
                    fontweight="bold")
    figure.tight_layout()
    figure.savefig(output_dir / "representation_vs_return.png", dpi=300,
                   bbox_inches="tight")
    plt.close(figure)


def as_rate(values: pd.Series) -> pd.Series:
    if values.dtype == bool:
        return values.astype(float)
    return values.astype(str).str.lower().map({"true": 1.0, "false": 0.0})


def plot_filter_rates(frame: pd.DataFrame, output_dir: Path) -> None:
    metrics = (
        ("passes_policy_equivalence", "Policy equivalent"),
        ("passes_intended_interaction", "Interact condition"),
        ("passes_return_distinction", "Return distinct"),
        ("passes_concrete_example", "All conditions"),
    )
    layouts = ordered_layouts(frame)
    series_names = ordered_series(frame)
    figure, axes = plt.subplots(
        1, len(series_names), figsize=(6.2 * len(series_names), 4.2), squeeze=False
    )
    x = np.arange(len(layouts))
    width = .8 / len(metrics)
    for axis, series in zip(axes[0], series_names):
        subset = frame[frame["series"] == series].copy()
        for offset, (column, label) in enumerate(metrics):
            subset[column] = as_rate(subset[column])
            rates = subset.groupby("layout")[column].mean().reindex(layouts)
            positions = x + (offset - (len(metrics) - 1) / 2) * width
            axis.bar(positions, rates, width=width, label=label,
                     edgecolor="black", linewidth=.4)
        axis.set_xticks(x)
        axis.set_xticklabels(
            [FAMILY_LABELS.get(layout, layout) for layout in layouts],
            rotation=25,
            ha="right",
        )
        axis.set_ylim(0, 1.05)
        axis.set_ylabel("Fraction of evaluations")
        axis.set_title(series, fontweight="bold")
        axis.grid(axis="y", alpha=.25)
        axis.legend(frameon=False, fontsize=8)
    figure.suptitle("Large-scale pair filter rates", fontweight="bold")
    figure.tight_layout()
    figure.savefig(output_dir / "filter_pass_rates.png", dpi=300,
                   bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-root", type=Path, required=True)
    parser.add_argument("--input-dir", type=Path)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()

    input_dir = args.input_dir or (
        args.model_root.expanduser() / "analysis" / "policy_value_large_scale"
    )
    if not input_dir.is_dir():
        parser.error(f"Large-scale result directory does not exist: {input_dir}")
    output_dir = args.output_dir or input_dir / "figures"
    output_dir.mkdir(parents=True, exist_ok=True)

    frame = load_metrics(input_dir)
    required = {
        "model", "num_envs", "seed", "mc_return_delta_a_minus_b",
        "policy_rep_cosine_distance", "value_rep_cosine_distance",
        "policy_rep_raw_euclidean_distance", "value_rep_raw_euclidean_distance",
        "policy_rep_zscored_euclidean_distance",
        "value_rep_zscored_euclidean_distance", "passes_policy_equivalence",
        "passes_intended_interaction", "passes_return_distinction",
        "passes_concrete_example",
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        parser.error("Missing metrics columns: " + ", ".join(missing))

    plot_layout_distances(
        frame, output_dir, "cosine",
        "policy_rep_cosine_distance", "value_rep_cosine_distance",
        "Cosine distance",
    )
    plot_layout_distances(
        frame, output_dir, "raw_l2",
        "policy_rep_raw_euclidean_distance", "value_rep_raw_euclidean_distance",
        "Raw Euclidean distance",
    )
    plot_layout_distances(
        frame, output_dir, "zscored_l2",
        "policy_rep_zscored_euclidean_distance",
        "value_rep_zscored_euclidean_distance",
        "Z-scored Euclidean distance",
    )
    plot_return_scatter(frame, output_dir)
    plot_filter_rates(frame, output_dir)
    print(f"Loaded {len(frame)} evaluations from {input_dir}")
    print(f"Saved large-scale PNG figures to {output_dir}")


if __name__ == "__main__":
    main()
