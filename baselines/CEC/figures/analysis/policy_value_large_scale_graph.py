"""Visualize layout-wise outputs from policy_value_concrete_example.py --large-scale."""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from policy_value_fixed_pairs_metrics import save_rsa_figures


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
MODEL_COLORS = {"CEC": "#117733", "CEC_IDAAC": "#0072B2"}
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


def load_cka(input_dir: Path) -> pd.DataFrame:
    frames = []
    for family_dir in sorted(path for path in input_dir.iterdir() if path.is_dir()):
        metrics = family_dir / "concrete_example_cka.csv"
        if not metrics.is_file():
            continue
        frame = pd.read_csv(metrics)
        frame["layout"] = family_dir.name
        frames.append(frame)
    if not frames:
        raise FileNotFoundError(
            "No concrete_example_cka.csv files were found. Re-run the large-scale "
            "analysis with the updated policy_value_concrete_example.py first."
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


def plot_saved_rsa(input_dir: Path) -> int:
    """Regenerate per-layout RSA figures from already evaluated CSV files."""
    count = 0
    for family_dir in sorted(path for path in input_dir.iterdir() if path.is_dir()):
        rsa_path = family_dir / "concrete_example_rsa.csv"
        if not rsa_path.is_file():
            continue
        rows = pd.read_csv(rsa_path).to_dict(orient="records")
        if not rows:
            continue
        save_rsa_figures(rows, family_dir, family_dir.name)
        count += 1
    if count == 0:
        raise FileNotFoundError(
            f"No layout-level concrete_example_rsa.csv files found under {input_dir}"
        )
    return count


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


def spearman_correlation(x: np.ndarray, y: np.ndarray) -> float:
    """Rank correlation with average ranks for tied observations."""
    valid = np.isfinite(x) & np.isfinite(y)
    if valid.sum() < 3:
        return float("nan")
    x_rank = pd.Series(x[valid]).rank(method="average").to_numpy()
    y_rank = pd.Series(y[valid]).rank(method="average").to_numpy()
    return correlation(x_rank, y_rank)


def retain_common_pairs(frame: pd.DataFrame) -> pd.DataFrame:
    """Use the same pair IDs for every checkpoint within each layout."""
    retained = []
    for layout, layout_frame in frame.groupby("layout", sort=False):
        pair_sets = [
            set(group["pair_id"].astype(int))
            for _, group in layout_frame.groupby(["series", "seed"], sort=False)
        ]
        common = set.intersection(*pair_sets) if pair_sets else set()
        if not common:
            raise ValueError(f"No pair is shared by every checkpoint for {layout}")
        retained.append(layout_frame[layout_frame["pair_id"].isin(common)])
    return pd.concat(retained, ignore_index=True)


def seed_layout_correlations(frame: pd.DataFrame) -> pd.DataFrame:
    """Compute Spearman correlations with checkpoint seeds as replicates."""
    rows = []
    group_columns = ["series", "model", "num_envs", "layout", "seed"]
    for keys, subset in frame.groupby(group_columns, sort=False):
        series, model, num_envs, layout, seed = keys
        x = subset["mc_return_delta_a_minus_b"].abs().to_numpy()
        policy = subset["policy_rep_zscored_rms_distance"].to_numpy()
        value = subset["value_rep_zscored_rms_distance"].to_numpy()
        policy_rho = spearman_correlation(x, policy)
        value_rho = spearman_correlation(x, value)
        rows.append({
            "series": series,
            "model": model,
            "num_envs": int(num_envs),
            "layout": layout,
            "seed": int(seed),
            "pairs": len(subset),
            "policy_spearman_rho": policy_rho,
            "value_spearman_rho": value_rho,
            "value_minus_policy_rho": value_rho - policy_rho,
        })
    result = pd.DataFrame(rows)
    # Each layout contributes equally to the across-layout result for a seed.
    overall = result.groupby(
        ["series", "model", "num_envs", "seed"], as_index=False
    ).agg({
        "pairs": "sum",
        "policy_spearman_rho": "mean",
        "value_spearman_rho": "mean",
        "value_minus_policy_rho": "mean",
    })
    overall["layout"] = "overall"
    return pd.concat([result, overall[result.columns]], ignore_index=True)


def bootstrap_mean_ci(
    values: np.ndarray, samples: int, seed: int
) -> tuple[float, float, float]:
    values = values[np.isfinite(values)]
    if not len(values):
        return float("nan"), float("nan"), float("nan")
    mean = float(values.mean())
    if len(values) == 1:
        return mean, float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(values), size=(samples, len(values)))
    bootstrap_means = values[indices].mean(axis=1)
    low, high = np.quantile(bootstrap_means, [.025, .975])
    return mean, float(low), float(high)


def summarize_seed_correlations(
    correlations: pd.DataFrame, bootstrap_samples: int
) -> pd.DataFrame:
    metrics = (
        "policy_spearman_rho",
        "value_spearman_rho",
        "value_minus_policy_rho",
    )
    rows = []
    for group_index, (keys, subset) in enumerate(correlations.groupby(
        ["series", "model", "num_envs", "layout"], sort=False
    )):
        series, model, num_envs, layout = keys
        row = {
            "series": series,
            "model": model,
            "num_envs": int(num_envs),
            "layout": layout,
            "seeds": int(subset["seed"].nunique()),
            "mean_pairs_per_seed": float(subset["pairs"].mean()),
        }
        for metric_index, metric in enumerate(metrics):
            mean, low, high = bootstrap_mean_ci(
                subset[metric].to_numpy(dtype=float), bootstrap_samples,
                seed=1701 + group_index * len(metrics) + metric_index,
            )
            row[f"{metric}_mean"] = mean
            row[f"{metric}_ci_low"] = low
            row[f"{metric}_ci_high"] = high
        rows.append(row)
    return pd.DataFrame(rows)


def asymmetric_errors(summary: pd.DataFrame, metric: str) -> np.ndarray:
    mean = summary[f"{metric}_mean"].to_numpy(dtype=float)
    low = summary[f"{metric}_ci_low"].to_numpy(dtype=float)
    high = summary[f"{metric}_ci_high"].to_numpy(dtype=float)
    return np.vstack((np.maximum(0, mean - low), np.maximum(0, high - mean)))


def plot_seed_layout_correlations(
    summary: pd.DataFrame, frame: pd.DataFrame, output_dir: Path
) -> None:
    layouts = ordered_layouts(frame) + ["overall"]
    series_names = ordered_series(frame)
    figure, axes = plt.subplots(
        1, len(series_names), figsize=(6.2 * len(series_names), 4.4), squeeze=False
    )
    x = np.arange(len(layouts))
    width = .36
    for axis, series in zip(axes[0], series_names):
        subset = summary[summary["series"] == series].set_index("layout").reindex(layouts)
        for positions, metric, label, color in (
            (x - width / 2, "policy_spearman_rho", "Policy", POLICY_COLOR),
            (x + width / 2, "value_spearman_rho", "Value", VALUE_COLOR),
        ):
            axis.bar(
                positions, subset[f"{metric}_mean"], width=width,
                yerr=asymmetric_errors(subset, metric), capsize=3,
                color=color, edgecolor="black", linewidth=.5, label=label,
            )
        overall_gap = subset.loc["overall", "value_minus_policy_rho_mean"]
        axis.set_title(
            f"{series}\noverall $\\Delta\\rho={overall_gap:.2f}$",
            fontweight="bold",
        )
        axis.set_xticks(x)
        axis.set_xticklabels(
            [
                FAMILY_LABELS.get(
                    layout, "Overall" if layout == "overall" else layout
                )
                for layout in layouts
            ],
            rotation=25, ha="right",
        )
        axis.set_ylim(-1.05, 1.05)
        axis.axhline(0, color="black", linewidth=.7)
        axis.set_ylabel("Spearman correlation with $|\\Delta G_{MC}|$")
        axis.grid(axis="y", alpha=.25)
        axis.legend(frameon=False)
    figure.suptitle(
        "Seed-level return sensitivity of policy and value representations",
        fontweight="bold",
    )
    figure.tight_layout()
    figure.savefig(
        output_dir / "return_correlation_by_seed_layout.png", dpi=300,
        bbox_inches="tight",
    )
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(7.2, 4.3))
    width = .8 / len(series_names)
    for offset, series in enumerate(series_names):
        subset = summary[summary["series"] == series].set_index("layout").reindex(layouts)
        metric = "value_minus_policy_rho"
        positions = x + (offset - (len(series_names) - 1) / 2) * width
        model = subset["model"].dropna().iloc[0]
        axis.bar(
            positions, subset[f"{metric}_mean"], width=width,
            yerr=asymmetric_errors(subset, metric), capsize=3,
            label=series, edgecolor="black", linewidth=.5,
            color=MODEL_COLORS.get(model, ".5"),
        )
    axis.axhline(0, color="black", linewidth=.8)
    axis.set_xticks(x)
    axis.set_xticklabels(
        [
            FAMILY_LABELS.get(
                layout, "Overall" if layout == "overall" else layout
            )
            for layout in layouts
        ],
        rotation=25, ha="right",
    )
    axis.set_ylabel(r"Return-sensitivity gap $\rho_V-\rho_\pi$")
    axis.set_title("Policy–value return-sensitivity separation", fontweight="bold")
    axis.grid(axis="y", alpha=.25)
    axis.legend(frameon=False)
    figure.tight_layout()
    figure.savefig(
        output_dir / "return_correlation_gap.png", dpi=300, bbox_inches="tight"
    )
    plt.close(figure)


def plot_return_scatter(frame: pd.DataFrame, output_dir: Path) -> None:
    series_names = ordered_series(frame)
    figure, axes = plt.subplots(
        1, len(series_names), figsize=(5.7 * len(series_names), 4.4), squeeze=False
    )
    for axis, series in zip(axes[0], series_names):
        subset = frame[frame["series"] == series]
        x = subset["mc_return_delta_a_minus_b"].abs().to_numpy()
        policy = subset["policy_rep_zscored_rms_distance"].to_numpy()
        value = subset["value_rep_zscored_rms_distance"].to_numpy()
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
        axis.set_ylabel("Z-scored representation RMS distance")
        axis.grid(alpha=.25)
        axis.legend(frameon=False)
    figure.suptitle("Representation distance versus future-return difference",
                    fontweight="bold")
    figure.tight_layout()
    figure.savefig(output_dir / "representation_vs_return.png", dpi=300,
                   bbox_inches="tight")
    plt.close(figure)


def plot_cka(frame: pd.DataFrame, output_dir: Path) -> None:
    layouts = ordered_layouts(frame)
    series_names = ordered_series(frame)
    metrics = (
        ("policy_a_b_linear_cka", "Policy A↔B"),
        ("value_a_b_linear_cka", "Value A↔B"),
    )
    figure, axes = plt.subplots(
        1, len(series_names), figsize=(6.2 * len(series_names), 4.2), squeeze=False
    )
    x = np.arange(len(layouts))
    width = .8 / len(metrics)
    for axis, series in zip(axes[0], series_names):
        for offset, (column, label) in enumerate(metrics):
            mean, sem = mean_sem_by_seed(frame, column, layouts, series)
            positions = x + (offset - (len(metrics) - 1) / 2) * width
            axis.bar(
                positions, mean, width=width, yerr=sem, capsize=3,
                label=label, edgecolor="black", linewidth=.4,
            )
        axis.set_xticks(x)
        axis.set_xticklabels(
            [FAMILY_LABELS.get(layout, layout) for layout in layouts],
            rotation=25,
            ha="right",
        )
        axis.set_ylim(0, 1.05)
        axis.set_ylabel("Linear CKA")
        axis.set_title(series, fontweight="bold")
        axis.grid(axis="y", alpha=.25)
        axis.legend(frameon=False, fontsize=8)
    figure.suptitle("Cross-environment representation alignment", fontweight="bold")
    figure.tight_layout()
    figure.savefig(output_dir / "representation_cka.png", dpi=300,
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
    parser.add_argument(
        "--bootstrap-samples", type=int, default=10000,
        help="Seed-bootstrap samples for 95%% correlation confidence intervals",
    )
    args = parser.parse_args()
    if args.bootstrap_samples < 1:
        parser.error("--bootstrap-samples must be positive")

    input_dir = args.input_dir or (
        args.model_root.expanduser() / "analysis" / "policy_value_large_scale"
    )
    if not input_dir.is_dir():
        parser.error(f"Large-scale result directory does not exist: {input_dir}")
    output_dir = args.output_dir or input_dir / "figures"
    output_dir.mkdir(parents=True, exist_ok=True)

    frame = load_metrics(input_dir)
    try:
        cka_frame = load_cka(input_dir)
    except FileNotFoundError as error:
        parser.error(str(error))
    required = {
        "model", "num_envs", "seed", "pair_id", "mc_return_delta_a_minus_b",
        "policy_rep_cosine_distance", "value_rep_cosine_distance",
        "policy_rep_raw_euclidean_distance", "value_rep_raw_euclidean_distance",
        "policy_rep_zscored_euclidean_distance",
        "value_rep_zscored_euclidean_distance", "passes_policy_equivalence",
        "policy_rep_zscored_rms_distance", "value_rep_zscored_rms_distance",
        "passes_intended_interaction", "passes_return_distinction",
        "passes_concrete_example",
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        parser.error("Missing metrics columns: " + ", ".join(missing))
    required_cka = {"policy_a_b_linear_cka", "value_a_b_linear_cka"}
    missing_cka = sorted(required_cka - set(cka_frame.columns))
    if missing_cka:
        parser.error(
            "Missing CKA columns: " + ", ".join(missing_cka)
            + ". Re-run policy_value_fixed_pairs_metrics.py with the saved pairs."
        )

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
    plot_layout_distances(
        frame, output_dir, "zscored_rms",
        "policy_rep_zscored_rms_distance", "value_rep_zscored_rms_distance",
        "Z-scored RMS distance",
    )
    plot_return_scatter(frame, output_dir)
    common_pair_frame = retain_common_pairs(frame)
    seed_correlations = seed_layout_correlations(common_pair_frame)
    correlation_summary = summarize_seed_correlations(
        seed_correlations, args.bootstrap_samples
    )
    seed_correlations.to_csv(
        output_dir / "return_correlations_by_seed_layout.csv", index=False
    )
    correlation_summary.to_csv(
        output_dir / "return_correlations_summary.csv", index=False
    )
    plot_seed_layout_correlations(correlation_summary, common_pair_frame, output_dir)
    plot_cka(cka_frame, output_dir)
    plot_filter_rates(frame, output_dir)
    try:
        rsa_layout_count = plot_saved_rsa(input_dir)
    except FileNotFoundError as error:
        parser.error(str(error))
    print(f"Loaded {len(frame)} evaluations from {input_dir}")
    print(f"Regenerated RSA figures for {rsa_layout_count} layouts")
    print(f"Saved large-scale PNG figures to {output_dir}")


if __name__ == "__main__":
    main()
