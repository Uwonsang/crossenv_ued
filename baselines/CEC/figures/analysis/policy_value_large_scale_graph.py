"""Visualize layout-wise outputs from policy_value_concrete_example.py --large-scale."""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


REPOSITORY_ROOT = Path(__file__).resolve().parents[4]
DEFAULT_GENERALIZATION_DIR = REPOSITORY_ROOT / "artifacts" / "generalization_gap"

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
FAMILY_ABBREVIATIONS = {
    "asymm_advantages": "AA",
    "counter_circuit": "CC",
    "coord_ring": "CR",
    "forced_coord": "FC",
    "cramped_room": "CRoom",
}
PAPER_RC = {
    "font.family": "DejaVu Sans",
    "font.size": 24,
    "axes.titlesize": 28,
    "axes.titleweight": "bold",
    "axes.labelsize": 24,
    "xtick.labelsize": 24,
    "ytick.labelsize": 24,
    "figure.titlesize": 28,
    "legend.fontsize": 24,
}
MODEL_LABELS = {"CEC": "CEC", "CEC_IDAAC": "DCEC"}
MODEL_COLORS = {"CEC": "#117733", "CEC_IDAAC": "#0072B2"}
POLICY_COLOR = "#0072B2"
VALUE_COLOR = "#D55E00"

GENERALIZATION_LAYOUTS = {
    "asymm_advantages_9": "asymm_advantages",
    "coord_ring_9": "coord_ring",
    "counter_circuit_9": "counter_circuit",
    "forced_coord_9": "forced_coord",
    "cramped_room_9": "cramped_room",
}


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


def load_rsa(input_dir: Path) -> pd.DataFrame:
    frames = []
    for family_dir in sorted(path for path in input_dir.iterdir() if path.is_dir()):
        metrics = family_dir / "concrete_example_rsa.csv"
        if not metrics.is_file():
            continue
        frame = pd.read_csv(metrics)
        frame["layout"] = family_dir.name
        frames.append(frame)
    if not frames:
        raise FileNotFoundError(
            f"No layout-level concrete_example_rsa.csv files found under {input_dir}"
        )
    return pd.concat(frames, ignore_index=True)


def plot_saved_rsa(input_dir: Path) -> int:
    """Regenerate per-layout RSA figures from already evaluated CSV files."""
    from policy_value_fixed_pairs_metrics import save_rsa_figures

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


def action_consistency_by_seed(frame: pd.DataFrame) -> pd.DataFrame:
    """Aggregate matched-pair behavior consistency within each checkpoint."""
    data = frame.copy()
    data["action_agreement"] = as_rate(data["argmax_same"])
    return data.groupby(
        ["series", "model", "num_envs", "layout", "seed"], as_index=False
    ).agg(
        pairs=("pair_id", "nunique"),
        mean_policy_js_nats=("policy_js_nats", "mean"),
        action_agreement=("action_agreement", "mean"),
    )


def summarize_action_consistency(seed_frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    keys = ["series", "model", "num_envs", "layout"]
    for group_key, subset in seed_frame.groupby(keys, sort=False):
        series, model, num_envs, layout = group_key
        row = {
            "series": series,
            "model": model,
            "num_envs": int(num_envs),
            "layout": layout,
            "seeds": int(subset["seed"].nunique()),
            "mean_pairs_per_seed": float(subset["pairs"].mean()),
        }
        for metric in ("mean_policy_js_nats", "action_agreement"):
            values = subset[metric].to_numpy(dtype=float)
            row[f"{metric}_mean"] = float(values.mean())
            row[f"{metric}_sem"] = (
                float(values.std(ddof=1) / np.sqrt(len(values)))
                if len(values) > 1 else 0.0
            )
        rows.append(row)
    return pd.DataFrame(rows)


def aggregate_action_consistency_across_layouts(
    seed_frame: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Give each held-out layout equal weight within every checkpoint seed."""
    keys = ["series", "model", "num_envs", "seed"]
    aggregate_by_seed = seed_frame.groupby(keys, as_index=False).agg(
        layouts=("layout", "nunique"),
        pairs=("pairs", "sum"),
        mean_policy_js_nats=("mean_policy_js_nats", "mean"),
        action_agreement=("action_agreement", "mean"),
    )
    rows = []
    for group_key, subset in aggregate_by_seed.groupby(
        ["series", "model", "num_envs"], sort=False
    ):
        series, model, num_envs = group_key
        row = {
            "series": series,
            "model": model,
            "num_envs": int(num_envs),
            "seeds": int(subset["seed"].nunique()),
            "mean_layouts_per_seed": float(subset["layouts"].mean()),
            "mean_pairs_per_seed": float(subset["pairs"].mean()),
        }
        for metric in ("mean_policy_js_nats", "action_agreement"):
            values = subset[metric].to_numpy(dtype=float)
            row[f"{metric}_mean"] = float(values.mean())
            row[f"{metric}_sem"] = (
                float(values.std(ddof=1) / np.sqrt(len(values)))
                if len(values) > 1 else 0.0
            )
        rows.append(row)
    return aggregate_by_seed, pd.DataFrame(rows)


def plot_action_consistency(
    summary: pd.DataFrame, frame: pd.DataFrame, output_dir: Path
) -> None:
    """Create Figure 3(a): JS divergence and argmax-action agreement."""
    requested_order = (
        "asymm_advantages", "counter_circuit", "coord_ring",
        "forced_coord", "cramped_room",
    )
    present = set(summary["layout"])
    layouts = [layout for layout in requested_order if layout in present]
    series_names = ordered_series(frame)
    x = np.arange(len(layouts))
    width = .8 / len(series_names)
    metrics = (
        ("mean_policy_js_nats", "Policy JS divergence (nats)"),
        ("action_agreement", "Action agreement"),
    )
    with plt.rc_context(PAPER_RC):
        figure, axes = plt.subplots(1, 2, figsize=(20.0, 5.694), squeeze=False)
        for axis, (metric, ylabel) in zip(axes[0], metrics):
            for offset, series in enumerate(series_names):
                subset = summary[summary["series"] == series].set_index(
                    "layout"
                ).reindex(layouts)
                positions = x + (offset - (len(series_names) - 1) / 2) * width
                model_values = subset["model"].dropna()
                model = model_values.iloc[0] if len(model_values) else series
                label = series.rsplit(" (", 1)[0]
                axis.bar(
                    positions,
                    subset[f"{metric}_mean"],
                    width=width,
                    yerr=subset[f"{metric}_sem"],
                    capsize=3,
                    color=MODEL_COLORS.get(model, ".5"),
                    edgecolor="black",
                    linewidth=.5,
                    label=label,
                )
            axis.set_xticks(x)
            axis.set_xticklabels([
                FAMILY_ABBREVIATIONS.get(layout, layout) for layout in layouts
            ])
            axis.set_ylabel(ylabel)
            axis.grid(axis="y", alpha=.25)
        axes[0, 1].set_ylim(0, 1.05)
        handles, labels = axes[0, 0].get_legend_handles_labels()
        figure.legend(
            handles, labels, loc="upper center", bbox_to_anchor=(.5, .99),
            ncol=max(1, len(labels)), frameon=False,
        )
        figure.tight_layout(rect=(0, 0, 1, .86), w_pad=2.0)
        figure.savefig(
            output_dir / "action_distribution_consistency.pdf",
            bbox_inches="tight", pad_inches=0,
        )
        figure.savefig(
            output_dir / "action_distribution_consistency_by_layout.pdf",
            bbox_inches="tight", pad_inches=0,
        )
        plt.close(figure)


def plot_aggregate_action_consistency(
    summary: pd.DataFrame, output_dir: Path
) -> None:
    """Plot the equal-layout aggregate used for the main consistency result."""
    series_names = [
        series for series in ordered_series(summary)
        if series in set(summary["series"])
    ]
    metrics = (
        ("mean_policy_js_nats", "Policy JS divergence (nats)"),
        ("action_agreement", "Action agreement"),
    )
    x = np.arange(len(series_names))
    with plt.rc_context(PAPER_RC):
        figure, axes = plt.subplots(1, 2, figsize=(13.0, 5.694), squeeze=False)
        for axis, (metric, ylabel) in zip(axes[0], metrics):
            means, sems, colors, labels = [], [], [], []
            for series in series_names:
                row = summary[summary["series"] == series].iloc[0]
                means.append(float(row[f"{metric}_mean"]))
                sems.append(float(row[f"{metric}_sem"]))
                colors.append(MODEL_COLORS.get(row["model"], ".5"))
                labels.append(series.rsplit(" (", 1)[0])
            axis.bar(
                x, means, yerr=sems, capsize=4, width=.62,
                color=colors, edgecolor="black", linewidth=.6,
            )
            axis.set_xticks(x)
            axis.set_xticklabels(labels)
            axis.set_ylabel(ylabel)
            axis.grid(axis="y", alpha=.25)
        axes[0, 1].set_ylim(0, 1.05)
        figure.tight_layout(w_pad=2.0)
        figure.savefig(
            output_dir / "action_distribution_consistency_aggregate.pdf",
            bbox_inches="tight", pad_inches=0,
        )
        plt.close(figure)


def save_action_consistency_outputs(
    frame: pd.DataFrame, output_dir: Path
) -> pd.DataFrame:
    """Save Figure 3(a) data/plots and return its common-pair frame."""
    common_pair_frame = retain_common_pairs(frame)
    action_seed_summary = action_consistency_by_seed(common_pair_frame)
    action_summary = summarize_action_consistency(action_seed_summary)
    action_aggregate_by_seed, action_aggregate_summary = (
        aggregate_action_consistency_across_layouts(action_seed_summary)
    )
    action_seed_summary.to_csv(
        output_dir / "action_distribution_consistency_by_seed.csv", index=False
    )
    action_summary.to_csv(
        output_dir / "action_distribution_consistency_summary.csv", index=False
    )
    action_aggregate_by_seed.to_csv(
        output_dir / "action_distribution_consistency_aggregate_by_seed.csv",
        index=False,
    )
    action_aggregate_summary.to_csv(
        output_dir / "action_distribution_consistency_aggregate_summary.csv",
        index=False,
    )
    plot_action_consistency(action_summary, common_pair_frame, output_dir)
    plot_aggregate_action_consistency(action_aggregate_summary, output_dir)
    return common_pair_frame


def plot_layout_distances(
    frame: pd.DataFrame,
    output_dir: Path,
    suffix: str,
    policy_column: str,
    value_column: str,
    ylabel: str,
) -> None:
    paper_version = suffix == "zscored_rms"
    layouts = ordered_layouts(frame)
    if paper_version:
        requested_order = (
            "asymm_advantages", "counter_circuit", "coord_ring",
            "forced_coord", "cramped_room",
        )
        present = set(layouts)
        layouts = [layout for layout in requested_order if layout in present]
    series_names = ordered_series(frame)
    rc = PAPER_RC if paper_version else {}
    subplot_width, figure_height = (
        (10.0, 5.694) if paper_version else (6.2, 4.2)
    )
    with plt.rc_context(rc):
        figure, axes = plt.subplots(
            1, len(series_names),
            figsize=(subplot_width * len(series_names), figure_height),
            squeeze=False, sharey=paper_version,
        )
        x = np.arange(len(layouts))
        width = .36
        for axis_index, (axis, series) in enumerate(zip(axes[0], series_names)):
            policy_mean, policy_sem = mean_sem_by_seed(
                frame, policy_column, layouts, series
            )
            value_mean, value_sem = mean_sem_by_seed(
                frame, value_column, layouts, series
            )
            axis.bar(
                x - width / 2, policy_mean, width, yerr=policy_sem, capsize=3,
                color=POLICY_COLOR, edgecolor="black", linewidth=.5,
                label="Policy",
            )
            axis.bar(
                x + width / 2, value_mean, width, yerr=value_sem, capsize=3,
                color=VALUE_COLOR, edgecolor="black", linewidth=.5,
                label="Value",
            )
            axis.set_xticks(x)
            axis.set_xticklabels(
                [
                    (FAMILY_ABBREVIATIONS if paper_version else FAMILY_LABELS)
                    .get(layout, layout)
                    for layout in layouts
                ],
                rotation=0 if paper_version else 25,
                ha="center" if paper_version else "right",
            )
            series_title = series.rsplit(" (", 1)[0] if paper_version else series
            axis.set_title(
                series_title, fontweight="bold",
                fontsize=28 if paper_version else None,
                pad=10 if paper_version else None,
            )
            axis.set_ylabel(
                "Normalized distance"
                if paper_version and axis_index == 0 else
                ("" if paper_version else ylabel)
            )
            if paper_version:
                axis.tick_params(axis="y", labelleft=True)
            axis.grid(axis="y", alpha=.25)
            if not paper_version:
                axis.legend(frameon=False)
        if paper_version:
            handles, labels = axes[0, 0].get_legend_handles_labels()
            figure.legend(
                handles, labels, loc="upper center",
                bbox_to_anchor=(.5, .99), ncol=2, frameon=False,
            )
        else:
            figure.suptitle(
                "Policy and value representation distance",
                fontweight="bold", y=.99,
            )
        figure.tight_layout(
            rect=(0, 0, 1, .86 if paper_version else .93), w_pad=2.0
        )
        extension = "pdf" if paper_version else "png"
        save_kwargs = {
            "bbox_inches": "tight",
            "pad_inches": 0 if paper_version else .1,
        }
        if not paper_version:
            save_kwargs["dpi"] = 300
        figure.savefig(
            output_dir / f"representation_distance_{suffix}.{extension}",
            **save_kwargs,
        )
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


def resolve_generalization_csv(
    requested: Path | None, model_root: Path
) -> Path | None:
    """Find the run-level results for the original five Overcooked layouts."""
    if requested is not None:
        requested = requested.expanduser()
        if not requested.is_file():
            raise FileNotFoundError(f"Generalization CSV does not exist: {requested}")
        return requested
    patterns = (
        "generalization_gap_runs_final_window_*.csv",
        "generalization_gap_runs_[0-9]*m.csv",  # legacy final-window name
        "generalization_gap_runs_last_*.csv",
    )
    search_directories = (
        model_root.expanduser() / "artifacts" / "generalization_gap",
        DEFAULT_GENERALIZATION_DIR,
    )
    for directory in search_directories:
        for pattern in patterns:
            matches = sorted(directory.glob(pattern))
            if matches:
                return matches[-1]
    return None


def aggregate_representation_statistics(frame: pd.DataFrame) -> pd.DataFrame:
    """Reduce fixed-pair measurements to one row per checkpoint and layout."""
    keys = ["model", "num_envs", "seed", "layout"]
    result = frame.groupby(keys, as_index=False).agg(
        pairs=("pair_id", "nunique"),
        policy_environment_sensitivity=(
            "policy_rep_zscored_rms_distance", "mean"
        ),
        value_environment_sensitivity=(
            "value_rep_zscored_rms_distance", "mean"
        ),
    )
    result["leakage_score"] = (
        result["policy_environment_sensitivity"]
        - result["value_environment_sensitivity"]
    )
    return result


def aggregate_rsa_statistics(rsa_frame: pd.DataFrame) -> pd.DataFrame:
    """Put cosine and z-scored-RMS RSA summaries on checkpoint rows."""
    keys = ["model", "num_envs", "seed", "layout"]
    metrics = [
        "policy_behavior_rsa", "policy_return_rsa",
        "value_behavior_rsa", "value_return_rsa",
        "policy_behavior_minus_return", "value_return_minus_behavior",
        "rsa_asymmetry_score",
    ]
    missing = sorted(set(keys + ["distance_metric", *metrics]) - set(rsa_frame))
    if missing:
        raise ValueError("Missing RSA columns: " + ", ".join(missing))
    wide = rsa_frame.pivot_table(
        index=keys, columns="distance_metric", values=metrics, aggfunc="mean"
    )
    wide.columns = [f"rsa_{distance}_{metric}" for metric, distance in wide.columns]
    return wide.reset_index()


def merge_generalization_results(
    pair_frame: pd.DataFrame, rsa_frame: pd.DataFrame, generalization_csv: Path
) -> pd.DataFrame:
    """Join representation diagnostics to held-out results by run and layout."""
    representation = aggregate_representation_statistics(pair_frame)
    rsa = aggregate_rsa_statistics(rsa_frame)
    keys = ["model", "num_envs", "seed", "layout"]
    representation = representation.merge(rsa, on=keys, how="left", validate="one_to_one")

    generalization = pd.read_csv(generalization_csv)
    required = {
        "model", "num_envs", "seed", "eval_layout", "eval_return_mean",
        "generalization_gap", "relative_gap",
    }
    missing = sorted(required - set(generalization))
    if missing:
        raise ValueError("Missing generalization columns: " + ", ".join(missing))
    generalization = generalization[
        generalization["eval_layout"].isin(GENERALIZATION_LAYOUTS)
    ].copy()
    generalization["layout"] = generalization["eval_layout"].map(
        GENERALIZATION_LAYOUTS
    )
    numeric = ["eval_return_mean", "generalization_gap", "relative_gap"]
    generalization = generalization.groupby(keys, as_index=False).agg(
        held_out_runs=("eval_return_mean", "size"),
        **{column: (column, "mean") for column in numeric},
    )
    return representation.merge(
        generalization, on=keys, how="inner", validate="one_to_one"
    )


def generalization_correlations(merged: pd.DataFrame) -> pd.DataFrame:
    """Compute raw Spearman associations at useful, explicit scopes."""
    metric_columns = [
        "policy_environment_sensitivity", "value_environment_sensitivity",
        "leakage_score",
        *sorted(column for column in merged if column.startswith("rsa_")),
    ]
    targets = ["eval_return_mean", "generalization_gap", "relative_gap"]
    grouping_specs = (
        ("overall", []),
        ("model", ["model"]),
        ("model_layout", ["model", "layout"]),
        ("model_num_envs", ["model", "num_envs"]),
    )
    rows = []
    for scope, group_columns in grouping_specs:
        grouped = [((), merged)] if not group_columns else merged.groupby(
            group_columns, sort=False, dropna=False
        )
        for group_key, subset in grouped:
            if group_columns and not isinstance(group_key, tuple):
                group_key = (group_key,)
            labels = dict(zip(group_columns, group_key)) if group_columns else {}
            for metric in metric_columns:
                for target in targets:
                    valid = subset[[metric, target]].replace(
                        [np.inf, -np.inf], np.nan
                    ).dropna()
                    rows.append({
                        "scope": scope,
                        "model": labels.get("model", "all"),
                        "num_envs": labels.get("num_envs", "all"),
                        "layout": labels.get("layout", "all"),
                        "metric": metric,
                        "target": target,
                        "n": len(valid),
                        "spearman_rho": spearman_correlation(
                            valid[metric].to_numpy(dtype=float),
                            valid[target].to_numpy(dtype=float),
                        ),
                    })
    return pd.DataFrame(rows)


def plot_generalization_relationship(
    merged: pd.DataFrame, metric: str, metric_label: str, output: Path
) -> None:
    """Plot checkpoint/layout diagnostics against held-out performance."""
    models = [model for model in ("CEC", "CEC_IDAAC") if model in set(merged["model"])]
    if not models or metric not in merged:
        return
    targets = (
        ("eval_return_mean", "Held-out return"),
        ("generalization_gap", "Generalization gap"),
    )
    figure, axes = plt.subplots(
        len(targets), len(models), figsize=(6.0 * len(models), 8.2),
        squeeze=False, sharex="col",
    )
    for column, model in enumerate(models):
        subset = merged[merged["model"] == model]
        for row, (target, target_label) in enumerate(targets):
            axis = axes[row, column]
            valid = subset[[metric, target, "layout"]].dropna()
            for layout, points in valid.groupby("layout", sort=False):
                axis.scatter(
                    points[metric], points[target], s=42, alpha=.72,
                    label=FAMILY_ABBREVIATIONS.get(layout, layout),
                )
            if len(valid) >= 2 and valid[metric].nunique() > 1:
                slope, intercept = np.polyfit(valid[metric], valid[target], 1)
                x_line = np.linspace(valid[metric].min(), valid[metric].max(), 100)
                axis.plot(x_line, slope * x_line + intercept, color="black", linewidth=1)
            rho = spearman_correlation(
                valid[metric].to_numpy(dtype=float),
                valid[target].to_numpy(dtype=float),
            )
            axis.text(.03, .97, f"Spearman $\\rho$={rho:.2f}\nn={len(valid)}",
                      transform=axis.transAxes, ha="left", va="top")
            axis.set_title(MODEL_LABELS.get(model, model), fontweight="bold")
            axis.set_ylabel(target_label)
            axis.grid(alpha=.25)
            if row == len(targets) - 1:
                axis.set_xlabel(metric_label)
            if row == 0 and column == len(models) - 1:
                axis.legend(frameon=False, title="Layout", ncol=2, fontsize=9)
    figure.tight_layout()
    figure.savefig(output, bbox_inches="tight")
    plt.close(figure)


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
        "--action-consistency-only", action="store_true",
        help=(
            "Only create JS-divergence/action-agreement CSVs and figures. "
            "This also supports concrete-example directories without CKA/RSA CSVs."
        ),
    )
    parser.add_argument(
        "--generalization-csv", type=Path,
        help=(
            "Run-level generalization_gap CSV for the original five Overcooked "
            "layouts. If omitted, discover it under artifacts/generalization_gap."
        ),
    )
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
    action_required = {
        "model", "num_envs", "seed", "pair_id", "policy_js_nats",
        "argmax_same",
    }
    missing_action = sorted(action_required - set(frame.columns))
    if missing_action:
        parser.error(
            "Missing action-consistency columns: " + ", ".join(missing_action)
        )
    if args.action_consistency_only:
        save_action_consistency_outputs(frame, output_dir)
        print(f"Loaded {len(frame)} evaluations from {input_dir}")
        print(f"Saved action-consistency figures to {output_dir}")
        return

    try:
        cka_frame = load_cka(input_dir)
    except FileNotFoundError as error:
        parser.error(str(error))
    required = {
        "model", "num_envs", "seed", "pair_id", "mc_return_delta_a_minus_b",
        "policy_js_nats", "argmax_same",
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
    common_pair_frame = save_action_consistency_outputs(frame, output_dir)
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
        rsa_frame = load_rsa(input_dir)
        rsa_layout_count = plot_saved_rsa(input_dir)
    except FileNotFoundError as error:
        parser.error(str(error))

    try:
        generalization_csv = resolve_generalization_csv(
            args.generalization_csv, args.model_root
        )
    except FileNotFoundError as error:
        parser.error(str(error))
    if generalization_csv is None:
        print(
            "No generalization-gap CSV found; skipping held-out prediction analysis. "
            "Pass --generalization-csv to enable it."
        )
    else:
        try:
            merged = merge_generalization_results(
                common_pair_frame, rsa_frame, generalization_csv
            )
        except ValueError as error:
            parser.error(str(error))
        if merged.empty:
            parser.error(
                "No rows matched between fixed-pair metrics and held-out results "
                "on model / num_envs / seed / layout."
            )
        correlations = generalization_correlations(merged)
        merged.to_csv(
            output_dir / "representation_generalization_merged.csv", index=False
        )
        correlations.to_csv(
            output_dir / "representation_generalization_correlations.csv", index=False
        )
        plot_generalization_relationship(
            merged, "leakage_score", r"Leakage score $d_\pi-d_V$",
            output_dir / "leakage_vs_generalization.pdf",
        )
        for distance in ("cosine", "zscored_rms"):
            metric = f"rsa_{distance}_rsa_asymmetry_score"
            plot_generalization_relationship(
                merged, metric, f"RSA asymmetry ({distance.replace('_', ' ')})",
                output_dir / f"rsa_asymmetry_{distance}_vs_generalization.pdf",
            )
        represented_layouts = set(merged["layout"])
        missing_layouts = set(GENERALIZATION_LAYOUTS.values()) - represented_layouts
        print(
            f"Matched {len(merged)} representation/held-out rows from "
            f"{generalization_csv}"
        )
        if missing_layouts:
            print(
                "No matched representation rows for: "
                + ", ".join(sorted(missing_layouts))
            )
    print(f"Loaded {len(frame)} evaluations from {input_dir}")
    print(f"Regenerated RSA figures for {rsa_layout_count} layouts")
    print(f"Saved large-scale PNG figures to {output_dir}")


if __name__ == "__main__":
    main()
