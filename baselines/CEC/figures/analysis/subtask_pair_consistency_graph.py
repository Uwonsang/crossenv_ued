"""Plot saved matched-subtask policy-consistency metrics."""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


SUBTASK_ORDER = (
    "plate_pickup", "cooked_soup_pickup", "serve", "onion_pickup",
    "onion_to_pot",
)
SUBTASK_LABELS = {
    "plate_pickup": "Plate\npickup",
    "cooked_soup_pickup": "Soup\npickup",
    "serve": "Serve",
    "onion_pickup": "Onion\npickup",
    "onion_to_pot": "Third onion\nto pot",
}
FAMILY_ORDER = (
    "asymm_advantages", "counter_circuit", "coord_ring",
    "forced_coord", "cramped_room",
)
FAMILY_LABELS = {
    "asymm_advantages": "AA", "counter_circuit": "CC",
    "coord_ring": "CR", "forced_coord": "FC", "cramped_room": "CRoom",
}
MODEL_LABELS = {"CEC": "CEC", "CEC_IDAAC": "DCEC"}
MODEL_COLORS = {"CEC": "#117733", "CEC_IDAAC": "#56B4E9"}
METRICS = (
    ("policy_js_nats", "Policy JS divergence (nats)"),
    ("action_agreement", "Action agreement"),
    ("both_interact", "Both choose Interact"),
)
REQUIRED = {
    "family", "pair_id", "subtask", "model", "num_envs", "seed",
    "policy_js_nats", "action_agreement", "both_interact",
}


plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 18,
    "axes.titlesize": 20,
    "axes.titleweight": "bold",
    "axes.labelsize": 18,
    "xtick.labelsize": 16,
    "ytick.labelsize": 16,
    "legend.fontsize": 18,
})


def as_float(series: pd.Series) -> pd.Series:
    if series.dtype == bool:
        return series.astype(float)
    mapped = series.astype(str).str.lower().map({"true": 1.0, "false": 0.0})
    return pd.to_numeric(series, errors="coerce").fillna(mapped)


def load_metrics(input_dir: Path) -> pd.DataFrame:
    frames = []
    direct = input_dir / "subtask_consistency_metrics.csv"
    paths = [direct] if direct.is_file() else []
    paths.extend(sorted(input_dir.glob("*/subtask_consistency_metrics.csv")))
    for path in paths:
        frame = pd.read_csv(path)
        missing = sorted(REQUIRED - set(frame))
        if missing:
            raise ValueError(f"{path}: missing columns {missing}")
        frames.append(frame)
    if not frames:
        raise FileNotFoundError(
            f"No subtask_consistency_metrics.csv found under {input_dir}"
        )
    frame = pd.concat(frames, ignore_index=True)
    for column in ("action_agreement", "both_interact"):
        frame[column] = as_float(frame[column])
    frame["series"] = frame.apply(
        lambda row: f"{MODEL_LABELS.get(row['model'], row['model'])} "
                    f"({int(row['num_envs'])})",
        axis=1,
    )
    return frame


def aggregate_seed_level(frame: pd.DataFrame) -> pd.DataFrame:
    return frame.groupby(
        ["series", "model", "num_envs", "seed", "family", "subtask"],
        as_index=False,
    ).agg(
        pairs=("pair_id", "nunique"),
        policy_js_nats=("policy_js_nats", "mean"),
        action_agreement=("action_agreement", "mean"),
        both_interact=("both_interact", "mean"),
    )


def summarize(seed_frame: pd.DataFrame, group_columns: list[str]) -> pd.DataFrame:
    rows = []
    keys = ["series", "model", "num_envs", *group_columns]
    for group_key, subset in seed_frame.groupby(keys, sort=False):
        if not isinstance(group_key, tuple):
            group_key = (group_key,)
        row = dict(zip(keys, group_key))
        row["seeds"] = int(subset["seed"].nunique())
        for metric, _ in METRICS:
            values = subset[metric].to_numpy(dtype=float)
            row[f"{metric}_mean"] = float(values.mean())
            row[f"{metric}_sem"] = (
                float(values.std(ddof=1) / np.sqrt(len(values)))
                if len(values) > 1 else 0.0
            )
        rows.append(row)
    return pd.DataFrame(rows)


def equal_weight_aggregate(
    seed_frame: pd.DataFrame, keep: str | None
) -> pd.DataFrame:
    keys = ["series", "model", "num_envs", "seed"]
    if keep is not None:
        keys.append(keep)
    return seed_frame.groupby(keys, as_index=False).agg(**{
        metric: (metric, "mean") for metric, _ in METRICS
    })


def ordered_series(frame: pd.DataFrame) -> list[str]:
    order = frame[["series", "model", "num_envs"]].drop_duplicates().copy()
    order["rank"] = order["model"].map({"CEC": 0, "CEC_IDAAC": 1}).fillna(2)
    return order.sort_values(["rank", "num_envs"])["series"].tolist()


def plot_grouped(
    summary: pd.DataFrame,
    category: str,
    categories: list[str],
    labels: list[str],
    output: Path,
) -> None:
    series_names = ordered_series(summary)
    figure, axes = plt.subplots(1, 3, figsize=(20, 6.0), squeeze=False)
    x = np.arange(len(categories))
    width = .8 / len(series_names)
    for axis, (metric, ylabel) in zip(axes[0], METRICS):
        for offset, series in enumerate(series_names):
            subset = (
                summary[summary["series"] == series]
                .set_index(category).reindex(categories)
            )
            positions = x + (offset - (len(series_names) - 1) / 2) * width
            model_values = subset["model"].dropna()
            model = model_values.iloc[0] if len(model_values) else series
            axis.bar(
                positions,
                subset[f"{metric}_mean"],
                width=width,
                yerr=subset[f"{metric}_sem"],
                capsize=3,
                color=MODEL_COLORS.get(model, ".5"),
                edgecolor="black",
                linewidth=.5,
                label=series.rsplit(" (", 1)[0],
            )
        axis.set_xticks(x)
        axis.set_xticklabels(labels)
        axis.set_ylabel(ylabel)
        axis.grid(axis="y", alpha=.25)
    axes[0, 1].set_ylim(0, 1.05)
    axes[0, 2].set_ylim(0, 1.05)
    handles, legend_labels = axes[0, 0].get_legend_handles_labels()
    figure.legend(
        handles, legend_labels, loc="upper center", bbox_to_anchor=(.5, 1.0),
        ncol=len(legend_labels), frameon=False,
    )
    figure.tight_layout(rect=(0, 0, 1, .9), w_pad=2)
    figure.savefig(output, bbox_inches="tight", pad_inches=.02)
    plt.close(figure)


def plot_overall(summary: pd.DataFrame, output: Path) -> None:
    series_names = ordered_series(summary)
    x = np.arange(len(series_names))
    figure, axes = plt.subplots(1, 3, figsize=(15, 5.6), squeeze=False)
    for axis, (metric, ylabel) in zip(axes[0], METRICS):
        rows = summary.set_index("series").reindex(series_names)
        colors = [MODEL_COLORS.get(model, ".5") for model in rows["model"]]
        axis.bar(
            x, rows[f"{metric}_mean"], yerr=rows[f"{metric}_sem"],
            capsize=4, width=.62, color=colors,
            edgecolor="black", linewidth=.5,
        )
        axis.set_xticks(x)
        axis.set_xticklabels([
            series.rsplit(" (", 1)[0] for series in series_names
        ])
        axis.set_ylabel(ylabel)
        axis.grid(axis="y", alpha=.25)
    axes[0, 1].set_ylim(0, 1.05)
    axes[0, 2].set_ylim(0, 1.05)
    figure.tight_layout(w_pad=2)
    figure.savefig(output, bbox_inches="tight", pad_inches=.02)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-root", type=Path, required=True)
    parser.add_argument("--input-dir", type=Path)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    input_dir = args.input_dir or (
        args.model_root.expanduser() / "analysis" / "subtask_pair_consistency"
    )
    output_dir = args.output_dir or input_dir / "figures"
    output_dir.mkdir(parents=True, exist_ok=True)
    try:
        frame = load_metrics(input_dir)
    except (FileNotFoundError, ValueError) as error:
        parser.error(str(error))
    seed_frame = aggregate_seed_level(frame)

    by_subtask_seed = equal_weight_aggregate(seed_frame, "subtask")
    by_subtask = summarize(by_subtask_seed, ["subtask"])
    subtasks = [
        subtask for subtask in SUBTASK_ORDER
        if subtask in set(by_subtask["subtask"])
    ]
    plot_grouped(
        by_subtask, "subtask", subtasks,
        [SUBTASK_LABELS[subtask] for subtask in subtasks],
        output_dir / "subtask_consistency_by_subtask.pdf",
    )

    by_layout_seed = equal_weight_aggregate(seed_frame, "family")
    by_layout = summarize(by_layout_seed, ["family"])
    families = [
        family for family in FAMILY_ORDER if family in set(by_layout["family"])
    ]
    plot_grouped(
        by_layout, "family", families,
        [FAMILY_LABELS.get(family, family) for family in families],
        output_dir / "subtask_consistency_by_layout.pdf",
    )

    overall_seed = equal_weight_aggregate(seed_frame, None)
    overall = summarize(overall_seed, [])
    plot_overall(overall, output_dir / "subtask_consistency_overall.pdf")

    seed_frame.to_csv(output_dir / "subtask_consistency_by_seed.csv", index=False)
    by_subtask.to_csv(
        output_dir / "subtask_consistency_by_subtask.csv", index=False
    )
    by_layout.to_csv(
        output_dir / "subtask_consistency_by_layout.csv", index=False
    )
    overall.to_csv(output_dir / "subtask_consistency_overall.csv", index=False)
    print(f"Loaded {len(frame)} evaluations from {input_dir}")
    print(f"Saved subtask consistency figures to {output_dir}")


if __name__ == "__main__":
    main()
