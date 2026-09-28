"""Plot conditional CKA results from an existing summary CSV."""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


FAMILY_ORDER = (
    "asymm_advantages", "coord_ring", "counter_circuit",
    "forced_coord", "cramped_room",
)
FAMILY_LABELS = {
    "asymm_advantages": "Asymmetric Advantages",
    "coord_ring": "Coordination Ring",
    "counter_circuit": "Counter Circuit",
    "forced_coord": "Forced Coordination",
    "cramped_room": "Cramped Room",
}
MODEL_LABELS = {"CEC": "CEC", "CEC_IDAAC": "DCEC"}
MODEL_COLORS = {"CEC": "#117733", "CEC_IDAAC": "#56B4E9"}
POLICY_COLOR = "#0072B2"
VALUE_COLOR = "#D55E00"
CONDITION_LABELS = {
    "policy_equivalent": "Policy-equivalent",
    "interact_equivalent": "Policy-equivalent Interact",
}
CONDITION_ORDER = tuple(CONDITION_LABELS)
REQUIRED_COLUMNS = {
    "layout", "model", "num_envs", "seed", "condition",
    "selected_fraction", "policy_a_b_linear_cka", "value_a_b_linear_cka",
}


def mean_sem(values: pd.Series) -> tuple[float, float]:
    values = pd.to_numeric(values, errors="coerce").dropna().to_numpy(dtype=float)
    if not len(values):
        return float("nan"), 0.0
    sem = float(values.std(ddof=1) / np.sqrt(len(values))) if len(values) > 1 else 0.0
    return float(values.mean()), sem


def ordered_layouts(frame: pd.DataFrame) -> list[str]:
    present = set(frame["layout"])
    return [layout for layout in FAMILY_ORDER if layout in present] + sorted(
        present - set(FAMILY_ORDER)
    )


def ordered_models(frame: pd.DataFrame) -> list[str]:
    present = set(frame["model"])
    return [model for model in ("CEC", "CEC_IDAAC") if model in present] + sorted(
        present - {"CEC", "CEC_IDAAC"}
    )


def save_condition_figure(frame: pd.DataFrame, condition: str, output: Path) -> None:
    subset = frame[frame["condition"] == condition]
    models = ordered_models(subset)
    layouts = ordered_layouts(subset)
    figure, axes = plt.subplots(
        1, len(models), figsize=(6.4 * len(models), 4.8), squeeze=False,
        sharey=True,
    )
    x = np.arange(len(layouts))
    width = .36
    for axis, model in zip(axes[0], models):
        model_frame = subset[subset["model"] == model]
        for offset, (column, label, color) in enumerate((
            ("policy_a_b_linear_cka", "Policy", POLICY_COLOR),
            ("value_a_b_linear_cka", "Value", VALUE_COLOR),
        )):
            means, sems = [], []
            for layout in layouts:
                values = model_frame[model_frame["layout"] == layout][column]
                mean, sem = mean_sem(values)
                means.append(mean)
                sems.append(sem)
            positions = x + (-width / 2 if offset == 0 else width / 2)
            axis.bar(
                positions, means, width=width, yerr=sems, capsize=3,
                color=color, edgecolor="black", linewidth=.5, label=label,
            )
        axis.set_xticks(x)
        axis.set_xticklabels(
            [FAMILY_LABELS.get(layout, layout) for layout in layouts],
            rotation=25, ha="right",
        )
        axis.set_ylim(0, 1.05)
        axis.set_title(MODEL_LABELS.get(model, model), fontweight="bold")
        axis.set_ylabel("Conditional linear CKA")
        axis.grid(axis="y", alpha=.25)
        axis.legend(frameon=False)
    figure.suptitle(CONDITION_LABELS.get(condition, condition), fontweight="bold")
    figure.tight_layout()
    figure.savefig(output, bbox_inches="tight")
    plt.close(figure)


def save_selection_figure(frame: pd.DataFrame, output: Path) -> None:
    models = ordered_models(frame)
    conditions = [
        condition for condition in CONDITION_ORDER
        if condition in set(frame["condition"])
    ]
    conditions += sorted(set(frame["condition"]) - set(conditions))
    layouts = ordered_layouts(frame)
    figure, axes = plt.subplots(
        1, len(models), figsize=(6.4 * len(models), 4.6), squeeze=False,
        sharey=True,
    )
    x = np.arange(len(layouts))
    width = .8 / len(conditions)
    for axis, model in zip(axes[0], models):
        model_frame = frame[frame["model"] == model]
        for offset, condition in enumerate(conditions):
            means = [
                model_frame[
                    (model_frame["layout"] == layout)
                    & (model_frame["condition"] == condition)
                ]["selected_fraction"].mean()
                for layout in layouts
            ]
            positions = x + (offset - (len(conditions) - 1) / 2) * width
            axis.bar(
                positions, means, width=width,
                label=CONDITION_LABELS.get(condition, condition),
                color=MODEL_COLORS.get(model, ".5"), alpha=.55 + .35 * offset,
                edgecolor="black", linewidth=.5,
            )
        axis.set_xticks(x)
        axis.set_xticklabels(
            [FAMILY_LABELS.get(layout, layout) for layout in layouts],
            rotation=25, ha="right",
        )
        axis.set_ylim(0, 1.05)
        axis.set_title(MODEL_LABELS.get(model, model), fontweight="bold")
        axis.set_ylabel("Fraction of fixed pairs retained")
        axis.grid(axis="y", alpha=.25)
        axis.legend(frameon=False, fontsize=8)
    figure.tight_layout()
    figure.savefig(output, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-root", type=Path, required=True)
    parser.add_argument("--input", type=Path)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    default_dir = (
        args.model_root.expanduser() / "analysis" / "policy_value_conditional_cka"
    )
    input_path = args.input.expanduser() if args.input else (
        default_dir / "conditional_cka_summary.csv"
    )
    output_dir = args.output_dir.expanduser() if args.output_dir else default_dir
    if not input_path.is_file():
        parser.error(f"Conditional CKA summary does not exist: {input_path}")
    frame = pd.read_csv(input_path)
    missing = sorted(REQUIRED_COLUMNS - set(frame))
    if missing:
        parser.error("Missing summary columns: " + ", ".join(missing))
    if frame.empty:
        parser.error(f"Conditional CKA summary is empty: {input_path}")
    output_dir.mkdir(parents=True, exist_ok=True)
    for condition in frame["condition"].drop_duplicates():
        save_condition_figure(
            frame, condition, output_dir / f"conditional_cka_{condition}.pdf"
        )
    save_selection_figure(
        frame, output_dir / "conditional_cka_selection_rates.pdf"
    )
    print(f"Loaded conditional CKA results from {input_path}")
    print(f"Saved conditional CKA figures to {output_dir}")


if __name__ == "__main__":
    main()
