"""Visualize model-level CSVs produced by environment_representation_probe.py."""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 18,
    "axes.titlesize": 22,
    "axes.titleweight": "bold",
    "axes.labelsize": 20,
    "xtick.labelsize": 18,
    "ytick.labelsize": 18,
    "figure.titlesize": 22,
    "legend.fontsize": 18,
})


REQUIRED_COLUMNS = {
    "model",
    "model_label",
    "model_num_envs",
    "checkpoint_seed",
    "probe_split",
    "accuracy",
    "chance_accuracy",
}
MODEL_COLORS = {"CEC": "#117733", "CEC_IDAAC": "#56B4E9"}
MODEL_ORDER = ("CEC", "CEC_IDAAC")
MODEL_LABELS = {"CEC": "CEC", "CEC_IDAAC": "DCEC"}
DEFAULT_NUM_ENVS = (32, 64, 128, 256)
BATCH_SIZE_LABELS = {32: "8K", 64: "16K", 128: "32K", 256: "65K"}
DEFAULT_MODEL_ROOT = Path("/mnt/nas/wonsang/crossenv_ued/models/ICRL")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        nargs="*",
        help=(
            "Explicit environment_probe_*.csv files. When omitted, files are "
            "discovered from --input-dir."
        ),
    )
    parser.add_argument("--model-root", type=Path, default=DEFAULT_MODEL_ROOT)
    parser.add_argument(
        "--input-dir", type=Path,
        help=(
            "Probe CSV directory. Defaults to "
            "<model-root>/analysis/representation_probe."
        ),
    )
    parser.add_argument(
        "--representation", choices=("actor", "value", "both"), default="both"
    )
    parser.add_argument(
        "--num-envs", nargs="+", type=int, default=list(DEFAULT_NUM_ENVS)
    )
    parser.add_argument(
        "--models", nargs="+", choices=("cec", "dcec"),
        default=["cec", "dcec"],
    )
    parser.add_argument(
        "--output",
        type=Path,
        help=(
            "Output path without an extension. Defaults to "
            "<first-input-dir>/environment_probe_comparison."
        ),
    )
    return parser.parse_args()


def discover_inputs(args: argparse.Namespace) -> list[Path]:
    if args.input:
        return [path.expanduser() for path in args.input]
    input_dir = (
        args.input_dir.expanduser()
        if args.input_dir is not None
        else args.model_root.expanduser() / "analysis" / "representation_probe"
    )
    representations = (
        ("actor", "value")
        if args.representation == "both" else (args.representation,)
    )
    paths = [
        input_dir
        / f"environment_probe_{'' if representation == 'actor' else 'value_'}"
        f"{model}_{num_envs}.csv"
        for representation in representations
        for model in args.models
        for num_envs in args.num_envs
    ]
    available = [path for path in paths if path.is_file()]
    missing = [path.name for path in paths if not path.is_file()]
    if missing:
        print("Missing probe CSVs: " + ", ".join(missing))
    if not available:
        raise FileNotFoundError(f"No matching probe CSVs found under {input_dir}")
    return available


def load_seed_means(paths: list[Path]) -> tuple[pd.DataFrame, float, tuple[str, ...]]:
    frames = []
    for path in paths:
        frame = pd.read_csv(path)
        missing = REQUIRED_COLUMNS - set(frame.columns)
        if missing:
            raise ValueError(f"{path}: missing columns {sorted(missing)}")
        if frame.empty:
            raise ValueError(f"{path}: CSV is empty")
        if "representation" not in frame:
            frame["representation"] = "actor"
        frames.append(frame)

    data = pd.concat(frames, ignore_index=True)
    numeric_columns = ["model_num_envs", "checkpoint_seed", "accuracy", "chance_accuracy"]
    for column in numeric_columns:
        data[column] = pd.to_numeric(data[column], errors="raise")
    if not np.isfinite(data[["accuracy", "chance_accuracy"]]).all().all():
        raise ValueError("Accuracy columns must contain only finite values")
    if not data["accuracy"].between(0.0, 1.0).all():
        raise ValueError("Probe accuracy must be between zero and one")

    chance_values = data["chance_accuracy"].unique()
    if len(chance_values) != 1:
        raise ValueError("Input CSVs use different chance accuracy values")
    representations = tuple(data["representation"].astype(str).unique())

    seed_means = (
        data.groupby(
            [
                "model", "model_label", "model_num_envs", "representation",
                "checkpoint_seed",
            ],
            as_index=False,
        )["accuracy"]
        .mean()
        .sort_values(["model", "model_num_envs", "checkpoint_seed"])
    )
    return seed_means, float(chance_values[0]), representations


def plot(
    seed_means: pd.DataFrame, chance: float, representations: tuple[str, ...],
    output_stem: Path,
) -> None:
    if seed_means.empty:
        raise ValueError("No model results to plot")
    num_envs_values = sorted(seed_means["model_num_envs"].astype(int).unique())
    present_models = set(seed_means["model"].astype(str))
    models = [model for model in MODEL_ORDER if model in present_models]
    models.extend(sorted(present_models - set(models)))
    representation_order = [
        representation for representation in ("value", "actor")
        if representation in set(representations)
    ]
    x = np.arange(len(representation_order), dtype=float)
    width = 0.72 / max(len(models), 1)
    representation_labels = {
        "actor": "Policy",
        "value": "Value",
    }
    if representation_order == ["actor"]:
        panel_title = "(b) Policy Representation Probe Accuracy"
    elif representation_order == ["value"]:
        panel_title = "(a) Value Representation Probe Accuracy"
    else:
        panel_title = "(a) Environment Probe Accuracy"

    output_stem.parent.mkdir(parents=True, exist_ok=True)
    for num_envs in num_envs_values:
        fig_width = max(7.2, 1.8 * len(representation_order) + 3.6)
        fig, ax = plt.subplots(figsize=(fig_width, 5.2))
        model_handles = {}
        for model_index, model in enumerate(models):
            color = MODEL_COLORS.get(model, "#777777")
            offset = (model_index - (len(models) - 1) / 2) * width
            label_added = False
            for representation_index, representation in enumerate(
                representation_order
            ):
                group = seed_means[
                    (seed_means["model"] == model)
                    & (seed_means["model_num_envs"] == num_envs)
                    & (seed_means["representation"] == representation)
                ]
                if group.empty:
                    continue
                values = group["accuracy"].to_numpy(dtype=float)
                mean = float(values.mean())
                sem = (
                    float(values.std(ddof=1) / np.sqrt(len(values)))
                    if len(values) > 1 else 0.0
                )
                position = x[representation_index] + offset
                bars = ax.bar(
                    position, mean, width=width * .9,
                    color=color, alpha=.85, edgecolor="none",
                    linewidth=0, zorder=2,
                    label=(
                        MODEL_LABELS.get(model, model)
                        if not label_added else None
                    ),
                )
                model_handles.setdefault(model, bars[0])
                label_added = True
                ax.errorbar(
                    position, mean, yerr=sem, color="black",
                    linewidth=2.0, capsize=5, zorder=3,
                )

        ax.set_xticks(x)
        ax.set_xticklabels([
            representation_labels[value] for value in representation_order
        ])
        chance_line = ax.axhline(
            chance, color="#555555", linestyle="--", linewidth=1.6,
            label=f"Random Guess ({chance:.0%})",
        )
        ax.set_ylabel("Probe Accuracy")
        ax.set_title(
            panel_title, pad=12,
            fontweight="normal",
        )
        ax.set_ylim(0.0, 1.0)
        ax.grid(False)
        ax.set_axisbelow(True)
        chance_legend = ax.legend(
            [chance_line], [f"Random Guess ({chance:.0%})"],
            frameon=False, loc="upper right", bbox_to_anchor=(.67, 1.0),
            ncol=1, fontsize=16,
        )
        ax.add_artist(chance_legend)
        ax.legend(
            [model_handles[model] for model in models if model in model_handles],
            [
                MODEL_LABELS.get(model, model)
                for model in models if model in model_handles
            ],
            frameon=False, loc="upper right", bbox_to_anchor=(.98, 1.0),
            ncol=1, fontsize=16,
        )
        fig.tight_layout()

        sized_stem = output_stem.with_name(
            f"{output_stem.name}_num_envs{num_envs}"
        )
        for suffix in ("pdf", "png"):
            path = sized_stem.with_suffix(f".{suffix}")
            fig.savefig(path, bbox_inches="tight", dpi=300)
            print(f"Saved: {path}")
        plt.close(fig)

    # When one representation is requested across several training-set
    # sizes, also save a compact scaling panel using the same paper style as
    # the Policy/Value comparison above.
    if len(representation_order) == 1 and len(num_envs_values) > 1:
        representation = representation_order[0]
        x = np.arange(len(num_envs_values), dtype=float)
        width = 0.72 / max(len(models), 1)
        fig, ax = plt.subplots(figsize=(7.2, 5.2))
        model_handles = {}
        for model_index, model in enumerate(models):
            color = MODEL_COLORS.get(model, "#777777")
            positions = x + (
                model_index - (len(models) - 1) / 2
            ) * width
            means, sems = [], []
            for num_envs in num_envs_values:
                values = seed_means[
                    (seed_means["model"] == model)
                    & (seed_means["model_num_envs"] == num_envs)
                    & (seed_means["representation"] == representation)
                ]["accuracy"].to_numpy(dtype=float)
                means.append(float(values.mean()) if len(values) else np.nan)
                sems.append(
                    float(values.std(ddof=1) / np.sqrt(len(values)))
                    if len(values) > 1 else 0.0
                )
            bars = ax.bar(
                positions, means, width=width * .9,
                color=color, alpha=.85, edgecolor="none",
                linewidth=0, zorder=2,
                label=MODEL_LABELS.get(model, model),
            )
            model_handles[model] = bars[0]
            ax.errorbar(
                positions, means, yerr=sems, color="black",
                linewidth=2.0, capsize=5, fmt="none", zorder=3,
            )

        ax.set_xticks(x)
        ax.set_xticklabels([
            BATCH_SIZE_LABELS.get(value, str(value))
            for value in num_envs_values
        ])
        ax.set_xlabel("Batch Size")
        chance_line = ax.axhline(
            chance, color="#555555", linestyle="--", linewidth=1.6,
        )
        ax.set_ylabel("Probe Accuracy")
        ax.set_title(
            panel_title, pad=12,
            fontweight="normal",
        )
        ax.set_ylim(0.0, 1.0)
        ax.grid(False)
        ax.set_axisbelow(True)
        chance_legend = ax.legend(
            [chance_line], [f"Random Guess ({chance:.0%})"],
            frameon=False, loc="upper right", bbox_to_anchor=(.67, 1.0),
            ncol=1, fontsize=16,
        )
        ax.add_artist(chance_legend)
        ax.legend(
            [model_handles[model] for model in models if model in model_handles],
            [
                MODEL_LABELS.get(model, model)
                for model in models if model in model_handles
            ],
            frameon=False, loc="upper right", bbox_to_anchor=(.98, 1.0),
            ncol=1, fontsize=16,
        )
        fig.tight_layout()
        aggregate_stem = output_stem.with_name(
            f"{output_stem.name}_by_num_envs"
        )
        for suffix in ("pdf", "png"):
            path = aggregate_stem.with_suffix(f".{suffix}")
            fig.savefig(path, bbox_inches="tight", dpi=300)
            print(f"Saved: {path}")
        plt.close(fig)


def main() -> None:
    args = parse_args()
    try:
        inputs = discover_inputs(args)
    except FileNotFoundError as error:
        raise SystemExit(str(error)) from error
    output_stem = (
        args.output.expanduser()
        if args.output is not None
        else inputs[0].parent
        / f"environment_probe_{'policy' if args.representation == 'actor' else args.representation}"
    )
    seed_means, chance, representations = load_seed_means(inputs)
    plot(seed_means, chance, representations, output_stem)


if __name__ == "__main__":
    main()
