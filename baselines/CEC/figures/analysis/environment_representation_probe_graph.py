"""Visualize model-level CSVs produced by environment_representation_probe.py."""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


REQUIRED_COLUMNS = {
    "model",
    "model_label",
    "model_num_envs",
    "checkpoint_seed",
    "probe_split",
    "accuracy",
    "chance_accuracy",
}
MODEL_COLORS = {"CEC": "#117733", "CEC_IDAAC": "#0072B2"}
BATCH_LABELS = {32: "8K", 64: "16K", 128: "32K", 256: "65K"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        nargs="+",
        required=True,
        help="One or more environment_probe_*.csv files.",
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


def load_seed_means(paths: list[Path]) -> tuple[pd.DataFrame, float, str]:
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
    representations = data["representation"].astype(str).unique()
    if len(representations) != 1:
        raise ValueError("Actor and value representation CSVs cannot share one plot")

    seed_means = (
        data.groupby(
            ["model", "model_label", "model_num_envs", "checkpoint_seed"],
            as_index=False,
        )["accuracy"]
        .mean()
        .sort_values(["model", "model_num_envs", "checkpoint_seed"])
    )
    return seed_means, float(chance_values[0]), str(representations[0])


def display_label(model_label: str, num_envs: int) -> str:
    batch = BATCH_LABELS.get(num_envs, f"{num_envs} envs")
    return f"{model_label}\n({batch})"


def plot(
    seed_means: pd.DataFrame, chance: float, representation: str,
    output_stem: Path,
) -> None:
    groups = list(
        seed_means.groupby(
            ["model", "model_label", "model_num_envs"], sort=False
        )
    )
    if not groups:
        raise ValueError("No model results to plot")

    fig, ax = plt.subplots(figsize=(7.2, 5.6))
    rng = np.random.default_rng(42)
    for x, ((model, model_label, num_envs), group) in enumerate(groups):
        values = group["accuracy"].to_numpy(dtype=float)
        mean = float(values.mean())
        sem = (
            float(values.std(ddof=1) / np.sqrt(len(values)))
            if len(values) > 1
            else 0.0
        )
        color = MODEL_COLORS.get(model, "#777777")
        ax.bar(
            x,
            mean,
            width=0.58,
            color=color,
            alpha=0.35,
            edgecolor=color,
            linewidth=1.8,
            zorder=1,
        )
        ax.errorbar(
            x, mean, yerr=sem, color=color, linewidth=2.0, capsize=5, zorder=3
        )
        jitter = rng.uniform(-0.075, 0.075, len(values))
        ax.scatter(
            x + jitter,
            values,
            s=58,
            color=color,
            edgecolor="white",
            linewidth=0.7,
            zorder=4,
        )

    labels = [
        display_label(str(model_label), int(num_envs))
        for (model, model_label, num_envs), _ in groups
    ]
    ax.set_xticks(np.arange(len(groups)))
    ax.set_xticklabels(labels)
    ax.axhline(
        chance,
        color="#555555",
        linestyle="--",
        linewidth=1.6,
        label=f"Chance ({chance:.0%})",
    )
    ax.set_ylabel("Held-out Episode Probe Accuracy")
    ax.set_ylim(0.0, 1.0)
    representation_label = "Policy" if representation == "actor" else "Value"
    ax.set_title(
        f"Layout Information in {representation_label} Representations",
        fontweight="bold",
    )
    ax.grid(axis="y", alpha=0.25)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, loc="upper right")
    fig.tight_layout()

    output_stem.parent.mkdir(parents=True, exist_ok=True)
    for suffix in ("pdf", "png"):
        path = output_stem.with_suffix(f".{suffix}")
        fig.savefig(path, bbox_inches="tight", dpi=300)
        print(f"Saved: {path}")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    inputs = [path.expanduser() for path in args.input]
    output_stem = (
        args.output.expanduser()
        if args.output is not None
        else inputs[0].parent / "environment_probe_comparison"
    )
    seed_means, chance, representation = load_seed_means(inputs)
    plot(seed_means, chance, representation, output_stem)


if __name__ == "__main__":
    main()
