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
MODEL_ORDER = ("CEC", "CEC_IDAAC")
MODEL_LABELS = {"CEC": "CEC", "CEC_IDAAC": "DCEC"}
DEFAULT_NUM_ENVS = (32, 64, 128, 256)
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
        "--representation", choices=("actor", "value"), default="actor"
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
    representation_prefix = "" if args.representation == "actor" else "value_"
    paths = [
        input_dir
        / f"environment_probe_{representation_prefix}{model}_{num_envs}.csv"
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


def plot(
    seed_means: pd.DataFrame, chance: float, representation: str,
    output_stem: Path,
) -> None:
    if seed_means.empty:
        raise ValueError("No model results to plot")
    num_envs_values = sorted(seed_means["model_num_envs"].astype(int).unique())
    present_models = set(seed_means["model"].astype(str))
    models = [model for model in MODEL_ORDER if model in present_models]
    models.extend(sorted(present_models - set(models)))
    x = np.arange(len(num_envs_values), dtype=float)
    width = 0.72 / max(len(models), 1)

    fig_width = max(8.5, 1.65 * len(num_envs_values) + 3.0)
    fig, ax = plt.subplots(figsize=(fig_width, 5.8))
    rng = np.random.default_rng(42)
    for model_index, model in enumerate(models):
        color = MODEL_COLORS.get(model, "#777777")
        offset = (model_index - (len(models) - 1) / 2) * width
        label_added = False
        for env_index, num_envs in enumerate(num_envs_values):
            group = seed_means[
                (seed_means["model"] == model)
                & (seed_means["model_num_envs"] == num_envs)
            ]
            if group.empty:
                continue
            values = group["accuracy"].to_numpy(dtype=float)
            mean = float(values.mean())
            sem = (
                float(values.std(ddof=1) / np.sqrt(len(values)))
                if len(values) > 1 else 0.0
            )
            position = x[env_index] + offset
            ax.bar(
                position, mean, width=width * .9, color=color, alpha=.42,
                edgecolor=color, linewidth=1.8, zorder=1,
                label=MODEL_LABELS.get(model, model) if not label_added else None,
            )
            label_added = True
            ax.errorbar(
                position, mean, yerr=sem, color=color, linewidth=2.0,
                capsize=5, zorder=3,
            )
            jitter = rng.uniform(-width * .16, width * .16, len(values))
            ax.scatter(
                position + jitter, values, s=48, color=color,
                edgecolor="white", linewidth=.7, zorder=4,
            )

    ax.set_xticks(x)
    ax.set_xticklabels([str(value) for value in num_envs_values])
    ax.set_xlabel("Number of Training Environments")
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
    ax.legend(frameon=False, loc="upper center", ncol=len(models) + 1)
    fig.tight_layout()

    output_stem.parent.mkdir(parents=True, exist_ok=True)
    for suffix in ("pdf", "png"):
        path = output_stem.with_suffix(f".{suffix}")
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
        / f"environment_probe_{args.representation}_by_num_envs"
    )
    seed_means, chance, representation = load_seed_means(inputs)
    plot(seed_means, chance, representation, output_stem)


if __name__ == "__main__":
    main()
