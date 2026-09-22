"""Redraw one concrete state pair as a direct CEC/DCEC comparison."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from policy_value_concrete_example import ROOT, instantiate, load_config


MODEL_ORDER = ("CEC", "CEC_IDAAC")
MODEL_LABELS = {"CEC": "CEC", "CEC_IDAAC": "DCEC"}
MODEL_COLORS = {"CEC": "#117733", "CEC_IDAAC": "#0072B2"}


def render_state(config: dict, record: dict, horizon: int) -> np.ndarray:
    """Rebuild the controlled state and render it with JaxMARL's renderer."""
    from jaxmarl.viz.overcooked_jitted_visualizer import render_fn

    _, state, _ = instantiate(config, record, horizon)
    return np.asarray(render_fn(state))


def draw_state(axis, record: dict, image: np.ndarray) -> None:
    axis.imshow(image)
    axis.axis("off")
    axis.set_title(
        f"Environment {record['variant']}",
        fontsize=13, fontweight="bold",
    )


def metric_text(row: pd.Series) -> str:
    action_a = str(row["argmax_a"])
    action_b = str(row["argmax_b"])
    probability_a = row[f"prob_{action_a.lower()}_a"]
    probability_b = row[f"prob_{action_b.lower()}_b"]
    return (
        f"Behavior:  A: {action_a} ({probability_a:.3f})   |   "
        f"B: {action_b} ({probability_b:.3f})\n"
        rf"$\hat{{V}}_1(s_0^A)={row['predicted_value_a']:.2f}$"
        rf"   |   $\hat{{V}}_1(s_0^B)={row['predicted_value_b']:.2f}$"
        rf"   |   $\Delta\hat{{V}}_1(A-B)="
        rf"{row['predicted_value_delta_a_minus_b']:.2f}$\n"
        f"Z-scored RMS:  policy: "
        f"{row['policy_rep_zscored_rms_distance']:.3f}   |   "
        f"value: {row['value_rep_zscored_rms_distance']:.3f}"
    )


def draw_summary(axis, row: pd.Series, model: str) -> None:
    axis.axis("off")
    axis.set_title(
        MODEL_LABELS[model], fontweight="bold", fontsize=16,
        color=MODEL_COLORS[model], pad=8,
    )
    axis.text(
        .06, .5, metric_text(row), ha="left", va="center", fontsize=10.5,
        linespacing=1.55,
        bbox=dict(
            boxstyle="round,pad=.7",
            facecolor="#F6F6F6",
            edgecolor=MODEL_COLORS[model], linewidth=1.6,
        ),
    )


def load_pair(analysis_dir: Path, pair_id: int) -> list[dict]:
    path = analysis_dir / "concrete_state_pairs.json"
    if not path.is_file():
        raise FileNotFoundError(f"State-pair JSON does not exist: {path}")
    pairs = json.loads(path.read_text(encoding="utf-8"))["pairs"]
    for pair in pairs:
        if int(pair["pair_id"]) == pair_id:
            return pair["states"]
    raise KeyError(f"Pair {pair_id} is absent from {path}")


def load_metrics(
    analysis_dir: Path, pair_id: int, num_envs: int, seed: int
) -> dict[str, pd.Series]:
    path = analysis_dir / "concrete_example_metrics.csv"
    if not path.is_file():
        raise FileNotFoundError(f"Metrics CSV does not exist: {path}")
    frame = pd.read_csv(path)
    selected = frame[
        (frame["pair_id"] == pair_id)
        & (frame["num_envs"] == num_envs)
        & (frame["seed"] == seed)
    ]
    rows = {}
    for model in MODEL_ORDER:
        model_rows = selected[selected["model"] == model]
        if len(model_rows) != 1:
            raise ValueError(
                f"Expected one {model} row for pair={pair_id}, "
                f"num_envs={num_envs}, seed={seed}; found {len(model_rows)}"
            )
        rows[model] = model_rows.iloc[0]
    return rows


def ensure_zscored_rms(rows: dict[str, pd.Series], config: dict) -> None:
    """Support older CSVs that contain z-scored L2 but not RMS distance."""
    default_dims = {
        "policy": int(config["GRU_HIDDEN_DIM"]) // 4,
        "value": int(config["FC_DIM_SIZE"]) // 2,
    }
    for row in rows.values():
        for prefix in ("policy", "value"):
            rms_column = f"{prefix}_rep_zscored_rms_distance"
            if rms_column in row.index and pd.notna(row[rms_column]):
                continue
            dim_column = f"{prefix}_rep_dim"
            dimension = (
                int(row[dim_column])
                if dim_column in row.index and pd.notna(row[dim_column])
                else default_dims[prefix]
            )
            row[rms_column] = (
                row[f"{prefix}_rep_zscored_euclidean_distance"]
                / np.sqrt(dimension)
            )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analysis-dir", type=Path, required=True)
    parser.add_argument("--pair-id", type=int, required=True)
    parser.add_argument("--num-envs", type=int, default=64)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--horizon", type=int, default=200)
    parser.add_argument(
        "--config", type=Path,
        default=ROOT / "baselines/CEC_UED/config/ippo_overcooked_CEC_gradient.yaml",
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    try:
        states = load_pair(args.analysis_dir, args.pair_id)
        rows = load_metrics(args.analysis_dir, args.pair_id, args.num_envs, args.seed)
    except (FileNotFoundError, KeyError, ValueError) as error:
        parser.error(str(error))
    if len(states) != 2:
        parser.error(f"Expected two states, found {len(states)}")
    config = load_config(args.config)
    ensure_zscored_rms(rows, config)
    state_images = [render_state(config, state, args.horizon) for state in states]

    figure = plt.figure(figsize=(16, 9.0))
    grid = figure.add_gridspec(
        2, 2, height_ratios=(3.0, .62), hspace=.12, wspace=.12
    )
    draw_state(figure.add_subplot(grid[0, 0]), states[0], state_images[0])
    draw_state(figure.add_subplot(grid[0, 1]), states[1], state_images[1])
    for column, model in enumerate(MODEL_ORDER):
        draw_summary(figure.add_subplot(grid[1, column]), rows[model], model)

    family = states[0].get("family", "unknown").replace("_", " ").title()
    figure.suptitle(
        f"Policy–value asymmetry · Pair {args.pair_id:02d} · {family} · seed {args.seed}",
        fontsize=20, fontweight="bold",
    )
    output = args.output or (
        args.analysis_dir / "comparison_figures" /
        f"pair{args.pair_id:02d}_cec_vs_dcec_{args.num_envs}_seed{args.seed}_compact.png"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=220, bbox_inches="tight")
    plt.close(figure)
    print(f"Saved data-driven comparison figure to {output}")


if __name__ == "__main__":
    main()
