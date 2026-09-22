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


ACTIONS = ("North", "South", "East", "West", "Stay", "Interact")
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
        f"Environment {record['variant']} · map seed {record['map_seed']}\n"
        f"Route cost = {record['route_cost']}",
        fontsize=13, fontweight="bold",
    )


def draw_numeric_table(axis, row: pd.Series, model: str) -> None:
    axis.axis("off")
    column_labels = ["Env", "N", "S", "E", "W", "Stay", "Interact", "V", "MC"]
    cell_text = []
    for variant in ("a", "b"):
        values = [row[f"prob_{action.lower()}_{variant}"] for action in ACTIONS]
        values.extend((row[f"predicted_value_{variant}"], row[f"mc_return_{variant}"]))
        cell_text.append([
            variant.upper(),
            *(f"{value:.3f}" for value in values[:6]),
            f"{values[6]:.2f}",
            f"{values[7]:.2f}",
        ])
    table = axis.table(
        cellText=cell_text,
        colLabels=column_labels,
        cellLoc="center", loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9.5)
    table.scale(1, 1.75)
    for column in range(len(column_labels)):
        table[(0, column)].set_facecolor("#E8E8E8")
        table[(0, column)].set_text_props(fontweight="bold")
    for row_index, variant in enumerate(("a", "b"), start=1):
        probabilities = np.asarray([
            row[f"prob_{action.lower()}_{variant}"] for action in ACTIONS
        ])
        best_column = 1 + int(probabilities.argmax())
        table[(row_index, best_column)].set_facecolor("#FFF2A8")
        table[(row_index, best_column)].set_text_props(fontweight="bold")
        table[(row_index, 0)].set_text_props(fontweight="bold")
        table[(row_index, 7)].set_facecolor("#FDE8B0")
        table[(row_index, 8)].set_facecolor("#DDF1E2")
    axis.set_title(
        f"{MODEL_LABELS[model]}: action probabilities and values",
        fontweight="bold", color=MODEL_COLORS[model], pad=12,
    )


def metric_text(row: pd.Series) -> str:
    passed = bool(row["passes_concrete_example"])
    return (
        f"Policy JS: {row['policy_js_nats']:.4f}   ·   "
        f"argmax A/B: {row['argmax_a']} / {row['argmax_b']}   ·   "
        f"Interact A/B: {row['interact_probability_a']:.3f} / "
        f"{row['interact_probability_b']:.3f}\n"
        f"Predicted ΔV(A−B): {row['predicted_value_delta_a_minus_b']:.3f}   ·   "
        f"MC ΔG(A−B): {row['mc_return_delta_a_minus_b']:.3f} "
        f"± {row['mc_return_delta_sem']:.3f}\n"
        f"Representation cosine policy/value: "
        f"{row['policy_rep_cosine_distance']:.4f} / "
        f"{row['value_rep_cosine_distance']:.4f}   ·   "
        f"z-scored L2 policy/value: "
        f"{row['policy_rep_zscored_euclidean_distance']:.3f} / "
        f"{row['value_rep_zscored_euclidean_distance']:.3f}\n"
        f"Automatic filter: {'PASS' if passed else 'DOES NOT PASS'}"
    )


def draw_summary(axis, row: pd.Series, model: str) -> None:
    passed = bool(row["passes_concrete_example"])
    axis.axis("off")
    axis.text(
        .5, .5, metric_text(row), ha="center", va="center", fontsize=10.5,
        linespacing=1.5,
        bbox=dict(
            boxstyle="round,pad=.7",
            facecolor="#EFF8F1" if passed else "#FFF2F2",
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
    state_images = [render_state(config, state, args.horizon) for state in states]

    figure = plt.figure(figsize=(17, 9.2))
    grid = figure.add_gridspec(
        3, 2, height_ratios=(1.65, .65, .72), hspace=.34, wspace=.18
    )
    draw_state(figure.add_subplot(grid[0, 0]), states[0], state_images[0])
    draw_state(figure.add_subplot(grid[0, 1]), states[1], state_images[1])
    for column, model in enumerate(MODEL_ORDER):
        draw_numeric_table(figure.add_subplot(grid[1, column]), rows[model], model)
        draw_summary(figure.add_subplot(grid[2, column]), rows[model], model)

    family = states[0].get("family", "unknown").replace("_", " ").title()
    figure.suptitle(
        f"Policy–value asymmetry · Pair {args.pair_id:02d} · {family} · seed {args.seed}",
        fontsize=20, fontweight="bold",
    )
    output = args.output or (
        args.analysis_dir / "comparison_figures" /
        f"pair{args.pair_id:02d}_cec_vs_dcec_{args.num_envs}_seed{args.seed}.png"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=220, bbox_inches="tight")
    plt.close(figure)
    print(f"Saved data-driven comparison figure to {output}")


if __name__ == "__main__":
    main()
