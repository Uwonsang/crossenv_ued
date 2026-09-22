"""Redraw one concrete state pair as a direct CEC/DCEC comparison."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle
import numpy as np
import pandas as pd


ACTIONS = ("North", "South", "East", "West", "Stay", "Interact")
MODEL_ORDER = ("CEC", "CEC_IDAAC")
MODEL_LABELS = {"CEC": "CEC", "CEC_IDAAC": "DCEC"}
MODEL_COLORS = {"CEC": "#117733", "CEC_IDAAC": "#0072B2"}
ENV_COLORS = {"A": "#56B4E9", "B": "#0072B2"}


def positions(layout: dict, key: str) -> set[tuple[int, int]]:
    width = int(layout["width"])
    return {(int(index) % width, int(index) // width) for index in layout[key]}


def draw_state(axis, record: dict) -> None:
    layout = record["layout"]
    width, height = int(layout["width"]), int(layout["height"])
    walls = positions(layout, "wall_idx")
    for y in range(height):
        for x in range(width):
            axis.add_patch(Rectangle(
                (x, y), 1, 1,
                facecolor="#707070" if (x, y) in walls else "#f7f7f7",
                edgecolor="#4a4a4a", linewidth=.45,
            ))
    for key, color, label in (
        ("goal_idx", "#20df36", "Serve"),
        ("onion_pile_idx", "#ffe600", "Onion"),
        ("plate_pile_idx", "white", "Plate"),
        ("pot_idx", "#1b1b1b", "Pot\n2/3"),
    ):
        for x, y in positions(layout, key):
            axis.add_patch(Rectangle(
                (x + .08, y + .08), .84, .84, facecolor=color,
                edgecolor="black", linewidth=.6,
            ))
            axis.text(
                x + .5, y + .5, label, ha="center", va="center", fontsize=7,
                color="white" if key == "pot_idx" else "black",
            )
    for position, color, label in (
        (record["ego"], "#d62728", "Ego\nOnion"),
        (record["teammate"], "#2455d6", "Mate"),
    ):
        x, y = position
        axis.add_patch(Circle(
            (x + .5, y + .5), .35, facecolor=color,
            edgecolor="black", linewidth=.8, zorder=5,
        ))
        axis.text(
            x + .5, y + .5, label, ha="center", va="center", fontsize=7,
            color="white", fontweight="bold", zorder=6,
        )
    ego_x, ego_y = record["ego"]
    pot_x, pot_y = record["pot"]
    axis.annotate(
        "", xy=(pot_x + .5, pot_y + .5), xytext=(ego_x + .5, ego_y + .5),
        arrowprops=dict(arrowstyle="->", color="#d62728", lw=2),
    )
    axis.set(xlim=(0, width), ylim=(height, 0), aspect="equal")
    axis.axis("off")
    axis.set_title(
        f"Environment {record['variant']} · map seed {record['map_seed']}\n"
        f"Route cost = {record['route_cost']}",
        fontsize=13, fontweight="bold",
    )


def draw_policy(axis, row: pd.Series, model: str) -> None:
    x = np.arange(len(ACTIONS))
    width = .38
    for offset, variant in ((-.5, "a"), (.5, "b")):
        probabilities = [row[f"prob_{action.lower()}_{variant}"] for action in ACTIONS]
        axis.bar(
            x + offset * width, probabilities, width,
            color=ENV_COLORS[variant.upper()], label=f"Environment {variant.upper()}",
            edgecolor="black", linewidth=.4,
        )
    axis.set_xticks(x)
    axis.set_xticklabels(ACTIONS, rotation=30, ha="right")
    axis.set_ylim(0, 1.05)
    axis.set_ylabel("Action probability")
    axis.set_title(f"{MODEL_LABELS[model]} policy", fontweight="bold",
                   color=MODEL_COLORS[model])
    axis.grid(axis="y", alpha=.25)
    axis.legend(frameon=False, fontsize=8)


def draw_value(axis, row: pd.Series, model: str) -> None:
    x = np.arange(2)
    width = .38
    axis.bar(
        x - width / 2, [row["predicted_value_a"], row["predicted_value_b"]],
        width, label="Predicted value", color="#E69F00",
        edgecolor="black", linewidth=.4,
    )
    axis.bar(
        x + width / 2, [row["mc_return_a"], row["mc_return_b"]],
        width, label="MC return", color="#117733",
        edgecolor="black", linewidth=.4,
    )
    axis.set_xticks(x)
    axis.set_xticklabels(("Environment A", "Environment B"))
    axis.set_title(f"{MODEL_LABELS[model]} value", fontweight="bold",
                   color=MODEL_COLORS[model])
    axis.grid(axis="y", alpha=.25)
    axis.legend(frameon=False, fontsize=8)


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
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    try:
        states = load_pair(args.analysis_dir, args.pair_id)
        rows = load_metrics(args.analysis_dir, args.pair_id, args.num_envs, args.seed)
    except (FileNotFoundError, KeyError, ValueError) as error:
        parser.error(str(error))
    if len(states) != 2:
        parser.error(f"Expected two states, found {len(states)}")

    figure = plt.figure(figsize=(19, 10.5))
    grid = figure.add_gridspec(
        3, 4, height_ratios=(1.55, 1.0, .72), hspace=.42, wspace=.32
    )
    draw_state(figure.add_subplot(grid[0, 0:2]), states[0])
    draw_state(figure.add_subplot(grid[0, 2:4]), states[1])
    for column, model in enumerate(MODEL_ORDER):
        offset = column * 2
        draw_policy(figure.add_subplot(grid[1, offset]), rows[model], model)
        draw_value(figure.add_subplot(grid[1, offset + 1]), rows[model], model)
        draw_summary(figure.add_subplot(grid[2, offset:offset + 2]), rows[model], model)

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
