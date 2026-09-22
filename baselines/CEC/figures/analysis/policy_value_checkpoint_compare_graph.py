"""Redraw one concrete state pair as a direct CEC/DCEC comparison."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

# This script only rebuilds states for visualization.  Keeping JAX on CPU avoids
# a GPU/XLA shutdown crash that can otherwise occur after the PDF is saved.
os.environ.setdefault("JAX_PLATFORMS", "cpu")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 24,
    "axes.titlesize": 28,
    "axes.titleweight": "bold",
    "axes.labelsize": 24,
    "xtick.labelsize": 24,
    "ytick.labelsize": 24,
    "figure.titlesize": 28,
    "legend.fontsize": 24,
    "mathtext.fontset": "dejavusans",
})

from policy_value_concrete_example import ROOT, instantiate, load_config


MODEL_ORDER = ("CEC", "CEC_IDAAC")
MODEL_LABELS = {"CEC": "CEC", "CEC_IDAAC": "DCEC"}
MODEL_COLORS = {"CEC": "#117733", "CEC_IDAAC": "#56B4E9"}


def render_state(config: dict, record: dict, horizon: int) -> np.ndarray:
    """Rebuild the state and render JaxMARL's centered 7x7 map view."""
    from jaxmarl.viz.overcooked_jitted_visualizer import render_state as render_map

    _, state, _ = instantiate(config, record, horizon)
    return np.asarray(render_map(
        state, highlight=False, agent_view_size=6,
    ))


def draw_state(axis, record: dict, image: np.ndarray) -> None:
    axis.imshow(image)
    axis.axis("off")
    axis.set_title(
        f"Environment {record['variant']}",
    )


def draw_behavior(axis, variant: str, rows: dict[str, pd.Series]):
    """Show both models' decisions directly below one environment."""
    axis.axis("off")
    axis.set_xlim(0, 1)
    axis.set_ylim(0, 1)
    artists = []
    for y, model in zip((.68, .22), MODEL_ORDER):
        label = MODEL_LABELS[model]
        action = rows[model][f"argmax_{variant.lower()}"]
        artists.append(axis.text(
            .48, y, f"{label}:",
            ha="right", va="center", fontweight="bold",
            color=MODEL_COLORS[model],
        ))
        artists.append(axis.text(
            .50, y, str(action),
            ha="left", va="center", color="black",
        ))
    return artists


def fit_summary_width(figure, artists, base_size=(10.0, 5.694)) -> None:
    """Grow the paper figure only when text would be clipped."""
    figure.canvas.draw()
    renderer = figure.canvas.get_renderer()
    widest = max(
        artist.get_window_extent(renderer).width
        for artist in artists
    )
    available = figure.bbox.width * .94
    if widest <= available:
        return
    scale = 1.03 * widest / available
    figure.set_size_inches(base_size[0] * scale, base_size[1] * scale)


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

    # Each map is followed by two model-colored behavior lines.
    # A near-square column keeps the two 7x7 maps close together.  With a
    # 12-inch-wide figure, imshow preserves its square aspect and leaves large
    # horizontal gaps inside each subplot regardless of GridSpec.wspace.
    paper_size = (9.0, 5.3)
    figure = plt.figure(figsize=paper_size)
    grid = figure.add_gridspec(
        2, 2, height_ratios=(3.2, .65), hspace=.08, wspace=.02
    )
    draw_state(figure.add_subplot(grid[0, 0]), states[0], state_images[0])
    draw_state(figure.add_subplot(grid[0, 1]), states[1], state_images[1])
    summary_artists = []
    summary_artists.extend(draw_behavior(
        figure.add_subplot(grid[1, 0]), "A", rows
    ))
    summary_artists.extend(draw_behavior(
        figure.add_subplot(grid[1, 1]), "B", rows
    ))

    figure.subplots_adjust(top=.97, bottom=.04, left=.03, right=.97)
    fit_summary_width(figure, summary_artists, paper_size)
    output = args.output or (
        args.analysis_dir / "comparison_figures" /
        f"pair{args.pair_id:02d}_cec_vs_dcec_{args.num_envs}_seed{args.seed}_paper.pdf"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, bbox_inches="tight")
    plt.close(figure)
    print(f"Saved data-driven comparison figure to {output}")


if __name__ == "__main__":
    main()
