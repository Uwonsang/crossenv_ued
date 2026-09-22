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
MODEL_COLORS = {"CEC": "#117733", "CEC_IDAAC": "#0072B2"}


def crop_counter_circuit_padding(image: np.ndarray, record: dict) -> np.ndarray:
    """Remove the 9x9 padding while preserving the map's actual orientation."""
    if record.get("family") != "counter_circuit":
        return image
    layout = record["layout"]
    height, width = int(layout["height"]), int(layout["width"])
    walls = {int(index) for index in layout["wall_idx"]}
    content = set(range(height * width)) - walls
    for key in (
        "agent_idx", "goal_idx", "plate_pile_idx", "onion_pile_idx", "pot_idx",
    ):
        content.update(int(index) for index in layout[key])
    if not content:
        return image

    # The generated Counter Circuit footprint is 5x8, or 8x5 after rotation,
    # and make_9x9_layout places it at the upper-left of the 9x9 canvas.
    candidates = ((5, 8), (8, 5))
    crop_height, crop_width = min(
        candidates,
        key=lambda shape: sum(
            index // width >= shape[0] or index % width >= shape[1]
            for index in content
        ),
    )
    tile_height = image.shape[0] // height
    tile_width = image.shape[1] // width
    return image[:crop_height * tile_height, :crop_width * tile_width]


def render_state(config: dict, record: dict, horizon: int) -> np.ndarray:
    """Rebuild the controlled state and render it with JaxMARL's renderer."""
    from jaxmarl.viz.overcooked_jitted_visualizer import render_fn

    _, state, _ = instantiate(config, record, horizon)
    image = np.asarray(render_fn(state))
    return crop_counter_circuit_padding(image, record)


def draw_state(axis, record: dict, image: np.ndarray) -> None:
    axis.imshow(image)
    axis.axis("off")
    axis.set_title(
        f"Environment {record['variant']}",
    )


def metric_text(row: pd.Series) -> str:
    action_a = str(row["argmax_a"])
    action_b = str(row["argmax_b"])
    return (
        f"A: {action_a}   |   B: {action_b}"
        rf"   |   Rep. distance: $d_\pi$ = "
        f"{row['policy_rep_zscored_rms_distance']:.3f},  "
        rf"$d_V$ = {row['value_rep_zscored_rms_distance']:.3f}"
    )


def draw_summary(axis, row: pd.Series, model: str):
    axis.axis("off")
    return axis.text(
        .5, .5, f"{MODEL_LABELS[model]}   |   {metric_text(row)}",
        ha="center", va="center",
        bbox=dict(
            boxstyle="round,pad=.35",
            facecolor="#F6F6F6",
            edgecolor=MODEL_COLORS[model], linewidth=1.6,
        ),
    )


def fit_summary_width(figure, artists, base_size=(10.0, 5.694)) -> None:
    """Grow the paper figure only when a one-line summary would be clipped."""
    figure.canvas.draw()
    renderer = figure.canvas.get_renderer()
    widest = max(
        artist.get_bbox_patch().get_window_extent(renderer).width
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

    # Maps are columns (environment A/B), while model results are full-width
    # rows so a model is not visually associated with only one environment.
    paper_size = (10.0, 5.694)
    figure = plt.figure(figsize=paper_size)
    grid = figure.add_gridspec(
        3, 2, height_ratios=(3.0, .30, .30), hspace=.08, wspace=.10
    )
    draw_state(figure.add_subplot(grid[0, 0]), states[0], state_images[0])
    draw_state(figure.add_subplot(grid[0, 1]), states[1], state_images[1])
    summary_artists = []
    for row_index, model in enumerate(MODEL_ORDER, start=1):
        summary_artists.append(draw_summary(
            figure.add_subplot(grid[row_index, :]), rows[model], model
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
