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
from matplotlib.offsetbox import AnchoredOffsetbox, HPacker, TextArea

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 16,
    "axes.titlesize": 20,
    "axes.titleweight": "bold",
    "axes.labelsize": 16,
    "xtick.labelsize": 16,
    "ytick.labelsize": 16,
    "figure.titlesize": 20,
    "legend.fontsize": 16,
    "mathtext.fontset": "dejavusans",
})

from policy_value_concrete_example import ROOT, instantiate, load_config


MODEL_ORDER = ("CEC", "CEC_IDAAC")
MODEL_LABELS = {"CEC": "CEC", "CEC_IDAAC": "DCEC"}
MODEL_COLORS = {"CEC": "#117733", "CEC_IDAAC": "#56B4E9"}


def crop_counter_circuit_view(image: np.ndarray, record: dict) -> np.ndarray:
    """Crop a rendered 7x7 view to 5x7 or 7x5 without losing visible tiles."""
    if record.get("family") != "counter_circuit":
        return image
    grid_size = 7
    tile_height = image.shape[0] // grid_size
    tile_width = image.shape[1] // grid_size
    wall_color = image[-1, -1]
    visible = np.zeros((grid_size, grid_size), dtype=bool)
    for row in range(grid_size):
        for column in range(grid_size):
            tile = image[
                row * tile_height:(row + 1) * tile_height,
                column * tile_width:(column + 1) * tile_width,
            ]
            visible[row, column] = np.any(tile != wall_color)
    visible_rows, visible_columns = np.nonzero(visible)
    if len(visible_rows) == 0:
        return image

    row_span = int(visible_rows.max() - visible_rows.min() + 1)
    column_span = int(visible_columns.max() - visible_columns.min() + 1)
    if row_span <= 5 and column_span >= row_span:
        crop_height, crop_width = 5, 7
    elif column_span <= 5:
        crop_height, crop_width = 7, 5
    else:
        return image

    def crop_start(minimum: int, maximum: int, size: int) -> int:
        start = min(minimum, grid_size - size)
        return max(0, min(start, maximum - size + 1))

    top = crop_start(int(visible_rows.min()), int(visible_rows.max()), crop_height)
    left = crop_start(
        int(visible_columns.min()), int(visible_columns.max()), crop_width
    )
    if np.any(visible[:top]) or np.any(visible[top + crop_height:]):
        return image
    if np.any(visible[:, :left]) or np.any(visible[:, left + crop_width:]):
        return image
    return image[
        top * tile_height:(top + crop_height) * tile_height,
        left * tile_width:(left + crop_width) * tile_width,
    ]


def trim_pair_to_common_content(
    images: list[np.ndarray], records: list[dict]
) -> list[np.ndarray]:
    """Apply one shared crop and equal horizontal padding to both maps."""
    if len(images) != 2 or not all(
        record.get("family") == "counter_circuit" for record in records
    ):
        return images
    if images[0].shape != images[1].shape:
        return images

    bounds = []
    wall_colors = []
    for image in images:
        wall_color = image[-1, -1]
        wall_colors.append(wall_color)
        content = np.any(image != wall_color, axis=-1)
        rows, columns = np.nonzero(content)
        if len(rows) == 0:
            return images
        bounds.append((
            int(rows.min()), int(rows.max()) + 1,
            int(columns.min()), int(columns.max()) + 1,
        ))
    top = min(bound[0] for bound in bounds)
    bottom = max(bound[1] for bound in bounds)
    left = min(bound[2] for bound in bounds)
    right = max(bound[3] for bound in bounds)
    horizontal_padding = 8
    padded = []
    for image, wall_color in zip(images, wall_colors):
        cropped = image[top:bottom, left:right]
        canvas = np.empty(
            (
                cropped.shape[0],
                cropped.shape[1] + 2 * horizontal_padding,
                cropped.shape[2],
            ),
            dtype=cropped.dtype,
        )
        canvas[...] = wall_color
        canvas[:, horizontal_padding:-horizontal_padding] = cropped
        padded.append(canvas)
    return padded


def render_state(config: dict, record: dict, horizon: int) -> np.ndarray:
    """Rebuild the state and render JaxMARL's centered 7x7 map view."""
    from jaxmarl.viz.overcooked_jitted_visualizer import render_state as render_map

    _, state, _ = instantiate(config, record, horizon)
    image = np.asarray(render_map(
        state, highlight=False, agent_view_size=6,
    ))
    return crop_counter_circuit_view(image, record)


def draw_state(axis, record: dict, image: np.ndarray) -> None:
    axis.imshow(image)
    axis.axis("off")
    axis.set_title(
        f"Environment {record['variant']}",
    )


def draw_behavior(axis, variant: str, rows: dict[str, pd.Series]):
    """Show both models' decisions on one centered horizontal line."""
    axis.axis("off")
    axis.set_xlim(0, 1)
    axis.set_ylim(0, 1)
    action_column = f"argmax_{variant.lower()}"
    parts = []
    for index, model in enumerate(MODEL_ORDER):
        if index:
            parts.append(TextArea(",  ", textprops={"fontsize": 16}))
        parts.append(TextArea(
            f"{MODEL_LABELS[model]}:",
            textprops={
                "fontsize": 16,
                "fontweight": "bold",
                "color": MODEL_COLORS[model],
                "fontfamily": "DejaVu Sans",
            },
        ))
        parts.append(TextArea(
            f" {rows[model][action_column]}",
            textprops={"fontsize": 16, "fontfamily": "DejaVu Sans"},
        ))
    packed = HPacker(children=parts, align="center", pad=0, sep=0)
    artist = AnchoredOffsetbox(
        loc="center", child=packed, frameon=False,
        bbox_to_anchor=(.5, .5), bbox_transform=axis.transAxes,
        borderpad=0,
    )
    axis.add_artist(artist)
    return [artist]


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
    state_images = trim_pair_to_common_content(state_images, states)

    # Each map is followed by two model-colored behavior lines.
    # Match the canvas width to the cropped map aspect ratios to avoid the
    # internal whitespace that imshow adds while preserving pixel geometry.
    aspect_sum = sum(image.shape[1] / image.shape[0] for image in state_images)
    paper_size = (max(7.5, 3.15 * aspect_sum + .2), 3.8)
    figure = plt.figure(figsize=paper_size)
    grid = figure.add_gridspec(
        2, 2, height_ratios=(3.2, .30), hspace=.01, wspace=.04
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

    figure.subplots_adjust(top=.95, bottom=.02, left=.005, right=.995)
    fit_summary_width(figure, summary_artists, paper_size)
    output = args.output or (
        args.analysis_dir / "comparison_figures" /
        f"pair{args.pair_id:02d}_cec_vs_dcec_{args.num_envs}_seed{args.seed}_paper.pdf"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, bbox_inches="tight", pad_inches=.01)
    plt.close(figure)
    print(f"Saved data-driven comparison figure to {output}")


if __name__ == "__main__":
    main()
