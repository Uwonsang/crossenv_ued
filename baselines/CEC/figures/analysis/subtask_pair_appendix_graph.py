"""Render controlled subtask-pair examples for the paper appendix."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

# Rendering does not need a GPU and CPU-only JAX avoids shutdown crashes.
os.environ.setdefault("JAX_PLATFORMS", "cpu")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import yaml
from PIL import Image


ROOT = Path(__file__).resolve().parents[4]
SUBTASKS = (
    "plate_pickup",
    "cooked_soup_pickup",
    "serve",
    "onion_pickup",
    "onion_to_pot",
)
SUBTASK_LABELS = {
    "plate_pickup": "Plate pickup",
    "cooked_soup_pickup": "Cooked soup\npickup",
    "serve": "Serve",
    "onion_pickup": "Onion pickup",
    "onion_to_pot": "Third onion\nto pot",
}


plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 16,
    "axes.titlesize": 20,
    "axes.titleweight": "normal",
    "axes.labelsize": 16,
    "xtick.labelsize": 16,
    "ytick.labelsize": 16,
    "figure.titlesize": 20,
    "legend.fontsize": 16,
})


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-root", type=Path, required=True)
    parser.add_argument(
        "--input-dir",
        type=Path,
        help=(
            "Directory containing subtask_state_pairs.json. Defaults to the "
            "Counter Circuit subtask analysis under --model-root."
        ),
    )
    parser.add_argument("--case-id", type=int, default=0)
    parser.add_argument(
        "--seed", type=int, default=0,
        help=(
            "Used only in the output filename for compatibility. The saved "
            "state pairs are constructed before model evaluation."
        ),
    )
    parser.add_argument("--horizon", type=int, default=200)
    parser.add_argument(
        "--subtasks", nargs="+", choices=SUBTASKS, default=list(SUBTASKS)
    )
    parser.add_argument(
        "--config", type=Path,
        default=ROOT / "baselines/CEC_UED/config/ippo_overcooked_CEC_gradient.yaml",
    )
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def load_examples(
    input_dir: Path,
    subtasks: list[str],
    case_id: int,
) -> list[tuple[str, list[dict]]]:
    pairs_path = input_dir / "subtask_state_pairs.json"
    if not pairs_path.is_file():
        raise FileNotFoundError(f"Pair JSON does not exist: {pairs_path}")

    pair_records = json.loads(pairs_path.read_text(encoding="utf-8"))["pairs"]
    selected_pairs = {
        str(pair["subtask"]): pair
        for pair in pair_records
        if int(pair["case_id"]) == case_id
    }
    examples = []
    for subtask in subtasks:
        if subtask not in selected_pairs:
            raise KeyError(f"No {subtask} pair has case_id={case_id}")
        pair = selected_pairs[subtask]
        examples.append((subtask, pair["states"]))
    return examples


def crop_9x9_view(image: np.ndarray, record: dict | None = None) -> np.ndarray:
    """Crop to 8x5 or 5x8 while preserving the top/left map origin."""
    grid_size = 9
    tile_height = image.shape[0] // grid_size
    tile_width = image.shape[1] // grid_size
    if (
        tile_height * grid_size != image.shape[0]
        or tile_width * grid_size != image.shape[1]
    ):
        raise ValueError(f"Expected a 9x9 tile render, found {image.shape}")

    visible = np.zeros((grid_size, grid_size), dtype=bool)
    layout = record.get("layout") if record else None
    if layout:
        # The saved layout and rendered image are both 9x9. Derive the useful
        # extent from semantic cells, preserving coordinates from the top-left
        # origin. Only padding below/right of that extent may be removed.
        layout_width = int(layout["width"])
        layout_height = int(layout["height"])
        wall_indices = {int(index) for index in layout["wall_idx"]}
        occupied = set(range(layout_width * layout_height)) - wall_indices
        for key in (
            "goal_idx", "onion_pile_idx", "plate_pile_idx", "pot_idx",
            "agent_idx",
        ):
            occupied.update(int(index) for index in layout.get(key, []))
        for index in occupied:
            row, column = divmod(index, layout_width)
            if row < grid_size and column < grid_size:
                visible[row, column] = True
        for key in ("ego", "teammate", "target"):
            position = record.get(key)
            if position is not None:
                column, row = map(int, position)
                if row < grid_size and column < grid_size:
                    visible[row, column] = True
    else:
        # Fallback for records without layout metadata. Ignore anti-aliased
        # tile borders and detect meaningful color area in each tile interior.
        corner_size = max(2, min(tile_height, tile_width) // 5)
        corner_pixels = np.concatenate([
            image[:corner_size, :corner_size].reshape(-1, image.shape[-1]),
            image[:corner_size, -corner_size:].reshape(-1, image.shape[-1]),
            image[-corner_size:, :corner_size].reshape(-1, image.shape[-1]),
            image[-corner_size:, -corner_size:].reshape(-1, image.shape[-1]),
        ])
        background = np.median(corner_pixels, axis=0)
        inset_y = max(2, tile_height // 8)
        inset_x = max(2, tile_width // 8)
        for row in range(grid_size):
            for column in range(grid_size):
                tile = image[
                    row * tile_height:(row + 1) * tile_height,
                    column * tile_width:(column + 1) * tile_width,
                ]
                interior = tile[inset_y:-inset_y, inset_x:-inset_x]
                color_delta = np.max(
                    np.abs(interior.astype(np.int16) - background), axis=-1
                )
                visible[row, column] = np.mean(color_delta > 8) >= .01
    visible_rows, visible_columns = np.nonzero(visible)
    if len(visible_rows) == 0:
        return image

    row_span = int(visible_rows.max() - visible_rows.min() + 1)
    column_span = int(visible_columns.max() - visible_columns.min() + 1)
    if row_span <= 5 and column_span >= row_span:
        # Several horizontal layouts contain a dispenser/pot in column 7.
        # Keep eight columns consistently so those edge objects survive.
        crop_height, crop_width = 5, 8
    elif column_span <= 5:
        # The corresponding vertical layouts can contain objects in row 7.
        crop_height, crop_width = 8, 5
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
    cropped = image[
        top * tile_height:(top + crop_height) * tile_height,
        left * tile_width:(left + crop_width) * tile_width,
    ]
    print(
        f"Applied map crop: 9x9 -> {crop_width}x{crop_height} "
        f"(columns {left}:{left + crop_width}, rows {top}:{top + crop_height})"
    )
    return cropped


def render_state(config: dict, record: dict, horizon: int) -> np.ndarray:
    from jaxmarl.viz.overcooked_jitted_visualizer import render_state as render_map
    from subtask_pair_consistency import instantiate_controlled_state

    _, state = instantiate_controlled_state(config, record, horizon)
    # These controlled states use a 17x17 padded map. agent_view_size=6 makes
    # padding=4 and exposes the complete saved 9x9 layout. Cropping before
    # this point would remove the top/left dispenser tiles.
    image = np.asarray(render_map(
        state, highlight=False, agent_view_size=6,
    ))
    return crop_9x9_view(image, record)


def draw_appendix_figure(
    examples: list[tuple[str, list[dict]]],
    config: dict,
    horizon: int,
    output: Path,
) -> None:
    row_count = len(examples)
    figure = plt.figure(figsize=(10.0, 2.6 * row_count + .8))
    grid = figure.add_gridspec(
        row_count, 2,
        hspace=.08,
        # Keep a narrow white separator so the two environments remain
        # visually distinct while still reading as a matched pair.
        wspace=.06,
    )
    column_axes: list[list[plt.Axes]] = [[], []]

    for row, (subtask, states) in enumerate(examples):
        if len(states) != 2:
            raise ValueError(
                f"Expected two states for {subtask}; found {len(states)}"
            )
        map_axes = []
        for column, state in enumerate(states):
            image = render_state(config, state, horizon)
            axis = figure.add_subplot(grid[row, column])
            # Preserve one rendered tile as a square.  Some Matplotlib style
            # configurations set image.aspect="auto", which stretches a
            # cropped 8x5/5x8 raster back into a square subplot and makes it
            # appear as though the gray wall tiles were never removed.
            axis.imshow(image, interpolation="nearest", aspect="equal")
            axis.set_aspect("equal", adjustable="box")
            if column == 0:
                axis.set_anchor("E")
            elif image.shape[0] > image.shape[1]:
                # Center narrow 5x8 maps within the Environment B column.
                axis.set_anchor("C")
            else:
                # Keep horizontal B maps close to their matched A maps.
                axis.set_anchor("W")
            axis.axis("off")
            map_axes.append(axis)
            column_axes[column].append(axis)
        map_axes[0].text(
            -.06, .5, SUBTASK_LABELS[subtask],
            transform=map_axes[0].transAxes,
            ha="right", va="center", fontsize=16,
            fontfamily="DejaVu Sans", linespacing=1.15,
        )

    figure.subplots_adjust(
        top=.955, bottom=.015, left=.19, right=.995,
    )
    # Resolve equal-aspect axes first, then center each header over the union
    # of all five rendered maps in that column. This keeps Environment B over
    # the full column even though its first-row map is narrow and vertical.
    figure.canvas.draw()
    for axes, label in zip(
        column_axes, ("Environment A", "Environment B"), strict=True
    ):
        positions = [axis.get_position() for axis in axes]
        column_left = min(position.x0 for position in positions)
        column_right = max(position.x1 for position in positions)
        figure.text(
            (column_left + column_right) / 2, .972, label,
            ha="center", va="bottom",
            fontsize=20, fontfamily="DejaVu Sans", fontweight="normal",
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    png_path = output.with_suffix(".png")
    pdf_path = output.with_suffix(".pdf")
    figure.savefig(
        png_path, bbox_inches="tight", pad_inches=.02,
        dpi=300, facecolor="white",
    )
    plt.close(figure)
    # Some PDF viewers fail to display raster images embedded directly by
    # Matplotlib. Building the PDF from the verified PNG makes both outputs
    # pixel-identical and avoids viewer-specific missing-map artifacts.
    with Image.open(png_path) as rendered:
        rendered.convert("RGB").save(
            pdf_path, "PDF", resolution=300.0, quality=95,
        )
    print(f"Saved PNG: {png_path.resolve()}")
    print(f"Saved PDF: {pdf_path.resolve()}")


def main() -> None:
    args = parse_args()
    if args.case_id < 0:
        raise SystemExit("--case-id must be nonnegative")
    input_dir = args.input_dir or (
        args.model_root.expanduser()
        / "analysis/subtask_pair_consistency/counter_circuit"
    )
    output = args.output or (
        input_dir / "figures"
        / f"subtask_pair_examples_seed{args.seed}_paper"
    )
    try:
        examples = load_examples(
            input_dir,
            list(dict.fromkeys(args.subtasks)),
            args.case_id,
        )
        with args.config.open(encoding="utf-8") as file:
            config = yaml.safe_load(file)
        config["ENV_NAME"] = "overcooked"
        config["CONV_NET"] = True
        config["LSTM"] = True
        print(f"Output PNG: {output.with_suffix('.png').resolve()}")
        print(f"Output PDF: {output.with_suffix('.pdf').resolve()}")
        draw_appendix_figure(examples, config, args.horizon, output)
    except (FileNotFoundError, KeyError, ValueError) as error:
        raise SystemExit(str(error)) from error


if __name__ == "__main__":
    main()
