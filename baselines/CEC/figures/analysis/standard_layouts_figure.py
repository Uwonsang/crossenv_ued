"""Render the five standard Overcooked layouts as a publication figure."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw
# These are the same five definitions as the unpadded standard layouts in
# jaxmarl/environments/overcooked/layouts.py.  Keeping the small static maps
# here makes figure generation independent of the training/JAX environment.
LAYOUTS = (
    ("Asymmetric\nAdvantages", (
        "WWWWWWWWW",
        "O WGWOW G",
        "W   W   W",
        "W R WB  W",
        "WWWPWPWWW",
    )),
    ("Coordination\nRing", (
        "WWWPW",
        "W R W",
        "DBW W",
        "O   W",
        "WOGWG",
    )),
    ("Counter\nCircuit", (
        "WWWPPWWG",
        "W R    W",
        "D WWWW G",
        "W     BW",
        "DWWOOWWW",
    )),
    ("Cramped\nRoom", (
        "WWPWW",
        "WR BO",
        "W   W",
        "WDWGG",
    )),
    ("Forced\nCoordination", (
        "WWWPW",
        "O WBP",
        "ORW W",
        "D W W",
        "WWWGG",
    )),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "artifacts" / "standard_layouts",
    )
    parser.add_argument("--filename-stem", default="five_standard_layouts")
    parser.add_argument("--dpi", type=int, default=300)
    return parser.parse_args()


def render_layout(rows, tile_size: int = 64) -> np.ndarray:
    """Render one standard layout with the project's Overcooked color scheme."""
    height, width = len(rows), len(rows[0])
    image = Image.new("RGB", (width * tile_size, height * tile_size), "black")
    draw = ImageDraw.Draw(image)
    colors = {
        "counter": (105, 105, 105),
        "grid": (70, 70, 70),
        "yellow": (255, 245, 0),
        "white": (255, 255, 255),
        "green": (20, 235, 35),
        "red": (255, 20, 20),
        "blue": (25, 25, 255),
    }

    def box(x, y, margin=0):
        return (
            x * tile_size + margin,
            y * tile_size + margin,
            (x + 1) * tile_size - 1 - margin,
            (y + 1) * tile_size - 1 - margin,
        )

    for y, row in enumerate(rows):
        if len(row) != width:
            raise ValueError("Every row in a layout must have the same width")
        for x, symbol in enumerate(row):
            if symbol in "WDOGP":
                draw.rectangle(box(x, y), fill=colors["counter"])
            else:
                draw.rectangle(box(x, y), fill=(0, 0, 0))
            draw.line(
                [(x * tile_size, y * tile_size), ((x + 1) * tile_size, y * tile_size)],
                fill=colors["grid"], width=1,
            )
            draw.line(
                [(x * tile_size, y * tile_size), (x * tile_size, (y + 1) * tile_size)],
                fill=colors["grid"], width=1,
            )

            if symbol == "G":
                draw.rectangle(box(x, y, tile_size // 10), fill=colors["green"])
            elif symbol == "O":
                radius = tile_size * 0.12
                for cx, cy in ((.50, .16), (.28, .42), (.75, .37), (.38, .76), (.72, .74)):
                    center_x, center_y = (x + cx) * tile_size, (y + cy) * tile_size
                    draw.ellipse(
                        (center_x-radius, center_y-radius, center_x+radius, center_y+radius),
                        fill=colors["yellow"],
                    )
            elif symbol == "D":
                radius = tile_size * 0.16
                for cx, cy in ((.30, .30), (.72, .43), (.40, .74)):
                    center_x, center_y = (x + cx) * tile_size, (y + cy) * tile_size
                    draw.ellipse(
                        (center_x-radius, center_y-radius, center_x+radius, center_y+radius),
                        fill=colors["white"],
                    )
            elif symbol == "P":
                draw.rectangle(
                    (x*tile_size+tile_size*.12, y*tile_size+tile_size*.34,
                     (x+1)*tile_size-tile_size*.12, (y+1)*tile_size-tile_size*.10),
                    fill=(20, 20, 20),
                )
                draw.rectangle(
                    (x*tile_size+tile_size*.10, y*tile_size+tile_size*.22,
                     (x+1)*tile_size-tile_size*.10, y*tile_size+tile_size*.27),
                    fill=(20, 20, 20),
                )
            elif symbol in "RB":
                color = colors["red"] if symbol == "R" else colors["blue"]
                left, top = x * tile_size, y * tile_size
                if symbol == "R":
                    points = (
                        (left+tile_size*.18, top+tile_size*.50),
                        (left+tile_size*.82, top+tile_size*.18),
                        (left+tile_size*.82, top+tile_size*.82),
                    )
                else:
                    points = (
                        (left+tile_size*.50, top+tile_size*.18),
                        (left+tile_size*.18, top+tile_size*.82),
                        (left+tile_size*.82, top+tile_size*.82),
                    )
                draw.polygon(points, fill=color)
    return np.asarray(image)


def plot_layouts(output_dir: Path, filename_stem: str, dpi: int):
    if not filename_stem or Path(filename_stem).name != filename_stem:
        raise ValueError("--filename-stem must be a filename without a path")
    if dpi <= 0:
        raise ValueError("--dpi must be positive")

    images = [(label, render_layout(rows)) for label, rows in LAYOUTS]
    figure = plt.figure(figsize=(12.0, 5.0), constrained_layout=False)

    def add_row(row_images, y, height, gap=0.004):
        figure_aspect = figure.get_figwidth() / figure.get_figheight()
        widths = [
            height * image.shape[1] / image.shape[0] / figure_aspect
            for _, image in row_images
        ]
        x = (1.0 - sum(widths) - gap * (len(widths) - 1)) / 2.0
        for (_, image), width in zip(row_images, widths):
            ax = figure.add_axes((x, y, width, height))
            ax.imshow(image, interpolation="nearest")
            ax.axis("off")
            x += width + gap

    # Put the three compact layouts above the two wide layouts.  The summed
    # aspect ratios of the rows are then similar, keeping the figure balanced.
    add_row([images[1], images[3], images[4]], y=0.505, height=0.485)
    add_row([images[0], images[2]], y=0.010, height=0.485)

    output_dir.mkdir(parents=True, exist_ok=True)
    for suffix in ("png", "pdf"):
        figure.savefig(
            output_dir / f"{filename_stem}.{suffix}",
            dpi=dpi,
            bbox_inches="tight",
            pad_inches=0.04,
        )
    plt.close(figure)


def main():
    args = parse_args()
    plot_layouts(
        args.output_dir.expanduser(),
        args.filename_stem,
        args.dpi,
    )
    print(f"Saved PNG/PDF to {args.output_dir.expanduser()}")


if __name__ == "__main__":
    main()
