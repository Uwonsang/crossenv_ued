"""Combine Dual Destination and Overcooked-AI environment visualizations."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageChops


REPO_ROOT = Path(__file__).resolve().parents[4]
DEFAULT_DIR = REPO_ROOT / "artifacts" / "standard_layouts"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dual-image", type=Path, default=DEFAULT_DIR / "dual_setting.png"
    )
    parser.add_argument(
        "--overcooked-image",
        type=Path,
        default=DEFAULT_DIR / "five_standard_layouts.png",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_DIR)
    parser.add_argument(
        "--filename-stem", default="combined_environment_visualizations"
    )
    parser.add_argument("--dpi", type=int, default=300)
    return parser.parse_args()


def load_and_crop(path: Path) -> np.ndarray:
    """Crop transparent/white exterior margins and composite onto white."""
    image = Image.open(path).convert("RGBA")
    alpha_bbox = image.getchannel("A").getbbox()
    if alpha_bbox is None:
        raise ValueError(f"Image is fully transparent: {path}")
    image = image.crop(alpha_bbox)
    background = Image.new("RGBA", image.size, "white")
    image = Image.alpha_composite(background, image).convert("RGB")
    white = Image.new("RGB", image.size, "white")
    content_bbox = ImageChops.difference(image, white).getbbox()
    if content_bbox is None:
        raise ValueError(f"Image contains no visible content: {path}")
    return np.asarray(image.crop(content_bbox))


def main():
    args = parse_args()
    if args.dpi <= 0:
        raise ValueError("--dpi must be positive")
    if not args.filename_stem or Path(args.filename_stem).name != args.filename_stem:
        raise ValueError("--filename-stem must be a filename without a path")

    dual = load_and_crop(args.dual_image.expanduser())
    overcooked = load_and_crop(args.overcooked_image.expanduser())

    figure = plt.figure(figsize=(14.0, 5.5), facecolor="white")
    panels = (
        (dual, (0.015, 0.13, 0.455, 0.83), "(a) Dual Destination"),
        (overcooked, (0.485, 0.13, 0.500, 0.83), "(b) Overcooked-AI"),
    )
    for image, bounds, label in panels:
        ax = figure.add_axes(bounds)
        ax.imshow(image, interpolation="nearest")
        ax.set_anchor("S")
        ax.axis("off")
        center = bounds[0] + bounds[2] / 2
        figure.text(
            center, 0.075, label, ha="center", va="center",
            fontsize=18, fontfamily="serif",
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    for suffix in ("png", "pdf"):
        figure.savefig(
            args.output_dir / f"{args.filename_stem}.{suffix}",
            dpi=args.dpi,
            bbox_inches="tight",
            pad_inches=0.04,
            facecolor="white",
        )
    plt.close(figure)
    print(f"Saved PNG/PDF to {args.output_dir}")


if __name__ == "__main__":
    main()
