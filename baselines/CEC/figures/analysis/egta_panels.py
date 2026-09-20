"""Plot baseline-pair subgames with a focal model at the top vertex."""
import argparse
import json
from itertools import combinations
from pathlib import Path

from egta_analysis import plt, np, pd, replicator, nash_gap, display_name


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--payoff", type=Path, required=True)
    parser.add_argument("--focal", required=True, help="Exact model identifier, not an alias")
    parser.add_argument("--pair", nargs=2, action="append",
                        help="Repeat for each panel; default: all pairs of remaining models")
    parser.add_argument("--resolution", type=int, default=8)
    parser.add_argument(
        "--arrow-scale", type=float, default=0.15,
        help="Maximum arrow length in simplex-coordinate units.",
    )
    parser.add_argument("--title", default="Role-averaged empirical meta-game")
    parser.add_argument("--model-fontsize", type=float, default=16,
                        help="Font size for model names at simplex vertices.")
    parser.add_argument("--title-fontsize", type=float, default=18,
                        help="Font size for the figure title.")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--filename-stem", default="baseline_panels")
    args = parser.parse_args()
    if args.resolution < 2:
        parser.error("resolution must be >= 2")
    if not np.isfinite(args.arrow_scale) or args.arrow_scale <= 0:
        parser.error("arrow-scale must be finite and positive")
    if (not np.isfinite(args.model_fontsize) or args.model_fontsize <= 0
            or not np.isfinite(args.title_fontsize) or args.title_fontsize <= 0):
        parser.error("font sizes must be finite and positive")
    if (not args.filename_stem
            or Path(args.filename_stem).name != args.filename_stem):
        parser.error("filename-stem must be a nonempty filename without a path")
    frame = pd.read_csv(args.payoff, index_col=0)
    if (frame.index.has_duplicates or frame.columns.has_duplicates
            or list(frame.index) != list(frame.columns)
            or not np.isfinite(frame.to_numpy(dtype=float)).all()):
        parser.error("Payoff must be a finite square matrix with matching unique labels")
    if args.focal not in frame.index:
        parser.error(f"Focal model {args.focal!r} absent; available: {list(frame.index)}")
    pairs = args.pair or list(combinations([m for m in frame.index if m != args.focal], 2))
    if not pairs:
        parser.error("At least two baseline models are required")
    for pair in pairs:
        if len(set([*pair, args.focal])) != 3 or any(m not in frame.index for m in pair):
            parser.error(f"Invalid baseline pair: {pair}")
    vertices = np.array([[0, 0], [1, 0], [.5, np.sqrt(3)/2]])
    n = args.resolution
    points = np.array([[i, j, n-i-j] for i in range(n+1) for j in range(n+1-i)]) / n
    xy = points @ vertices
    panels = []
    for left, right in pairs:
        names = [left, right, args.focal]
        payoff = frame.loc[names, names].to_numpy(dtype=float)
        velocity = np.array([replicator(x, payoff) for x in points]) @ vertices
        panels.append((names, payoff, velocity, np.linalg.norm(velocity, axis=1)))
    # One shared normalization across panels; preserve relative speed differences.
    maximum = max(max(float(p[3].max()) for p in panels), 1e-12)
    cols = min(3, len(panels))
    rows = (len(panels) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(3.75*cols, 3.9*rows),
                             squeeze=False, constrained_layout=True)
    fig.set_constrained_layout_pads(
        w_pad=0.01, h_pad=0.02, wspace=0.01, hspace=0.02
    )
    checks = []
    for ax, (names, payoff, velocity, speed) in zip(axes.flat, panels):
        border = vertices[[0, 1, 2, 0]]
        ax.plot(*border.T, color="0.5", linewidth=.7)
        nonzero = speed > 1e-12
        arrows = velocity / maximum * args.arrow_scale
        q = ax.quiver(*xy[nonzero].T, *arrows[nonzero].T, speed[nonzero]/maximum,
                      angles="xy", scale_units="xy", scale=1, cmap="viridis",
                      clim=(0, 1), width=.009, headwidth=3.5)
        ax.text(.5, .90, display_name(names[2]), ha="center", va="bottom",
                fontsize=args.model_fontsize)
        ax.text(0, -.055, display_name(names[0]), ha="center", va="top",
                fontsize=args.model_fontsize)
        ax.text(1, -.055, display_name(names[1]), ha="center", va="top",
                fontsize=args.model_fontsize)
        ax.set(xlim=(-.10, 1.10), ylim=(-.19, .99), aspect="equal")
        ax.axis("off")
        for baseline in range(2):
            differences = payoff[2] - payoff[baseline]
            checks.append(dict(focal=args.focal, left=names[0], right=names[1],
                               baseline=names[baseline],
                               min_focal_advantage=float(differences.min()),
                               strictly_dominates=bool((differences > 0).all()),
                               focal_invades_baseline=float(payoff[2, baseline]-payoff[baseline, baseline]),
                               baseline_invades_focal=float(payoff[baseline, 2]-payoff[2, 2]),
                               focal_pure_nash_gap=nash_gap(np.array([0., 0., 1.]), payoff)))
    for ax in list(axes.flat)[len(panels):]:
        ax.axis("off")
    fig.suptitle(args.title, fontsize=args.title_fontsize, fontweight="bold")
    fig.colorbar(q, ax=list(axes.flat), shrink=.7, label="Relative speed (shared scale)")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for suffix in ("png", "pdf"):
        fig.savefig(
            args.output_dir / f"{args.filename_stem}.{suffix}", dpi=220
        )
    plt.close(fig)
    pd.DataFrame(checks).to_csv(args.output_dir / "baseline_checks.csv", index=False)
    metadata = dict(payoff=str(args.payoff.resolve()), focal=args.focal, pairs=pairs,
                    resolution=n, grid_points_per_panel=len(points), title=args.title,
                    arrow_scale=args.arrow_scale,
                    model_fontsize=args.model_fontsize,
                    title_fontsize=args.title_fontsize,
                    filename_stem=args.filename_stem,
                    shared_speed_max=maximum,
                    interpretation="Point estimates within each selected subgame; not statistical significance or full-game validation")
    (args.output_dir / "panels_metadata.json").write_text(json.dumps(metadata, indent=2)+"\n")
    print(f"Saved {len(panels)} panels to {args.output_dir}")


if __name__ == "__main__":
    main()
