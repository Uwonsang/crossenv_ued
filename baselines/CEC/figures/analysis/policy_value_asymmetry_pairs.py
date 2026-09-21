"""Generate controlled state pairs from the five existing PCG layout families."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "baselines/CEC"))

try:  # Support both direct execution and package-style imports in tests.
    from .policy_value_asymmetry_common import (
        FAMILIES, delivery_plan, initial_state,
    )
except ImportError:
    from policy_value_asymmetry_common import (  # noqa: E402
        FAMILIES, delivery_plan, initial_state,
    )


FAMILY_LABELS = {
    "asymm_advantages": "Asymmetric Advantages",
    "coord_ring": "Coordination Ring",
    "counter_circuit": "Counter Circuit",
    "forced_coord": "Forced Coordination",
    "cramped_room": "Cramped Room",
}
DIRECTION_VECTORS = ((0, -1), (0, 1), (1, 0), (-1, 0))
DIRECTION_MARKERS = ("^", "v", ">", "<")


def positions(indices, width):
    return {(int(index) % width, int(index) // width) for index in indices}


def path_positions(record):
    layout = record["layout"]
    width = int(layout["width"])
    walls = positions(layout["wall_idx"], width)
    agents = list(layout["agent_idx"])
    current = (int(agents[0]) % width, int(agents[0]) // width)
    partner = (int(agents[1]) % width, int(agents[1]) // width)
    result = [current]
    for action in record["plan"]:
        if 0 <= int(action) < 4:
            dx, dy = DIRECTION_VECTORS[int(action)]
            target = (current[0] + dx, current[1] + dy)
            if target not in walls and target != partner:
                current = target
            result.append(current)
    return result


def draw_layout(ax, record, variant):
    from matplotlib.patches import Circle, Rectangle

    layout = record["layout"]
    height, width = int(layout["height"]), int(layout["width"])
    walls = positions(layout["wall_idx"], width)
    for y in range(height):
        for x in range(width):
            ax.add_patch(Rectangle(
                (x, y), 1, 1,
                facecolor="#707070" if (x, y) in walls else "#f7f7f7",
                edgecolor="#4a4a4a", linewidth=.45,
            ))
    facility_styles = (
        ("goal_idx", "#20df36", "G", "black"),
        ("onion_pile_idx", "#ffe600", "O", "black"),
        ("plate_pile_idx", "white", "P", "black"),
        ("pot_idx", "#1b1b1b", "Pot", "white"),
    )
    for key, color, label, text_color in facility_styles:
        for x, y in positions(layout[key], width):
            ax.add_patch(Rectangle(
                (x + .12, y + .12), .76, .76, facecolor=color,
                edgecolor="#303030", linewidth=.65,
            ))
            ax.text(x + .5, y + .52, label, ha="center", va="center",
                    fontsize=7, color=text_color, fontweight="bold")
    route = path_positions(record)
    if len(route) > 1:
        xs = [x + .5 for x, _ in route]
        ys = [y + .5 for _, y in route]
        ax.plot(xs, ys, color="#d62728", linewidth=2, alpha=.8, zorder=5)
        ax.scatter(xs[1:-1], ys[1:-1], s=9, color="#d62728", zorder=6)
    agents = list(layout["agent_idx"])
    direction = int(record["direction"])
    for index, color in ((int(agents[0]), "#d62728"),
                         (int(agents[1]), "#2455d6")):
        x, y = index % width, index // width
        ax.scatter(x + .5, y + .5, s=150, marker=DIRECTION_MARKERS[direction],
                   color=color, edgecolor="black", linewidth=.6, zorder=8)
    focal_x, focal_y = int(agents[0]) % width, int(agents[0]) // width
    ax.add_patch(Circle(
        (focal_x + .72, focal_y + .25), .12,
        facecolor="white", edgecolor="black", linewidth=.6, zorder=9,
    ))
    ax.set(xlim=(0, width), ylim=(height, 0), aspect="equal")
    ax.axis("off")
    ax.set_title(
        f"{variant.title()} · {len(record['plan'])} steps · seed {record['map_seed']}",
        fontsize=10, fontweight="bold", pad=5,
    )


def save_map_visualizations(payload, output_dir, dpi):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    pairs = payload["pairs"]
    if not pairs:
        print("No matched pairs; no map visualizations were created.")
        return
    output_dir.mkdir(parents=True, exist_ok=True)
    families = list(dict.fromkeys(pair["family"] for pair in pairs))
    for family in families:
        family_pairs = [pair for pair in pairs if pair["family"] == family]
        figure, axes = plt.subplots(
            len(family_pairs), 2,
            figsize=(7, max(3.1, 3 * len(family_pairs))), squeeze=False,
        )
        for row, pair in enumerate(family_pairs):
            draw_layout(axes[row, 0], pair["short"], "short")
            draw_layout(axes[row, 1], pair["long"], "long")
            axes[row, 0].text(
                -.04, .5, f"Pair {row + 1}", transform=axes[row, 0].transAxes,
                ha="right", va="center", rotation=90, fontsize=9,
            )
        figure.suptitle(
            FAMILY_LABELS.get(family, family), fontsize=14, fontweight="bold",
        )
        figure.legend(handles=(
            Line2D([0], [0], color="#d62728", marker=">", lw=2,
                   label="Agent 0 holding soup / shortest route"),
            Line2D([0], [0], color="#2455d6", marker=">", lw=0,
                   label="Stationary partner"),
        ), loc="lower center", ncol=2, frameon=False, fontsize=8)
        figure.tight_layout(rect=(0, .035, 1, .97), h_pad=1, w_pad=.5)
        stem = output_dir / family
        figure.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
        figure.savefig(stem.with_suffix(".png"), dpi=dpi, bbox_inches="tight")
        plt.close(figure)
        print(f"Saved: {stem.with_suffix('.pdf')}")
        print(f"Saved: {stem.with_suffix('.png')}")


def generate_pairs(args):
    import jax
    import numpy as np
    from jaxmarl.environments.overcooked import layouts
    from jaxmarl.environments.overcooked.common import DIR_TO_VEC
    from environment_representation_probe import load_config

    config = load_config(args.config)
    if config["ENV_KWARGS"].get("partial_obs", False):
        raise ValueError("This analysis requires the model's full 9x9 observation")
    directions = [tuple(map(int, value)) for value in np.asarray(DIR_TO_VEC)]
    pairs, diagnostics = [], {}
    for family_index, family in enumerate(FAMILIES):
        generator = getattr(layouts, "make_" + family + "_9x9")
        groups, selected, reachable = {}, [], 0
        for candidate in range(args.candidates_per_family):
            seed = args.map_seed + family_index * args.candidates_per_family + candidate
            layout = generator(jax.random.PRNGKey(seed), ik=True)
            record = dict(
                layout={key: np.asarray(value).tolist() for key, value in layout.items()},
                reset_seed=seed,
                direction=directions.index((1, 0)),
            )
            _, state = initial_state(config, record, args.horizon)
            walls = np.asarray(state.wall_map)
            partner = tuple(np.asarray(state.agent_pos[1]).tolist())
            start = tuple(np.asarray(state.agent_pos[0]).tolist())
            floor = {
                (x, y) for y, x in np.argwhere(~walls.astype(bool))
            } - {partner}
            goals = set(map(tuple, np.asarray(state.goal_pos).tolist()))
            plan, first_actions = delivery_plan(
                floor, goals, start, directions, record["direction"]
            )
            if plan is None or len(first_actions) != 1 or len(plan) > args.horizon:
                continue
            reachable += 1
            # Keep the controlled agent configuration and optimal first action
            # fixed, while allowing the full 9x9 layout observation to differ.
            key = (start, partner, first_actions[0])
            record.update(plan=plan, map_seed=seed)
            for other in groups.get(key, []):
                if abs(len(other["plan"]) - len(plan)) >= args.min_step_gap:
                    short, long = sorted((other, record), key=lambda item: len(item["plan"]))
                    selected.append(dict(
                        pair=f"{family}_{len(selected)}",
                        family=family,
                        short=short,
                        long=long,
                    ))
                    groups.pop(key, None)
                    break
            else:
                groups.setdefault(key, []).append(record)
                continue
            if len(selected) >= args.pairs_per_family:
                break
        pairs.extend(selected)
        diagnostics[family] = dict(
            sampled=candidate + 1,
            reachable_unique_first=reachable,
            matched_pairs=len(selected),
        )
        print(f"{family}: {len(selected)} matched pairs", flush=True)

    payload = dict(
        pairs=pairs,
        diagnostics=diagnostics,
        map_seed=args.map_seed,
        min_step_gap=args.min_step_gap,
        config=str(args.config),
        observation_scope="full 9x9x26 model observation",
        matching=("same focal/partner positions, direction, and unique oracle "
                  "first action; different PCG layout variants"),
        intervention="agent 0 holds soup; stationary partner",
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2))
    print(f"Saved: {args.output}")
    visualization_dir = (
        args.visualization_dir or args.output.parent / "map_visualizations"
    )
    save_map_visualizations(payload, visualization_dir, args.dpi)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--map-seed", type=int, default=1701)
    parser.add_argument("--candidates-per-family", type=int, default=500)
    parser.add_argument("--pairs-per-family", type=int, default=5)
    parser.add_argument("--min-step-gap", type=int, default=2)
    parser.add_argument("--horizon", type=int, default=400)
    parser.add_argument(
        "--config", type=Path,
        default=ROOT / "baselines/CEC_UED/config/ippo_overcooked_CEC_gradient.yaml",
    )
    parser.add_argument(
        "--output", type=Path,
        default=ROOT / "artifacts/policy_value_asymmetry/state_pairs.json",
    )
    parser.add_argument(
        "--visualization-dir", type=Path,
        help="Default: <state_pairs directory>/map_visualizations",
    )
    parser.add_argument("--dpi", type=int, default=250)
    args = parser.parse_args()
    if min(
        args.candidates_per_family, args.pairs_per_family,
        args.min_step_gap, args.horizon,
        args.dpi,
    ) < 1:
        parser.error("Invalid pair-generation parameters")
    args.output = args.output.expanduser()
    if args.visualization_dir is not None:
        args.visualization_dir = args.visualization_dir.expanduser()
    generate_pairs(args)


if __name__ == "__main__":
    main()
