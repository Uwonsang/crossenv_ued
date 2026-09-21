"""Generate controlled state pairs from the five existing PCG layout families."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "baselines/CEC"))

from policy_value_asymmetry_common import (  # noqa: E402
    FAMILIES, delivery_plan, initial_state,
)


def generate_pairs(args):
    import jax
    import numpy as np
    from jaxmarl.environments.overcooked import layouts
    from jaxmarl.environments.overcooked.common import DIR_TO_VEC
    from environment_representation_probe import load_config

    config = load_config(args.config)
    if config["ENV_KWARGS"].get("partial_obs", False):
        raise ValueError("Pair matching currently requires full spatial observations")
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
            env, state = initial_state(config, record, args.horizon)
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
            observation = np.asarray(env.get_obs(state)["agent_0"])
            if observation.ndim != 3 or observation.shape[:2] != walls.shape:
                raise ValueError("Expected full H x W x channels observation")
            radius = args.local_radius
            padded = np.pad(
                observation,
                ((radius, radius), (radius, radius), (0, 0)),
                constant_values=-1,
            )
            x, y = start
            patch = padded[y:y + 2 * radius + 1, x:x + 2 * radius + 1]
            key = (start, partner, first_actions[0], patch.tobytes())
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
        local_radius=args.local_radius,
        min_step_gap=args.min_step_gap,
        config=str(args.config),
        intervention="agent 0 holds soup; stationary partner",
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2))
    print(f"Saved: {args.output}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--map-seed", type=int, default=1701)
    parser.add_argument("--candidates-per-family", type=int, default=500)
    parser.add_argument("--pairs-per-family", type=int, default=5)
    parser.add_argument("--local-radius", type=int, default=1)
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
    args = parser.parse_args()
    if min(
        args.candidates_per_family, args.pairs_per_family,
        args.min_step_gap, args.horizon,
    ) < 1 or args.local_radius < 0:
        parser.error("Invalid pair-generation parameters")
    args.output = args.output.expanduser()
    generate_pairs(args)


if __name__ == "__main__":
    main()
