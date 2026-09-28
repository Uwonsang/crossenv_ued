"""Evaluate policy consistency on matched Overcooked subtask states.

The five subtasks are plate pickup, cooked-soup pickup, serving, onion pickup,
and placing the third onion into a pot. Within every A/B pair, the ego agent
has the same inventory, faces
the same kind of object in the same direction, and has the same intended
immediate action (Interact). The map and teammate placement differ. This
isolates whether a policy preserves its immediate subtask decision when the
environment changes. RSA and CKA are deliberately outside this analysis.
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from policy_value_concrete_example import (
    ACTION_NAMES,
    DIRECTIONS,
    FAMILY_LABELS,
    MODEL_SPECS,
    ROOT,
    checkpoint_path,
    distances,
    js_divergence,
    layout_record,
    load_config,
    load_params,
    neighbors,
    parse_model_spec,
)


SUBTASKS = (
    "plate_pickup",
    "cooked_soup_pickup",
    "serve",
    "onion_pickup",
    "onion_to_pot",
)
SUBTASK_LABELS = {
    "plate_pickup": "Plate pickup",
    "cooked_soup_pickup": "Cooked soup pickup",
    "serve": "Serve",
    "onion_pickup": "Onion pickup",
    "onion_to_pot": "Third onion to pot",
}
SUBTASK_SPECS = {
    "plate_pickup": {"target": "plate_pile", "inventory": "empty"},
    "cooked_soup_pickup": {"target": "pot", "inventory": "plate"},
    "serve": {"target": "goal", "inventory": "dish"},
    "onion_pickup": {"target": "onion_pile", "inventory": "empty"},
    "onion_to_pot": {"target": "pot", "inventory": "onion"},
}
INTERACT_INDEX = ACTION_NAMES.index("Interact")


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError(f"Cannot write an empty CSV: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def base_state(config: dict, record: dict, horizon: int):
    import jax
    import jax.numpy as jnp
    import jaxmarl

    layout = {
        key: value if key in ("height", "width") else jnp.asarray(value)
        for key, value in record["layout"].items()
    }
    kwargs = dict(config["ENV_KWARGS"])
    kwargs.update(
        layout=layout,
        random_reset=False,
        check_held_out=False,
        shuffle_inv_and_pot=False,
        max_steps=horizon,
    )
    env = jaxmarl.make("overcooked", **kwargs)
    _, state = env.custom_reset(
        jax.random.PRNGKey(record["map_seed"]),
        layout=layout,
        random_reset=False,
        shuffle_inv_and_pot=False,
    )
    return env, state, layout


def object_positions(state, layout: dict, object_name: str) -> list[tuple[int, int]]:
    """Return unpadded (x, y) locations for an Overcooked object type."""
    from jaxmarl.environments.overcooked.common import OBJECT_TO_INDEX

    height, width = int(layout["height"]), int(layout["width"])
    padding = (state.maze_map.shape[0] - height) // 2
    object_layer = np.asarray(
        state.maze_map[padding:padding + height, padding:padding + width, 0]
    )
    return [
        (int(x), int(y))
        for y, x in np.argwhere(object_layer == OBJECT_TO_INDEX[object_name])
    ]


def candidate_records(config: dict, base: dict, horizon: int) -> list[dict]:
    """Create at most one controlled state per subtask/orientation/map."""
    _, state, layout = base_state(config, base, horizon)
    floor = {
        (int(x), int(y)) for y, x in np.argwhere(~np.asarray(state.wall_map))
    }
    if len(floor) < 2:
        return []

    candidates = []
    for subtask in SUBTASKS:
        target_name = SUBTASK_SPECS[subtask]["target"]
        targets = object_positions(state, layout, target_name)
        for target in targets:
            for ego in neighbors(target, floor):
                delta = (target[0] - ego[0], target[1] - ego[1])
                if delta not in DIRECTIONS:
                    continue
                direction_index = DIRECTIONS.index(delta)
                teammate_candidates = floor - {ego}
                if not teammate_candidates:
                    continue
                from_ego = distances(floor, [ego])
                teammate = max(
                    teammate_candidates,
                    key=lambda position: (from_ego.get(position, -1), position),
                )
                candidates.append(dict(
                    base,
                    subtask=subtask,
                    subtask_label=SUBTASK_LABELS[subtask],
                    ego=list(ego),
                    teammate=list(teammate),
                    target=list(target),
                    target_object=target_name,
                    ego_inventory=SUBTASK_SPECS[subtask]["inventory"],
                    ego_direction=direction_index,
                    orientation=ACTION_NAMES[direction_index],
                    intended_action="Interact",
                    intended_action_index=INTERACT_INDEX,
                    pot_onions=2 if subtask == "onion_to_pot" else None,
                ))

    # A map contributes at most one state to each subtask/orientation stratum.
    unique = {}
    for record in candidates:
        unique.setdefault((record["subtask"], record["ego_direction"]), record)
    return list(unique.values())


def instantiate_controlled_state(config: dict, record: dict, horizon: int):
    """Instantiate a state whose Interact transition realizes the subtask."""
    import jax.numpy as jnp
    from jaxmarl.environments.overcooked.common import (
        COLOR_TO_INDEX,
        DIR_TO_VEC,
        OBJECT_TO_INDEX,
    )

    env, state, layout = base_state(config, record, horizon)
    padding = (state.maze_map.shape[0] - int(layout["height"])) // 2
    maze = state.maze_map
    empty = jnp.array([OBJECT_TO_INDEX["empty"], 0, 0], dtype=maze.dtype)
    for x, y in np.asarray(state.agent_pos):
        maze = maze.at[padding + int(y), padding + int(x)].set(empty)

    positions = jnp.asarray(
        [record["ego"], record["teammate"]], dtype=state.agent_pos.dtype
    )
    directions = jnp.asarray(
        [record["ego_direction"], 0], dtype=state.agent_dir_idx.dtype
    )
    for index, ((x, y), facing) in enumerate(zip(positions, directions)):
        agent = jnp.array([
            OBJECT_TO_INDEX["agent"],
            COLOR_TO_INDEX["red"] + index * 2,
            facing,
        ], dtype=maze.dtype)
        maze = maze.at[padding + y, padding + x].set(agent)

    # Ready pot = 0. A pot with one onion = 22 and with two onions = 21.
    pot_positions = object_positions(state, layout, "pot")
    if record["subtask"] in ("plate_pickup", "cooked_soup_pickup"):
        ready_pot = (
            tuple(record["target"])
            if record["target_object"] == "pot"
            else (pot_positions[0] if pot_positions else None)
        )
        if ready_pot is not None:
            x, y = ready_pot
            maze = maze.at[padding + y, padding + x, 2].set(
                jnp.asarray(0, dtype=maze.dtype)
            )
    elif record["subtask"] == "onion_pickup" and pot_positions:
        x, y = pot_positions[0]
        maze = maze.at[padding + y, padding + x, 2].set(
            jnp.asarray(22, dtype=maze.dtype)
        )
    elif record["subtask"] == "onion_to_pot":
        x, y = record["target"]
        maze = maze.at[padding + y, padding + x, 2].set(
            jnp.asarray(21, dtype=maze.dtype)
        )

    state = state.replace(
        agent_pos=positions,
        agent_dir_idx=directions,
        agent_dir=DIR_TO_VEC[directions],
        agent_inv=jnp.asarray([
            OBJECT_TO_INDEX[record["ego_inventory"]],
            OBJECT_TO_INDEX["empty"],
        ], dtype=state.agent_inv.dtype),
        maze_map=maze,
    )
    return env, state


def _pair_candidates(records: list[dict], count: int) -> list[tuple[dict, dict]]:
    """Pair different maps while cycling evenly over ego orientations."""
    by_direction = defaultdict(list)
    for record in sorted(records, key=lambda item: item["map_seed"]):
        by_direction[record["ego_direction"]].append(record)
    available = {
        direction: [
            (items[index], items[index + 1])
            for index in range(0, len(items) - 1, 2)
            if items[index]["map_seed"] != items[index + 1]["map_seed"]
        ]
        for direction, items in by_direction.items()
    }
    paired = []
    cursor = defaultdict(int)
    while len(paired) < count:
        added = False
        for direction in range(4):
            index = cursor[direction]
            if index < len(available.get(direction, [])):
                paired.append(available[direction][index])
                cursor[direction] += 1
                added = True
                if len(paired) == count:
                    break
        if not added:
            break
    return paired


def generate_pairs(config: dict, args: argparse.Namespace) -> list[dict]:
    import jax
    from jaxmarl.environments.overcooked import layouts
    try:
        from tqdm.auto import tqdm
    except ImportError:
        def tqdm(iterable, **_kwargs):
            return iterable

    requested_subtasks = tuple(args.subtasks)
    candidates = defaultdict(list)
    generator = getattr(layouts, f"make_{args.family}_9x9")
    for offset in tqdm(
        range(args.map_candidates), total=args.map_candidates,
        desc=f"Generating {args.family} subtask candidates", unit="map",
        dynamic_ncols=True,
    ):
        map_seed = args.map_seed + offset
        base = layout_record(
            generator(jax.random.PRNGKey(map_seed), ik=True),
            map_seed,
            args.family,
        )
        for record in candidate_records(config, base, args.horizon):
            if record["subtask"] in requested_subtasks:
                candidates[record["subtask"]].append(record)

    pairs = []
    counts = {}
    pair_id = 0
    for subtask in requested_subtasks:
        subtask_pairs = _pair_candidates(
            candidates[subtask], args.pairs_per_subtask
        )
        counts[subtask] = len(subtask_pairs)
        for case_id, (left, right) in enumerate(subtask_pairs):
            pairs.append({
                "pair_id": pair_id,
                "case_id": case_id,
                "subtask": subtask,
                "subtask_label": SUBTASK_LABELS[subtask],
                "orientation": left["orientation"],
                "intended_action": "Interact",
                "states": [dict(left, variant="A"), dict(right, variant="B")],
                "pair_selection": {
                    "method": "matched_subtask_and_orientation",
                    "uses_model_outputs": False,
                    "same_subtask": True,
                    "same_ego_inventory": True,
                    "same_target_object": True,
                    "same_orientation": True,
                    "intended_action": "Interact",
                    "pot_onions": 2 if subtask == "onion_to_pot" else None,
                    "different_map_seeds": True,
                },
            })
            pair_id += 1

    missing = [
        subtask for subtask in requested_subtasks
        if counts[subtask] < args.pairs_per_subtask
    ]
    if missing:
        detail = ", ".join(
            f"{subtask}={counts[subtask]}/{args.pairs_per_subtask}"
            for subtask in missing
        )
        if not args.allow_fewer:
            raise RuntimeError(
                f"Not enough balanced subtask pairs ({detail}). Increase "
                "--map-candidates or pass --allow-fewer."
            )
        print(f"Warning: using fewer pairs for {detail}")
    if not pairs:
        raise RuntimeError("No subtask pairs were generated")
    return pairs


def make_predictor(config: dict, model: str, checkpoint: Path, horizon: int):
    import jax
    import jax.numpy as jnp
    from actor_networks import ScannedRNN

    spec = MODEL_SPECS[model]
    network = spec["value_network"](len(ACTION_NAMES), config=config)
    params = load_params(checkpoint)
    hidden_dim = int(config["GRU_HIDDEN_DIM"])

    @jax.jit
    def predict(observation_batch, positions):
        hidden = ScannedRNN.initialize_carry(2, hidden_dim)
        if spec["value_separate_hidden"]:
            hidden = (hidden, hidden)
        _, policy, _ = network.apply(
            params,
            hidden,
            (
                observation_batch[jnp.newaxis],
                jnp.zeros((1, 2), dtype=bool),
                positions[jnp.newaxis],
            ),
        )
        return policy.probs[0, 0]

    def predict_record(record: dict) -> np.ndarray:
        env, state = instantiate_controlled_state(config, record, horizon)
        observations = env.get_obs(state)
        observation_batch = jnp.stack([
            observations[agent].reshape(-1) for agent in env.agents
        ])
        return np.asarray(predict(observation_batch, state.agent_pos))

    return predict_record


def evaluate_pairs(
    config: dict, pairs: list[dict], args: argparse.Namespace
) -> list[dict]:
    rows = []
    for model, num_envs in args.models:
        for seed in args.seeds:
            checkpoint = checkpoint_path(args.model_root, model, num_envs, seed)
            if checkpoint is None:
                print(f"Missing checkpoint: {model} {num_envs} seed {seed}")
                continue
            predict = make_predictor(config, model, checkpoint, args.horizon)
            for pair in pairs:
                left, right = pair["states"]
                probabilities_a = predict(left)
                probabilities_b = predict(right)
                argmax_a = int(probabilities_a.argmax())
                argmax_b = int(probabilities_b.argmax())
                rows.append({
                    "family": args.family,
                    "pair_id": pair["pair_id"],
                    "case_id": pair["case_id"],
                    "subtask": pair["subtask"],
                    "subtask_label": pair["subtask_label"],
                    "orientation": pair["orientation"],
                    "intended_action": "Interact",
                    "pot_onions": left.get("pot_onions"),
                    "map_seed_a": left["map_seed"],
                    "map_seed_b": right["map_seed"],
                    "model": model,
                    "num_envs": num_envs,
                    "seed": seed,
                    "checkpoint": str(checkpoint),
                    "policy_js_nats": js_divergence(
                        probabilities_a, probabilities_b
                    ),
                    "argmax_a": ACTION_NAMES[argmax_a],
                    "argmax_b": ACTION_NAMES[argmax_b],
                    "action_agreement": argmax_a == argmax_b,
                    "interact_a": argmax_a == INTERACT_INDEX,
                    "interact_b": argmax_b == INTERACT_INDEX,
                    "both_interact": (
                        argmax_a == INTERACT_INDEX
                        and argmax_b == INTERACT_INDEX
                    ),
                    "interact_probability_a": float(
                        probabilities_a[INTERACT_INDEX]
                    ),
                    "interact_probability_b": float(
                        probabilities_b[INTERACT_INDEX]
                    ),
                    **{
                        f"prob_{action.lower()}_a": float(probability)
                        for action, probability in zip(ACTION_NAMES, probabilities_a)
                    },
                    **{
                        f"prob_{action.lower()}_b": float(probability)
                        for action, probability in zip(ACTION_NAMES, probabilities_b)
                    },
                })
            print(
                f"Evaluated {model} {num_envs} seed {seed} on {len(pairs)} "
                "subtask pairs"
            )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-root", type=Path, required=True)
    parser.add_argument(
        "--models", nargs="+", type=parse_model_spec,
        default=[("CEC", 64), ("CEC_IDAAC", 64)],
    )
    parser.add_argument("--seeds", type=int, nargs="+", default=list(range(6)))
    parser.add_argument(
        "--family", choices=tuple(FAMILY_LABELS), default="counter_circuit"
    )
    parser.add_argument(
        "--subtasks", nargs="+", choices=SUBTASKS, default=list(SUBTASKS)
    )
    parser.add_argument("--map-seed", type=int, default=1701)
    parser.add_argument("--map-candidates", type=int, default=3000)
    parser.add_argument("--pairs-per-subtask", type=int, default=50)
    parser.add_argument("--allow-fewer", action="store_true")
    parser.add_argument("--horizon", type=int, default=200)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument(
        "--reuse-pairs", action="store_true",
        help="Load subtask_state_pairs.json instead of generating maps again.",
    )
    parser.add_argument("--output-root", type=Path)
    parser.add_argument(
        "--config", type=Path,
        default=ROOT / "baselines/CEC_UED/config/ippo_overcooked_CEC_gradient.yaml",
    )
    args = parser.parse_args()
    if args.map_candidates < 1 or args.pairs_per_subtask < 1:
        parser.error(
            "--map-candidates and --pairs-per-subtask must be positive"
        )
    if args.prepare_only and args.reuse_pairs:
        parser.error("--prepare-only and --reuse-pairs cannot be used together")
    args.subtasks = tuple(dict.fromkeys(args.subtasks))
    args.model_root = args.model_root.expanduser()
    output_root = args.output_root or (
        args.model_root / "analysis" / "subtask_pair_consistency"
    )
    output_dir = output_root / args.family
    output_dir.mkdir(parents=True, exist_ok=True)

    config = load_config(args.config)
    pairs_path = output_dir / "subtask_state_pairs.json"
    if args.reuse_pairs:
        if not pairs_path.is_file():
            parser.error(f"Saved pair file does not exist: {pairs_path}")
        payload = json.loads(pairs_path.read_text(encoding="utf-8"))
        if payload.get("family") != args.family:
            parser.error(
                f"{pairs_path} contains family={payload.get('family')!r}, "
                f"expected {args.family!r}"
            )
        pairs = payload.get("pairs", [])
        if not pairs:
            parser.error(f"Saved pair file contains no pairs: {pairs_path}")
        print(f"Loaded {len(pairs)} saved subtask pairs from {pairs_path}")
    else:
        pairs = generate_pairs(config, args)
        pairs_path.write_text(
            json.dumps({
                "family": args.family,
                "subtasks": list(args.subtasks),
                "pairs_per_subtask_requested": args.pairs_per_subtask,
                "pair_definition": (
                    "same subtask, inventory, target type, orientation, and "
                    "intended Interact action; different map seeds"
                ),
                "pairs": pairs,
            }, indent=2),
            encoding="utf-8",
        )
        print(f"Saved {len(pairs)} subtask pairs to {pairs_path}")
    if args.prepare_only:
        return
    rows = evaluate_pairs(config, pairs, args)
    if not rows:
        raise RuntimeError("No usable checkpoint was found")
    write_csv(output_dir / "subtask_consistency_metrics.csv", rows)
    print(f"Saved subtask consistency metrics to {output_dir}")


if __name__ == "__main__":
    main()
