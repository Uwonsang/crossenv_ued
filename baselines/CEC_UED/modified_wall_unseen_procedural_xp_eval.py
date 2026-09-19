import json
import os

import hydra
import jax
import numpy as np
import pandas as pd
from omegaconf import OmegaConf
from tqdm import tqdm

from modified_wall_procedural_xp_eval import (
    MODEL_SPECS,
    bank_hash,
    evaluation_rngs,
    load_checkpoints,
    load_or_create_state_bank,
    make_pair_evaluator,
    resolve_path,
    stack_states,
    state_from_arrays,
    summarize_results,
)


UNSEEN_WALLS = {
    "anti_diagonal": ((1, 3), (2, 2), (3, 1)),
    "horizontal": ((1, 2), (2, 2), (3, 2)),
    "vertical": ((2, 1), (2, 2), (2, 3)),
    "checker": ((1, 1), (3, 1), (1, 3), (3, 3)),
}


def wall_map_from_coordinates(coordinates):
    wall_map = np.zeros((5, 5), dtype=bool)
    for x, y in coordinates:
        wall_map[y, x] = True
    return wall_map


def sample_unique_states(wall_map, count, rng):
    free_yx = np.argwhere(~wall_map)
    free_xy = free_yx[:, ::-1]
    states = []
    signatures = set()
    while len(states) < count:
        selected = free_xy[rng.choice(len(free_xy), size=4, replace=False)]
        agent_pos = selected[:2].astype(np.int32)
        goal_pos = selected[2:].astype(np.int32)
        canonical_goals = goal_pos[np.lexsort((goal_pos[:, 0], goal_pos[:, 1]))]
        signature = (agent_pos.tobytes(), canonical_goals.tobytes())
        if signature in signatures:
            continue
        signatures.add(signature)
        states.append((agent_pos, goal_pos, wall_map.copy()))
    return states


def generate_unseen_bank(config, base_bank):
    base_count = int(config["NUM_PROCEDURAL_TASKS"])
    block_count = int(config["TASKS_PER_UNSEEN_LAYOUT"])
    if base_count != 100 or block_count != 100:
        raise ValueError("This cumulative evaluation requires 100 tasks per block")

    agent_positions = [np.asarray(value) for value in base_bank["agent_pos"][:base_count]]
    goal_positions = [np.asarray(value) for value in base_bank["goal_pos"][:base_count]]
    wall_maps = [np.asarray(value) for value in base_bank["wall_map"][:base_count]]
    task_blocks = ["heldout_seen_walls"] * base_count
    eval_layouts = [str(value) for value in base_bank["procedural_layouts"][:base_count]]

    seed_sequence = np.random.SeedSequence(int(config["STATE_SEED"]))
    child_seeds = seed_sequence.spawn(len(UNSEEN_WALLS))
    for (layout_name, coordinates), child_seed in zip(UNSEEN_WALLS.items(), child_seeds):
        wall_map = wall_map_from_coordinates(coordinates)
        states = sample_unique_states(
            wall_map, block_count, np.random.default_rng(child_seed)
        )
        for agent_pos, goal_pos, state_wall_map in states:
            agent_positions.append(agent_pos)
            goal_positions.append(goal_pos)
            wall_maps.append(state_wall_map)
            task_blocks.append(layout_name)
            eval_layouts.append(layout_name)

    agent_pos = np.stack(agent_positions)
    goal_pos = np.stack(goal_positions)
    wall_map = np.stack(wall_maps)
    return {
        "agent_pos": agent_pos,
        "goal_pos": goal_pos,
        "wall_map": wall_map,
        "task_blocks": np.asarray(task_blocks),
        "eval_layouts": np.asarray(eval_layouts),
        "hash": bank_hash(agent_pos, goal_pos, wall_map),
    }


def load_or_create_unseen_bank(config, base_bank, base_manifest):
    bank_path = resolve_path(config["UNSEEN_BANK_PATH"])
    bank_path.parent.mkdir(parents=True, exist_ok=True)
    expected = generate_unseen_bank(config, base_bank)

    if bank_path.exists():
        with np.load(bank_path, allow_pickle=False) as stored:
            bank = {name: stored[name] for name in stored.files}
        actual_hash = bank_hash(bank["agent_pos"], bank["goal_pos"], bank["wall_map"])
        if actual_hash != expected["hash"]:
            raise ValueError(
                f"Unseen state bank mismatch at {bank_path}. "
                "Remove it only if you intentionally changed the task definitions."
            )
        bank["hash"] = actual_hash
    else:
        temporary_path = bank_path.with_suffix(".tmp.npz")
        np.savez_compressed(
            temporary_path,
            agent_pos=expected["agent_pos"],
            goal_pos=expected["goal_pos"],
            wall_map=expected["wall_map"],
            task_blocks=expected["task_blocks"],
            eval_layouts=expected["eval_layouts"],
        )
        os.replace(temporary_path, bank_path)
        bank = expected

    block_counts = {
        block: int(np.sum(bank["task_blocks"] == block))
        for block in ("heldout_seen_walls", *UNSEEN_WALLS)
    }
    manifest = {
        "sha256": bank["hash"],
        "base_heldout_sha256": base_manifest["sha256"],
        "num_tasks": int(len(bank["agent_pos"])),
        "task_limits": [int(value) for value in config["TASK_LIMITS"]],
        "block_counts": block_counts,
        "wall_coordinates": {
            name: [list(coordinate) for coordinate in coordinates]
            for name, coordinates in UNSEEN_WALLS.items()
        },
        "state_seed": int(config["STATE_SEED"]),
    }
    manifest_path = bank_path.with_suffix(".json")
    with open(manifest_path, "w", encoding="utf-8") as file:
        json.dump(manifest, file, indent=2)
    print(f"Unseen procedural bank: {bank_path}")
    print(json.dumps(manifest, indent=2))
    return bank, bank_path, manifest_path


def states_from_unseen_bank(bank):
    states = [
        state_from_arrays(agent_pos, goal_pos, wall_map)
        for agent_pos, goal_pos, wall_map in zip(
            bank["agent_pos"], bank["goal_pos"], bank["wall_map"]
        )
    ]
    return stack_states(states)


def evaluate_in_batches(evaluate_pair, params_1, params_2, states, rngs, batch_size):
    rewards = []
    num_states = len(rngs)
    for start in range(0, num_states, batch_size):
        stop = min(start + batch_size, num_states)
        state_batch = jax.tree.map(lambda value: value[start:stop], states)
        rewards.append(
            np.asarray(
                jax.device_get(
                    evaluate_pair(
                        params_1,
                        params_2,
                        state_batch,
                        rngs[start:stop],
                    )
                )
            )
        )
    return np.concatenate(rewards)


def add_metadata(frame, model_group, spec, num_envs, task_count):
    frame.insert(0, "model_group", model_group)
    frame.insert(1, "algorithm", spec["algorithm"])
    frame.insert(2, "num_envs", num_envs)
    frame.insert(3, "train_map", spec["train_map"])
    frame.insert(4, "task_count", task_count)


@hydra.main(
    version_base=None,
    config_path="xp_config",
    config_name="modified_wall_unseen_procedural_xp",
)
def main(config):
    config = OmegaConf.to_container(config, resolve=True)
    model_group = str(config["MODEL_GROUP"])
    if model_group not in MODEL_SPECS:
        raise ValueError(f"Unknown MODEL_GROUP={model_group!r}")
    if int(config["CKPT_NUM_ENVS"]) != 256:
        raise ValueError("Unseen procedural XP is defined for 65K/NUM_ENVS=256 checkpoints")

    task_limits = [int(value) for value in config["TASK_LIMITS"]]
    if task_limits != [100, 200, 300, 400, 500]:
        raise ValueError(f"TASK_LIMITS must be [100, 200, 300, 400, 500], got {task_limits}")
    seeds = [int(seed) for seed in config["SEEDS"]]
    if len(seeds) < 2:
        raise ValueError("Cross-play requires at least two seeds")
    eval_batch_size = int(config["EVAL_BATCH_SIZE"])
    if eval_batch_size <= 0:
        raise ValueError("EVAL_BATCH_SIZE must be positive")

    base_bank, _, _, base_manifest = load_or_create_state_bank(config)
    bank, _, _ = load_or_create_unseen_bank(config, base_bank, base_manifest)
    params, _ = load_checkpoints(config, model_group)
    spec = MODEL_SPECS[model_group]
    evaluate_pair = make_pair_evaluator(config, spec)
    states = states_from_unseen_bank(bank)
    rngs = evaluation_rngs(int(config["ACTION_SEED"]), len(bank["agent_pos"]))

    rows = []
    ordered_pairs = [
        (seed_1, seed_2)
        for seed_1 in seeds
        for seed_2 in seeds
        if seed_1 != seed_2
    ]
    for seed_1, seed_2 in tqdm(ordered_pairs, desc=f"{model_group} unseen XP"):
        rewards = evaluate_in_batches(
            evaluate_pair,
            params[seed_1],
            params[seed_2],
            states,
            rngs,
            eval_batch_size,
        )
        for state_id, reward in enumerate(rewards):
            rows.append(
                {
                    "model_group": model_group,
                    "algorithm": spec["algorithm"],
                    "num_envs": 256,
                    "train_map": spec["train_map"],
                    "split": "procedural",
                    "state_id": state_id,
                    "task_block": str(bank["task_blocks"][state_id]),
                    "eval_layout": str(bank["eval_layouts"][state_id]),
                    "policy_1": seed_1,
                    "policy_2": seed_2,
                    "reward": float(reward),
                    "normalized_return": float(reward)
                    / (2.0 * int(config["NUM_STEPS"])),
                    "success": bool(reward > -int(config["NUM_STEPS"])),
                }
            )

    all_episodes = pd.DataFrame(rows)
    results_dir = resolve_path(config["RESULTS_DIR"])
    results_dir.mkdir(parents=True, exist_ok=True)
    result_prefix = f"{model_group}_numenv256"
    cumulative_summaries = []

    for task_count in task_limits:
        episodes = all_episodes[all_episodes["state_id"] < task_count].copy()
        episodes.insert(4, "task_count", task_count)
        ordered, pairs, summary = summarize_results(episodes)
        for frame in (ordered, pairs, summary):
            add_metadata(frame, model_group, spec, 256, task_count)
        cumulative_summaries.append(summary)

        prefix = results_dir / f"{result_prefix}_tasks{task_count}"
        episodes.to_csv(f"{prefix}_episodes.csv", index=False)
        ordered.to_csv(f"{prefix}_ordered_pairs.csv", index=False)
        pairs.to_csv(f"{prefix}_pairs.csv", index=False)
        summary.to_csv(f"{prefix}_summary.csv", index=False)

    cumulative_summary = pd.concat(cumulative_summaries, ignore_index=True)
    cumulative_path = results_dir / f"{result_prefix}_cumulative_summary.csv"
    cumulative_summary.to_csv(cumulative_path, index=False)
    print(cumulative_summary.to_string(index=False))
    print(f"Saved cumulative results under {results_dir}")

    jax.effects_barrier()
    jax.clear_caches()


if __name__ == "__main__":
    main()
