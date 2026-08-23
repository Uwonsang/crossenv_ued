import glob
import hashlib
import json
import os
import pickle
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import distrax
import hydra
import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
import yaml
from hydra.utils import get_original_cwd
from omegaconf import OmegaConf
from tqdm import tqdm

from jaxmarl.environments.toy_coop.toy_coop_no_pink import State, ToyCoopNoPink


MODEL_SPECS = {
    "ippo_empty": {
        "family": "ippo",
        "algorithm": "IPPO",
        "train_map": "empty",
        "fixed_layouts": ["empty"],
        "wandb_group": "XP_IPPO_64_EMPTY_x_IPPO_64_EMPTY",
    },
    "ippo_wall_a": {
        "family": "ippo",
        "algorithm": "IPPO",
        "train_map": "wall_a",
        "fixed_layouts": ["wall_a"],
        "wandb_group": "XP_IPPO_64_WALL_A_x_IPPO_64_WALL_A",
    },
    "e3t_empty": {
        "family": "e3t",
        "algorithm": "E3T",
        "train_map": "empty",
        "fixed_layouts": ["empty"],
        "wandb_group": "XP_E3T_64_EMPTY_x_E3T_64_EMPTY",
    },
    "e3t_wall_a": {
        "family": "e3t",
        "algorithm": "E3T",
        "train_map": "wall_a",
        "fixed_layouts": ["wall_a"],
        "wandb_group": "XP_E3T_64_WALL_A_x_E3T_64_WALL_A",
    },
    "cec": {
        "family": "ippo",
        "algorithm": "CEC",
        "train_map": "mixed",
        "fixed_layouts": ["empty", "wall_a"],
        "wandb_group": "XP_ONLY_CEC_64_x_ONLY_CEC_64",
    },
    "idaac_cec": {
        "family": "idaac",
        "algorithm": "CEC+IDAAC",
        "train_map": "mixed",
        "fixed_layouts": ["empty", "wall_a"],
        "wandb_group": "XP_IDAAC_CEC_256_x_IDAAC_CEC_256",
    },
}


def resolve_path(path):
    path = Path(path)
    if path.is_absolute():
        return path
    return Path(get_original_cwd()) / path


def checkpoint_patterns(config, model_group, seed):
    base = Path("ckpts")
    if model_group.startswith("ippo_"):
        train_map = MODEL_SPECS[model_group]["train_map"]
        return [
            str(
                base
                / "ippo"
                / config["ENV_NAME"]
                / "modified_wall"
                / train_map
                / "ikFalse"
                / "reset_all"
                / "ippo_layout_eval"
                / "**"
                / f"seed{seed}_ckpt0_improved_updates*.pkl"
            )
        ]
    if model_group.startswith("e3t_"):
        train_map = MODEL_SPECS[model_group]["train_map"]
        return [
            str(
                base
                / "e3t"
                / config["ENV_NAME"]
                / "modified_wall"
                / train_map
                / "ikFalse"
                / "reset_all"
                / "e3t"
                / "**"
                / f"seed{seed}_ckpt0_e3t_updates*.pkl"
            )
        ]
    if model_group == "cec":
        root = f"mixed_empty_wall_a_{config['CEC_CKPT_TAG']}"
        return [
            str(
                base
                / "ippo"
                / config["ENV_NAME"]
                / "modified_wall"
                / root
                / "ikTrue"
                / "reset_all"
                / "cec_layout_eval"
                / "**"
                / f"seed{seed}_ckpt0_improved_updates*.pkl"
            )
        ]
    if model_group == "idaac_cec":
        root = f"mixed_empty_wall_a_{config['IDAAC_CKPT_TAG']}"
        return [
            str(
                base
                / "idaac"
                / config["ENV_NAME"]
                / "modified_wall"
                / root
                / "ikTrue"
                / "reset_all"
                / "**"
                / f"seed{seed}_ckpt0_improved_updates*.pkl"
            )
        ]
    raise ValueError(f"Unknown MODEL_GROUP: {model_group}")


def load_checkpoints(config, model_group):
    params = {}
    paths = {}
    repo_root = Path(get_original_cwd())
    for seed in [int(seed) for seed in config["SEEDS"]]:
        matches = []
        for pattern in checkpoint_patterns(config, model_group, seed):
            matches.extend(glob.glob(str(repo_root / pattern), recursive=True))
        if not matches:
            searched = "\n".join(checkpoint_patterns(config, model_group, seed))
            raise FileNotFoundError(
                f"Missing {model_group} seed{seed} checkpoint. Searched:\n{searched}"
            )
        path = max(matches, key=os.path.getmtime)
        with open(path, "rb") as file:
            checkpoint = pickle.load(file)
        params[seed] = checkpoint["params"]
        paths[seed] = path
        print(f"Loaded {model_group} seed{seed}: {path}")
    return params, paths


def stack_states(states):
    return jax.tree.map(lambda *values: jnp.stack(values), *states)


def state_from_arrays(agent_pos, goal_pos, wall_map):
    return State(
        agent_pos=jnp.asarray(agent_pos),
        goal_pos=jnp.asarray(goal_pos),
        wall_map=jnp.asarray(wall_map),
        time=jnp.asarray(0, dtype=jnp.int32),
        terminal=jnp.asarray(False),
    )


def bank_hash(agent_pos, goal_pos, wall_map):
    digest = hashlib.sha256()
    for array in (agent_pos, goal_pos, wall_map):
        digest.update(np.ascontiguousarray(array).tobytes())
    return digest.hexdigest()


def generate_state_bank(config):
    layout_names = list(config["layout_names"])
    env = ToyCoopNoPink(
        max_steps=int(config["ENV_KWARGS"]["max_steps"]),
        random_reset=True,
        check_held_out=False,
        partial_obs=bool(config["ENV_KWARGS"]["partial_obs"]),
        incentivize_strat=int(config["ENV_KWARGS"]["incentivize_strat"]),
        map_name="mixed",
        layout_names=layout_names,
    )
    procedural_states = [
        env.custom_reset_fn(jax.random.key(index), random_reset=True)
        for index in range(int(config["NUM_PROCEDURAL_TASKS"]))
    ]
    fixed_states = []
    for layout_name in layout_names:
        fixed_env = ToyCoopNoPink(
            max_steps=int(config["ENV_KWARGS"]["max_steps"]),
            random_reset=False,
            check_held_out=False,
            partial_obs=bool(config["ENV_KWARGS"]["partial_obs"]),
            incentivize_strat=int(config["ENV_KWARGS"]["incentivize_strat"]),
            map_name=layout_name,
        )
        fixed_states.append(
            fixed_env.custom_reset_fn(jax.random.key(0), random_reset=False)
        )
    all_states = procedural_states + fixed_states
    agent_pos = np.asarray(jax.device_get(jnp.stack([s.agent_pos for s in all_states])))
    goal_pos = np.asarray(jax.device_get(jnp.stack([s.goal_pos for s in all_states])))
    wall_map = np.asarray(jax.device_get(jnp.stack([s.wall_map for s in all_states])))

    fixed_wall_maps = {
        name: np.asarray(jax.device_get(fixed_states[index].wall_map))
        for index, name in enumerate(layout_names)
    }
    procedural_layouts = []
    for wall in wall_map[: len(procedural_states)]:
        matches = [
            name for name, fixed_wall in fixed_wall_maps.items()
            if np.array_equal(wall, fixed_wall)
        ]
        if len(matches) != 1:
            raise ValueError("A procedural state did not match exactly one wall layout")
        procedural_layouts.append(matches[0])

    return {
        "agent_pos": agent_pos,
        "goal_pos": goal_pos,
        "wall_map": wall_map,
        "procedural_layouts": np.asarray(procedural_layouts),
        "fixed_layouts": np.asarray(layout_names),
        "hash": bank_hash(agent_pos, goal_pos, wall_map),
    }


def load_or_create_state_bank(config):
    bank_path = resolve_path(config["HELDOUT_PATH"])
    bank_path.parent.mkdir(parents=True, exist_ok=True)
    expected = generate_state_bank(config)
    if bank_path.exists():
        with np.load(bank_path, allow_pickle=False) as stored:
            actual = {name: stored[name] for name in stored.files}
        actual_hash = bank_hash(
            actual["agent_pos"], actual["goal_pos"], actual["wall_map"]
        )
        if actual_hash != expected["hash"]:
            raise ValueError(
                f"Held-out bank mismatch at {bank_path}. "
                "Keep the training-time code, layout order, and JAX version aligned."
            )
        bank = actual
        bank["hash"] = actual_hash
    else:
        temporary_path = bank_path.with_suffix(".tmp.npz")
        np.savez_compressed(
            temporary_path,
            agent_pos=expected["agent_pos"],
            goal_pos=expected["goal_pos"],
            wall_map=expected["wall_map"],
            procedural_layouts=expected["procedural_layouts"],
            fixed_layouts=expected["fixed_layouts"],
        )
        os.replace(temporary_path, bank_path)
        bank = expected

    num_procedural = int(config["NUM_PROCEDURAL_TASKS"])
    layout_counts = {
        name: int(np.sum(bank["procedural_layouts"] == name))
        for name in config["layout_names"]
    }
    signatures = set()
    for index in range(num_procedural):
        goals = bank["goal_pos"][index]
        goal_order = np.lexsort((goals[:, 0], goals[:, 1]))
        signature = (
            bank["agent_pos"][index].tobytes(),
            goals[goal_order].tobytes(),
            bank["wall_map"][index].tobytes(),
        )
        signatures.add(signature)
    manifest = {
        "sha256": bank["hash"],
        "num_procedural": num_procedural,
        "num_fixed": len(config["layout_names"]),
        "num_unique_procedural": len(signatures),
        "layout_counts": layout_counts,
        "layout_names": list(config["layout_names"]),
        "generator_keys": [0, num_procedural - 1],
    }
    manifest_path = bank_path.with_suffix(".json")
    with open(manifest_path, "w", encoding="utf-8") as file:
        json.dump(manifest, file, indent=2)
    print(f"Held-out bank: {bank_path}")
    print(json.dumps(manifest, indent=2))
    return bank, bank_path, manifest_path, manifest


def make_network(config, spec):
    network_config = dict(config)
    network_config["layout_name"] = spec["train_map"]
    family = spec["family"]
    if family == "ippo":
        from modified_wall_ippo_general_dual_destination_with_xp import (
            ActorCriticRNN,
            ScannedRNN,
        )

        network = ActorCriticRNN(5, config=network_config)
        initialize_carry = lambda: ScannedRNN.initialize_carry(
            2, int(config["GRU_HIDDEN_DIM"])
        )
    elif family == "e3t":
        from modified_wall_e3t_dual_destination_with_xp import (
            ActorCriticRNN,
            ScannedRNN,
        )

        network = ActorCriticRNN(5, config=network_config)
        initialize_carry = lambda: ScannedRNN.initialize_carry(
            2, int(config["GRU_HIDDEN_DIM"])
        )
    elif family == "idaac":
        from modified_wall_idaac_general_gradient_with_xp import ActorCriticRNN

        network = ActorCriticRNN(5, config=network_config)
        initialize_carry = lambda: ActorCriticRNN.initialize_carry(
            2, int(config["GRU_HIDDEN_DIM"])
        )
    else:
        raise ValueError(f"Unknown network family: {family}")
    return network, initialize_carry


def make_pair_evaluator(config, spec):
    env = ToyCoopNoPink(
        max_steps=int(config["ENV_KWARGS"]["max_steps"]),
        random_reset=False,
        check_held_out=False,
        partial_obs=bool(config["ENV_KWARGS"]["partial_obs"]),
        incentivize_strat=int(config["ENV_KWARGS"]["incentivize_strat"]),
        map_name="mixed",
        layout_names=list(config["layout_names"]),
    )
    network, initialize_carry = make_network(config, spec)
    beta = float(config["BETA"])
    argmax = bool(config["ARGMAX"])
    num_steps = int(config["NUM_STEPS"])

    def rollout(params_1, params_2, initial_state, rng):
        obs = env.get_obs(initial_state)
        done = jnp.zeros((2,), dtype=bool)
        carry = (
            initial_state,
            obs,
            done,
            initialize_carry(),
            initialize_carry(),
            rng,
        )

        def step(carry, _):
            state, obs, done, hidden_1, hidden_2, rng = carry
            rng, action_rng_1, action_rng_2, step_rng = jax.random.split(rng, 4)
            obs_batch = jnp.stack([obs[agent].reshape(-1) for agent in env.agents])
            agent_positions = jnp.stack([state.agent_pos for _ in env.agents])
            ac_in = (
                obs_batch[jnp.newaxis, :],
                done[jnp.newaxis, :],
                agent_positions[jnp.newaxis, :],
            )
            output_1 = network.apply(params_1, hidden_1, ac_in)
            output_2 = network.apply(params_2, hidden_2, ac_in)
            hidden_1, pi_1 = output_1[0], output_1[1]
            hidden_2, pi_2 = output_2[0], output_2[1]
            pi_1 = distrax.Categorical(logits=pi_1.logits * beta)
            pi_2 = distrax.Categorical(logits=pi_2.logits * beta)
            sampled_1 = pi_1.sample(seed=action_rng_1)[0]
            sampled_2 = pi_2.sample(seed=action_rng_2)[0]
            greedy_1 = jnp.argmax(pi_1.probs, axis=-1)[0]
            greedy_2 = jnp.argmax(pi_2.probs, axis=-1)[0]
            actions_1 = jnp.where(argmax, greedy_1, sampled_1)
            actions_2 = jnp.where(argmax, greedy_2, sampled_2)
            actions = {
                env.agents[0]: actions_1[0],
                env.agents[1]: actions_2[1],
            }
            next_obs, next_state, reward, dones, _ = env.step_env(
                step_rng, state, actions
            )
            next_done = jnp.asarray([dones[agent] for agent in env.agents])
            next_carry = (
                next_state,
                next_obs,
                next_done,
                hidden_1,
                hidden_2,
                rng,
            )
            return next_carry, reward[env.agents[0]]

        _, rewards = jax.lax.scan(step, carry, None, num_steps)
        return rewards.sum()

    @jax.jit
    def evaluate_pair(params_1, params_2, initial_states, rngs):
        return jax.vmap(
            lambda state, rng: rollout(params_1, params_2, state, rng)
        )(initial_states, rngs)

    return evaluate_pair


def states_from_bank(bank, config, spec):
    num_procedural = int(config["NUM_PROCEDURAL_TASKS"])
    procedural = [
        state_from_arrays(
            bank["agent_pos"][index],
            bank["goal_pos"][index],
            bank["wall_map"][index],
        )
        for index in range(num_procedural)
    ]
    fixed_offset = num_procedural
    fixed_lookup = {
        name: state_from_arrays(
            bank["agent_pos"][fixed_offset + index],
            bank["goal_pos"][fixed_offset + index],
            bank["wall_map"][fixed_offset + index],
        )
        for index, name in enumerate(config["layout_names"])
    }
    fixed_names = list(spec["fixed_layouts"])
    fixed = [fixed_lookup[name] for name in fixed_names]
    return stack_states(procedural), stack_states(fixed), fixed_names


def evaluation_rngs(seed, count, offset=0):
    base = jax.random.PRNGKey(seed)
    indices = jnp.arange(count, dtype=jnp.uint32) + jnp.uint32(offset)
    return jax.vmap(lambda index: jax.random.fold_in(base, index))(indices)


def summarize_results(episodes):
    ordered_pairs = (
        episodes.groupby(["split", "policy_1", "policy_2"], as_index=False)
        .agg(
            reward_mean=("reward", "mean"),
            normalized_return_mean=("normalized_return", "mean"),
            success_rate=("success", "mean"),
            num_tasks=("state_id", "count"),
        )
    )
    ordered_pairs["seed_pair"] = ordered_pairs.apply(
        lambda row: f"{min(row.policy_1, row.policy_2)}x{max(row.policy_1, row.policy_2)}",
        axis=1,
    )
    seat_symmetric = (
        ordered_pairs.groupby(["split", "seed_pair"], as_index=False)
        .agg(
            reward_mean=("reward_mean", "mean"),
            normalized_return_mean=("normalized_return_mean", "mean"),
            success_rate=("success_rate", "mean"),
        )
    )
    summary_rows = []
    for split, split_frame in seat_symmetric.groupby("split"):
        summary_rows.append(
            {
                "split": split,
                "reward_mean": float(split_frame["reward_mean"].mean()),
                "reward_sem": float(split_frame["reward_mean"].sem()),
                "normalized_return_mean": float(
                    split_frame["normalized_return_mean"].mean()
                ),
                "normalized_return_sem": float(
                    split_frame["normalized_return_mean"].sem()
                ),
                "success_rate": float(split_frame["success_rate"].mean()),
                "num_seed_pairs": int(len(split_frame)),
            }
        )
    return ordered_pairs, seat_symmetric, pd.DataFrame(summary_rows)


def initialize_wandb(config, spec, model_group, checkpoint_paths, manifest):
    if config["WANDB_MODE"] == "disabled":
        return None
    import wandb

    if config["WANDB_MODE"] == "online":
        with open(resolve_path("private.yaml"), encoding="utf-8") as file:
            private_info = yaml.load(file, Loader=yaml.FullLoader)
        wandb.login(key=private_info["wandb_key"])
    wandb_config = dict(config)
    wandb_config["checkpoint_paths"] = {
        str(seed): path for seed, path in checkpoint_paths.items()
    }
    wandb_config["heldout_manifest"] = manifest
    group = config.get("WANDB_GROUP") or spec["wandb_group"]
    return wandb.init(
        entity=config["ENTITY"],
        project=config["PROJECT"],
        group=group,
        name=f"FINAL_XP_{model_group}",
        job_type="final_xp_eval",
        tags=["final_xp", "procedural_100", model_group],
        config=wandb_config,
        mode=config["WANDB_MODE"],
    )


@hydra.main(
    version_base=None,
    config_path="xp_config",
    config_name="modified_wall_procedural_xp",
)
def main(config):
    config = OmegaConf.to_container(config, resolve=True)
    model_group = str(config["MODEL_GROUP"])
    if model_group not in MODEL_SPECS:
        valid = ", ".join(MODEL_SPECS)
        raise ValueError(f"Unknown MODEL_GROUP={model_group!r}. Choose one of: {valid}")
    spec = MODEL_SPECS[model_group]
    seeds = [int(seed) for seed in config["SEEDS"]]
    if len(seeds) < 2:
        raise ValueError("Cross-play requires at least two seeds")

    bank, bank_path, manifest_path, manifest = load_or_create_state_bank(config)
    params, checkpoint_paths = load_checkpoints(config, model_group)
    run = initialize_wandb(
        config, spec, model_group, checkpoint_paths, manifest
    )
    evaluate_pair = make_pair_evaluator(config, spec)
    procedural_states, fixed_states, fixed_names = states_from_bank(
        bank, config, spec
    )
    procedural_rngs = evaluation_rngs(
        int(config["ACTION_SEED"]), len(bank["procedural_layouts"])
    )
    fixed_rngs = evaluation_rngs(
        int(config["ACTION_SEED"]), len(fixed_names), offset=10000
    )

    rows = []
    ordered_pairs = [(seed_1, seed_2) for seed_1 in seeds for seed_2 in seeds if seed_1 != seed_2]
    for seed_1, seed_2 in tqdm(ordered_pairs, desc=f"{model_group} procedural XP"):
        procedural_rewards = np.asarray(
            jax.device_get(
                evaluate_pair(
                    params[seed_1],
                    params[seed_2],
                    procedural_states,
                    procedural_rngs,
                )
            )
        )
        fixed_rewards = np.asarray(
            jax.device_get(
                evaluate_pair(
                    params[seed_1],
                    params[seed_2],
                    fixed_states,
                    fixed_rngs,
                )
            )
        )
        for state_id, reward in enumerate(procedural_rewards):
            rows.append(
                {
                    "model_group": model_group,
                    "algorithm": spec["algorithm"],
                    "train_map": spec["train_map"],
                    "split": "procedural",
                    "state_id": int(state_id),
                    "eval_layout": str(bank["procedural_layouts"][state_id]),
                    "policy_1": seed_1,
                    "policy_2": seed_2,
                    "reward": float(reward),
                    "normalized_return": float(reward) / (2.0 * int(config["NUM_STEPS"])),
                    "success": bool(reward > -int(config["NUM_STEPS"])),
                }
            )
        for state_id, (layout_name, reward) in enumerate(zip(fixed_names, fixed_rewards)):
            rows.append(
                {
                    "model_group": model_group,
                    "algorithm": spec["algorithm"],
                    "train_map": spec["train_map"],
                    "split": "fixed",
                    "state_id": int(state_id),
                    "eval_layout": layout_name,
                    "policy_1": seed_1,
                    "policy_2": seed_2,
                    "reward": float(reward),
                    "normalized_return": float(reward) / (2.0 * int(config["NUM_STEPS"])),
                    "success": bool(reward > -int(config["NUM_STEPS"])),
                }
            )

    episodes = pd.DataFrame(rows)
    ordered_summary, pair_summary, summary = summarize_results(episodes)
    for frame in (ordered_summary, pair_summary, summary):
        frame.insert(0, "model_group", model_group)
        frame.insert(1, "algorithm", spec["algorithm"])
        frame.insert(2, "train_map", spec["train_map"])

    results_dir = resolve_path(config["RESULTS_DIR"])
    results_dir.mkdir(parents=True, exist_ok=True)
    episodes_path = results_dir / f"{model_group}_episodes.csv"
    ordered_path = results_dir / f"{model_group}_ordered_pairs.csv"
    pairs_path = results_dir / f"{model_group}_pairs.csv"
    summary_path = results_dir / f"{model_group}_summary.csv"
    episodes.to_csv(episodes_path, index=False)
    ordered_summary.to_csv(ordered_path, index=False)
    pair_summary.to_csv(pairs_path, index=False)
    summary.to_csv(summary_path, index=False)
    print(summary.to_string(index=False))
    print(f"Saved results under {results_dir}")

    if run is not None:
        import wandb

        log = {
            "final_xp/episodes": wandb.Table(dataframe=episodes),
            "final_xp/seed_pairs": wandb.Table(dataframe=pair_summary),
            "final_xp/summary": wandb.Table(dataframe=summary),
        }
        for row in summary.to_dict("records"):
            split = row["split"]
            log[f"final_xp/{split}_return"] = row["reward_mean"]
            log[f"final_xp/{split}_return_sem"] = row["reward_sem"]
            log[f"final_xp/{split}_normalized_return"] = row[
                "normalized_return_mean"
            ]
            log[f"final_xp/{split}_normalized_return_sem"] = row[
                "normalized_return_sem"
            ]
            log[f"final_xp/{split}_success_rate"] = row["success_rate"]
        run.log(log)
        artifact = wandb.Artifact(
            f"modified-wall-procedural-xp-{model_group}", type="evaluation"
        )
        for path in (
            bank_path,
            manifest_path,
            episodes_path,
            ordered_path,
            pairs_path,
            summary_path,
        ):
            artifact.add_file(str(path))
        run.log_artifact(artifact)
        run.finish()

    jax.effects_barrier()
    jax.clear_caches()


if __name__ == "__main__":
    main()
