import glob
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

from ippo_general_dual_destination import ActorCriticRNN, ScannedRNN, initialize_environment
from jaxmarl.wrappers.baselines import LogWrapper
from jaxmarl.viz.toy_coop_jitted_visualizer import render_fn


def latest_match(patterns):
    matches = []
    for pattern in patterns:
        matches.extend(glob.glob(pattern, recursive=True))
    if not matches:
        return None
    return sorted(matches, key=lambda p: os.path.getmtime(p))[-1]


def model_patterns(root, model_name, seed, stage=None):
    if model_name == "IPPO":
        return [
            f"{root}/ikFalse/reset_all/ippo/**/seed{seed}_final.pkl",
        ]
    if model_name == "IPPO_POP":
        if stage is None:
            raise ValueError("IPPO_POP requires a checkpoint stage")
        return [
            f"{root}/ikFalse/reset_all/ippo/**/seed{seed}_{stage}.pkl",
        ]
    if model_name == "CEC":
        return [
            f"{root}/ikTrue/reset_all/cec/**/seed{seed}_ckpt0_improved_updates*.pkl",
        ]
    if model_name == "FCP":
        return [
            f"{root}/ikFalse/reset_all/fcp/**/fcp_seed{seed}_best.pkl",
        ]
    raise ValueError(f"Unknown model_name: {model_name}")


def load_params(root, model_name, seeds, stages=None):
    params = []
    labels = []
    paths = []
    specs = []
    if model_name == "IPPO_POP":
        stages = stages or ["init", "mid", "final"]
        specs = [(seed, stage, f"{seed}_{stage}") for seed in seeds for stage in stages]
    else:
        specs = [(seed, None, str(seed)) for seed in seeds]

    for seed, stage, label in specs:
        path = latest_match(model_patterns(root, model_name, seed, stage))
        if path is None:
            print(f"Missing {model_name} {label}")
            continue
        with open(path, "rb") as f:
            ckpt = pickle.load(f)
        params.append(ckpt["params"])
        labels.append(label)
        paths.append(path)
        print(f"Loaded {model_name} {label}: {path}")
    if not params:
        raise FileNotFoundError(f"No {model_name} checkpoints found under {root}")
    return jax.tree_map(lambda *x: jnp.stack(x), *params), labels, paths


def resolve_path(path):
    path = Path(path)
    if path.is_absolute():
        return str(path)
    return str(Path(get_original_cwd()) / path)


def safe_name(name):
    return "".join(c if c.isalnum() or c in ("-", "_", ".") else "_" for c in str(name))


ACTION_NAMES = {
    0: "right",
    1: "down",
    2: "left",
    3: "up",
    4: "stay",
}


def goal_name(pos, goals):
    if np.array_equal(pos, goals[0]):
        return "goal_0"
    if np.array_equal(pos, goals[1]):
        return "goal_1"
    return "none"


def goal_orientation(goals, goal_idx):
    other_idx = 1 - goal_idx
    if goals[goal_idx][1] < goals[other_idx][1]:
        return "top"
    if goals[goal_idx][1] > goals[other_idx][1]:
        return "bottom"
    if goals[goal_idx][0] < goals[other_idx][0]:
        return "left"
    if goals[goal_idx][0] > goals[other_idx][0]:
        return "right"
    return f"goal_{goal_idx}"


def summarize_debug_rollout(states_np):
    final_agent_pos = np.asarray(states_np.agent_pos)[-1]
    goals = np.asarray(states_np.goal_pos)[-1]
    agent_0_goal = goal_name(final_agent_pos[0], goals)
    agent_1_goal = goal_name(final_agent_pos[1], goals)

    def orient(goal):
        if goal == "goal_0":
            return goal_orientation(goals, 0)
        if goal == "goal_1":
            return goal_orientation(goals, 1)
        return "none"

    both_on_distinct_goals = (
        agent_0_goal != "none"
        and agent_1_goal != "none"
        and agent_0_goal != agent_1_goal
    )
    same_goal = (
        agent_0_goal != "none"
        and agent_1_goal != "none"
        and agent_0_goal == agent_1_goal
    )
    return {
        "agent_0_final_x": int(final_agent_pos[0][0]),
        "agent_0_final_y": int(final_agent_pos[0][1]),
        "agent_1_final_x": int(final_agent_pos[1][0]),
        "agent_1_final_y": int(final_agent_pos[1][1]),
        "goal_0_x": int(goals[0][0]),
        "goal_0_y": int(goals[0][1]),
        "goal_1_x": int(goals[1][0]),
        "goal_1_y": int(goals[1][1]),
        "agent_0_final_goal": agent_0_goal,
        "agent_0_final_goal_orientation": orient(agent_0_goal),
        "agent_1_final_goal": agent_1_goal,
        "agent_1_final_goal_orientation": orient(agent_1_goal),
        "both_on_distinct_goals": bool(both_on_distinct_goals),
        "same_goal": bool(same_goal),
    }


def get_rollouts(param_1, param_2, config, env, network, seed=0):
    def _step(carry, _):
        params_1, params_2, env_state, last_obs, last_done, hstate_1, hstate_2, rng = carry

        rng, action_rng_1, action_rng_2 = jax.random.split(rng, 3)
        obs_batch = jnp.stack([last_obs[a].flatten() for a in env.agents])
        agent_positions = jnp.stack([env_state.env_state.agent_pos for _ in env.agents])
        ac_in = (
            obs_batch[jnp.newaxis, :],
            last_done[jnp.newaxis, :],
            agent_positions[jnp.newaxis, :],
        )

        hstate_1, pi_1, _ = network.apply(params_1, hstate_1, ac_in)
        hstate_2, pi_2, _ = network.apply(params_2, hstate_2, ac_in)
        pi_1 = distrax.Categorical(logits=pi_1.logits * config["beta"])
        pi_2 = distrax.Categorical(logits=pi_2.logits * config["beta"])

        action_1 = pi_1.sample(seed=action_rng_1)[0]
        action_2 = pi_2.sample(seed=action_rng_2)[0]
        action_1 = jnp.where(config["argmax"], jnp.argmax(pi_1.probs, axis=-1)[0], action_1)
        action_2 = jnp.where(config["argmax"], jnp.argmax(pi_2.probs, axis=-1)[0], action_2)
        env_act = {env.agents[0]: action_1[0], env.agents[1]: action_2[1]}

        rng, step_rng = jax.random.split(rng)
        obs, env_state, reward, done, _info = env.step(step_rng, env_state, env_act)
        done_batch = jnp.array([done[a] for a in env.agents])
        carry = (params_1, params_2, env_state, obs, done_batch, hstate_1, hstate_2, rng)
        return carry, reward["agent_0"]

    def _rollout(rng):
        rng, reset_rng = jax.random.split(rng)
        obs, env_state = env.reset(reset_rng)
        hstate_1 = ScannedRNN.initialize_carry(env.num_agents, config["hidden_dim"])
        hstate_2 = ScannedRNN.initialize_carry(env.num_agents, config["hidden_dim"])
        done = jnp.zeros(env.num_agents, dtype=bool)
        carry = (param_1, param_2, env_state, obs, done, hstate_1, hstate_2, rng)
        _, rewards = jax.lax.scan(_step, carry, None, config["num_steps"])
        return rewards.sum()

    rollout_rngs = jax.random.split(jax.random.PRNGKey(seed), config["num_trajs"])
    return jax.vmap(_rollout)(rollout_rngs)


def get_debug_rollout(param_1, param_2, config, env, network, seed=0):
    def _step(carry, _):
        params_1, params_2, env_state, last_obs, last_done, hstate_1, hstate_2, rng = carry

        rng, action_rng_1, action_rng_2 = jax.random.split(rng, 3)
        obs_batch = jnp.stack([last_obs[a].flatten() for a in env.agents])
        agent_positions = jnp.stack([env_state.env_state.agent_pos for _ in env.agents])
        ac_in = (
            obs_batch[jnp.newaxis, :],
            last_done[jnp.newaxis, :],
            agent_positions[jnp.newaxis, :],
        )

        hstate_1, pi_1, _ = network.apply(params_1, hstate_1, ac_in)
        hstate_2, pi_2, _ = network.apply(params_2, hstate_2, ac_in)
        pi_1 = distrax.Categorical(logits=pi_1.logits * config["beta"])
        pi_2 = distrax.Categorical(logits=pi_2.logits * config["beta"])

        action_1 = pi_1.sample(seed=action_rng_1)[0]
        action_2 = pi_2.sample(seed=action_rng_2)[0]
        action_1 = jnp.where(config["argmax"], jnp.argmax(pi_1.probs, axis=-1)[0], action_1)
        action_2 = jnp.where(config["argmax"], jnp.argmax(pi_2.probs, axis=-1)[0], action_2)
        chosen_actions = jnp.array([action_1[0], action_2[1]])
        env_act = {env.agents[0]: chosen_actions[0], env.agents[1]: chosen_actions[1]}

        rng, step_rng = jax.random.split(rng)
        obs, env_state, reward, done, _info = env.step(step_rng, env_state, env_act)
        done_batch = jnp.array([done[a] for a in env.agents])
        carry = (params_1, params_2, env_state, obs, done_batch, hstate_1, hstate_2, rng)
        debug_step = (env_state.env_state, reward["agent_0"], chosen_actions)
        return carry, debug_step

    rng = jax.random.PRNGKey(seed)
    rng, reset_rng = jax.random.split(rng)
    obs, env_state = env.reset(reset_rng)
    initial_state = env_state.env_state
    hstate_1 = ScannedRNN.initialize_carry(env.num_agents, config["hidden_dim"])
    hstate_2 = ScannedRNN.initialize_carry(env.num_agents, config["hidden_dim"])
    done = jnp.zeros(env.num_agents, dtype=bool)
    carry = (param_1, param_2, env_state, obs, done, hstate_1, hstate_2, rng)
    _, (states, rewards, actions) = jax.lax.scan(_step, carry, None, config["num_steps"])
    return initial_state, states, rewards, actions


def save_debug_gif(path, initial_state, states, fps):
    import imageio.v2 as imageio

    frames = [np.asarray(jax.device_get(render_fn(initial_state)))]
    step_frames = jax.device_get(jax.vmap(render_fn)(states))
    frames.extend([np.asarray(frame) for frame in step_frames])
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    imageio.mimsave(path, frames, duration=1.0 / fps)


@hydra.main(version_base=None, config_path="xp_config", config_name="dual_xp")
def main(config):
    config = OmegaConf.to_container(config, resolve=True)
    model_name = config["model_name"]
    partner_model_name = config.get("partner_model_name") or model_name
    split = "procedural" if config["ENV_KWARGS"]["random_reset"] else "fixed"
    run_name = f"XP_{model_name}_{split}" if model_name == partner_model_name else f"XP_{model_name}_x_{partner_model_name}_{split}"
    wandb_run = None
    if config["WANDB_MODE"] != "disabled":
        import wandb

        if config["WANDB_MODE"] == "online":
            with open(resolve_path("private.yaml")) as f:
                private_info = yaml.load(f, Loader=yaml.FullLoader)
            wandb.login(key=private_info["wandb_key"])

        wandb_run = wandb.init(
            entity=config["ENTITY"],
            project=config["PROJECT"],
            mode=config["WANDB_MODE"],
            name=run_name,
            config=config,
        )

    seeds = [int(s) for s in config["SEEDS"]]
    partner_seeds = [int(s) for s in config.get("PARTNER_SEEDS", config["SEEDS"])]
    ippo_pop_stages = list(config.get("IPPO_POP_STAGES", ["init", "mid", "final"]))
    model_root = resolve_path(config["MODEL_ROOT"])
    param_stack_1, label_list_1, path_list_1 = load_params(model_root, model_name, seeds, ippo_pop_stages)
    if model_name == partner_model_name and seeds == partner_seeds:
        param_stack_2, label_list_2, path_list_2 = param_stack_1, label_list_1, path_list_1
    else:
        param_stack_2, label_list_2, path_list_2 = load_params(
            model_root,
            partner_model_name,
            partner_seeds,
            ippo_pop_stages,
        )

    env = initialize_environment(config)
    env = LogWrapper(env, env_params={"random_reset_fn": config["ENV_KWARGS"]["random_reset_fn"]})
    network = ActorCriticRNN(env.action_space(env.agents[0]).n, config=config)

    eval_config = {
        "num_trajs": config["TEST_KWARGS"]["num_trajs"],
        "num_steps": config["NUM_STEPS"],
        "hidden_dim": config["GRU_HIDDEN_DIM"],
        "argmax": config["TEST_KWARGS"]["argmax"],
        "beta": config["TEST_KWARGS"]["beta"],
    }
    debug_cfg = config.get("DEBUG_GIFS", {})
    debug_enabled = bool(debug_cfg.get("enabled", False))
    debug_max_pairs = int(debug_cfg.get("max_pairs", 4))
    debug_pairs = set(str(pair) for pair in debug_cfg.get("pairs", []))
    debug_only_cross_play = bool(debug_cfg.get("only_cross_play", True))
    debug_seed = int(debug_cfg.get("traj_seed", 0))
    debug_fps = int(debug_cfg.get("fps", 4))
    debug_upload_wandb = bool(debug_cfg.get("upload_wandb", True))
    debug_out_dir = resolve_path(debug_cfg.get("out_dir", f"{config['SAVE_PATH']}/debug_gifs"))
    debug_records = []

    rows = []
    pair_rows = []
    n_1 = len(label_list_1)
    n_2 = len(label_list_2)
    pair_idx = 0
    for i in tqdm(range(n_1), desc=f"{model_name} x {partner_model_name} XP"):
        for j in range(n_2):
            label_1 = label_list_1[i]
            label_2 = label_list_2[j]
            path_1 = path_list_1[i]
            path_2 = path_list_2[j]
            is_self_pair = model_name == partner_model_name and label_1 == label_2
            pair_name = f"{label_1}x{label_2}"
            param_1 = jax.tree_map(lambda x: x[i], param_stack_1)
            param_2 = jax.tree_map(lambda x: x[j], param_stack_2)
            rewards = get_rollouts(
                param_1,
                param_2,
                eval_config,
                env,
                network,
                seed=int(i * 1000 + j),
            )
            rewards_np = jax.device_get(rewards)
            normalized_rewards_np = rewards_np / (2.0 * config["ENV_KWARGS"]["max_steps"])
            success_np = rewards_np > -config["ENV_KWARGS"]["max_steps"]
            pair_mean = float(rewards_np.mean())
            pair_std = float(rewards_np.std())
            pair_normalized_mean = float(normalized_rewards_np.mean())
            pair_success_rate = float(success_np.mean())
            pair_row = {
                "model": model_name,
                "partner_model": partner_model_name,
                "policy_1": label_1,
                "policy_2": label_2,
                "checkpoint_1": path_1,
                "checkpoint_2": path_2,
                "reward_mean": pair_mean,
                "reward_std": pair_std,
                "normalized_reward_mean": pair_normalized_mean,
                "success_rate": pair_success_rate,
                "is_self_pair": bool(is_self_pair),
                "random_reset": bool(config["ENV_KWARGS"]["random_reset"]),
            }
            pair_rows.append(pair_row)
            if wandb_run is not None:
                wandb_run.log(
                    {
                        "pair_idx": pair_idx,
                        "pair/reward_mean": pair_mean,
                        "pair/reward_std": pair_std,
                        "pair/normalized_reward_mean": pair_normalized_mean,
                        "pair/success_rate": pair_success_rate,
                        "pair/is_self_pair": int(is_self_pair),
                        f"pair_reward/{pair_name}": pair_mean,
                        f"pair_normalized_reward/{pair_name}": pair_normalized_mean,
                        f"pair_success_rate/{pair_name}": pair_success_rate,
                    }
                )

            debug_pair_requested = not debug_pairs or pair_name in debug_pairs
            debug_pair_allowed = not debug_only_cross_play or not is_self_pair
            debug_pair_budget = len(debug_records) < debug_max_pairs
            if debug_enabled and debug_pair_requested and debug_pair_allowed and debug_pair_budget:
                debug_rollout_seed = int(debug_seed + i * 1000 + j)
                initial_state, states, debug_rewards, debug_actions = get_debug_rollout(
                    param_1,
                    param_2,
                    eval_config,
                    env,
                    network,
                    seed=debug_rollout_seed,
                )
                debug_prefix = safe_name(f"{run_name}_{pair_name}_{split}")
                gif_path = f"{debug_out_dir}/{debug_prefix}.gif"
                actions_path = f"{debug_out_dir}/{debug_prefix}_actions.csv"
                save_debug_gif(gif_path, initial_state, states, debug_fps)

                debug_rewards_np = np.asarray(jax.device_get(debug_rewards))
                debug_actions_np = np.asarray(jax.device_get(debug_actions))
                states_np = jax.device_get(states)
                agent_pos_np = np.asarray(states_np.agent_pos)
                goal_pos_np = np.asarray(states_np.goal_pos)
                debug_summary = summarize_debug_rollout(states_np)
                action_df = pd.DataFrame(
                    {
                        "step": np.arange(len(debug_rewards_np)),
                        "action_agent_0": debug_actions_np[:, 0],
                        "action_agent_0_name": [ACTION_NAMES[int(a)] for a in debug_actions_np[:, 0]],
                        "action_agent_1": debug_actions_np[:, 1],
                        "action_agent_1_name": [ACTION_NAMES[int(a)] for a in debug_actions_np[:, 1]],
                        "agent_0_x": agent_pos_np[:, 0, 0],
                        "agent_0_y": agent_pos_np[:, 0, 1],
                        "agent_1_x": agent_pos_np[:, 1, 0],
                        "agent_1_y": agent_pos_np[:, 1, 1],
                        "goal_0_x": goal_pos_np[:, 0, 0],
                        "goal_0_y": goal_pos_np[:, 0, 1],
                        "goal_1_x": goal_pos_np[:, 1, 0],
                        "goal_1_y": goal_pos_np[:, 1, 1],
                        "reward": debug_rewards_np,
                        "cumulative_reward": np.cumsum(debug_rewards_np),
                    }
                )
                action_df.to_csv(actions_path, index=False)
                debug_record = {
                    "pair_idx": pair_idx,
                    "pair": pair_name,
                    "model": model_name,
                    "partner_model": partner_model_name,
                    "policy_1": label_1,
                    "policy_2": label_2,
                    "reward": float(debug_rewards_np.sum()),
                    "gif_path": gif_path,
                    "actions_path": actions_path,
                    **debug_summary,
                }
                debug_records.append(debug_record)
                print(f"Saved debug GIF for {pair_name}: {gif_path}")
                if wandb_run is not None and debug_upload_wandb:
                    import wandb

                    wandb_run.log(
                        {
                            f"debug_gifs/{debug_prefix}": wandb.Video(gif_path, fps=debug_fps, format="gif"),
                            f"debug_rollout_reward/{pair_name}": float(debug_rewards_np.sum()),
                        }
                    )
            pair_idx += 1

            for reward in rewards_np:
                normalized_reward = float(reward / (2.0 * config["ENV_KWARGS"]["max_steps"]))
                rows.append(
                    {
                        "model": model_name,
                        "partner_model": partner_model_name,
                        "policy_1": label_1,
                        "policy_2": label_2,
                        "reward": float(reward),
                        "normalized_reward": normalized_reward,
                        "success": bool(reward > -config["ENV_KWARGS"]["max_steps"]),
                        "is_self_pair": bool(is_self_pair),
                        "random_reset": bool(config["ENV_KWARGS"]["random_reset"]),
                    }
                )

    out = config.get("OUTPUT_FILE")
    if out is None:
        out_prefix = model_name if model_name == partner_model_name else f"{model_name}_x_{partner_model_name}"
        out = f"{config['SAVE_PATH']}/{out_prefix}_{split}_XP_results.csv"
    out = resolve_path(out)
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    pair_df = pd.DataFrame(pair_rows)
    df.to_csv(out, index=False)
    pair_out = out.replace(".csv", "_pair_summary.csv")
    pair_df.to_csv(pair_out, index=False)
    debug_df = pd.DataFrame(debug_records)
    debug_out = out.replace(".csv", "_debug_gifs.csv")
    if len(debug_df):
        debug_df.to_csv(debug_out, index=False)
    print(f"Saved {len(rows)} rows to {out}")
    print(f"Saved {len(pair_rows)} pair summaries to {pair_out}")
    if len(debug_df):
        print(f"Saved {len(debug_df)} debug GIF records to {debug_out}")

    if wandb_run is not None:
        import wandb

        self_df = df[df["is_self_pair"]]
        xp_df = df[~df["is_self_pair"]]
        self_pair_df = pair_df[pair_df["is_self_pair"]]
        xp_pair_df = pair_df[~pair_df["is_self_pair"]]
        summary_log = {
            "xp/reward_mean": float(df["reward"].mean()),
            "xp/reward_std": float(df["reward"].std()),
            "xp/reward_min": float(df["reward"].min()),
            "xp/reward_max": float(df["reward"].max()),
            "xp/normalized_reward_mean": float(df["normalized_reward"].mean()),
            "xp/normalized_reward_std": float(df["normalized_reward"].std()),
            "xp/success_rate": float(df["success"].mean()),
            "xp/cross_play_reward_mean": float(xp_df["reward"].mean()) if len(xp_df) else float("nan"),
            "xp/cross_play_reward_std": float(xp_df["reward"].std()) if len(xp_df) else float("nan"),
            "xp/cross_play_normalized_reward_mean": float(xp_df["normalized_reward"].mean()) if len(xp_df) else float("nan"),
            "xp/cross_play_success_rate": float(xp_df["success"].mean()) if len(xp_df) else float("nan"),
            "xp/self_play_reward_mean": float(self_df["reward"].mean()) if len(self_df) else float("nan"),
            "xp/self_play_reward_std": float(self_df["reward"].std()) if len(self_df) else float("nan"),
            "xp/self_play_normalized_reward_mean": float(self_df["normalized_reward"].mean()) if len(self_df) else float("nan"),
            "xp/self_play_success_rate": float(self_df["success"].mean()) if len(self_df) else float("nan"),
            "xp/cross_play_pair_mean": float(xp_pair_df["reward_mean"].mean()) if len(xp_pair_df) else float("nan"),
            "xp/self_play_pair_mean": float(self_pair_df["reward_mean"].mean()) if len(self_pair_df) else float("nan"),
            "xp/cross_play_pair_normalized_mean": float(xp_pair_df["normalized_reward_mean"].mean()) if len(xp_pair_df) else float("nan"),
            "xp/self_play_pair_normalized_mean": float(self_pair_df["normalized_reward_mean"].mean()) if len(self_pair_df) else float("nan"),
            "xp/num_pairs": int(len(pair_df)),
            "xp/num_cross_play_pairs": int(len(xp_pair_df)),
            "xp/num_self_play_pairs": int(len(self_pair_df)),
            "xp/num_model_1_policies": int(n_1),
            "xp/num_model_2_policies": int(n_2),
            "xp/num_rollouts": int(len(df)),
            "xp/reward_histogram": wandb.Histogram(df["reward"].to_numpy()),
            "xp/pair_summary": wandb.Table(dataframe=pair_df),
            "xp/rollouts": wandb.Table(dataframe=df),
        }
        if len(debug_df):
            summary_log["xp/debug_gifs"] = wandb.Table(dataframe=debug_df)
        wandb_run.log(summary_log)
        wandb_run.finish()


if __name__ == "__main__":
    main()


# Examples:
# python3 baselines/CEC_UED/dual_xp_test.py model_name=IPPO
# python3 baselines/CEC_UED/dual_xp_test.py model_name=FCP
# python3 baselines/CEC_UED/dual_xp_test.py model_name=CEC
#
# python3 baselines/CEC_UED/dual_xp_test.py model_name=IPPO ENV_KWARGS.random_reset=true ENV_KWARGS.check_held_out=true
# python3 baselines/CEC_UED/dual_xp_test.py model_name=FCP ENV_KWARGS.random_reset=true ENV_KWARGS.check_held_out=true
# python3 baselines/CEC_UED/dual_xp_test.py model_name=CEC ENV_KWARGS.random_reset=true ENV_KWARGS.check_held_out=true
#
# FCP partner-population diagnostics:
# python3 baselines/CEC_UED/dual_xp_test.py model_name=FCP partner_model_name=IPPO_POP
# python3 baselines/CEC_UED/dual_xp_test.py model_name=IPPO_POP partner_model_name=FCP
