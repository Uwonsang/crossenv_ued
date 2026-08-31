"""
Based on PureJaxRL Implementation of PPO.

Note, this file will only work for MPE environments with homogenous agents (e.g. Simple Spread).

"""
import os
import glob
import pickle
import jax
import jax.numpy as jnp
import flax.linen as nn
import numpy as np
import optax
from flax.linen.initializers import constant, orthogonal
from typing import Sequence, NamedTuple, Any, Dict
from flax.training.train_state import TrainState
import distrax
import hydra
from omegaconf import OmegaConf

import jaxmarl
from jaxmarl.wrappers.baselines import LogWrapper
from jaxmarl.environments.toy_coop.toy_coop_no_pink import ToyCoopNoPink
from jaxmarl.environments.overcooked import overcooked_layouts
from jaxmarl.environments.overcooked.layouts import make_counter_circuit_9x9, make_forced_coord_9x9, make_coord_ring_9x9, make_asymm_advantages_9x9, make_cramped_room_9x9

import wandb
import functools
import pdb
from jax_tqdm import scan_tqdm
import yaml
import time
from baselines.CEC_UED.algo_utils import (
    BCPolicy,
    EVAL_LAYOUTS_9,
    load_human_proxy_params,
    make_eval_envs_overcooked,
)
from baselines.CEC.evaluation_rollout_utils import (
    evaluate_cross_play_layout,
    evaluate_self_play_layout,
)

TOY_LAYOUT_NAMES = ["empty", "wall_a"]


def get_wall_map_name(config):
    return config.get("map_name", config["ENV_KWARGS"].get("map_name", "empty"))


def get_toy_layout_names(config):
    return list(config.get("layout_names", TOY_LAYOUT_NAMES))


def get_wall_map_dir_name(config):
    map_name = get_wall_map_name(config)
    ckpt_tag = str(config.get("CKPT_TAG", "")).strip()
    return f"{map_name}_{ckpt_tag}" if ckpt_tag else map_name


def make_modified_wall_env(config, map_name=None, evaluation=False):
    allowed = {
        "max_steps",
        "random_reset",
        "debug",
        "check_held_out",
        "partial_obs",
        "incentivize_strat",
    }
    env_kwargs = {
        key: value
        for key, value in config["ENV_KWARGS"].items()
        if key in allowed
    }
    env_kwargs["map_name"] = map_name or get_wall_map_name(config)
    if evaluation:
        env_kwargs["random_reset"] = False
        env_kwargs["check_held_out"] = False
    config["ENV_KWARGS"]["map_name"] = get_wall_map_name(config)
    return ToyCoopNoPink(**env_kwargs)


def make_modified_wall_eval_envs(config):
    return {
        name: LogWrapper(
            make_modified_wall_env(config, map_name=name, evaluation=True),
            env_params={
                "random_reset_fn": config["EVAL_KWARGS"]["random_reset_fn"]
            },
        )
        for name in get_toy_layout_names(config)
    }


def initialize_environment(config):
    map_name = get_wall_map_name(config)
    config["layout_name"] = map_name
    env = make_modified_wall_env(config)
    config["obs_dim"] = env.observation_space(env.agents[0]).shape
    return env


def load_xp_partner_params(config):
    partner_seed = int(config.get("XP_KWARGS", {}).get("partner_seed", 98))
    root = (
        f"ckpts/e3t/{config['ENV_NAME']}/modified_wall/{get_wall_map_dir_name(config)}"
        f"/ikFalse/{config['ENV_KWARGS']['random_reset_fn']}/e3t"
    )
    matches = glob.glob(
        f"{root}/**/seed{partner_seed}_ckpt0_e3t_updates*.pkl",
        recursive=True,
    )
    if not matches:
        raise FileNotFoundError(
            f"Missing E3T XP partner seed{partner_seed} under {root}"
        )
    path = max(matches, key=os.path.getmtime)
    with open(path, "rb") as f:
        checkpoint = pickle.load(f)
    print(f"Loaded E3T XP partner seed{partner_seed}: {path}")
    return checkpoint["params"], path


class ScannedRNN(nn.Module):
    @functools.partial(
        nn.scan,
        variable_broadcast="params",
        in_axes=0,
        out_axes=0,
        split_rngs={"params": False},
    )
    @nn.compact
    def __call__(self, carry, x):
        """Applies the module."""
        lstm_state = carry
        ins, resets = x
        
        # Reset LSTM state on episode boundaries
        lstm_state = jax.tree.map(
            lambda x: jnp.where(resets[:, np.newaxis], jnp.zeros_like(x), x),
            lstm_state
        )
        
        new_lstm_state, y = nn.OptimizedLSTMCell(features=ins.shape[-1])(lstm_state, ins)
        return new_lstm_state, y

    @staticmethod
    def initialize_carry(batch_size, hidden_size):
        return nn.OptimizedLSTMCell(features=hidden_size).initialize_carry(
            jax.random.PRNGKey(0), (batch_size, hidden_size)
        )


class ActorCriticRNN(nn.Module):
    action_dim: Sequence[int]
    config: Dict

    @nn.compact
    def __call__(self, hidden, x):
        obs, dones, agent_positions = x
        if self.config["CONV_NET"]:
            batch_size, num_envs, flattened_obs_dim = obs.shape
            if self.config["ENV_NAME"] == "overcooked":
                reshaped_obs = obs.reshape(-1, 9,9,26)
            else:
                reshaped_obs = obs.reshape(-1, 5,5,4)

            embedding = nn.Conv(
                features=64 if "9" in self.config['layout_name'] else 2 * self.config["FC_DIM_SIZE"],
                kernel_size=(2, 2),
                kernel_init=orthogonal(np.sqrt(2)),
                bias_init=constant(0.0),
            )(reshaped_obs)
            embedding = nn.relu(embedding)
            embedding = nn.Conv(
                features=32 if "9" in self.config['layout_name'] else self.config["FC_DIM_SIZE"],
                kernel_size=(2, 2),
                kernel_init=orthogonal(np.sqrt(2)),
                bias_init=constant(0.0),
            )(embedding)
            embedding = nn.relu(embedding)

            embedding = embedding.reshape((batch_size, num_envs, -1))
        else:
            embedding = obs

        embedding = nn.Dense(
            self.config["FC_DIM_SIZE"] * 2, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)
        )(embedding)
        embedding = nn.relu(embedding)
        # embedding = nn.Dense(
        #     self.config["FC_DIM_SIZE"], kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)
        # )(embedding)
        # embedding = nn.relu(embedding)
        embedding = nn.Dense(
            self.config["FC_DIM_SIZE"] * 2 if "9" in self.config['layout_name'] else self.config["FC_DIM_SIZE"], kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)
        )(embedding)
        embedding = nn.relu(embedding)

        rnn_in = (embedding, dones)
        hidden, embedding = ScannedRNN()(hidden, rnn_in)

        #########
        # Model of other agent (patner_prediction module-> 7,8,9 index in original e3t paper)
        #########
        prediction_other = nn.Dense(64, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(embedding)
        prediction_other = nn.leaky_relu(prediction_other)
        prediction_other = nn.Dense(64, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(prediction_other)
        prediction_other = nn.leaky_relu(prediction_other)
        prediction_other = nn.Dense(64, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(prediction_other)
        prediction_other = nn.leaky_relu(prediction_other)
        prediction_other = nn.Dense(64, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(prediction_other)
        prediction_other = nn.tanh(prediction_other)
        prediction_other = nn.Dense(self.action_dim, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(prediction_other)
        prediction_other = prediction_other / jnp.sqrt(jnp.sum(prediction_other**2, axis=-1, keepdims=True) + 1e-10)  # L2 normalization
        other_pi = distrax.Categorical(logits=prediction_other)

        #########
        # Actor
        #########
        actor_embedding = jnp.concatenate([embedding, prediction_other], axis=-1)
        actor_mean = nn.Dense(self.config["GRU_HIDDEN_DIM"] , kernel_init=orthogonal(2), bias_init=constant(0.0))(
            actor_embedding
        )
        actor_mean = nn.relu(actor_mean)
        actor_mean = nn.Dense(self.config["GRU_HIDDEN_DIM"] * 3 // 4, kernel_init=orthogonal(2), bias_init=constant(0.0))(
            actor_mean
        )
        actor_mean = nn.relu(actor_mean)
        actor_mean = nn.Dense(
            self.config["GRU_HIDDEN_DIM"] // 2, kernel_init=orthogonal(2), bias_init=constant(0.0)
        )(actor_mean)
        actor_mean = nn.relu(actor_mean)
        if self.config["ENV_NAME"] == "overcooked":
            actor_mean = nn.Dense(self.config["GRU_HIDDEN_DIM"] // 4, kernel_init=orthogonal(2), bias_init=constant(0.0))(
                actor_mean
            )
            actor_mean = nn.relu(actor_mean)  # extra layer 1

        actor_mean = nn.Dense(
            self.action_dim, kernel_init=orthogonal(0.01), bias_init=constant(0.0)
        )(actor_mean)        

        pi = distrax.Categorical(logits=actor_mean)

        #########
        # Critic
        #########
        critic = nn.Dense(self.config["FC_DIM_SIZE"]*2, kernel_init=orthogonal(2), bias_init=constant(0.0))(
            embedding
        )
        critic = nn.relu(critic)
        critic = nn.Dense(self.config["FC_DIM_SIZE"], kernel_init=orthogonal(2), bias_init=constant(0.0))(
            critic
        )
        critic = nn.relu(critic)
        if self.config["ENV_NAME"] == "overcooked":
            critic = nn.Dense(self.config["FC_DIM_SIZE"] * 3 // 4, kernel_init=orthogonal(2), bias_init=constant(0.0))(
                critic
            )
            critic = nn.relu(critic)  # extra layer 1
            critic = nn.Dense(self.config["FC_DIM_SIZE"] // 2, kernel_init=orthogonal(2), bias_init=constant(0.0))(
                critic
            )
            critic = nn.relu(critic)  # extra layer 2
        critic = nn.Dense(1, kernel_init=orthogonal(1.0), bias_init=constant(0.0))(
            critic
        )

        return hidden, pi, jnp.squeeze(critic, axis=-1), other_pi


class Transition(NamedTuple):
    global_done: jnp.ndarray
    done: jnp.ndarray
    action: jnp.ndarray
    value: jnp.ndarray
    reward: jnp.ndarray
    log_prob: jnp.ndarray
    obs: jnp.ndarray
    info: jnp.ndarray
    agent_positions: jnp.ndarray
    other_action: jnp.ndarray

def batchify(x: dict, agent_list, num_actors):
    x = jnp.stack([x[a] for a in agent_list])
    return x.reshape((num_actors, -1))


def unbatchify(x: jnp.ndarray, agent_list, num_envs, num_actors):
    x = x.reshape((num_actors, num_envs, -1))
    return {a: x[i] for i, a in enumerate(agent_list)}


def make_train(config, update_step=0, xp_partner_params=None):
    # env = jaxmarl.make(config["ENV_NAME"], **config["ENV_KWARGS"])
    env = initialize_environment(config)
    
    config["NUM_ACTORS"] = env.num_agents * config["NUM_ENVS"]
    config["NUM_UPDATES"] = (
        config["TOTAL_TIMESTEPS"] // config["NUM_STEPS"] // config["NUM_ENVS"]
    )
    resume_update_step = update_step * (config["NUM_MINIBATCHES"] * config["UPDATE_EPOCHS"])
    config["MAX_TRAIN_UPDATES"] = (
        config["MAX_TRAIN_STEPS"] // config["NUM_STEPS"] // config["NUM_ENVS"]
    )
    config["NUM_REWARD_SHAPING_STEPS"] = config["MAX_TRAIN_UPDATES"] // 2  # used for annealing reward shaping
    config["MINIBATCH_SIZE"] = (
        config["NUM_ACTORS"] * config["NUM_STEPS"] // config["NUM_MINIBATCHES"]
    )
    config["CLIP_EPS"] = (
        config["CLIP_EPS"] / env.num_agents
        if config["SCALE_CLIP_EPS"]
        else config["CLIP_EPS"]
    )
    config["obs_dim"] = env.observation_space(env.agents[0]).shape

    obs, state = env.reset(jax.random.PRNGKey(0), params={'random_reset_fn': config['ENV_KWARGS']['random_reset_fn']})
    

    env = LogWrapper(env, env_params={'random_reset_fn': config['ENV_KWARGS']['random_reset_fn']})

    eval_envs = make_eval_envs_overcooked(config)
    eval_all_layouts = (
        config["ENV_NAME"] == "overcooked"
        and bool(config["ENV_KWARGS"]["random_reset"])
        and config["ENV_KWARGS"]["random_reset_fn"] == "reset_all"
        and bool(config["ENV_KWARGS"]["check_held_out"])
        and len(eval_envs) > 0
    )
    human_proxy_params = (
        load_human_proxy_params(
            config["EVAL_KWARGS"]["human_proxy_ckpt_dir"],
            int(config["EVAL_KWARGS"]["human_proxy_num_seeds"]),
        )
        if eval_all_layouts
        else {}
    )
    training_xp_enabled = (
        bool(config.get("XP_KWARGS", {}).get("enabled", False))
        and xp_partner_params is not None
    )
    xp_layout_names = get_toy_layout_names(config)
    xp_eval_envs = (
        make_modified_wall_eval_envs(config) if training_xp_enabled else {}
    )
    LOG_INTERVAL = max(1, int(config["NUM_UPDATES"]) // 100)

    def linear_schedule(count):
        frac = (
            1.0
            - ((count + resume_update_step) // (config["NUM_MINIBATCHES"] * config["UPDATE_EPOCHS"]))
            / config["MAX_TRAIN_UPDATES"]
        )
        frac = jnp.maximum(1e-9, frac)
        return config["LR"] * frac

    def train(rng, model_params=None, update_step=0):
        # INIT NETWORK
        network = ActorCriticRNN(env.action_space(env.agents[0]).n, config=config)
        bc_network = BCPolicy()
        rng, _rng = jax.random.split(rng)
        # get flattened obs dim
        flattened_obs_dim = 1
        for dim in env.observation_space(env.agents[0]).shape:
            flattened_obs_dim *= dim
        init_x = (
            jnp.zeros(
                (1, config["NUM_ENVS"], flattened_obs_dim)
            ),
            jnp.zeros((1, config["NUM_ENVS"])),
            jnp.zeros((1, config["NUM_ENVS"], 2, 2)).astype(jnp.int32)
        )
        init_hstate = ScannedRNN.initialize_carry(config["NUM_ENVS"], config["GRU_HIDDEN_DIM"])
        network_params = network.init(_rng, init_hstate, init_x)
        if model_params is not None:
            network_params = model_params
        if config["ANNEAL_LR"]:
            tx = optax.chain(
                optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
                optax.adam(learning_rate=linear_schedule, eps=1e-5),
            )
        else:
            tx = optax.chain(
                optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
                optax.adam(config["LR"], eps=1e-5),
            )
        train_state = TrainState.create(
            apply_fn=network.apply,
            params=network_params,
            tx=tx,
        )

        # INIT ENV
        rng, _rng = jax.random.split(rng)
        reset_rng = jax.random.split(_rng, config["NUM_ENVS"])
        obsv, env_state = jax.vmap(env.reset, in_axes=(0,))(reset_rng)
        init_hstate = ScannedRNN.initialize_carry(config["NUM_ACTORS"], config["GRU_HIDDEN_DIM"])

        def eval_layout_sp(eval_env, params, eval_rng):
            return evaluate_self_play_layout(
                eval_env=eval_env,
                params=params,
                eval_rng=eval_rng,
                network_apply=network.apply,
                initialize_carry=ScannedRNN.initialize_carry,
                batchify=batchify,
                unbatchify=unbatchify,
                num_eval_envs=int(config["EVAL_KWARGS"]["num_envs"]),
                num_steps=int(config["EVAL_KWARGS"]["num_steps"]),
                hidden_dim=config["GRU_HIDDEN_DIM"],
                beta=config["EVAL_KWARGS"]["beta"],
                argmax=config["EVAL_KWARGS"]["argmax"],
            )

        def eval_layout_xp(
            eval_env, main_params, bc_params_stacked, eval_rng,
        ):
            return evaluate_cross_play_layout(
                eval_env=eval_env,
                main_params=main_params,
                bc_params_stacked=bc_params_stacked,
                eval_rng=eval_rng,
                network_apply=network.apply,
                bc_network_apply=bc_network.apply,
                initialize_carry=ScannedRNN.initialize_carry,
                num_eval_envs=int(config["EVAL_KWARGS"]["num_envs"]),
                num_steps=int(config["EVAL_KWARGS"]["num_steps"]),
                hidden_dim=config["GRU_HIDDEN_DIM"],
                beta=config["EVAL_KWARGS"]["beta"],
                argmax=config["EVAL_KWARGS"]["argmax"],
                num_human_proxy_seeds=int(
                    config["EVAL_KWARGS"]["human_proxy_num_seeds"]
                ),
            )

        def eval_training_xp(eval_env, params_1, params_2, eval_rng):
            xp_cfg = config.get("XP_KWARGS", {})
            num_envs = int(
                xp_cfg.get("num_envs", config["EVAL_KWARGS"]["num_envs"])
            )
            num_steps = int(
                xp_cfg.get("num_steps", config["EVAL_KWARGS"]["num_steps"])
            )
            beta = float(
                xp_cfg.get("beta", config["EVAL_KWARGS"]["beta"])
            )
            argmax = bool(
                xp_cfg.get("argmax", config["EVAL_KWARGS"]["argmax"])
            )
            num_actors = eval_env.num_agents * num_envs

            eval_rng, reset_rng = jax.random.split(eval_rng)
            obs, state = jax.vmap(eval_env.reset)(
                jax.random.split(reset_rng, num_envs)
            )
            done = jnp.zeros((num_actors,), dtype=bool)
            hstate_1 = ScannedRNN.initialize_carry(
                num_actors, config["GRU_HIDDEN_DIM"]
            )
            hstate_2 = ScannedRNN.initialize_carry(
                num_actors, config["GRU_HIDDEN_DIM"]
            )
            returns = jnp.zeros((num_envs,), dtype=jnp.float32)

            def _xp_step(carry, _):
                state, obs, done, hstate_1, hstate_2, returns, rng = carry
                rng, action_rng_1, action_rng_2, step_rng = jax.random.split(
                    rng, 4
                )
                obs_batch = batchify(obs, eval_env.agents, num_actors)
                positions = batchify(
                    {
                        "agent_0": state.env_state.agent_pos,
                        "agent_1": state.env_state.agent_pos,
                    },
                    eval_env.agents,
                    num_actors,
                )
                network_input = (
                    obs_batch[jnp.newaxis, :],
                    done[jnp.newaxis, :],
                    positions[jnp.newaxis, :],
                )
                hstate_1, pi_1, _, _ = network.apply(
                    params_1, hstate_1, network_input
                )
                hstate_2, pi_2, _, _ = network.apply(
                    params_2, hstate_2, network_input
                )
                pi_1 = distrax.Categorical(logits=pi_1.logits * beta)
                pi_2 = distrax.Categorical(logits=pi_2.logits * beta)
                action_1 = jnp.where(
                    argmax,
                    jnp.argmax(pi_1.probs, axis=-1)[0],
                    pi_1.sample(seed=action_rng_1)[0],
                )
                action_2 = jnp.where(
                    argmax,
                    jnp.argmax(pi_2.probs, axis=-1)[0],
                    pi_2.sample(seed=action_rng_2)[0],
                )
                action = jnp.concatenate(
                    [action_1[:num_envs], action_2[num_envs:]], axis=0
                )
                env_action = unbatchify(
                    action, eval_env.agents, num_envs, eval_env.num_agents
                )
                env_action = {
                    key: value.squeeze() for key, value in env_action.items()
                }
                obs, state, reward, done_dict, _ = jax.vmap(eval_env.step)(
                    jax.random.split(step_rng, num_envs), state, env_action
                )
                done = batchify(
                    done_dict, eval_env.agents, num_actors
                ).squeeze()
                returns = returns + reward["agent_0"]
                return (
                    state,
                    obs,
                    done,
                    hstate_1,
                    hstate_2,
                    returns,
                    rng,
                ), None

            carry = (
                state,
                obs,
                done,
                hstate_1,
                hstate_2,
                returns,
                eval_rng,
            )
            carry, _ = jax.lax.scan(_xp_step, carry, None, num_steps)
            return carry[5].mean()

        # TRAIN LOOP
        @scan_tqdm(int(config["NUM_UPDATES"]))
        def _update_step(update_runner_state, unused):
            # COLLECT TRAJECTORIES
            runner_state, update_steps = update_runner_state

            def _env_step(runner_state, unused):
                train_state, env_state, last_obs, last_done, hstate, rng, update_step, beta_agent = runner_state

                # SELECT ACTION
                rng, _rng = jax.random.split(rng)
                obs_batch = batchify(last_obs, env.agents, config["NUM_ACTORS"])
                agent_positions = {'agent_0': env_state.env_state.agent_pos, 'agent_1': env_state.env_state.agent_pos}  
                agent_positions = batchify(agent_positions, env.agents, config["NUM_ACTORS"])
                ac_in = (
                    obs_batch[np.newaxis, :],
                    last_done[np.newaxis, :],
                    agent_positions[np.newaxis, :],
                )
                hstate, pi, value, other_pi = network.apply(train_state.params, hstate, ac_in)

                unbatched_logits = unbatchify(pi.logits, env.agents, config["NUM_ENVS"], env.num_agents)
                # agent 0 mask is 1 if beta_agent is 0, 0 otherwise
                agent_0_mask = jnp.where(beta_agent == 0, config["TRAIN_KWARGS"]["e3t_beta"], 1.00)
                agent_1_mask = jnp.where(beta_agent == 1, config["TRAIN_KWARGS"]["e3t_beta"], 1.00)
                multiply_row = lambda x, y: x * y
                unbatched_logits['agent_0'] = jax.vmap(multiply_row)(unbatched_logits['agent_0'], agent_0_mask)
                unbatched_logits['agent_1'] = jax.vmap(multiply_row)(unbatched_logits['agent_1'], agent_1_mask)
                batched_logits = batchify(unbatched_logits, env.agents, config["NUM_ACTORS"])
                pi = distrax.Categorical(logits=batched_logits)
                action = pi.sample(seed=_rng)
                log_prob = pi.log_prob(action)
                env_act = unbatchify(
                    action, env.agents, config["NUM_ENVS"], env.num_agents
                )
                env_act = {k: v.squeeze() for k, v in env_act.items()}
                other_env_act = {'agent_0': env_act['agent_1'], 'agent_1': env_act['agent_0']}  # get other agent's action
                other_action = batchify(other_env_act, env.agents, config["NUM_ACTORS"])

                # STEP ENV
                rng, _rng = jax.random.split(rng)
                rng_step = jax.random.split(_rng, config["NUM_ENVS"])
                obsv, env_state, reward, done, info = jax.vmap(
                    env.step, in_axes=(0, 0, 0)
                )(rng_step, env_state, env_act)
                shaped_reward = info['shaped_reward']
                reward_shaping_frac = jnp.maximum(0.0, 1.0 - (update_step / config["NUM_REWARD_SHAPING_STEPS"]))
                reward = jax.tree.map(lambda x, y: x + y * reward_shaping_frac, reward, shaped_reward)
                
                # remove shaped rewards
                del info['shaped_reward']

                info = jax.tree.map(lambda x: x.reshape((config["NUM_ACTORS"])), info)
                done_batch = batchify(done, env.agents, config["NUM_ACTORS"]).squeeze()
                transition = Transition(
                    jnp.tile(done["__all__"], env.num_agents),
                    last_done,
                    action.squeeze(),
                    value.squeeze(),
                    batchify(reward, env.agents, config["NUM_ACTORS"]).squeeze(),
                    log_prob.squeeze(),
                    obs_batch,
                    info,
                    agent_positions,
                    other_action.squeeze()
                )
                runner_state = (train_state, env_state, obsv, done_batch, hstate, rng, update_step, beta_agent)
                return runner_state, transition

            initial_hstate = runner_state[-2]
            (train_state, env_state, obsv, done_batch, hstate, rng) = runner_state
            # sample which agent we'll increase beta to
            beta_agent = jax.random.choice(rng, jnp.arange(env.num_agents), shape=(config["NUM_ENVS"],))
            rng, _rng = jax.random.split(rng)
            runner_state = (train_state, env_state, obsv, done_batch, hstate, rng, update_steps, beta_agent)
            runner_state, traj_batch = jax.lax.scan(
                _env_step, runner_state, None, config["NUM_STEPS"]
            )

            # CALCULATE ADVANTAGE
            train_state, env_state, last_obs, last_done, hstate, rng, update_steps, beta_agent = runner_state
            runner_state = (train_state, env_state, last_obs, last_done, hstate, rng)
            last_obs_batch = batchify(last_obs, env.agents, config["NUM_ACTORS"])
            agent_positions = {'agent_0': env_state.env_state.agent_pos, 'agent_1': env_state.env_state.agent_pos}
            agent_positions = batchify(agent_positions, env.agents, config["NUM_ACTORS"])
            ac_in = (
                last_obs_batch[np.newaxis, :],
                last_done[np.newaxis, :],
                agent_positions[np.newaxis, :],
            )
            _, _, last_val, _ = network.apply(train_state.params, hstate, ac_in)
            last_val = last_val.squeeze()

            def _calculate_gae(traj_batch, last_val):
                def _get_advantages(gae_and_next_value, transition):
                    gae, next_value = gae_and_next_value
                    done, value, reward = (
                        transition.global_done,
                        transition.value,
                        transition.reward,
                    )
                    delta = reward + config["GAMMA"] * next_value * (1 - done) - value
                    gae = (
                        delta
                        + config["GAMMA"] * config["GAE_LAMBDA"] * (1 - done) * gae
                    )
                    return (gae, value), gae

                _, advantages = jax.lax.scan(
                    _get_advantages,
                    (jnp.zeros_like(last_val), last_val),
                    traj_batch,
                    reverse=True,
                    unroll=16,
                )
                return advantages, advantages + traj_batch.value

            advantages, targets = _calculate_gae(traj_batch, last_val)

            # UPDATE NETWORK
            def _update_epoch(update_state, unused):
                def _update_minbatch(train_state, batch_info):
                    init_hstate, traj_batch, advantages, targets = batch_info

                    def _loss_fn(params, init_hstate, traj_batch, gae, targets):
                        # RERUN NETWORK
                        _, pi, value, other_pi = network.apply(
                            params,
                            jax.tree.map(lambda h: h.squeeze(), init_hstate),
                            (traj_batch.obs, traj_batch.done, traj_batch.agent_positions),
                        )
                        log_prob = pi.log_prob(traj_batch.action)
                        other_log_prob = other_pi.log_prob(traj_batch.other_action)
                        moa_nll_loss = -jnp.mean(other_log_prob)

                        # CALCULATE VALUE LOSS
                        value_pred_clipped = traj_batch.value + (
                            value - traj_batch.value
                        ).clip(-config["CLIP_EPS"], config["CLIP_EPS"])
                        value_losses = jnp.square(value - targets)
                        value_losses_clipped = jnp.square(value_pred_clipped - targets)
                        value_loss = 0.5 * jnp.maximum(
                            value_losses, value_losses_clipped
                        ).mean()

                        # CALCULATE ACTOR LOSS
                        logratio = log_prob - traj_batch.log_prob
                        ratio = jnp.exp(logratio)
                        gae = (gae - gae.mean()) / (gae.std() + 1e-8)
                        loss_actor1 = ratio * gae
                        loss_actor2 = (
                            jnp.clip(
                                ratio,
                                1.0 - config["CLIP_EPS"],
                                1.0 + config["CLIP_EPS"],
                            )
                            * gae
                        )
                        loss_actor = -jnp.minimum(loss_actor1, loss_actor2)
                        loss_actor = loss_actor.mean()
                        entropy = pi.entropy().mean()

                        # debug
                        approx_kl = ((ratio - 1) - logratio).mean()
                        clip_frac = jnp.mean(jnp.abs(ratio - 1) > config["CLIP_EPS"])

                        total_loss = (
                            loss_actor
                            + config["MOA_COEF"] * moa_nll_loss
                            + config["VF_COEF"] * value_loss
                            - config["ENT_COEF"] * entropy
                        )
                        return total_loss, (value_loss, loss_actor, entropy, ratio, approx_kl, clip_frac)

                    grad_fn = jax.value_and_grad(_loss_fn, has_aux=True)
                    total_loss, grads = grad_fn(
                        train_state.params, init_hstate, traj_batch, advantages, targets
                    )
                    train_state = train_state.apply_gradients(grads=grads)
                    return train_state, total_loss

                (
                    train_state,
                    init_hstate,
                    traj_batch,
                    advantages,
                    targets,
                    rng,
                ) = update_state
                rng, _rng = jax.random.split(rng)

                init_hstate = jax.tree.map(lambda h: jnp.reshape(h, (1, config["NUM_ACTORS"], -1)), init_hstate)
                batch = (
                    init_hstate,
                    traj_batch,
                    advantages.squeeze(),
                    targets.squeeze(),
                )
                permutation = jax.random.permutation(_rng, config["NUM_ACTORS"])

                shuffled_batch = jax.tree_util.tree_map(
                    lambda x: jnp.take(x, permutation, axis=1), batch
                )

                minibatches = jax.tree_util.tree_map(
                    lambda x: jnp.swapaxes(
                        jnp.reshape(
                            x,
                            [x.shape[0], config["NUM_MINIBATCHES"], -1]
                            + list(x.shape[2:]),
                        ),
                        1,
                        0,
                    ),
                    shuffled_batch,
                )

                train_state, total_loss = jax.lax.scan(
                    _update_minbatch, train_state, minibatches
                )
                update_state = (
                    train_state,
                    jax.tree.map(lambda h: h.squeeze(), init_hstate),
                    traj_batch,
                    advantages,
                    targets,
                    rng,
                )
                return update_state, total_loss

            update_state = (
                train_state,
                initial_hstate,
                traj_batch,
                advantages,
                targets,
                rng,
            )
            update_state, loss_info = jax.lax.scan(
                _update_epoch, update_state, None, config["UPDATE_EPOCHS"]
            )
            train_state = update_state[0]
            metric = traj_batch.info
            metric = jax.tree.map(
                lambda x: x.reshape(
                    (config["NUM_STEPS"], config["NUM_ENVS"], env.num_agents)
                ),
                traj_batch.info,
            )
            ratio_0 = loss_info[1][3].at[0,0].get().mean()
            loss_info = jax.tree.map(lambda x: x.mean(), loss_info)
            metric["loss"] = {
                "total_loss": loss_info[0],
                "value_loss": loss_info[1][0],
                "actor_loss": loss_info[1][1],
                "entropy": loss_info[1][2],
                "ratio": loss_info[1][3],
                "ratio_0": ratio_0,
                "approx_kl": loss_info[1][4],
                "clip_frac": loss_info[1][5],
            }
            rng = update_state[-1]

            if eval_all_layouts:
                run_eval = (
                    (update_steps % LOG_INTERVAL == 0)
                    | (update_steps == int(config["NUM_UPDATES"]) - 1)
                )

                def _do_eval(_):
                    base = jax.random.fold_in(rng, update_steps)
                    eval_layout_names = EVAL_LAYOUTS_9
                    out = {}
                    for i, eval_layout_name in enumerate(eval_layout_names):
                        out[eval_layout_name] = eval_layout_sp(
                            eval_envs[eval_layout_name],
                            train_state.params,
                            jax.random.fold_in(base, i),
                        )
                        out[f"{eval_layout_name}_xp"] = eval_layout_xp(
                            eval_envs[eval_layout_name],
                            train_state.params,
                            human_proxy_params[eval_layout_name],
                            jax.random.fold_in(base, 1000 + i),
                        )
                    out["mean"] = jnp.mean(jnp.stack([
                        out[name] for name in eval_layout_names
                    ]))
                    out["mean_xp"] = jnp.mean(jnp.stack([
                        out[f"{name}_xp"] for name in eval_layout_names
                    ]))
                    return out

                def _skip_eval(_):
                    nan = jnp.array(jnp.nan, dtype=jnp.float32)
                    eval_layout_names = EVAL_LAYOUTS_9
                    out = {name: nan for name in eval_layout_names}
                    out.update({
                        f"{name}_xp": nan for name in eval_layout_names
                    })
                    out["mean"] = nan
                    out["mean_xp"] = nan
                    return out

                metric["eval_returns"] = jax.lax.cond(
                    run_eval, _do_eval, _skip_eval, operand=None
                )

            if training_xp_enabled:
                run_training_xp = (
                    (
                        update_steps
                        % int(config["EVAL_KWARGS"]["eval_interval"])
                        == 0
                    )
                    | (update_steps == int(config["NUM_UPDATES"]) - 1)
                )

                def _do_training_xp(_):
                    base = jax.random.fold_in(rng, update_steps)
                    out = {}
                    for i, xp_layout_name in enumerate(xp_layout_names):
                        out[xp_layout_name] = eval_training_xp(
                            xp_eval_envs[xp_layout_name],
                            train_state.params,
                            xp_partner_params,
                            jax.random.fold_in(base, 2000 + i),
                        )
                    out["mean"] = jnp.mean(
                        jnp.stack([out[name] for name in xp_layout_names])
                    )
                    return out

                def _skip_training_xp(_):
                    out = {
                        name: jnp.asarray(jnp.nan, dtype=jnp.float32)
                        for name in xp_layout_names
                    }
                    out["mean"] = jnp.asarray(
                        jnp.nan, dtype=jnp.float32
                    )
                    return out

                metric["training_xp_returns"] = jax.lax.cond(
                    run_training_xp,
                    _do_training_xp,
                    _skip_training_xp,
                    operand=None,
                )

            def callback(metric):
                log_data = {
                    "returns": metric["returns"],
                    "env_step": metric["update_steps"]
                    * config["NUM_ENVS"]
                    * config["NUM_STEPS"],
                    **metric["loss"],
                }
                if "eval_returns" in metric:
                    eval_layout_names = EVAL_LAYOUTS_9
                    sp_mean = float(metric["eval_returns"]["mean"])
                    xp_mean = float(metric["eval_returns"]["mean_xp"])
                    if np.isfinite(sp_mean):
                        log_data["eval/mean"] = sp_mean
                        for eval_layout_name in eval_layout_names:
                            log_data[f"eval/{eval_layout_name}"] = float(
                                metric["eval_returns"][eval_layout_name]
                            )
                    if np.isfinite(xp_mean):
                        log_data["eval_xp/mean"] = xp_mean
                        for eval_layout_name in eval_layout_names:
                            log_data[f"eval_xp/{eval_layout_name}"] = float(
                                metric["eval_returns"][f"{eval_layout_name}_xp"]
                            )
                if "training_xp_returns" in metric:
                    xp_mean = float(metric["training_xp_returns"]["mean"])
                    if np.isfinite(xp_mean):
                        return_scale = 2.0 * float(
                            config["ENV_KWARGS"]["max_steps"]
                        )
                        log_data["xp/return_mean_fixed_heldout"] = xp_mean
                        log_data[
                            "xp/normalized_return_mean_fixed_heldout"
                        ] = xp_mean / return_scale
                        for xp_layout_name in xp_layout_names:
                            xp_return = float(
                                metric["training_xp_returns"][xp_layout_name]
                            )
                            log_data[
                                f"xp/return_{xp_layout_name}"
                            ] = xp_return
                            log_data[
                                f"xp/normalized_return_{xp_layout_name}"
                            ] = xp_return / return_scale
                wandb.log(log_data, step=int(metric["update_steps"]))
                current_return = float(metric["returns"])
                if current_return > best_return[0]:
                    best_return[0] = current_return
                    os.makedirs(config['filepath'], exist_ok=True)
                    ckpt_path = f"{config['filepath']}/{config['fcp_prefix']}seed{config['SEED']}_best_e3t.pkl"
                    with open(ckpt_path, "wb") as f:
                        pickle.dump({
                            'params': metric["params"],
                            'returns': current_return,
                            'update_steps': int(metric['update_steps']),
                        }, f)

            returns = metric["returned_episode_returns"][:, :, 0][
                            metric["returned_episode"][:, :, 0].astype(jnp.int32)
                        ].mean()
            metric["returns"] = returns
            metric["update_steps"] = update_steps
            metric["params"] = train_state.params
            jax.experimental.io_callback(callback, None, metric)
            update_steps = update_steps + 1
            runner_state = (train_state, env_state, last_obs, last_done, hstate, rng)  # hstate resets automatically
            return (runner_state, update_steps), metric

        best_return = [float('-inf')]
        
        rng, _rng = jax.random.split(rng)
        runner_state = (
            train_state,
            env_state,
            obsv,
            jnp.zeros((config["NUM_ACTORS"]), dtype=bool),
            init_hstate,
            _rng,
        )
        runner_state, metric = jax.lax.scan(
            _update_step, (runner_state, update_step), jnp.arange(int(config["NUM_UPDATES"])), int(config["NUM_UPDATES"])
        )
        return {"runner_state": runner_state}

    return train


@hydra.main(
    version_base=None,
    config_path="config",
    config_name="e3t_modified_wall_dual_destination_with_xp",
)
def main(config):
    config = OmegaConf.to_container(config, resolve=True)
    config["model_name"] = "E3T"
    map_name = get_wall_map_name(config)
    xpid = "lr-%s" % time.strftime("%Y%m%d-%H%M%S")

    xp_partner_params = None
    if bool(config.get("XP_KWARGS", {}).get("enabled", False)):
        xp_partner_params, xp_partner_path = load_xp_partner_params(config)
        config["XP_KWARGS"]["partner_path"] = xp_partner_path

    if config["WANDB_MODE"] == "online":
        with open("private.yaml") as f:
            private_info = yaml.load(f, Loader=yaml.FullLoader)
        wandb.login(key=private_info["wandb_key"])

    wandb.init(
        entity=config["ENTITY"],
        project=config["PROJECT"],
        group=config.get("WANDB_GROUP") or None,
        tags=["E3T", "RNN", "SP", "modified_wall"],
        config=config,
        mode=config["WANDB_MODE"],
        name=f"E3T_modified_wall_{map_name}_seed{config['SEED']}",
    )

    filepath = (
        f"ckpts/e3t/{config['ENV_NAME']}/modified_wall/{get_wall_map_dir_name(config)}"
        f"/ikFalse/{config['ENV_KWARGS']['random_reset_fn']}/e3t/{xpid}"
    )
    config["filepath"] = filepath
    config["fcp_prefix"] = ""
    print(f"Working on: \n{filepath}\n")

    model_params = None
    final_update_step = 0
    rng = jax.random.PRNGKey(config["SEED"])

    print(f"Starting from update step {final_update_step}")
    train_jit = jax.jit(
        make_train(
            config,
            final_update_step,
            xp_partner_params=xp_partner_params,
        ),
        device=jax.devices()[0],
    )
    out = train_jit(rng, model_params, final_update_step)
    jax.effects_barrier()

    runner_state, _ = out["runner_state"]
    model_state = runner_state[0]
    rng = runner_state[-1]
    num_updates = int(
        config["TOTAL_TIMESTEPS"]
        // config["NUM_STEPS"]
        // config["NUM_ENVS"]
    )

    os.makedirs(filepath, exist_ok=True)
    ckpt_path = (
        f"{filepath}/seed{config['SEED']}_ckpt"
        f"{config['TRAIN_KWARGS']['ckpt_id']}_e3t_updates"
        f"{num_updates}.pkl"
    )
    with open(ckpt_path, "wb") as f:
        pickle.dump(
            {
                "key": rng,
                "params": model_state.params,
                "update_steps": num_updates,
            },
            f,
        )

    print(f"Saved model to {ckpt_path}")
    wandb.finish()


if __name__ == "__main__":
    main()
