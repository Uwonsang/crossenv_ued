"""
Based on PureJaxRL Implementation of PPO.

Note, this file will only work for MPE environments with homogenous agents (e.g. Simple Spread).

"""
import os
import pickle
import jax
import jax.numpy as jnp
import flax.linen as nn
import numpy as np
import optax
from flax.linen.initializers import constant, orthogonal
from typing import Sequence, NamedTuple, Dict
from flax.training.train_state import TrainState
import distrax
import hydra
from omegaconf import OmegaConf

import jaxmarl
from jaxmarl.wrappers.baselines import LogWrapper
from jaxmarl.environments.overcooked import overcooked_layouts
from jaxmarl.environments.overcooked.layouts import make_counter_circuit_9x9, make_forced_coord_9x9, make_coord_ring_9x9, make_asymm_advantages_9x9, make_cramped_room_9x9

import wandb
import functools
from jax_tqdm import scan_tqdm
import time
import yaml
from algo_utils import EVAL_LAYOUTS_9
from environment_gradient import (
    compute_static_grid_conditioned_gradients,
    empty_environment_gradient_metrics,
    environment_gradient_log_key,
)
from paper_stiffness import (
    advance_rollout_rng,
    compute_paper_stiffness,
    count_unique_static_signatures,
    encode_static_grid_signature,
    empty_paper_stiffness_metrics,
    first_minibatch_actor_indices,
    select_stiffness_batch,
)
from representation_metrics import (
    compute_pooled_feature_rank_metrics,
    empty_pooled_feature_rank_metrics,
)


# Parameter groups used consistently by gradient diagnostics and norm metrics.
# Each group contains the shared encoder/RNN, its branch, and its output head.
ACTOR_TRUNK_KEYS = (
    "Conv_0", "Conv_1", "Dense_0", "Dense_1", "ScannedRNN_0",
    "Dense_2", "Dense_3", "Dense_4", "Dense_5", "Dense_6",
)
VALUE_TRUNK_KEYS = (
    "Conv_0", "Conv_1", "Dense_0", "Dense_1", "ScannedRNN_0",
    "Dense_7", "Dense_8", "Dense_9", "Dense_10", "Dense_11",
)
IPPO_FEATURE_RANK_NAMES = (
    ("shared", "shared"),
    ("policy", "policy_penultimate"),
    ("value", "value_penultimate"),
)
SHARED_TRUNK_KEYS = (
    "Conv_0", "Conv_1", "Dense_0", "Dense_1", "ScannedRNN_0",
)


def initialize_environment(config):
    layout_name = config["ENV_KWARGS"]["layout"]
    config["DIAGNOSTIC_AGGREGATION"] = "unique_static_grid"
    config["ENV_KWARGS"]["layout"] = overcooked_layouts[layout_name]
    env = jaxmarl.make(config["ENV_NAME"], **config["ENV_KWARGS"])

    if config["ENV_NAME"] == "overcooked":
        def reset_env(key):
            def reset_sub_dict(key, fn):
                key, subkey = jax.random.split(key)
                sampled_layout_dict = fn(subkey, ik=True)
                temp_o, temp_s = env.custom_reset(key, layout=sampled_layout_dict, random_reset=False, shuffle_inv_and_pot=False)
                key, subkey = jax.random.split(key)
                return (temp_o, temp_s), key
                
            asymm_reset, key = reset_sub_dict(key, make_asymm_advantages_9x9)
            coord_ring_reset, key = reset_sub_dict(key, make_coord_ring_9x9)
            counter_circuit_reset, key = reset_sub_dict(key, make_counter_circuit_9x9)
            forced_coord_reset, key = reset_sub_dict(key, make_forced_coord_9x9)
            cramped_room_reset, key = reset_sub_dict(key, make_cramped_room_9x9)
            layout_resets = [asymm_reset, coord_ring_reset, counter_circuit_reset, forced_coord_reset, cramped_room_reset]
            # stack all layouts
            stacked_layout_reset = jax.tree.map(lambda *x: jnp.stack(x), *layout_resets)
            # sample an index from 0 to 4
            index = jax.random.randint(key, (), minval=0, maxval=5)
            sampled_reset = jax.tree.map(lambda x: x[index], stacked_layout_reset)
            return sampled_reset
        @scan_tqdm(100)
        def gen_held_out(runner_state, unused):
            (i,) = runner_state
            _, ho_state = reset_env(jax.random.key(i))
            res = (ho_state.goal_pos, ho_state.wall_map, ho_state.pot_pos)
            carry = (i+1,)
            return carry, res
        carry, res = jax.lax.scan(gen_held_out, (0,), jnp.arange(100), 100)
        ho_goal, ho_wall, ho_pot = [], [], []
        for layout_name, layout_dict in overcooked_layouts.items():  # add hand crafted ones to heldout set
            if "9" in layout_name:
                _, ho_state = env.custom_reset(jax.random.PRNGKey(0), random_reset=False, shuffle_inv_and_pot=False, layout=layout_dict)
                ho_goal.append(ho_state.goal_pos)
                ho_wall.append(ho_state.wall_map)
                ho_pot.append(ho_state.pot_pos)
        ho_goal = jnp.stack(ho_goal, axis=0)
        ho_wall = jnp.stack(ho_wall, axis=0)
        ho_pot = jnp.stack(ho_pot, axis=0)
        ho_goal = jnp.concatenate([res[0], ho_goal], axis=0)
        ho_wall = jnp.concatenate([res[1], ho_wall], axis=0)
        ho_pot = jnp.concatenate([res[2], ho_pot], axis=0)
        env.held_out_goal, env.held_out_wall, env.held_out_pot = (ho_goal, ho_wall, ho_pot)
    elif config["ENV_NAME"] == "ToyCoop":
        # Generate 100 held-out states for ToyCoop
        @scan_tqdm(100)
        def gen_held_out_toycoop(runner_state, unused):
            (i,) = runner_state
            key = jax.random.key(i)
            state = env.custom_reset_fn(key, random_reset=True)
            res = (state.agent_pos, state.goal_pos, state.other_goal_pos)
            carry = (i+1,)
            return carry, res
        
        carry, res = jax.lax.scan(gen_held_out_toycoop, (0,), jnp.arange(100), 100)
        ho_agent_pos, ho_goal_pos, ho_other_goal_pos = res
        
        # Set the held-out states in the environment
        env.held_out_agent_pos = ho_agent_pos
        env.held_out_goal_pos = ho_goal_pos
        env.held_out_other_goal_pos = ho_other_goal_pos
    config["obs_dim"] = env.observation_space(env.agents[0]).shape
    return env

def _is_connected_jax(passable_9x9):
    """Flood fill: True iff all passable cells form one connected component."""
    flat = passable_9x9.astype(jnp.float32).flatten()
    visited = jnp.zeros_like(flat).at[jnp.argmax(flat)].set(1.0).reshape(9, 9)
    def _spread(v, _):
        nbr = jnp.maximum(
            jnp.maximum(jnp.pad(v[1:],   ((0,1),(0,0))), jnp.pad(v[:-1],  ((1,0),(0,0)))),
            jnp.maximum(jnp.pad(v[:,1:], ((0,0),(0,1))), jnp.pad(v[:,:-1],((0,0),(1,0)))),
        )
        return passable_9x9.astype(jnp.float32) * jnp.maximum(v, nbr), None
    visited, _ = jax.lax.scan(_spread, visited, None, 18)
    return jnp.sum(visited) == jnp.sum(passable_9x9.astype(jnp.float32))

def _classify_layout_jax(maze_map_9x9_ch0):
    """Return layout ID: 0=cramped_room_9, 1=asymm_advantages_9, 2=coord_ring_9,
                         3=counter_circuit_9, 4=forced_coord_9"""
    passable = (maze_map_9x9_ch0 == 1) | (maze_map_9x9_ch0 == 10)
    n = passable.sum()
    conn = _is_connected_jax(passable)
    return jnp.where(n == 8, 2,
           jnp.where(n == 6,
               jnp.where(conn, 0, 4),
               jnp.where(conn, 3, 1)
           )).astype(jnp.int32)


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
        batch_size, num_envs, flattened_obs_dim = obs.shape

        # Intermediate feature norms are only materialized during the
        # diagnostic apply(..., mutable=["intermediates"]) call. Recording
        # per-sample norms avoids retaining every Conv/Dense activation while
        # adding no norm-computation overhead to ordinary rollout/update
        # forwards.

        collect_intermediates = (
            not self.is_initializing()
            and self.is_mutable_collection("intermediates")
        )

        def record_feature_norm(name, features):
            if collect_intermediates:
                feature_vectors = features.reshape(
                    (batch_size, num_envs, -1)
                )
                sample_norms = jnp.linalg.norm(feature_vectors, axis=-1)
                self.sow(
                    "intermediates", f"feature_norm_{name}", sample_norms
                )

        if self.config["CONV_NET"]:
            if self.config["ENV_NAME"] == "overcooked":
                reshaped_obs = obs.reshape(-1, 9,9,26)
            else:
                reshaped_obs = obs.reshape(-1, 5,5,4)

            embedding = nn.Conv(
                # features=64 if "9" in self.config['layout_name'] and self.config["ENV_NAME"] == "overcooked")else 2 * self.config["FC_DIM_SIZE"],
                features=64,
                kernel_size=(2, 2),
                kernel_init=orthogonal(np.sqrt(2)),
                bias_init=constant(0.0),
            )(reshaped_obs)
            embedding = nn.relu(embedding)
            record_feature_norm("shared_conv_0", embedding)

            embedding = nn.Conv(
                # features=32 if "9" in self.config['layout_name'] and self.config["ENV_NAME"] == "overcooked") else self.config["FC_DIM_SIZE"],
                features=32,
                kernel_size=(2, 2),
                kernel_init=orthogonal(np.sqrt(2)),
                bias_init=constant(0.0),
            )(embedding)
            embedding = nn.relu(embedding)
            record_feature_norm("shared_conv_1", embedding)

            embedding = embedding.reshape((batch_size, num_envs, -1))
        else:
            embedding = obs

        embedding = nn.Dense(
            self.config["FC_DIM_SIZE"] * 2, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)
        )(embedding)
        embedding = nn.relu(embedding)
        record_feature_norm("shared_dense_0", embedding)

        embedding = nn.Dense(
            # self.config["FC_DIM_SIZE"] * 2 if "9" in self.config['layout_name'] else self.config["FC_DIM_SIZE"], 
            self.config["FC_DIM_SIZE"] * 2,
            kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)
        )(embedding)
        embedding = nn.relu(embedding)
        record_feature_norm("shared_dense_1", embedding)

        if self.config["LSTM"]:
            rnn_in = (embedding, dones)
            hidden, embedding = ScannedRNN()(hidden, rnn_in)
        else:
            embedding = nn.Dense(self.config["GRU_HIDDEN_DIM"], kernel_init=orthogonal(2), bias_init=constant(0.0))(embedding)
            embedding = nn.relu(embedding)
        embedding = embedding.reshape((batch_size, num_envs, -1))
        record_feature_norm("shared_recurrent", embedding)

        if (
            not self.is_initializing()
            and self.is_mutable_collection("feature_rank")
        ):
            self.sow("feature_rank", "shared", embedding)

        if not self.is_initializing():
            self.sow("intermediates", "shared_penultimate", embedding)

        #########
        # Actor
        #########
        actor_mean = nn.Dense(self.config["GRU_HIDDEN_DIM"] , kernel_init=orthogonal(2), bias_init=constant(0.0))(
            embedding
        )
        actor_mean = nn.relu(actor_mean)
        record_feature_norm("actor_hidden_0", actor_mean)
        actor_mean = nn.Dense(self.config["GRU_HIDDEN_DIM"] * 3 // 4, kernel_init=orthogonal(2), bias_init=constant(0.0))(
            actor_mean
        )
        actor_mean = nn.relu(actor_mean)
        record_feature_norm("actor_hidden_1", actor_mean)
        actor_mean = nn.Dense(
            self.config["GRU_HIDDEN_DIM"] // 2, kernel_init=orthogonal(2), bias_init=constant(0.0)
        )(actor_mean)
        actor_mean = nn.relu(actor_mean)
        record_feature_norm("actor_hidden_2", actor_mean)
        if self.config["ENV_NAME"] == "overcooked":
            actor_mean = nn.Dense(self.config["GRU_HIDDEN_DIM"] // 4, kernel_init=orthogonal(2), bias_init=constant(0.0))(
                actor_mean
            )
            actor_mean = nn.relu(actor_mean)  # extra layer 1
            record_feature_norm("actor_hidden_3", actor_mean)

        if not self.is_initializing():
            self.sow("intermediates", "actor_penultimate", actor_mean)
        if (
            not self.is_initializing()
            and self.is_mutable_collection("feature_rank")
        ):
            self.sow("feature_rank", "policy_penultimate", actor_mean)

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
        record_feature_norm("critic_hidden_0", critic)
        critic = nn.Dense(self.config["FC_DIM_SIZE"], kernel_init=orthogonal(2), bias_init=constant(0.0))(
            critic
        )
        critic = nn.relu(critic)
        record_feature_norm("critic_hidden_1", critic)
        if self.config["ENV_NAME"] == "overcooked":
            critic = nn.Dense(self.config["FC_DIM_SIZE"] * 3 // 4, kernel_init=orthogonal(2), bias_init=constant(0.0))(
                critic
            )
            critic = nn.relu(critic)  # extra layer 1
            record_feature_norm("critic_hidden_2", critic)
            critic = nn.Dense(self.config["FC_DIM_SIZE"] // 2, kernel_init=orthogonal(2), bias_init=constant(0.0))(
                critic
            )
            critic = nn.relu(critic)  # extra layer 2
            record_feature_norm("critic_hidden_3", critic)

        if not self.is_initializing():
            self.sow("intermediates", "critic_penultimate", critic)
        if (
            not self.is_initializing()
            and self.is_mutable_collection("feature_rank")
        ):
            self.sow("feature_rank", "value_penultimate", critic)
            
        critic = nn.Dense(1, kernel_init=orthogonal(1.0), bias_init=constant(0.0))(
            critic
        )

        return hidden, pi, jnp.squeeze(critic, axis=-1)


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
    layout_id: jnp.ndarray
    static_signature: jnp.ndarray


def ppo_loss(
    network, params, initial_hstate, traj_batch, advantages, targets, config
):
    """Compute the PPO training objective."""
    _, pi, value = network.apply(
        params,
        initial_hstate,
        (traj_batch.obs, traj_batch.done, traj_batch.agent_positions),
    )
    log_prob = pi.log_prob(traj_batch.action)
    value_pred_clipped = traj_batch.value + (
        value - traj_batch.value
    ).clip(-config["CLIP_EPS"], config["CLIP_EPS"])
    value_loss = 0.5 * jnp.maximum(
        jnp.square(value - targets),
        jnp.square(value_pred_clipped - targets),
    ).mean()

    advantages = (
        (advantages - advantages.mean()) / (advantages.std() + 1e-8)
    )
    logratio = log_prob - traj_batch.log_prob
    ratio = jnp.exp(logratio)
    actor_loss = -jnp.minimum(
        ratio * advantages,
        jnp.clip(
            ratio, 1.0 - config["CLIP_EPS"], 1.0 + config["CLIP_EPS"]
        ) * advantages,
    ).mean()
    entropy = pi.entropy().mean()
    approx_kl = ((ratio - 1) - logratio).mean()
    clip_frac = jnp.mean(jnp.abs(ratio - 1) > config["CLIP_EPS"])
    total_loss = (
        actor_loss
        + config["VF_COEF"] * value_loss
        - config["ENT_COEF"] * entropy
    )
    return total_loss, (
        value_loss, actor_loss, entropy, ratio, approx_kl, clip_frac
    )


POLICY_VALUE_METRIC_NAMES = (
    "shared_cosine",
    "policy_grad_norm",
    "weighted_value_grad_norm",
    "shared_conflict_rate",
)


def empty_policy_value_metrics(dtype=jnp.float32):
    return {
        name: jnp.asarray(jnp.nan, dtype=dtype)
        for name in POLICY_VALUE_METRIC_NAMES
    }


def compute_policy_value_interference(
    *,
    network,
    params,
    initial_hstate,
    traj_batch,
    advantages,
    targets,
    config,
    epsilon=1e-12,
):
    """Compare policy and value gradients on the shared PPO trunk.

    ``shared_conflict_rate`` is a 0/1 observation for this measurement;
    averaging it over measurement times (and seeds) gives the conflict rate.
    """

    def loss_components(candidate_params):
        _, pi, value = network.apply(
            candidate_params,
            initial_hstate,
            (
                traj_batch.obs,
                traj_batch.done,
                traj_batch.agent_positions,
            ),
        )
        log_prob = pi.log_prob(traj_batch.action)
        value_pred_clipped = traj_batch.value + (
            value - traj_batch.value
        ).clip(-config["CLIP_EPS"], config["CLIP_EPS"])
        value_loss = 0.5 * jnp.maximum(
            jnp.square(value - targets),
            jnp.square(value_pred_clipped - targets),
        ).mean()

        normalized_advantages = (
            advantages - advantages.mean()
        ) / (advantages.std() + 1e-8)
        ratio = jnp.exp(log_prob - traj_batch.log_prob)
        actor_loss = -jnp.minimum(
            ratio * normalized_advantages,
            jnp.clip(
                ratio,
                1.0 - config["CLIP_EPS"],
                1.0 + config["CLIP_EPS"],
            ) * normalized_advantages,
        ).mean()
        policy_loss = actor_loss - config["ENT_COEF"] * pi.entropy().mean()
        return policy_loss, value_loss

    policy_grads = jax.grad(lambda candidate: loss_components(candidate)[0])(
        params
    )
    raw_value_grads = jax.grad(
        lambda candidate: loss_components(candidate)[1]
    )(params)

    policy_squared_norm = jnp.asarray(0.0, dtype=jnp.float32)
    value_squared_norm = jnp.asarray(0.0, dtype=jnp.float32)
    dot_product = jnp.asarray(0.0, dtype=jnp.float32)
    for key in SHARED_TRUNK_KEYS:
        policy_leaves = jax.tree_util.tree_leaves(
            policy_grads["params"][key]
        )
        value_leaves = jax.tree_util.tree_leaves(
            raw_value_grads["params"][key]
        )
        for policy_leaf, value_leaf in zip(policy_leaves, value_leaves):
            policy_leaf = policy_leaf.astype(jnp.float32)
            value_leaf = value_leaf.astype(jnp.float32)
            policy_squared_norm += jnp.sum(jnp.square(policy_leaf))
            value_squared_norm += jnp.sum(jnp.square(value_leaf))
            dot_product += jnp.sum(policy_leaf * value_leaf)

    policy_norm = jnp.sqrt(policy_squared_norm)
    raw_value_norm = jnp.sqrt(value_squared_norm)
    weighted_value_norm = jnp.abs(
        jnp.asarray(config["VF_COEF"], dtype=jnp.float32)
    ) * raw_value_norm
    valid = jnp.logical_and(policy_norm > epsilon, raw_value_norm > epsilon)
    shared_cosine = jnp.where(
        valid,
        dot_product / jnp.maximum(policy_norm * raw_value_norm, epsilon),
        jnp.asarray(jnp.nan, dtype=jnp.float32),
    )
    shared_conflict = jnp.where(
        valid,
        (shared_cosine < 0.0).astype(jnp.float32),
        jnp.asarray(jnp.nan, dtype=jnp.float32),
    )
    return {
        "shared_cosine": shared_cosine,
        "policy_grad_norm": policy_norm,
        "weighted_value_grad_norm": weighted_value_norm,
        "shared_conflict_rate": shared_conflict,
    }


def batchify(x: dict, agent_list, num_actors):
    x = jnp.stack([x[a] for a in agent_list])
    return x.reshape((num_actors, -1))


def unbatchify(x: jnp.ndarray, agent_list, num_envs, num_actors):
    x = x.reshape((num_actors, num_envs, -1))
    return {a: x[i] for i, a in enumerate(agent_list)}


def make_train(
    config, update_step=0, save_info=None, opt_state=None,
    train_state_step=None,
):
    # env = jaxmarl.make(config["ENV_NAME"], **config["ENV_KWARGS"])
    env = initialize_environment(config)

    config["NUM_ACTORS"] = env.num_agents * config["NUM_ENVS"]
    config["NUM_UPDATES"] = (
        config["TOTAL_TIMESTEPS"] // config["NUM_STEPS"] // config["NUM_ENVS"]
    )
    # If opt_state is restored from a mid-run checkpoint, the optimizer's own step
    # count already reflects progress, so the manual offset would double-count it.
    resume_update_step = 0 if opt_state is not None else update_step * (config["NUM_MINIBATCHES"] * config["UPDATE_EPOCHS"])
    config["MAX_TRAIN_UPDATES"] = (
        config["MAX_TRAIN_STEPS"] // config["NUM_STEPS"] // config["NUM_ENVS"]
    )
    config["NUM_REWARD_SHAPING_STEPS"] = config["MAX_TRAIN_UPDATES"] // 2  # used for annealing reward shaping
    config["CLIP_EPS"] = (
        config["CLIP_EPS"] / env.num_agents
        if config["SCALE_CLIP_EPS"]
        else config["CLIP_EPS"]
    )
    config["obs_dim"] = env.observation_space(env.agents[0]).shape
    config["ACTION_DIM"] = env.action_space(env.agents[0]).n

    stiffness_config = config["STIFFNESS"]
    stiffness_enabled = bool(stiffness_config["ENABLED"])
    stiffness_chunk_size = int(stiffness_config["CHUNK_SIZE"])
    if int(config["NUM_ACTORS"]) % int(config["NUM_MINIBATCHES"]) != 0:
        raise ValueError("NUM_ACTORS must be divisible by NUM_MINIBATCHES.")
    actors_per_minibatch = int(config["NUM_ACTORS"]) // int(
        config["NUM_MINIBATCHES"]
    )
    stiffness_sample_size = int(config["NUM_STEPS"]) * actors_per_minibatch
    if stiffness_enabled and (
        stiffness_chunk_size <= 0
        or stiffness_sample_size % stiffness_chunk_size != 0
    ):
        raise ValueError(
            "STIFFNESS.CHUNK_SIZE must be positive and evenly divide the "
            "number of actor-states in one training minibatch."
        )
    env_steps_per_update = int(config["NUM_ENVS"]) * int(
        config["NUM_STEPS"]
    )
    stiffness_interval_env_steps = int(
        stiffness_config.get("INTERVAL_ENV_STEPS", 0)
    )
    if stiffness_interval_env_steps <= 0:
        stiffness_interval_env_steps = max(
            1, int(config["TOTAL_TIMESTEPS"]) // 100
        )
    stiffness_interval_updates = max(
        1,
        int(np.ceil(stiffness_interval_env_steps / env_steps_per_update)),
    )
    obs, state = env.reset(jax.random.PRNGKey(0), params={'random_reset_fn': config['ENV_KWARGS']['random_reset_fn']})

    env = LogWrapper(env, env_params={'random_reset_fn': config['ENV_KWARGS']['random_reset_fn']})

    def linear_schedule(count):
        frac = (
            1.0
            - ((count + resume_update_step) // (config["NUM_MINIBATCHES"] * config["UPDATE_EPOCHS"]))
            / config["MAX_TRAIN_UPDATES"]
        )
        frac = jnp.maximum(1e-9, frac)
        return config["LR"] * frac

    remaining_updates = int(config["NUM_UPDATES"]) - update_step

    def train(rng, model_params=None, resume_runner_state=None):
        # INIT NETWORK
        network = ActorCriticRNN(env.action_space(env.agents[0]).n, config=config)
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
        if opt_state is not None:
            train_state = train_state.replace(
                opt_state=opt_state,
                step=train_state.step if train_state_step is None else train_state_step,
            )

        # INIT OR RESTORE ENV RUNNER STATE
        if resume_runner_state is None:
            rng, _rng = jax.random.split(rng)
            reset_rng = jax.random.split(_rng, config["NUM_ENVS"])
            obsv, env_state = jax.vmap(env.reset, in_axes=(0,))(reset_rng)
            init_hstate = ScannedRNN.initialize_carry(
                config["NUM_ACTORS"], config["GRU_HIDDEN_DIM"]
            )
            rng, runner_rng = jax.random.split(rng)
        else:
            env_state, obsv, restored_done, init_hstate, runner_rng = resume_runner_state

        # WandB logging: cap at ~100 points over the full run. PPO scalars are
        # update-averaged; target/critic/TD metrics are logging-step snapshots.
        LOG_INTERVAL = max(1, int(config["NUM_UPDATES"]) // 100)
        _log_accum = {
            "sum": {},
            "count": {},
            "layout_sum": {},
            "layout_count": {},
        }

        # TRAIN LOOP
        @scan_tqdm(remaining_updates)
        def _update_step(update_runner_state, unused):
            # COLLECT TRAJECTORIES
            runner_state, update_steps = update_runner_state
            if stiffness_enabled:
                post_rollout_rng = advance_rollout_rng(
                    runner_state[-1], int(config["NUM_STEPS"])
                )
                _, permutation_rng = jax.random.split(post_rollout_rng)
                stiffness_actor_indices = first_minibatch_actor_indices(
                    permutation_rng,
                    int(config["NUM_ACTORS"]),
                    int(config["NUM_MINIBATCHES"]),
                )

            def _env_step(runner_state, unused):
                train_state, env_state, last_obs, last_done, hstate, rng, update_step = runner_state

                # layout BEFORE env.step: the layout this transition's action/reward belong to
                pre_maze_map = env_state.env_state.maze_map
                layout_id = jax.vmap(_classify_layout_jax)(pre_maze_map[:, 4:13, 4:13, 0])  # (NUM_ENVS,)
                layout_id = jnp.tile(layout_id, [env.num_agents])  # (NUM_ACTORS,), matches agent_positions
                static_signature = jax.vmap(encode_static_grid_signature)(
                    env_state.env_state.wall_map,
                    pre_maze_map[:, 4:13, 4:13, 0],
                    env_state.env_state.goal_pos,
                    env_state.env_state.pot_pos,
                )
                static_signature = jnp.tile(
                    static_signature, (env.num_agents, 1)
                )

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
                if stiffness_enabled:
                    sampled_hstate = jax.tree.map(
                        lambda value: jnp.take(
                            value, stiffness_actor_indices, axis=0
                        ),
                        hstate,
                    )
                hstate, pi, value = network.apply(train_state.params, hstate, ac_in)
                action = pi.sample(seed=_rng)
                log_prob = pi.log_prob(action)
                env_act = unbatchify(
                    action, env.agents, config["NUM_ENVS"], env.num_agents
                )
                env_act = {k: v.squeeze() for k, v in env_act.items()}

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
                    layout_id,
                    static_signature,
                )
                runner_state = (train_state, env_state, obsv, done_batch, hstate, rng, update_step)
                if stiffness_enabled:
                    return runner_state, (transition, sampled_hstate)
                return runner_state, transition

            initial_hstate = runner_state[-2]
            (train_state, env_state, obsv, done_batch, hstate, rng) = runner_state
            runner_state = (train_state, env_state, obsv, done_batch, hstate, rng, update_steps)
            runner_state, rollout_output = jax.lax.scan(
                _env_step, runner_state, jnp.arange(config["NUM_STEPS"])
            )
            if stiffness_enabled:
                traj_batch, sampled_hstates = rollout_output
            else:
                traj_batch = rollout_output

            # CALCULATE ADVANTAGE
            train_state, env_state, last_obs, last_done, hstate, rng, update_steps = runner_state
            runner_state = (train_state, env_state, last_obs, last_done, hstate, rng)
            last_obs_batch = batchify(last_obs, env.agents, config["NUM_ACTORS"])
            agent_positions = {'agent_0': env_state.env_state.agent_pos, 'agent_1': env_state.env_state.agent_pos}
            agent_positions = batchify(agent_positions, env.agents, config["NUM_ACTORS"])
            ac_in = (
                last_obs_batch[np.newaxis, :],
                last_done[np.newaxis, :],
                agent_positions[np.newaxis, :],
            )
            _, _, last_val = network.apply(train_state.params, hstate, ac_in)
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

            # ── per-layout gradient conflict ──────────────────────────────
            original_params = train_state.params

            # per-step layout id, classified pre-step inside _env_step, already tiled to actors
            _layout_ids_full = traj_batch.layout_id[:, :config["NUM_ENVS"]]

            run_stiffness = jnp.logical_or(
                jnp.logical_or(
                    jnp.equal(
                        update_steps % stiffness_interval_updates, 0
                    ),
                    jnp.equal(
                        update_steps, int(config["NUM_UPDATES"]) - 1
                    ),
                ),
                jnp.equal(update_steps, update_step),
            )

            if stiffness_enabled:
                selected_static_signatures_by_state = jnp.take(
                    traj_batch.static_signature,
                    stiffness_actor_indices,
                    axis=1,
                )
                selected_static_signatures = (
                    selected_static_signatures_by_state.reshape(
                        (-1, traj_batch.static_signature.shape[-1])
                    )
                )
                selected_layout_ids_by_state = jnp.take(
                    traj_batch.layout_id, stiffness_actor_indices, axis=1
                )
                selected_static_sample_mask_by_state = jnp.ones(
                    selected_static_signatures_by_state.shape[:2],
                    dtype=jnp.bool_,
                )
                policy_value_hstate = jax.tree.map(
                    lambda value: jnp.take(
                        value, stiffness_actor_indices, axis=0
                    ),
                    initial_hstate,
                )
                policy_value_traj = jax.tree.map(
                    lambda value: jnp.take(
                        value, stiffness_actor_indices, axis=1
                    ),
                    traj_batch,
                )
                policy_value_advantages = jnp.take(
                    advantages, stiffness_actor_indices, axis=1
                )
                policy_value_targets = jnp.take(
                    targets, stiffness_actor_indices, axis=1
                )
                feature_rank_metrics = jax.lax.cond(
                    run_stiffness,
                    lambda _: compute_pooled_feature_rank_metrics(
                        network=network,
                        params=original_params,
                        initial_hstate=policy_value_hstate,
                        network_inputs=(
                            policy_value_traj.obs,
                            policy_value_traj.done,
                            policy_value_traj.agent_positions,
                        ),
                        feature_names=IPPO_FEATURE_RANK_NAMES,
                        sample_mask=selected_static_sample_mask_by_state,
                        static_signatures=selected_static_signatures_by_state,
                        sample_layout_ids=selected_layout_ids_by_state,
                        max_groups=stiffness_sample_size,
                    ),
                    lambda _: empty_pooled_feature_rank_metrics(
                        IPPO_FEATURE_RANK_NAMES
                    ),
                    operand=None,
                )

                selected_advantage_mean = policy_value_advantages.mean()
                selected_advantage_std = policy_value_advantages.std()

                def _compute_stiffness(_):
                    (
                        stiffness_hstates,
                        stiffness_observations,
                        stiffness_dones,
                        stiffness_agent_positions,
                        stiffness_targets,
                    ) = select_stiffness_batch(
                        sampled_hstates=sampled_hstates,
                        observations=traj_batch.obs,
                        dones=traj_batch.done,
                        agent_positions=traj_batch.agent_positions,
                        targets=targets,
                        actor_indices=stiffness_actor_indices,
                    )
                    return compute_paper_stiffness(
                        network=network,
                        params=original_params,
                        sampled_hstates=stiffness_hstates,
                        observations=stiffness_observations,
                        dones=stiffness_dones,
                        agent_positions=stiffness_agent_positions,
                        targets=stiffness_targets,
                        sample_static_signatures=selected_static_signatures,
                        sample_layout_ids=selected_layout_ids_by_state.reshape(-1),
                        sample_mask=selected_static_sample_mask_by_state.reshape(-1),
                        max_static_grids=stiffness_sample_size,
                        value_param_keys=VALUE_TRUNK_KEYS,
                        chunk_size=stiffness_chunk_size,
                    )

                stiffness_metrics = jax.lax.cond(
                    run_stiffness,
                    _compute_stiffness,
                    lambda _: empty_paper_stiffness_metrics(),
                    operand=None,
                )
                static_unique_count = jax.lax.cond(
                    run_stiffness,
                    lambda _: count_unique_static_signatures(
                        selected_static_signatures
                    ).astype(jnp.float32),
                    lambda _: jnp.asarray(jnp.nan, dtype=jnp.float32),
                    operand=None,
                )
                policy_value_metrics = jax.lax.cond(
                    run_stiffness,
                    lambda _: compute_policy_value_interference(
                        network=network,
                        params=original_params,
                        initial_hstate=policy_value_hstate,
                        traj_batch=policy_value_traj,
                        advantages=policy_value_advantages,
                        targets=policy_value_targets,
                        config=config,
                    ),
                    lambda _: empty_policy_value_metrics(),
                    operand=None,
                )
                environment_gradient_metrics = jax.lax.cond(
                    run_stiffness,
                    lambda _: compute_static_grid_conditioned_gradients(
                        network=network,
                        params=original_params,
                        initial_hstates=policy_value_hstate,
                        trajectories=policy_value_traj,
                        normalized_advantages=(
                            policy_value_advantages
                            - selected_advantage_mean
                        ) / (selected_advantage_std + 1e-8),
                        targets=policy_value_targets,
                        sample_mask=selected_static_sample_mask_by_state,
                        static_signatures=selected_static_signatures_by_state,
                        max_static_grids=stiffness_sample_size,
                        shared_param_keys=SHARED_TRUNK_KEYS,
                        policy_gsnr_param_keys=SHARED_TRUNK_KEYS,
                        value_gsnr_param_keys=SHARED_TRUNK_KEYS,
                        value_loss_coefficient=config["VF_COEF"],
                        clip_eps=config["CLIP_EPS"],
                        entropy_coef=config["ENT_COEF"],
                        chunk_size=stiffness_chunk_size,
                    ),
                    lambda _: empty_environment_gradient_metrics(),
                    operand=None,
                )
            else:
                stiffness_metrics = empty_paper_stiffness_metrics()
                static_unique_count = jnp.asarray(jnp.nan, dtype=jnp.float32)
                policy_value_metrics = empty_policy_value_metrics()
                feature_rank_metrics = empty_pooled_feature_rank_metrics(
                    IPPO_FEATURE_RANK_NAMES
                )
                environment_gradient_metrics = (
                    empty_environment_gradient_metrics()
                )

            # UPDATE NETWORK
            def _update_epoch(update_state, unused):
                def _update_minbatch(train_state, batch_info):
                    init_hstate, traj_batch, advantages, targets = batch_info

                    def _loss_fn(params, init_hstate, traj_batch, gae, targets):
                        return ppo_loss(
                            network,
                            params,
                            jax.tree.map(lambda h: h.squeeze(), init_hstate),
                            traj_batch,
                            gae,
                            targets,
                            config,
                        )

                    grad_fn = jax.value_and_grad(_loss_fn, has_aux=True)
                    total_loss, grads = grad_fn(
                        train_state.params, init_hstate, traj_batch, advantages, targets
                    )

                    train_state = train_state.apply_gradients(grads=grads)
                    loss, loss_aux = total_loss
                    return train_state, (loss, loss_aux)

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

            # Save the small final checkpoint immediately after the final PPO
            # update, before loss-surface snapshots, evaluation, WandB, and the
            # larger resumable checkpoint can delay or interrupt finalization.
            if save_info is not None:
                num_updates_total = save_info["num_updates"]

                def final_save_callback(params):
                    ckpt_path = save_info["final_ckpt_path"]
                    os.makedirs(os.path.dirname(ckpt_path), exist_ok=True)
                    with open(ckpt_path, "wb") as f:
                        pickle.dump({
                            'key': save_info["rng"],
                            'params': params,
                            'update_steps': num_updates_total,
                        }, f)
                    print(f"Saved final model to {ckpt_path}")
                    print(
                        f"Finished training for seed {config['SEED']} with "
                        f"ckpt {config['TRAIN_KWARGS']['ckpt_id']}"
                        f"_updates{num_updates_total}"
                    )
                    print("--------------------------------")

                is_last_step = jnp.equal(
                    update_steps, num_updates_total - 1
                )
                jax.lax.cond(
                    is_last_step,
                    lambda _: jax.experimental.io_callback(
                        final_save_callback,
                        None,
                        train_state.params,
                        ordered=True,
                    ),
                    lambda _: None,
                    operand=None,
                )

            metric = traj_batch.info
            metric = jax.tree.map(
                lambda x: x.reshape(
                    (config["NUM_STEPS"], config["NUM_ENVS"], env.num_agents)
                ),
                traj_batch.info,
            )

            # 'returned_episode', 'returned_episode_lengths', 'returned_episode_returns'
            returns = metric["returned_episode_returns"][:, :, 0][
                metric["returned_episode"][:, :, 0].astype(jnp.int32)
            ].mean()
            # Save before reduction for per-layout return logging in callback
            episode_returns_step = metric["returned_episode_returns"][:, :, 0]  # (NUM_STEPS, NUM_ENVS)
            episode_done_step = metric["returned_episode"][:, :, 0]             # (NUM_STEPS, NUM_ENVS)
            # Reduce to scalars so scan output stays O(NUM_UPDATES), not O(NUM_UPDATES*NUM_STEPS*...)
            metric = jax.tree.map(lambda x: x.mean(), metric)
            
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
            metric["stiffness"] = stiffness_metrics
            metric["static_unique_count"] = static_unique_count
            metric["policy_value"] = policy_value_metrics
            metric["feature_rank"] = feature_rank_metrics
            metric["environment_gradient"] = environment_gradient_metrics
            rng = update_state[-1]

            def callback(metric):
                step = int(metric["update_steps"])
                # Metrics are produced after this zero-based update finishes.
                env_step = int((step + 1) * env_steps_per_update)
                is_standard_log_step = (
                    step % LOG_INTERVAL == 0
                    or step == int(config["NUM_UPDATES"]) - 1
                    or step == update_step
                )
                # Average finite scalar training metrics over the interval.
                def _accumulate(key, value):
                    value = float(value)
                    _log_accum["sum"].setdefault(key, 0.0)
                    _log_accum["count"].setdefault(key, 0)
                    if np.isfinite(value):
                        _log_accum["sum"][key] += value
                        _log_accum["count"][key] += 1

                _accumulate("returns", metric["returns"])
                for k, v in metric["loss"].items():
                    _accumulate(k, v)

                stiffness_log = {
                    f"stiffness/paper_{k}": float(v)
                    for k, v in metric["stiffness"].items()
                    if np.isfinite(float(v))
                }
                diversity_log = {}
                if np.isfinite(float(metric["static_unique_count"])):
                    diversity_log["diversity/static_grid_unique_count"] = int(
                        metric["static_unique_count"]
                    )
                policy_value_log = {
                    f"policy_value/{k}": float(v)
                    for k, v in metric["policy_value"].items()
                    if np.isfinite(float(v))
                }
                feature_rank_log = {
                    k: float(v)
                    for k, v in metric["feature_rank"].items()
                    if np.isfinite(float(v))
                }
                environment_gradient_log = {
                    environment_gradient_log_key(k): float(v)
                    for k, v in metric["environment_gradient"].items()
                    if np.isfinite(float(v))
                }
                if (
                    stiffness_log
                    or diversity_log
                    or policy_value_log
                    or feature_rank_log
                    or environment_gradient_log
                ) and not is_standard_log_step:
                    wandb.log(
                        {
                            "update_step": step,
                            "env_step": env_step,
                            **stiffness_log,
                            **diversity_log,
                            **policy_value_log,
                            **feature_rank_log,
                            **environment_gradient_log,
                        },
                        step=env_step,
                    )

                if config["ENV_NAME"] == "overcooked":
                    ep_rets = np.array(metric["episode_returns_step"])   # (NUM_STEPS, NUM_ENVS)
                    ep_done = np.array(metric["episode_done_step"]).astype(bool)
                    layout_ids = np.array(metric["layout_ids"])  # (NUM_STEPS, NUM_ENVS), pre-step layout
                    for t, e in np.argwhere(ep_done):
                        label = EVAL_LAYOUTS_9[int(layout_ids[t, e])]
                        _log_accum["layout_sum"][label] = _log_accum["layout_sum"].get(label, 0.0) + float(ep_rets[t, e])
                        _log_accum["layout_count"][label] = _log_accum["layout_count"].get(label, 0) + 1

                if is_standard_log_step:
                    log_dict = {
                        "update_step": step,
                        "env_step": env_step,
                    }
                    for k, s in _log_accum["sum"].items():
                        cnt = _log_accum["count"][k]
                        log_dict[k] = s / cnt if cnt > 0 else float("nan")

                    log_dict.update(stiffness_log)
                    log_dict.update(diversity_log)
                    log_dict.update(policy_value_log)
                    log_dict.update(feature_rank_log)
                    log_dict.update(environment_gradient_log)

                    if config["ENV_NAME"] == "overcooked":
                        for name in EVAL_LAYOUTS_9:
                            c = _log_accum["layout_count"].get(name, 0)
                            log_dict[f"train_returns/{name}"] = (
                                _log_accum["layout_sum"][name] / c if c > 0 else float("nan")
                            )

                    # Use environment interactions as WandB's global x-axis.
                    wandb.log(log_dict, step=env_step)

                    _log_accum["sum"] = {}
                    _log_accum["count"] = {}
                    _log_accum["layout_sum"] = {}
                    _log_accum["layout_count"] = {}

            metric["returns"] = returns
            metric["update_steps"] = update_steps

            callback_metric = {
                **metric,
                "episode_returns_step": episode_returns_step,
                "episode_done_step": episode_done_step,
                "layout_ids": _layout_ids_full,
            }

            jax.experimental.io_callback(
                callback, None, callback_metric, ordered=True
            )

            def ckpt_callback(
                params, opt_state_, tx_step, step,
                env_state_, last_obs_, last_done_, hstate_, rng_,
            ):
                step = int(step)
                mid_ckpt_dir = config["MID_CKPT_DIR"]
                os.makedirs(mid_ckpt_dir, exist_ok=True)
                mid_ckpt_path = os.path.join(mid_ckpt_dir, "resume_ckpt.pkl")
                with open(mid_ckpt_path, "wb") as f:
                    pickle.dump({
                        'params': params,
                        'opt_state': opt_state_,
                        'tx_step': tx_step,
                        'final_update_step': step + 1,
                        'wandb_run_id': wandb.run.id,
                        'runner_state': (
                            env_state_, last_obs_, last_done_, hstate_, rng_,
                        ),
                    }, f)

            # Keep periodic checkpoints.
            save_ckpt_interval = LOG_INTERVAL
            if save_ckpt_interval > 0:
                is_scheduled_ckpt = jnp.equal(
                    update_steps % save_ckpt_interval, 0
                )
                jax.lax.cond(
                    is_scheduled_ckpt,
                    lambda _: jax.experimental.io_callback(
                        ckpt_callback, None,
                        train_state.params, train_state.opt_state, train_state.step,
                        update_steps, env_state, last_obs, last_done, hstate, rng,
                        ordered=True,
                    ),
                    lambda _: None,
                    operand=None,
                )

            update_steps = update_steps + 1
            runner_state = (train_state, env_state, last_obs, last_done, hstate, rng)  # hstate resets automatically
            return (runner_state, update_steps), metric

        initial_done = (
            jnp.zeros((config["NUM_ACTORS"]), dtype=bool)
            if resume_runner_state is None else restored_done
        )
        runner_state = (
            train_state,
            env_state,
            obsv,
            initial_done,
            init_hstate,
            runner_rng,
        )
        runner_state, metric = jax.lax.scan(
            _update_step, (runner_state, update_step), jnp.arange(remaining_updates), remaining_updates
        )

        final_runner_state, _ = runner_state
        return {"params": final_runner_state[0].params}

    return train


@hydra.main(version_base=None, config_path="config", config_name="ippo_overcooked_CEC_gradient")
def main(config):
    config = OmegaConf.to_container(config)
    xpid = "lr-%s" % time.strftime("%Y%m%d-%H%M%S")

    if config['TRAIN_KWARGS']['finetune']:
        config['LR'] = config['LR'] / 10
        finetune_appendage = "_improved_finetune"
        if config['FCP']:
            fcp_prefix = "fcp_"
        else:
            fcp_prefix = ""
    elif config['ENV_NAME'] == 'overcooked':
        fcp_prefix = ""
        finetune_appendage = "_improved"
    else:
        fcp_prefix = ""
        finetune_appendage = "_improved"
    
    if config['ENV_KWARGS']['partial_obs']:
        finetune_appendage += "_partial_obs"
    if not config['LSTM']:
        finetune_appendage += "_no_lstm"
    if config['ENV_KWARGS']['incentivize_strat'] != 2:
        finetune_appendage += f"_incentivize_strat_{config['ENV_KWARGS']['incentivize_strat']}"
    
    if config["WANDB_MODE"] == "online":
        with open("private.yaml") as f:
            private_info = yaml.load(f, Loader=yaml.FullLoader)
        wandb.login(key=private_info["wandb_key"])

    resume_xpid = config["RESUME_XPID"]
    active_xpid = resume_xpid if resume_xpid else xpid

    filepath_base = f"ckpts/ippo/{config['ENV_NAME']}"
    if config["ENV_NAME"] == "overcooked":
        filepath_base += f"/{config['ENV_KWARGS']['layout']}"
    filepath_base += f"/ik{config['ENV_KWARGS']['random_reset']}/{config['ENV_KWARGS']['random_reset_fn']}"
    filepath = f"{filepath_base}/{active_xpid}"
    print(f"Working on: \n{filepath}\n")

    config['MID_CKPT_DIR'] = os.path.join(filepath, f"seed{config['SEED']}_mid_ckpts")

    mid_ckpt_path = os.path.join(config['MID_CKPT_DIR'], "resume_ckpt.pkl")
    _has_mid_ckpt = bool(resume_xpid) and os.path.exists(mid_ckpt_path)
    wandb_resume_id = None
    resume_runner_state = None
    resume_train_state_step = None
    if _has_mid_ckpt:
        with open(mid_ckpt_path, "rb") as f:
            _peek = pickle.load(f)
        wandb_resume_id = _peek.get('wandb_run_id', None)

    layout_name = config["ENV_KWARGS"]["layout"]
    if wandb_resume_id:
        wandb.init(
            entity=config["ENTITY"],
            project=config["PROJECT"],
            id=wandb_resume_id,
            resume="must",
            mode=config["WANDB_MODE"],
        )
    else:
        wandb.init(
            entity=config["ENTITY"],
            project=config["PROJECT"],
            tags=["IPPO", "RNN", "SP", "UNIQUE_STATIC_GRID"],
            config=config,
            mode=config["WANDB_MODE"],
            name=f"CEC_gradient_{layout_name}_seed{config['SEED']}"
        )

    num_updates = int(
        config["TOTAL_TIMESTEPS"] // config["NUM_STEPS"] // config["NUM_ENVS"]
    )
    final_ckpt_path = os.path.join(
        filepath,
        f"{fcp_prefix}seed{config['SEED']}_ckpt"
        f"{config['TRAIN_KWARGS']['ckpt_id']}{finetune_appendage}"
        f"_updates{num_updates}.pkl",
    )
    legacy_final_ckpt_path = os.path.join(
        filepath,
        f"{fcp_prefix}seed{config['SEED']}_ckpt"
        f"{config['TRAIN_KWARGS']['ckpt_id']}{finetune_appendage}.pkl",
    )
    if not config['TRAIN_KWARGS']['overwrite_ckpt']:
        if (
            os.path.exists(final_ckpt_path)
            or os.path.exists(legacy_final_ckpt_path)
        ):
            print(f"Checkpoint {config['TRAIN_KWARGS']['ckpt_id']} already exists, exiting")
            exit(0)

    if _has_mid_ckpt:
        print(f"Found mid-run checkpoint: {mid_ckpt_path}")
        model_params = _peek['params']
        opt_state = _peek.get('opt_state', None)
        resume_train_state_step = _peek.get('tx_step', None)
        resume_runner_state = _peek.get('runner_state', None)
        final_update_step = _peek['final_update_step']
        rng = jax.random.PRNGKey(config["SEED"])
        print(f"Resuming from update step {final_update_step}")
    elif config['TRAIN_KWARGS']['ckpt_id'] > 0:
        print("Loading checkpoint")
        with open(f"{filepath}/{fcp_prefix}seed{config['SEED']}_ckpt{config['TRAIN_KWARGS']['ckpt_id'] - 1}{finetune_appendage}.pkl", "rb") as f:
            previous_ckpt = pickle.load(f)
            model_params = previous_ckpt['params']
            opt_state = None
            final_update_step = previous_ckpt['final_update_step']
            rng = previous_ckpt['key']
            rng, _rng = jax.random.split(jax.random.PRNGKey(rng))

    elif config['TRAIN_KWARGS']['finetune']:
        finetune_filepath =f"ckpts/ippo/{config['ENV_NAME']}"
        if config["ENV_NAME"] == "overcooked":
            finetune_filepath += f"/cramped_room_9"
        if config['FCP']:
            finetune_filepath = f"{finetune_filepath}/ikFalse/{xpid}"
            finetune_ckpt_num = 19 if config['ENV_NAME'] == 'ToyCoop' else 6
        else:
            finetune_filepath = f"{finetune_filepath}/ikTrue/{config['ENV_KWARGS']['random_reset_fn']}/{xpid}"
            finetune_ckpt_num = 29 if config['ENV_NAME'] == 'overcooked' else 19
        print(f"Loading checkpoint for finetuning: {finetune_filepath}/{fcp_prefix}seed{config['SEED']}_ckpt{finetune_ckpt_num}_improved.pkl")
        with open(f"{finetune_filepath}/{fcp_prefix}seed{config['SEED']}_ckpt{finetune_ckpt_num}_improved.pkl", "rb") as f:  # need to resume from last checkpoint
            previous_ckpt = pickle.load(f)
            model_params = previous_ckpt['params']
            opt_state = None
            # final_update_step = previous_ckpt['final_update_step']
            final_update_step = 0
            rng = previous_ckpt['key']
            rng, _rng = jax.random.split(jax.random.PRNGKey(rng))
    else:
        model_params = None
        opt_state = None
        final_update_step = 0
        rng = jax.random.PRNGKey(config["SEED"])

    save_info = {
        "filepath": filepath,
        "fcp_prefix": fcp_prefix,
        "finetune_appendage": finetune_appendage,
        "rng": rng,
        "num_updates": num_updates,
        "final_ckpt_path": final_ckpt_path,
    }

    print(f"Starting from update step {final_update_step}")
    train_jit = jax.jit(
        make_train(
            config, final_update_step, save_info, opt_state,
            resume_train_state_step,
        ),
        device=jax.devices()[0],
    )
    train_jit(rng, model_params, resume_runner_state)

    jax.effects_barrier()
    jax.clear_caches()
    wandb.finish()


if __name__ == "__main__":
    main()
