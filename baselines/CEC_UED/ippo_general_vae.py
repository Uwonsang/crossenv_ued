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
from typing import Sequence, NamedTuple, Any, Dict
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
from baselines.CEC_UED.VAE.utils import load_checkpoint
from baselines.CEC_UED.regret_z_generator import AdversarialZ
from jaxmarl.viz.overcooked_visualizer import OvercookedVisualizer
from flax import struct
import chex
import imageio

def initialize_environment(config):
    layout_name = config["ENV_KWARGS"]["layout"]
    config['layout_name'] = layout_name
    config["ENV_KWARGS"]["layout"] = overcooked_layouts[layout_name]
    env = jaxmarl.make(config["ENV_NAME"], **config["ENV_KWARGS"])

    if config["ENV_NAME"] == "overcooked" or config["ENV_NAME"] == "overcooked_vae" or config["ENV_NAME"] == "overcooked_vae_crop":
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
            stacked_layout_reset = jax.tree_map(lambda *x: jnp.stack(x), *layout_resets)
            # sample an index from 0 to 4
            index = jax.random.randint(key, (), minval=0, maxval=5)
            sampled_reset = jax.tree_map(lambda x: x[index], stacked_layout_reset)
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
        lstm_state = jax.tree_map(
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
        if self.config["CONV_NET"]:
            if self.config["ENV_NAME"] == "overcooked" or self.config["ENV_NAME"] == "overcooked_vae":
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
            embedding = nn.Conv(
                # features=32 if "9" in self.config['layout_name'] and self.config["ENV_NAME"] == "overcooked") else self.config["FC_DIM_SIZE"],
                features=32,
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

        embedding = nn.Dense(
            # self.config["FC_DIM_SIZE"] * 2 if "9" in self.config['layout_name'] else self.config["FC_DIM_SIZE"], 
            self.config["FC_DIM_SIZE"] * 2,
            kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)
        )(embedding)
        embedding = nn.relu(embedding)

        if self.config["LSTM"]:
            rnn_in = (embedding, dones)
            hidden, embedding = ScannedRNN()(hidden, rnn_in)
        else:
            embedding = nn.Dense(self.config["GRU_HIDDEN_DIM"], kernel_init=orthogonal(2), bias_init=constant(0.0))(embedding)
            embedding = nn.relu(embedding)
        embedding = embedding.reshape((batch_size, num_envs, -1))

        #########
        # Actor
        #########
        actor_mean = nn.Dense(self.config["GRU_HIDDEN_DIM"] , kernel_init=orthogonal(2), bias_init=constant(0.0))(
            embedding
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
        if self.config["ENV_NAME"] == "overcooked" or self.config["ENV_NAME"] == "overcooked_vae":
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
        if self.config["ENV_NAME"] == "overcooked" or self.config["ENV_NAME"] == "overcooked_vae":
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

@struct.dataclass
class FilteredState:
    agent_dir_idx: chex.Array
    agent_inv: chex.Array
    maze_map: chex.Array


def batchify(x: dict, agent_list, num_actors):
    x = jnp.stack([x[a] for a in agent_list])
    return x.reshape((num_actors, -1))


def unbatchify(x: jnp.ndarray, agent_list, num_envs, num_actors):
    x = x.reshape((num_actors, num_envs, -1))
    return {a: x[i] for i, a in enumerate(agent_list)}


def make_train(config, update_step=0):
    # env = jaxmarl.make(config["ENV_NAME"], **config["ENV_KWARGS"])
    env = initialize_environment(config)
    agent_view_size = env.agent_view_size
    viz = OvercookedVisualizer()
    
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

    rng, _rng1, _rng2 = jax.random.split(jax.random.PRNGKey(0), 3)
    init_z = jax.random.normal(_rng1, (1, config["ENV_KWARGS"]['vae_config']['latent_dim']))
    obs, state = env.reset(_rng2, params={"random_reset_fn": config["ENV_KWARGS"]["random_reset_fn"], "z": init_z})

    env = LogWrapper(env, env_params={"random_reset_fn": config["ENV_KWARGS"]["random_reset_fn"]})

    def linear_schedule(count):
        frac = (
            1.0
            - ((count + resume_update_step) // (config["NUM_MINIBATCHES"] * config["UPDATE_EPOCHS"]))
            / config["MAX_TRAIN_UPDATES"]
        )
        frac = jnp.maximum(1e-9, frac)
        return config["LR"] * frac

    def train(rng, model_params=None, update_step=0):
        # Initialize z generator
        rng, _rng = jax.random.split(rng)
        z_gen = AdversarialZ(config, linear_schedule, _rng)

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
        
        # adversarial z generator state
        adversary_state = z_gen.init_state

        # INIT ENV
        rng, _rng = jax.random.split(rng)

        def _reset_all_envs(rng_key, adversary_state):
            """Sample a separate z for each env and reset each env with its own z."""
            key_z, key_env = jax.random.split(rng_key)
            keys_z = jax.random.split(key_z, config["NUM_ENVS"])

            def _sample_z(k):
                return z_gen.get_z(adversary_state, k)

            z_new, z_old, z_prior = jax.vmap(_sample_z, in_axes=0, out_axes=0)(keys_z) # z_new, z_old (batch_size, z_dim)       
            keys_env = jax.random.split(key_env, config["NUM_ENVS"])

            def _reset_env(k, z):
                obsv, env_state = env.reset(k, params={"z": z[None, :]})
                return obsv, env_state

            obsv, env_state = jax.vmap(_reset_env, in_axes=(0, 0), out_axes=0)(keys_env, z_new)

            return obsv, env_state, adversary_state, z_new, z_old, z_prior

        obsv, env_state, adversary_state, z_new, z_old, z_prior = \
            _reset_all_envs(_rng, adversary_state)
        
        init_hstate = ScannedRNN.initialize_carry(config["NUM_ACTORS"], config["GRU_HIDDEN_DIM"])

        # TRAIN LOOP
        @scan_tqdm(int(config["NUM_UPDATES"]))
        def _update_step(update_runner_state, unused):
            # COLLECT TRAJECTORIES
            runner_state, update_steps = update_runner_state

            def _env_step(runner_state, unused):
                (train_state, env_state, last_obs, last_done, hstate, rng, adversary_state,
                z_new, z_old, z_prior, update_step) = runner_state

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
                    env.step_env, in_axes=(0, 0, 0) # not use auto-reset
                )(rng_step, env_state, env_act)
                shaped_reward = info['shaped_reward']
                reward_shaping_frac = jnp.maximum(0.0, 1.0 - (update_step / config["NUM_REWARD_SHAPING_STEPS"]))
                reward = jax.tree_map(lambda x, y: x + y * reward_shaping_frac, reward, shaped_reward)
                
                # remove shaped rewards
                del info['shaped_reward']
                
                filtered_state = {
                    "agent_dir_idx": env_state.env_state.agent_dir_idx[0],
                    "agent_inv": env_state.env_state.agent_inv[0],
                    "maze_map": env_state.env_state.maze_map[0]}

                done_all = done["__all__"]
                # --- Manually reset only done envs (no auto-reset) ---
                rng, _rng = jax.random.split(rng)
                rng_reset = jax.random.split(_rng, config["NUM_ENVS"])
                #TODO : check the layout of reset, is it reset with after done?
                def _reset_if_done(done_i, rng_i, obsv_i, env_state_i, z_new_i, z_old_i, z_prior_i, adversary_state):
                    """Reset a single env if its episode is done, otherwise keep as is."""
                    def do_reset(_):
                        key_z, key_env = jax.random.split(rng_i)
                        z_new_i_new, z_old_i_new, z_prior_i_new = z_gen.get_z(adversary_state, key_z)

                        obsv_new_i, env_state_new_i = env.reset(
                            key_env, params={"z": z_new_i_new[None, :]}
                        )
                        return (obsv_new_i, env_state_new_i, z_new_i_new, z_old_i_new, z_prior_i_new)

                    def keep(_):
                        return (obsv_i, env_state_i, z_new_i, z_old_i, z_prior_i)

                    return jax.lax.cond(done_i, do_reset, keep, operand=None)

                obsv, env_state, z_new, z_old, z_prior = jax.vmap(
                    _reset_if_done, in_axes=(0, 0, 0, 0, 0, 0, 0, None), out_axes=(0, 0, 0, 0, 0)
                )(done_all, rng_reset, obsv, env_state, z_new, z_old, z_prior, adversary_state)

                info = jax.tree_map(lambda x: x.reshape((config["NUM_ACTORS"])), info)
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
                    agent_positions
                )
                runner_state = (train_state, env_state, obsv, done_batch, hstate, rng, adversary_state,
                                z_new, z_old, z_prior, update_step)

                return runner_state, (transition, FilteredState(**filtered_state))

            (train_state, env_state, obsv, done_batch, hstate, rng, adversary_state, 
            z_new, z_old, z_prior) = runner_state
            initial_hstate = hstate
            eval_start_state = (env_state, obsv, done_batch, hstate, rng, update_steps)

            runner_state = (train_state, env_state, obsv, done_batch, hstate, rng, 
            adversary_state, z_new, z_old, z_prior, update_steps)
            runner_state, (traj_batch, train_filtered_state) = jax.lax.scan(
                _env_step, runner_state, None, config["NUM_STEPS"]
            )

            # CALCULATE ADVANTAGE
            (train_state, env_state, last_obs, last_done, hstate, rng, adversary_state,
             z_new, z_old, z_prior, update_steps) = runner_state

            z_sample_info = {
                'z_sample/mean': z_new.mean(),
                'z_sample/std': z_new.std(),
                'log_prob': z_prior.log_prob(z_new).mean()}

            runner_state = (train_state, env_state, last_obs, last_done, hstate, rng, adversary_state,
                            z_new, z_old, z_prior)
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

            # UPDATE NETWORK
            def _update_epoch(update_state, unused):
                def _update_minbatch(train_state, batch_info):
                    init_hstate, traj_batch, advantages, targets = batch_info

                    def _loss_fn(params, init_hstate, traj_batch, gae, targets):
                        # RERUN NETWORK
                        _, pi, value = network.apply(
                            params,
                            jax.tree_map(lambda h: h.squeeze(), init_hstate),
                            (traj_batch.obs, traj_batch.done, traj_batch.agent_positions),
                        )
                        log_prob = pi.log_prob(traj_batch.action)

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

                init_hstate = jax.tree_map(lambda h: jnp.reshape(h, (1, config["NUM_ACTORS"], -1)), init_hstate)
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
                    jax.tree_map(lambda h: h.squeeze(), init_hstate),
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
            metric = jax.tree_map(
                lambda x: x.reshape(
                    (config["NUM_STEPS"], config["NUM_ENVS"], env.num_agents)
                ),
                traj_batch.info,
            )

            # 'returned_episode', 'returned_episode_lengths', 'returned_episode_returns'
            returns = metric["returned_episode_returns"][:, :, 0][
                metric["returned_episode"][:, :, 0].astype(jnp.int32)
            ].mean()
            # Reduce to scalars so scan output stays O(NUM_UPDATES), not O(NUM_UPDATES*NUM_STEPS*...)
            metric = jax.tree_map(lambda x: x.mean(), metric)
            
            ratio_0 = loss_info[1][3].at[0,0].get().mean()
            loss_info = jax.tree_map(lambda x: x.mean(), loss_info)
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

            # TODO check the layout of evaluation, make layout with same Z-sampling
            def _env_step_eval(runner_state_eval, unused):
                train_state, env_state_eval, obsv_eval, done_eval, hstate_eval, rng_eval, update_step = runner_state_eval

                rng_eval, _rng = jax.random.split(rng_eval)
                obs_batch = batchify(obsv_eval, env.agents, config["NUM_ACTORS"])
                agent_positions = {"agent_0": env_state_eval.env_state.agent_pos, "agent_1": env_state_eval.env_state.agent_pos}
                agent_positions = batchify(agent_positions, env.agents, config["NUM_ACTORS"])
                ac_in = (
                    obs_batch[np.newaxis, :],
                    done_eval[np.newaxis, :],
                    agent_positions[np.newaxis, :],
                )
                hstate_eval, pi_eval, _ = network.apply(train_state.params, hstate_eval, ac_in)
                action_eval = pi_eval.sample(seed=_rng)

                env_act_eval = unbatchify(
                    action_eval, env.agents, config["NUM_ENVS"], env.num_agents
                )
                env_act_eval = {k: v.squeeze() for k, v in env_act_eval.items()}

                rng_eval, _rng = jax.random.split(rng_eval)
                rng_step = jax.random.split(_rng, config["NUM_ENVS"])
                obsv_next, env_state_next, reward_eval, done_next, info_eval = jax.vmap(
                    env.step_env, in_axes=(0, 0, 0)
                )(rng_step, env_state_eval, env_act_eval)
                shaped_reward_eval = info_eval['shaped_reward']
                reward_shaping_frac = jnp.maximum(0.0, 1.0 - (update_step / config["NUM_REWARD_SHAPING_STEPS"]))
                reward_eval = jax.tree_map(lambda x, y: x + y * reward_shaping_frac, reward_eval, shaped_reward_eval)

                done_batch_next = batchify(done_next, env.agents, config["NUM_ACTORS"]).squeeze()

                runner_state_eval_next = (
                    train_state,
                    env_state_next,
                    obsv_next,
                    done_batch_next,
                    hstate_eval,
                    rng_eval,
                    update_step,
                )

                filtered_state = {
                    "agent_dir_idx": env_state_next.env_state.agent_dir_idx[0],
                    "agent_inv": env_state_next.env_state.agent_inv[0],
                    "maze_map": env_state_next.env_state.maze_map[0]}

                reward_eval = batchify(reward_eval, env.agents, config["NUM_ACTORS"]).squeeze()
                return runner_state_eval_next, (reward_eval, FilteredState(**filtered_state))

            env_state0, obsv0, done_batch0, hstate0, rng0, update_step = eval_start_state
            runner_state_eval0 = (train_state, env_state0, obsv0, done_batch0, hstate0, rng0, update_step)    
            runner_state_eval, (reward_eval_seq, test_filtered_state) = jax.lax.scan(
                _env_step_eval, runner_state_eval0, None, config["NUM_STEPS"])
            
            current_reward = jnp.transpose(traj_batch.reward, (1, 0)).reshape((config["NUM_ENVS"], env.num_agents, config["NUM_STEPS"]))
            trained_reward = jnp.transpose(reward_eval_seq, (1, 0)).reshape((config["NUM_ENVS"], env.num_agents, config["NUM_STEPS"]))
            current_reward = current_reward[:,0,:].sum(axis=1) # choose agent 0
            trained_reward = trained_reward[:,0,:].sum(axis=1)

            metric["current_reward_per_env"] = current_reward
            metric["trained_reward_per_env"] = trained_reward

            rng, _rng = jax.random.split(rng)
            adversary_state, train_info = z_gen.train_step(adversary_state, _rng, current_reward, trained_reward, 
                                                           z_new, z_old, z_prior)

            def callback(metric):
                wandb.log(
                    {
                        "returns": metric["returns"],
                        "env_step": int(metric["update_steps"] * config["NUM_ENVS"] * config["NUM_STEPS"]),
                        **metric["loss"],
                    }
                )
                z_gen.log_train_info(metric["adversary"])
                
                step = int(metric["update_steps"])
                save_interval = max(1, config["NUM_UPDATES"] // 19)
                if (step % save_interval == 0 or step == config["NUM_UPDATES"] - 1):
                    z_gen.save(metric["adversary_params"], step)

                def save_frames(filtered_state, step, file_path):
                    frames = [viz.custom_get_frame(jax.tree_map(lambda x: x[step], filtered_state), agent_view_size)
                        for step in range(config["NUM_STEPS"])]
                    
                    file_path = os.path.join(file_path, str(step))
                    os.makedirs(file_path, exist_ok=True)
                    for i, frame in enumerate(frames):
                        imageio.imwrite(os.path.join(file_path, f"layout_{i:03d}.png"), frame)
            
                if config["save_frames"]:
                    jax.debug.print("Saving frames")
                    save_frames(metric["train_filtered_state"], step, "/app/ckpts/ippo/overcooked_vae/train_images")
                    save_frames(metric["test_filtered_state"], step, "/app/ckpts/ippo/overcooked_vae/test_images")

            metric["adversary"] = {**train_info, **z_sample_info}
            metric["returns"] = returns
            metric["update_steps"] = update_steps

            callback_metric = {
                **metric,
                "adversary_params": adversary_state.params,
                "train_filtered_state": train_filtered_state,
                "test_filtered_state": test_filtered_state,
            }

            jax.experimental.io_callback(callback, None, callback_metric)
            update_steps = update_steps + 1
            runner_state = (train_state, env_state, last_obs, last_done, hstate, rng, adversary_state,
                            z_new, z_old, z_prior)
            return (runner_state, update_steps), metric

        rng, _rng = jax.random.split(rng)
        runner_state = (
            train_state,
            env_state,
            obsv,
            jnp.zeros((config["NUM_ACTORS"]), dtype=bool),
            init_hstate,
            _rng,
            adversary_state,
            z_new, z_old, z_prior
        )
        runner_state, metric = jax.lax.scan(
            _update_step, (runner_state, update_step), jnp.arange(int(config["NUM_UPDATES"])), int(config["NUM_UPDATES"])
        )
        return {"runner_state": runner_state}

    return train


@hydra.main(version_base=None, config_path="config", config_name="ippo_overcooked_CEC_VAE")
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
    elif config['ENV_NAME'] == 'overcooked_vae':
        fcp_prefix = ""
        finetune_appendage = "_improved_vae"
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
    
    layout_name = config["ENV_KWARGS"]["layout"]
    wandb.init(
        entity=config["ENTITY"],
        project=config["PROJECT"],
        tags=["IPPO", "RNN", "SP"],
        config=config,
        mode=config["WANDB_MODE"],
        name=f"CEC_VAE_{layout_name}_seed{config['SEED']}"
    )
    filepath = f"ckpts/ippo/{config['ENV_NAME']}"
    if config["ENV_NAME"] == "overcooked":
        filepath += f"/{config['ENV_KWARGS']['layout']}"
    filepath = f'{filepath}/ik{config["ENV_KWARGS"]["random_reset"]}/{config["ENV_KWARGS"]["random_reset_fn"]}/{xpid}'
    print(f"Working on: \n{filepath}\n")

    if not config['TRAIN_KWARGS']['overwrite_ckpt']:
        # check if ckpt exists
        if os.path.exists(f"{filepath}/{fcp_prefix}seed{config['SEED']}_ckpt{config['TRAIN_KWARGS']['ckpt_id']}{finetune_appendage}.pkl"):
            print(f"Checkpoint {config['TRAIN_KWARGS']['ckpt_id']} already exists, exiting")
            exit(0)

    if config['TRAIN_KWARGS']['ckpt_id'] > 0:
        print("Loading checkpoint")
        with open(f"{filepath}/{fcp_prefix}seed{config['SEED']}_ckpt{config['TRAIN_KWARGS']['ckpt_id'] - 1}{finetune_appendage}.pkl", "rb") as f:
            previous_ckpt = pickle.load(f)
            model_params = previous_ckpt['params']
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
            # final_update_step = previous_ckpt['final_update_step']
            final_update_step = 0
            rng = previous_ckpt['key']
            rng, _rng = jax.random.split(jax.random.PRNGKey(rng))
    else:
        model_params = None
        final_update_step = 0
        rng = jax.random.PRNGKey(config["SEED"])
    
    # load vae decoder params
    params, ckpt_config = load_checkpoint(config["VAE_CKPT_PATH"])
    decoder_params = {"params": params["params"]["Decoder_0"]}

    config["ENV_KWARGS"]["vae_decoder_params"] = decoder_params
    config["ENV_KWARGS"]["vae_config"] = ckpt_config
    config["filepath"] = filepath
    
    print(f"Starting from update step {final_update_step}")
    train_jit = jax.jit(make_train(config, final_update_step), device=jax.devices()[0])
    out = train_jit(rng, model_params, final_update_step)
    runner_state = out['runner_state']
    train_state = runner_state[0]
    model_state = train_state[0]
    rng = train_state[5]
    adversary_state = train_state[6]

    num_updates = (config["TOTAL_TIMESTEPS"] // config["NUM_STEPS"] // config["NUM_ENVS"])

    # save model
    os.makedirs(filepath, exist_ok=True)
    with open(f"{filepath}/{fcp_prefix}seed{config['SEED']}_ckpt{config['TRAIN_KWARGS']['ckpt_id']}{finetune_appendage}_updates{num_updates}.pkl", "wb") as f:
        ckpt = {'key': rng, 'params': model_state.params, 'update_steps': num_updates, 'adversary_params': adversary_state.params}
        pickle.dump(ckpt, f)

    print(f"Finished training for seed {config['SEED']} with ckpt {config['TRAIN_KWARGS']['ckpt_id']}")
    print(f'Saved to {filepath}/{fcp_prefix}train_info_seed{config["SEED"]}_ckpt{config["TRAIN_KWARGS"]["ckpt_id"]}{finetune_appendage}.png')

if __name__ == "__main__":
    main()
