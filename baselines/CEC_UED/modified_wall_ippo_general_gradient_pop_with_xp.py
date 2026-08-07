"""
Based on PureJaxRL Implementation of PPO.

Note, this file will only work for MPE environments with homogenous agents (e.g. Simple Spread).

"""
import os
import pickle
import glob
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
from jaxmarl.environments.toy_coop.toy_coop_no_pink import ToyCoopNoPink

import wandb
import functools
import pdb
from jax_tqdm import scan_tqdm
import time
import yaml
from jaxmarl.viz.overcooked_visualizer import OvercookedVisualizer
import flax.core
import imageio
from algo_utils import init_hdf5, make_eval_envs_overcooked, EVAL_LAYOUTS_9

TOY_LAYOUT_NAMES = ["empty", "wall_a", "wall_b", "wall_c"]

def get_wall_map_name(config):
    return config.get("map_name", config["ENV_KWARGS"].get("map_name", "empty"))

def get_toy_layout_names(config):
    return list(config.get("layout_names", TOY_LAYOUT_NAMES))

def get_wall_map_dir_name(config):
    map_name = get_wall_map_name(config)
    if map_name != "mixed":
        dir_name = map_name
    else:
        dir_name = "mixed_" + "_".join(get_toy_layout_names(config))
    ckpt_tag = str(config.get("CKPT_TAG", "")).strip()
    return f"{dir_name}_{ckpt_tag}" if ckpt_tag else dir_name

def get_toy_layout_wall_maps(layout_names):
    return jnp.array(
        [
            [[token == "B" for token in row] for row in ToyCoopNoPink.LAYOUTS[name]]
            for name in layout_names
        ],
        dtype=bool,
    )

def make_modified_wall_env(config):
    env_kwargs = dict(config["ENV_KWARGS"])
    env_kwargs["map_name"] = get_wall_map_name(config)
    allowed_keys = {
        "random_reset",
        "max_steps",
        "check_held_out",
        "debug",
        "partial_obs",
        "incentivize_strat",
        "map_name",
    }
    env_kwargs = {k: v for k, v in env_kwargs.items() if k in allowed_keys}
    if env_kwargs["map_name"] == "mixed":
        env_kwargs["layout_names"] = get_toy_layout_names(config)
    return ToyCoopNoPink(**env_kwargs)

def modified_wall_ckpt_root(config):
    return f"ckpts/ippo/{config['ENV_NAME']}/modified_wall/{get_wall_map_dir_name(config)}"

def latest_match(patterns):
    matches = []
    for pattern in patterns:
        matches.extend(glob.glob(pattern, recursive=True))
    if not matches:
        return None
    return sorted(matches, key=lambda p: os.path.getmtime(p))[-1]

def load_xp_partner_params(config):
    xp_cfg = config.get("XP_KWARGS", {})
    partner_seed = int(xp_cfg.get("partner_seed", 98))
    root = modified_wall_ckpt_root(config)
    path = latest_match([
        f"{root}/ikTrue/{config['ENV_KWARGS']['random_reset_fn']}/cec_popart_layout_eval/**/seed{partner_seed}_ckpt0_improved_pop_updates*.pkl",
    ])
    if path is None:
        raise FileNotFoundError(f"Missing CEC PopArt XP partner seed{partner_seed} under {root}")
    with open(path, "rb") as f:
        ckpt = pickle.load(f)
    print(f"Loaded XP partner CEC PopArt seed{partner_seed}: {path}")
    return ckpt["params"], path

def make_fixed_toy_env_kwargs(config, map_name):
    allowed = {"max_steps", "debug", "partial_obs", "incentivize_strat"}
    env_kwargs = {k: v for k, v in config["ENV_KWARGS"].items() if k in allowed}
    env_kwargs.update(
        {
            "map_name": map_name,
            "random_reset": False,
            "check_held_out": False,
        }
    )
    return env_kwargs

def make_modified_wall_eval_envs(config):
    if config["ENV_NAME"] != "ToyCoopNoPink":
        return {}
    envs = {}
    for map_name in get_toy_layout_names(config):
        base_env = ToyCoopNoPink(**make_fixed_toy_env_kwargs(config, map_name))
        envs[map_name] = LogWrapper(
            base_env,
            env_params={"random_reset_fn": config["EVAL_KWARGS"]["random_reset_fn"]},
        )
    return envs

def fixed_toy_states_for_layout_eval(config):
    states = []
    for map_name in get_toy_layout_names(config):
        env = ToyCoopNoPink(**make_fixed_toy_env_kwargs(config, map_name))
        states.append(env.custom_reset_fn(jax.random.PRNGKey(0), random_reset=False))
    return jax.tree.map(lambda *x: jnp.stack(x), *states)

def initialize_environment(config):
    if config["ENV_NAME"] == "ToyCoopNoPink":
        env = make_modified_wall_env(config)
        toy_heldout_num = int(config.get("TOY_HELDOUT_NUM", 100))

        @scan_tqdm(toy_heldout_num)
        def gen_held_out_toycoop(runner_state, unused):
            (i,) = runner_state
            key = jax.random.key(i)
            state = env.custom_reset_fn(key, random_reset=True)
            res = (state.agent_pos, state.goal_pos, state.wall_map)
            carry = (i+1,)
            return carry, res

        carry, res = jax.lax.scan(
            gen_held_out_toycoop,
            (0,),
            jnp.arange(toy_heldout_num),
            toy_heldout_num,
        )
        ho_agent_pos, ho_goal_pos, ho_wall_map = res
        fixed_states = fixed_toy_states_for_layout_eval(config)
        ho_agent_pos = jnp.concatenate([ho_agent_pos, fixed_states.agent_pos], axis=0)
        ho_goal_pos = jnp.concatenate([ho_goal_pos, fixed_states.goal_pos], axis=0)
        ho_wall_map = jnp.concatenate([ho_wall_map, fixed_states.wall_map], axis=0)

        env.held_out_agent_pos = ho_agent_pos
        env.held_out_goal_pos = ho_goal_pos
        env.held_out_wall_map = ho_wall_map
        config["obs_dim"] = env.observation_space(env.agents[0]).shape
        return env

    layout_name = config["ENV_KWARGS"]["layout"]
    config['layout_name'] = layout_name
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
    elif config["ENV_NAME"] == "ToyCoopNoPink":
        toy_heldout_num = int(config.get("TOY_HELDOUT_NUM", 100))

        @scan_tqdm(toy_heldout_num)
        def gen_held_out_toycoop(runner_state, unused):
            (i,) = runner_state
            key = jax.random.key(i)
            state = env.custom_reset_fn(key, random_reset=True)
            res = (state.agent_pos, state.goal_pos, state.wall_map)
            carry = (i+1,)
            return carry, res
        
        carry, res = jax.lax.scan(
            gen_held_out_toycoop,
            (0,),
            jnp.arange(toy_heldout_num),
            toy_heldout_num,
        )
        ho_agent_pos, ho_goal_pos, ho_wall_map = res
        
        # Set the held-out states in the environment
        env.held_out_agent_pos = ho_agent_pos
        env.held_out_goal_pos = ho_goal_pos
        env.held_out_wall_map = ho_wall_map
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


def _classify_toy_layout_jax(wall_map, layout_wall_maps):
    matches = jnp.all(layout_wall_maps == wall_map[None, :, :], axis=(1, 2))
    return jnp.argmax(matches).astype(jnp.int32)


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
        if self.config["CONV_NET"]:
            if self.config["ENV_NAME"] == "overcooked":
                reshaped_obs = obs.reshape(-1, 9,9,26)
            else:
                reshaped_obs = obs.reshape(-1, 5,5,5)

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
        critic = nn.Dense(1, kernel_init=orthogonal(1.0), bias_init=constant(0.0), name='critic_output')(
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


def batchify(x: dict, agent_list, num_actors):
    x = jnp.stack([x[a] for a in agent_list])
    return x.reshape((num_actors, -1))


def unbatchify(x: jnp.ndarray, agent_list, num_envs, num_actors):
    x = x.reshape((num_actors, num_envs, -1))
    return {a: x[i] for i, a in enumerate(agent_list)}


def make_train(config, update_step=0, save_info=None, opt_state=None, xp_partner_params=None):
    # env = jaxmarl.make(config["ENV_NAME"], **config["ENV_KWARGS"])
    env = initialize_environment(config)
    is_overcooked = config["ENV_NAME"] == "overcooked"
    agent_view_size = env.agent_view_size if is_overcooked else None
    viz = OvercookedVisualizer() if is_overcooked else None
    layout_names = EVAL_LAYOUTS_9 if is_overcooked else get_toy_layout_names(config)
    num_layouts = len(layout_names)
    toy_layout_wall_maps = None if is_overcooked else get_toy_layout_wall_maps(layout_names)
    
    config["NUM_ACTORS"] = env.num_agents * config["NUM_ENVS"]
    config["NUM_UPDATES"] = (
        config["TOTAL_TIMESTEPS"] // config["NUM_STEPS"] // config["NUM_ENVS"]
    )
    # If opt_state is restored from a mid-run checkpoint, the optimizer's own step
    # count already reflects progress, so the manual offset would double-count it.
    resume_update_step = 0 if opt_state is not None else update_step * (config["NUM_MINIBATCHES"] * config["UPDATE_EPOCHS"])
    remaining_updates = int(config["NUM_UPDATES"]) - update_step
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

    eval_envs = make_eval_envs_overcooked(config) if is_overcooked else make_modified_wall_eval_envs(config)

    def linear_schedule(count):
        frac = (
            1.0
            - ((count + resume_update_step) // (config["NUM_MINIBATCHES"] * config["UPDATE_EPOCHS"]))
            / config["MAX_TRAIN_UPDATES"]
        )
        frac = jnp.maximum(1e-9, frac)
        return config["LR"] * frac

    def train(rng, model_params=None, init_popart_mu=None, init_popart_sigma=None):
        save_xpid = "lr-%s" % time.strftime("%Y%m%d-%H%M%S")
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
            params=flax.core.freeze(network_params),
            tx=tx,
        )
        if opt_state is not None:
            train_state = train_state.replace(opt_state=opt_state)

        # INIT ENV
        rng, _rng = jax.random.split(rng)
        reset_rng = jax.random.split(_rng, config["NUM_ENVS"])
        obsv, env_state = jax.vmap(env.reset, in_axes=(0,))(reset_rng)
        init_hstate = ScannedRNN.initialize_carry(config["NUM_ACTORS"], config["GRU_HIDDEN_DIM"])

        # PopArt running statistics: network predicts normalized values
        popart_mu = init_popart_mu
        popart_sigma = init_popart_sigma

        # TRAIN LOOP
        @scan_tqdm(remaining_updates)
        def _update_step(update_runner_state, unused):
            # COLLECT TRAJECTORIES
            runner_state, update_steps, popart_mu, popart_sigma = update_runner_state

            def _env_step(runner_state, unused):
                train_state, env_state, last_obs, last_done, hstate, rng, update_step = runner_state

                # layout BEFORE env.step: the layout this transition's action/reward belong to
                if is_overcooked:
                    pre_maze_map = env_state.env_state.maze_map
                    layout_id = jax.vmap(_classify_layout_jax)(pre_maze_map[:, 4:13, 4:13, 0])  # (NUM_ENVS,)
                else:
                    layout_id = jax.vmap(lambda wall_map: _classify_toy_layout_jax(wall_map, toy_layout_wall_maps))(env_state.env_state.wall_map)  # (NUM_ENVS,)
                layout_id = jnp.tile(layout_id, [env.num_agents])  # (NUM_ACTORS,), matches agent_positions

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
                )
                runner_state = (train_state, env_state, obsv, done_batch, hstate, rng, update_step)
                return runner_state, transition

            initial_hstate = runner_state[-2]
            (train_state, env_state, obsv, done_batch, hstate, rng) = runner_state
            runner_state = (train_state, env_state, obsv, done_batch, hstate, rng, update_steps)
            runner_state, traj_batch = jax.lax.scan(
                _env_step, runner_state, None, config["NUM_STEPS"]
            )

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
                # Denormalize network outputs (normalized) to real scale for GAE
                last_val_real = last_val * popart_sigma + popart_mu

                def _get_advantages(gae_and_next_value, transition):
                    gae, next_value = gae_and_next_value
                    done, value_norm, reward = (
                        transition.global_done,
                        transition.value,
                        transition.reward,
                    )
                    value_real = value_norm * popart_sigma + popart_mu
                    delta = reward + config["GAMMA"] * next_value * (1 - done) - value_real
                    gae = (
                        delta
                        + config["GAMMA"] * config["GAE_LAMBDA"] * (1 - done) * gae
                    )
                    return (gae, value_real), (gae, delta)

                _, (advantages, td_errors) = jax.lax.scan(
                    _get_advantages,
                    (jnp.zeros_like(last_val_real), last_val_real),
                    traj_batch,
                    reverse=True,
                    unroll=16,
                )
                # targets are real-scale returns; _loss_fn will normalize before comparing
                targets_real = advantages + traj_batch.value * popart_sigma + popart_mu
                return advantages, targets_real, td_errors

            advantages, targets, td_errors = _calculate_gae(traj_batch, last_val)

            # ── per-layout gradient conflict ──────────────────────────────
            _LAYOUT_NAMES = layout_names
            original_params = train_state.params

            # per-step layout id, classified pre-step inside _env_step, already tiled to actors
            _actor_layout_full = traj_batch.layout_id  # (NUM_STEPS, NUM_ACTORS)
            _layout_ids_full = _actor_layout_full[:, :config["NUM_ENVS"]]  # (NUM_STEPS, NUM_ENVS)

            # ── value target statistics: raw / popart-normalized / between-within decomposition ──
            _targets_norm = (targets - popart_mu) / popart_sigma

            target_stats = {}
            target_stats["target_raw/mean"] = targets.mean()
            target_stats["target_popart/mean"] = _targets_norm.mean()

            # ── critic quality: how well does the critic fit the targets it's trained on? ──
            _value_norm = traj_batch.value
            _err_norm = _targets_norm - _value_norm

            target_stats["critic/explained_var"] = 1.0 - _err_norm.var() / (_targets_norm.var() + 1e-8)
            target_stats["critic/bias"] = _err_norm.mean()
            target_stats["critic/rmse"] = jnp.sqrt((_err_norm ** 2).mean())

            target_stats["td_error/mean_abs"] = jnp.abs(td_errors).mean()
            target_stats["td_error/rmse"] = jnp.sqrt((td_errors ** 2).mean())

            _N = targets.size
            _layer_means, _layer_vars, _layer_counts, _layer_stds_n = [], [], [], []
            for _lid, _name in enumerate(_LAYOUT_NAMES):
                _mask = (_actor_layout_full == _lid).astype(jnp.float32)
                _cnt_raw = _mask.sum()
                _cnt = _cnt_raw + 1e-8
                _mean_l = (targets * _mask).sum() / _cnt
                _var_l = ((targets - _mean_l) ** 2 * _mask).sum() / _cnt
                _layer_means.append(_mean_l)
                _layer_vars.append(_var_l)
                _layer_counts.append(_cnt_raw)

                _masked = jnp.where(_mask.astype(bool), targets, jnp.nan)
                target_stats[f"target_raw/{_name}/mean"] = _mean_l
                target_stats[f"target_scale/{_name}/std"] = jnp.sqrt(_var_l + 1e-8)

                _sum_n = (_targets_norm * _mask).sum()
                _mean_ln = _sum_n / _cnt
                _var_ln = ((_targets_norm - _mean_ln) ** 2 * _mask).sum() / _cnt
                _std_ln = jnp.sqrt(_var_ln)
                target_stats[f"target_popart/{_name}/mean"] = _mean_ln
                target_stats[f"target_popart/{_name}/std"] = _std_ln
                _layer_stds_n.append(_std_ln)

                _err_l = _err_norm * _mask
                _bias_l = _err_l.sum() / _cnt
                _mse_l = ((_err_norm ** 2) * _mask).sum() / _cnt
                _resid_var_l = (((_err_norm - _bias_l) ** 2) * _mask).sum() / _cnt
                target_stats[f"critic/{_name}/explained_var"] = 1.0 - _resid_var_l / (_var_ln + 1e-8)
                target_stats[f"critic/{_name}/bias"] = _bias_l
                target_stats[f"critic/{_name}/rmse"] = jnp.sqrt(_mse_l)

                _td_l = td_errors * _mask
                target_stats[f"td_error/{_name}/mean_abs"] = jnp.abs(_td_l).sum() / _cnt
                target_stats[f"td_error/{_name}/rmse"] = jnp.sqrt((_td_l ** 2).sum() / _cnt)

            # law of total variance: total_var ≈ within_var + between_var
            _within_var = sum(c * v for c, v in zip(_layer_counts, _layer_vars)) / _N
            _grand_mean = sum(c * m for c, m in zip(_layer_counts, _layer_means)) / _N
            _between_var = sum(c * (m - _grand_mean) ** 2 for c, m in zip(_layer_counts, _layer_means)) / _N
            target_stats["target_variance_decomp/within"] = _within_var
            target_stats["target_variance_decomp/between"] = _between_var
            target_stats["target_variance_decomp/between_ratio"] = _between_var / (_within_var + _between_var + 1e-8)

            _layer_stds = jnp.sqrt(jnp.stack(_layer_vars) + 1e-8)
            target_stats["target_scale/std_max"] = jnp.max(_layer_stds)
            target_stats["target_scale/std_min"] = jnp.min(_layer_stds)
            target_stats["target_scale/std_ratio"] = (
                jnp.max(_layer_stds) / (jnp.min(_layer_stds) + 1e-8)
            )
            target_stats["target_scale/std_cv"] = (
                jnp.std(_layer_stds) / (jnp.mean(_layer_stds) + 1e-8)
            )

            _layer_stds_n = jnp.stack(_layer_stds_n)
            target_stats["target_scale_popart/std_max"] = jnp.max(_layer_stds_n)
            target_stats["target_scale_popart/std_min"] = jnp.min(_layer_stds_n)
            target_stats["target_scale_popart/std_ratio"] = (
                jnp.max(_layer_stds_n) / (jnp.min(_layer_stds_n) + 1e-8)
            )
            target_stats["target_scale_popart/std_cv"] = (
                jnp.std(_layer_stds_n) / (jnp.mean(_layer_stds_n) + 1e-8)
            )

            _ev_vals = jnp.stack([target_stats[f"critic/{_n}/explained_var"] for _n in _LAYOUT_NAMES])
            target_stats["critic/worst_family_ev"] = jnp.min(_ev_vals)
            # ── end value target statistics ─────────────────────────────────

            # subsample: use only the first _GC_STEPS steps to reduce activation memory
            _GC_STEPS = config["GRAD_CONFLICT_STEPS"]
            _gc_traj = jax.tree.map(lambda x: x[:_GC_STEPS], traj_batch)
            _gc_adv  = advantages[:_GC_STEPS]
            # targets are real-scale; normalize for value loss in _fwd (network outputs normalized)
            _gc_tgt  = (targets[:_GC_STEPS] - popart_mu) / popart_sigma

            # reuse full-trajectory classification for the gradient-conflict subsample
            _layout_ids = _layout_ids_full[:_GC_STEPS]
            _actor_layout = _actor_layout_full[:_GC_STEPS]

            def _tdot(g1, g2):
                return sum(
                    jnp.sum(a * b)
                    for a, b in zip(jax.tree_util.tree_leaves(g1), jax.tree_util.tree_leaves(g2))
                )

            def _tnorm2(g):
                return sum(jnp.sum(a ** 2) for a in jax.tree_util.tree_leaves(g))

            # Per-loss-type scalar accumulators: norms_sq[lid], dots[(i,j)], prev[lid]
            # actor and value are logged; entropy is excluded (less interpretable).
            _gc_state = {
                'actor': {'norms_sq': [], 'dots': {}, 'prev': []},
                'value': {'norms_sq': [], 'dots': {}, 'prev': []},
            }

            _sample_counts = []  # raw sample count per layout (for quality monitoring)
            for _lid in range(num_layouts):
                _mask = (_actor_layout == _lid).astype(jnp.float32)  # (_GC_STEPS, NUM_ACTORS)
                _cnt = _mask.sum() + 1e-8
                _sample_counts.append(_mask.sum())

                # Single forward pass per layout; 2 backward passes via vjp cotangents.
                def _fwd(p, mask=_mask, cnt=_cnt):
                    _, pi, value = jax.checkpoint(network.apply)(
                        p, initial_hstate,
                        (_gc_traj.obs, _gc_traj.done, _gc_traj.agent_positions),
                    )
                    lp = pi.log_prob(_gc_traj.action)
                    adv_mean = (_gc_adv * mask).sum() / cnt
                    adv_std = jnp.sqrt(((_gc_adv - adv_mean) ** 2 * mask).sum() / cnt + 1e-8)
                    gae = (_gc_adv - adv_mean) / (adv_std + 1e-8)
                    ratio = jnp.exp(lp - _gc_traj.log_prob)
                    al = -(jnp.minimum(
                        ratio * gae,
                        jnp.clip(ratio, 1 - config["CLIP_EPS"], 1 + config["CLIP_EPS"]) * gae,
                    ) * mask).sum() / cnt
                    vpc = _gc_traj.value + (value - _gc_traj.value).clip(
                        -config["CLIP_EPS"], config["CLIP_EPS"]
                    )
                    vl = 0.5 * (jnp.maximum(
                        jnp.square(value - _gc_tgt), jnp.square(vpc - _gc_tgt)
                    ) * mask).sum() / cnt
                    return al, vl

                _, _vjp_fn = jax.vjp(_fwd, original_params)
                # cotangent (1, 0) → actor gradient; (0, 1) → value gradient
                _g_actor, = _vjp_fn((1.0, 0.0))
                _g_value, = _vjp_fn((0.0, 1.0))

                for _loss_type, _g in [('actor', _g_actor), ('value', _g_value)]:
                    _s = _gc_state[_loss_type]
                    _s['norms_sq'].append(_tnorm2(_g))
                    for _prev_lid, _g_prev in enumerate(_s['prev']):
                        _s['dots'][(_prev_lid, _lid)] = _tdot(_g_prev, _g)
                    _s['prev'].append(_g)

            grad_conflict = {}
            # per-layout sample counts — interpret cosine similarity values alongside these:
            # low count → noisy gradient → cosine similarity less reliable
            for _lid, _name in enumerate(_LAYOUT_NAMES):
                grad_conflict[f"grad_conflict/sample_count/{_name}"] = _sample_counts[_lid]

            _sample_counts_arr = jnp.stack(_sample_counts)
            _sample_total = _sample_counts_arr.sum() + 1e-8
            for _i in range(num_layouts):
                grad_conflict[f"sample_share/{_LAYOUT_NAMES[_i]}"] = _sample_counts_arr[_i] / _sample_total

            for _loss_type, _s in _gc_state.items():
                # per-layout gradient norms
                for _i in range(num_layouts):
                    grad_conflict[f"grad_conflict_{_loss_type}/norm/{_LAYOUT_NAMES[_i]}"] = (
                        jnp.sqrt(_s['norms_sq'][_i])
                    )
                # gradient share p_f, dominance ratio D, norm CV
                _norms = jnp.stack([jnp.sqrt(_s['norms_sq'][_i]) for _i in range(num_layouts)])
                _norm_sum = _norms.sum() + 1e-8
                for _i in range(num_layouts):
                    grad_conflict[f"grad_share_{_loss_type}/{_LAYOUT_NAMES[_i]}"] = _norms[_i] / _norm_sum
                grad_conflict[f"grad_dominance_{_loss_type}"] = jnp.max(_norms) / (jnp.median(_norms) + 1e-8)
                grad_conflict[f"grad_norm_cv_{_loss_type}"] = jnp.std(_norms) / (jnp.mean(_norms) + 1e-8)

                # sample-weighted gradient share: weights each layout's (per-sample-mean) norm
                # by its actual sample count, approximating its contribution to the real,
                # unmasked combined gradient (which averages over all samples, not per layout).
                _weighted_norms = _norms * _sample_counts_arr
                _weighted_norm_sum = _weighted_norms.sum() + 1e-8
                for _i in range(num_layouts):
                    grad_conflict[f"grad_share_weighted_{_loss_type}/{_LAYOUT_NAMES[_i]}"] = (
                        _weighted_norms[_i] / _weighted_norm_sum
                    )
                # pairwise cosine similarities
                for _i in range(num_layouts):
                    for _j in range(_i + 1, num_layouts):
                        cos = _s['dots'][(_i, _j)] / (
                            jnp.sqrt(_s['norms_sq'][_i] * _s['norms_sq'][_j]) + 1e-8
                        )
                        grad_conflict[
                            f"grad_conflict_{_loss_type}/{_LAYOUT_NAMES[_i]}_vs_{_LAYOUT_NAMES[_j]}"
                        ] = cos
                        # neg_dot_ij = max(0, -g_i · g_j): magnitude of conflict, 0 when aligned
                        neg_dot = jnp.maximum(0.0, -_s['dots'][(_i, _j)])
                        grad_conflict[
                            f"grad_neg_dot_{_loss_type}/{_LAYOUT_NAMES[_i]}_vs_{_LAYOUT_NAMES[_j]}"
                        ] = neg_dot
                # joint update alignment: cos(g_i, g_all) where g_all = sum of all 5 layout gradients
                # measures how much each layout's update aligns with the combined training direction
                _g_all = jax.tree.map(lambda *gs: sum(gs), *_s['prev'])
                _norm_all_sq = _tnorm2(_g_all)
                for _i in range(num_layouts):
                    _dot_i_all = _tdot(_s['prev'][_i], _g_all)
                    _align = _dot_i_all / (jnp.sqrt(_s['norms_sq'][_i] * _norm_all_sq) + 1e-8)
                    grad_conflict[f"grad_conflict_{_loss_type}/alignment/{_LAYOUT_NAMES[_i]}"] = _align
                    # leave-one-out: cos(g_i, g_all - g_i) — removes self-contribution
                    _g_others = jax.tree.map(lambda ga, gi: ga - gi, _g_all, _s['prev'][_i])
                    _norm_others_sq = _tnorm2(_g_others)
                    _dot_i_others = _tdot(_s['prev'][_i], _g_others)
                    _align_loo = _dot_i_others / (
                        jnp.sqrt(_s['norms_sq'][_i] * _norm_others_sq) + 1e-8
                    )
                    grad_conflict[f"grad_conflict_{_loss_type}/alignment_loo/{_LAYOUT_NAMES[_i]}"] = _align_loo
            # ── end gradient conflict ──────────────────────────────────────

            # UPDATE NETWORK
            def _update_epoch(update_state, unused):
                def _update_minbatch(train_state, batch_info):
                    init_hstate, traj_batch, advantages, targets = batch_info

                    def _loss_fn(params, init_hstate, traj_batch, gae, targets):
                        # RERUN NETWORK
                        _, pi, value = network.apply(
                            params,
                            jax.tree.map(lambda h: h.squeeze(), init_hstate),
                            (traj_batch.obs, traj_batch.done, traj_batch.agent_positions),
                        )
                        log_prob = pi.log_prob(traj_batch.action)

                        # CALCULATE VALUE LOSS (in normalized space)
                        targets_norm = (targets - popart_mu) / popart_sigma
                        value_pred_clipped = traj_batch.value + (
                            value - traj_batch.value
                        ).clip(-config["CLIP_EPS"], config["CLIP_EPS"])
                        value_losses = jnp.square(value - targets_norm)
                        value_losses_clipped = jnp.square(value_pred_clipped - targets_norm)
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

            # ── PopArt: update EMA stats and correct output layer weights ──
            _pa_alpha = config["POPART_ALPHA"]
            _batch_mu = targets.mean()
            _batch_var = targets.var()
            _mu_new = (1 - _pa_alpha) * popart_mu + _pa_alpha * _batch_mu
            _sigma_new = jnp.sqrt(jnp.maximum(
                (1 - _pa_alpha) * (popart_sigma ** 2 + popart_mu ** 2)
                + _pa_alpha * (_batch_var + _batch_mu ** 2)
                - _mu_new ** 2,
                1e-8,
            ))
            # Preserve outputs precisely: rescale critic_output layer so the
            # real-scale prediction is unchanged despite the new normalization.
            _pa_params = flax.core.unfreeze(train_state.params)
            _pa_params['params']['critic_output']['kernel'] = (
                popart_sigma / _sigma_new
            ) * _pa_params['params']['critic_output']['kernel']
            _pa_params['params']['critic_output']['bias'] = (
                popart_sigma * _pa_params['params']['critic_output']['bias'] + popart_mu - _mu_new
            ) / _sigma_new
            train_state = train_state.replace(params=flax.core.freeze(_pa_params))
            popart_mu, popart_sigma = _mu_new, _sigma_new
            # ── end PopArt ────────────────────────────────────────────────

            metric = traj_batch.info
            metric = jax.tree.map(
                lambda x: x.reshape(
                    (config["NUM_STEPS"], config["NUM_ENVS"], env.num_agents)
                ),
                traj_batch.info,
            )

            # Save before reduction for per-layout return logging in callback
            episode_returns_step = metric["returned_episode_returns"][:, :, 0]  # (NUM_STEPS, NUM_ENVS)
            episode_done_step = metric["returned_episode"][:, :, 0]             # (NUM_STEPS, NUM_ENVS)
            done_count = jnp.maximum(episode_done_step.sum(), 1.0)
            returns = (episode_returns_step * episode_done_step).sum() / done_count
            success_rate = (
                ((episode_returns_step > -config["ENV_KWARGS"]["max_steps"]) * episode_done_step).sum()
                / done_count
            )
            normalized_returns = returns / (2.0 * config["ENV_KWARGS"]["max_steps"])
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
                "popart/mu": popart_mu,
                "popart/sigma": popart_sigma,
                **grad_conflict,
                **target_stats,
            }
            rng = update_state[-1]

            def eval_layout(eval_env, params, eval_rng):
                num_eval_envs = int(config["EVAL_KWARGS"]["num_envs"])
                num_actors_eval = eval_env.num_agents * num_eval_envs 

                eval_rng, reset_rng = jax.random.split(eval_rng)
                reset_rngs = jax.random.split(reset_rng, num_eval_envs)
                init_obs, init_state = jax.vmap(eval_env.reset, in_axes=(0,))(reset_rngs)
                init_hstate = ScannedRNN.initialize_carry(num_actors_eval, config["GRU_HIDDEN_DIM"])
                init_done = jnp.zeros((num_actors_eval,), dtype=bool)
                init_returns = jnp.zeros((num_eval_envs,), dtype=jnp.float32)
                runner_state = (init_state, init_obs, init_done, init_hstate, init_returns, eval_rng)

                def _eval_step(carry, _):
                    env_state_e, obs_e, done_e, hstate_e, returns_e, rng_e = carry

                    rng_e, _rng_e = jax.random.split(rng_e)
                    obs_batch = batchify(obs_e, eval_env.agents, num_actors_eval)
                    agent_positions = {'agent_0': env_state_e.env_state.agent_pos, 'agent_1': env_state_e.env_state.agent_pos}
                    agent_positions = batchify(agent_positions, eval_env.agents, num_actors_eval)
                    ac_in = (
                        obs_batch[np.newaxis, :],
                        done_e[np.newaxis, :],
                        agent_positions[np.newaxis, :],
                    )
                    hstate_next, pi, _ = network.apply(params, hstate_e, ac_in)
                    pi = distrax.Categorical(logits=pi.logits * config["EVAL_KWARGS"]["beta"])
                    sampled_action = pi.sample(seed=_rng_e)[0]
                    greedy_action = jnp.argmax(pi.probs, axis=-1)[0]
                    action = jnp.where(config["EVAL_KWARGS"]["argmax"], greedy_action, sampled_action)

                    env_act = unbatchify(action, eval_env.agents, num_eval_envs, eval_env.num_agents)
                    env_act = {k: v.squeeze() for k, v in env_act.items()} #TODO: check

                    rng_e, _rng_e = jax.random.split(rng_e)
                    rng_step_e = jax.random.split(_rng_e, num_eval_envs)
                    obs_next, state_next, reward, done, _info = jax.vmap(
                        eval_env.step, in_axes=(0, 0, 0)
                    )(rng_step_e, env_state_e, env_act)

                    done_next = batchify(done, eval_env.agents, num_actors_eval).squeeze()
                    returns_next = returns_e + reward["agent_0"]

                    return (state_next, obs_next, done_next, hstate_next, returns_next, rng_e), None
                
                runner_state, _ = jax.lax.scan(_eval_step, runner_state, None, int(config["EVAL_KWARGS"]["num_steps"]))
                state, obs, done, h_state, returns, rng = runner_state

                return returns.mean()

            def xp_layout(eval_env, params_1, params_2, eval_rng):
                xp_cfg = config.get("XP_KWARGS", {})
                num_eval_envs = int(xp_cfg.get("num_envs", config["EVAL_KWARGS"]["num_envs"]))
                num_steps = int(xp_cfg.get("num_steps", config["EVAL_KWARGS"]["num_steps"]))
                beta = float(xp_cfg.get("beta", config["EVAL_KWARGS"]["beta"]))
                argmax = bool(xp_cfg.get("argmax", config["EVAL_KWARGS"]["argmax"]))
                num_actors_eval = eval_env.num_agents * num_eval_envs

                eval_rng, reset_rng = jax.random.split(eval_rng)
                reset_rngs = jax.random.split(reset_rng, num_eval_envs)
                init_obs, init_state = jax.vmap(eval_env.reset, in_axes=(0,))(reset_rngs)
                init_hstate_1 = ScannedRNN.initialize_carry(num_actors_eval, config["GRU_HIDDEN_DIM"])
                init_hstate_2 = ScannedRNN.initialize_carry(num_actors_eval, config["GRU_HIDDEN_DIM"])
                init_done = jnp.zeros((num_actors_eval,), dtype=bool)
                init_returns = jnp.zeros((num_eval_envs,), dtype=jnp.float32)
                runner_state = (init_state, init_obs, init_done, init_hstate_1, init_hstate_2, init_returns, eval_rng)

                def _xp_step(carry, _):
                    env_state_e, obs_e, done_e, hstate_1, hstate_2, returns_e, rng_e = carry

                    rng_e, rng_1, rng_2 = jax.random.split(rng_e, 3)
                    obs_batch = batchify(obs_e, eval_env.agents, num_actors_eval)
                    agent_positions = {'agent_0': env_state_e.env_state.agent_pos, 'agent_1': env_state_e.env_state.agent_pos}
                    agent_positions = batchify(agent_positions, eval_env.agents, num_actors_eval)
                    ac_in = (
                        obs_batch[np.newaxis, :],
                        done_e[np.newaxis, :],
                        agent_positions[np.newaxis, :],
                    )
                    hstate_1_next, pi_1, _ = network.apply(params_1, hstate_1, ac_in)
                    hstate_2_next, pi_2, _ = network.apply(params_2, hstate_2, ac_in)
                    pi_1 = distrax.Categorical(logits=pi_1.logits * beta)
                    pi_2 = distrax.Categorical(logits=pi_2.logits * beta)
                    sampled_1 = pi_1.sample(seed=rng_1)[0]
                    sampled_2 = pi_2.sample(seed=rng_2)[0]
                    greedy_1 = jnp.argmax(pi_1.probs, axis=-1)[0]
                    greedy_2 = jnp.argmax(pi_2.probs, axis=-1)[0]
                    action_1 = jnp.where(argmax, greedy_1, sampled_1)
                    action_2 = jnp.where(argmax, greedy_2, sampled_2)
                    action = jnp.concatenate(
                        [action_1[:num_eval_envs], action_2[num_eval_envs:]],
                        axis=0,
                    )

                    env_act = unbatchify(action, eval_env.agents, num_eval_envs, eval_env.num_agents)
                    env_act = {k: v.squeeze() for k, v in env_act.items()}

                    rng_e, _rng_e = jax.random.split(rng_e)
                    rng_step_e = jax.random.split(_rng_e, num_eval_envs)
                    obs_next, state_next, reward, done, _info = jax.vmap(
                        eval_env.step, in_axes=(0, 0, 0)
                    )(rng_step_e, env_state_e, env_act)

                    done_next = batchify(done, eval_env.agents, num_actors_eval).squeeze()
                    returns_next = returns_e + reward["agent_0"]

                    return (state_next, obs_next, done_next, hstate_1_next, hstate_2_next, returns_next, rng_e), None

                runner_state, _ = jax.lax.scan(_xp_step, runner_state, None, num_steps)
                state, obs, done, h_state_1, h_state_2, returns, rng = runner_state
                return returns.mean()

            run_eval = jnp.equal(update_steps % config["EVAL_KWARGS"]["eval_interval"], 0)
            run_xp = bool(config.get("XP_KWARGS", {}).get("enabled", False)) and xp_partner_params is not None

            if len(eval_envs) > 0:
                def _do_eval(_):
                    out = {}
                    base = jax.random.fold_in(rng, update_steps)
                    for i, layout_name in enumerate(layout_names):
                        if run_xp:
                            out[layout_name] = xp_layout(
                                eval_envs[layout_name],
                                train_state.params,
                                xp_partner_params,
                                jax.random.fold_in(base, i),
                            )
                        else:
                            out[layout_name] = eval_layout(
                                eval_envs[layout_name],
                                train_state.params,
                                jax.random.fold_in(base, i),
                            )
                    out["mean"] = jnp.mean(jnp.stack([out[n] for n in layout_names]))
                    return out

                def _skip_eval(_):
                    out = {n: jnp.array(jnp.nan, dtype=jnp.float32) for n in layout_names}
                    out["mean"] = jnp.array(jnp.nan, dtype=jnp.float32)
                    return out

                metric["eval_returns"] = jax.lax.cond(run_eval, _do_eval, _skip_eval, operand=None)
                metric["using_xp_eval"] = jnp.array(run_xp, dtype=jnp.bool_)

            def callback(metric):
                log_dict = {
                    "returns": metric["returns"],
                    "normalized_returns": metric["normalized_returns"],
                    "success_rate": metric["success_rate"],
                    "env_step": int(metric["update_steps"] * config["NUM_ENVS"] * config["NUM_STEPS"]),
                    **metric["loss"],
                }
                if "eval_returns" in metric:
                    if np.isfinite(float(metric["eval_returns"]["mean"])):
                        metric_prefix = "xp" if bool(metric.get("using_xp_eval", False)) else "eval"
                        if metric_prefix == "xp":
                            return_scale = 2.0 * float(config["ENV_KWARGS"]["max_steps"])
                            mean_return = float(metric["eval_returns"]["mean"])
                            log_dict["xp/return_mean_fixed_heldout"] = mean_return
                            log_dict["xp/normalized_return_mean_fixed_heldout"] = mean_return / return_scale
                            for _ln in layout_names:
                                layout_return = float(metric["eval_returns"][_ln])
                                log_dict[f"xp/return_{_ln}"] = layout_return
                                log_dict[f"xp/normalized_return_{_ln}"] = layout_return / return_scale
                        else:
                            log_dict["eval/return_mean_fixed_heldout"] = float(metric["eval_returns"]["mean"])
                            for _ln in layout_names:
                                log_dict[f"eval/return_{_ln}"] = float(metric["eval_returns"][_ln])

                if config["ENV_NAME"] == "overcooked":
                    ep_rets = np.array(metric["episode_returns_step"])   # (NUM_STEPS, NUM_ENVS)
                    ep_done = np.array(metric["episode_done_step"]).astype(bool)
                    layout_ids = np.array(metric["layout_ids"])  # (NUM_STEPS, NUM_ENVS), pre-step layout
                    layout_returns = {name: [] for name in EVAL_LAYOUTS_9}
                    for t in range(ep_done.shape[0]):
                        for e in range(ep_done.shape[1]):
                            if ep_done[t, e]:
                                label = EVAL_LAYOUTS_9[int(layout_ids[t, e])]
                                layout_returns[label].append(float(ep_rets[t, e]))
                    for name in EVAL_LAYOUTS_9:
                        returns_for_layout = layout_returns[name]
                        log_dict[f"train_returns/{name}"] = (
                            float(np.mean(returns_for_layout))
                            if len(returns_for_layout) > 0
                            else float("nan")
                        )
                wandb.log(log_dict)
            metric["returns"] = returns
            metric["normalized_returns"] = normalized_returns
            metric["success_rate"] = success_rate
            metric["update_steps"] = update_steps

            callback_metric = {
                **metric,
                "episode_returns_step": episode_returns_step,
                "episode_done_step": episode_done_step,
                "layout_ids": _layout_ids_full,
            }
            jax.experimental.io_callback(callback, None, callback_metric)

            def ckpt_callback(params, opt_state_, tx_step, step, mu, sigma):
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
                        'popart_mu': mu,
                        'popart_sigma': sigma,
                    }, f)

            save_ckpt_interval = int(config.get("SAVE_CKPT_INTERVAL", 5000))
            if save_ckpt_interval > 0:
                run_save_ckpt = jnp.equal(update_steps % save_ckpt_interval, 0)
                jax.lax.cond(
                    run_save_ckpt,
                    lambda _: jax.experimental.io_callback(ckpt_callback, None, train_state.params, train_state.opt_state, train_state.step, update_steps, popart_mu, popart_sigma),
                    lambda _: None,
                    operand=None,
                )

            if save_info is not None:
                num_updates_total = save_info["num_updates"]
                def final_save_callback(params, step, mu, sigma):
                    fp = save_info["filepath"]
                    prefix = save_info["fcp_prefix"]
                    appendage = save_info["finetune_appendage"]
                    rng_key = save_info["rng"]
                    os.makedirs(fp, exist_ok=True)
                    ckpt_path = f"{fp}/{prefix}seed{config['SEED']}_ckpt{config['TRAIN_KWARGS']['ckpt_id']}{appendage}_pop_updates{num_updates_total}.pkl"
                    with open(ckpt_path, "wb") as f:
                        pickle.dump({'key': rng_key, 'params': params, 'update_steps': num_updates_total,
                                     'popart_mu': mu, 'popart_sigma': sigma}, f)
                    print(f"Saved final model to {ckpt_path}")
                    print(f"Finished training for seed {config['SEED']} with ckpt {config['TRAIN_KWARGS']['ckpt_id']}_updates{num_updates_total}")
                    print(f"--------------------------------")

                is_last_step = jnp.equal(update_steps, num_updates_total - 1)
                jax.lax.cond(
                    is_last_step,
                    lambda _: jax.experimental.io_callback(final_save_callback, None, train_state.params, update_steps, popart_mu, popart_sigma),
                    lambda _: None,
                    operand=None,
                )

            update_steps = update_steps + 1
            runner_state = (train_state, env_state, last_obs, last_done, hstate, rng)  # hstate resets automatically
            return (runner_state, update_steps, popart_mu, popart_sigma), metric

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
            _update_step, (runner_state, update_step, popart_mu, popart_sigma), jnp.arange(remaining_updates), remaining_updates
        )
        return {"runner_state": runner_state}

    return train


@hydra.main(version_base=None, config_path="config", config_name="ippo_overcooked_CEC_dual_destination_popart_with_xp")
def main(config):
    config = OmegaConf.to_container(config)
    config['model_name'] = "CEC_POPART"
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

    resume_xpid = config.get("RESUME_XPID", "")
    active_xpid = resume_xpid if resume_xpid else xpid

    if config["ENV_NAME"] == "ToyCoopNoPink":
        filepath_base = modified_wall_ckpt_root(config)
    else:
        filepath_base = f"ckpts/ippo/{config['ENV_NAME']}"
    if config["ENV_NAME"] == "overcooked":
        filepath_base += f"/{config['ENV_KWARGS']['layout']}"
    ckpt_group = "cec_popart_layout_eval" if config["ENV_KWARGS"]["random_reset"] else "ippo_popart_layout_eval"
    filepath_base += f"/ik{config['ENV_KWARGS']['random_reset']}/{config['ENV_KWARGS']['random_reset_fn']}/{ckpt_group}"
    filepath = f"{filepath_base}/{active_xpid}"
    print(f"Working on: \n{filepath}\n")

    config['MID_CKPT_DIR'] = os.path.join(filepath, f"seed{config['SEED']}_mid_ckpts")

    mid_ckpt_path = os.path.join(config['MID_CKPT_DIR'], "resume_ckpt.pkl")
    _has_mid_ckpt = bool(resume_xpid) and os.path.exists(mid_ckpt_path)
    wandb_resume_id = None
    if _has_mid_ckpt:
        with open(mid_ckpt_path, "rb") as f:
            _peek = pickle.load(f)
        wandb_resume_id = _peek.get('wandb_run_id', None)

    layout_name = get_wall_map_name(config) if config["ENV_NAME"] == "ToyCoopNoPink" else config["ENV_KWARGS"]["layout"]
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
            tags=["IPPO", "RNN", "SP"],
            config=config,
            mode=config["WANDB_MODE"],
            name=f"CEC_POPART_LAYOUT_EVAL_modified_wall_{layout_name}_seed{config['SEED']}" if config["ENV_NAME"] == "ToyCoopNoPink" else f"CEC_gradient_pop_layout_eval_{layout_name}_seed{config['SEED']}"
        )

    if not config['TRAIN_KWARGS']['overwrite_ckpt']:
        # check if ckpt exists
        if os.path.exists(f"{filepath}/{fcp_prefix}seed{config['SEED']}_ckpt{config['TRAIN_KWARGS']['ckpt_id']}{finetune_appendage}.pkl"):
            print(f"Checkpoint {config['TRAIN_KWARGS']['ckpt_id']} already exists, exiting")
            exit(0)

    init_popart_mu = None
    init_popart_sigma = None
    if _has_mid_ckpt:
        print(f"Found mid-run checkpoint: {mid_ckpt_path}")
        model_params = _peek['params']
        opt_state = _peek.get('opt_state', None)
        final_update_step = _peek['final_update_step']
        rng = jax.random.PRNGKey(config["SEED"])
        init_popart_mu = _peek.get('popart_mu', None)
        init_popart_sigma = _peek.get('popart_sigma', None)
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
            finetune_ckpt_num = 19 if config['ENV_NAME'] == 'ToyCoopNoPink' else 6
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

    num_updates = int(config["TOTAL_TIMESTEPS"] // config["NUM_STEPS"] // config["NUM_ENVS"])
    save_info = {
        "filepath": filepath,
        "fcp_prefix": fcp_prefix,
        "finetune_appendage": finetune_appendage,
        "rng": rng,
        "num_updates": num_updates,
    }

    if init_popart_mu is None:
        init_popart_mu = jnp.zeros(())
    if init_popart_sigma is None:
        init_popart_sigma = jnp.ones(())

    xp_partner_params = None
    if bool(config.get("XP_KWARGS", {}).get("enabled", False)):
        xp_partner_params, xp_partner_path = load_xp_partner_params(config)
        config["XP_KWARGS"]["partner_path"] = xp_partner_path

    print(f"Starting from update step {final_update_step}")
    train_jit = jax.jit(make_train(config, final_update_step, save_info, opt_state, xp_partner_params), device=jax.devices()[0])
    out = train_jit(rng, model_params, init_popart_mu, init_popart_sigma)

    jax.effects_barrier()
    jax.clear_caches()
    wandb.finish()

if __name__ == "__main__":
    main()
