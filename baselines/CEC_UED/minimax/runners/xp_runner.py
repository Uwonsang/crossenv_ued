"""
Copyright (c) Meta Platforms, Inc. and affiliates.
All rights reserved.

This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.
"""

from functools import partial
import os
import time

import imageio
import numpy as np
import jax
import jax.numpy as jnp
from jax.experimental import mesh_utils
from jax.experimental.shard_map import shard_map


from .eval_runner import EvalRunner
from .dr_runner import DRRunner
from .plr_runner import PLRRunner
from minimax.model import ActorCriticRNN
import minimax.agents as agents

import jaxmarl
from jaxmarl.wrappers.baselines import LogWrapper
from jaxmarl.environments.overcooked import overcooked_layouts
from jaxmarl.environments.overcooked.layouts import make_counter_circuit_9x9, make_forced_coord_9x9, make_coord_ring_9x9, make_asymm_advantages_9x9, make_cramped_room_9x9
from jaxmarl.viz.overcooked_visualizer import OvercookedVisualizer
from jax_tqdm import scan_tqdm


def initialize_environment(config):
    layout_name = config["ENV_KWARGS"]["layout"]
    config['layout_name'] = layout_name
    config["ENV_KWARGS"]["layout"] = overcooked_layouts[layout_name]
    env = jaxmarl.make(config["ENV_NAME"], **config["ENV_KWARGS"])

    if "overcooked" in config["ENV_NAME"]:
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


class RunnerInfo:
    def __init__(
            self,
            runner_cls):
        self.runner_cls = runner_cls


RUNNER_INFO = {
    'dr': RunnerInfo(
        runner_cls=DRRunner,
    ),
    'plr': RunnerInfo(
        runner_cls=PLRRunner,
    )
}


class ExperimentRunner:
    def __init__(
            self,
            config,
            train_runner,
            env_name,
            n_devices=1,
            xpid=None
    ):
        self.env_name = env_name
        self.xpid = xpid

        env = initialize_environment(config)
        obs, state = env.reset(jax.random.PRNGKey(0), params={'random_reset_fn': config['ENV_KWARGS']['random_reset_fn']})
        env = LogWrapper(env, env_params={'random_reset_fn': config['ENV_KWARGS']['random_reset_fn']})

        config["NUM_ACTORS"] = env.num_agents * config["NUM_ENVS"]
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

        # ---- Make agent ----
        network = ActorCriticRNN(env.action_space(env.agents[0]).n, config=config)

        student_agent = agents.IPPOAgent(
            model=network,
            config=config,
            obs_dim = env.observation_space(env.agents[0]).shape,
            n_devices=n_devices
        )
        
        # ---- Set up train runner ----
        runner_cls = RUNNER_INFO[train_runner].runner_cls

        # Set up learning rate annealing parameters
        lr_init = train_runner_kwargs.lr
        lr_final = train_runner_kwargs.lr_final
        lr_anneal_steps = train_runner_kwargs.lr_anneal_steps

        if lr_final is None:
            train_runner_kwargs.lr_final = lr_init
        if train_runner_kwargs.lr_final == train_runner_kwargs.lr:
            train_runner_kwargs.lr_anneal_steps = 0

        use_shaped_reward = (shaped_reward_steps is not None and shaped_reward_steps > 0) or (
            shaped_reward_n_updates is not None and shaped_reward_n_updates > 0)

        self.runner = runner_cls(
            env_name=env_name,
            env_kwargs=env_kwargs,
            student_agents=[student_agent],
            n_devices=n_devices,
            shaped_reward=use_shaped_reward,
            **train_runner_kwargs)

        # # ---- Make eval runner ----
        # if eval_kwargs.get('env_names') is None:
        #     self.eval_runner = None
        # else:
        #     self.eval_runner = EvalRunner(
        #         pop=self.runner.student_pop,
        #         env_kwargs=eval_env_kwargs,
        #         **eval_kwargs)

        self._start_tick = 0

        # ---- Set up device parallelism ----
        self.n_devices = n_devices
        if n_devices > 1:
            dummy_runner_state = self.reset_train_runner(jax.random.PRNGKey(0))
            self._shmap_run = self._make_shmap_run(dummy_runner_state)
        else:
            self._shmap_run = None

    @partial(jax.jit, static_argnums=(0,))
    def step(self, runner_state, evaluate=False):
        if self.n_devices > 1:
            run_fn = self._shmap_run
        else:
            run_fn = self.runner.run

        stats, *runner_state = run_fn(*runner_state)

        rng = runner_state[0]
        rng, subrng = jax.random.split(rng)

        if self.eval_runner is not None:
            params = runner_state[1].actor_params
            eval_stats = jax.lax.cond(
                evaluate,
                self.eval_runner.run,
                self.eval_runner.fake_run,
                *(subrng, params)
            )
        else:
            eval_stats = {}

        return stats, eval_stats, rng, *runner_state[1:]

    def _make_shmap_run(self, runner_state):
        devices = mesh_utils.create_device_mesh((self.n_devices,))
        mesh = Mesh(devices, axis_names=('device'))

        in_specs, out_specs = self.runner.get_shmap_spec()

        return partial(shard_map,
                       mesh=mesh,
                       in_specs=in_specs,
                       out_specs=out_specs,
                       check_rep=False
                       )(self.runner.run)

    def train(
            self,
            rng,
            agent_algo='ppo',
            algo_runner='dr',
            n_total_updates=1000,
            logger=None,
            log_interval=1,
            test_interval=1,
            checkpoint_interval=0,
            archive_interval=0,
            archive_init_checkpoint=False,
            from_last_checkpoint=False
    ):
        """
        Entry-point for training
        """
        # Load checkpoint if any
        runner_state = self.runner.reset(rng)

        if from_last_checkpoint:
            last_checkpoint_state = logger.load_last_checkpoint_state()
            if last_checkpoint_state is not None:
                runner_state = self.runner.load_checkpoint_state(
                    runner_state,
                    last_checkpoint_state
                )
                self._start_tick = runner_state[1].n_iters[0]

        # Archive initialization weights if necessary
        if archive_init_checkpoint:
            logger.checkpoint(
                self.runner.get_checkpoint_state(runner_state),
                index=0,
                archive_interval=1)

        # Train loop
        log_on = logger is not None and log_interval > 0
        checkpoint_on = checkpoint_interval > 0 or archive_interval > 0
        train_state = runner_state[1]

        tick = self._start_tick
        train_steps = tick*self.runner.step_batch_size * \
            self.runner.n_rollout_steps*self.n_devices
        real_train_steps = train_steps//self.runner.n_students

        while (train_state.n_updates < n_total_updates).any():
            evaluate = test_interval > 0 and (tick+1) % test_interval == 0

            start = time.time()
            stats, eval_stats, *runner_state = \
                self.step(runner_state, evaluate)
            end = time.time()

            start_state = runner_state[-1]
            runner_state = runner_state[:-1]

            if evaluate:
                stats.update(eval_stats)
            else:
                stats.update({k: None for k in eval_stats.keys()})

            train_state = runner_state[1]

            dsteps = self.runner.step_batch_size*self.runner.n_rollout_steps*self.n_devices
            real_dsteps = dsteps//self.runner.n_students
            train_steps += dsteps
            real_train_steps += real_dsteps

            if (self.shaped_reward_steps is not None and self.shaped_reward_steps > 0) or (self.shaped_reward_n_updates is not None and self.shaped_reward_n_updates > 0):
                if self.shaped_reward_n_updates:  # Meassure based on n_updates
                    new_shaped_reward_coeff_value = max(
                        0.0, 1.0 - (train_state.n_updates[0]/self.shaped_reward_n_updates))
                else:  # Meassure based on steps in the env
                    new_shaped_reward_coeff_value = max(
                        0.0, 1.0 - (real_train_steps/self.shaped_reward_steps))

                new_shaped_reward_coeff = jnp.full(
                    runner_state[1].shaped_reward_coeff.shape, fill_value=new_shaped_reward_coeff_value)
                jax.debug.print("Shaped reward coeff: {a}, real_dsteps: {b}, shaped_reward_steps: {c}",
                                a=new_shaped_reward_coeff, b=real_dsteps, c=self.shaped_reward_steps)
                # runner_state[1] is the training state object where the shaped reward coefficient is stored
                runner_state[1] = runner_state[1].set_new_shaped_reward_coeff(
                    new_shaped_reward_coeff)

            sps = int(dsteps/(end-start))
            real_sps = int(real_dsteps/(end-start))
            time_per_iter = float(end-start)
            stats.update(dict(
                steps=train_steps,
                sps=sps,
                real_steps=real_train_steps,
                real_sps=real_sps,
                time_per_iter=time_per_iter,
            ))

            tick += 1

            jax.debug.print("-----\n{stats}", stats=stats)
            if log_on and tick % log_interval == 0:
                logger.log(stats, tick, ignore_val=-np.inf)

            if checkpoint_on and tick > 0:
                if tick % checkpoint_interval == 0 \
                        or (archive_interval > 0 and tick % archive_interval == 0):

                    maze_map = start_state.maze_map
                    agent_dir_idx = start_state.agent_dir_idx
                    agent_inv = start_state.agent_inv
                    for i in range(1):  # self.runner.n_parallel

                        padding = 4 #TODO : check this
                        grid = np.asarray(
                            maze_map[0, i, padding:-padding, padding:-padding, :])
                        # Render the state
                        frame = OvercookedVisualizer._render_grid(
                            grid,
                            tile_size=32,
                            highlight_mask=None,
                            agent_dir_idx=agent_dir_idx[0][i],
                            agent_inv=agent_inv[0][i]
                        )

                        # Save the numpy frame as image
                        dir = f"{os.getcwd()}/overcooked_teacher_layout_imgs/{self.xpid}/"

                        os.makedirs(os.path.dirname(dir), exist_ok=True)
                        imageio.imwrite(
                            dir + f"{tick}_{i}.png", frame)

                    # Also produce an image of the teachers env output currently
                    checkpoint_state = \
                        self.runner.get_checkpoint_state(runner_state)
                    logger.checkpoint(
                        checkpoint_state,
                        index=tick,
                        archive_interval=archive_interval)
