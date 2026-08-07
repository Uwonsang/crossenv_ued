from functools import partial
from typing import Dict, Tuple

import chex
import jax
import jax.numpy as jnp
from flax import struct

from jaxmarl.environments import spaces
from .modified_wall_toy_coop import Actions, ModifiedWallToyCoop


@struct.dataclass
class State:
    agent_pos: chex.Array
    goal_pos: chex.Array
    wall_map: chex.Array
    time: int
    terminal: bool


class ToyCoopNoPink(ModifiedWallToyCoop):
    """Dual Destination with exactly one pair of identical goal cells."""

    def __init__(
        self,
        max_steps: int = 100,
        random_reset: bool = False,
        debug: bool = False,
        check_held_out: bool = False,
        partial_obs: bool = False,
        incentivize_strat: int = 2,
        map_name: str = "empty",
        layout_names=None,
    ):
        del incentivize_strat
        super().__init__(
            max_steps=max_steps,
            random_reset=random_reset,
            debug=debug,
            check_held_out=check_held_out,
            partial_obs=partial_obs,
            incentivize_strat=2,
            map_name=map_name,
            layout_names=layout_names,
        )
        self.held_out_agent_pos = None
        self.held_out_goal_pos = None
        self.held_out_wall_map = None

    def _matches_held_out(self, state):
        agent_match = jax.vmap(lambda pos: jnp.all(pos == state.agent_pos))(
            self.held_out_agent_pos
        )
        def canonical_goals(pos):
            flat = pos[:, 1] * self.width + pos[:, 0]
            return jnp.sort(flat)

        state_goals = canonical_goals(state.goal_pos)
        goal_match = jax.vmap(
            lambda pos: jnp.all(canonical_goals(pos) == state_goals)
        )(self.held_out_goal_pos)
        wall_match = jax.vmap(lambda wall: jnp.all(wall == state.wall_map))(
            self.held_out_wall_map
        )
        return jnp.any(agent_match & goal_match & wall_match)

    def reset(
        self, key: chex.PRNGKey, params={"random_reset_fn": "reset_all"}
    ) -> Tuple[Dict[str, chex.Array], State]:
        del params
        has_held_out = (
            self.held_out_agent_pos is not None
            and self.held_out_goal_pos is not None
            and self.held_out_wall_map is not None
        )

        first_key, retry_key = jax.random.split(key)
        state = self.custom_reset_fn(
            first_key, random_reset=self.random_reset, debug=self.debug
        )
        if self.random_reset and self.check_held_out and has_held_out:
            state = jax.lax.cond(
                self._matches_held_out(state),
                lambda rng: self.custom_reset_fn(
                    rng, random_reset=self.random_reset, debug=self.debug
                ),
                lambda rng: state,
                retry_key,
            )

        return self.get_obs(state), state

    def custom_reset_fn(self, key, random_reset=False, debug=False):
        del debug
        layout_key, position_key = jax.random.split(key)
        sampled_layout_idx = jax.random.randint(
            layout_key, (), minval=0, maxval=self.num_layouts
        )
        layout_idx = jnp.where(self.mixed_layout & random_reset, sampled_layout_idx, 0)
        wall_map = self.wall_maps[layout_idx]
        default_agent_pos = self.default_agent_positions[layout_idx]
        default_goal_pos = self.default_goal_positions[layout_idx]

        scores = jax.random.uniform(position_key, (self.height * self.width,))
        scores = jnp.where(self.free_masks[layout_idx], scores, -1.0)
        locations = self.all_pos[jnp.argsort(scores)[-4:]]

        return State(
            agent_pos=jnp.where(random_reset, locations[:2], default_agent_pos),
            goal_pos=jnp.where(random_reset, locations[2:], default_goal_pos),
            wall_map=wall_map,
            time=0,
            terminal=False,
        )

    @partial(jax.jit, static_argnums=[0])
    def step_agents(self, key, state: State, actions):
        del key
        next_pos = state.agent_pos + self.action_to_dir[actions]
        next_pos = jnp.clip(next_pos, 0, self.width - 1)
        hit_wall = state.wall_map[next_pos[:, 1], next_pos[:, 0]]
        next_pos = jnp.where(hit_wall[:, None], state.agent_pos, next_pos)
        collision = jnp.all(next_pos[0] == next_pos[1])
        next_pos = jnp.where(collision, state.agent_pos, next_pos)

        on_goal = jax.vmap(
            lambda pos: jnp.any(jnp.all(pos == state.goal_pos, axis=-1))
        )(next_pos)
        distinct_goals = ~jnp.all(next_pos[0] == next_pos[1])
        reward = jnp.float32(jnp.all(on_goal) & distinct_goals) * 3.0 - 1.0
        return state.replace(agent_pos=next_pos), reward

    @partial(jax.jit, static_argnums=[0])
    def get_obs(self, state: State) -> Dict[str, chex.Array]:
        obs = jnp.zeros((self.height, self.width, 4))
        obs = obs.at[state.agent_pos[0, 1], state.agent_pos[0, 0], 0].set(1)
        obs = obs.at[state.agent_pos[1, 1], state.agent_pos[1, 0], 1].set(1)
        obs = obs.at[state.goal_pos[:, 1], state.goal_pos[:, 0], 2].set(1)
        obs = obs.at[:, :, 3].set(state.wall_map.astype(obs.dtype))

        obs_1 = obs.at[:, :, 0].set(obs[:, :, 1])
        obs_1 = obs_1.at[:, :, 1].set(obs[:, :, 0])

        if self.partial_obs:
            obs = self._partial_obs(obs)
            obs_1 = self._partial_obs(obs_1)

        return {"agent_0": obs.reshape(-1), "agent_1": obs_1.reshape(-1)}

    def _partial_obs(self, obs):
        ego_y, ego_x = jnp.where(obs[:, :, 0] == 1, size=1)
        ys = jnp.arange(self.height)[:, None]
        xs = jnp.arange(self.width)[None, :]
        visible = (jnp.abs(ys - ego_y[0]) <= 1) & (jnp.abs(xs - ego_x[0]) <= 1)
        return jnp.where(visible[:, :, None], obs, -jnp.ones_like(obs))

    @property
    def name(self):
        return "ToyCoopNoPink"

    def observation_space(self, agent_id: str = "") -> spaces.Box:
        del agent_id
        return spaces.Box(0, 1, (self.height * self.width * 4,))
