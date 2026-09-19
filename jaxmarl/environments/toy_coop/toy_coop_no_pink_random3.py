from typing import Dict, Tuple

import chex
import jax
import jax.numpy as jnp

from .toy_coop_no_pink import State, ToyCoopNoPink


def _state_signature(state, width):
    def canonical_positions(positions):
        positions = jax.device_get(positions).tolist()
        return tuple(sorted(int(y) * width + int(x) for x, y in positions))

    wall_map = tuple(bool(value) for value in jax.device_get(state.wall_map).ravel())
    return (
        wall_map,
        canonical_positions(state.agent_pos),
        canonical_positions(state.goal_pos),
    )


def generate_unique_held_out_states(env, num_states, fixed_states, seed=0):
    """Generate deterministic valid states that exclude fixed evaluation tasks."""
    seen = {_state_signature(state, env.width) for state in fixed_states}
    held_out = []
    candidate_seed = int(seed)

    while len(held_out) < num_states:
        state = env.custom_reset_fn(
            jax.random.PRNGKey(candidate_seed), random_reset=True
        )
        candidate_seed += 1
        signature = _state_signature(state, env.width)
        if signature in seen:
            continue
        seen.add(signature)
        held_out.append(state)

    return jax.tree_util.tree_map(lambda *values: jnp.stack(values), *held_out)


class ToyCoopNoPinkRandom3(ToyCoopNoPink):
    """Dual Destination with three procedurally placed wall cells."""

    def __init__(
        self,
        max_steps: int = 100,
        random_reset: bool = False,
        debug: bool = False,
        check_held_out: bool = False,
        partial_obs: bool = False,
        incentivize_strat: int = 2,
        map_name: str = "random_3_walls",
        layout_names=None,
        randomize_walls: bool = True,
        num_random_walls: int = 3,
        heldout_ignore_agent_order: bool = True,
    ):
        requested_map_name = map_name
        base_map_name = "empty" if map_name == "random_3_walls" else map_name
        super().__init__(
            max_steps=max_steps,
            random_reset=random_reset,
            debug=debug,
            check_held_out=check_held_out,
            partial_obs=partial_obs,
            incentivize_strat=incentivize_strat,
            map_name=base_map_name,
            layout_names=layout_names,
        )
        if not 0 <= num_random_walls <= self.width * self.height - 4:
            raise ValueError(
                "num_random_walls must leave at least four cells for agents and goals"
            )
        self.map_name = requested_map_name
        self.randomize_walls = randomize_walls
        self.num_random_walls = num_random_walls
        self.heldout_ignore_agent_order = heldout_ignore_agent_order

    def _matches_held_out(self, state):
        def canonical_positions(pos):
            flat = pos[:, 1] * self.width + pos[:, 0]
            return jnp.sort(flat)

        if self.heldout_ignore_agent_order:
            state_agents = canonical_positions(state.agent_pos)
            agent_match = jax.vmap(
                lambda pos: jnp.all(canonical_positions(pos) == state_agents)
            )(self.held_out_agent_pos)
        else:
            agent_match = jax.vmap(lambda pos: jnp.all(pos == state.agent_pos))(
                self.held_out_agent_pos
            )

        state_goals = canonical_positions(state.goal_pos)
        goal_match = jax.vmap(
            lambda pos: jnp.all(canonical_positions(pos) == state_goals)
        )(self.held_out_goal_pos)
        wall_match = jax.vmap(lambda wall: jnp.all(wall == state.wall_map))(
            self.held_out_wall_map
        )
        return jnp.any(agent_match & goal_match & wall_match)

    def reset(
        self, key: chex.PRNGKey, params={"random_reset_fn": "reset_all"}
    ) -> Tuple[Dict[str, chex.Array], State]:
        del params
        key, sample_key = jax.random.split(key)
        state = self.custom_reset_fn(
            sample_key, random_reset=self.random_reset, debug=self.debug
        )
        has_held_out = (
            self.held_out_agent_pos is not None
            and self.held_out_goal_pos is not None
            and self.held_out_wall_map is not None
        )

        if self.random_reset and self.check_held_out and has_held_out:
            def needs_retry(carry):
                _, candidate = carry
                return self._matches_held_out(candidate)

            def retry(carry):
                rng, _ = carry
                rng, retry_key = jax.random.split(rng)
                candidate = self.custom_reset_fn(
                    retry_key, random_reset=True, debug=self.debug
                )
                return rng, candidate

            _, state = jax.lax.while_loop(needs_retry, retry, (key, state))

        return self.get_obs(state), state

    def custom_reset_fn(self, key, random_reset=False, debug=False):
        if not self.randomize_walls or not random_reset:
            return super().custom_reset_fn(
                key, random_reset=random_reset, debug=debug
            )
        return self._sample_valid_random_wall_state(key)

    def _sample_random_wall_candidate(self, key):
        wall_key, position_key = jax.random.split(key)
        wall_indices = jax.random.choice(
            wall_key,
            self.width * self.height,
            shape=(self.num_random_walls,),
            replace=False,
        )
        wall_map = jnp.zeros((self.width * self.height,), dtype=bool)
        wall_map = wall_map.at[wall_indices].set(True).reshape(
            (self.height, self.width)
        )

        scores = jax.random.uniform(position_key, (self.height * self.width,))
        scores = jnp.where(~wall_map.reshape(-1), scores, -1.0)
        locations = self.all_pos[jnp.argsort(scores)[-4:]]
        return State(
            agent_pos=locations[:2],
            goal_pos=locations[2:],
            wall_map=wall_map,
            time=0,
            terminal=False,
        )

    def _state_is_reachable(self, state):
        reachable = jnp.zeros_like(state.wall_map)
        start = state.agent_pos[0]
        reachable = reachable.at[start[1], start[0]].set(True)

        def expand(_, current):
            padded = jnp.pad(current, 1, constant_values=False)
            neighbors = (
                padded[:-2, 1:-1]
                | padded[2:, 1:-1]
                | padded[1:-1, :-2]
                | padded[1:-1, 2:]
            )
            return (current | neighbors) & ~state.wall_map

        reachable = jax.lax.fori_loop(
            0, self.width * self.height, expand, reachable
        )
        required = jnp.concatenate([state.agent_pos, state.goal_pos], axis=0)
        return jnp.all(reachable[required[:, 1], required[:, 0]])

    def _sample_valid_random_wall_state(self, key):
        key, sample_key = jax.random.split(key)
        state = self._sample_random_wall_candidate(sample_key)

        def invalid(carry):
            _, candidate = carry
            return ~self._state_is_reachable(candidate)

        def resample(carry):
            rng, _ = carry
            rng, sample_rng = jax.random.split(rng)
            return rng, self._sample_random_wall_candidate(sample_rng)

        _, state = jax.lax.while_loop(invalid, resample, (key, state))
        return state
