"""JAX port of Overcooked-AI ``featurize_state`` computed from a JaxMARL observation.

Reference: overcooked_ai_py/mdp/overcooked_mdp.py, OvercookedGridworld.featurize_state
(HumanCompatibleAI/overcooked_ai, num_pots=2). Output per agent is 96-dim:

  [p_self (46), p_other (46), other_pos - self_pos (2), self_pos (2)]

  p_k (46) = orientation one-hot (4)
           + held object one-hot [onion, soup, dish, tomato] (4)
           + closest {onion, tomato, dish, soup} deltas (8)
           + closest soup n_{onions, tomatoes} (2)
           + closest {serving, empty_counter} deltas (4)
           + 2 closest pots x [exists, is_empty, is_full, is_cooking, is_ready,
                               n_onions, n_tomatoes, cook_time, dx, dy] (20)
           + adjacent non-floor terrain [N, S, E, W] (4)

Differences from the reference (documented, unavoidable in JaxMARL):
  * "closest" uses BFS path length over walkable cells (stand next to the
    object) instead of the medium-level motion planner, which also counts turns.
  * JaxMARL has no tomatoes, so tomato entries are always 0.
  * Everything is computed from the 26-channel observation, so the observing
    agent is p_self (channel 0) and the partner is p_other (channel 1).
"""

import jax
import jax.numpy as jnp

INF = 1e6

# JaxMARL Overcooked observation channels (overcooked.py: get_obs)
SELF_POS, OTHER_POS = 0, 1
SELF_DIR, OTHER_DIR = slice(2, 6), slice(6, 10)
POT, COUNTER, ONION_PILE, PLATE_PILE, GOAL = 10, 11, 12, 14, 15
ONIONS_IN_POT, ONIONS_IN_SOUP, COOK_TIME, SOUP_READY = 16, 18, 20, 21
PLATE, ONION = 22, 23

# (dy, dx) for N, S, E, W (Overcooked-AI Direction.ALL_DIRECTIONS order)
NEIGHBORS = ((-1, 0), (1, 0), (0, 1), (0, -1))


def _shift(grid, dy, dx, fill):
    """out[y, x] = grid[y + dy, x + dx] (``fill`` outside the grid)."""
    h, w = grid.shape
    padded = jnp.pad(grid, 1, constant_values=fill)
    return padded[1 + dy:1 + dy + h, 1 + dx:1 + dx + w]


def _bfs_distance(walkable, start):
    """Shortest path length from ``start`` (one-hot) over walkable cells."""
    h, w = walkable.shape

    def relax(_, dist):
        neighbor = jnp.min(
            jnp.stack([_shift(dist, dy, dx, INF) for dy, dx in NEIGHBORS]), axis=0
        )
        return jnp.where(walkable, jnp.minimum(dist, neighbor + 1.0), INF)

    dist = jnp.where(start, 0.0, INF)
    return jax.lax.fori_loop(0, h * w, relax, dist)


def _position(onehot):
    """(x, y) of a one-hot grid."""
    y, x = jnp.unravel_index(jnp.argmax(onehot), onehot.shape)
    return jnp.array([x, y], dtype=jnp.float32)


def _closest(object_distance, mask):
    """Closest cell of ``mask`` -> (found, (x, y))."""
    dist = jnp.where(mask, object_distance, INF)
    y, x = jnp.unravel_index(jnp.argmin(dist), dist.shape)
    return dist[y, x] < INF, jnp.array([x, y], dtype=jnp.float32)


def _player_features(obs, pos_channel, dir_channels):
    me = obs[..., pos_channel] > 0
    agents = (obs[..., SELF_POS] + obs[..., OTHER_POS]) > 0
    pos = _position(me)
    walkable = (
        obs[..., POT] + obs[..., COUNTER] + obs[..., ONION_PILE]
        + obs[..., PLATE_PILE] + obs[..., GOAL]
    ) == 0
    walk_dist = _bfs_distance(walkable, me)
    # Interacting with a cell = standing on a walkable neighbor, +1 step.
    object_distance = jnp.min(
        jnp.stack([_shift(walk_dist, dy, dx, INF) for dy, dx in NEIGHBORS]), axis=0
    ) + 1.0

    def at_me(channel):
        return jnp.sum(obs[..., channel] * me)

    held_onion = at_me(ONION) > 0
    held_soup = at_me(SOUP_READY) > 0
    held_dish = at_me(PLATE) > 0
    orientation = jnp.sum(obs[..., dir_channels] * me[..., None], axis=(0, 1))
    held = jnp.array([held_onion, held_soup, held_dish, False], dtype=jnp.float32)

    pot = obs[..., POT] > 0
    counter_soup = (obs[..., SOUP_READY] > 0) & ~pot & ~agents
    counter_dish = (obs[..., PLATE] > 0) & ~agents
    counter_onion = (obs[..., ONION] > 0) & ~agents
    empty_counter = (obs[..., COUNTER] > 0) & (
        obs[..., SOUP_READY] + obs[..., PLATE] + obs[..., ONION] == 0
    )

    def delta(mask, holding=False):
        found, loc = _closest(object_distance, mask)
        return found, jnp.where(found & ~holding, loc - pos, 0.0)

    _, onion_delta = delta((obs[..., ONION_PILE] > 0) | counter_onion, held_onion)
    _, dish_delta = delta((obs[..., PLATE_PILE] > 0) | counter_dish, held_dish)
    soup_found, soup_delta = delta(counter_soup, held_soup)
    soup_onions = jnp.where(held_soup | soup_found, 3.0, 0.0)
    _, serving_delta = delta(obs[..., GOAL] > 0)
    _, empty_counter_delta = delta(empty_counter)

    def pot_features(mask):
        found, loc = _closest(object_distance, mask)
        x, y = loc.astype(jnp.int32)
        n_raw = obs[y, x, ONIONS_IN_POT]
        cook_time = obs[y, x, COOK_TIME]
        is_ready = obs[y, x, SOUP_READY] > 0
        is_cooking = cook_time > 0
        idle = ~is_cooking & ~is_ready
        features = jnp.concatenate([
            jnp.array([
                1.0,
                idle & (n_raw == 0),
                idle & (n_raw >= 3),
                is_cooking,
                is_ready,
                n_raw + obs[y, x, ONIONS_IN_SOUP],
                0.0,
                cook_time,
            ], dtype=jnp.float32),
            loc - pos,
        ])
        return jnp.where(found, features, 0.0), mask & ~(
            (jnp.arange(mask.shape[1])[None, :] == x)
            & (jnp.arange(mask.shape[0])[:, None] == y)
            & found
        )

    pot_0, remaining_pots = pot_features(pot)
    pot_1, _ = pot_features(remaining_pots)

    walls = jnp.stack([
        1.0 - jnp.sum(_shift(walkable.astype(jnp.float32), dy, dx, 0.0) * me)
        for dy, dx in NEIGHBORS
    ])
    features = jnp.concatenate([
        orientation.astype(jnp.float32),
        held,
        onion_delta,
        jnp.zeros(2),               # closest tomato
        dish_delta,
        soup_delta,
        jnp.array([soup_onions, 0.0]),
        serving_delta,
        empty_counter_delta,
        pot_0,
        pot_1,
        walls,
    ])
    return features, pos


def featurize_obs(obs):
    """[H, W, 26] JaxMARL observation -> [96] Overcooked-AI style features."""
    obs = obs.astype(jnp.float32)
    self_features, self_pos = _player_features(obs, SELF_POS, SELF_DIR)
    other_features, other_pos = _player_features(obs, OTHER_POS, OTHER_DIR)
    return jnp.concatenate([
        self_features, other_features, other_pos - self_pos, self_pos,
    ])
