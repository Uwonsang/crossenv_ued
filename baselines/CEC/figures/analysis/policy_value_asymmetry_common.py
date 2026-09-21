"""Shared state construction and planning for the asymmetry analysis."""
from __future__ import annotations

from collections import deque


FAMILIES = (
    "asymm_advantages",
    "coord_ring",
    "counter_circuit",
    "forced_coord",
    "cramped_room",
)


def delivery_plan(floor, goals, start, directions, initial_direction):
    """BFS preserving all equally optimal first actions (partner is an obstacle)."""
    queue = deque([(tuple(start), initial_direction, [])])
    seen = set()
    solutions = []
    best = None
    while queue:
        pos, facing, actions = queue.popleft()
        if best is not None and len(actions) + 1 > best:
            break
        key = (pos, facing, actions[0] if actions else -1)
        if key in seen:
            continue
        seen.add(key)
        dx, dy = directions[facing]
        if (pos[0] + dx, pos[1] + dy) in goals:
            best = len(actions) + 1
            solutions.append(actions + [5])
            continue
        for action, (dx, dy) in enumerate(directions):
            target = (pos[0] + dx, pos[1] + dy)
            queue.append((target if target in floor else pos, action, actions + [action]))
    return (solutions[0], sorted({p[0] for p in solutions})) if solutions else (None, [])


def initial_state(config, record, horizon):
    """Reconstruct one saved controlled state in the Overcooked environment."""
    import jax
    import jax.numpy as jnp
    import jaxmarl
    from jaxmarl.environments.overcooked.common import DIR_TO_VEC, OBJECT_TO_INDEX

    layout = {
        key: value if key in ("height", "width") else jnp.asarray(value)
        for key, value in record["layout"].items()
    }
    kwargs = dict(config["ENV_KWARGS"])
    kwargs.update(
        layout=layout,
        random_reset=False,
        check_held_out=False,
        shuffle_inv_and_pot=False,
        max_steps=horizon,
    )
    env = jaxmarl.make("overcooked", **kwargs)
    _, state = env.custom_reset(
        jax.random.PRNGKey(record["reset_seed"]),
        layout=layout,
        random_reset=False,
        shuffle_inv_and_pot=False,
    )
    facing = record["direction"]
    state = state.replace(
        agent_dir_idx=jnp.array([facing, facing]),
        agent_dir=jnp.stack([DIR_TO_VEC[facing], DIR_TO_VEC[facing]]),
        agent_inv=jnp.array([
            OBJECT_TO_INDEX["dish"], OBJECT_TO_INDEX["empty"],
        ]),
    )
    pad = (state.maze_map.shape[0] - int(layout["height"])) // 2
    maze = state.maze_map
    for x, y in state.agent_pos.tolist():
        maze = maze.at[pad + y, pad + x, 2].set(facing)
    return env, state.replace(maze_map=maze)
