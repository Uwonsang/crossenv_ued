
import jax.numpy as jnp
from jaxmarl.environments.overcooked.common import (
    OBJECT_TO_INDEX,
    COLOR_TO_INDEX,
)
from jaxmarl.environments.overcooked.overcooked import POT_EMPTY_STATUS


def restore_from_obs(obs, agent_view_size=5):
    """
    obs: (H, W, 26) agent_0 기준
    returns: agent_dir_idx (scalar), agent_inv (scalar), maze_map (H+pad*2, W+pad*2, 3)
    """
    H, W = obs.shape[:2]
    padding = agent_view_size - 1  # 4

    # agent_dir_idx
    agent_0_dir_layers = obs[:, :, 2:6]
    agent_0_dir_idx = jnp.argmax(jnp.sum(agent_0_dir_layers, axis=(0, 1)))

    agent_1_dir_layers = obs[:, :, 6:10]
    agent_1_dir_idx = jnp.argmax(jnp.sum(agent_1_dir_layers, axis=(0, 1)))

    agent_dir_idx = jnp.array([agent_0_dir_idx, agent_1_dir_idx])


    # agent_inv (empty for both agents)
    agent_inv =  jnp.array([OBJECT_TO_INDEX['empty'], OBJECT_TO_INDEX['empty']])

    # maze_map channel 0: object type (H, W)
    obj_map = jnp.zeros((H, W), dtype=jnp.uint8)
    obj_map = jnp.where(obs[:, :, 11], OBJECT_TO_INDEX['wall'],       obj_map)
    obj_map = jnp.where(obs[:, :, 10], OBJECT_TO_INDEX['pot'],        obj_map)
    obj_map = jnp.where(obs[:, :, 12], OBJECT_TO_INDEX['onion_pile'], obj_map)
    obj_map = jnp.where(obs[:, :, 14], OBJECT_TO_INDEX['plate_pile'], obj_map)
    obj_map = jnp.where(obs[:, :, 15], OBJECT_TO_INDEX['goal'],       obj_map)
    obj_map = jnp.where(obs[:, :, 23], OBJECT_TO_INDEX['onion'],      obj_map)
    obj_map = jnp.where(obs[:, :, 22], OBJECT_TO_INDEX['plate'],      obj_map)
    obj_map = jnp.where(obs[:, :, 21], OBJECT_TO_INDEX['dish'],       obj_map)
    obj_map = jnp.where(obs[:, :,  0], OBJECT_TO_INDEX['agent'],      obj_map)
    obj_map = jnp.where(obs[:, :,  1], OBJECT_TO_INDEX['agent'],      obj_map)

    # maze_map channel 1: color
    agent_0_pos = jnp.argwhere(obs[:, :, 0], size=1)[0]
    y0, x0 = agent_0_pos[0], agent_0_pos[1]
    agent_1_pos = jnp.argwhere(obs[:, :, 1], size=1)[0]
    y1, x1 = agent_1_pos[0], agent_1_pos[1]

    
    color_map = jnp.zeros((H, W), dtype=jnp.uint8)
    color_map = color_map.at[y0, x0].set(COLOR_TO_INDEX['red'])
    color_map = color_map.at[y1, x1].set(COLOR_TO_INDEX['blue'])

    # maze_map channel 2: pot status
    pot_status_map = jnp.zeros((H, W), dtype=jnp.uint8)
    pot_status_map = jnp.where(obs[:, :, 10],
                               POT_EMPTY_STATUS - obs[:, :, 16] - obs[:, :, 20],
                               pot_status_map)

    # (H, W, 3)
    maze_map = jnp.stack([obj_map, color_map, pot_status_map], axis=-1)

    pad_value_obj = OBJECT_TO_INDEX['wall']
    padded_maze_map = jnp.pad(
        maze_map,
        pad_width=((padding, padding), (padding, padding), (0, 0)),
        mode='constant',
        constant_values=0
    )

    padded_color = jnp.pad(
        color_map,
        pad_width=((padding, padding), (padding, padding)),
        mode='constant',
        constant_values=0)

    padded_maze_map = padded_maze_map.at[:, :, 1].set(padded_color)

    padded_obj = jnp.pad(
        obj_map,
        pad_width=((padding, padding), (padding, padding)),
        mode='constant',
        constant_values=pad_value_obj
    )
    padded_maze_map = padded_maze_map.at[:, :, 0].set(padded_obj)

    state = {
        "agent_dir_idx": agent_dir_idx,
        "agent_inv": agent_inv,
        "maze_map": padded_maze_map,
    }

    return state

if __name__ == "__main__":
    obs = jnp.load('/app/baselines/CEC_UED/VAE/dataset/env_states_3e6_lr-20260224-071423.h5')
    agent_dir_idx, agent_inv, maze_map = restore_from_obs(obs)
    print(agent_dir_idx)
    print(agent_inv)
    print(maze_map)