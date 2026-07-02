import pickle

import jax
import jax.numpy as jnp

import jaxmarl
from jaxmarl.environments.overcooked import overcooked_layouts
from jaxmarl.viz.overcooked_visualizer import OvercookedVisualizer

import imageio
from tqdm import tqdm
import os


def load_pickle(file_path: str):
    with open(file_path, "rb") as f:
        data = pickle.load(f)
    plr_buffer = jax.device_put(jax.tree.map(jnp.array, data))
    return plr_buffer

def slice_levels(buffer, n=100):
    out= {}
    for k, v in buffer['sampled']["levels"].items():
        if v.ndim == 0:
            out[k] = v
            continue
        out[k] = v[:n]
    return out

def slot_to_layout(sliced_levels):
    layouts = []
    
    for idx in range(len(sliced_levels['width'])):
        out= {}
        for k, v in sliced_levels.items():
            row = v[idx]
            if k in ("height", "width"):
                out[k] = jnp.asarray(int(jnp.asarray(row).item()), dtype=jnp.int32)
            else:
                out[k] = jnp.asarray(row, dtype=jnp.int32)
        layouts.append(out)
        
    return layouts

def render_level_frame(env, viz, layout_dict, agent_view_size=5, seed=0):
    
    key = jax.random.PRNGKey(seed)
    _, state = env.custom_reset(
        key,
        random_reset=False,
        shuffle_inv_and_pot=False,
        layout=layout_dict,
    )
    frame = viz.custom_get_frame(state, agent_view_size)
    return frame


def visualize_buffer(buffer, n = 100):

    save_dir = "/app/baselines/CEC_UED/minimax/analysis/sampled_viz"
    new_save_dir = os.path.join(save_dir, str(buffer['update_step']))
    os.makedirs(new_save_dir, exist_ok=True)
    
    sliced_levels = slice_levels(buffer, n=n)
    level_layout = slot_to_layout(sliced_levels)

    env = jaxmarl.make(
        "overcooked",
        layout=overcooked_layouts["cramped_room_9"],
        random_reset=False,
        max_steps=256,
        check_held_out=False,
        shuffle_inv_and_pot=False,
    )
    viz = OvercookedVisualizer()
    
    for idx, layout in tqdm(enumerate(level_layout)):
        frame = render_level_frame(env, viz, layout)
        imageio.imwrite(os.path.join(new_save_dir, f"buffer_frame_{idx:03d}.png"), frame)


def main():
    file_list = [
        "/app/ckpts/ippo/overcooked/cramped_room_9/ikTrue/reset_all/lr-20260414-113107/plr_sampled_levels/sampled_levels_step_2926.pkl",
        "/app/ckpts/ippo/overcooked/cramped_room_9/ikTrue/reset_all/lr-20260414-113107/plr_sampled_levels/sampled_levels_step_2927.pkl",
        "/app/ckpts/ippo/overcooked/cramped_room_9/ikTrue/reset_all/lr-20260414-113107/plr_sampled_levels/sampled_levels_step_2928.pkl",
    ]
        
    for file_path in file_list:
        buffer = load_pickle(file_path)
        visualize_buffer(buffer, n=100)

if __name__ == "__main__":
    main()
