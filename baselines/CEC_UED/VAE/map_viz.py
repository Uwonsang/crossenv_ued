import pickle
import os
from tqdm import tqdm
import jax
import jax.numpy as jnp
from jaxmarl.environments.overcooked import overcooked_layouts

from jaxmarl.viz.overcooked_visualizer import OvercookedVisualizer
from flax import struct
import chex
import hydra
from omegaconf import OmegaConf
import jaxmarl
import imageio


def load_pickle(path):
    with open(path, "rb") as f:
        data = pickle.load(f)
    return data

@struct.dataclass
class FilteredState:
    agent_pos: chex.Array
    agent_dir: chex.Array
    agent_dir_idx: chex.Array
    agent_inv: chex.Array
    maze_map: chex.Array

def filtered_state(ep_state):
    dtype_map = {
        "agent_pos": jnp.uint32,
        "agent_dir": jnp.int8,
        "agent_dir_idx": jnp.int32,
        "agent_inv": jnp.int32,
        "maze_map": jnp.uint8,
    }
    state_dict = {
        k: jnp.asarray(v, dtype=dtype_map.get(k, v.dtype))
        for k, v in ep_state.items()
        if k in dtype_map
    }

    state_dict = {
        k: v.reshape(-1, *v.shape[2:]) for k, v in state_dict.items()
    }
    
    return FilteredState(**state_dict)

def layout_render(env_state, config, save_dir):
    env = jaxmarl.make(config["ENV_NAME"], **config["ENV_KWARGS"])
    agent_view_size = env.agent_view_size
    viz = OvercookedVisualizer()
    
    state = filtered_state(env_state)
    # num_layouts = state.agent_pos.shape[0]
    frame = [
        viz.custom_get_frame(jax.tree_map(lambda x: x[step], state), agent_view_size)
        for step in tqdm(range(100))
    ]

    os.makedirs(save_dir, exist_ok=True)
    for i, frame in enumerate(frame):
        imageio.imwrite(os.path.join(save_dir, f"layout_{i:03d}.png"), frame)

    
@hydra.main(version_base=None, config_path="config", config_name="collect_overcooked")
def visualize_layout(config):
    config = OmegaConf.to_container(config)
    config["data_dir"] = '/app/baselines/CEC_UED/VAE/dataset/lr-20260221-062635'

    data_path = os.path.join(config["data_dir"], "env_states.pkl")    
    env_state = load_pickle(data_path)

    save_dir = "/app/baselines/CEC_UED/VAE/dataset/img"
    layout_render(env_state, config, save_dir)


if __name__ == "__main__":
    visualize_layout()