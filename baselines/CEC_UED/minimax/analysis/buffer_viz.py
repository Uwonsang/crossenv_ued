import pickle
import os
import jax
import jax.numpy as jnp


def load_pickle(file_path):
    with open(file_path, "rb") as f:
        data = pickle.load(f)
        plr_buffer = jax.device_put(jax.tree_map(jnp.array, data["plr_buffer"]))
    return plr_buffer

def visualize_buffer(buffer):
    
    breakpoint()
    print(buffer)


def main(file_path):
    buffer = load_pickle(file_path)
    visualize_buffer(buffer)

if __name__ == "__main__":
    file_path = "/app/ckpts/ippo/overcooked/cramped_room_9/ikTrue/reset_all/lr-20260407-124217/plr_buffer/plr_buffer_step_000.pkl"
    main(file_path)