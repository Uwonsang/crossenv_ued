import os
import h5py
import numpy as np
import jaxmarl
from jaxmarl.environments.overcooked import overcooked_layouts
from jaxmarl.wrappers.baselines import LogWrapper

EVAL_LAYOUTS_9 = [
    "cramped_room_9",
    "asymm_advantages_9",
    "coord_ring_9",
    "counter_circuit_9",
    "forced_coord_9",
]

def init_hdf5(save_path, field_names, field_shapes, field_dtypes, num_updates, num_envs):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with h5py.File(save_path, "w") as f:
        for k in field_names:
            f.create_dataset(
                k,
                shape=(num_updates, num_envs, *field_shapes[k]),
                dtype=field_dtypes[k],
                chunks=(1, num_envs, *field_shapes[k]),
            )
        f.create_dataset(
            "update_step",
            shape=(num_updates,),
            dtype=np.int32,
            chunks=(1,),
        )
    print(f"Initialized HDF5 at {save_path}")

def save_to_hdf5(save_path, field_names, arrays, update_idx, update_step):
    idx = int(update_idx)
    with h5py.File(save_path, "a") as f:
        for k, v in zip(field_names, arrays):
            f[k][idx] = np.asarray(v)
        f["update_step"][idx] = int(update_step)

def make_eval_envs_overcooked(config):
    if config["ENV_NAME"] != "overcooked":
        return {}
    if "EVAL_KWARGS" not in config:
        return {}

    allowed = {"max_steps", "random_reset", "check_held_out", "shuffle_inv_and_pot"}
    base_env_kwargs = {k: v for k, v in config["EVAL_KWARGS"].items() if k in allowed}

    envs = {}
    for layout_name in EVAL_LAYOUTS_9:
        env_kwargs = dict(base_env_kwargs)
        env_kwargs["layout"] = overcooked_layouts[layout_name]
        base_env = jaxmarl.make(config["ENV_NAME"], **env_kwargs)
        envs[layout_name] = LogWrapper(
            base_env,
            env_params={"random_reset_fn": config["EVAL_KWARGS"]["random_reset_fn"]},
        )
    return envs