import os
import pickle
import time

import hydra
import jax
import yaml
from omegaconf import OmegaConf

import wandb
from baselines.CEC import e3t as e3t_base
from baselines.CEC_UED.modified_wall_ippo_general_dual_destination import (
    get_wall_map_name,
    make_modified_wall_env,
)


def initialize_environment(config):
    map_name = get_wall_map_name(config)
    config["layout_name"] = map_name
    env = make_modified_wall_env(config)
    config["obs_dim"] = env.observation_space(env.agents[0]).shape
    return env


e3t_base.initialize_environment = initialize_environment


@hydra.main(
    version_base=None,
    config_path="config",
    config_name="e3t_modified_wall_dual_destination",
)
def main(config):
    config = OmegaConf.to_container(config, resolve=True)
    config["model_name"] = "E3T"
    map_name = get_wall_map_name(config)
    xpid = "lr-%s" % time.strftime("%Y%m%d-%H%M%S")

    if config["WANDB_MODE"] == "online":
        with open("private.yaml") as f:
            private_info = yaml.load(f, Loader=yaml.FullLoader)
        wandb.login(key=private_info["wandb_key"])

    wandb.init(
        entity=config["ENTITY"],
        project=config["PROJECT"],
        tags=["E3T", "RNN", "SP", "modified_wall"],
        config=config,
        mode=config["WANDB_MODE"],
        name=f"E3T_modified_wall_{map_name}_seed{config['SEED']}",
    )

    filepath = (
        f"ckpts/e3t/{config['ENV_NAME']}/modified_wall/{map_name}"
        f"/ikFalse/{config['ENV_KWARGS']['random_reset_fn']}/e3t/{xpid}"
    )
    config["filepath"] = filepath
    config["fcp_prefix"] = ""
    print(f"Working on: \n{filepath}\n")

    model_params = None
    final_update_step = 0
    rng = jax.random.PRNGKey(config["SEED"])

    print(f"Starting from update step {final_update_step}")
    train_jit = jax.jit(
        e3t_base.make_train(config, final_update_step),
        device=jax.devices()[0],
    )
    out = train_jit(rng, model_params, final_update_step)
    jax.effects_barrier()

    runner_state, _ = out["runner_state"]
    model_state = runner_state[0]
    rng = runner_state[-1]
    num_updates = int(
        config["TOTAL_TIMESTEPS"] // config["NUM_STEPS"] // config["NUM_ENVS"]
    )

    os.makedirs(filepath, exist_ok=True)
    ckpt_path = (
        f"{filepath}/seed{config['SEED']}_ckpt"
        f"{config['TRAIN_KWARGS']['ckpt_id']}_e3t_updates{num_updates}.pkl"
    )
    with open(ckpt_path, "wb") as f:
        pickle.dump(
            {
                "key": rng,
                "params": model_state.params,
                "update_steps": num_updates,
            },
            f,
        )

    print(f"Saved model to {ckpt_path}")
    wandb.finish()


if __name__ == "__main__":
    main()
