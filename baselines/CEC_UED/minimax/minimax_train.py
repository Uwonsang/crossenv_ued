"""
Copyright (c) Meta Platforms, Inc. and affiliates.
All rights reserved.

This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.
"""

import jax
import wandb

import hydra
from omegaconf import OmegaConf
import yaml
import time


@hydra.main(version_base=None, config_path="./config", config_name="ippo_overcooked_CEC_minimax")
def main(config):
    save_xpid = "lr-%s" % time.strftime("%Y%m%d-%H%M%S")

    with jax.disable_jit(False):
        config = OmegaConf.to_container(config)
        # === Setup the main runner ===
        from minimax.runners_ma import ExperimentRunner
        xp_runner = ExperimentRunner(
            config=config,
            train_runner=config["train_runner"],
            env_name=config["env_name"],
            n_devices=config["n_devices"],
            xpid=save_xpid
        )

        # === Configure logging ===
        # Set up wandb
        if config["WANDB_MODE"] == "online":
            with open("private.yaml") as f:
                private_info = yaml.load(f, Loader=yaml.FullLoader)
            wandb.login(key=private_info["wandb_key"])

        layout_name = config["ENV_KWARGS"]["layout"]
        wandb.init(
            project=config["PROJECT"],
            entity=config["ENTITY"],
            config=config,
            name=f"CEC_UED_{layout_name}_seed{config['SEED']}",
            mode=config["WANDB_MODE"]
        )


        config["NUM_UPDATES"] = (
            config["TOTAL_TIMESTEPS"] // config["NUM_STEPS"] // config["NUM_ENVS"]
        )
        # === Start training ===
        rng = jax.random.PRNGKey(config["SEED"])
        xp_runner.train(
            rng=rng,
            n_total_updates=config["NUM_UPDATES"],
            log_interval=config["log_interval"],
            test_interval=config["test_interval"],
            checkpoint_interval=config["checkpoint_interval"],
            archive_interval=config["archive_interval"],
            archive_init_checkpoint=config["archive_init_checkpoint"],
            from_last_checkpoint=config["from_last_checkpoint"])

if __name__ == "__main__":
    main()