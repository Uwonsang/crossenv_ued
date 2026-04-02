"""
Copyright (c) Meta Platforms, Inc. and affiliates.
All rights reserved.

This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.
"""

import os
import copy

import jax
import wandb

from minimax.util.loggers import Logger
import hydra
from omegaconf import OmegaConf



@hydra.main(version_base=None, config_path="./config", config_name="ippo_overcooked_CEC_minimax")
def main(config):
    with jax.disable_jit(False):
        config = OmegaConf.to_container(config)

        # === Setup the main runner ===
        from minimax.runners_ma import ExperimentRunner


        xp_runner = ExperimentRunner(
            config=config,
            train_runner=config["train_runner"],
            env_name=config["env_name"],
            agent_rl_algo=config["agent_rl_algo"],
            student_agent_kind=config["student_agent_kind"],
            train_runner_kwargs=config["train_runner_args"],
            env_kwargs=config["env_args"],
            ued_env_kwargs=config["ued_env_args"],
            student_rl_kwargs=config["student_rl_args"],
            student_model_kwargs=config["student_model_args"],
            eval_kwargs=config["eval_args"],
            eval_env_kwargs=config["eval_env_args"],
            n_devices=config["n_devices"],
            shaped_reward_steps=config["n_shaped_reward_steps"],
            shaped_reward_n_updates=config["n_shaped_reward_updates"],
            xpid=config["xpid"]
        )

        # === Configure logging ===
        # Set up wandb
        wandb_args = args.wandb_args
        if wandb_args.base_url:
            os.environ["WANDB_BASE_URL"] = wandb_args.base_url
        # if wandb_args.api_key:
        #     os.environ["WANDB_API_KEY"] = wandb_args.api_key
        if wandb_args.base_url:  # and wandb_args.api_key:
            os.environ["WANDB_CACHE_DIR"] = '~/.cache/wandb'
            wandb.init(project=wandb_args.project,
                    entity=wandb_args.entity,
                    config=args,
                    name=args.xpid,
                    group=wandb_args.group,
                    mode=wandb_args.mode
                    )
            callback = wandb.log
        else:
            callback = None

        logger = Logger(
            log_dir=args.log_dir,
            xpid=args.xpid,
            xp_args=args,
            callback=callback,
            from_last_checkpoint=args.from_last_checkpoint,
            verbose=args.verbose)

        # === Start training ===
        rng = jax.random.PRNGKey(args.seed)
        xp_runner.train(
            rng=rng,
            n_total_updates=args.n_total_updates,
            logger=logger,
            log_interval=args.log_interval,
            test_interval=args.test_interval,
            checkpoint_interval=args.checkpoint_interval,
            archive_interval=args.archive_interval,
            archive_init_checkpoint=args.archive_init_checkpoint,
            from_last_checkpoint=args.from_last_checkpoint)

if __name__ == "__main__":
    main()