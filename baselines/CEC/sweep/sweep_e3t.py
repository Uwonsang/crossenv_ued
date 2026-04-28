"""
W&B Sweep for e3t.py hyperparameter search.
Sweeps over: LR, VF_COEF, MOA_COEF, NUM_MINIBATCHES, ENT_COEF, layout, e3t_beta

Usage (run from repo root):
  python -m baselines.CEC.sweep.sweep_e3t --create
  python -m baselines.CEC.sweep.sweep_e3t --run --sweep_id <ID>
  python -m baselines.CEC.sweep.sweep_e3t --create --run
"""
import copy
import os
import argparse
import pickle
import time
import jax
import wandb

SWEEP_CONFIG = {
    "method": "bayes",
    "metric": {
        "name": "returns",
        "goal": "maximize",
    },
    "parameters": {
        "LR": {
            "values": [1e-4, 3e-4, 1e-3],
        },
        "VF_COEF": {
            "values": [0.5, 1.0, 2.0],
        },
        "MOA_COEF": {
            "values": [0.5, 1.0, 2.0],
        },
        "NUM_MINIBATCHES": {
            "values": [2, 4, 8],
        },
        "ENT_COEF": {
            "values": [0.005, 0.01, 0.02],
        },
        # nested under ENV_KWARGS — applied manually
        "layout": {
            "values": [
                "cramped_room_9",
                "asymm_advantages_9",
                "coord_ring_9",
                "counter_circuit_9",
                "forced_coord_9",
            ],
        },
        # nested under TRAIN_KWARGS — applied manually
        "e3t_beta": {
            "values": [0.0, 0.3, 0.55, 1.0],
        },
    },
}

# mirrors repro_config/e3t_final_baseline.yaml
BASE_CONFIG = {
    "ENV_NAME": "overcooked",
    "LR": 1e-3,
    "NUM_ENVS": 256,
    "NUM_SEEDS": 1,
    "NUM_STEPS": 400,
    "FC_DIM_SIZE": 256,
    "GRU_HIDDEN_DIM": 256,
    "TOTAL_TIMESTEPS": 5e7,
    "MAX_TRAIN_STEPS": 5e7,
    "UPDATE_EPOCHS": 4,
    "NUM_MINIBATCHES": 4,
    "GAMMA": 0.99,
    "GAE_LAMBDA": 0.95,
    "CLIP_EPS": 0.2,
    "SCALE_CLIP_EPS": False,
    "ENT_COEF": 0.01,
    "VF_COEF": 1.0,
    "MOA_COEF": 1.0,
    "MAX_GRAD_NORM": 0.5,
    "ACTIVATION": "relu",
    "ANNEAL_LR": True,
    "SEED": 0,
    "ENV_KWARGS": {
        "random_reset": False,
        "max_steps": 400,
        "check_held_out": False,
        "debug": False,
        "partial_obs": False,
        "incentivize_strat": 2,
        "shuffle_inv_and_pot": False,
        "layout": "cramped_room_9",
        "random_reset_fn": "reset_all",
    },
    "TRAINING": True,
    "TRAIN_KWARGS": {
        "ckpt_id": 0,
        "overwrite_ckpt": True,
        "finetune": False,
        "e3t_beta": 0.55,
    },
    "TEST_KWARGS": {
        "beta": 1.0,
        "argmax": False,
        "num_trajs": 2,
        "plot": False,
        "self_play": False,
        "ik": False,
        "debug": False,
        "use_ckpt": True,
    },
    "CONV_NET": True,
    "LSTM": True,
    "FCP": False,
    "FCP_KWARGS": {"train_oracle": True},
    "ENTITY": "overcooked_ai",
    "PROJECT": "crossenv_ued_test",
    "WANDB_MODE": "online",
    "model_name": "e3t",
}


def run_sweep_agent():
    from baselines.CEC.e3t import make_train

    run = wandb.init()
    sp = dict(wandb.config)

    config = copy.deepcopy(BASE_CONFIG)

    # flat params
    config["LR"]              = sp["LR"]
    config["VF_COEF"]         = sp["VF_COEF"]
    config["MOA_COEF"]        = sp["MOA_COEF"]
    config["NUM_MINIBATCHES"] = sp["NUM_MINIBATCHES"]
    config["ENT_COEF"]        = sp["ENT_COEF"]

    # nested params
    config["ENV_KWARGS"]["layout"]     = sp["layout"]
    config["TRAIN_KWARGS"]["e3t_beta"] = sp["e3t_beta"]

    layout_name = sp["layout"]
    xpid = "sweep-%s" % time.strftime("%Y%m%d-%H%M%S")
    filepath = (
        f"ckpts/e3t/{config['ENV_NAME']}/{layout_name}"
        f"/ik{config['ENV_KWARGS']['random_reset']}"
        f"/{config['ENV_KWARGS']['random_reset_fn']}/{xpid}"
    )
    os.makedirs(filepath, exist_ok=True)

    run.name = (
        f"{layout_name}"
        f"_lr{sp['LR']:.0e}_vf{sp['VF_COEF']}_moa{sp['MOA_COEF']}"
        f"_mb{sp['NUM_MINIBATCHES']}_ent{sp['ENT_COEF']:.3f}"
        f"_e3t{sp['e3t_beta']}"
    )

    print(
        f"[sweep/e3t] layout={layout_name}  e3t_beta={sp['e3t_beta']}\n"
        f"            LR={sp['LR']}  VF_COEF={sp['VF_COEF']}  MOA_COEF={sp['MOA_COEF']}\n"
        f"            NUM_MINIBATCHES={sp['NUM_MINIBATCHES']}  ENT_COEF={sp['ENT_COEF']}\n"
        f"            ckpt dir: {filepath}"
    )

    rng = jax.random.PRNGKey(config["SEED"])
    train_jit = jax.jit(make_train(config, 0), device=jax.devices()[0])
    out = train_jit(rng, None, 0)

    # e3t.make_train has no filepath arg — save checkpoint here
    runner_state = out["runner_state"]
    train_state = runner_state[0]
    rng_out = runner_state[-1]
    num_updates = int(config["TOTAL_TIMESTEPS"] // config["NUM_STEPS"] // config["NUM_ENVS"])
    with open(f"{filepath}/seed{config['SEED']}_ckpt0_e3t_updates{num_updates}.pkl", "wb") as f:
        pickle.dump({"key": rng_out, "params": train_state.params, "update_steps": num_updates}, f)


def create_sweep(entity: str, project: str) -> str:
    sweep_id = wandb.sweep(SWEEP_CONFIG, entity=entity, project=project)
    print(f"[sweep/e3t] Created sweep: {sweep_id}")
    print(f"[sweep/e3t] View at: https://wandb.ai/{entity}/{project}/sweeps/{sweep_id}")
    return sweep_id


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--create",   action="store_true")
    parser.add_argument("--run",      action="store_true")
    parser.add_argument("--sweep_id", type=str, default=None)
    parser.add_argument("--count",    type=int, default=None)
    parser.add_argument("--entity",   type=str, default=BASE_CONFIG["ENTITY"])
    parser.add_argument("--project",  type=str, default=BASE_CONFIG["PROJECT"])
    args = parser.parse_args()

    if not args.create and not args.run:
        parser.print_help()
        return

    sweep_id = args.sweep_id
    if args.create:
        sweep_id = create_sweep(args.entity, args.project)

    if args.run:
        if sweep_id is None:
            raise ValueError("Provide --sweep_id or use --create --run together.")
        full_sweep_id = f"{args.entity}/{args.project}/{sweep_id}"
        print(f"[sweep/e3t] Starting agent for {full_sweep_id}")
        wandb.agent(full_sweep_id, function=run_sweep_agent, count=args.count)


if __name__ == "__main__":
    main()
