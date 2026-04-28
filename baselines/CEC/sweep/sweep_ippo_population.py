"""
W&B Sweep for ippo_general_population.py hyperparameter search.
Sweeps over: LR, VF_COEF, NUM_MINIBATCHES, ENT_COEF, layout, FCP, e3t_beta

Usage (run from repo root):
  # 1. Create sweep (prints sweep ID)
  python -m baselines.CEC.sweep.sweep_ippo_population --create

  # 2. Join an existing sweep
  python -m baselines.CEC.sweep.sweep_ippo_population --run --sweep_id <ID>

  # 3. Create + immediately start one agent
  python -m baselines.CEC.sweep.sweep_ippo_population --create --run
"""
import copy
import os
import argparse
import time
import jax
import wandb

# ── Sweep configuration ────────────────────────────────────────────────────────
# W&B sweep params must be flat — nested keys (layout, e3t_beta) are applied
# manually in run_sweep_agent() below.
SWEEP_CONFIG = {
    "method": "bayes",   # "grid" | "random" | "bayes"
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
        "NUM_MINIBATCHES": {
            "values": [4, 8],
        },
        "ENT_COEF": {
            "values": [0.005, 0.01],
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
        # FCP (Fictitious Co-Play) on/off
        "FCP": {
            "values": [True, False],
        },
        # nested under TRAIN_KWARGS — applied manually
        # 0 = uniform partner, 1 = original probs, >1 = magnify extremes
        "e3t_beta": {
            "values": [0.0, 0.3, 0.55, 1.0],
        },
    },
}

# ── Base config (mirrors repro_config/ippo_final_baseline.yaml) ───────────────
BASE_CONFIG = {
    "ENV_NAME": "overcooked",
    "LR": 3e-4,
    "NUM_ENVS": 256,
    "NUM_SEEDS": 1,
    "NUM_STEPS": 400,
    "FC_DIM_SIZE": 256,
    "GRU_HIDDEN_DIM": 256,
    "TOTAL_TIMESTEPS": 3e9,
    "MAX_TRAIN_STEPS": 3e9,
    "UPDATE_EPOCHS": 4,
    "NUM_MINIBATCHES": 2,
    "GAMMA": 0.99,
    "GAE_LAMBDA": 0.95,
    "CLIP_EPS": 0.2,
    "SCALE_CLIP_EPS": False,
    "ENT_COEF": 0.005,
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
    "model_name": "IPPO",
}


def run_sweep_agent():
    """Single sweep trial — called by wandb.agent."""
    from baselines.CEC.ippo_general_population import make_train

    run = wandb.init()
    sp = dict(wandb.config)

    # Deep-copy so each trial has an independent config
    config = copy.deepcopy(BASE_CONFIG)

    # ── flat params ────────────────────────────────────────────────────────────
    config["LR"]              = sp["LR"]
    config["VF_COEF"]         = sp["VF_COEF"]
    config["NUM_MINIBATCHES"] = sp["NUM_MINIBATCHES"]
    config["ENT_COEF"]        = sp["ENT_COEF"]
    config["FCP"]             = sp["FCP"]

    # ── nested params ──────────────────────────────────────────────────────────
    config["ENV_KWARGS"]["layout"]      = sp["layout"]
    config["TRAIN_KWARGS"]["e3t_beta"]  = sp["e3t_beta"]

    layout_name = sp["layout"]
    fcp_tag     = "fcp" if sp["FCP"] else "nofcp"
    xpid = "sweep-%s" % time.strftime("%Y%m%d-%H%M%S")
    filepath = (
        f"ckpts/ippo/{config['ENV_NAME']}/{layout_name}"
        f"/ik{config['ENV_KWARGS']['random_reset']}"
        f"/{config['ENV_KWARGS']['random_reset_fn']}/{fcp_tag}/{xpid}"
    )
    os.makedirs(filepath, exist_ok=True)

    run.name = (
        f"{layout_name}_{fcp_tag}"
        f"_lr{sp['LR']:.0e}_vf{sp['VF_COEF']}"
        f"_mb{sp['NUM_MINIBATCHES']}_ent{sp['ENT_COEF']:.3f}"
        f"_e3t{sp['e3t_beta']}"
    )

    print(
        f"[sweep] layout={layout_name}  FCP={sp['FCP']}  e3t_beta={sp['e3t_beta']}\n"
        f"        LR={sp['LR']}  VF_COEF={sp['VF_COEF']}  "
        f"NUM_MINIBATCHES={sp['NUM_MINIBATCHES']}  ENT_COEF={sp['ENT_COEF']}\n"
        f"        ckpt dir: {filepath}"
    )

    rng = jax.random.PRNGKey(config["SEED"])
    train_jit = jax.jit(make_train(config, 0, filepath), device=jax.devices()[0])
    train_jit(rng, None, 0)


def create_sweep(entity: str, project: str) -> str:
    sweep_id = wandb.sweep(SWEEP_CONFIG, entity=entity, project=project)
    print(f"[sweep] Created sweep: {sweep_id}")
    print(f"[sweep] View at: https://wandb.ai/{entity}/{project}/sweeps/{sweep_id}")
    return sweep_id


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--create",   action="store_true", help="Create a new W&B sweep")
    parser.add_argument("--run",      action="store_true", help="Start a sweep agent")
    parser.add_argument("--sweep_id", type=str,  default=None, help="Existing sweep ID to join")
    parser.add_argument("--count",    type=int,  default=None, help="Max trials per agent (default: unlimited)")
    parser.add_argument("--entity",   type=str,  default=BASE_CONFIG["ENTITY"])
    parser.add_argument("--project",  type=str,  default=BASE_CONFIG["PROJECT"])
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
        print(f"[sweep] Starting agent for {full_sweep_id}")
        wandb.agent(full_sweep_id, function=run_sweep_agent, count=args.count)


if __name__ == "__main__":
    main()
