"""Per-layout gradient-norm line plot, fetched directly from a wandb run."""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import wandb

# ──────────────────────────────────────────────
# Config (defaults, overridable via CLI args)
# ──────────────────────────────────────────────
ENTITY = "overcooked_ai"
PROJECT = "crossenv_ued_gradient"
RUN_ID = "5rwobcx9"

LOSS_TYPE = "value"  # "actor" or "value"
SMOOTH_WINDOW = 50  # rolling-mean window, in samples
X_AXIS = "env_step"  # "env_step" or "update_step"

SAVE_DIR = Path(__file__).parent.parent / "results" / "grad_norm_graph"

LAYOUT_NAMES = [
    "cramped_room_9", "asymm_advantages_9", "coord_ring_9",
    "counter_circuit_9", "forced_coord_9",
]
LAYOUT_LABEL = {
    "cramped_room_9": "Cramped Room",
    "asymm_advantages_9": "Asymm Advantages",
    "coord_ring_9": "Coord Ring",
    "counter_circuit_9": "Counter Circuit",
    "forced_coord_9": "Forced Coord",
}
LAYOUT_COLORS = ["#d65f5f", "#956cb4", "#d5bb67", "#6acc64", "#4878d0"]


# ──────────────────────────────────────────────
# Args
# ──────────────────────────────────────────────
def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--entity", default=ENTITY)
    parser.add_argument("--project", default=PROJECT)
    parser.add_argument("--run-id", default=RUN_ID)
    parser.add_argument("--loss-type", default=LOSS_TYPE, choices=["actor", "value"])
    parser.add_argument("--smooth-window", type=int, default=SMOOTH_WINDOW)
    parser.add_argument("--x-axis", default=X_AXIS, choices=["env_step", "update_step"])
    return parser.parse_args()


# ──────────────────────────────────────────────
# Data loading
# ──────────────────────────────────────────────
def fetch_history(entity: str, project: str, run_id: str, loss_type: str):
    api = wandb.Api()
    run = api.run(f"{entity}/{project}/{run_id}")
    keys = ["env_step"] + [f"grad_conflict_{loss_type}/norm/{name}" for name in LAYOUT_NAMES]
    history = run.history(keys=keys, samples=10000)
    # update_step = env_step / (NUM_ENVS * NUM_STEPS), derived from the same run config
    # used to compute env_step at logging time (see ippo_general_gradient*.py callbacks)
    steps_per_update = run.config["NUM_ENVS"] * run.config["NUM_STEPS"]
    history["update_step"] = history["env_step"] / steps_per_update
    return run.name, run.config.get("model_name", "unknown"), history


# ──────────────────────────────────────────────
# Plotting
# ──────────────────────────────────────────────
def plot_grad_norm(history, run_name: str, loss_type: str, x_axis: str, smooth_window: int, out_path: Path):
    cols = [f"grad_conflict_{loss_type}/norm/{name}" for name in LAYOUT_NAMES]
    history = history.dropna(subset=cols)
    smoothed = history[cols].rolling(smooth_window, min_periods=1).mean()

    fig, ax = plt.subplots(figsize=(8, 5))
    for name, color in zip(LAYOUT_NAMES, LAYOUT_COLORS):
        ax.plot(history[x_axis], smoothed[f"grad_conflict_{loss_type}/norm/{name}"],
                 label=LAYOUT_LABEL[name], color=color, linewidth=1.5)

    variant = "Gradient-Pop" if "_pop_" in run_name else "Gradient"

    ax.set_xlabel("Env Step" if x_axis == "env_step" else "Update Step")
    ax.set_ylabel(f"Gradient Norm ({loss_type})")
    ax.set_title(f"[{variant}] Gradient Norm per Layout — {run_name}")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved: {out_path}")
    plt.close(fig)


def main():
    args = parse_args()
    SAVE_DIR.mkdir(parents=True, exist_ok=True)
    run_name, model_name, history = fetch_history(args.entity, args.project, args.run_id, args.loss_type)
    out_path = SAVE_DIR / f"grad_norm_{args.loss_type}_{model_name}_{args.run_id}_{args.x_axis}.png"
    plot_grad_norm(history, run_name, args.loss_type, args.x_axis, args.smooth_window, out_path)


if __name__ == "__main__":
    main()
