"""Weighted gradient-share line plot, fetched directly from a wandb run."""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import wandb

# ──────────────────────────────────────────────
# Config
# ──────────────────────────────────────────────
ENTITY = "overcooked_ai"
PROJECT = "crossenv_ued_gradient"
RUN_ID = "gov0m8v9"

LOSS_TYPE = "value"  # "actor" or "value"
SMOOTH_WINDOW = 50  # rolling-mean window, in samples

SAVE_DIR = Path(__file__).parent.parent / "results" / "share_gradient_graph"
SAVE_DIR.mkdir(parents=True, exist_ok=True)

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
# Data loading
# ──────────────────────────────────────────────
def fetch_history(entity: str, project: str, run_id: str, loss_type: str):
    api = wandb.Api()
    run = api.run(f"{entity}/{project}/{run_id}")
    keys = ["env_step"] + [f"grad_share_weighted_{loss_type}/{name}" for name in LAYOUT_NAMES]
    return run.name, run.history(keys=keys, samples=10000)


# ──────────────────────────────────────────────
# Plotting
# ──────────────────────────────────────────────
def plot_share(history, run_name: str, loss_type: str, out_path: Path):
    cols = [f"grad_share_weighted_{loss_type}/{name}" for name in LAYOUT_NAMES]
    history = history.dropna(subset=cols)
    smoothed = history[cols].rolling(SMOOTH_WINDOW, min_periods=1).mean()

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.stackplot(
        history["env_step"],
        [smoothed[col] for col in cols],
        labels=[LAYOUT_LABEL[name] for name in LAYOUT_NAMES],
        colors=LAYOUT_COLORS,
    )

    ax.set_xlabel("Env Step")
    ax.set_ylabel(f"Weighted Gradient Share ({loss_type})")
    ax.set_ylim(0, 1)
    variant = "Gradient-Pop" if "_pop_" in run_name else "Gradient"
    ax.set_title(f"[{variant}] Weighted Gradient Share per Layout — {run_name}")
    ax.legend(fontsize=9, loc="upper right")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved: {out_path}")
    plt.close(fig)


def main():
    run_name, history = fetch_history(ENTITY, PROJECT, RUN_ID, LOSS_TYPE)
    out_path = SAVE_DIR / f"grad_share_weighted_{LOSS_TYPE}_{RUN_ID}.png"
    plot_share(history, run_name, LOSS_TYPE, out_path)


if __name__ == "__main__":
    main()
