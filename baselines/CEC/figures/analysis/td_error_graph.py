"""Per-layout TD-error line plot, fetched directly from a wandb run."""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import wandb

# ──────────────────────────────────────────────
# Config
# ──────────────────────────────────────────────
ENTITY = "overcooked_ai"
PROJECT = "crossenv_ued_gradient"
RUN_ID = "9g9abvem"

TD_METRIC = "mean_abs"  # "mean_abs" or "rmse"
SMOOTH_WINDOW = 50  # rolling-mean window, in samples

SAVE_DIR = Path(__file__).parent.parent / "results" / "td_error_graph"
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
def fetch_history(entity: str, project: str, run_id: str, metric: str):
    api = wandb.Api()
    run = api.run(f"{entity}/{project}/{run_id}")
    keys = ["env_step", f"td_error/{metric}"] + [f"td_error/{name}/{metric}" for name in LAYOUT_NAMES]
    return run.name, run.history(keys=keys, samples=10000)


# ──────────────────────────────────────────────
# Plotting
# ──────────────────────────────────────────────
def plot_td_error(history, run_name: str, metric: str, out_path: Path):
    cols = [f"td_error/{name}/{metric}" for name in LAYOUT_NAMES]
    overall_col = f"td_error/{metric}"
    history = history.dropna(subset=cols + [overall_col])
    smoothed = history[cols + [overall_col]].rolling(SMOOTH_WINDOW, min_periods=1).mean()

    fig, ax = plt.subplots(figsize=(8, 5))
    for name, color in zip(LAYOUT_NAMES, LAYOUT_COLORS):
        ax.plot(history["env_step"], smoothed[f"td_error/{name}/{metric}"],
                 label=LAYOUT_LABEL[name], color=color, linewidth=1.5)
    ax.plot(history["env_step"], smoothed[overall_col],
             label="Overall", color="black", linewidth=1.2, linestyle="--")

    variant = "Gradient-Pop" if "_pop_" in run_name else "Gradient"

    ax.set_xlabel("Env Step")
    ax.set_ylabel(f"TD Error ({metric})")
    ax.set_title(f"[{variant}] TD Error per Layout — {run_name}")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved: {out_path}")
    plt.close(fig)


def main():
    run_name, history = fetch_history(ENTITY, PROJECT, RUN_ID, TD_METRIC)
    out_path = SAVE_DIR / f"td_error_{TD_METRIC}_{RUN_ID}.png"
    plot_td_error(history, run_name, TD_METRIC, out_path)


if __name__ == "__main__":
    main()
