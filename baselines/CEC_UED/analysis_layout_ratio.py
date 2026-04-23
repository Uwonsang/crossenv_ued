"""
Analyze the ratio of the 5 layout variants in the saved training dataset.

Classification rule (from maze_map channel 0, active 9x9 region rows 4-12, cols 4-12):
  Passable cells = value==1 (floor) or value==10 (agent)

  passable | connected components | layout
  ---------|---------------------|------------------
      6    |          1          | cramped_room
      6    |          2          | forced_coord
      8    |          1          | coord_ring
     14    |          1          | counter_circuit
     14    |          2          | asymm_advantages
"""

import h5py
import numpy as np
from scipy import ndimage

DATA_PATH = "baselines/CEC_UED/dataset/train_env_states.h5"

LAYOUT_NAMES = [
    "cramped_room",
    "asymm_advantages",
    "coord_ring",
    "counter_circuit",
    "forced_coord",
]

def classify_layout(maze_map_9x9_ch0: np.ndarray) -> str:
    passable = (maze_map_9x9_ch0 == 1) | (maze_map_9x9_ch0 == 10)
    n_passable = int(passable.sum())
    _, n_components = ndimage.label(passable)

    if n_passable == 6 and n_components == 1:
        return "cramped_room"
    elif n_passable == 6 and n_components == 2:
        return "forced_coord"
    elif n_passable == 8:
        return "coord_ring"
    elif n_passable == 14 and n_components == 1:
        return "counter_circuit"
    elif n_passable == 14 and n_components >= 2:
        # asymm_advantages IK variants add 2 "wall holes" that can be isolated,
        # fragmenting into 2, 3, or 4 components depending on placement
        return "asymm_advantages"


def main():
    with h5py.File(DATA_PATH, "r") as f:
        maze_map = f["maze_map"][:]   # shape: (num_saves, num_envs, 17, 17, 3)
        update_steps = f["update_step"][:]

    num_saves, num_envs = maze_map.shape[:2]
    print(f"Loaded: {num_saves} snapshots × {num_envs} envs = {num_saves * num_envs} total samples")
    print(f"Update steps saved: {update_steps.tolist()}\n")

    # Extract active 9x9 area, channel 0
    active = maze_map[:, :, 4:13, 4:13, 0]  # (num_saves, num_envs, 9, 9)

    counts = {name: 0 for name in LAYOUT_NAMES}
    counts["unknown"] = 0
    total = 0

    per_step = []
    for s in range(num_saves):
        step_counts = {name: 0 for name in LAYOUT_NAMES}
        step_counts["unknown"] = 0
        for e in range(num_envs):
            label = classify_layout(active[s, e])
            if label in counts:
                counts[label] += 1
            else:
                counts["unknown"] += 1
                label = "unknown"
            if label in step_counts:
                step_counts[label] += 1
            else:
                step_counts["unknown"] += 1
            total += 1
        per_step.append((int(update_steps[s]), step_counts))

    print("=" * 50)
    print("Overall layout distribution")
    print("=" * 50)
    for name in LAYOUT_NAMES + ["unknown"]:
        n = counts[name]
        pct = 100.0 * n / total if total > 0 else 0
        bar = "#" * int(pct / 2)
        print(f"  {name:<22} {n:5d}  ({pct:5.1f}%)  {bar}")
    print(f"  {'Total':<22} {total:5d}")

    print()
    print("=" * 50)
    print("Per-snapshot breakdown")
    print("=" * 50)
    header = f"{'step':>8} | " + " | ".join(f"{n[:10]:>10}" for n in LAYOUT_NAMES)
    print(header)
    print("-" * len(header))
    for step, sc in per_step:
        row = f"{step:>8} | " + " | ".join(f"{sc[n]:>10}" for n in LAYOUT_NAMES)
        if sc.get("unknown", 0) > 0:
            row += f" | unknown={sc['unknown']}"
        print(row)


if __name__ == "__main__":
    main()
