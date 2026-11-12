"""Fit layout probes on data saved by environment_representation_probe_collect.py.

For each saved rollout, linear probes are trained on actor features, value
features and two input baselines with the same episode-level train/test splits:
  raw        - flattened 9x9x26 grid observation
  featurized - Overcooked-AI style 96-dim feature vector (overcooked_featurize.py)
The input probes are fit once per split and shared by the actor and value rows:
  gap            = raw_accuracy - accuracy
  featurized_gap = featurized_accuracy - accuracy

Probes run on the GPU (probe_jax.py: sklearn StandardScaler +
LogisticRegression objective solved with jaxopt.LBFGS, all splits at once).

Writes CSVs compatible with environment_representation_probe_graph.py:
  <output-dir>/behavior_<b>/environment_probe_{,value_}<cec|dcec>_<num_envs>.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import jax
import numpy as np

from probe_jax import probe_accuracies
from environment_representation_probe_review import (
    LAYOUT_FAMILIES,
    MODEL_FILE_LABELS,
    MODEL_SPECS,
    parse_model_spec,
    stratified_episode_split,
    write_csv,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fit layout probes on saved data.")
    parser.add_argument(
        "--model-root", type=Path,
        default=Path("/mnt/nas/wonsang/crossenv_ued/models/ICRL"),
    )
    parser.add_argument(
        "--models", nargs="+", type=parse_model_spec,
        default=[("CEC", 64), ("CEC_IDAAC", 64)], metavar="MODEL_SIZE",
    )
    parser.add_argument(
        "--behaviors", nargs="+", choices=("own", "random", "cross"),
        default=["own", "random", "cross"],
    )
    parser.add_argument("--seeds", type=int, nargs="+", default=list(range(6)))
    parser.add_argument("--probe-splits", type=int, default=10)
    parser.add_argument("--train-fraction", type=float, default=0.7)
    parser.add_argument("--rollout-seed", type=int, default=1701)
    parser.add_argument(
        "--data-dir", type=Path, default=None,
        help="Defaults to <model-root>/analysis/representation_probe/data.",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=None,
        help="Defaults to <model-root>/analysis/representation_probe/data_results.",
    )
    return parser.parse_args()


def run_probes(data, args, model: str, num_envs: int, seed: int, behavior: str):
    labels, episode_ids = data["labels"], data["episode_ids"]
    train_masks, test_masks = zip(*(
        stratified_episode_split(
            labels, episode_ids, args.train_fraction,
            np.random.default_rng(args.rollout_seed + 7919 * seed + split),
        )
        for split in range(args.probe_splits)
    ))
    train_masks, test_masks = np.stack(train_masks), np.stack(test_masks)
    accuracies = {
        name: probe_accuracies(
            data[name], labels, train_masks, test_masks, len(LAYOUT_FAMILIES)
        )
        for name in ("raw", "featurized", "actor", "value")
    }

    rows = {"actor": [], "value": []}
    for split in range(args.probe_splits):
        raw_accuracy = float(accuracies["raw"][split])
        featurized_accuracy = float(accuracies["featurized"][split])
        for representation in ("actor", "value"):
            accuracy = float(accuracies[representation][split])
            rows[representation].append({
                "model": model,
                "model_label": MODEL_SPECS[model]["display"],
                "model_num_envs": num_envs,
                "representation": representation,
                "behavior": behavior,
                "checkpoint_seed": seed,
                "probe_split": split,
                "accuracy": accuracy,
                "raw_accuracy": raw_accuracy,
                "gap": raw_accuracy - accuracy,
                "featurized_accuracy": featurized_accuracy,
                "featurized_gap": featurized_accuracy - accuracy,
                "chance_accuracy": 1.0 / len(LAYOUT_FAMILIES),
                "num_train_samples": int(train_masks[split].sum()),
                "num_test_samples": int(test_masks[split].sum()),
                "checkpoint": str(data["checkpoint"]),
            })
    return rows


def main():
    args = parse_args()
    # JAX uses a visible GPU automatically; report it so a CPU fallback is obvious.
    print(f"JAX devices: {jax.devices()}")
    if jax.default_backend() != "gpu":
        print("WARNING: no GPU visible to JAX, running on CPU (much slower).")
    args.model_root = args.model_root.expanduser()
    probe_root = args.model_root / "analysis" / "representation_probe"
    data_dir = args.data_dir or probe_root / "data"
    output_dir = args.output_dir or probe_root / "data_results"
    saved_csv_paths = []

    for model, num_envs in args.models:
        label = MODEL_FILE_LABELS[model]
        for behavior in args.behaviors:
            rows = {"actor": [], "value": []}
            for seed in args.seeds:
                path = data_dir / f"behavior_{behavior}" / f"{label}_{num_envs}_seed{seed}.npz"
                if not path.exists():
                    print(f"Missing data: {path}")
                    continue
                print(f"Probing {path}")
                with np.load(path) as data:
                    seed_rows = run_probes(
                        dict(data), args, model, num_envs, seed, behavior,
                    )
                for representation in rows:
                    rows[representation].extend(seed_rows[representation])

            for representation, representation_rows in rows.items():
                if not representation_rows:
                    continue
                prefix = "" if representation == "actor" else "value_"
                csv_path = output_dir / f"behavior_{behavior}" / (
                    f"environment_probe_{prefix}{label}_{num_envs}.csv"
                )
                csv_path.parent.mkdir(parents=True, exist_ok=True)
                write_csv(representation_rows, csv_path)
                saved_csv_paths.append(csv_path)
                print(f"Saved: {csv_path}")

    if not saved_csv_paths:
        raise RuntimeError("No probe data was found")


if __name__ == "__main__":
    main()
