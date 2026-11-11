"""Fit layout probes on data saved by environment_representation_probe_collect.py.

For each saved rollout, linear probes are trained on actor features, value
features and two input baselines with the same episode-level train/test splits:
  raw        - flattened 9x9x26 grid observation
  featurized - Overcooked-AI style 96-dim feature vector (overcooked_featurize.py)
The input probes are fit once per split and shared by the actor and value rows:
  gap            = raw_accuracy - accuracy
  featurized_gap = featurized_accuracy - accuracy

Writes CSVs compatible with environment_representation_probe_graph.py:
  <output-dir>/behavior_<b>/environment_probe_{,value_}<cec|dcec>_<num_envs>.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

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


def probe_accuracy(x, labels, train_mask, test_mask) -> float:
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    probe = make_pipeline(
        StandardScaler(), LogisticRegression(max_iter=3000, solver="lbfgs"),
    )
    probe.fit(x[train_mask], labels[train_mask])
    return float(probe.score(x[test_mask], labels[test_mask]))


def run_probes(data, args, model: str, num_envs: int, seed: int, behavior: str):
    labels, episode_ids = data["labels"], data["episode_ids"]
    rows = {"actor": [], "value": []}
    for split in range(args.probe_splits):
        rng = np.random.default_rng(args.rollout_seed + 7919 * seed + split)
        train_mask, test_mask = stratified_episode_split(
            labels, episode_ids, args.train_fraction, rng
        )
        raw_accuracy = probe_accuracy(data["raw"], labels, train_mask, test_mask)
        featurized_accuracy = probe_accuracy(
            data["featurized"], labels, train_mask, test_mask
        )
        for representation in ("actor", "value"):
            accuracy = probe_accuracy(
                data[representation], labels, train_mask, test_mask
            )
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
                "num_train_samples": int(train_mask.sum()),
                "num_test_samples": int(test_mask.sum()),
                "checkpoint": str(data["checkpoint"]),
            })
    return rows


def main():
    args = parse_args()
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
