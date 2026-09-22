"""Evaluate saved concrete state pairs without generating new map samples."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from policy_value_concrete_example import (
    ROOT,
    add_dataset_normalized_distances,
    checkpoint_path,
    compute_cka_rows,
    compute_rsa_rows,
    evaluate_checkpoint,
    load_config,
    parse_model_spec,
)


def discover_pair_files(root: Path) -> list[Path]:
    direct = root / "concrete_state_pairs.json"
    files = [direct] if direct.is_file() else []
    files.extend(sorted(root.glob("*/concrete_state_pairs.json")))
    unique = []
    seen = set()
    for path in files:
        resolved = path.resolve()
        if resolved not in seen:
            seen.add(resolved)
            unique.append(path)
    return unique


def load_pairs(path: Path) -> list[tuple[int, list[dict]]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    pairs = []
    for index, pair in enumerate(payload.get("pairs", [])):
        states = pair.get("states", [])
        if len(states) != 2:
            raise ValueError(
                f"Pair {pair.get('pair_id', index)} in {path} has {len(states)} states"
            )
        pairs.append((int(pair.get("pair_id", index)), states))
    if not pairs:
        raise ValueError(f"No state pairs found in {path}")
    return pairs


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError(f"Cannot write an empty CSV: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-root", type=Path, required=True)
    parser.add_argument("--pairs-root", type=Path)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--models", nargs="+", type=parse_model_spec,
                        default=[("CEC", 64), ("CEC_IDAAC", 64)])
    parser.add_argument("--seeds", type=int, nargs="+", default=list(range(6)))
    parser.add_argument("--horizon", type=int, default=200)
    parser.add_argument("--rollouts", type=int, default=100)
    parser.add_argument("--rollout-seed", type=int, default=2701)
    parser.add_argument("--gamma", type=float, default=.99)
    parser.add_argument("--max-policy-js", type=float, default=.15)
    parser.add_argument("--min-interact-probability", type=float, default=.7)
    parser.add_argument("--min-abs-return-delta", type=float, default=1.0)
    parser.add_argument(
        "--config", type=Path,
        default=ROOT / "baselines/CEC_UED/config/ippo_overcooked_CEC_gradient.yaml",
    )
    args = parser.parse_args()
    args.model_root = args.model_root.expanduser()
    pairs_root = args.pairs_root or (
        args.model_root / "analysis" / "policy_value_large_scale"
    )
    output_root = args.output_root or pairs_root
    pair_files = discover_pair_files(pairs_root)
    if not pair_files:
        parser.error(f"No concrete_state_pairs.json found under {pairs_root}")

    config = load_config(args.config)
    for pair_file in pair_files:
        try:
            pairs = load_pairs(pair_file)
        except ValueError as error:
            parser.error(str(error))
        relative_dir = pair_file.parent.relative_to(pairs_root)
        output_dir = output_root / relative_dir
        rows = []
        for model, num_envs in args.models:
            for seed in args.seeds:
                checkpoint = checkpoint_path(
                    args.model_root, model, num_envs, seed
                )
                if checkpoint is None:
                    print(f"Missing checkpoint: {model} {num_envs} seed {seed}")
                    continue
                for pair_id, states in pairs:
                    rows.append(evaluate_checkpoint(
                        config, states, pair_id, model, num_envs, seed,
                        checkpoint, args,
                    ))
                print(
                    f"Evaluated {model} {num_envs} seed {seed} on "
                    f"{len(pairs)} fixed pairs from {pair_file.parent.name}"
                )
        if not rows:
            raise RuntimeError(f"No usable checkpoint found for {pair_file}")

        cka_rows = compute_cka_rows(rows)
        rsa_rows = compute_rsa_rows(rows)
        add_dataset_normalized_distances(rows)
        write_csv(output_dir / "concrete_example_metrics.csv", rows)
        write_csv(output_dir / "concrete_example_cka.csv", cka_rows)
        write_csv(output_dir / "concrete_example_rsa.csv", rsa_rows)
        print(f"Saved fixed-pair metrics to {output_dir}")


if __name__ == "__main__":
    main()
