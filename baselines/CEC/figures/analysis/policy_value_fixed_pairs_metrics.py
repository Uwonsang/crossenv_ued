"""Evaluate saved concrete state pairs without generating new map samples."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

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


MODEL_LABELS = {"CEC": "CEC", "CEC_IDAAC": "DCEC"}
RSA_COLORS = (
    "#0072B2", "#56B4E9", "#D55E00", "#E69F00", "#117733",
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


def ordered_series(rows: list[dict]) -> list[tuple[str, int]]:
    order = []
    for row in rows:
        key = (row["model"], int(row["num_envs"]))
        if key not in order:
            order.append(key)
    rank = {"CEC": 0, "CEC_IDAAC": 1}
    return sorted(order, key=lambda item: (rank.get(item[0], 2), item[1]))


def rsa_values(
    rows: list[dict], distance: str, series: tuple[str, int], column: str
) -> np.ndarray:
    model, num_envs = series
    return np.asarray([
        float(row[column]) for row in rows
        if row["distance_metric"] == distance
        and row["model"] == model
        and int(row["num_envs"]) == num_envs
        and np.isfinite(float(row[column]))
    ])


def save_rsa_grouped_figure(
    rows: list[dict], output: Path, title: str,
    metrics: tuple[tuple[str, str], ...], ylabel: str,
    fixed_limits: tuple[float, float] | None = None,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    distances = ("cosine", "zscored_rms")
    distance_labels = {"cosine": "Cosine RDM", "zscored_rms": "Z-scored RMS RDM"}
    series_order = ordered_series(rows)
    figure, axes = plt.subplots(1, 2, figsize=(12.4, 4.5), squeeze=False)
    x = np.arange(len(series_order))
    width = .8 / len(metrics)
    for axis, distance in zip(axes[0], distances):
        for offset, (column, label) in enumerate(metrics):
            means, sems, seed_values = [], [], []
            for series in series_order:
                values = rsa_values(rows, distance, series, column)
                seed_values.append(values)
                means.append(float(values.mean()) if len(values) else np.nan)
                sems.append(
                    float(values.std(ddof=1) / np.sqrt(len(values)))
                    if len(values) > 1 else 0.0
                )
            positions = x + (offset - (len(metrics) - 1) / 2) * width
            axis.bar(
                positions, means, width=width, yerr=sems, capsize=3,
                color=RSA_COLORS[offset], edgecolor="black", linewidth=.5,
                label=label, zorder=2,
            )
            for position, values in zip(positions, seed_values):
                if len(values):
                    axis.scatter(
                        np.full(len(values), position), values, s=13,
                        color="black", alpha=.48, linewidths=0, zorder=3,
                    )
        axis.axhline(0, color="black", linewidth=.7)
        axis.set_xticks(x)
        axis.set_xticklabels([
            f"{MODEL_LABELS.get(model, model)} ({num_envs})"
            for model, num_envs in series_order
        ])
        axis.set_title(distance_labels[distance], fontweight="bold")
        axis.set_ylabel(ylabel)
        if fixed_limits is not None:
            axis.set_ylim(*fixed_limits)
        axis.grid(axis="y", alpha=.25)
        axis.legend(frameon=False, fontsize=8)
    figure.suptitle(title, fontweight="bold")
    figure.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=300, bbox_inches="tight")
    plt.close(figure)


def save_rsa_figures(rows: list[dict], output_dir: Path, layout_name: str) -> None:
    figure_dir = output_dir / "rsa_figures"
    layout_title = layout_name.replace("_", " ").title()
    save_rsa_grouped_figure(
        rows, figure_dir / "rsa_basic_alignment.png",
        f"RSA geometry alignment · {layout_title}",
        (
            ("policy_value_rsa", "Policy ↔ Value"),
            ("policy_a_b_rsa", "Policy A ↔ B"),
            ("value_a_b_rsa", "Value A ↔ B"),
        ),
        "Spearman RSA", fixed_limits=(-1.05, 1.05),
    )
    save_rsa_grouped_figure(
        rows, figure_dir / "rsa_target_alignment.png",
        f"RSA target alignment · {layout_title}",
        (
            ("policy_behavior_rsa", "Policy ↔ Behavior"),
            ("policy_return_rsa", "Policy ↔ Return"),
            ("value_behavior_rsa", "Value ↔ Behavior"),
            ("value_return_rsa", "Value ↔ Return"),
        ),
        "Spearman RSA", fixed_limits=(-1.05, 1.05),
    )
    save_rsa_grouped_figure(
        rows, figure_dir / "rsa_asymmetry.png",
        f"RSA policy–value selectivity · {layout_title}",
        (
            ("policy_behavior_minus_return", "Policy: behavior − return"),
            ("value_return_minus_behavior", "Value: return − behavior"),
            ("rsa_asymmetry_score", "Combined asymmetry"),
        ),
        "RSA selectivity",
    )


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
        save_rsa_figures(rsa_rows, output_dir, pair_file.parent.name)
        print(f"Saved fixed-pair metrics to {output_dir}")


if __name__ == "__main__":
    main()
