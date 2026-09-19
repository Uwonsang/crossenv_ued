import argparse
import hashlib
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from plot_modified_wall_xp import COLORS


MODEL_GROUPS = (
    "ippo_empty",
    "ippo_wall_a",
    "e3t_empty",
    "e3t_wall_a",
    "cec",
    "idaac_cec",
)

ALGORITHM_GROUPS = {
    "IPPO": ("ippo_empty", "ippo_wall_a"),
    "E3T": ("e3t_empty", "e3t_wall_a"),
    "CEC": ("cec",),
    "DCEC": ("idaac_cec",),
}

STRATA = (
    ("heldout_empty", "heldout_seen_walls", "empty"),
    ("heldout_wall_a", "heldout_seen_walls", "wall_a"),
    ("anti_diagonal", "anti_diagonal", None),
    ("horizontal", "horizontal", None),
    ("vertical", "vertical", None),
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Build a balanced 100-task XP subset from completed 400-task XP."
    )
    parser.add_argument(
        "--source-dir",
        type=Path,
        default=Path("baselines/CEC_UED/results/unseen_procedural_xp"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(
            "baselines/CEC_UED/results/unseen_procedural_xp/balanced_unseen100_xp"
        ),
    )
    parser.add_argument("--sample-seed", type=int, default=20260920)
    parser.add_argument("--tasks-per-stratum", type=int, default=20)
    return parser.parse_args()


def select_tasks(source_dir, sample_seed, tasks_per_stratum):
    bank_path = source_dir / "procedural_500.npz"
    if not bank_path.exists():
        raise FileNotFoundError(f"Missing procedural state bank: {bank_path}")
    with np.load(bank_path, allow_pickle=False) as bank:
        task_blocks = bank["task_blocks"]
        eval_layouts = bank["eval_layouts"]

    child_seeds = np.random.SeedSequence(sample_seed).spawn(len(STRATA))
    rows = []
    balanced_task_id = 0
    for (stratum, task_block, eval_layout), child_seed in zip(STRATA, child_seeds):
        mask = task_blocks == task_block
        if eval_layout is not None:
            mask &= eval_layouts == eval_layout
        candidates = np.flatnonzero(mask)
        if len(candidates) < tasks_per_stratum:
            raise ValueError(
                f"{stratum} has only {len(candidates)} candidates; "
                f"cannot sample {tasks_per_stratum}"
            )
        selected = np.sort(
            np.random.default_rng(child_seed).choice(
                candidates, size=tasks_per_stratum, replace=False
            )
        )
        for state_id in selected:
            rows.append(
                {
                    "balanced_task_id": balanced_task_id,
                    "source_state_id": int(state_id),
                    "stratum": stratum,
                    "task_block": str(task_blocks[state_id]),
                    "eval_layout": str(eval_layouts[state_id]),
                }
            )
            balanced_task_id += 1
    return pd.DataFrame(rows), bank_path


def summarize(episodes):
    ordered = (
        episodes.groupby(["policy_1", "policy_2"], as_index=False)
        .agg(
            reward_mean=("reward", "mean"),
            normalized_return_mean=("normalized_return", "mean"),
            success_rate=("success", "mean"),
            num_tasks=("balanced_task_id", "count"),
        )
    )
    ordered["seed_pair"] = ordered.apply(
        lambda row: f"{int(min(row.policy_1, row.policy_2))}x"
        f"{int(max(row.policy_1, row.policy_2))}",
        axis=1,
    )
    pairs = (
        ordered.groupby("seed_pair", as_index=False)
        .agg(
            reward_mean=("reward_mean", "mean"),
            normalized_return_mean=("normalized_return_mean", "mean"),
            success_rate=("success_rate", "mean"),
        )
    )
    summary = pd.DataFrame(
        [
            {
                "reward_mean": pairs["reward_mean"].mean(),
                "reward_sem": pairs["reward_mean"].sem(),
                "normalized_return_mean": pairs["normalized_return_mean"].mean(),
                "normalized_return_sem": pairs["normalized_return_mean"].sem(),
                "success_rate": pairs["success_rate"].mean(),
                "num_seed_pairs": len(pairs),
            }
        ]
    )
    return ordered, pairs, summary


def add_metadata(frame, model_group, algorithm, train_map):
    frame.insert(0, "model_group", model_group)
    frame.insert(1, "algorithm", algorithm)
    frame.insert(2, "num_envs", 256)
    frame.insert(3, "budget", "65K")
    frame.insert(4, "train_map", train_map)
    frame.insert(5, "task_count", 100)


def build_group_results(source_dir, output_dir, selection, model_group):
    source_path = source_dir / f"{model_group}_numenv256_tasks400_episodes.csv"
    if not source_path.exists():
        raise FileNotFoundError(f"Missing completed XP episodes: {source_path}")
    source = pd.read_csv(source_path)
    source = source.rename(columns={"state_id": "source_state_id"})
    source["source_task_count"] = source.pop("task_count")
    episodes = source.merge(
        selection[["balanced_task_id", "source_state_id", "stratum"]],
        on="source_state_id",
        how="inner",
        validate="many_to_one",
    )
    episodes.insert(4, "task_count", 100)
    episodes = episodes.sort_values(
        ["policy_1", "policy_2", "balanced_task_id"]
    ).reset_index(drop=True)
    expected_rows = 30 * len(selection)
    if len(episodes) != expected_rows:
        raise ValueError(
            f"Expected {expected_rows} rows for {model_group}, found {len(episodes)}"
        )

    algorithm = str(episodes["algorithm"].iloc[0])
    train_map = str(episodes["train_map"].iloc[0])
    ordered, pairs, summary = summarize(episodes)
    for frame in (ordered, pairs, summary):
        add_metadata(frame, model_group, algorithm, train_map)

    prefix = output_dir / f"{model_group}_numenv256_balanced100"
    episodes.to_csv(f"{prefix}_episodes.csv", index=False)
    ordered.to_csv(f"{prefix}_ordered_pairs.csv", index=False)
    pairs.to_csv(f"{prefix}_pairs.csv", index=False)
    summary.to_csv(f"{prefix}_summary.csv", index=False)
    return summary


def build_plot_values(output_dir):
    rows = []
    for algorithm, model_groups in ALGORITHM_GROUPS.items():
        frames = []
        for model_group in model_groups:
            path = output_dir / f"{model_group}_numenv256_balanced100_pairs.csv"
            frames.append(pd.read_csv(path))
        pairs = (
            pd.concat(frames, ignore_index=True)
            .groupby("seed_pair", as_index=False)
            .agg(
                reward=("reward_mean", "mean"),
                normalized=("normalized_return_mean", "mean"),
                success_rate=("success_rate", "mean"),
            )
        )
        rows.append(
            {
                "algorithm": algorithm,
                "num_envs": 256,
                "budget": "65K",
                "task_count": 100,
                "reward_mean": pairs["reward"].mean(),
                "reward_sem": pairs["reward"].sem(),
                "normalized_mean": pairs["normalized"].mean(),
                "normalized_sem": pairs["normalized"].sem(),
                "success_rate": pairs["success_rate"].mean(),
                "num_seed_pairs": len(pairs),
            }
        )
    return pd.DataFrame(rows)


def algorithm_color(algorithm):
    return COLORS["DCEC_65K"] if algorithm == "DCEC" else COLORS[algorithm]


def draw_plot(plot_values, metric, output_dir):
    algorithms = list(ALGORITHM_GROUPS)
    frame = plot_values.set_index("algorithm")
    fig, axis = plt.subplots(figsize=(7.2, 4.4))
    axis.bar(
        algorithms,
        [frame.loc[name, f"{metric}_mean"] for name in algorithms],
        yerr=[frame.loc[name, f"{metric}_sem"] for name in algorithms],
        capsize=4,
        color=[algorithm_color(name) for name in algorithms],
        edgecolor="#222222",
        linewidth=0.8,
        error_kw={"elinewidth": 1.1, "capthick": 1.1},
    )
    axis.axhline(0, color="#222222", linewidth=0.8)
    axis.set_axisbelow(True)
    axis.grid(axis="y", color="#D9D9D9", linewidth=0.8, alpha=0.75)
    axis.set_title("100 balanced procedural tasks", fontweight="bold")
    axis.set_ylabel("Normalized XP reward" if metric == "normalized" else "Mean XP reward")
    if metric == "normalized":
        axis.set_ylim(-0.55, 1.05)
    fig.tight_layout()

    stem = f"balanced_unseen100_{metric}"
    png_path = output_dir / f"{stem}.png"
    pdf_path = output_dir / f"{stem}.pdf"
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)
    return png_path, pdf_path


def main():
    args = parse_args()
    if args.tasks_per_stratum != 20:
        raise ValueError("tasks-per-stratum must be 20 for the requested 100-task set")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    selection, bank_path = select_tasks(
        args.source_dir, args.sample_seed, args.tasks_per_stratum
    )
    selection_path = args.output_dir / "selected_tasks.csv"
    selection.to_csv(selection_path, index=False)
    selection_sha256 = hashlib.sha256(selection.to_csv(index=False).encode()).hexdigest()
    manifest = {
        "sample_seed": args.sample_seed,
        "tasks_per_stratum": args.tasks_per_stratum,
        "total_tasks": len(selection),
        "source_bank": str(bank_path),
        "selection_sha256": selection_sha256,
        "stratum_counts": selection["stratum"].value_counts().sort_index().to_dict(),
        "selected_state_ids": {
            stratum: selection.loc[
                selection["stratum"] == stratum, "source_state_id"
            ].tolist()
            for stratum, _, _ in STRATA
        },
    }
    with open(args.output_dir / "selection_manifest.json", "w", encoding="utf-8") as file:
        json.dump(manifest, file, indent=2)

    summaries = []
    for model_group in MODEL_GROUPS:
        summaries.append(
            build_group_results(
                args.source_dir, args.output_dir, selection, model_group
            )
        )
    pd.concat(summaries, ignore_index=True).to_csv(
        args.output_dir / "model_group_summary.csv", index=False
    )

    plot_values = build_plot_values(args.output_dir)
    plot_values_path = args.output_dir / "plot_values.csv"
    plot_values.to_csv(plot_values_path, index=False)
    output_paths = []
    for metric in ("reward", "normalized"):
        output_paths.extend(draw_plot(plot_values, metric, args.output_dir))

    print(json.dumps(manifest, indent=2))
    print(plot_values.to_string(index=False))
    print(f"Saved results and {len(output_paths)} plot files to {args.output_dir}")


if __name__ == "__main__":
    main()
