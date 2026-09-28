"""Generate an XP generalization-technique comparison figure."""
from __future__ import annotations

import argparse
from pathlib import Path

from eval_xp_model_graph import fetch_model_histories, plot_eval_xp_by_model


ENTITY = "overcooked_ai"
PROJECT = "crossenv_ICLR"
NUM_ENVS = 256
OUTPUT_DIR = (
    Path(__file__).parent.parent / "results" / "eval_xp_comparisons"
)

GENERALIZATION_MODELS = [
    "CEC", "CEC_WD", "CEC_LAYERNORM", "CEC_BATCHNORM"
]
GENERALIZATION_SEEDS = {
    "CEC": (0, 1),
    "CEC_WD": (0, 1),
    "CEC_LAYERNORM": (0, 1),
    "CEC_BATCHNORM": (4, 5),
}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--entity", default=ENTITY)
    parser.add_argument("--project", default=PROJECT)
    parser.add_argument("--num-envs", type=int, default=NUM_ENVS)
    parser.add_argument("--seeds", type=int, nargs="+")
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--smooth-window", type=int, default=1)
    return parser.parse_args()


def main():
    args = parse_args()
    if args.seeds:
        seeds_by_model = {
            model_name: tuple(args.seeds)
            for model_name in GENERALIZATION_MODELS
        }
        seed_title = f"; SEEDS={','.join(str(seed) for seed in args.seeds)}"
        label_suffixes = None
    else:
        seeds_by_model = GENERALIZATION_SEEDS
        seed_title = ""
        label_suffixes = {"CEC_BATCHNORM": " [seeds 4,5]"}
        if args.num_envs == 256:
            label_suffixes["CEC_LAYERNORM"] = " [incomplete: 4 evals]"

    histories, _ = fetch_model_histories(
        entity=args.entity,
        project=args.project,
        model_names=GENERALIZATION_MODELS,
        num_envs=args.num_envs,
        seeds_by_model=seeds_by_model,
    )
    plot_eval_xp_by_model(
        histories=histories,
        model_names=GENERALIZATION_MODELS,
        smooth_window=args.smooth_window,
        min_run_fraction=1.0,
        out_path=(
            args.output_dir
            / f"xp_generalization_num_envs{args.num_envs}.png"
        ),
        title=(
            "XP Performance: Generalization Techniques "
            f"(NUM_ENVS={args.num_envs}{seed_title})"
        ),
        label_suffixes=label_suffixes,
        num_envs_values=(args.num_envs,),
    )


if __name__ == "__main__":
    main()
