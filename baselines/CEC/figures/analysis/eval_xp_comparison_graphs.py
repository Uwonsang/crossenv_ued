"""Generate the NUM_ENVS=256 XP comparison figures for the ICLR study."""
from __future__ import annotations

import argparse
from pathlib import Path

from eval_xp_model_graph import fetch_model_histories, plot_eval_xp_by_model


ENTITY = "overcooked_ai"
PROJECT = "crossenv_ICLR"
NUM_ENVS = 256
SEEDS = (0, 1)
OUTPUT_DIR = (
    Path(__file__).parent.parent / "results" / "eval_xp_comparisons"
)

BASELINE_MODELS = ["CEC", "CEC_IDAAC", "CEC_IDAAC_POP"]
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
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--smooth-window", type=int, default=1)
    return parser.parse_args()


def make_figure(
    entity,
    project,
    model_names,
    seeds_by_model,
    title,
    output,
    smooth_window,
    label_suffixes=None,
):
    histories, run_counts = fetch_model_histories(
        entity=entity,
        project=project,
        model_names=model_names,
        num_envs=NUM_ENVS,
        seeds_by_model=seeds_by_model,
    )
    plot_eval_xp_by_model(
        histories=histories,
        run_counts=run_counts,
        model_names=model_names,
        smooth_window=smooth_window,
        min_run_fraction=1.0,
        out_path=output,
        title=title,
        label_suffixes=label_suffixes,
    )


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    make_figure(
        entity=args.entity,
        project=args.project,
        model_names=BASELINE_MODELS,
        seeds_by_model={name: SEEDS for name in BASELINE_MODELS},
        title="XP Performance: Baseline vs Method (NUM_ENVS=256)",
        output=args.output_dir / "xp_baseline_vs_method_num_envs256.png",
        smooth_window=args.smooth_window,
    )
    make_figure(
        entity=args.entity,
        project=args.project,
        model_names=GENERALIZATION_MODELS,
        seeds_by_model=GENERALIZATION_SEEDS,
        title="XP Performance: Generalization Techniques (NUM_ENVS=256)",
        output=args.output_dir / "xp_generalization_num_envs256.png",
        smooth_window=args.smooth_window,
        label_suffixes={
            "CEC_LAYERNORM": " [incomplete: 4 evals]",
            "CEC_BATCHNORM": " [seeds 4,5]",
        },
    )


if __name__ == "__main__":
    main()
