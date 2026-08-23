import argparse
import shutil
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import yaml


ALGORITHM_FILES = {
    "IPPO": ["ippo_empty_pairs.csv", "ippo_wall_a_pairs.csv"],
    "E3T": ["e3t_empty_pairs.csv", "e3t_wall_a_pairs.csv"],
    "CEC": ["cec_pairs.csv"],
    "CEC+IDAAC": ["idaac_cec_pairs.csv"],
}

MODEL_GROUPS = (
    "ippo_empty",
    "ippo_wall_a",
    "e3t_empty",
    "e3t_wall_a",
    "cec",
    "idaac_cec",
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--results-dir",
        default="baselines/CEC_UED/results/procedural_xp",
    )
    parser.add_argument("--entity", default="overcooked_ai")
    parser.add_argument("--project", default="crossenv_ued_ICLR_dual_destination")
    parser.add_argument("--wandb-mode", default="online")
    parser.add_argument("--download-artifacts", action="store_true")
    return parser.parse_args()


def load_algorithm_results(results_dir, algorithm, filenames):
    frames = []
    for filename in filenames:
        path = results_dir / filename
        if not path.exists():
            raise FileNotFoundError(f"Missing evaluation result: {path}")
        frames.append(pd.read_csv(path))
    frame = pd.concat(frames, ignore_index=True)
    frame = (
        frame.groupby(["split", "seed_pair"], as_index=False)
        .agg(
            normalized_return_mean=("normalized_return_mean", "mean"),
            success_rate=("success_rate", "mean"),
        )
    )
    rows = []
    for split, split_frame in frame.groupby("split"):
        rows.append(
            {
                "algorithm": algorithm,
                "split": split,
                "normalized_return_mean": float(
                    split_frame["normalized_return_mean"].mean()
                ),
                "normalized_return_sem": float(
                    split_frame["normalized_return_mean"].sem()
                ),
                "success_rate": float(split_frame["success_rate"].mean()),
                "num_seed_pair_samples": int(len(split_frame)),
            }
        )
    return rows


def download_evaluation_artifacts(args, results_dir):
    import wandb

    api = wandb.Api()
    artifact_root = results_dir / "wandb_artifacts"
    for model_group in MODEL_GROUPS:
        artifact = api.artifact(
            f"{args.entity}/{args.project}/"
            f"modified-wall-procedural-xp-{model_group}:latest"
        )
        download_dir = Path(
            artifact.download(root=str(artifact_root / model_group))
        )
        source = download_dir / f"{model_group}_pairs.csv"
        if not source.exists():
            matches = list(download_dir.rglob(f"{model_group}_pairs.csv"))
            if len(matches) != 1:
                raise FileNotFoundError(
                    f"Artifact for {model_group} does not contain its pair results"
                )
            source = matches[0]
        shutil.copy2(source, results_dir / source.name)


def make_figure(summary, output_png, output_pdf):
    algorithms = list(ALGORITHM_FILES)
    split_titles = {
        "fixed": "Fixed Tasks",
        "procedural": "100 Procedural Tasks",
    }
    colors = ["#4C78A8", "#F58518", "#54A24B", "#B279A2"]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
    for axis, split in zip(axes, ("fixed", "procedural")):
        split_frame = summary[summary["split"] == split].set_index("algorithm")
        means = [split_frame.loc[name, "normalized_return_mean"] for name in algorithms]
        sems = [split_frame.loc[name, "normalized_return_sem"] for name in algorithms]
        axis.bar(algorithms, means, yerr=sems, capsize=4, color=colors)
        axis.set_title(split_titles[split])
        axis.set_xlabel("Algorithm")
        axis.grid(axis="y", alpha=0.25)
        axis.tick_params(axis="x", rotation=15)
    axes[0].set_ylabel("Normalized XP Return")
    fig.tight_layout()
    fig.savefig(output_png, dpi=300, bbox_inches="tight")
    fig.savefig(output_pdf, bbox_inches="tight")
    plt.close(fig)


def log_wandb(args, summary, output_paths):
    if args.wandb_mode == "disabled":
        return
    import wandb

    if args.wandb_mode == "online":
        with open("private.yaml", encoding="utf-8") as file:
            private_info = yaml.load(file, Loader=yaml.FullLoader)
        wandb.login(key=private_info["wandb_key"])
    run = wandb.init(
        entity=args.entity,
        project=args.project,
        group="FINAL_XP_BAR_PLOTS",
        name="FIGURE3_MODIFIED_WALL",
        job_type="final_plot",
        tags=["final_xp", "bar_plot", "procedural_100"],
        mode=args.wandb_mode,
        config=vars(args),
    )
    table = wandb.Table(dataframe=summary)
    fixed_table = wandb.Table(
        dataframe=summary[summary["split"] == "fixed"]
    )
    procedural_table = wandb.Table(
        dataframe=summary[summary["split"] == "procedural"]
    )
    run.log(
        {
            "final_xp/bar_plot": wandb.Image(str(output_paths["png"])),
            "final_xp/summary": table,
            "final_xp/fixed_bar": wandb.plot.bar(
                fixed_table,
                "algorithm",
                "normalized_return_mean",
                title="Fixed Tasks",
            ),
            "final_xp/procedural_bar": wandb.plot.bar(
                procedural_table,
                "algorithm",
                "normalized_return_mean",
                title="100 Procedural Tasks",
            ),
        }
    )
    artifact = wandb.Artifact("modified-wall-final-xp-figure", type="figure")
    for path in output_paths.values():
        artifact.add_file(str(path))
    run.log_artifact(artifact)
    run.finish()


def main():
    args = parse_args()
    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    if args.download_artifacts:
        download_evaluation_artifacts(args, results_dir)
    rows = []
    for algorithm, filenames in ALGORITHM_FILES.items():
        rows.extend(
            load_algorithm_results(results_dir, algorithm, filenames)
        )
    summary = pd.DataFrame(rows)
    summary_path = results_dir / "final_xp_bar_values.csv"
    output_png = results_dir / "figure3_modified_wall.png"
    output_pdf = results_dir / "figure3_modified_wall.pdf"
    summary.to_csv(summary_path, index=False)
    make_figure(summary, output_png, output_pdf)
    print(summary.to_string(index=False))
    print(f"Saved {output_png}")
    log_wandb(
        args,
        summary,
        {"png": output_png, "pdf": output_pdf, "csv": summary_path},
    )


if __name__ == "__main__":
    main()
