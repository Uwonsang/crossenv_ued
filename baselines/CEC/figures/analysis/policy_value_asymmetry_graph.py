"""Plot the outputs of policy_value_asymmetry.py."""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


MODEL_LABELS = {"CEC": "CEC", "CEC_IDAAC": "DCEC"}
COLORS = {"CEC": "#117733", "CEC_IDAAC": "#0072B2"}


def seed_summary(frame: pd.DataFrame, column: str) -> pd.DataFrame:
    """Average pairs within a seed, then compute uncertainty across seeds."""
    per_seed = frame.groupby(["model", "seed"], as_index=False)[column].mean()
    summary = per_seed.groupby("model")[column].agg(["mean", "count", "std"])
    summary["sem"] = summary["std"] / np.sqrt(summary["count"])
    summary.loc[summary["count"] < 2, "sem"] = np.nan
    return summary


def ordered_models(frame: pd.DataFrame) -> list[str]:
    preferred = [name for name in ("CEC", "CEC_IDAAC") if name in set(frame["model"])]
    return preferred + sorted(set(frame["model"]) - set(preferred))


def bars(ax, frame: pd.DataFrame, column: str, ylabel: str, models: list[str]):
    summary = seed_summary(frame, column).reindex(models)
    x = np.arange(len(models))
    errors = summary["sem"].fillna(0).to_numpy()
    ax.bar(x, summary["mean"], yerr=errors,
           color=[COLORS.get(model, ".45") for model in models],
           edgecolor="black", linewidth=.7, capsize=3)
    ax.set_xticks(x)
    ax.set_xticklabels([MODEL_LABELS.get(model, model) for model in models])
    ax.set_ylabel(ylabel)
    ax.grid(axis="y", alpha=.25)


def plot_summary(frame: pd.DataFrame, output: Path):
    models = ordered_models(frame)
    fig, axes = plt.subplots(1, 4, figsize=(13.4, 3.2))
    bars(axes[0], frame, "policy_js_nats", "Policy JS divergence (nats)", models)
    bars(axes[1], frame, "both_policy_oracle_correct", "Oracle-action accuracy", models)
    axes[1].set_ylim(0, 1.05)

    metric_labels = (
        ("value_delta", "Predicted value"),
        ("mc_delta", "MC return"),
        ("first_delivery_oracle_delta", "First-delivery oracle"),
    )
    width = .8 / len(metric_labels)
    x = np.arange(len(models))
    for offset, (column, label) in enumerate(metric_labels):
        summary = seed_summary(frame, column).reindex(models)
        positions = x + (offset - (len(metric_labels) - 1) / 2) * width
        axes[2].bar(positions, summary["mean"], width=width, label=label,
                    yerr=summary["sem"].fillna(0), capsize=2,
                    edgecolor="black", linewidth=.5)
    axes[2].axhline(0, color="black", linewidth=.7)
    axes[2].set_xticks(x)
    axes[2].set_xticklabels([MODEL_LABELS.get(model, model) for model in models])
    axes[2].set_ylabel("Short − long difference")
    axes[2].grid(axis="y", alpha=.25)
    axes[2].legend(frameon=False, fontsize=8)

    representation_labels = (
        ("actor_penultimate_cosine_distance", "Actor"),
        ("critic_penultimate_cosine_distance", "Critic"),
    )
    width = .8 / len(representation_labels)
    for offset, (column, label) in enumerate(representation_labels):
        summary = seed_summary(frame, column).reindex(models)
        positions = x + (offset - .5) * width
        axes[3].bar(positions, summary["mean"], width=width, label=label,
                    yerr=summary["sem"].fillna(0), capsize=2,
                    edgecolor="black", linewidth=.5)
    axes[3].set_xticks(x)
    axes[3].set_xticklabels([MODEL_LABELS.get(model, model) for model in models])
    axes[3].set_ylabel("Cosine distance across layouts")
    axes[3].grid(axis="y", alpha=.25)
    axes[3].legend(frameon=False, fontsize=8)

    axes[0].set_title("(a) Policy similarity", fontweight="bold")
    axes[1].set_title("(b) Oracle action", fontweight="bold")
    axes[2].set_title("(c) Value sensitivity", fontweight="bold")
    axes[3].set_title("(d) Representation sensitivity", fontweight="bold")
    fig.tight_layout()
    fig.savefig(output.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(output.with_suffix(".png"), dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_pairs(frame: pd.DataFrame, output: Path):
    fig, ax = plt.subplots(figsize=(5.2, 4.0))
    for model in ordered_models(frame):
        subset = frame[frame["model"] == model]
        ax.scatter(subset["policy_js_nats"], subset["value_delta"], s=34,
                   alpha=.72, color=COLORS.get(model),
                   label=MODEL_LABELS.get(model, model))
    ax.axhline(0, color="black", linewidth=.7)
    ax.set_xlabel("Policy JS divergence (nats)")
    ax.set_ylabel("Predicted value difference (short − long)")
    ax.set_title("Policy–value asymmetry by state pair", fontweight="bold")
    ax.grid(alpha=.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(output.with_suffix(".png"), dpi=300, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metrics", type=Path, required=True,
                        help="paired_metrics.csv created by policy_value_asymmetry.py")
    parser.add_argument("--output-dir", type=Path,
                        help="Default: the metrics file directory")
    args = parser.parse_args()
    if not args.metrics.is_file():
        parser.error(f"Metrics file does not exist: {args.metrics}")
    output_dir = args.output_dir or args.metrics.parent
    output_dir.mkdir(parents=True, exist_ok=True)

    frame = pd.read_csv(args.metrics)
    required = {
        "model", "seed", "pair", "policy_js_nats", "policy_argmax_same",
        "both_policy_oracle_correct", "value_delta", "mc_delta",
        "first_delivery_oracle_delta", "actor_penultimate_cosine_distance",
        "critic_penultimate_cosine_distance",
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        parser.error("Missing metrics columns: " + ", ".join(missing))
    frame["policy_argmax_same"] = frame["policy_argmax_same"].astype(float)
    frame["both_policy_oracle_correct"] = frame[
        "both_policy_oracle_correct"
    ].astype(float)

    plot_summary(frame, output_dir / "policy_value_asymmetry_summary")
    plot_pairs(frame, output_dir / "policy_value_asymmetry_pairs")
    print(f"Saved plots to {output_dir}")


if __name__ == "__main__":
    main()
