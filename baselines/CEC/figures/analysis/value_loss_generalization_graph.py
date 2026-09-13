"""Plot value loss against held-out evaluation return from W&B runs.

One point represents one NUM_ENVS setting. Independent seeds are first
averaged within each (model, NUM_ENVS) group, matching the presentation used
in value-loss/generalization correlation figures. The script writes both the
run-level data and the seed-aggregated data used for plotting.
"""
from __future__ import annotations

import argparse
import csv
import math
import re
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import wandb


DEFAULT_ENTITY = "overcooked_ai"
DEFAULT_PROJECT = "crossenv_ICLR"
DEFAULT_TARGET_ENV_STEPS = 3_000_000_000
REPOSITORY_ROOT = Path(__file__).resolve().parents[4]
DEFAULT_OUTPUT_DIR = REPOSITORY_ROOT / "artifacts" / "value_loss_generalization"

LAYOUTS = [
    "cramped_room_9",
    "asymm_advantages_9",
    "coord_ring_9",
    "counter_circuit_9",
    "forced_coord_9",
]
LAYOUT_LABELS = {
    "mean": "Mean",
    "cramped_room_9": "Cramped Room",
    "asymm_advantages_9": "Asymmetric Advantages",
    "coord_ring_9": "Coordination Ring",
    "counter_circuit_9": "Counter Circuit",
    "forced_coord_9": "Forced Coordination",
}
MODEL_LABELS = {
    "CEC": "CEC",
    "CEC_IDAAC": "CEC-IDAAC",
}
COLORS = {
    32: "#1f77b4",
    64: "#2ca02c",
    128: "#ff7f0e",
    256: "#d62728",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--entity", default=DEFAULT_ENTITY)
    parser.add_argument("--project", default=DEFAULT_PROJECT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--target-env-steps", type=int, default=DEFAULT_TARGET_ENV_STEPS,
        help="Compare runs at this common environment-step horizon.",
    )
    parser.add_argument(
        "--tail-fraction", type=float, default=0.10,
        help="Average metrics over the final fraction of the target horizon.",
    )
    parser.add_argument(
        "--aggregation", choices=("final-window", "last"),
        default="final-window",
        help=(
            "Use a mean over the final window, or the last valid logged value "
            "at or before --target-env-steps."
        ),
    )
    parser.add_argument(
        "--min-progress", type=float, default=0.95,
        help="Minimum fraction of target steps a run must have completed.",
    )
    parser.add_argument(
        "--model-names", nargs="+", default=["CEC", "CEC_IDAAC"],
        help="Exact canonical model_name values to include.",
    )
    parser.add_argument(
        "--num-envs", nargs="*", type=int, default=[32, 64, 128, 256],
    )
    parser.add_argument(
        "--num-minibatches", type=int, default=2,
        help="Only include runs whose NUM_MINIBATCHES equals this value.",
    )
    parser.add_argument("--seeds", nargs="*", type=int, default=None)
    parser.add_argument(
        "--run-name-contains", default=None,
        help="Optional substring filter for W&B run names.",
    )
    parser.add_argument(
        "--include-running", action="store_true",
        help="Also include running runs that satisfy --min-progress.",
    )
    parser.add_argument(
        "--all-matching-runs", action="store_true",
        help=(
            "Keep duplicate runs. By default only the newest run for each "
            "model/NUM_ENVS/seed/training-layout combination is used."
        ),
    )
    parser.add_argument(
        "--history-samples", type=int, default=10_000,
        help="Maximum sampled history rows fetched per run.",
    )
    parser.add_argument(
        "--show-error-bars", action="store_true",
        help="Show across-seed standard deviations on both axes.",
    )
    parser.add_argument(
        "--plot-mode", choices=("aggregate", "individual", "both"),
        default="aggregate",
        help=(
            "Plot the original seed-averaged points, all individual seed runs, "
            "or both. Individual plots use a separate filename."
        ),
    )
    parser.add_argument(
        "--refresh-wandb", action="store_true",
        help="Ignore an existing compatible run CSV and fetch W&B again.",
    )
    return parser.parse_args()


def finite_float(value):
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def integer_config(config, *keys):
    for key in keys:
        if key not in config:
            continue
        value = finite_float(config[key])
        if value is not None:
            return int(value)
    return None


def canonical_model(run) -> str:
    configured = run.config.get("model_name", run.config.get("MODEL_NAME"))
    if configured:
        text = str(configured).upper().replace("-", "_")
    else:
        text = str(run.name).upper().replace("-", "_")
    if "IDAAC" in text or "IDDAC" in text:
        return "CEC_IDAAC"
    if "CEC" in text or "IPPO" in text:
        return "CEC"
    return re.sub(r"[^A-Z0-9_]+", "_", text).strip("_") or "UNKNOWN"


def configured_model(run) -> str | None:
    """Return the exact normalized model_name stored in the run config."""
    value = run.config.get("model_name", run.config.get("MODEL_NAME"))
    if value is None:
        return None
    return str(value).upper().replace("-", "_")


def training_layout(run) -> str:
    env_kwargs = run.config.get("ENV_KWARGS", {})
    if isinstance(env_kwargs, dict):
        return str(env_kwargs.get("layout", "unknown"))
    return "unknown"


def max_logged_step(run):
    for key in ("env_step", "_step"):
        value = finite_float(run.summary.get(key))
        if value is not None:
            return value
    return None


def select_runs(args: argparse.Namespace):
    api = wandb.Api()
    candidates = []
    allowed_states = {"finished", "running"} if args.include_running else {"finished"}

    for run in api.runs(f"{args.entity}/{args.project}"):
        if run.state not in allowed_states:
            continue
        if args.run_name_contains and args.run_name_contains not in run.name:
            continue
        exact_model = configured_model(run)
        model = canonical_model(run)
        # When model_name exists, require an exact match so variants such as
        # CEC_WD and CEC_IDAAC_SMALL cannot be silently grouped as base models.
        if exact_model is not None:
            if exact_model not in args.model_names:
                continue
            model = exact_model
        elif model not in args.model_names:
            continue
        num_envs = integer_config(run.config, "NUM_ENVS", "num_envs")
        num_minibatches = integer_config(
            run.config, "NUM_MINIBATCHES", "num_minibatches"
        )
        seed = integer_config(run.config, "SEED", "seed")
        if num_envs is None or num_minibatches is None or seed is None:
            continue
        if args.num_envs and num_envs not in args.num_envs:
            continue
        if num_minibatches != args.num_minibatches:
            continue
        if args.seeds is not None and seed not in args.seeds:
            continue
        configured_steps = integer_config(
            run.config, "TOTAL_TIMESTEPS", "total_timesteps"
        )
        # Do not mix experiments trained with a different intended horizon.
        if configured_steps is not None and configured_steps != args.target_env_steps:
            continue
        logged_step = max_logged_step(run)
        progress = (
            logged_step / args.target_env_steps if logged_step is not None else 0.0
        )
        if progress < args.min_progress:
            print(
                f"Skipping incomplete run {run.id} ({run.name}): "
                f"progress={100.0 * progress:.1f}%"
            )
            continue
        candidates.append(
            {
                "run": run,
                "model": model,
                "num_envs": num_envs,
                "num_minibatches": num_minibatches,
                "seed": seed,
                "training_layout": training_layout(run),
                "configured_steps": configured_steps,
                "max_logged_step": logged_step,
                "progress": progress,
            }
        )

    if not args.all_matching_runs:
        newest = {}
        for item in candidates:
            key = (
                item["model"], item["num_envs"], item["seed"],
                item["training_layout"],
            )
            run = item["run"]
            recency = (str(getattr(run, "created_at", "") or ""), run.id)
            if key not in newest or recency > newest[key][0]:
                newest[key] = (recency, item)
        candidates = [value[1] for value in newest.values()]

    return sorted(
        candidates,
        key=lambda item: (item["model"], item["num_envs"], item["seed"]),
    )


def fetch_run_rows(args: argparse.Namespace):
    eval_names = ["mean", *LAYOUTS]
    eval_keys = {name: f"eval/{name}" for name in eval_names}
    history_keys = ["env_step", "value_loss", *eval_keys.values()]
    window_start = args.target_env_steps * (1.0 - args.tail_fraction)
    selection_start = window_start if args.aggregation == "final-window" else 0
    rows = []

    for item in select_runs(args):
        run = item["run"]
        print(
            f"Fetching {run.id}: {item['model']}, NUM_ENVS={item['num_envs']}, "
            f"seed={item['seed']}"
        )
        value_losses = []
        eval_values = {name: [] for name in eval_names}
        history = run.history(
            keys=history_keys,
            samples=args.history_samples,
            pandas=False,
        )
        for history_row in history:
            env_step = finite_float(history_row.get("env_step"))
            if env_step is None:
                continue
            if env_step > args.target_env_steps:
                continue
            if args.aggregation == "final-window" and env_step < window_start:
                continue
            value_loss = finite_float(history_row.get("value_loss"))
            if value_loss is not None:
                value_losses.append((env_step, value_loss))
            for name, key in eval_keys.items():
                value = finite_float(history_row.get(key))
                if value is not None:
                    eval_values[name].append((env_step, value))

        if args.aggregation == "last":
            value_losses = [max(value_losses, key=lambda pair: pair[0])[1]] if value_losses else []
            eval_values = {
                name: [max(values, key=lambda pair: pair[0])[1]] if values else []
                for name, values in eval_values.items()
            }
        else:
            value_losses = [value for _, value in value_losses]
            eval_values = {
                name: [value for _, value in values]
                for name, values in eval_values.items()
            }

        if not value_losses:
            print(f"No value_loss in final window: {run.id} ({run.name})")
            continue
        for eval_name, scores in eval_values.items():
            if not scores:
                continue
            rows.append(
                {
                    "run_id": run.id,
                    "run_name": run.name,
                    "created_at": str(getattr(run, "created_at", "") or ""),
                    "model": item["model"],
                    "num_envs": item["num_envs"],
                    "num_minibatches": item["num_minibatches"],
                    "aggregation": args.aggregation,
                    "seed": item["seed"],
                    "training_layout": item["training_layout"],
                    "target_env_steps": args.target_env_steps,
                    "window_start_env_step": int(selection_start),
                    "max_logged_env_step": int(item["max_logged_step"]),
                    "eval_layout": eval_name,
                    "value_loss_mean": float(np.mean(value_losses)),
                    "value_loss_count": len(value_losses),
                    "eval_score_mean": float(np.mean(scores)),
                    "eval_score_count": len(scores),
                }
            )
        if not any(eval_values.values()):
            print(f"No eval/* metric in final window: {run.id} ({run.name})")
    return rows


def aggregate_seeds(run_rows):
    grouped = defaultdict(list)
    for row in run_rows:
        grouped[(row["model"], row["num_envs"], row["eval_layout"])].append(row)

    aggregated = []
    for (model, num_envs, eval_layout), group in sorted(grouped.items()):
        x = np.asarray([row["value_loss_mean"] for row in group], dtype=float)
        y = np.asarray([row["eval_score_mean"] for row in group], dtype=float)
        aggregated.append(
            {
                "model": model,
                "num_envs": num_envs,
                "eval_layout": eval_layout,
                "value_loss_mean": float(x.mean()),
                "value_loss_std": float(x.std(ddof=1)) if len(x) > 1 else 0.0,
                "eval_score_mean": float(y.mean()),
                "eval_score_std": float(y.std(ddof=1)) if len(y) > 1 else 0.0,
                "num_seeds": len(group),
                "seeds": " ".join(str(row["seed"]) for row in sorted(group, key=lambda r: r["seed"])),
            }
        )
    return aggregated


def correlation(x, y):
    if len(x) < 2 or np.allclose(x, x[0]) or np.allclose(y, y[0]):
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def plot_model(
    model, rows, output_path: Path, show_error_bars: bool, aggregation: str,
):
    panels = ["mean", *LAYOUTS]
    fig, axes = plt.subplots(2, 3, figsize=(11.2, 7.0))
    axes = axes.ravel()

    for ax, layout in zip(axes, panels):
        points = sorted(
            (row for row in rows if row["model"] == model and row["eval_layout"] == layout),
            key=lambda row: row["num_envs"],
        )
        if not points:
            ax.set_visible(False)
            continue
        x = np.asarray([row["value_loss_mean"] for row in points])
        y = np.asarray([row["eval_score_mean"] for row in points])
        xerr = np.asarray([row["value_loss_std"] for row in points])
        yerr = np.asarray([row["eval_score_std"] for row in points])
        x_mid = (float(x.min()) + float(x.max())) / 2.0
        y_mid = (float(y.min()) + float(y.max())) / 2.0

        for point, px, py, pxe, pye in zip(points, x, y, xerr, yerr):
            color = COLORS.get(point["num_envs"], "#1f4e99")
            ax.errorbar(
                px, py,
                xerr=pxe if show_error_bars and pxe > 0 else None,
                yerr=pye if show_error_bars and pye > 0 else None,
                fmt="o", color=color,
                ecolor=color, elinewidth=1.0, capsize=2.5, markersize=5.5,
                zorder=3,
            )
            # Put labels toward the plot interior. This keeps labels such as
            # 32/64 away from the top/right spines and prevents the fitted
            # line from running through their text.
            label_dx = -5 if px > x_mid else 5
            label_dy = -5 if py > y_mid else 5
            ax.annotate(
                str(point["num_envs"]), (px, py),
                xytext=(label_dx, label_dy), textcoords="offset points",
                ha="right" if label_dx < 0 else "left",
                va="top" if label_dy < 0 else "bottom",
                fontsize=8, color=color, zorder=4,
                bbox={"facecolor": "white", "edgecolor": "none", "pad": 0.15,
                      "alpha": 0.85},
            )

        r_value = correlation(x, y)
        if len(x) >= 2 and not np.allclose(x, x[0]):
            slope, intercept = np.polyfit(x, y, 1)
            line_x = np.linspace(float(x.min()), float(x.max()), 100)
            ax.plot(
                line_x, slope * line_x + intercept,
                color="#f28e00", linewidth=2.0, zorder=1,
            )
        r_text = f"r = {r_value:.2f}" if math.isfinite(r_value) else "r = n/a"
        ax.set_title(f"{LAYOUT_LABELS[layout]}     {r_text}", fontsize=11)
        ax.grid(alpha=0.25)
        ax.margins(x=0.06, y=0.08)

    for index, ax in enumerate(axes):
        if not ax.get_visible():
            continue
        if index // 3 == 1:
            x_label = (
                "Value loss (last logged value)"
                if aggregation == "last"
                else "Value loss (final-window mean)"
            )
            ax.set_xlabel(x_label)
        if index % 3 == 0:
            ax.set_ylabel("Evaluation return")

    label = MODEL_LABELS.get(model, model.replace("_", "-"))
    statistic_label = (
        "Last logged metrics"
        if aggregation == "last"
        else "Final-window mean metrics"
    )
    # Draw the heading and subtitle separately.  A multiline suptitle can have
    # inconsistent line spacing across matplotlib/font versions and collide
    # with the first row of subplot titles.
    fig.text(
        0.5, 0.982, f"{label}: Value Loss and Generalization",
        ha="center", va="top", fontsize=14,
    )
    fig.text(
        0.5, 0.948,
        f"{statistic_label}; point labels denote parallel training environments",
        ha="center", va="top", fontsize=12,
    )
    # tight_layout reserves too much space below the two-line figure title.
    # Explicit margins keep the panels compact without clipping axis labels.
    fig.subplots_adjust(
        left=0.075, right=0.985, bottom=0.095, top=0.855,
        wspace=0.30, hspace=0.32,
    )
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def plot_model_individual_runs(
    model, rows, output_path: Path, aggregation: str,
):
    """Plot every seed run separately for each NUM_ENVS setting."""
    panels = ["mean", *LAYOUTS]
    fig, axes = plt.subplots(2, 3, figsize=(11.2, 7.0))
    axes = axes.ravel()

    for ax, layout in zip(axes, panels):
        points = [
            row for row in rows
            if row["model"] == model and row["eval_layout"] == layout
        ]
        if not points:
            ax.set_visible(False)
            continue

        all_x = np.asarray([row["value_loss_mean"] for row in points], dtype=float)
        all_y = np.asarray([row["eval_score_mean"] for row in points], dtype=float)
        r_value = correlation(all_x, all_y)
        if len(all_x) >= 2 and not np.allclose(all_x, all_x[0]):
            slope, intercept = np.polyfit(all_x, all_y, 1)
            line_x = np.linspace(float(all_x.min()), float(all_x.max()), 100)
            ax.plot(
                line_x, slope * line_x + intercept,
                color="#f28e00", linewidth=2.0, zorder=1,
            )

        for num_envs in sorted({row["num_envs"] for row in points}):
            group = [row for row in points if row["num_envs"] == num_envs]
            x = np.asarray([row["value_loss_mean"] for row in group], dtype=float)
            y = np.asarray([row["eval_score_mean"] for row in group], dtype=float)
            color = COLORS.get(num_envs, "#1f4e99")
            ax.scatter(
                x, y, s=27, color=color, alpha=0.62,
                edgecolors="white", linewidths=0.45, zorder=3,
            )

        # The figure heading already identifies these as individual runs, so
        # keep panel titles short enough for the three-column layout.
        r_text = f"r = {r_value:.2f}" if math.isfinite(r_value) else "r = n/a"
        ax.set_title(f"{LAYOUT_LABELS[layout]}     {r_text}", fontsize=11)
        ax.grid(alpha=0.25)
        ax.margins(x=0.06, y=0.08)

    for index, ax in enumerate(axes):
        if not ax.get_visible():
            continue
        if index // 3 == 1:
            x_label = (
                "Value loss (last logged value)"
                if aggregation == "last"
                else "Value loss (final-window mean)"
            )
            ax.set_xlabel(x_label)
        if index % 3 == 0:
            ax.set_ylabel("Evaluation return")

    env_counts = sorted({row["num_envs"] for row in rows if row["model"] == model})
    legend_handles = []
    for num_envs in env_counts:
        legend_handles.append(Line2D(
            [0], [0], marker="o", linestyle="none", markersize=6,
            markerfacecolor=COLORS.get(num_envs, "#1f4e99"),
            markeredgecolor="white", label=f"{num_envs} envs",
        ))
    fig.legend(
        handles=legend_handles, loc="upper center", ncol=len(legend_handles),
        bbox_to_anchor=(0.5, 0.925), frameon=True, fontsize=9,
    )

    label = MODEL_LABELS.get(model, model.replace("_", "-"))
    statistic_label = (
        "Last logged metrics" if aggregation == "last"
        else "Final-window mean metrics"
    )
    fig.text(
        0.5, 0.982, f"{label}: Value Loss and Generalization (Individual Runs)",
        ha="center", va="top", fontsize=14,
    )
    fig.text(
        0.5, 0.949,
        f"{statistic_label}; circles denote individual seed runs",
        ha="center", va="top", fontsize=11,
    )
    fig.subplots_adjust(
        left=0.075, right=0.985, bottom=0.095, top=0.82,
        wspace=0.30, hspace=0.32,
    )
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def write_csv(path: Path, rows):
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


INT_COLUMNS = {
    "num_envs", "num_minibatches", "seed", "target_env_steps",
    "window_start_env_step", "max_logged_env_step", "value_loss_count",
    "eval_score_count", "train_history_count", "eval_history_count",
}
FLOAT_COLUMNS = {
    "value_loss_mean", "eval_score_mean", "train_return_mean",
    "eval_return_mean", "generalization_gap", "relative_gap",
}


def read_run_csv(path: Path):
    with path.open(newline="", encoding="utf-8") as file:
        rows = list(csv.DictReader(file))
    for row in rows:
        for key in INT_COLUMNS & row.keys():
            row[key] = int(float(row[key]))
        for key in FLOAT_COLUMNS & row.keys():
            row[key] = float(row[key])
    return rows


def compatible_cached_rows(path: Path, args: argparse.Namespace):
    """Load and filter a cache only when it covers every requested group."""
    if not path.exists() or args.refresh_wandb:
        return None
    if args.run_name_contains or args.include_running or args.all_matching_runs:
        return None

    rows = read_run_csv(path)
    selected = [
        row for row in rows
        if (
            row.get("aggregation") == args.aggregation
            or (
                not row.get("aggregation")
                and args.aggregation == "final-window"
            )
        )
        and row.get("model") in args.model_names
        and row.get("num_envs") in args.num_envs
        and row.get("num_minibatches") == args.num_minibatches
        and row.get("target_env_steps") == args.target_env_steps
        and (args.seeds is None or row.get("seed") in args.seeds)
    ]
    available_groups = {
        (row["model"], row["num_envs"]) for row in selected
    }
    requested_groups = {
        (model, num_envs)
        for model in args.model_names for num_envs in args.num_envs
    }
    if not selected or not requested_groups.issubset(available_groups):
        return None
    if args.seeds is not None:
        available_seed_groups = {
            (row["model"], row["num_envs"], row["seed"])
            for row in selected
        }
        requested_seed_groups = {
            (model, num_envs, seed)
            for model in args.model_names
            for num_envs in args.num_envs
            for seed in args.seeds
        }
        if not requested_seed_groups.issubset(available_seed_groups):
            return None
    print(f"Using cached W&B data: {path}")
    return selected


def main():
    args = parse_args()
    if args.target_env_steps <= 0:
        raise ValueError("--target-env-steps must be positive.")
    if not 0.0 < args.tail_fraction <= 1.0:
        raise ValueError("--tail-fraction must be in (0, 1].")
    if not 0.0 < args.min_progress <= 1.0:
        raise ValueError("--min-progress must be in (0, 1].")
    if args.history_samples <= 0:
        raise ValueError("--history-samples must be positive.")
    if args.num_minibatches <= 0:
        raise ValueError("--num-minibatches must be positive.")

    args.model_names = [name.upper().replace("-", "_") for name in args.model_names]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    horizon = f"{args.target_env_steps // 1_000_000}m"
    mode = args.aggregation.replace("-", "_")
    suffix = f"{mode}_{horizon}"
    run_csv = args.output_dir / f"value_loss_generalization_runs_{suffix}.csv"
    aggregate_csv = args.output_dir / f"value_loss_generalization_by_num_envs_{suffix}.csv"

    run_rows = compatible_cached_rows(run_csv, args)
    if run_rows is None and args.aggregation == "final-window":
        legacy_run_csv = args.output_dir / (
            f"value_loss_generalization_runs_{horizon}.csv"
        )
        run_rows = compatible_cached_rows(legacy_run_csv, args)
    loaded_from_cache = run_rows is not None
    if run_rows is None:
        print("No compatible CSV cache; fetching W&B data.")
        run_rows = fetch_run_rows(args)
    if not run_rows:
        raise RuntimeError(
            "No matching W&B value_loss/eval data found. Check --project, "
            "--target-env-steps, --model-names, and run completion."
        )
    aggregate_rows = aggregate_seeds(run_rows)

    if not loaded_from_cache:
        write_csv(run_csv, run_rows)
        write_csv(aggregate_csv, aggregate_rows)
        print(f"Saved: {run_csv}")
        print(f"Saved: {aggregate_csv}")

    for model in args.model_names:
        if not any(row["model"] == model for row in aggregate_rows):
            print(f"No plot data for model: {model}")
            continue
        safe_model = model.lower()
        if args.plot_mode in ("aggregate", "both"):
            figure_path = args.output_dir / f"value_loss_vs_eval_{safe_model}_{suffix}.pdf"
            plot_model(
                model, aggregate_rows, figure_path, args.show_error_bars,
                args.aggregation,
            )
            print(f"Saved: {figure_path}")
        if args.plot_mode in ("individual", "both"):
            individual_path = args.output_dir / (
                f"value_loss_vs_eval_{safe_model}_individual_runs_{suffix}.pdf"
            )
            plot_model_individual_runs(
                model, run_rows, individual_path, args.aggregation,
            )
            print(f"Saved: {individual_path}")


if __name__ == "__main__":
    main()
