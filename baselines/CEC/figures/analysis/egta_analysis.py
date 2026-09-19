"""Role-averaged, shared-reward EGTA from cross_algo.py evaluation CSVs."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp
from scipy.special import softmax


def display_name(model):
    """Use publication names while preserving raw CSV model identifiers."""
    if model.startswith("CEC_IDAAC"):
        name = model.replace("CEC_IDAAC", "DCEC", 1)
        if "_envs" in name:
            family, count = name.split("_envs", 1)
            batch_label = {
                "32": "8K",
                "64": "16K",
                "128": "32K",
                "256": "65K",
            }.get(count, count)
            return f"{family}({batch_label})"
        return name
    if model.startswith("CEC_envs"):
        return "CEC"
    return model


def build_payoffs(frame, models, xp_only=False):
    """Average trajectories, seed pairs, then layouts with equal weights."""
    keys = ["layout", "algo_1", "algo_2", "seed_1", "seed_2", "trajectory"]
    missing = set(keys + ["reward"]) - set(frame.columns)
    if missing:
        raise ValueError(f"Missing CSV columns: {sorted(missing)}")
    frame = frame.loc[
        frame.algo_1.isin(models) & frame.algo_2.isin(models)
    ].copy()
    if xp_only:
        frame = frame.loc[~((frame.algo_1 == frame.algo_2)
                            & (frame.seed_1 == frame.seed_2))]
    if frame.empty or frame[keys + ["reward"]].isna().any().any():
        raise ValueError("Empty selection or missing evaluation values")
    if frame.duplicated(keys).any():
        raise ValueError("Duplicate evaluation keys; select one evaluation per condition")
    frame["reward"] = pd.to_numeric(frame.reward, errors="raise")
    if not np.isfinite(frame.reward).all():
        raise ValueError("Rewards must be finite")
    pairs = frame.groupby(keys[:-1]).reward.agg(["mean", "count"]).reset_index()
    cells = pairs.groupby(keys[:3])["mean"].mean()
    directional, symmetric = {}, {}
    for layout, values in cells.groupby(level="layout"):
        matrix = values.droplevel("layout").unstack("algo_2").reindex(
            index=models, columns=models
        ).to_numpy(dtype=float)
        if not np.isfinite(matrix).all():
            absent = [(models[i], models[j]) for i, j in np.argwhere(~np.isfinite(matrix))]
            raise ValueError(f"{layout}: missing model pairs (including diagonal): {absent}")
        directional[layout] = matrix
        symmetric[layout] = (matrix + matrix.T) / 2
    return directional, symmetric, pairs


def replicator(x, payoff):
    fitness = payoff @ x
    return x * (fitness - x @ fitness)


def nash_gap(x, payoff):
    """Best unilateral gain in the symmetric meta-game, in raw return units."""
    fitness = payoff @ x
    return max(0.0, float(fitness.max() - x @ fitness))


def integrate(payoff, initial, horizon, samples):
    # Log coordinates preserve positivity and unit mass. A global positive
    # scale changes time only; no row/column-wise normalization is applied.
    scale = max(float(np.ptp(payoff)), 1.0)
    normalized = (payoff - payoff.mean()) / scale
    def rhs(_, logits):
        x = softmax(logits)
        fitness = normalized @ x
        return fitness - x @ fitness
    result = solve_ivp(rhs, (0, horizon), np.log(initial),
                       t_eval=np.linspace(0, horizon, samples),
                       rtol=1e-8, atol=1e-10)
    if not result.success:
        raise RuntimeError(result.message)
    return result.t, softmax(result.y.T, axis=1), scale


def plot_simplex(payoff, models, path, trajectories=(), resolution=8):
    vertices = np.array([[0, 0], [1, 0], [0.5, np.sqrt(3) / 2]])
    points = np.array([[i, j, resolution-i-j]
                       for i in range(resolution + 1)
                       for j in range(resolution + 1-i)]) / resolution
    xy = points @ vertices
    velocity = np.array([replicator(x, payoff) for x in points]) @ vertices
    speed = np.linalg.norm(velocity, axis=1)
    arrows = velocity / max(float(speed.max()), 1e-12) * 0.04
    fig, ax = plt.subplots(figsize=(9, 7))
    border = vertices[[0, 1, 2, 0]]
    ax.plot(*border.T, color="black")
    q = ax.quiver(*xy.T, *arrows.T, speed, angles="xy", scale_units="xy",
                  scale=1, cmap="viridis", width=0.003)
    fig.colorbar(q, ax=ax, label="Projected replicator speed (raw return units)")
    for trajectory in trajectories:
        ax.plot(*(trajectory @ vertices).T, alpha=0.6, linewidth=1)
    for name, position in zip(models, vertices):
        ax.text(position[0], position[1] + 0.045, display_name(name), ha="center", fontsize=9)
    ax.set(xlim=(-0.2, 1.2), ylim=(-0.1, 1.04), aspect="equal")
    ax.set_title("Role-averaged shared-reward meta-game")
    ax.axis("off")
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, nargs="+", required=True,
                        help="CSV files or directories containing *_cross_algo_results.csv")
    parser.add_argument("--models", nargs="+", required=True)
    parser.add_argument("--layouts", nargs="+", help="Subset of layout names")
    parser.add_argument("--xp-only", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/egta"))
    parser.add_argument("--starts", type=int, default=16)
    parser.add_argument("--horizon", type=float, default=100)
    parser.add_argument("--samples", type=int, default=501)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--resolution", type=int, default=8,
                        help="Simplex grid subdivisions; unrelated to evaluation episodes")
    parser.add_argument("--show-trajectories", action="store_true")
    args = parser.parse_args()
    if len(args.models) < 2 or len(set(args.models)) != len(args.models):
        parser.error("Provide at least two distinct models")
    if args.starts < 1 or args.samples < 2 or not np.isfinite(args.horizon) or args.horizon <= 0:
        parser.error("starts >= 1, samples >= 2 and finite horizon > 0 required")
    if args.resolution < 2:
        parser.error("resolution must be >= 2")
    paths = []
    for source in args.input:
        found = sorted(source.glob("*_cross_algo_results.csv")) if source.is_dir() else [source]
        if not found or any(not p.is_file() for p in found):
            parser.error(f"No input CSVs at {source}")
        paths.extend(p.resolve() for p in found)
    if len(set(paths)) != len(paths):
        parser.error("An input file was selected more than once")
    frame = pd.concat([pd.read_csv(p) for p in paths], ignore_index=True)
    if args.layouts:
        if "layout" not in frame or set(args.layouts) - set(frame.layout):
            parser.error("Requested layouts are absent from inputs")
        frame = frame.loc[frame.layout.isin(args.layouts)]
    directional, symmetric, pairs = build_payoffs(frame, args.models, args.xp_only)
    payoff = np.mean(list(symmetric.values()), axis=0)
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    pairs.to_csv(out / "seed_pair_summary.csv", index=False)
    records = []
    for layout in directional:
        for i, row in enumerate(args.models):
            for j, column in enumerate(args.models):
                records.append(dict(layout=layout, algo_1=row, algo_2=column,
                                    directional=directional[layout][i, j],
                                    role_averaged=symmetric[layout][i, j]))
    pd.DataFrame(records).to_csv(out / "layout_payoffs.csv", index=False)
    pd.DataFrame(payoff, index=args.models, columns=args.models).to_csv(out / "payoff.csv")
    rng = np.random.default_rng(args.seed)
    initial = np.vstack([np.ones(len(args.models)) / len(args.models),
                         rng.dirichlet(np.ones(len(args.models)), args.starts - 1)])
    trajectories, endpoints, histories = [], [], []
    for run, x0 in enumerate(initial):
        time, xs, scale = integrate(payoff, x0, args.horizon, args.samples)
        trajectories.append(xs)
        history = pd.DataFrame(xs, columns=args.models)
        history.insert(0, "time", time)
        history.insert(0, "start", run)
        histories.append(history)
        endpoints.append(dict(start=run, nash_gap=nash_gap(xs[-1], payoff),
                              derivative_norm=float(np.linalg.norm(replicator(xs[-1], payoff))),
                              **dict(zip(args.models, xs[-1]))))
    pd.concat(histories).to_csv(out / "trajectories.csv", index=False)
    pd.DataFrame(endpoints).to_csv(out / "endpoints.csv", index=False)
    pd.DataFrame([dict(model=name, nash_gap=nash_gap(x, payoff))
                  for name, x in zip(args.models, np.eye(len(args.models)))
                  ]).to_csv(out / "pure_strategy_gaps.csv", index=False)
    fig, ax = plt.subplots(figsize=(9, 5))
    for i, model in enumerate(args.models):
        line, = ax.plot(time, trajectories[0][:, i], label=display_name(model))
        for xs in trajectories[1:]:
            ax.plot(time, xs[:, i], color=line.get_color(), alpha=0.15)
    ax.set(xlabel="Time (globally scaled payoff)", ylabel="Population fraction", ylim=(0, 1))
    ax.legend()
    fig.savefig(out / "population.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    if len(args.models) == 3:
        plot_simplex(payoff, args.models, out / "simplex.png",
                     trajectories if args.show_trajectories else (), args.resolution)
    metadata = dict(inputs=[str(p) for p in paths], models=args.models,
                    layouts=list(symmetric), xp_only=args.xp_only, seed=args.seed,
                    starts=args.starts, horizon=args.horizon, samples=args.samples,
                    resolution=args.resolution, show_trajectories=args.show_trajectories,
                    payoff_time_scale=scale,
                    assumption="Shared reward; uniformly random seats; equal layout and seed-pair weights",
                    interpretation="Finite-time endpoints, not certified equilibria or basin probabilities; no bootstrap uncertainty estimated")
    (out / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(f"Saved EGTA for {len(args.models)} models, {len(symmetric)} layouts to {out}")


if __name__ == "__main__":
    main()
