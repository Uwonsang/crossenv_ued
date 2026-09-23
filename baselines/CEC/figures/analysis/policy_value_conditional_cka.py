"""Compute cross-environment CKA after conditioning on equivalent behavior.

The script reuses saved ``concrete_state_pairs.json`` files, extracts frozen
actor/critic penultimate features, and computes A-to-B linear CKA only over
pairs whose action distributions satisfy the requested behavior condition.
"""
from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import jax.numpy as jnp
import numpy as np

from policy_value_concrete_example import (
    ACTION_NAMES,
    MODEL_SPECS,
    ROOT,
    checkpoint_path,
    instantiate,
    js_divergence,
    linear_cka,
    load_config,
    load_params,
    parse_model_spec,
)
from policy_value_fixed_pairs_metrics import discover_pair_files, load_pairs
from actor_networks import ScannedRNN


MODEL_LABELS = {"CEC": "CEC", "CEC_IDAAC": "DCEC"}
CONDITION_LABELS = {
    "policy_equivalent": "Policy-equivalent",
    "interact_equivalent": "Policy-equivalent Interact",
}
CONDITION_ORDER = tuple(CONDITION_LABELS)


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError(f"Cannot write an empty CSV: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def extract_pair_features(
    config: dict, records: list[dict], model: str, params, horizon: int
) -> dict:
    """Extract agent-0 action probabilities and actor/critic representations."""
    spec = MODEL_SPECS[model]
    network = spec["value_network"](len(ACTION_NAMES), config=config)
    hidden_dim = int(config["GRU_HIDDEN_DIM"])

    def initial_hidden():
        hidden = ScannedRNN.initialize_carry(2, hidden_dim)
        return (hidden, hidden) if spec["value_separate_hidden"] else hidden

    result = {}
    for record in records:
        env, state, _ = instantiate(config, record, horizon)
        observations = env.get_obs(state)
        observation_batch = jnp.stack([
            observations[agent].reshape(-1) for agent in env.agents
        ])
        network_input = (
            observation_batch[jnp.newaxis, :],
            jnp.zeros((1, len(env.agents)), dtype=bool),
            state.agent_pos[jnp.newaxis, :],
        )
        (_, policy, _), captured = network.apply(
            params, initial_hidden(), network_input, mutable=["intermediates"]
        )
        intermediates = captured["intermediates"]
        result[record["variant"]] = {
            "probabilities": np.asarray(policy.probs[0, 0]),
            "policy_rep": np.asarray(
                intermediates["actor_penultimate"][0][0, 0]
            ),
            "value_rep": np.asarray(
                intermediates["critic_penultimate"][0][0, 0]
            ),
        }
    if set(result) != {"A", "B"}:
        raise ValueError(f"Expected A/B variants, found {sorted(result)}")
    return result


def classify_pair(
    evaluated: dict, max_policy_js: float, min_interact_probability: float
) -> dict:
    probabilities_a = evaluated["A"]["probabilities"]
    probabilities_b = evaluated["B"]["probabilities"]
    argmax_a = int(probabilities_a.argmax())
    argmax_b = int(probabilities_b.argmax())
    policy_js = js_divergence(probabilities_a, probabilities_b)
    policy_equivalent = argmax_a == argmax_b and policy_js < max_policy_js
    interact_equivalent = bool(
        policy_equivalent
        and argmax_a == ACTION_NAMES.index("Interact")
        and probabilities_a[argmax_a] >= min_interact_probability
        and probabilities_b[argmax_b] >= min_interact_probability
    )
    return {
        "policy_js_nats": policy_js,
        "argmax_a": ACTION_NAMES[argmax_a],
        "argmax_b": ACTION_NAMES[argmax_b],
        "interact_probability_a": float(
            probabilities_a[ACTION_NAMES.index("Interact")]
        ),
        "interact_probability_b": float(
            probabilities_b[ACTION_NAMES.index("Interact")]
        ),
        "policy_equivalent": bool(policy_equivalent),
        "interact_equivalent": interact_equivalent,
    }


def compute_conditional_cka(
    evaluated_rows: list[dict], conditions: tuple[str, ...], min_pairs: int
) -> list[dict]:
    grouped = defaultdict(list)
    for row in evaluated_rows:
        grouped[(row["layout"], row["model"], row["num_envs"], row["seed"])].append(row)

    summaries = []
    for (layout, model, num_envs, seed), group in grouped.items():
        group = sorted(group, key=lambda row: row["pair_id"])
        for condition in conditions:
            selected = [row for row in group if row[condition]]
            enough = len(selected) >= min_pairs
            policy_cka = value_cka = float("nan")
            if enough:
                policy_a = np.stack([row["_policy_rep_a"] for row in selected])
                policy_b = np.stack([row["_policy_rep_b"] for row in selected])
                value_a = np.stack([row["_value_rep_a"] for row in selected])
                value_b = np.stack([row["_value_rep_b"] for row in selected])
                policy_cka = linear_cka(policy_a, policy_b)
                value_cka = linear_cka(value_a, value_b)
            summaries.append({
                "layout": layout,
                "model": model,
                "model_label": MODEL_LABELS.get(model, model),
                "num_envs": int(num_envs),
                "seed": int(seed),
                "condition": condition,
                "total_pairs": len(group),
                "selected_pairs": len(selected),
                "selected_fraction": len(selected) / len(group),
                "min_pairs_required": min_pairs,
                "has_enough_pairs": enough,
                "policy_a_b_linear_cka": policy_cka,
                "value_a_b_linear_cka": value_cka,
                "policy_minus_value_cka": policy_cka - value_cka,
            })
    return summaries


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-root", type=Path, required=True)
    parser.add_argument("--pairs-root", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument(
        "--models", nargs="+", type=parse_model_spec,
        default=[("CEC", 64), ("CEC_IDAAC", 64)],
    )
    parser.add_argument("--seeds", type=int, nargs="+", default=list(range(6)))
    parser.add_argument(
        "--conditions", nargs="+", choices=CONDITION_ORDER,
        default=list(CONDITION_ORDER),
    )
    parser.add_argument("--max-policy-js", type=float, default=.15)
    parser.add_argument("--min-interact-probability", type=float, default=.7)
    parser.add_argument("--min-pairs", type=int, default=3)
    parser.add_argument("--horizon", type=int, default=200)
    parser.add_argument(
        "--config", type=Path,
        default=ROOT / "baselines/CEC_UED/config/ippo_overcooked_CEC_gradient.yaml",
    )
    args = parser.parse_args()
    if args.min_pairs < 2:
        parser.error("--min-pairs must be at least 2")
    args.model_root = args.model_root.expanduser()
    pairs_root = args.pairs_root or (
        args.model_root / "analysis" / "policy_value_large_scale"
    )
    output_dir = args.output_dir or (
        args.model_root / "analysis" / "policy_value_conditional_cka"
    )
    pair_files = discover_pair_files(pairs_root)
    if not pair_files:
        parser.error(f"No concrete_state_pairs.json found under {pairs_root}")

    config = load_config(args.config)
    filter_rows = []
    for pair_file in pair_files:
        try:
            pairs = load_pairs(pair_file)
        except ValueError as error:
            parser.error(str(error))
        layout = pair_file.parent.name
        for model, num_envs in args.models:
            for seed in args.seeds:
                checkpoint = checkpoint_path(args.model_root, model, num_envs, seed)
                if checkpoint is None:
                    print(f"Missing checkpoint: {model} {num_envs} seed {seed}")
                    continue
                params = load_params(checkpoint)
                for pair_id, states in pairs:
                    evaluated = extract_pair_features(
                        config, states, model, params, args.horizon
                    )
                    classification = classify_pair(
                        evaluated, args.max_policy_js,
                        args.min_interact_probability,
                    )
                    filter_rows.append({
                        "layout": layout,
                        "model": model,
                        "model_label": MODEL_LABELS.get(model, model),
                        "num_envs": int(num_envs),
                        "seed": int(seed),
                        "pair_id": int(pair_id),
                        "checkpoint": str(checkpoint),
                        **classification,
                        "_policy_rep_a": evaluated["A"]["policy_rep"],
                        "_policy_rep_b": evaluated["B"]["policy_rep"],
                        "_value_rep_a": evaluated["A"]["value_rep"],
                        "_value_rep_b": evaluated["B"]["value_rep"],
                    })
                print(
                    f"Evaluated {layout}: {model} {num_envs}, seed {seed}, "
                    f"{len(pairs)} fixed pairs"
                )

    if not filter_rows:
        raise RuntimeError("No usable checkpoints were found")
    conditions = tuple(dict.fromkeys(args.conditions))
    summary_rows = compute_conditional_cka(filter_rows, conditions, args.min_pairs)
    public_filter_rows = [
        {key: value for key, value in row.items() if not key.startswith("_")}
        for row in filter_rows
    ]
    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(output_dir / "conditional_cka_pair_filters.csv", public_filter_rows)
    write_csv(output_dir / "conditional_cka_summary.csv", summary_rows)
    print(f"Saved conditional CKA CSV results to {output_dir}")


if __name__ == "__main__":
    main()
