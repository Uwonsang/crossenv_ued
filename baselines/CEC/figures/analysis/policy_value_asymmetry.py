"""Policy/value environment-fitting diagnostic on five PCG layout families.

Models receive their original full 9x9 observations. Pairs share controlled
agent conditions and a unique shortest-delivery first action, but use different
layout variants and delivery delays.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
# Prefer this checkout over an unrelated editable installation of jaxmarl.
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "baselines/CEC"))


CLI_MODEL_NAMES = {"cec": "CEC", "dcec": "CEC_IDAAC"}

try:  # Support both direct execution and package-style imports in tests.
    from .policy_value_asymmetry_common import initial_state
except ImportError:
    from policy_value_asymmetry_common import initial_state  # noqa: E402


def parse_model_spec(value):
    """Parse cec_64 or dcec_256 into an internal model name and size."""
    try:
        name, num_envs_text = value.lower().rsplit("_", 1)
        model = CLI_MODEL_NAMES[name]
        num_envs = int(num_envs_text)
    except (KeyError, ValueError) as exc:
        raise argparse.ArgumentTypeError(
            f"Invalid model '{value}'; use cec_<num_envs> or dcec_<num_envs>"
        ) from exc
    if num_envs <= 0:
        raise argparse.ArgumentTypeError("The number of environments must be positive")
    return model, num_envs


def write_rows(path, rows):
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def evaluate(args):
    import jax
    import jax.numpy as jnp
    import numpy as np
    from jaxmarl.environments.overcooked.overcooked import DELIVERY_REWARD
    from environment_representation_probe import (
        MODEL_SPECS, checkpoint_path, load_config, load_params,
    )
    from actor_networks import ScannedRNN

    cfg = load_config(args.config)
    gamma = float(cfg["GAMMA"] if args.gamma is None else args.gamma)
    if not 0 < gamma < 1:
        raise ValueError("This delay-based diagnostic requires 0 < gamma < 1")
    if args.checkpoints is not None:
        manifest = json.loads(args.checkpoints.read_text())
    else:
        manifest = []
        for model, num_envs in args.models:
            for seed in args.seeds:
                checkpoint = checkpoint_path(args.model_root, model, num_envs, seed)
                if checkpoint is None:
                    print(
                        f"Missing checkpoint: model={model}, envs={num_envs}, "
                        f"seed={seed}", flush=True,
                    )
                    continue
                manifest.append(dict(
                    model=model, num_envs=num_envs, seed=seed,
                    checkpoint=str(checkpoint),
                ))
        if not manifest:
            raise RuntimeError("No usable checkpoints were found under --model-root")
    payload = json.loads(args.pairs.read_text())
    pairs = payload["pairs"]
    if not pairs:
        raise ValueError("No matched pairs. Inspect state_pairs.json diagnostics first.")
    rows, raw = [], []
    for entry in manifest:
        model = entry["model"]
        spec = MODEL_SPECS[model]
        params = load_params(Path(entry["checkpoint"]).expanduser())
        net = spec["value_network"](6, config=cfg)
        for pair in pairs:
            pair_results = {}
            for variant in ("short", "long"):
                env, initial = initial_state(cfg, pair[variant], args.horizon)
                plan = pair[variant]["plan"]
                if len(plan) > args.horizon:
                    raise ValueError("Horizon shorter than oracle delivery path")
                state, oracle_return = initial, 0.0
                for t, action in enumerate(plan):
                    _, state, rewards, _, _ = env.step_env(jax.random.PRNGKey(t), state,
                        {"agent_0": jnp.array(action), "agent_1": jnp.array(4)})
                    oracle_return += gamma**t * float(rewards["agent_0"])
                expected = float(DELIVERY_REWARD) * gamma**(len(plan)-1)
                if not np.isclose(oracle_return, expected):
                    raise ValueError("Geometry oracle disagrees with actual environment transitions")

                def carry():
                    h = ScannedRNN.initialize_carry(2, int(cfg["GRU_HIDDEN_DIM"]))
                    return (h, h) if spec["value_separate_hidden"] else h

                def inputs(state, done):
                    obs = env.get_obs(state)
                    return (jnp.stack([obs[a].reshape(-1) for a in env.agents])[None],
                            done[None], state.agent_pos[None])

                (_, distribution, value), captured = net.apply(params, carry(),
                    inputs(initial, jnp.zeros(2, dtype=bool)), mutable=["intermediates"])
                probs = np.asarray(distribution.probs[0, 0])

                def one(key):
                    def step(c, t):
                        state, h, key, finished = c
                        key, ak, ek = jax.random.split(key, 3)
                        h, pi, _ = net.apply(params, h, inputs(state, jnp.zeros(2, dtype=bool)))
                        action = pi.sample(seed=ak)[0, 0]
                        _, state, reward, done, _ = env.step_env(ek, state,
                            {"agent_0": action, "agent_1": jnp.array(4)})
                        r = jnp.where(finished, 0., reward["agent_0"])
                        return (state, h, key, finished | done["__all__"]), gamma**t*r
                    _, r = jax.lax.scan(step, (initial, carry(), key, jnp.array(False)),
                                        jnp.arange(args.horizon))
                    return r.sum()
                returns = np.asarray(jax.jit(jax.vmap(one))(
                    jax.random.split(jax.random.PRNGKey(args.rollout_seed), args.rollouts)))
                item = dict(predicted_value=float(value[0, 0]), mc_return=float(returns.mean()),
                            first_delivery_oracle_return=oracle_return, oracle_steps=len(plan),
                            oracle_first_action=plan[0], oracle_action_probability=float(probs[plan[0]]))
                features = {k: np.asarray(captured["intermediates"][k][0][0, 0])
                            for k in ("actor_penultimate", "critic_penultimate")}
                pair_results[variant] = (item, probs, features, returns)
                for rep, g in enumerate(returns):
                    raw.append(dict(model=model, seed=entry["seed"], pair=pair["pair"],
                                    variant=variant, rollout=rep, discounted_return=float(g)))
            a, pa, fa, ra = pair_results["short"]
            b, pb, fb, rb = pair_results["long"]
            mid = (pa+pb)/2
            def kl(p):
                mask = p > 0
                return float(np.sum(p[mask]*np.log(p[mask]/mid[mask])))
            delta = ra-rb
            record = dict(model=model, num_envs=entry.get("num_envs", ""),
                seed=entry["seed"], checkpoint=entry["checkpoint"],
                pair=pair["pair"], policy_js_nats=(kl(pa)+kl(pb))/2,
                policy_argmax_same=bool(pa.argmax()==pb.argmax()),
                oracle_first_action_same=a["oracle_first_action"]==b["oracle_first_action"],
                short_policy_argmax=int(pa.argmax()),
                long_policy_argmax=int(pb.argmax()),
                short_policy_oracle_correct=bool(pa.argmax()==a["oracle_first_action"]),
                long_policy_oracle_correct=bool(pb.argmax()==b["oracle_first_action"]),
                both_policy_oracle_correct=bool(
                    pa.argmax()==a["oracle_first_action"]
                    and pb.argmax()==b["oracle_first_action"]
                ),
                value_delta=a["predicted_value"]-b["predicted_value"],
                mc_delta=float(delta.mean()), mc_delta_sem=float(delta.std(ddof=1)/np.sqrt(len(delta))),
                first_delivery_oracle_delta=a["first_delivery_oracle_return"]-b["first_delivery_oracle_return"])
            for variant, item in (("short", a), ("long", b)):
                record.update({f"{variant}_{k}": v for k, v in item.items()})
            # Descriptive distances only: do not infer a common actor/critic scale.
            for feature in fa:
                norm = np.linalg.norm(fa[feature])*np.linalg.norm(fb[feature])
                record[f"{feature}_cosine_distance"] = (
                    float(1-np.dot(fa[feature], fb[feature])/norm) if norm > 0 else float("nan"))
            rows.append(record)
            write_rows(args.output_dir / "paired_metrics.csv", rows)
            write_rows(args.output_dir / "rollout_returns.csv", raw)
            print(f"Saved {model} seed {entry['seed']} {pair['pair']}", flush=True)
    (args.output_dir / "metadata.json").write_text(json.dumps(dict(
        gamma=gamma, horizon=args.horizon, rollouts=args.rollouts, rollout_seed=args.rollout_seed,
        checkpoints=manifest, pair_source=str(args.pairs), pair_selection=payload, partner="stationary agent at variant reset position", recurrent_history="zero at initial state",
        caveat="Existing-family PCG variants with injected soup and zero recurrent history. "
               "Models receive different full 9x9 layout observations; matching the shortest "
               "first-delivery action does not imply full-policy equivalence. "
               "First-delivery oracle is not full-episode policy value. Critic training partner/reward may differ. "
               "Do not treat pairs or rollouts as independent training seeds."
    ), indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--pairs", type=Path,
        default=ROOT / "artifacts/policy_value_asymmetry/state_pairs.json",
        help="Prepared state_pairs.json (default: local artifacts directory)",
    )
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--checkpoints", type=Path,
                        help="Optional JSON list: model, seed, checkpoint")
    source.add_argument("--model-root", type=Path,
                        help="Root containing MODEL/NUM_ENVS/seedN checkpoints")
    parser.add_argument(
        "--models", nargs="+", type=parse_model_spec,
        default=[("CEC", 64), ("CEC_IDAAC", 64)],
        metavar="MODEL_SIZE",
        help="cec_<num_envs> or dcec_<num_envs> (default: cec_64 dcec_64)",
    )
    parser.add_argument("--seeds", type=int, nargs="+", default=list(range(6)))
    parser.add_argument("--config", type=Path, default=ROOT / "baselines/CEC_UED/config/ippo_overcooked_CEC_gradient.yaml")
    parser.add_argument(
        "--output-dir", type=Path,
        help=("Override output directory. Evaluation with --model-root defaults to "
              "<model-root>/analysis/policy_value_asymmetry."),
    )
    parser.add_argument("--horizon", type=int, default=400)
    parser.add_argument("--rollouts", type=int, default=100)
    parser.add_argument("--rollout-seed", type=int, default=1701)
    parser.add_argument("--gamma", type=float)
    args = parser.parse_args()
    if args.checkpoints is None and args.model_root is None:
        parser.error("Supply --model-root or --checkpoints")
    if args.rollouts < 2 or args.horizon < 1:
        parser.error("rollouts >= 2 and horizon >= 1 required")
    if args.output_dir is None:
        if args.model_root is not None:
            args.output_dir = (
                args.model_root.expanduser() / "analysis" / "policy_value_asymmetry"
            )
        else:
            args.output_dir = ROOT / "artifacts/policy_value_asymmetry"
    else:
        args.output_dir = args.output_dir.expanduser()
    if args.model_root is not None:
        args.model_root = args.model_root.expanduser()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if not args.pairs.is_file():
        parser.error(f"Prepared pairs file does not exist: {args.pairs}")
    evaluate(args)


if __name__ == "__main__":
    main()
