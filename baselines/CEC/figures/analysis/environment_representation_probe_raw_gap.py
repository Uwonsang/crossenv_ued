"""Layout probe on frozen representations vs. raw observations (probe gap).

For every rollout, two linear probes are trained on the same timesteps and the
same episode-level train/test split:
  * representation probe - penultimate actor/value features of the model
  * raw probe            - the flattened raw observation the model received
gap = raw_accuracy - accuracy measures how much layout information that is
linearly available in the input the representation discards.

``--behaviors`` chooses the policy that generates the observation sequences:
  own    - each model's own rollouts (as in environment_representation_probe.py)
  random - uniform-random actions; every model sees identical inputs
  cross  - the other model's checkpoint (CEC <-> DCEC, same num_envs and seed)
"""

from __future__ import annotations

import argparse
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from environment_representation_probe_review import (
    CROSS_MODELS,
    LAYOUT_FAMILIES,
    MODEL_FILE_LABELS,
    MODEL_SPECS,
    ScannedRNN,
    checkpoint_path,
    load_config,
    load_params,
    make_family_env,
    parse_model_spec,
    stratified_episode_split,
    write_csv,
    REPO_ROOT,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Probe layout information in features vs. raw observations."
    )
    parser.add_argument(
        "--model-root", type=Path,
        default=Path("/mnt/nas/wonsang/crossenv_ued/models/ICRL"),
    )
    parser.add_argument(
        "--models", nargs="+", type=parse_model_spec,
        default=[("CEC", 64), ("CEC_IDAAC", 64)], metavar="MODEL_SIZE",
        help="cec_<num_envs> / dcec_<num_envs> (default: cec_64 dcec_64).",
    )
    parser.add_argument(
        "--representation", choices=("actor", "value"), default="actor",
    )
    parser.add_argument(
        "--behaviors", nargs="+", choices=("own", "random", "cross"),
        default=["own", "random", "cross"],
    )
    parser.add_argument("--seeds", type=int, nargs="+", default=list(range(6)))
    parser.add_argument("--episodes-per-layout", type=int, default=20)
    parser.add_argument("--steps", type=int, default=128)
    parser.add_argument("--samples-per-episode", type=int, default=32)
    parser.add_argument("--probe-splits", type=int, default=10)
    parser.add_argument("--train-fraction", type=float, default=0.7)
    parser.add_argument("--rollout-seed", type=int, default=1701)
    parser.add_argument("--argmax", action="store_true")
    parser.add_argument(
        "--config", type=Path,
        default=REPO_ROOT / "baselines" / "CEC_UED" / "config"
        / "ippo_overcooked_CEC_gradient.yaml",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=None,
        help="Defaults to <model-root>/analysis/representation_probe/raw_gap.",
    )
    return parser.parse_args()


def rollout_family(
    network, params, env, reset_name: str, config: dict, episode_keys,
    steps: int, argmax: bool, representation: str, separate_hidden: bool,
    behavior: str, behavior_network, behavior_params,
):
    """Return (features, raw_obs), each shaped [episode, step, agent, dim]."""
    num_agents = env.num_agents
    hidden_dim = int(config["GRU_HIDDEN_DIM"])
    feature_name = (
        "actor_penultimate" if representation == "actor" else "critic_penultimate"
    )

    def one_episode(key):
        key, reset_key = jax.random.split(key)
        obs, env_state = env.reset(
            reset_key, params={"random_reset_fn": reset_name}
        )
        hidden = ScannedRNN.initialize_carry(num_agents, hidden_dim)
        if separate_hidden:
            hidden = (hidden, hidden)
        behavior_hidden = ScannedRNN.initialize_carry(num_agents, hidden_dim)
        done = jnp.zeros((num_agents,), dtype=bool)

        def step(carry, _):
            env_state, obs, done, hidden, behavior_hidden, key = carry
            obs_batch = jnp.stack(
                [obs[agent].reshape(-1) for agent in env.agents]
            )
            positions = jnp.stack([env_state.agent_pos for _ in env.agents])
            network_input = (
                obs_batch[jnp.newaxis, :],
                done[jnp.newaxis, :],
                positions[jnp.newaxis, :],
            )
            outputs, captured = network.apply(
                params, hidden, network_input, mutable=["intermediates"],
            )
            hidden, policy, _ = outputs
            features = captured["intermediates"][feature_name][0][0]
            key, action_key, step_key = jax.random.split(key, 3)
            if behavior == "random":
                actions = jax.random.randint(
                    action_key, (num_agents,), 0, env.action_space("agent_0").n
                )
            else:
                if behavior == "cross":
                    behavior_hidden, policy, _ = behavior_network.apply(
                        behavior_params, behavior_hidden, network_input
                    )
                actions = (
                    jnp.argmax(policy.logits[0], axis=-1)
                    if argmax else policy.sample(seed=action_key)[0]
                )
            env_actions = {
                agent: actions[index] for index, agent in enumerate(env.agents)
            }
            obs, env_state, _, dones, _ = env.step(step_key, env_state, env_actions)
            done = jnp.asarray([dones[agent] for agent in env.agents])
            carry = (env_state, obs, done, hidden, behavior_hidden, key)
            return carry, (features, obs_batch.astype(jnp.float32))

        carry = (env_state, obs, done, hidden, behavior_hidden, key)
        _, outputs = jax.lax.scan(step, carry, None, length=steps)
        return outputs

    features, raw_obs = jax.jit(jax.vmap(one_episode))(episode_keys)
    return np.asarray(features), np.asarray(raw_obs)


def collect_dataset(
    network, params, env, config: dict, args: argparse.Namespace, model_seed: int,
    separate_hidden: bool, behavior: str, behavior_network, behavior_params,
):
    all_features, all_raw, all_labels, all_episodes = [], [], [], []
    sample_steps = np.linspace(
        0, args.steps - 1, min(args.samples_per_episode, args.steps), dtype=int,
    )
    base_key = jax.random.PRNGKey(args.rollout_seed + 1009 * model_seed)
    family_keys = jax.random.split(base_key, len(LAYOUT_FAMILIES))

    for label, ((layout_name, reset_name), family_key) in enumerate(
        zip(LAYOUT_FAMILIES, family_keys)
    ):
        episode_keys = jax.random.split(family_key, args.episodes_per_layout)
        features, raw_obs = rollout_family(
            network, params, env, reset_name, config, episode_keys,
            args.steps, args.argmax, args.representation, separate_hidden,
            behavior, behavior_network, behavior_params,
        )
        features = features[:, sample_steps]
        raw_obs = raw_obs[:, sample_steps]
        per_episode = features.shape[1] * features.shape[2]
        all_features.append(features.reshape(-1, features.shape[-1]))
        all_raw.append(raw_obs.reshape(-1, raw_obs.shape[-1]))
        all_labels.append(np.full(features.shape[0] * per_episode, label))
        all_episodes.append(np.repeat(
            label * args.episodes_per_layout + np.arange(features.shape[0]),
            per_episode,
        ))
        print(f"  collected {layout_name}: {features.shape[0] * per_episode} vectors")

    return (
        np.concatenate(all_features),
        np.concatenate(all_raw),
        np.concatenate(all_labels),
        np.concatenate(all_episodes),
    )


def run_probes(
    features, raw_obs, labels, episode_ids, args, model: str, num_envs: int,
    seed: int, ckpt: Path, behavior: str,
):
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    def probe_accuracy(x, train_mask, test_mask):
        probe = make_pipeline(
            StandardScaler(), LogisticRegression(max_iter=3000, solver="lbfgs"),
        )
        probe.fit(x[train_mask], labels[train_mask])
        return float(probe.score(x[test_mask], labels[test_mask]))

    rows = []
    for split in range(args.probe_splits):
        rng = np.random.default_rng(args.rollout_seed + 7919 * seed + split)
        train_mask, test_mask = stratified_episode_split(
            labels, episode_ids, args.train_fraction, rng
        )
        accuracy = probe_accuracy(features, train_mask, test_mask)
        raw_accuracy = probe_accuracy(raw_obs, train_mask, test_mask)
        rows.append({
            "model": model,
            "model_label": MODEL_SPECS[model]["display"],
            "model_num_envs": num_envs,
            "representation": args.representation,
            "behavior": behavior,
            "checkpoint_seed": seed,
            "probe_split": split,
            "accuracy": accuracy,
            "raw_accuracy": raw_accuracy,
            "gap": raw_accuracy - accuracy,
            "chance_accuracy": 1.0 / len(LAYOUT_FAMILIES),
            "num_train_samples": int(train_mask.sum()),
            "num_test_samples": int(test_mask.sum()),
            "checkpoint": str(ckpt),
        })
    return rows


def main():
    args = parse_args()
    args.model_root = args.model_root.expanduser()
    if args.output_dir is None:
        args.output_dir = (
            args.model_root / "analysis" / "representation_probe" / "raw_gap"
        )
    config = load_config(args.config)
    env = make_family_env(config)
    action_dim = env.action_space("agent_0").n
    saved_csv_paths = []

    for model, num_envs in args.models:
        spec = MODEL_SPECS[model]
        model_rows = {behavior: [] for behavior in args.behaviors}
        network = spec[f"{args.representation}_network"](action_dim, config=config)
        separate_hidden = (
            args.representation == "value" and bool(spec["value_separate_hidden"])
        )
        cross_model = CROSS_MODELS[model]
        cross_network = MODEL_SPECS[cross_model]["actor_network"](
            action_dim, config=config,
        )
        for seed in args.seeds:
            ckpt = checkpoint_path(args.model_root, model, num_envs, seed)
            if ckpt is None:
                print(f"Missing checkpoint: model={model}, envs={num_envs}, seed={seed}")
                continue
            print(f"Loading {spec['display']} seed {seed}: {ckpt}")
            params = load_params(ckpt)
            for behavior in args.behaviors:
                behavior_params = None
                if behavior == "cross":
                    cross_ckpt = checkpoint_path(
                        args.model_root, cross_model, num_envs, seed
                    )
                    if cross_ckpt is None:
                        print(f"Missing cross checkpoint: {cross_model} seed {seed}")
                        continue
                    behavior_params = load_params(cross_ckpt)
                print(f"  behavior={behavior}")
                features, raw_obs, labels, episode_ids = collect_dataset(
                    network, params, env, config, args, seed, separate_hidden,
                    behavior, cross_network, behavior_params,
                )
                model_rows[behavior].extend(run_probes(
                    features, raw_obs, labels, episode_ids, args, model,
                    num_envs, seed, ckpt, behavior,
                ))

        representation_prefix = (
            "" if args.representation == "actor" else f"{args.representation}_"
        )
        for behavior, rows in model_rows.items():
            if not rows:
                continue
            csv_path = args.output_dir / f"behavior_{behavior}" / (
                "environment_probe_"
                f"{representation_prefix}{MODEL_FILE_LABELS[model]}_{num_envs}.csv"
            )
            csv_path.parent.mkdir(parents=True, exist_ok=True)
            write_csv(rows, csv_path)
            saved_csv_paths.append(csv_path)
            print(f"Saved: {csv_path}")

    if not saved_csv_paths:
        raise RuntimeError("No usable checkpoints were found")


if __name__ == "__main__":
    main()
