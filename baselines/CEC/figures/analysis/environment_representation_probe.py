"""Linear probe for layout information in frozen policy representations.

The probe independently evaluates CEC (shared policy/value trunk) and DCEC
(policy/value-decoupled trunks).  Data are generated from the five procedural
Overcooked layout families used by ``reset_all``.  Entire episodes are assigned
to either the probe-training or probe-test split, so adjacent states from one
trajectory cannot leak across the split.

This is a representation *information* diagnostic, not by itself proof of
overfitting: layout geometry is useful for acting and is visible in the raw
observation.  A stronger overfitting claim requires combining probe accuracy
with held-out return/generalization-gap results.
"""

from __future__ import annotations

import argparse
import csv
import glob
import pickle
import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import yaml


REPO_ROOT = Path(__file__).resolve().parents[4]
CEC_DIR = REPO_ROOT / "baselines" / "CEC"
if str(CEC_DIR) not in sys.path:
    sys.path.insert(0, str(CEC_DIR))

import jaxmarl  # noqa: E402
from jaxmarl.environments.overcooked import overcooked_layouts  # noqa: E402

from actor_networks import (  # noqa: E402
    ActorCriticRNN,
    IDAACActorRNN,
    ScannedRNN,
)


LAYOUT_FAMILIES = (
    ("Asymmetric Advantages", "reset_asymm_advantages"),
    ("Coordination Ring", "reset_coord_ring"),
    ("Counter Circuit", "reset_counter_circuit"),
    ("Forced Coordination", "reset_forced_coord"),
    ("Cramped Room", "reset_cramped_room"),
)
MODEL_SPECS = {
    "CEC": {"display": "CEC", "network": ActorCriticRNN},
    "CEC_IDAAC": {
        "display": "DCEC",
        "network": IDAACActorRNN,
    },
}
MODEL_FILE_LABELS = {"CEC": "cec", "CEC_IDAAC": "dcec"}
CLI_MODEL_NAMES = {"cec": "CEC", "dcec": "CEC_IDAAC"}


def parse_model_spec(value: str) -> tuple[str, int]:
    """Parse e.g. ``cec_64`` or ``dcec_256`` into checkpoint coordinates."""
    try:
        model_name, num_envs_text = value.lower().rsplit("_", 1)
        model = CLI_MODEL_NAMES[model_name]
        num_envs = int(num_envs_text)
    except (KeyError, ValueError) as exc:
        raise argparse.ArgumentTypeError(
            f"Invalid model '{value}'; use cec_<num_envs> or dcec_<num_envs>"
        ) from exc
    if num_envs <= 0:
        raise argparse.ArgumentTypeError("The number of environments must be positive")
    return model, num_envs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Probe layout-family information in policy features."
    )
    parser.add_argument(
        "--model-root", type=Path,
        default=Path("/mnt/nas/wonsang/crossenv_ued/models/ICRL"),
    )
    parser.add_argument(
        "--models",
        nargs="+",
        type=parse_model_spec,
        default=[("CEC", 64), ("CEC_IDAAC", 64)],
        metavar="MODEL_SIZE",
        help=(
            "Models to probe, formatted as cec_<num_envs> or "
            "dcec_<num_envs> (default: cec_64 dcec_64)."
        ),
    )
    parser.add_argument("--seeds", type=int, nargs="+", default=list(range(6)))
    parser.add_argument("--episodes-per-layout", type=int, default=20)
    parser.add_argument("--steps", type=int, default=128)
    parser.add_argument(
        "--samples-per-episode", type=int, default=32,
        help="Evenly subsampled timesteps; both agents are retained.",
    )
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
        "--output-dir", type=Path,
        default=None,
        help=(
            "Output directory. Defaults to "
            "<model-root>/analysis/representation_probe."
        ),
    )
    return parser.parse_args()


def load_config(path: Path) -> dict:
    with path.open(encoding="utf-8") as file:
        config = yaml.safe_load(file)
    config["ENV_NAME"] = "overcooked"
    config["CONV_NET"] = True
    config["LSTM"] = True
    return config


def checkpoint_path(root: Path, model: str, num_envs: int, seed: int) -> Path | None:
    pattern = root / model / str(num_envs) / f"seed{seed}" / f"seed{seed}_ckpt*.pkl"
    matches = sorted(glob.glob(str(pattern)))
    return Path(matches[-1]) if matches else None


def load_params(path: Path):
    with path.open("rb") as file:
        checkpoint = pickle.load(file)
    if "params" not in checkpoint:
        raise KeyError(f"Checkpoint has no 'params': {path}")
    return checkpoint["params"]


def make_family_env(config: dict):
    kwargs = dict(config["ENV_KWARGS"])
    kwargs.update({
        "layout": overcooked_layouts["cramped_room_9"],
        "random_reset": True,
        "check_held_out": False,
        "shuffle_inv_and_pot": False,
    })
    return jaxmarl.make("overcooked", **kwargs)


def rollout_family(
    network, params, env, reset_name: str, config: dict, episode_keys,
    steps: int, argmax: bool,
) -> np.ndarray:
    """Return features with shape [episode, step, agent, feature]."""
    num_agents = env.num_agents
    hidden_dim = int(config["GRU_HIDDEN_DIM"])

    def one_episode(key):
        key, reset_key = jax.random.split(key)
        obs, env_state = env.reset(
            reset_key, params={"random_reset_fn": reset_name}
        )
        hidden = ScannedRNN.initialize_carry(num_agents, hidden_dim)
        done = jnp.zeros((num_agents,), dtype=bool)

        def step(carry, _):
            env_state, obs, done, hidden, key = carry
            obs_batch = jnp.stack(
                [obs[agent].reshape(-1) for agent in env.agents]
            )
            positions = jnp.stack(
                [env_state.agent_pos for _ in env.agents]
            )
            network_input = (
                obs_batch[jnp.newaxis, :],
                done[jnp.newaxis, :],
                positions[jnp.newaxis, :],
            )
            outputs, captured = network.apply(
                params, hidden, network_input,
                mutable=["intermediates"],
            )
            hidden, policy, _ = outputs
            features = captured["intermediates"]["actor_penultimate"][0][0]
            key, action_key, step_key = jax.random.split(key, 3)
            actions = (
                jnp.argmax(policy.logits[0], axis=-1)
                if argmax else policy.sample(seed=action_key)[0]
            )
            env_actions = {
                agent: actions[index]
                for index, agent in enumerate(env.agents)
            }
            obs, env_state, _, dones, _ = env.step(
                step_key, env_state, env_actions
            )
            done = jnp.asarray([dones[agent] for agent in env.agents])
            return (env_state, obs, done, hidden, key), features

        carry = (env_state, obs, done, hidden, key)
        _, features = jax.lax.scan(step, carry, None, length=steps)
        return features

    collect = jax.jit(jax.vmap(one_episode))
    return np.asarray(collect(episode_keys))


def collect_dataset(
    network, params, env, config: dict, args: argparse.Namespace, model_seed: int,
):
    all_features = []
    all_labels = []
    all_episodes = []
    sample_steps = np.linspace(
        0, args.steps - 1,
        min(args.samples_per_episode, args.steps),
        dtype=int,
    )
    base_key = jax.random.PRNGKey(args.rollout_seed + 1009 * model_seed)
    family_keys = jax.random.split(base_key, len(LAYOUT_FAMILIES))

    for label, ((layout_name, reset_name), family_key) in enumerate(
        zip(LAYOUT_FAMILIES, family_keys)
    ):
        episode_keys = jax.random.split(family_key, args.episodes_per_layout)
        features = rollout_family(
            network, params, env, reset_name, config, episode_keys,
            args.steps, args.argmax,
        )
        features = features[:, sample_steps, :, :]
        per_episode = features.shape[1] * features.shape[2]
        all_features.append(features.reshape(-1, features.shape[-1]))
        all_labels.append(np.full(features.shape[0] * per_episode, label))
        episode_ids = np.repeat(
            label * args.episodes_per_layout + np.arange(features.shape[0]),
            per_episode,
        )
        all_episodes.append(episode_ids)
        print(
            f"  collected {layout_name}: {features.shape[0]} episodes, "
            f"{features.shape[0] * per_episode} feature vectors"
        )

    return (
        np.concatenate(all_features),
        np.concatenate(all_labels),
        np.concatenate(all_episodes),
    )


def stratified_episode_split(labels, episode_ids, train_fraction, rng):
    train_episodes = []
    test_episodes = []
    for label in np.unique(labels):
        candidates = np.unique(episode_ids[labels == label])
        candidates = rng.permutation(candidates)
        count = int(round(len(candidates) * train_fraction))
        count = min(max(count, 1), len(candidates) - 1)
        train_episodes.extend(candidates[:count])
        test_episodes.extend(candidates[count:])
    return (
        np.isin(episode_ids, train_episodes),
        np.isin(episode_ids, test_episodes),
    )


def run_probes(
    features, labels, episode_ids, args, model: str, num_envs: int,
    seed: int, ckpt: Path,
):
    try:
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler
    except ImportError as exc:
        raise RuntimeError(
            "scikit-learn is required for the linear representation probe"
        ) from exc

    rows = []
    for split in range(args.probe_splits):
        rng = np.random.default_rng(args.rollout_seed + 7919 * seed + split)
        train_mask, test_mask = stratified_episode_split(
            labels, episode_ids, args.train_fraction, rng
        )
        probe = make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=3000, solver="lbfgs"),
        )
        probe.fit(features[train_mask], labels[train_mask])
        accuracy = float(probe.score(features[test_mask], labels[test_mask]))
        rows.append({
            "model": model,
            "model_label": MODEL_SPECS[model]["display"],
            "model_num_envs": num_envs,
            "checkpoint_seed": seed,
            "probe_split": split,
            "accuracy": accuracy,
            "chance_accuracy": 1.0 / len(LAYOUT_FAMILIES),
            "num_train_samples": int(train_mask.sum()),
            "num_test_samples": int(test_mask.sum()),
            "checkpoint": str(ckpt),
        })
    return rows


def write_csv(rows, path: Path):
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    args = parse_args()
    if args.episodes_per_layout < 2:
        raise ValueError("--episodes-per-layout must be at least 2")
    if not 0.0 < args.train_fraction < 1.0:
        raise ValueError("--train-fraction must be between 0 and 1")
    args.model_root = args.model_root.expanduser()
    if args.output_dir is None:
        args.output_dir = args.model_root / "analysis" / "representation_probe"
    else:
        args.output_dir = args.output_dir.expanduser()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    config = load_config(args.config)
    env = make_family_env(config)
    saved_csv_paths = []

    for model, num_envs in args.models:
        spec = MODEL_SPECS[model]
        model_rows = []
        network = spec["network"](
            env.action_space("agent_0").n,
            config=config,
        )
        for seed in args.seeds:
            ckpt = checkpoint_path(args.model_root, model, num_envs, seed)
            if ckpt is None:
                print(
                    f"Missing checkpoint: model={model}, envs={num_envs}, "
                    f"seed={seed}"
                )
                continue
            print(f"Loading {spec['display']} seed {seed}: {ckpt}")
            params = load_params(ckpt)
            features, labels, episode_ids = collect_dataset(
                network, params, env, config, args, seed
            )
            model_rows.extend(run_probes(
                features, labels, episode_ids, args, model, num_envs,
                seed, ckpt
            ))

        if model_rows:
            csv_path = args.output_dir / (
                f"environment_probe_{MODEL_FILE_LABELS[model]}_{num_envs}.csv"
            )
            write_csv(model_rows, csv_path)
            saved_csv_paths.append(csv_path)
            print(f"Saved: {csv_path}")

    if not saved_csv_paths:
        raise RuntimeError("No usable checkpoints were found")


if __name__ == "__main__":
    main()
