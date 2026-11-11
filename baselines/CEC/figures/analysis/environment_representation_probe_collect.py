"""Collect probe data once: actor/value features + raw observations.

One rollout per (model, num_envs, seed, behavior) stores everything the probes
need, so probing (environment_representation_probe_fit.py) can be rerun without
rolling out again:
  actor   [N, actor_dim]   penultimate actor features
  value   [N, value_dim]   penultimate value features
  raw     [N, obs_dim]     flattened raw observation (9x9x26)
  featurized [N, 96]       Overcooked-AI style feature vector (overcooked_featurize.py)
  labels  [N]              layout family index
  episode_ids [N]          episode index (for episode-level splits)

Behaviors (policy that generates the observation sequences):
  own    - the model itself acts (as in environment_representation_probe.py)
  random - uniform-random actions; every model sees identical inputs
  cross  - the other model's checkpoint (CEC <-> DCEC, same num_envs and seed)

Saved to <output-dir>/behavior_<b>/<cec|dcec>_<num_envs>_seed<s>.npz.
Existing files are skipped unless --overwrite.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from overcooked_featurize import featurize_obs
from environment_representation_probe_review import (
    CROSS_MODELS,
    LAYOUT_FAMILIES,
    MODEL_FILE_LABELS,
    MODEL_SPECS,
    REPO_ROOT,
    ScannedRNN,
    checkpoint_path,
    load_config,
    load_params,
    make_family_env,
    parse_model_spec,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Collect layout-probe data.")
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
        "--behaviors", nargs="+", choices=("own", "random", "cross"),
        default=["own", "random", "cross"],
    )
    parser.add_argument("--seeds", type=int, nargs="+", default=list(range(6)))
    parser.add_argument("--episodes-per-layout", type=int, default=20)
    parser.add_argument("--steps", type=int, default=128)
    parser.add_argument("--samples-per-episode", type=int, default=32)
    parser.add_argument("--rollout-seed", type=int, default=1701)
    parser.add_argument("--argmax", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--config", type=Path,
        default=REPO_ROOT / "baselines" / "CEC_UED" / "config"
        / "ippo_overcooked_CEC_gradient.yaml",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=None,
        help="Defaults to <model-root>/analysis/representation_probe/data.",
    )
    return parser.parse_args()


def make_rollout_fn(
    network, env, reset_name: str, config: dict, steps: int, argmax: bool,
    separate_hidden: bool, behavior: str, behavior_network,
):
    """Jitted (params, behavior_params, episode_keys) -> (actor, value, raw, featurized).

    Params are arguments (not constants), so one compilation serves all seeds.
    """
    num_agents = env.num_agents
    hidden_dim = int(config["GRU_HIDDEN_DIM"])

    def one_episode(params, behavior_params, key):
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
            featurized = jax.vmap(featurize_obs)(
                jnp.stack([obs[agent] for agent in env.agents])
            )
            network_input = (
                obs_batch[jnp.newaxis, :],
                done[jnp.newaxis, :],
                positions[jnp.newaxis, :],
            )
            outputs, captured = network.apply(
                params, hidden, network_input, mutable=["intermediates"],
            )
            hidden, policy, _ = outputs
            actor = captured["intermediates"]["actor_penultimate"][0][0]
            value = captured["intermediates"]["critic_penultimate"][0][0]
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
            return carry, (actor, value, obs_batch.astype(jnp.float32), featurized)

        carry = (env_state, obs, done, hidden, behavior_hidden, key)
        _, outputs = jax.lax.scan(step, carry, None, length=steps)
        return outputs

    return jax.jit(jax.vmap(one_episode, in_axes=(None, None, 0)))


def collect(rollout_fns, params, behavior_params, args, model_seed: int):
    sample_steps = np.linspace(
        0, args.steps - 1, min(args.samples_per_episode, args.steps), dtype=int,
    )
    base_key = jax.random.PRNGKey(args.rollout_seed + 1009 * model_seed)
    family_keys = jax.random.split(base_key, len(LAYOUT_FAMILIES))
    data = {
        "actor": [], "value": [], "raw": [], "featurized": [],
        "labels": [], "episode_ids": [],
    }

    for label, (rollout_fn, family_key) in enumerate(zip(rollout_fns, family_keys)):
        episode_keys = jax.random.split(family_key, args.episodes_per_layout)
        outputs = rollout_fn(params, behavior_params, episode_keys)
        # [episode, step, agent, dim] -> sampled steps -> [N, dim]
        actor, value, raw, featurized = (
            np.asarray(x)[:, sample_steps] for x in outputs
        )
        per_episode = actor.shape[1] * actor.shape[2]
        for name, x in (
            ("actor", actor), ("value", value), ("raw", raw),
            ("featurized", featurized),
        ):
            data[name].append(x.reshape(-1, x.shape[-1]))
        data["labels"].append(np.full(actor.shape[0] * per_episode, label))
        data["episode_ids"].append(np.repeat(
            label * args.episodes_per_layout + np.arange(actor.shape[0]),
            per_episode,
        ))
        print(f"    {LAYOUT_FAMILIES[label][0]}: {actor.shape[0] * per_episode} vectors")

    return {name: np.concatenate(x) for name, x in data.items()}


def main():
    args = parse_args()
    args.model_root = args.model_root.expanduser()
    if args.output_dir is None:
        args.output_dir = args.model_root / "analysis" / "representation_probe" / "data"
    config = load_config(args.config)
    env = make_family_env(config)
    action_dim = env.action_space("agent_0").n

    for model, num_envs in args.models:
        spec = MODEL_SPECS[model]
        # The value network also sows actor_penultimate, so one rollout gives both.
        network = spec["value_network"](action_dim, config=config)
        cross_model = CROSS_MODELS[model]
        cross_network = MODEL_SPECS[cross_model]["actor_network"](
            action_dim, config=config,
        )
        for behavior in args.behaviors:
            rollout_fns = [
                make_rollout_fn(
                    network, env, reset_name, config, args.steps, args.argmax,
                    bool(spec["value_separate_hidden"]), behavior, cross_network,
                )
                for _, reset_name in LAYOUT_FAMILIES
            ]
            for seed in args.seeds:
                out_path = args.output_dir / f"behavior_{behavior}" / (
                    f"{MODEL_FILE_LABELS[model]}_{num_envs}_seed{seed}.npz"
                )
                if out_path.exists() and not args.overwrite:
                    print(f"Exists, skipping: {out_path}")
                    continue
                ckpt = checkpoint_path(args.model_root, model, num_envs, seed)
                if ckpt is None:
                    print(f"Missing checkpoint: model={model}, envs={num_envs}, seed={seed}")
                    continue
                behavior_ckpt = ""
                behavior_params = None
                if behavior == "cross":
                    behavior_ckpt = checkpoint_path(
                        args.model_root, cross_model, num_envs, seed
                    )
                    if behavior_ckpt is None:
                        print(f"Missing cross checkpoint: {cross_model} seed {seed}")
                        continue
                    behavior_params = load_params(behavior_ckpt)
                print(f"{spec['display']} envs{num_envs} seed {seed} behavior={behavior}")
                data = collect(
                    rollout_fns, load_params(ckpt), behavior_params, args, seed,
                )
                out_path.parent.mkdir(parents=True, exist_ok=True)
                np.savez_compressed(
                    out_path, **data,
                    model=model, num_envs=num_envs, seed=seed, behavior=behavior,
                    checkpoint=str(ckpt), behavior_checkpoint=str(behavior_ckpt),
                    rollout_seed=args.rollout_seed, steps=args.steps,
                    episodes_per_layout=args.episodes_per_layout,
                    samples_per_episode=args.samples_per_episode,
                )
                print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
