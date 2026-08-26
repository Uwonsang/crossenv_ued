"""Environment-conditioned policy/value gradient alignment diagnostics."""

from typing import Sequence

import flax
import jax
import jax.numpy as jnp


ENVIRONMENT_GRADIENT_METRIC_NAMES = (
    "policy_policy_mean_cosine",
    "value_value_mean_cosine",
    "policy_value_same_env_cosine",
    "policy_value_cross_env_cosine",
)


def empty_environment_gradient_metrics(dtype=jnp.float32):
    return {
        name: jnp.asarray(jnp.nan, dtype=dtype)
        for name in ENVIRONMENT_GRADIENT_METRIC_NAMES
    }


def _partition_params(params, selected_param_keys: Sequence[str]):
    mutable_params = flax.core.unfreeze(params)
    root_params = mutable_params["params"]
    selected_keys = set(selected_param_keys)
    selected_params = {
        key: value for key, value in root_params.items()
        if key in selected_keys
    }
    frozen_params = {
        key: value for key, value in root_params.items()
        if key not in selected_keys
    }
    other_collections = {
        key: value for key, value in mutable_params.items()
        if key != "params"
    }

    def merge(candidate_params):
        merged = dict(frozen_params)
        merged.update(candidate_params)
        return flax.core.freeze({**other_collections, "params": merged})

    return flax.core.freeze(selected_params), merge


def compute_environment_conditioned_cosines(
    *,
    network,
    params,
    initial_hstates,
    trajectories,
    normalized_advantages: jnp.ndarray,
    targets: jnp.ndarray,
    sample_mask: jnp.ndarray,
    shared_param_keys: Sequence[str],
    clip_eps: float,
    entropy_coef: float,
    chunk_size: int,
    epsilon: float = 1e-12,
):
    """Compute alignment between gradients conditioned on static grids.

    Inputs have a leading environment dimension. Each environment contains
    the selected agent trajectories assigned to the optimizer's first
    minibatch. ``sample_mask`` excludes unselected agents and states after a
    mid-rollout reset changes the static grid.
    """

    num_environments = int(sample_mask.shape[0])
    if num_environments % int(chunk_size) != 0:
        raise ValueError("chunk_size must evenly divide NUM_ENVS.")

    shared_params, merge_params = _partition_params(
        params, shared_param_keys
    )

    def losses_for_environment(
        candidate_shared_params,
        initial_hstate,
        trajectory,
        environment_advantages,
        environment_targets,
        environment_mask,
    ):
        full_params = merge_params(candidate_shared_params)
        _, pi, value = network.apply(
            full_params,
            initial_hstate,
            (
                trajectory.obs,
                trajectory.done,
                trajectory.agent_positions,
            ),
        )
        mask = environment_mask.astype(jnp.float32)
        denominator = jnp.maximum(mask.sum(), 1.0)

        log_prob = pi.log_prob(trajectory.action)
        ratio = jnp.exp(log_prob - trajectory.log_prob)
        actor_terms = -jnp.minimum(
            ratio * environment_advantages,
            jnp.clip(ratio, 1.0 - clip_eps, 1.0 + clip_eps)
            * environment_advantages,
        )
        policy_terms = actor_terms - entropy_coef * pi.entropy()
        policy_loss = jnp.sum(policy_terms * mask) / denominator

        value_pred_clipped = trajectory.value + (
            value - trajectory.value
        ).clip(-clip_eps, clip_eps)
        value_terms = 0.5 * jnp.maximum(
            jnp.square(value - environment_targets),
            jnp.square(value_pred_clipped - environment_targets),
        )
        value_loss = jnp.sum(value_terms * mask) / denominator
        return policy_loss, value_loss

    policy_gradient = jax.grad(
        lambda candidate, *args: losses_for_environment(
            candidate, *args
        )[0]
    )
    value_gradient = jax.grad(
        lambda candidate, *args: losses_for_environment(
            candidate, *args
        )[1]
    )

    num_chunks = num_environments // int(chunk_size)

    def chunk(values):
        return values.reshape(
            (num_chunks, int(chunk_size)) + values.shape[1:]
        )

    chunked_inputs = (
        jax.tree.map(chunk, initial_hstates),
        jax.tree.map(chunk, trajectories),
        chunk(normalized_advantages),
        chunk(targets),
        chunk(sample_mask),
    )
    zero_sum = jax.tree.map(jnp.zeros_like, shared_params)
    initial_state = (
        zero_sum,
        zero_sum,
        jnp.asarray(0.0, dtype=jnp.float32),
        jnp.asarray(0.0, dtype=jnp.float32),
    )

    def accumulate_chunk(state, batch):
        policy_sum, value_sum, same_dot_sum, valid_count = state
        (
            batch_hstates,
            batch_trajectories,
            batch_advantages,
            batch_targets,
            batch_mask,
        ) = batch
        policy_grads = jax.vmap(
            policy_gradient,
            in_axes=(None, 0, 0, 0, 0, 0),
        )(
            shared_params,
            batch_hstates,
            batch_trajectories,
            batch_advantages,
            batch_targets,
            batch_mask,
        )
        value_grads = jax.vmap(
            value_gradient,
            in_axes=(None, 0, 0, 0, 0, 0),
        )(
            shared_params,
            batch_hstates,
            batch_trajectories,
            batch_advantages,
            batch_targets,
            batch_mask,
        )

        policy_squared_norm = jnp.zeros(
            (int(chunk_size),), dtype=jnp.float32
        )
        value_squared_norm = jnp.zeros(
            (int(chunk_size),), dtype=jnp.float32
        )
        for policy_leaf, value_leaf in zip(
            jax.tree_util.tree_leaves(policy_grads),
            jax.tree_util.tree_leaves(value_grads),
        ):
            axes = tuple(range(1, policy_leaf.ndim))
            policy_squared_norm += jnp.sum(
                jnp.square(policy_leaf), axis=axes
            )
            value_squared_norm += jnp.sum(
                jnp.square(value_leaf), axis=axes
            )

        policy_norm = jnp.sqrt(policy_squared_norm)
        value_norm = jnp.sqrt(value_squared_norm)
        valid = jnp.logical_and(
            batch_mask.sum(axis=(1, 2)) > 0,
            jnp.logical_and(policy_norm > epsilon, value_norm > epsilon),
        )
        policy_inverse_norm = jnp.where(
            valid, 1.0 / jnp.maximum(policy_norm, epsilon), 0.0
        )
        value_inverse_norm = jnp.where(
            valid, 1.0 / jnp.maximum(value_norm, epsilon), 0.0
        )

        chunk_same_dot = jnp.zeros(
            (int(chunk_size),), dtype=jnp.float32
        )

        def add_normalized(total, gradient_leaf, inverse_norm):
            shape = (int(chunk_size),) + (1,) * (gradient_leaf.ndim - 1)
            return total + jnp.sum(
                gradient_leaf * inverse_norm.reshape(shape), axis=0
            )

        for policy_leaf, value_leaf in zip(
            jax.tree_util.tree_leaves(policy_grads),
            jax.tree_util.tree_leaves(value_grads),
        ):
            shape = (int(chunk_size),) + (1,) * (policy_leaf.ndim - 1)
            normalized_policy = (
                policy_leaf * policy_inverse_norm.reshape(shape)
            )
            normalized_value = (
                value_leaf * value_inverse_norm.reshape(shape)
            )
            axes = tuple(range(1, policy_leaf.ndim))
            chunk_same_dot += jnp.sum(
                normalized_policy * normalized_value, axis=axes
            )

        policy_sum = jax.tree.map(
            lambda total, gradient: add_normalized(
                total, gradient, policy_inverse_norm
            ),
            policy_sum,
            policy_grads,
        )
        value_sum = jax.tree.map(
            lambda total, gradient: add_normalized(
                total, gradient, value_inverse_norm
            ),
            value_sum,
            value_grads,
        )
        return (
            policy_sum,
            value_sum,
            same_dot_sum + jnp.sum(chunk_same_dot),
            valid_count + jnp.sum(valid.astype(jnp.float32)),
        ), None

    (
        policy_sum,
        value_sum,
        same_dot_sum,
        valid_count,
    ), _ = jax.lax.scan(accumulate_chunk, initial_state, chunked_inputs)

    policy_sum_squared_norm = jnp.asarray(0.0, dtype=jnp.float32)
    value_sum_squared_norm = jnp.asarray(0.0, dtype=jnp.float32)
    policy_value_sum_dot = jnp.asarray(0.0, dtype=jnp.float32)
    for policy_leaf, value_leaf in zip(
        jax.tree_util.tree_leaves(policy_sum),
        jax.tree_util.tree_leaves(value_sum),
    ):
        policy_sum_squared_norm += jnp.sum(jnp.square(policy_leaf))
        value_sum_squared_norm += jnp.sum(jnp.square(value_leaf))
        policy_value_sum_dot += jnp.sum(policy_leaf * value_leaf)

    off_diagonal_count = valid_count * (valid_count - 1.0)
    policy_policy = jnp.where(
        off_diagonal_count > 0,
        (policy_sum_squared_norm - valid_count) / off_diagonal_count,
        jnp.asarray(jnp.nan, dtype=jnp.float32),
    )
    value_value = jnp.where(
        off_diagonal_count > 0,
        (value_sum_squared_norm - valid_count) / off_diagonal_count,
        jnp.asarray(jnp.nan, dtype=jnp.float32),
    )
    policy_value_same = jnp.where(
        valid_count > 0,
        same_dot_sum / valid_count,
        jnp.asarray(jnp.nan, dtype=jnp.float32),
    )
    policy_value_cross = jnp.where(
        off_diagonal_count > 0,
        (policy_value_sum_dot - same_dot_sum) / off_diagonal_count,
        jnp.asarray(jnp.nan, dtype=jnp.float32),
    )
    return {
        "policy_policy_mean_cosine": policy_policy,
        "value_value_mean_cosine": value_value,
        "policy_value_same_env_cosine": policy_value_same,
        "policy_value_cross_env_cosine": policy_value_cross,
    }
