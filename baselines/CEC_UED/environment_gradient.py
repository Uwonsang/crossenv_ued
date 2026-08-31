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
    "policy_effective_rank",
    "value_effective_rank",
    "policy_effective_rank_ratio",
    "value_effective_rank_ratio",
    "policy_snr",
    "value_snr",
    "policy_log_snr",
    "value_log_snr",
    "policy_parameterwise_gsnr_mean",
    "value_parameterwise_gsnr_mean",
    "policy_parameterwise_gsnr_mean_log10",
    "value_parameterwise_gsnr_mean_log10",
)


def empty_environment_gradient_metrics(dtype=jnp.float32):
    return {
        name: jnp.asarray(jnp.nan, dtype=dtype)
        for name in ENVIRONMENT_GRADIENT_METRIC_NAMES
    }


def environment_gradient_log_key(metric_name: str) -> str:
    """Group environment-gradient metrics by their analysis purpose."""
    if metric_name.endswith("_cosine"):
        namespace = "env_gradient_cosine"
    elif "effective_rank" in metric_name:
        namespace = "env_gradient_rank"
    elif metric_name.endswith("_snr"):
        namespace = "env_gradient_snr"
    elif "parameterwise_gsnr" in metric_name:
        namespace = "env_gradient_gsnr"
    else:
        raise ValueError(
            f"Unknown environment-gradient metric: {metric_name}"
        )
    return f"{namespace}/{metric_name}"


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
    sketch_size: int = 512,
    epsilon: float = 1e-12,
    policy_gsnr_param_keys: Sequence[str] = None,
    value_gsnr_param_keys: Sequence[str] = None,
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
    policy_gsnr_param_keys = tuple(
        shared_param_keys
        if policy_gsnr_param_keys is None else policy_gsnr_param_keys
    )
    value_gsnr_param_keys = tuple(
        shared_param_keys
        if value_gsnr_param_keys is None else value_gsnr_param_keys
    )
    selected_key_set = set(shared_param_keys)
    if not set(policy_gsnr_param_keys).issubset(selected_key_set):
        raise ValueError("policy GSNR keys must be in shared_param_keys.")
    if not set(value_gsnr_param_keys).issubset(selected_key_set):
        raise ValueError("value GSNR keys must be in shared_param_keys.")
    if int(sketch_size) <= 0:
        raise ValueError("sketch_size must be positive.")

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
        zero_sum,
        zero_sum,
        zero_sum,
        zero_sum,
        jnp.asarray(0.0, dtype=jnp.float32),
        jnp.asarray(0.0, dtype=jnp.float32),
        jnp.asarray(0.0, dtype=jnp.float32),
        jnp.asarray(0.0, dtype=jnp.float32),
    )

    def accumulate_chunk(state, batch):
        (
            policy_sum,
            value_sum,
            raw_policy_sum,
            raw_value_sum,
            raw_policy_squared_sum,
            raw_value_squared_sum,
            raw_policy_squared_norm_sum,
            raw_value_squared_norm_sum,
            same_dot_sum,
            valid_count,
        ) = state
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
        policy_sketch = jnp.zeros(
            (int(chunk_size), int(sketch_size)), dtype=jnp.float32
        )
        value_sketch = jnp.zeros_like(policy_sketch)

        def add_normalized(total, gradient_leaf, inverse_norm):
            shape = (int(chunk_size),) + (1,) * (gradient_leaf.ndim - 1)
            return total + jnp.sum(
                gradient_leaf * inverse_norm.reshape(shape), axis=0
            )

        parameter_offset = 0
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

            flat_policy = normalized_policy.reshape(
                (int(chunk_size), -1)
            )
            flat_value = normalized_value.reshape(
                (int(chunk_size), -1)
            )
            parameter_indices = (
                jnp.arange(flat_policy.shape[1], dtype=jnp.uint32)
                + jnp.asarray(parameter_offset, dtype=jnp.uint32)
            )
            mixed_indices = parameter_indices + jnp.uint32(0x9E3779B9)
            mixed_indices = mixed_indices ^ (mixed_indices >> jnp.uint32(16))
            mixed_indices = mixed_indices * jnp.uint32(0x7FEB352D)
            mixed_indices = mixed_indices ^ (mixed_indices >> jnp.uint32(15))
            mixed_indices = mixed_indices * jnp.uint32(0x846CA68B)
            mixed_indices = mixed_indices ^ (mixed_indices >> jnp.uint32(16))
            sketch_bins = (
                mixed_indices % jnp.uint32(int(sketch_size))
            ).astype(jnp.int32)
            sign_hash = mixed_indices ^ jnp.uint32(0xA5A5A5A5)
            sign_hash = sign_hash * jnp.uint32(0x27D4EB2D)
            sign_hash = sign_hash ^ (sign_hash >> jnp.uint32(15))
            sign_bits = sign_hash & jnp.uint32(1)
            sketch_signs = jnp.where(
                sign_bits == 0, 1.0, -1.0
            ).astype(jnp.float32)

            def add_to_sketch(row):
                return jnp.zeros(
                    (int(sketch_size),), dtype=jnp.float32
                ).at[sketch_bins].add(row * sketch_signs)

            policy_sketch += jax.vmap(add_to_sketch)(flat_policy)
            value_sketch += jax.vmap(add_to_sketch)(flat_value)
            parameter_offset += int(flat_policy.shape[1])

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

        def add_raw(total, gradient):
            shape = (int(chunk_size),) + (1,) * (gradient.ndim - 1)
            return total + jnp.sum(
                gradient * valid.astype(gradient.dtype).reshape(shape),
                axis=0,
            )

        raw_policy_sum = jax.tree.map(
            add_raw, raw_policy_sum, policy_grads
        )
        raw_value_sum = jax.tree.map(
            add_raw, raw_value_sum, value_grads
        )

        def add_raw_squared(total, gradient):
            shape = (int(chunk_size),) + (1,) * (gradient.ndim - 1)
            return total + jnp.sum(
                jnp.square(gradient)
                * valid.astype(gradient.dtype).reshape(shape),
                axis=0,
            )

        raw_policy_squared_sum = jax.tree.map(
            add_raw_squared, raw_policy_squared_sum, policy_grads
        )
        raw_value_squared_sum = jax.tree.map(
            add_raw_squared, raw_value_squared_sum, value_grads
        )
        next_state = (
            policy_sum,
            value_sum,
            raw_policy_sum,
            raw_value_sum,
            raw_policy_squared_sum,
            raw_value_squared_sum,
            raw_policy_squared_norm_sum + jnp.sum(
                jnp.where(valid, policy_squared_norm, 0.0)
            ),
            raw_value_squared_norm_sum + jnp.sum(
                jnp.where(valid, value_squared_norm, 0.0)
            ),
            same_dot_sum + jnp.sum(chunk_same_dot),
            valid_count + jnp.sum(valid.astype(jnp.float32)),
        )
        return next_state, (policy_sketch, value_sketch, valid)

    (
        policy_sum,
        value_sum,
        raw_policy_sum,
        raw_value_sum,
        raw_policy_squared_sum,
        raw_value_squared_sum,
        raw_policy_squared_norm_sum,
        raw_value_squared_norm_sum,
        same_dot_sum,
        valid_count,
    ), sketch_outputs = jax.lax.scan(
        accumulate_chunk, initial_state, chunked_inputs
    )
    policy_sketches, value_sketches, valid_environments = sketch_outputs
    policy_sketches = policy_sketches.reshape(
        (num_environments, int(sketch_size))
    )
    value_sketches = value_sketches.reshape(
        (num_environments, int(sketch_size))
    )
    valid_environments = valid_environments.reshape((num_environments,))

    def normalize_sketches(sketches):
        norms = jnp.linalg.norm(sketches, axis=1, keepdims=True)
        return jnp.where(
            valid_environments[:, None],
            sketches / jnp.maximum(norms, epsilon),
            jnp.zeros_like(sketches),
        )

    policy_sketches = normalize_sketches(policy_sketches)
    value_sketches = normalize_sketches(value_sketches)

    def effective_rank(sketches):
        gram = sketches @ sketches.T
        eigenvalues = jnp.maximum(jnp.linalg.eigvalsh(gram), 0.0)
        eigenvalue_sum = jnp.sum(eigenvalues)
        probabilities = eigenvalues / jnp.maximum(eigenvalue_sum, epsilon)
        entropy = -jnp.sum(
            jnp.where(
                probabilities > 0,
                probabilities * jnp.log(
                    jnp.maximum(probabilities, epsilon)
                ),
                0.0,
            )
        )
        return jnp.where(
            eigenvalue_sum > epsilon,
            jnp.exp(entropy),
            jnp.asarray(jnp.nan, dtype=jnp.float32),
        )

    policy_effective_rank = effective_rank(policy_sketches)
    value_effective_rank = effective_rank(value_sketches)
    policy_effective_rank_ratio = jnp.where(
        valid_count > 0,
        policy_effective_rank / valid_count,
        jnp.asarray(jnp.nan, dtype=jnp.float32),
    )
    value_effective_rank_ratio = jnp.where(
        valid_count > 0,
        value_effective_rank / valid_count,
        jnp.asarray(jnp.nan, dtype=jnp.float32),
    )

    def gradient_snr(raw_sum, raw_squared_norm_sum):
        count = jnp.maximum(valid_count, 1.0)
        mean_squared_norm = jnp.asarray(0.0, dtype=jnp.float32)
        for leaf in jax.tree_util.tree_leaves(raw_sum):
            mean_squared_norm += jnp.sum(jnp.square(leaf / count))
        mean_noise_squared_norm = jnp.maximum(
            raw_squared_norm_sum / count - mean_squared_norm,
            0.0,
        )
        snr = jnp.sqrt(mean_squared_norm) / jnp.sqrt(
            jnp.maximum(mean_noise_squared_norm, epsilon)
        )
        return jnp.where(
            valid_count > 1,
            snr,
            jnp.asarray(jnp.nan, dtype=jnp.float32),
        )

    policy_snr = gradient_snr(
        raw_policy_sum, raw_policy_squared_norm_sum
    )
    value_snr = gradient_snr(
        raw_value_sum, raw_value_squared_norm_sum
    )
    policy_log_snr = jnp.log10(jnp.maximum(policy_snr, epsilon))
    value_log_snr = jnp.log10(jnp.maximum(value_snr, epsilon))

    def parameterwise_gsnr(raw_sum, raw_squared_sum, param_keys):
        count = jnp.maximum(valid_count, 1.0)
        variance_denominator = jnp.maximum(valid_count - 1.0, 1.0)
        gsnr_sum = jnp.asarray(0.0, dtype=jnp.float32)
        log10_gsnr_sum = jnp.asarray(0.0, dtype=jnp.float32)
        active_parameter_count = jnp.asarray(0.0, dtype=jnp.float32)
        for param_key in param_keys:
            for sum_leaf, squared_sum_leaf in zip(
                jax.tree_util.tree_leaves(raw_sum[param_key]),
                jax.tree_util.tree_leaves(raw_squared_sum[param_key]),
            ):
                sum_leaf = sum_leaf.astype(jnp.float32)
                squared_sum_leaf = squared_sum_leaf.astype(jnp.float32)
                mean = sum_leaf / count
                centered_squared_sum = jnp.maximum(
                    squared_sum_leaf - count * jnp.square(mean), 0.0
                )
                sample_variance = (
                    centered_squared_sum / variance_denominator
                )
                coordinate_gsnr = jnp.square(mean) / (
                    sample_variance + epsilon
                )
                gsnr_sum += jnp.sum(coordinate_gsnr)
                log10_gsnr_sum += jnp.sum(
                    jnp.log10(jnp.maximum(coordinate_gsnr, epsilon))
                )
                active_parameter_count += jnp.asarray(
                    sum_leaf.size, dtype=jnp.float32
                )

        valid_result = jnp.logical_and(
            valid_count > 1, active_parameter_count > 0
        )
        denominator = jnp.maximum(active_parameter_count, 1.0)
        mean_gsnr = gsnr_sum / denominator
        mean_log10_gsnr = log10_gsnr_sum / denominator
        nan = jnp.asarray(jnp.nan, dtype=jnp.float32)
        return (
            jnp.where(valid_result, mean_gsnr, nan),
            jnp.where(valid_result, mean_log10_gsnr, nan),
        )

    (
        policy_parameterwise_gsnr_mean,
        policy_parameterwise_gsnr_mean_log10,
    ) = parameterwise_gsnr(
        raw_policy_sum,
        raw_policy_squared_sum,
        policy_gsnr_param_keys,
    )
    (
        value_parameterwise_gsnr_mean,
        value_parameterwise_gsnr_mean_log10,
    ) = parameterwise_gsnr(
        raw_value_sum,
        raw_value_squared_sum,
        value_gsnr_param_keys,
    )

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
        "policy_effective_rank": policy_effective_rank,
        "value_effective_rank": value_effective_rank,
        "policy_effective_rank_ratio": policy_effective_rank_ratio,
        "value_effective_rank_ratio": value_effective_rank_ratio,
        "policy_snr": policy_snr,
        "value_snr": value_snr,
        "policy_log_snr": policy_log_snr,
        "value_log_snr": value_log_snr,
        "policy_parameterwise_gsnr_mean": policy_parameterwise_gsnr_mean,
        "value_parameterwise_gsnr_mean": value_parameterwise_gsnr_mean,
        "policy_parameterwise_gsnr_mean_log10": (
            policy_parameterwise_gsnr_mean_log10
        ),
        "value_parameterwise_gsnr_mean_log10": (
            value_parameterwise_gsnr_mean_log10
        ),
    }
