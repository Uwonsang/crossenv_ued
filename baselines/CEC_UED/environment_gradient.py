"""Environment-conditioned policy/value gradient alignment diagnostics."""

from typing import Sequence

import flax
import jax
import jax.numpy as jnp

from static_grid_grouping import unique_static_grid_signatures


ENVIRONMENT_GRADIENT_NORM_METRIC_NAMES = tuple(
    f"{gradient_name}_{statistic}"
    for gradient_name in (
        "policy_norm",
        "weighted_value_norm",
    )
    for statistic in ("mean", "cv", "p10", "p90", "iqm")
)


ENVIRONMENT_GRADIENT_METRIC_NAMES = (
    "policy_policy_mean_cosine",
    "value_value_mean_cosine",
    "policy_value_same_static_grid_cosine",
    "policy_value_cross_static_grid_cosine",
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
) + ENVIRONMENT_GRADIENT_NORM_METRIC_NAMES


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
    elif metric_name in ENVIRONMENT_GRADIENT_NORM_METRIC_NAMES:
        namespace = "env_gradient_norm"
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


def compute_static_grid_conditioned_gradients(
    *,
    network,
    params,
    initial_hstates,
    trajectories,
    normalized_advantages: jnp.ndarray,
    targets: jnp.ndarray,
    sample_mask: jnp.ndarray,
    static_signatures: jnp.ndarray,
    max_static_grids: int,
    shared_param_keys: Sequence[str],
    clip_eps: float,
    entropy_coef: float,
    chunk_size: int,
    sketch_size: int = 512,
    epsilon: float = 1e-12,
    policy_gsnr_param_keys: Sequence[str] = None,
    value_gsnr_param_keys: Sequence[str] = None,
    value_loss_coefficient: float = 1.0,
):
    """Compute diagnostics over distinct static-grid gradients.

    Every actor-state in the first recurrent minibatch is assigned using its
    current signature, including configurations introduced by rollout resets.
    The implementation differentiates one configuration at a time and never
    materializes a ``unique_grids x parameters`` tensor.
    """
    del chunk_size
    num_slots = int(max_static_grids)
    if int(sketch_size) <= 0:
        raise ValueError("sketch_size must be positive.")
    shared_params, merge_params = _partition_params(params, shared_param_keys)
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

    def losses_for_slot(candidate_shared_params, hstate, trajectory, adv, target, mask):
        _, pi, value = network.apply(
            merge_params(candidate_shared_params),
            hstate,
            (trajectory.obs, trajectory.done, trajectory.agent_positions),
        )
        weights = mask.astype(jnp.float32)
        denominator = jnp.maximum(jnp.sum(weights), 1.0)
        ratio = jnp.exp(pi.log_prob(trajectory.action) - trajectory.log_prob)
        actor_terms = -jnp.minimum(
            ratio * adv,
            jnp.clip(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * adv,
        )
        policy_loss = jnp.sum(
            (actor_terms - entropy_coef * pi.entropy()) * weights
        ) / denominator
        clipped_value = trajectory.value + (value - trajectory.value).clip(
            -clip_eps, clip_eps
        )
        value_terms = 0.5 * jnp.maximum(
            jnp.square(value - target), jnp.square(clipped_value - target)
        )
        value_loss = jnp.sum(value_terms * weights) / denominator
        return policy_loss, value_loss

    policy_gradient = jax.grad(
        lambda candidate, *args: losses_for_slot(candidate, *args)[0]
    )
    value_gradient = jax.grad(
        lambda candidate, *args: losses_for_slot(candidate, *args)[1]
    )
    representatives, _, retained_group_count, _ = (
        unique_static_grid_signatures(
            static_signatures,
            sample_mask,
            max_groups=max_static_grids,
        )
    )
    zero_tree = jax.tree.map(jnp.zeros_like, shared_params)
    nan_vector = jnp.full((num_slots,), jnp.nan, dtype=jnp.float32)
    initial_state = (
        zero_tree, zero_tree, jnp.asarray(0.0, jnp.float32),
        zero_tree, zero_tree,
        zero_tree, zero_tree, zero_tree, zero_tree,
        jnp.asarray(0.0, jnp.float32), jnp.asarray(0.0, jnp.float32),
        jnp.asarray(0.0, jnp.float32), jnp.asarray(0.0, jnp.float32),
        jnp.zeros((int(sketch_size), int(sketch_size)), jnp.float32),
        jnp.zeros((int(sketch_size), int(sketch_size)), jnp.float32),
        nan_vector, nan_vector,
        jnp.asarray(0, jnp.int32),
    )

    def finalize_group(state):
        (
            current_policy, current_value, current_sample_count,
            normalized_policy_sum, normalized_value_sum,
            raw_policy_sum, raw_value_sum,
            raw_policy_squared_sum, raw_value_squared_sum,
            raw_policy_squared_norm_sum, raw_value_squared_norm_sum,
            same_dot_sum, valid_count,
            policy_sketch_covariance, value_sketch_covariance,
            policy_norms, value_norms, group_index,
        ) = state
        denominator = jnp.maximum(current_sample_count, 1.0)
        policy = jax.tree.map(lambda x: x / denominator, current_policy)
        value = jax.tree.map(lambda x: x / denominator, current_value)
        policy_squared_norm = sum(
            jnp.sum(jnp.square(x)) for x in jax.tree_util.tree_leaves(policy)
        )
        value_squared_norm = sum(
            jnp.sum(jnp.square(x)) for x in jax.tree_util.tree_leaves(value)
        )
        policy_norm = jnp.sqrt(policy_squared_norm)
        value_norm = jnp.sqrt(value_squared_norm)
        has_samples = current_sample_count > 0.0
        policy_finite = jnp.logical_and(has_samples, jnp.isfinite(policy_norm))
        value_finite = jnp.logical_and(has_samples, jnp.isfinite(value_norm))
        valid = jnp.logical_and(
            jnp.logical_and(policy_finite, value_finite),
            jnp.logical_and(policy_norm > epsilon, value_norm > epsilon),
        )
        policy_inverse = jnp.where(valid, 1.0 / jnp.maximum(policy_norm, epsilon), 0.0)
        value_inverse = jnp.where(valid, 1.0 / jnp.maximum(value_norm, epsilon), 0.0)
        normalized_policy = jax.tree.map(lambda x: x * policy_inverse, policy)
        normalized_value = jax.tree.map(lambda x: x * value_inverse, value)
        normalized_policy_sum = jax.tree.map(jnp.add, normalized_policy_sum, normalized_policy)
        normalized_value_sum = jax.tree.map(jnp.add, normalized_value_sum, normalized_value)
        valid_float = valid.astype(jnp.float32)
        raw_policy_sum = jax.tree.map(lambda total, x: total + x * valid_float, raw_policy_sum, policy)
        raw_value_sum = jax.tree.map(lambda total, x: total + x * valid_float, raw_value_sum, value)
        raw_policy_squared_sum = jax.tree.map(
            lambda total, x: total + jnp.square(x) * valid_float,
            raw_policy_squared_sum, policy,
        )
        raw_value_squared_sum = jax.tree.map(
            lambda total, x: total + jnp.square(x) * valid_float,
            raw_value_squared_sum, value,
        )
        same_dot = sum(
            jnp.sum(p * v) for p, v in zip(
                jax.tree_util.tree_leaves(normalized_policy),
                jax.tree_util.tree_leaves(normalized_value),
            )
        )
        policy_sketch = jnp.zeros((int(sketch_size),), dtype=jnp.float32)
        value_sketch = jnp.zeros_like(policy_sketch)
        parameter_offset = 0
        for policy_leaf, value_leaf in zip(
            jax.tree_util.tree_leaves(normalized_policy),
            jax.tree_util.tree_leaves(normalized_value),
        ):
            flat_policy = policy_leaf.reshape(-1)
            flat_value = value_leaf.reshape(-1)
            parameter_indices = (
                jnp.arange(flat_policy.size, dtype=jnp.uint32)
                + jnp.asarray(parameter_offset, dtype=jnp.uint32)
            )
            mixed = parameter_indices + jnp.uint32(0x9E3779B9)
            mixed = (mixed ^ (mixed >> jnp.uint32(16))) * jnp.uint32(0x7FEB352D)
            mixed = (mixed ^ (mixed >> jnp.uint32(15))) * jnp.uint32(0x846CA68B)
            mixed = mixed ^ (mixed >> jnp.uint32(16))
            bins = (mixed % jnp.uint32(int(sketch_size))).astype(jnp.int32)
            sign_hash = (mixed ^ jnp.uint32(0xA5A5A5A5)) * jnp.uint32(0x27D4EB2D)
            signs = jnp.where(
                ((sign_hash ^ (sign_hash >> jnp.uint32(15))) & jnp.uint32(1)) == 0,
                1.0, -1.0,
            ).astype(jnp.float32)
            policy_sketch = policy_sketch.at[bins].add(flat_policy * signs)
            value_sketch = value_sketch.at[bins].add(flat_value * signs)
            parameter_offset += int(flat_policy.size)
        policy_sketch_norm = jnp.linalg.norm(policy_sketch)
        value_sketch_norm = jnp.linalg.norm(value_sketch)
        normalized_policy_sketch = jnp.where(
            valid,
            policy_sketch / jnp.maximum(policy_sketch_norm, epsilon),
            jnp.zeros_like(policy_sketch),
        )
        normalized_value_sketch = jnp.where(
            valid,
            value_sketch / jnp.maximum(value_sketch_norm, epsilon),
            jnp.zeros_like(value_sketch),
        )
        policy_sketch_covariance += jnp.outer(
            normalized_policy_sketch, normalized_policy_sketch
        )
        value_sketch_covariance += jnp.outer(
            normalized_value_sketch, normalized_value_sketch
        )
        policy_norms = policy_norms.at[group_index].set(jnp.where(policy_finite, policy_norm, jnp.nan))
        value_norms = value_norms.at[group_index].set(jnp.where(value_finite, value_norm, jnp.nan))
        return (
            zero_tree, zero_tree, jnp.asarray(0.0, jnp.float32),
            normalized_policy_sum, normalized_value_sum,
            raw_policy_sum, raw_value_sum,
            raw_policy_squared_sum, raw_value_squared_sum,
            raw_policy_squared_norm_sum + policy_squared_norm * valid_float,
            raw_value_squared_norm_sum + value_squared_norm * valid_float,
            same_dot_sum + same_dot, valid_count + valid_float,
            policy_sketch_covariance, value_sketch_covariance,
            policy_norms, value_norms,
            group_index + 1,
        )

    def accumulate_group(group_index, state):
        signature_match = jnp.all(
            static_signatures == representatives[group_index], axis=-1
        )
        mask = jnp.logical_and(sample_mask, signature_match)
        policy_gradient_value = policy_gradient(
            shared_params,
            initial_hstates,
            trajectories,
            normalized_advantages,
            targets,
            mask,
        )
        value_gradient_value = value_gradient(
            shared_params,
            initial_hstates,
            trajectories,
            normalized_advantages,
            targets,
            mask,
        )
        sample_count = jnp.sum(mask.astype(jnp.float32))
        current_policy = jax.tree.map(
            lambda value: value * sample_count, policy_gradient_value
        )
        current_value = jax.tree.map(
            lambda value: value * sample_count, value_gradient_value
        )
        state = (
            current_policy, current_value, sample_count, *state[3:]
        )
        return finalize_group(state)

    state = jax.lax.fori_loop(
        0,
        retained_group_count,
        accumulate_group,
        initial_state,
    )
    (
        _, _, _, normalized_policy_sum, normalized_value_sum,
        raw_policy_sum, raw_value_sum,
        raw_policy_squared_sum, raw_value_squared_sum,
        raw_policy_squared_norm_sum, raw_value_squared_norm_sum,
        same_dot_sum, valid_count,
        policy_sketch_covariance, value_sketch_covariance,
        policy_norms, value_norms, _,
    ) = state

    def distribution_statistics(values):
        finite = jnp.isfinite(values)
        count = jnp.sum(finite.astype(jnp.int32))
        count_float = jnp.maximum(count.astype(jnp.float32), 1.0)
        mean = jnp.sum(jnp.where(finite, values, 0.0)) / count_float
        variance = jnp.sum(jnp.where(finite, jnp.square(values - mean), 0.0)) / count_float
        sorted_values = jnp.sort(jnp.where(finite, values, jnp.inf))
        def quantile(probability):
            position = probability * jnp.maximum(count_float - 1.0, 0.0)
            lower = jnp.floor(position).astype(jnp.int32)
            upper = jnp.ceil(position).astype(jnp.int32)
            fraction = position - lower.astype(jnp.float32)
            return sorted_values[lower] + fraction * (sorted_values[upper] - sorted_values[lower])
        ranks = jnp.arange(num_slots, dtype=jnp.float32)
        trim_start, trim_end = 0.25 * count_float, 0.75 * count_float
        weights = jnp.maximum(jnp.minimum(ranks + 1.0, trim_end) - jnp.maximum(ranks, trim_start), 0.0)
        iqm = jnp.sum(jnp.where(weights > 0.0, sorted_values, 0.0) * weights) / jnp.maximum(jnp.sum(weights), epsilon)
        nan = jnp.asarray(jnp.nan, jnp.float32)
        return {
            "mean": jnp.where(count > 0, mean, nan),
            "cv": jnp.where(count > 0, jnp.sqrt(jnp.maximum(variance, 0.0)) / jnp.maximum(jnp.abs(mean), epsilon), nan),
            "p10": jnp.where(count > 0, quantile(0.10), nan),
            "p90": jnp.where(count > 0, quantile(0.90), nan),
            "iqm": jnp.where(count > 0, iqm, nan),
        }

    norm_metrics = {}
    for name, values in {
        "policy_norm": policy_norms,
        "weighted_value_norm": jnp.abs(jnp.asarray(value_loss_coefficient, jnp.float32)) * value_norms,
    }.items():
        for statistic, result in distribution_statistics(values).items():
            norm_metrics[f"{name}_{statistic}"] = result

    def effective_rank(sketch_covariance):
        # G G^T and G^T G share the same non-zero eigenvalues. Using the
        # sketch-space covariance keeps this matrix 512 x 512 even when the
        # first minibatch contains more than NUM_ENVS unique grids.
        eigenvalues = jnp.maximum(
            jnp.linalg.eigvalsh(sketch_covariance), 0.0
        )
        eigenvalue_sum = jnp.sum(eigenvalues)
        probabilities = eigenvalues / jnp.maximum(eigenvalue_sum, epsilon)
        entropy = -jnp.sum(jnp.where(probabilities > 0.0, probabilities * jnp.log(jnp.maximum(probabilities, epsilon)), 0.0))
        return jnp.where(eigenvalue_sum > epsilon, jnp.exp(entropy), jnp.nan)

    policy_effective_rank = effective_rank(policy_sketch_covariance)
    value_effective_rank = effective_rank(value_sketch_covariance)
    def gradient_snr(raw_sum, squared_norm_sum):
        count = jnp.maximum(valid_count, 1.0)
        mean_squared_norm = sum(jnp.sum(jnp.square(x / count)) for x in jax.tree_util.tree_leaves(raw_sum))
        noise = jnp.maximum(squared_norm_sum / count - mean_squared_norm, 0.0)
        result = jnp.sqrt(mean_squared_norm) / jnp.sqrt(jnp.maximum(noise, epsilon))
        return jnp.where(valid_count > 1.0, result, jnp.nan)
    policy_snr = gradient_snr(raw_policy_sum, raw_policy_squared_norm_sum)
    value_snr = gradient_snr(raw_value_sum, raw_value_squared_norm_sum)

    def parameterwise_gsnr(raw_sum, raw_squared_sum, param_keys):
        count = jnp.maximum(valid_count, 1.0)
        variance_denominator = jnp.maximum(valid_count - 1.0, 1.0)
        total = jnp.asarray(0.0, jnp.float32)
        log_total = jnp.asarray(0.0, jnp.float32)
        parameter_count = jnp.asarray(0.0, jnp.float32)
        for key in param_keys:
            for sum_leaf, squared_leaf in zip(
                jax.tree_util.tree_leaves(raw_sum[key]),
                jax.tree_util.tree_leaves(raw_squared_sum[key]),
            ):
                mean = sum_leaf / count
                variance = jnp.maximum(squared_leaf - count * jnp.square(mean), 0.0) / variance_denominator
                coordinate = jnp.square(mean) / (variance + epsilon)
                total += jnp.sum(coordinate)
                log_total += jnp.sum(jnp.log10(jnp.maximum(coordinate, epsilon)))
                parameter_count += jnp.asarray(sum_leaf.size, jnp.float32)
        valid_result = jnp.logical_and(valid_count > 1.0, parameter_count > 0.0)
        return (
            jnp.where(valid_result, total / jnp.maximum(parameter_count, 1.0), jnp.nan),
            jnp.where(valid_result, log_total / jnp.maximum(parameter_count, 1.0), jnp.nan),
        )
    policy_gsnr, policy_log_gsnr = parameterwise_gsnr(raw_policy_sum, raw_policy_squared_sum, policy_gsnr_param_keys)
    value_gsnr, value_log_gsnr = parameterwise_gsnr(raw_value_sum, raw_value_squared_sum, value_gsnr_param_keys)
    policy_sum_norm = sum(jnp.sum(jnp.square(x)) for x in jax.tree_util.tree_leaves(normalized_policy_sum))
    value_sum_norm = sum(jnp.sum(jnp.square(x)) for x in jax.tree_util.tree_leaves(normalized_value_sum))
    policy_value_sum_dot = sum(
        jnp.sum(p * v) for p, v in zip(
            jax.tree_util.tree_leaves(normalized_policy_sum),
            jax.tree_util.tree_leaves(normalized_value_sum),
        )
    )
    off_diagonal_count = valid_count * (valid_count - 1.0)
    nan = jnp.asarray(jnp.nan, jnp.float32)
    return {
        "policy_policy_mean_cosine": jnp.where(off_diagonal_count > 0.0, (policy_sum_norm - valid_count) / off_diagonal_count, nan),
        "value_value_mean_cosine": jnp.where(off_diagonal_count > 0.0, (value_sum_norm - valid_count) / off_diagonal_count, nan),
        "policy_value_same_static_grid_cosine": jnp.where(valid_count > 0.0, same_dot_sum / valid_count, nan),
        "policy_value_cross_static_grid_cosine": jnp.where(off_diagonal_count > 0.0, (policy_value_sum_dot - same_dot_sum) / off_diagonal_count, nan),
        "policy_effective_rank": policy_effective_rank,
        "value_effective_rank": value_effective_rank,
        "policy_effective_rank_ratio": jnp.where(valid_count > 0.0, policy_effective_rank / valid_count, nan),
        "value_effective_rank_ratio": jnp.where(valid_count > 0.0, value_effective_rank / valid_count, nan),
        "policy_snr": policy_snr,
        "value_snr": value_snr,
        "policy_log_snr": jnp.log10(jnp.maximum(policy_snr, epsilon)),
        "value_log_snr": jnp.log10(jnp.maximum(value_snr, epsilon)),
        "policy_parameterwise_gsnr_mean": policy_gsnr,
        "value_parameterwise_gsnr_mean": value_gsnr,
        "policy_parameterwise_gsnr_mean_log10": policy_log_gsnr,
        "value_parameterwise_gsnr_mean_log10": value_log_gsnr,
        **norm_metrics,
    }
