"""JAX/Flax representation diagnostics."""

import jax
import jax.numpy as jnp


def weight_l2_norm(params):
    """Global L2 norm of every trainable parameter, including biases."""
    squared_norm = sum(
        jnp.sum(jnp.square(x)) for x in jax.tree_util.tree_leaves(params)
    )
    return jnp.sqrt(squared_norm)


def parameter_group_l2_norm(params, module_names):
    """L2 norm for selected top-level Flax parameter modules."""
    param_tree = params["params"] if "params" in params else params
    selected = {name: param_tree[name] for name in module_names}
    return weight_l2_norm(selected)


def feature_metrics(features, cutoff=0.01):
    """Compute norm and Lyle et al. feature rank for (..., feature_dim)."""
    feature_matrix = features.reshape((-1, features.shape[-1]))
    n_samples = feature_matrix.shape[0]

    # Mean per-sample representation norm. This is independent of batch size.
    mean_feature_norm = jnp.mean(jnp.linalg.norm(feature_matrix, axis=-1))

    # Lyle et al. (2022): singular values of Phi / sqrt(N) above epsilon. (https://arxiv.org/pdf/2204.09560.pdf)
    singular_values = jnp.linalg.svd(feature_matrix, compute_uv=False)
    normalized_singular_values = singular_values / jnp.sqrt(
        jnp.asarray(n_samples, dtype=feature_matrix.dtype)
    )
    feature_rank = jnp.sum(normalized_singular_values > cutoff).astype(
        feature_matrix.dtype
    )

    return {
        "feature_norm": mean_feature_norm,
        "feature_rank": feature_rank,
        "lambda_1": normalized_singular_values[0],
        "lambda_N": normalized_singular_values[-1],
    }


def compute_penultimate_metrics(
    network,
    params,
    initial_hstate,
    network_inputs,
    actor_param_keys,
    critic_param_keys,
    cutoff=0.01,
):
    """Measure weights and actor/critic penultimate activations.

    The network must sow ``actor_penultimate`` and ``critic_penultimate`` into
    Flax's ``intermediates`` collection. Time and actor axes are combined into
    the observation/sample axis before computing the SVD.
    """
    _, intermediates = network.apply(
        params,
        initial_hstate,
        network_inputs,
        mutable=["intermediates"],
    )
    captured = intermediates["intermediates"]
    actor_features = captured["actor_penultimate"][0]
    critic_features = captured["critic_penultimate"][0]

    actor_metrics = feature_metrics(actor_features, cutoff=cutoff)
    critic_metrics = feature_metrics(critic_features, cutoff=cutoff)

    metrics = {
        "representation/weight_norm": weight_l2_norm(params),
        "representation/actor_weight_norm": parameter_group_l2_norm(
            params, actor_param_keys
        ),
        "representation/critic_weight_norm": parameter_group_l2_norm(
            params, critic_param_keys
        ),
    }
    metrics.update({
        f"representation/actor_{name}": value
        for name, value in actor_metrics.items()
    })
    metrics.update({
        f"representation/critic_{name}": value
        for name, value in critic_metrics.items()
    })
    return metrics


def empty_penultimate_metrics(dtype=jnp.float32):
    """Shape/dtype-compatible output for skipped logging steps."""
    names = (
        "representation/weight_norm",
        "representation/actor_weight_norm",
        "representation/critic_weight_norm",
        "representation/actor_feature_norm",
        "representation/actor_feature_rank",
        "representation/actor_lambda_1",
        "representation/actor_lambda_N",
        "representation/critic_feature_norm",
        "representation/critic_feature_rank",
        "representation/critic_lambda_1",
        "representation/critic_lambda_N",
    )
    return {name: jnp.asarray(jnp.nan, dtype=dtype) for name in names}
