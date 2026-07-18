"""JAX/Flax representation diagnostics."""

import jax
import jax.numpy as jnp


def weight_l2_norm(params):
    """Global L2 norm of matrix/tensor weights, excluding 1-D parameters.

    This matches the common layer-weight convention used by the reference
    PyTorch metrics: Dense/Conv/RNN kernels are included, while biases and
    LayerNorm scale/shift parameters are excluded.
    """
    squared_norm = sum(
        (
            jnp.sum(jnp.square(x))
            for x in jax.tree_util.tree_leaves(params)
            if x.ndim > 1
        ),
        jnp.asarray(0.0),
    )
    return jnp.sqrt(squared_norm)


def parameter_group_l2_norm(params, module_names):
    """Matrix/tensor-weight L2 norm for selected Flax parameter modules."""
    param_tree = params["params"] if "params" in params else params
    selected = {name: param_tree[name] for name in module_names}
    return weight_l2_norm(selected)


def feature_metrics(features, cutoff=0.01):
    """Compute norm and singular-spectrum ranks for (..., feature_dim).

    ``effective_rank_vetterli`` is the entropy-based continuous effective
    rank. ``srank_kumar`` is the number of leading singular values needed to
    explain ``1 - cutoff`` of their sum. ``approximate_rank_pca`` uses the
    same threshold on the squared singular values, while ``matrix_rank`` uses
    the standard NumPy/PyTorch numerical-rank tolerance.
    """
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

    # Roy & Vetterli (2007): exp(entropy) of the normalized singular-value
    # distribution. Unlike the threshold feature rank, this is invariant to a
    # uniform rescaling of all features.
    singular_value_sum = jnp.sum(singular_values)
    nonzero_spectrum = singular_value_sum > 0
    singular_value_distribution = jnp.where(
        nonzero_spectrum,
        singular_values / jnp.where(
            nonzero_spectrum,
            singular_value_sum,
            jnp.ones_like(singular_value_sum),
        ),
        jnp.zeros_like(singular_values),
    )
    entropy_terms = jnp.where(
        singular_value_distribution > 0,
        singular_value_distribution
        * jnp.log(jnp.where(
            singular_value_distribution > 0,
            singular_value_distribution,
            jnp.ones_like(singular_value_distribution),
        )),
        jnp.zeros_like(singular_value_distribution),
    )
    effective_rank_vetterli = jnp.where(
        nonzero_spectrum,
        jnp.exp(-jnp.sum(entropy_terms)),
        jnp.asarray(0.0, dtype=feature_matrix.dtype),
    )
    lambda_1_ratio = jnp.where(
        nonzero_spectrum,
        singular_value_distribution[0],
        jnp.asarray(0.0, dtype=feature_matrix.dtype),
    )

    # Kumar et al. (2021): smallest k whose leading singular values explain
    # 1 - cutoff of the nuclear norm. Return zero for an all-zero feature
    # matrix, for which no direction is active.
    cumulative_singular_values = jnp.cumsum(singular_values)
    srank_kumar = jnp.where(
        nonzero_spectrum,
        jnp.sum(
            cumulative_singular_values
            < (1.0 - cutoff) * singular_value_sum
        )
        + 1,
        0,
    ).astype(feature_matrix.dtype)

    # Yang et al. (2020): smallest k explaining 1 - cutoff of the PCA
    # variance, i.e. the squared singular-value sum.
    squared_singular_values = jnp.square(singular_values)
    squared_singular_value_sum = jnp.sum(squared_singular_values)
    nonzero_variance = squared_singular_value_sum > 0
    approximate_rank_pca = jnp.where(
        nonzero_variance,
        jnp.sum(
            jnp.cumsum(squared_singular_values)
            < (1.0 - cutoff) * squared_singular_value_sum
        )
        + 1,
        0,
    ).astype(feature_matrix.dtype)

    # NumPy/PyTorch default matrix-rank threshold. Reuse the SVD above rather
    # than computing it again through jnp.linalg.matrix_rank.
    matrix_rank_tolerance = (
        max(feature_matrix.shape)
        * jnp.finfo(feature_matrix.dtype).eps
        * singular_values[0]
    )
    matrix_rank = jnp.sum(
        singular_values > matrix_rank_tolerance
    ).astype(feature_matrix.dtype)

    metrics = {
        "feature_norm": mean_feature_norm,
        "feature_rank": feature_rank,
        "effective_rank_vetterli": effective_rank_vetterli,
        "srank_kumar": srank_kumar,
        "approximate_rank_pca": approximate_rank_pca,
        "matrix_rank": matrix_rank,
        "lambda_1": normalized_singular_values[0],
        "lambda_1_ratio": lambda_1_ratio,
        "lambda_N": normalized_singular_values[-1],
    }
    return metrics


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
        "representation_weight/weight_norm": weight_l2_norm(params),
        "representation_weight/actor_weight_norm": parameter_group_l2_norm(
            params, actor_param_keys
        ),
        "representation_weight/critic_weight_norm": parameter_group_l2_norm(
            params, critic_param_keys
        ),
    }
    rank_names = {
        "feature_rank",
        "effective_rank_vetterli",
        "srank_kumar",
        "approximate_rank_pca",
        "matrix_rank",
    }
    for role, role_metrics in (
        ("actor", actor_metrics),
        ("critic", critic_metrics),
    ):
        for name, value in role_metrics.items():
            namespace = (
                "representation_rank"
                if name in rank_names
                else "representation_feature"
            )
            metrics[f"{namespace}/{role}_{name}"] = value
    return metrics


def empty_penultimate_metrics(dtype=jnp.float32):
    """Shape/dtype-compatible output for skipped logging steps."""
    names = (
        "representation_weight/weight_norm",
        "representation_weight/actor_weight_norm",
        "representation_weight/critic_weight_norm",
        "representation_feature/actor_feature_norm",
        "representation_feature/actor_lambda_1",
        "representation_feature/actor_lambda_1_ratio",
        "representation_feature/actor_lambda_N",
        "representation_feature/critic_feature_norm",
        "representation_feature/critic_lambda_1",
        "representation_feature/critic_lambda_1_ratio",
        "representation_feature/critic_lambda_N",
        "representation_rank/actor_feature_rank",
        "representation_rank/actor_effective_rank_vetterli",
        "representation_rank/actor_srank_kumar",
        "representation_rank/actor_approximate_rank_pca",
        "representation_rank/actor_matrix_rank",
        "representation_rank/critic_feature_rank",
        "representation_rank/critic_effective_rank_vetterli",
        "representation_rank/critic_srank_kumar",
        "representation_rank/critic_approximate_rank_pca",
        "representation_rank/critic_matrix_rank",
    )
    return {name: jnp.asarray(jnp.nan, dtype=dtype) for name in names}
