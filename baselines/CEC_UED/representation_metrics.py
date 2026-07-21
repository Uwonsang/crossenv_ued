"""JAX/Flax representation diagnostics."""

import jax
import jax.numpy as jnp


INTERMEDIATE_FEATURE_GROUPS = {
    "shared": (
        "shared_conv_0",
        "shared_conv_1",
        "shared_dense_0",
        "shared_dense_1",
        "shared_recurrent",
    ),
    "actor": (
        "actor_hidden_0",
        "actor_hidden_1",
        "actor_hidden_2",
        "actor_hidden_3",
    ),
    "critic": (
        "critic_hidden_0",
        "critic_hidden_1",
        "critic_hidden_2",
        "critic_hidden_3",
    ),
}

INTERMEDIATE_FEATURE_NAMES = tuple(
    name
    for names in INTERMEDIATE_FEATURE_GROUPS.values()
    for name in names
)


def tree_global_l2_norm(tree):
    """Global L2 norm over every array leaf in a pytree."""
    squared_norm = sum(
        (
            jnp.sum(jnp.square(x))
            for x in jax.tree_util.tree_leaves(tree)
        ),
        jnp.asarray(0.0),
    )
    return jnp.sqrt(squared_norm)


def weight_l2_norm(params):
    """Global L2 norm over every parameter leaf.

    Dense/Conv/RNN kernels, biases, and any 1-D learned scale/shift parameters
    are all included. This is equivalent to flattening and concatenating the
    complete parameter pytree before taking one L2 norm.
    """
    return tree_global_l2_norm(params)


def parameter_group_l2_norm(params, module_names):
    """All-leaf global L2 norm for selected Flax parameter modules."""
    param_tree = params["params"] if "params" in params else params
    selected = {name: param_tree[name] for name in module_names}
    return weight_l2_norm(selected)


def tree_leaf_count_weighted_rms_norm(tree):
    """Leaf-size-weighted RMS of leaf L2 norms for an array pytree."""
    leaves = jax.tree_util.tree_leaves(tree)
    dtype = leaves[0].dtype
    numerator = sum(
        (
            jnp.asarray(leaf.size, dtype=dtype)
            * jnp.sum(jnp.square(leaf))
            for leaf in leaves
        ),
        jnp.asarray(0.0, dtype=dtype),
    )
    denominator = sum(leaf.size for leaf in leaves)
    return jnp.sqrt(
        numerator / jnp.asarray(denominator, dtype=dtype)
    )


def parameter_count_weighted_rms(params):
    """SimbaV2-style parameter-count weighted RMS of leaf L2 norms.

    For leaves ``theta_i`` with ``n_i`` elements, this computes
    ``sqrt(sum_i n_i * ||theta_i||_2^2 / sum_i n_i)``. All leaves, including
    biases and learned 1-D scale/shift parameters, participate.
    """
    return tree_leaf_count_weighted_rms_norm(params)


def tree_group_leaf_count_weighted_rms_norm(tree, module_names):
    """Leaf-size-weighted RMS for selected Flax parameter modules."""
    module_tree = tree["params"] if "params" in tree else tree
    selected = {name: module_tree[name] for name in module_names}
    return tree_leaf_count_weighted_rms_norm(selected)


def parameter_group_count_weighted_rms(params, module_names):
    """Weighted RMS for all leaves in selected Flax parameter modules."""
    return tree_group_leaf_count_weighted_rms_norm(params, module_names)


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
    sigma_1_ratio = jnp.where(
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
        "normalized_sigma_1": normalized_singular_values[0],
        "sigma_1_ratio": sigma_1_ratio,
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
    """Measure weights, penultimate spectra, and layerwise feature norms.

    The network must sow ``actor_penultimate`` and ``critic_penultimate`` into
    Flax's ``intermediates`` collection. It also sows scalar mean per-sample
    L2 norms for the shared, actor, and critic hidden layers. Time and actor
    axes are combined into the observation/sample axis before computing the
    penultimate SVD.
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
        "representation_weight/weighted_rms_norm": (
            parameter_count_weighted_rms(params)
        ),
        "representation_weight/actor_weighted_rms_norm": (
            parameter_group_count_weighted_rms(params, actor_param_keys)
        ),
        "representation_weight/critic_weighted_rms_norm": (
            parameter_group_count_weighted_rms(params, critic_param_keys)
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

    # Match SimbaV2's layer-level definition internally: each value is the
    # batch mean of per-sample L2 feature norms. Only actor/critic path totals
    # are exported, so the individual layer values do not clutter W&B.
    layer_norms = {}
    for name in INTERMEDIATE_FEATURE_NAMES:
        captured_name = f"feature_norm_{name}"
        layer_norms[name] = (
            captured[captured_name][0]
            if captured_name in captured
            else jnp.asarray(jnp.nan, dtype=actor_features.dtype)
        )

    def _sum_present(group):
        present_values = [
            layer_norms[name]
            for name in INTERMEDIATE_FEATURE_GROUPS[group]
            if f"feature_norm_{name}" in captured
        ]
        return sum(
            present_values,
            jnp.asarray(0.0, dtype=actor_features.dtype),
        )

    # Totals are simple sums of layer mean norms, as in SimbaV2's
    # featnorm_total. Actor/critic totals include the shared feature path.
    shared_total = _sum_present("shared")
    actor_branch_total = _sum_present("actor")
    critic_branch_total = _sum_present("critic")
    metrics["representation_feature_total/actor_feature_norm"] = (
        shared_total + actor_branch_total
    )
    metrics["representation_feature_total/critic_feature_norm"] = (
        shared_total + critic_branch_total
    )
    return metrics


def empty_penultimate_metrics(dtype=jnp.float32):
    """Shape/dtype-compatible output for skipped logging steps."""
    names = (
        "representation_weight/weight_norm",
        "representation_weight/actor_weight_norm",
        "representation_weight/critic_weight_norm",
        "representation_weight/weighted_rms_norm",
        "representation_weight/actor_weighted_rms_norm",
        "representation_weight/critic_weighted_rms_norm",
        "representation_feature/actor_feature_norm",
        "representation_feature/actor_normalized_sigma_1",
        "representation_feature/actor_sigma_1_ratio",
        "representation_feature/critic_feature_norm",
        "representation_feature/critic_normalized_sigma_1",
        "representation_feature/critic_sigma_1_ratio",
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
    names = names + (
        "representation_feature_total/actor_feature_norm",
        "representation_feature_total/critic_feature_norm",
    )
    return {name: jnp.asarray(jnp.nan, dtype=dtype) for name in names}
