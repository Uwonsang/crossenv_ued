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


def parameter_group_l2_norm(params, module_names):
    """All-leaf global L2 norm for selected Flax parameter modules."""
    param_tree = params["params"] if "params" in params else params
    selected = {name: param_tree[name] for name in module_names}
    return tree_global_l2_norm(selected)


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


def tree_group_leaf_count_weighted_rms_norm(tree, module_names):
    """Leaf-size-weighted RMS for selected Flax parameter modules."""
    module_tree = tree["params"] if "params" in tree else tree
    selected = {name: module_tree[name] for name in module_names}
    return tree_leaf_count_weighted_rms_norm(selected)


def compute_weight_metrics(params, actor_param_keys, critic_param_keys):
    """Measure SimBaV2-style parameter norms at one optimizer-update state."""
    return {
        "representation_weight/weight_norm": tree_global_l2_norm(params),
        "representation_weight/actor_weight_norm": parameter_group_l2_norm(
            params, actor_param_keys
        ),
        "representation_weight/critic_weight_norm": parameter_group_l2_norm(
            params, critic_param_keys
        ),
        "representation_weight/weighted_rms_norm": (
            tree_leaf_count_weighted_rms_norm(params)
        ),
        "representation_weight/actor_weighted_rms_norm": (
            tree_group_leaf_count_weighted_rms_norm(
                params, actor_param_keys
            )
        ),
        "representation_weight/critic_weighted_rms_norm": (
            tree_group_leaf_count_weighted_rms_norm(
                params, critic_param_keys
            )
        ),
    }


def feature_scale_metrics(feature_matrix, singular_values):
    """Compute feature magnitude and leading-spectrum scale metrics."""
    n_samples = feature_matrix.shape[0]
    normalized_singular_values = singular_values / jnp.sqrt(
        jnp.asarray(n_samples, dtype=feature_matrix.dtype)
    )

    return {
        "feature_norm": jnp.mean(
            jnp.linalg.norm(feature_matrix, axis=-1)
        ),
        "normalized_sigma_1": normalized_singular_values[0],
        "sigma_1_ratio": singular_values[0] / jnp.sum(singular_values),
    }


# source: https://github.com/DAVIAN-Robotics/SimbaV2/blob/86899c277cdc697b2b02d827243de1ea93f20a1d/scale_rl/agents/jax_utils/metrics.py#L135
def feature_metrics(features, cutoff=0.01):

    threshold = 1 - cutoff
    feature_matrix = features.reshape((-1, features.shape[-1]))

    # Roy & Vetterli (2007): exp(entropy) of the normalized singular-value
    # distribution. Unlike the threshold feature rank, this is invariant to a
    # uniform rescaling of all features.
    svals = jnp.linalg.svdvals(feature_matrix)

    sval_sum = jnp.sum(svals)
    sval_dist = svals / sval_sum
    # Replace 0 with 1. This is a safe trick to avoid log(0) = -inf
    # as Roy & Vetterli assume 0*log(0) = 0 = 1*log(1).
    sval_dist_fixed = jnp.where(sval_dist == 0, jnp.ones_like(sval_dist), sval_dist)
    effective_rank_vetterli = jnp.exp(-jnp.sum(sval_dist_fixed * jnp.log(sval_dist_fixed)))
        
    # Yang et al. (2020): smallest k explaining 1 - cutoff of the PCA
    # variance, i.e. the squared singular-value sum.
    sval_squares = svals**2
    sval_squares_sum = jnp.sum(sval_squares)
    cumsum_squares = jnp.cumsum(sval_squares)
    threshold_crossed = cumsum_squares >= (threshold * sval_squares_sum)
    approximate_rank_pca = (~threshold_crossed).sum() + 1

    # Kumar et al. (2020): smallest k whose leading singular values explain
    # 1 - cutoff of the nuclear norm. Return zero for an all-zero feature
    # matrix, for which no direction is active.
    cumsum_svals = jnp.cumsum(svals)
    threshold_crossed = cumsum_svals >= threshold * sval_sum
    srank_kumar = (~threshold_crossed).sum() + 1

    # Lyle et al. (2022): singular values of Phi / sqrt(N) above epsilon.
    n_obs = feature_matrix.shape[0]
    svals_of_normalized = svals / jnp.sqrt(n_obs)
    over_cutoff = svals_of_normalized > cutoff
    feature_rank = over_cutoff.sum()

    # Note that this determines the matrix rank same with (4), but some reasonable tau is chosen automatically based on the floating point precision of the input.
    jnp_ranks = jnp.linalg.matrix_rank(feature_matrix)

    metrics = {
        "feature_rank": feature_rank,
        "effective_rank_vetterli": effective_rank_vetterli,
        "srank_kumar": srank_kumar,
        "approximate_rank_pca": approximate_rank_pca,
        "matrix_rank": jnp_ranks,
    }

    metrics.update(feature_scale_metrics(feature_matrix, svals))

    return metrics


def first_epoch_first_minibatch_indices(
    update_rng,
    num_actors,
    num_minibatches,
):
    """Reproduce the actor indices used by the first PPO minibatch.

    ``_update_epoch`` splits its incoming RNG once and uses the second key for
    the actor-axis permutation. Calling this helper with that same incoming
    RNG therefore selects exactly the actors that the first epoch's first
    minibatch will consume, without advancing the training RNG.
    """
    if num_actors % num_minibatches != 0:
        raise ValueError(
            "NUM_ACTORS must be divisible by NUM_MINIBATCHES, got "
            f"{num_actors} and {num_minibatches}"
        )
    _, permutation_key = jax.random.split(update_rng)
    permutation = jax.random.permutation(permutation_key, num_actors)
    actors_per_minibatch = num_actors // num_minibatches
    return permutation[:actors_per_minibatch]


def compute_minibatch_penultimate_metrics(
    network,
    params,
    initial_hstate,
    network_inputs,
    cutoff=0.01,
):
    """Measure representation metrics on one recurrent PPO minibatch.

    The input keeps the complete rollout-time axis and a subset of the actor
    axis, together with the matching initial recurrent states. The network
    therefore reconstructs features with the same sequence and reset handling
    as PPO. Time and actor axes are flattened only after features are computed;
    actors from the same environment remain separate rows.
    """
    (_, _, _), intermediates = network.apply(
        params,
        initial_hstate,
        network_inputs,
        mutable=["intermediates"],
    )

    actor_features = intermediates["intermediates"]["actor_penultimate"][0]
    critic_features = intermediates["intermediates"]["critic_penultimate"][0]
    layer_norms = {
        name: jnp.mean(intermediates["intermediates"][f"feature_norm_{name}"][0])
        for name in INTERMEDIATE_FEATURE_NAMES
    }

    actor_metrics = feature_metrics(actor_features, cutoff=cutoff)
    critic_metrics = feature_metrics(critic_features, cutoff=cutoff)

    metrics = {}
    rank_names = {
        "feature_rank",
        "effective_rank_vetterli",
        "srank_kumar",
        "approximate_rank_pca",
        "matrix_rank",
    }
    for role, role_metrics in (("actor", actor_metrics), ("critic", critic_metrics)):
        for name, value in role_metrics.items():
            namespace = (
                "representation_rank"
                if name in rank_names
                else "representation_feature"
            )
            metrics[f"{namespace}/{role}_{name}"] = value

    def _sum_group(group):
        return sum(
            layer_norms[name]
            for name in INTERMEDIATE_FEATURE_GROUPS[group]
        )

    # Match SimbaV2's featnorm_total: sum the mean per-sample feature norm of
    # each layer. Actor/critic totals both include the shared feature path.
    shared_total = _sum_group("shared")
    actor_branch_total = _sum_group("actor")
    critic_branch_total = _sum_group("critic")
    metrics["representation_feature_total/actor_feature_norm"] = (shared_total + actor_branch_total)
    metrics["representation_feature_total/critic_feature_norm"] = (shared_total + critic_branch_total)

    return metrics


def empty_penultimate_metrics(dtype=jnp.float32):
    """Shape/dtype-compatible output for skipped logging steps."""
    names = (
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
