"""Value-network stiffness on the first recurrent training minibatch.

For every actor-state in the first PPO/IDAAC minibatch, compute an
individual unclipped value-loss gradient and average cosine similarity
over all distinct state pairs.
"""

from typing import Sequence

import flax
import jax
import jax.numpy as jnp


STIFFNESS_METRIC_NAMES = (
    "value_off_diagonal",
    "value_same_layout",
    "value_different_layout",
    "value_layout_gap",
)


def advance_rollout_rng(rng: jax.Array, num_steps: int) -> jax.Array:
    """Predict the RNG carried out of a rollout without sampling tensors."""

    def advance(key, _):
        key, _ = jax.random.split(key)  # action sampling
        key, _ = jax.random.split(key)  # environment stepping
        return key, None

    rng, _ = jax.lax.scan(advance, rng, None, length=int(num_steps))
    return rng


def first_minibatch_actor_indices(
    permutation_rng: jax.Array,
    num_actors: int,
    num_minibatches: int,
) -> jax.Array:
    """Return exactly the actor trajectories assigned to minibatch zero."""

    if num_actors % num_minibatches != 0:
        raise ValueError(
            "NUM_ACTORS must be divisible by NUM_MINIBATCHES, got "
            f"{num_actors} and {num_minibatches}."
        )
    actors_per_minibatch = num_actors // num_minibatches
    permutation = jax.random.permutation(permutation_rng, num_actors)
    return permutation[:actors_per_minibatch]


def select_stiffness_batch(
    *,
    sampled_hstates,
    observations: jnp.ndarray,
    dones: jnp.ndarray,
    agent_positions: jnp.ndarray,
    targets: jnp.ndarray,
    layout_ids: jnp.ndarray,
    actor_indices: jax.Array,
):
    """Flatten every actor-state from the selected recurrent minibatch."""

    def select_and_flatten(values):
        selected = jnp.take(values, actor_indices, axis=1)
        return selected.reshape((-1,) + selected.shape[2:])

    def flatten_hstates(values):
        return values.reshape((-1,) + values.shape[2:])

    return (
        jax.tree.map(flatten_hstates, sampled_hstates),
        select_and_flatten(observations),
        select_and_flatten(dones),
        select_and_flatten(agent_positions),
        select_and_flatten(targets),
        select_and_flatten(layout_ids),
    )



def empty_paper_stiffness_metrics(dtype=jnp.float32):
    """Shape-compatible result for updates where stiffness is not measured."""

    return {
        name: jnp.asarray(jnp.nan, dtype=dtype)
        for name in STIFFNESS_METRIC_NAMES
    }


def _partition_value_params(params, value_param_keys: Sequence[str]):
    """Differentiate only root modules belonging to the value pathway."""

    mutable_params = flax.core.unfreeze(params)
    if "params" not in mutable_params:
        raise ValueError("Expected Flax variables with a top-level 'params'.")

    root_params = mutable_params["params"]
    selected_keys = set(value_param_keys)
    value_params = {
        key: value
        for key, value in root_params.items()
        if key in selected_keys
    }
    if not value_params:
        raise ValueError(
            "No value parameters matched STIFFNESS value_param_keys."
        )
    frozen_root_params = {
        key: value
        for key, value in root_params.items()
        if key not in selected_keys
    }
    other_collections = {
        key: value
        for key, value in mutable_params.items()
        if key != "params"
    }

    def merge(candidate_value_params):
        merged_root = dict(frozen_root_params)
        merged_root.update(candidate_value_params)
        return flax.core.freeze(
            {**other_collections, "params": merged_root}
        )

    return flax.core.freeze(value_params), merge


def compute_paper_stiffness(
    *,
    network,
    params,
    sampled_hstates,
    observations: jnp.ndarray,
    dones: jnp.ndarray,
    agent_positions: jnp.ndarray,
    targets: jnp.ndarray,
    layout_ids: jnp.ndarray,
    value_param_keys: Sequence[str],
    chunk_size: int,
    num_layouts: int = 5,
    epsilon: float = 1e-12,
):
    """Compute mean cosine similarity between distinct per-state value grads.

    The diagnostic loss is the paper's unclipped
    ``0.5 * (V_phi(s) - stop_gradient(return)) ** 2``. Recurrent carries are
    treated as part of the sampled state and are not differentiated through.
    """

    sample_size = int(observations.shape[0])
    if sample_size % int(chunk_size) != 0:
        raise ValueError("chunk_size must evenly divide the sampled batch.")

    value_params, merge_params = _partition_value_params(
        params, value_param_keys
    )

    def single_value_loss(
        candidate_value_params,
        sample_hstate,
        observation,
        done,
        agent_position,
        target,
    ):
        full_params = merge_params(candidate_value_params)
        batched_hstate = jax.tree.map(
            lambda value: jax.lax.stop_gradient(value)[None, :],
            sample_hstate,
        )
        network_input = (
            observation[None, None, ...],
            done[None, None, ...],
            agent_position[None, None, ...],
        )
        network_output = network.apply(
            full_params, batched_hstate, network_input
        )
        predicted_value = network_output[2].reshape(())
        residual = predicted_value - jax.lax.stop_gradient(target)
        return 0.5 * jnp.square(residual)

    individual_gradient = jax.grad(single_value_loss)
    chunked_hstates = jax.tree.map(
        lambda value: value.reshape(
            (sample_size // chunk_size, chunk_size) + value.shape[1:]
        ),
        sampled_hstates,
    )

    def chunk(values):
        return values.reshape(
            (sample_size // chunk_size, chunk_size) + values.shape[1:]
        )

    chunked_inputs = (
        chunked_hstates,
        chunk(observations),
        chunk(dones),
        chunk(agent_positions),
        chunk(targets),
        chunk(layout_ids),
    )
    normalized_gradient_sum_by_layout = jax.tree.map(
        lambda value: jnp.zeros(
            (num_layouts,) + value.shape, dtype=value.dtype
        ),
        value_params,
    )
    initial_state = (
        normalized_gradient_sum_by_layout,
        jnp.zeros((num_layouts,), dtype=jnp.float32),
    )

    def accumulate_chunk(state, batch):
        normalized_sum_by_layout, valid_count_by_layout = state
        (
            batch_hstates,
            batch_observations,
            batch_dones,
            batch_agent_positions,
            batch_targets,
            batch_layout_ids,
        ) = batch
        gradients = jax.vmap(
            individual_gradient,
            in_axes=(None, 0, 0, 0, 0, 0),
        )(
            value_params,
            batch_hstates,
            batch_observations,
            batch_dones,
            batch_agent_positions,
            batch_targets,
        )

        squared_norms = jnp.zeros((chunk_size,), dtype=jnp.float32)
        for gradient_leaf in jax.tree_util.tree_leaves(gradients):
            reduction_axes = tuple(range(1, gradient_leaf.ndim))
            squared_norms = squared_norms + jnp.sum(
                jnp.square(gradient_leaf), axis=reduction_axes
            )
        gradient_norms = jnp.sqrt(squared_norms)
        valid = gradient_norms > epsilon
        inverse_norms = jnp.where(
            valid, 1.0 / jnp.maximum(gradient_norms, epsilon), 0.0
        )

        layout_one_hot = jax.nn.one_hot(
            batch_layout_ids.astype(jnp.int32),
            num_layouts,
            dtype=jnp.float32,
        )
        valid_layout_weights = layout_one_hot * valid[:, None]

        def add_normalized_by_layout(total, gradient_leaf):
            broadcast_shape = (chunk_size,) + (1,) * (gradient_leaf.ndim - 1)
            normalized = gradient_leaf * inverse_norms.reshape(
                broadcast_shape
            )
            return total + jnp.tensordot(
                valid_layout_weights.T,
                normalized,
                axes=((1,), (0,)),
            )

        normalized_sum_by_layout = jax.tree.map(
            add_normalized_by_layout,
            normalized_sum_by_layout,
            gradients,
        )
        return (
            normalized_sum_by_layout,
            valid_count_by_layout + jnp.sum(
                valid_layout_weights, axis=0
            ),
        ), None

    (
        normalized_gradient_sum_by_layout,
        valid_sample_count_by_layout,
    ), _ = jax.lax.scan(accumulate_chunk, initial_state, chunked_inputs)

    squared_sum_norm_by_layout = sum(
        jnp.sum(
            jnp.square(leaf),
            axis=tuple(range(1, leaf.ndim)),
        )
        for leaf in jax.tree_util.tree_leaves(
            normalized_gradient_sum_by_layout
        )
    )
    normalized_gradient_sum = jax.tree.map(
        lambda leaf: jnp.sum(leaf, axis=0),
        normalized_gradient_sum_by_layout,
    )
    squared_sum_norm = sum(
        jnp.sum(jnp.square(leaf))
        for leaf in jax.tree_util.tree_leaves(normalized_gradient_sum)
    )
    valid_sample_count = jnp.sum(valid_sample_count_by_layout)
    # For normalized gradients u_i, ||sum_i u_i||^2 contains every ordered
    # pair cosine. Subtracting the valid count removes the i == j terms.
    off_diagonal_sum = squared_sum_norm - valid_sample_count
    off_diagonal_count = valid_sample_count * (valid_sample_count - 1.0)
    same_layout_sum = jnp.sum(
        squared_sum_norm_by_layout - valid_sample_count_by_layout
    )
    same_layout_count = jnp.sum(
        valid_sample_count_by_layout
        * (valid_sample_count_by_layout - 1.0)
    )
    # Removing the within-layout contribution from all distinct pairs leaves
    # exactly the pairs whose layout-family IDs differ.
    different_layout_sum = off_diagonal_sum - same_layout_sum
    different_layout_count = off_diagonal_count - same_layout_count

    def safe_mean(total, count):
        return jnp.where(
            count > 0.0,
            total / jnp.maximum(count, 1.0),
            jnp.asarray(jnp.nan, dtype=jnp.float32),
        )

    off_diagonal = safe_mean(off_diagonal_sum, off_diagonal_count)
    same_layout = safe_mean(same_layout_sum, same_layout_count)
    different_layout = safe_mean(
        different_layout_sum, different_layout_count
    )
    return {
        "value_off_diagonal": off_diagonal,
        "value_same_layout": same_layout,
        "value_different_layout": different_layout,
        "value_layout_gap": same_layout - different_layout,
    }
