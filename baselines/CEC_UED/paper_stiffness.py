"""Value-network stiffness on the first recurrent training minibatch.

For every actor-state in the first PPO/IDAAC minibatch, compute an
individual unclipped value-loss gradient and average cosine similarity
over all distinct state pairs.
"""

from typing import Sequence

import flax
import jax
import jax.numpy as jnp

from static_grid_grouping import unique_static_grid_signatures


STIFFNESS_METRIC_NAMES = (
    "value_off_diagonal",
    "value_same_layout_per_static_grid",
    "value_different_layout_per_static_grid",
)


def encode_static_grid_signature(
    wall_map: jnp.ndarray,
    maze_object_grid: jnp.ndarray,
    goal_positions: jnp.ndarray,
    pot_positions: jnp.ndarray,
) -> jnp.ndarray:
    """Losslessly encode the static 9x9 layout into seven uint32 values.

    Cell codes distinguish floor, wall/counter, onion pile, plate pile,
    goal, and pot. Twelve base-6 cells fit in one uint32, so the encoding
    has no hash collisions.
    """

    codes = wall_map.astype(jnp.uint32)
    codes = jnp.where(maze_object_grid == 4, 2, codes)  # onion pile
    codes = jnp.where(maze_object_grid == 6, 3, codes)  # plate pile

    goal_mask = jnp.zeros_like(wall_map, dtype=jnp.bool_).at[
        goal_positions[:, 1], goal_positions[:, 0]
    ].set(True)
    pot_mask = jnp.zeros_like(wall_map, dtype=jnp.bool_).at[
        pot_positions[:, 1], pot_positions[:, 0]
    ].set(True)
    codes = jnp.where(goal_mask, 4, codes)
    codes = jnp.where(pot_mask, 5, codes)

    flat_codes = jnp.pad(codes.reshape(-1), (0, 3)).reshape(7, 12)
    base6_weights = jnp.asarray(
        [6**power for power in range(12)], dtype=jnp.uint32
    )
    return jnp.sum(
        flat_codes * base6_weights[None, :],
        axis=1,
        dtype=jnp.uint32,
    )


def count_unique_static_signatures(signatures: jnp.ndarray) -> jnp.ndarray:
    """Count distinct static layouts in selected actor-state samples."""

    signatures = signatures.reshape((-1, signatures.shape[-1]))
    keys = tuple(signatures[:, index] for index in range(signatures.shape[-1]))
    sorted_keys = jax.lax.sort(
        keys,
        dimension=0,
        is_stable=True,
        num_keys=len(keys),
    )
    adjacent_difference = jnp.zeros(
        (signatures.shape[0] - 1,), dtype=jnp.bool_
    )
    for key in sorted_keys:
        adjacent_difference = jnp.logical_or(
            adjacent_difference, key[1:] != key[:-1]
        )
    return 1 + jnp.sum(adjacent_difference, dtype=jnp.int32)


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
    sample_static_signatures: jnp.ndarray,
    sample_layout_ids: jnp.ndarray,
    sample_mask: jnp.ndarray,
    max_static_grids: int,
    value_param_keys: Sequence[str],
    chunk_size: int,
    num_layouts: int = 5,
    sketch_size: int = 512,
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
    if int(sketch_size) <= 0:
        raise ValueError("sketch_size must be positive.")
    (
        _,
        _,
        retained_static_grid_count,
        sample_static_grid_ids,
    ) = (
        unique_static_grid_signatures(
            sample_static_signatures,
            sample_mask,
            max_groups=max_static_grids,
        )
    )

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
        chunk(sample_static_grid_ids),
        chunk(sample_layout_ids),
        chunk(sample_mask),
    )
    normalized_gradient_sum = jax.tree.map(jnp.zeros_like, value_params)
    initial_state = (
        normalized_gradient_sum,
        jnp.asarray(0.0, dtype=jnp.float32),
        jnp.zeros(
            (int(max_static_grids), int(sketch_size)), dtype=jnp.float32
        ),
        jnp.zeros((int(max_static_grids),), dtype=jnp.float32),
        jnp.zeros(
            (int(max_static_grids), int(num_layouts)), dtype=jnp.float32
        ),
    )

    def accumulate_chunk(state, batch):
        (
            normalized_sum,
            valid_count,
            raw_gradient_sketch_by_static_grid,
            sample_count_by_static_grid,
            layout_count_by_static_grid,
        ) = state
        (
            batch_hstates,
            batch_observations,
            batch_dones,
            batch_agent_positions,
            batch_targets,
            batch_static_grid_ids,
            batch_layout_ids,
            batch_sample_mask,
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

        def add_normalized(total, gradient_leaf):
            broadcast_shape = (chunk_size,) + (1,) * (gradient_leaf.ndim - 1)
            normalized = gradient_leaf * inverse_norms.reshape(
                broadcast_shape
            )
            return total + jnp.sum(normalized, axis=0)

        normalized_sum = jax.tree.map(
            add_normalized,
            normalized_sum,
            gradients,
        )
        raw_gradient_sketches = jnp.zeros(
            (int(chunk_size), int(sketch_size)), dtype=jnp.float32
        )
        parameter_offset = 0
        for gradient_leaf in jax.tree_util.tree_leaves(gradients):
            flat_gradient = gradient_leaf.reshape((int(chunk_size), -1))
            parameter_indices = (
                jnp.arange(flat_gradient.shape[1], dtype=jnp.uint32)
                + jnp.asarray(parameter_offset, dtype=jnp.uint32)
            )
            mixed_indices = parameter_indices + jnp.uint32(0x9E3779B9)
            mixed_indices = mixed_indices ^ (
                mixed_indices >> jnp.uint32(16)
            )
            mixed_indices = mixed_indices * jnp.uint32(0x7FEB352D)
            mixed_indices = mixed_indices ^ (
                mixed_indices >> jnp.uint32(15)
            )
            mixed_indices = mixed_indices * jnp.uint32(0x846CA68B)
            mixed_indices = mixed_indices ^ (
                mixed_indices >> jnp.uint32(16)
            )
            sketch_bins = (
                mixed_indices % jnp.uint32(int(sketch_size))
            ).astype(jnp.int32)
            sign_hash = mixed_indices ^ jnp.uint32(0xA5A5A5A5)
            sign_hash = sign_hash * jnp.uint32(0x27D4EB2D)
            sign_hash = sign_hash ^ (sign_hash >> jnp.uint32(15))
            sketch_signs = jnp.where(
                (sign_hash & jnp.uint32(1)) == 0, 1.0, -1.0
            ).astype(jnp.float32)

            def add_to_sketch(row):
                return jnp.zeros(
                    (int(sketch_size),), dtype=jnp.float32
                ).at[sketch_bins].add(row * sketch_signs)

            raw_gradient_sketches += jax.vmap(add_to_sketch)(flat_gradient)
            parameter_offset += int(flat_gradient.shape[1])

        retained_sample = jnp.logical_and(
            batch_static_grid_ids >= 0,
            jnp.logical_and(valid, batch_sample_mask.astype(jnp.bool_)),
        )
        safe_group_ids = jnp.clip(
            batch_static_grid_ids, 0, int(max_static_grids) - 1
        )
        sample_weights = retained_sample.astype(jnp.float32)
        raw_gradient_sketch_by_static_grid = (
            raw_gradient_sketch_by_static_grid.at[safe_group_ids].add(
                raw_gradient_sketches * sample_weights[:, None]
            )
        )
        sample_count_by_static_grid = sample_count_by_static_grid.at[
            safe_group_ids
        ].add(sample_weights)
        layout_count_by_static_grid = layout_count_by_static_grid.at[
            safe_group_ids
        ].add(jax.nn.one_hot(
            batch_layout_ids.astype(jnp.int32),
            int(num_layouts),
            dtype=jnp.float32,
        ) * sample_weights[:, None])
        return (
            normalized_sum,
            valid_count + jnp.sum(valid.astype(jnp.float32)),
            raw_gradient_sketch_by_static_grid,
            sample_count_by_static_grid,
            layout_count_by_static_grid,
        ), None

    (
        normalized_gradient_sum,
        valid_sample_count,
        raw_gradient_sketch_by_static_grid,
        sample_count_by_static_grid,
        layout_count_by_static_grid,
    ), _ = jax.lax.scan(accumulate_chunk, initial_state, chunked_inputs)

    squared_sum_norm = sum(
        jnp.sum(jnp.square(leaf))
        for leaf in jax.tree_util.tree_leaves(normalized_gradient_sum)
    )
    # For normalized gradients u_i, ||sum_i u_i||^2 contains every ordered
    # pair cosine. Subtracting the valid count removes the i == j terms.
    off_diagonal_sum = squared_sum_norm - valid_sample_count
    off_diagonal_count = valid_sample_count * (valid_sample_count - 1.0)
    def safe_mean(total, count):
        return jnp.where(
            count > 0.0,
            total / jnp.maximum(count, 1.0),
            jnp.asarray(jnp.nan, dtype=jnp.float32),
        )

    off_diagonal = safe_mean(off_diagonal_sum, off_diagonal_count)

    static_grid_layout_ids = jnp.argmax(
        layout_count_by_static_grid, axis=1
    ).astype(jnp.int32)
    static_grid_indices = jnp.arange(int(max_static_grids), dtype=jnp.int32)
    valid_group_index = static_grid_indices < retained_static_grid_count
    static_grid_sketch_norms = jnp.linalg.norm(
        raw_gradient_sketch_by_static_grid, axis=1
    )
    valid_static_grids = jnp.logical_and(
        valid_group_index,
        jnp.logical_and(
            sample_count_by_static_grid > 0.0,
            static_grid_sketch_norms > epsilon,
        ),
    )
    normalized_static_grid_sketches = jnp.where(
        valid_static_grids[:, None],
        raw_gradient_sketch_by_static_grid
        / jnp.maximum(static_grid_sketch_norms[:, None], epsilon),
        jnp.zeros_like(raw_gradient_sketch_by_static_grid),
    )
    static_grid_layout_weights = jax.nn.one_hot(
        static_grid_layout_ids.astype(jnp.int32),
        num_layouts,
        dtype=jnp.float32,
    ) * valid_static_grids.astype(jnp.float32)[:, None]
    normalized_static_grid_sum_by_layout = (
        static_grid_layout_weights.T @ normalized_static_grid_sketches
    )
    static_grid_count_by_layout = jnp.sum(
        static_grid_layout_weights, axis=0
    )
    static_grid_squared_sum_by_layout = jnp.sum(
        jnp.square(normalized_static_grid_sum_by_layout), axis=1
    )
    static_grid_count = jnp.sum(static_grid_count_by_layout)
    static_grid_same_layout_sum = jnp.sum(
        static_grid_squared_sum_by_layout - static_grid_count_by_layout
    )
    static_grid_same_layout_count = jnp.sum(
        static_grid_count_by_layout * (static_grid_count_by_layout - 1.0)
    )
    static_grid_total_sum = (
        jnp.sum(jnp.square(jnp.sum(
            normalized_static_grid_sum_by_layout, axis=0
        )))
        - static_grid_count
    )
    static_grid_total_count = static_grid_count * (static_grid_count - 1.0)
    static_grid_different_layout_sum = (
        static_grid_total_sum - static_grid_same_layout_sum
    )
    static_grid_different_layout_count = (
        static_grid_total_count - static_grid_same_layout_count
    )
    return {
        "value_off_diagonal": off_diagonal,
        "value_same_layout_per_static_grid": safe_mean(
            static_grid_same_layout_sum, static_grid_same_layout_count
        ),
        "value_different_layout_per_static_grid": safe_mean(
            static_grid_different_layout_sum,
            static_grid_different_layout_count,
        ),
    }
