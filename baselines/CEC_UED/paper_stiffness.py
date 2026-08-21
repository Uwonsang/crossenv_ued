"""Paper-style value-network stiffness for recurrent CEC agents.

This follows Moon et al. (2022), Section 3.1: compute an individual
unclipped value-loss gradient for every state in a fixed-size diagnostic
mini-batch, then average cosine similarity over all state pairs.
"""

from dataclasses import dataclass
from typing import Sequence

import flax
import jax
import jax.numpy as jnp
import numpy as np


STIFFNESS_METRIC_NAMES = ("value_all_pairs",)


@dataclass(frozen=True)
class PaperStiffnessPlan:
    """Static, approximately uniform sample allocation over a rollout."""

    actor_indices: jnp.ndarray
    valid_slot_indices: jnp.ndarray
    sample_size: int
    chunk_size: int
    num_steps: int
    num_actors: int


def build_paper_stiffness_plan(
    *,
    num_steps: int,
    num_actors: int,
    sample_size: int = 2**14,
    chunk_size: int = 16,
) -> PaperStiffnessPlan:
    """Allocate a fixed global sample without fixing samples per environment.

    Samples are spread as evenly as possible across rollout timesteps and
    actors. A cyclic actor offset can be applied at runtime so repeated
    measurements do not always use precisely the same actor trajectories.
    """

    total_samples = int(num_steps) * int(num_actors)
    if sample_size <= 1:
        raise ValueError("STIFFNESS.SAMPLE_SIZE must be greater than one.")
    if sample_size > total_samples:
        raise ValueError(
            "Paper-style stiffness needs at least SAMPLE_SIZE actor-states "
            f"in one rollout, but NUM_STEPS * NUM_ACTORS is {total_samples} "
            f"and SAMPLE_SIZE is {sample_size}. Increase NUM_STEPS/NUM_ENVS "
            "or reduce STIFFNESS.SAMPLE_SIZE explicitly."
        )
    if chunk_size <= 0 or sample_size % chunk_size != 0:
        raise ValueError(
            "STIFFNESS.CHUNK_SIZE must be positive and evenly divide "
            "STIFFNESS.SAMPLE_SIZE."
        )

    base_count, remainder = divmod(sample_size, num_steps)
    counts = np.full((num_steps,), base_count, dtype=np.int32)
    counts[:remainder] += 1
    max_count = int(counts.max())
    if max_count > num_actors:
        raise ValueError(
            "The requested stiffness sample requires more than one copy of "
            "an actor at a timestep. This indicates SAMPLE_SIZE exceeds the "
            "rollout population."
        )

    actor_indices = np.zeros((num_steps, max_count), dtype=np.int32)
    valid_mask = np.zeros((num_steps, max_count), dtype=bool)
    for time_index, count in enumerate(counts):
        if count == 0:
            continue
        # Midpoints of equal-width actor bins give deterministic uniform
        # coverage without constructing a full rollout-sized permutation.
        selected = np.floor(
            (np.arange(count, dtype=np.float64) + 0.5)
            * num_actors
            / count
        ).astype(np.int32)
        actor_indices[time_index, :count] = selected
        valid_mask[time_index, :count] = True

    valid_slot_indices = np.flatnonzero(valid_mask.reshape(-1)).astype(
        np.int32
    )
    if valid_slot_indices.size != sample_size:
        raise AssertionError("Internal stiffness sample-plan size mismatch.")

    return PaperStiffnessPlan(
        actor_indices=jnp.asarray(actor_indices),
        valid_slot_indices=jnp.asarray(valid_slot_indices),
        sample_size=int(sample_size),
        chunk_size=int(chunk_size),
        num_steps=int(num_steps),
        num_actors=int(num_actors),
    )


def sampled_actor_indices(
    plan: PaperStiffnessPlan,
    time_index: jnp.ndarray,
    update_step: jnp.ndarray,
) -> jnp.ndarray:
    """Actor indices to snapshot at one timestep of one training update."""

    base_indices = plan.actor_indices[time_index]
    # Rotating the actor IDs preserves the fixed allocation and adds coverage
    # across measurements without introducing a large random permutation.
    return (base_indices + update_step) % plan.num_actors


def select_stiffness_batch(
    *,
    plan: PaperStiffnessPlan,
    sampled_hstates,
    observations: jnp.ndarray,
    dones: jnp.ndarray,
    agent_positions: jnp.ndarray,
    targets: jnp.ndarray,
    update_step: jnp.ndarray,
):
    """Gather the fixed-size actor-state batch and its saved recurrent carry."""

    time_offsets = (
        jnp.arange(plan.num_steps, dtype=jnp.int32)[:, None]
        * plan.num_actors
    )
    rotated_actor_indices = (
        plan.actor_indices + update_step
    ) % plan.num_actors
    transition_indices = (time_offsets + rotated_actor_indices).reshape(-1)
    transition_indices = transition_indices[plan.valid_slot_indices]

    def gather_rollout(values):
        flattened = values.reshape(
            (plan.num_steps * plan.num_actors,) + values.shape[2:]
        )
        return jnp.take(flattened, transition_indices, axis=0)

    def gather_hstates(values):
        flattened = values.reshape(
            (-1,) + values.shape[2:]
        )
        return jnp.take(flattened, plan.valid_slot_indices, axis=0)

    return (
        jax.tree.map(gather_hstates, sampled_hstates),
        gather_rollout(observations),
        gather_rollout(dones),
        gather_rollout(agent_positions),
        gather_rollout(targets),
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
    value_param_keys: Sequence[str],
    chunk_size: int,
    epsilon: float = 1e-12,
):
    """Compute fixed-batch mean cosine similarity of per-state value grads.

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
    )
    normalized_gradient_sum = jax.tree.map(jnp.zeros_like, value_params)
    initial_state = (
        normalized_gradient_sum,
        jnp.asarray(0.0, dtype=jnp.float32),
    )

    def accumulate_chunk(state, batch):
        normalized_sum, valid_count = state
        (
            batch_hstates,
            batch_observations,
            batch_dones,
            batch_agent_positions,
            batch_targets,
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
            add_normalized, normalized_sum, gradients
        )
        return (
            normalized_sum,
            valid_count + jnp.sum(valid.astype(jnp.float32)),
        ), None

    (
        normalized_gradient_sum,
        valid_sample_count,
    ), _ = jax.lax.scan(accumulate_chunk, initial_state, chunked_inputs)

    squared_sum_norm = sum(
        jnp.sum(jnp.square(leaf))
        for leaf in jax.tree_util.tree_leaves(normalized_gradient_sum)
    )
    all_pairs = squared_sum_norm / jnp.maximum(
        jnp.square(valid_sample_count), 1.0
    )
    return {"value_all_pairs": all_pairs}
