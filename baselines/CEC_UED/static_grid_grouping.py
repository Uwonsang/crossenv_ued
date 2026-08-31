"""Fixed-shape helpers for grouping minibatch samples by static grid."""

import jax
import jax.numpy as jnp


def unique_static_grid_signatures(
    signatures: jnp.ndarray,
    sample_mask: jnp.ndarray,
    max_groups: int,
):
    """Return fixed-size representatives for valid sample signatures.

    Representatives are lexicographically ordered. ``group_count`` is the
    uncapped number of unique valid signatures, while ``retained_count`` is
    capped at ``max_groups`` for JAX-static diagnostic buffers.
    """
    flat_signatures = signatures.reshape((-1, signatures.shape[-1]))
    flat_valid = sample_mask.reshape(-1).astype(jnp.bool_)
    sample_indices = jnp.arange(flat_signatures.shape[0], dtype=jnp.int32)
    invalid_key = jnp.logical_not(flat_valid).astype(jnp.uint32)
    operands = (invalid_key,) + tuple(
        flat_signatures[:, index]
        for index in range(flat_signatures.shape[1])
    ) + (sample_indices,)
    sorted_operands = jax.lax.sort(
        operands,
        dimension=0,
        is_stable=True,
        num_keys=flat_signatures.shape[1] + 1,
    )
    sorted_invalid = sorted_operands[0].astype(jnp.bool_)
    sorted_signatures = jnp.stack(sorted_operands[1:-1], axis=1)
    sorted_sample_indices = sorted_operands[-1]
    sorted_valid = jnp.logical_not(sorted_invalid)
    signature_changed = jnp.concatenate(
        (
            jnp.ones((1,), dtype=jnp.bool_),
            jnp.any(
                sorted_signatures[1:] != sorted_signatures[:-1], axis=1
            ),
        )
    )
    boundaries = jnp.logical_and(sorted_valid, signature_changed)
    sorted_group_ids = jnp.cumsum(boundaries.astype(jnp.int32)) - 1
    retained_boundary = jnp.logical_and(
        boundaries, sorted_group_ids < int(max_groups)
    )
    safe_group_ids = jnp.clip(sorted_group_ids, 0, int(max_groups) - 1)
    representatives = jnp.zeros(
        (int(max_groups), flat_signatures.shape[1]),
        dtype=flat_signatures.dtype,
    ).at[safe_group_ids].add(
        jnp.where(
            retained_boundary[:, None],
            sorted_signatures,
            jnp.zeros_like(sorted_signatures),
        ),
    )
    group_count = jnp.sum(boundaries.astype(jnp.int32))
    retained_count = jnp.minimum(group_count, int(max_groups))
    sorted_retained = jnp.logical_and(
        sorted_valid, sorted_group_ids < int(max_groups)
    )
    sample_group_ids = jnp.full(
        (flat_signatures.shape[0],), -1, dtype=jnp.int32
    ).at[sorted_sample_indices].set(
        jnp.where(sorted_retained, sorted_group_ids, -1)
    )
    return representatives, group_count, retained_count, sample_group_ids
