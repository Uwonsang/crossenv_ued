"""Shared gradient-conflict diagnostics for CEC IPPO variants."""

import jax
import jax.numpy as jnp


def compute_projected_gradient_cosine_matrices(
    *,
    network,
    original_params,
    initial_hstate,
    traj_batch,
    advantages,
    value_targets,
    config,
    num_agents,
    value_trunk_keys,
    actor_trunk_keys,
):
    """Compute memory-bounded per-environment gradient cosine heatmaps.

    At each configured time, one sample is one environment's joint transition:
    the two agents' losses are averaged before differentiating. Full gradients
    are computed one environment at a time and immediately reduced to a fixed
    signed feature-hash sketch, so no ``NUM_ENVS x NUM_PARAMS`` tensor is kept.

    Returns:
        Array with shape ``(2, num_times, NUM_ENVS, NUM_ENVS)``. Axis 0 is
        ``(actor, value)``. The matrices are projected cosine/Gram matrices,
        not statistically centered covariance matrices.
    """
    timesteps = tuple(
        int(t) for t in config["GRADIENT_COVARIANCE_TIMESTEPS"]
    )
    sketch_dim = int(config["GRADIENT_COVARIANCE_SKETCH_DIM"])
    sketch_seed = int(config["GRADIENT_COVARIANCE_SKETCH_SEED"])
    num_envs = int(config["NUM_ENVS"])
    num_actors = num_envs * int(num_agents)
    if int(traj_batch.obs.shape[1]) != num_actors:
        raise ValueError(
            "trajectory actor axis does not match NUM_ENVS * num_agents"
        )

    value_keys = tuple(value_trunk_keys)
    actor_keys = tuple(actor_trunk_keys)

    def _select_params(params, keys):
        return {key: params["params"][key] for key in keys}

    value_params = _select_params(original_params, value_keys)
    actor_params = _select_params(original_params, actor_keys)
    other_value_params = {
        key: value
        for key, value in original_params["params"].items()
        if key not in value_keys
    }
    other_actor_params = {
        key: value
        for key, value in original_params["params"].items()
        if key not in actor_keys
    }

    def _rebuild_value_params(selected):
        merged = dict(other_value_params)
        merged.update(selected)
        return {"params": merged}

    def _rebuild_actor_params(selected):
        merged = dict(other_actor_params)
        merged.update(selected)
        return {"params": merged}

    # Match the existing actor diagnostic's common advantage normalization.
    diagnostic_steps = int(config["DIAGNOSTIC_WINDOW_STEPS"])
    diagnostic_advantages = advantages[:diagnostic_steps]
    advantage_mean = jnp.mean(diagnostic_advantages)
    advantage_std = jnp.sqrt(
        jnp.mean((diagnostic_advantages - advantage_mean) ** 2) + 1e-8
    )

    def _value_loss(
        selected_params, hstate, obs, done, positions, old_value, target
    ):
        _, _, value = jax.checkpoint(network.apply)(
            _rebuild_value_params(selected_params),
            hstate,
            (obs, done, positions),
        )
        value_clipped = old_value + (value - old_value).clip(
            -config["CLIP_EPS"], config["CLIP_EPS"]
        )
        # obs has shape (1, num_agents, ...), so mean() first combines the two
        # actors into one environment-level loss and then grad() is applied.
        return 0.5 * jnp.maximum(
            jnp.square(value - target),
            jnp.square(value_clipped - target),
        ).mean()

    def _actor_loss(
        selected_params, hstate, obs, done, positions, action, old_log_prob,
        advantage,
    ):
        _, policy, _ = jax.checkpoint(network.apply)(
            _rebuild_actor_params(selected_params),
            hstate,
            (obs, done, positions),
        )
        ratio = jnp.exp(policy.log_prob(action) - old_log_prob)
        normalized_advantage = (
            (advantage - advantage_mean) / (advantage_std + 1e-8)
        )
        return -jnp.minimum(
            ratio * normalized_advantage,
            jnp.clip(
                ratio,
                1.0 - config["CLIP_EPS"],
                1.0 + config["CLIP_EPS"],
            )
            * normalized_advantage,
        ).mean()

    value_grad = jax.grad(_value_loss)
    actor_grad = jax.grad(_actor_loss)

    def _make_sign_tree(params, ordered_keys, salt):
        base_key = jax.random.fold_in(
            jax.random.PRNGKey(sketch_seed), int(salt)
        )
        signs = {}
        for module_index, key in enumerate(ordered_keys):
            leaves, treedef = jax.tree_util.tree_flatten(params[key])
            module_key = jax.random.fold_in(base_key, module_index)
            module_signs = [
                jax.random.bernoulli(
                    jax.random.fold_in(module_key, leaf_index),
                    p=0.5,
                    shape=leaf.shape,
                )
                for leaf_index, leaf in enumerate(leaves)
            ]
            signs[key] = jax.tree_util.tree_unflatten(
                treedef, module_signs
            )
        return signs

    # Use the supplied logical module order rather than JAX's sorted dict-key
    # order. This keeps the projection aligned between the standard critic's
    # Dense_11 and PopArt's equivalently positioned critic_output module.
    value_signs = _make_sign_tree(value_params, value_keys, salt=0)
    actor_signs = _make_sign_tree(actor_params, actor_keys, salt=1)

    def _signed_feature_hash(gradient, signs, ordered_keys):
        """Balanced CountSketch-style projection without a dense P x r map."""
        sketch = jnp.zeros((sketch_dim,), dtype=jnp.float32)
        global_offset = 0
        for key in ordered_keys:
            gradient_leaves = jax.tree_util.tree_leaves(gradient[key])
            sign_leaves = jax.tree_util.tree_leaves(signs[key])
            for grad_leaf, sign_leaf in zip(gradient_leaves, sign_leaves):
                flat_gradient = jnp.asarray(
                    grad_leaf, dtype=jnp.float32
                ).reshape(-1)
                flat_sign = sign_leaf.reshape(-1)
                signed_gradient = jnp.where(
                    flat_sign, flat_gradient, -flat_gradient
                )

                # Coordinate i is assigned to bucket i mod sketch_dim.
                # Independent Rademacher signs make cross-bucket collisions
                # cancel in expectation.
                front_pad = global_offset % sketch_dim
                end_pad = (-(front_pad + int(grad_leaf.size))) % sketch_dim
                padded = jnp.pad(signed_gradient, (front_pad, end_pad))
                sketch = sketch + padded.reshape(
                    (-1, sketch_dim)
                ).sum(axis=0)
                global_offset += int(grad_leaf.size)
        return sketch

    def _env_major_transition(x, timestep):
        # batchify is agent-major: (agents * envs, ...) -> (envs, 1, agents, ...)
        x_t = x[timestep].reshape(
            (num_agents, num_envs) + x.shape[2:]
        )
        return jnp.moveaxis(x_t, 1, 0)[:, None, ...]

    def _env_major_hstate(hstate):
        return jax.tree.map(
            lambda h: jnp.moveaxis(
                h.reshape((num_agents, num_envs) + h.shape[1:]), 1, 0
            ),
            hstate,
        )

    # Recover the actual recurrent carry before each selected transition by
    # replaying only the required prefix segments. Carries are treated as fixed
    # inputs to the independent-transition gradient, matching the paper's
    # per-data-point view without storing the full rollout carry history.
    hstates_at_time = {}
    current_hstate = initial_hstate
    previous_timestep = 0
    for timestep in timesteps:
        if timestep > previous_timestep:
            current_hstate, _, _ = network.apply(
                original_params,
                current_hstate,
                (
                    traj_batch.obs[previous_timestep:timestep],
                    traj_batch.done[previous_timestep:timestep],
                    traj_batch.agent_positions[previous_timestep:timestep],
                ),
            )
        current_hstate = jax.tree.map(jax.lax.stop_gradient, current_hstate)
        hstates_at_time[timestep] = current_hstate
        previous_timestep = timestep

    actor_matrices = []
    value_matrices = []
    for timestep in timesteps:
        env_data = (
            _env_major_hstate(hstates_at_time[timestep]),
            _env_major_transition(traj_batch.obs, timestep),
            _env_major_transition(traj_batch.done, timestep),
            _env_major_transition(traj_batch.agent_positions, timestep),
            _env_major_transition(traj_batch.value, timestep),
            _env_major_transition(value_targets, timestep),
            _env_major_transition(traj_batch.action, timestep),
            _env_major_transition(traj_batch.log_prob, timestep),
            _env_major_transition(advantages, timestep),
        )

        def _project_one_environment(_, sample):
            (
                hstate, obs, done, positions, old_value, target,
                action, old_log_prob, advantage,
            ) = sample
            gradient_value = value_grad(
                value_params,
                hstate,
                obs,
                done,
                positions,
                old_value,
                target,
            )
            value_sketch = _signed_feature_hash(
                gradient_value, value_signs, value_keys
            )

            gradient_actor = actor_grad(
                actor_params,
                hstate,
                obs,
                done,
                positions,
                action,
                old_log_prob,
                advantage,
            )
            actor_sketch = _signed_feature_hash(
                gradient_actor, actor_signs, actor_keys
            )
            return None, (actor_sketch, value_sketch)

        _, (actor_sketches, value_sketches) = jax.lax.scan(
            _project_one_environment, None, env_data
        )

        def _cosine_matrix(sketches):
            row_norm = jnp.linalg.norm(sketches, axis=1)
            valid = row_norm > 1e-12
            unit_sketches = sketches / (row_norm[:, None] + 1e-12)
            matrix = jnp.clip(unit_sketches @ unit_sketches.T, -1.0, 1.0)
            valid_pairs = valid[:, None] & valid[None, :]
            matrix = jnp.where(valid_pairs, matrix, jnp.nan)
            diagonal = jnp.arange(num_envs)
            return matrix.at[diagonal, diagonal].set(
                jnp.where(valid, 1.0, jnp.nan)
            )

        actor_matrices.append(_cosine_matrix(actor_sketches))
        value_matrices.append(_cosine_matrix(value_sketches))

    return jnp.stack(
        (jnp.stack(actor_matrices), jnp.stack(value_matrices)), axis=0
    )


def render_projected_gradient_cosine_heatmaps(matrices, timesteps):
    """Render host-side RGBA images for WandB without retaining figures."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    matrices = np.asarray(matrices)
    timesteps = tuple(int(t) for t in timesteps)
    expected_prefix = (2, len(timesteps))
    if matrices.shape[:2] != expected_prefix:
        raise ValueError(
            f"expected matrix prefix {expected_prefix}, got {matrices.shape}"
        )

    images = {}
    for loss_index, loss_name in enumerate(("actor", "value")):
        for time_index, timestep in enumerate(timesteps):
            matrix = matrices[loss_index, time_index]
            fig, axis = plt.subplots(figsize=(6.4, 5.6), dpi=120)
            image = axis.imshow(
                matrix,
                cmap="coolwarm",
                vmin=-1.0,
                vmax=1.0,
                interpolation="nearest",
                origin="upper",
            )
            axis.set_title(
                "Projected normalized gradient Gram matrix\n"
                f"{loss_name}, t={timestep}; one joint sample per env"
            )
            axis.set_xlabel("environment slot j")
            axis.set_ylabel("environment slot i")
            fig.colorbar(image, ax=axis, label="projected cosine similarity")
            fig.tight_layout()
            fig.canvas.draw()
            rgba = np.asarray(fig.canvas.buffer_rgba()).copy()
            plt.close(fig)
            images[
                f"gradient_covariance_projected/{loss_name}/t_{timestep}"
            ] = rgba
    return images


def empty_gradient_conflict_metrics(layout_names, dtype=jnp.float32):
    """Shape-compatible skipped output for interval-only diagnostics."""
    names = []
    for layout_name in layout_names:
        names.append(f"sample_share/{layout_name}")

    for loss_type in ("actor", "value"):
        norm_role = "actor" if loss_type == "actor" else "critic"
        for layout_name in layout_names:
            names.append(f"grad_norm_{norm_role}/{layout_name}")
            names.append(
                f"grad_contribution_magnitude_{norm_role}/{layout_name}"
            )
            names.append(
                f"grad_contribution_signed_{norm_role}/{layout_name}"
            )
        names.append(f"grad_norm_{norm_role}/total")

        for i, layout_i in enumerate(layout_names):
            for layout_j in layout_names[i + 1:]:
                names.append(
                    f"grad_conflict_{loss_type}/{layout_i}_vs_{layout_j}"
                )
                names.append(
                    f"grad_neg_dot_{loss_type}/{layout_i}_vs_{layout_j}"
                )

        for layout_name in layout_names:
            names.append(f"grad_conflict_{loss_type}/alignment/{layout_name}")
            names.append(
                f"grad_conflict_{loss_type}/alignment_loo/{layout_name}"
            )

        for metric_name in (
            "avg_pairwise_cosine",
            "conflict_rate",
            "avg_negative_cosine",
            "alignment",
        ):
            names.append(f"grad_conflict_sample_{loss_type}/{metric_name}")

    return {
        name: jnp.asarray(jnp.nan, dtype=dtype)
        for name in names
    }


def compute_gradient_conflict_metrics(
    *,
    network,
    original_params,
    initial_hstate,
    traj_batch,
    advantages,
    value_targets,
    layout_ids_full,
    layout_names,
    config,
    num_agents,
    pairing_key,
    value_trunk_keys,
    actor_trunk_keys,
):
    _num_layouts = len(layout_names)
    # subsample: use only the first _GC_STEPS steps to reduce activation memory
    _GC_STEPS = config["DIAGNOSTIC_WINDOW_STEPS"]
    _gc_traj = jax.tree.map(lambda x: x[:_GC_STEPS], traj_batch)
    _gc_adv  = advantages[:_GC_STEPS]
    _gc_tgt  = value_targets[:_GC_STEPS]
    # Use one common normalization for every layout so count-weighted layout
    # gradients reconstruct the gradient of the combined GC-window loss.
    _gc_adv_mean = jnp.mean(_gc_adv)
    _gc_adv_std = jnp.sqrt(
        jnp.mean((_gc_adv - _gc_adv_mean) ** 2) + 1e-8
    )
    
    # Reuse environment-level classification for the conflict subsample.
    _layout_ids = layout_ids_full[:_GC_STEPS]
    
    def _tdot(g1, g2):
        return sum(
            jnp.sum(a * b)
            for a, b in zip(jax.tree_util.tree_leaves(g1), jax.tree_util.tree_leaves(g2))
        )
    
    def _tnorm2(g):
        return sum(jnp.sum(a ** 2) for a in jax.tree_util.tree_leaves(g))
    
    # Per-loss-type scalar accumulators: norms_sq[lid], dots[(i,j)], prev[lid]
    # actor and value are logged; entropy is excluded (less interpretable).
    _gc_state = {
        'actor': {'norms_sq': [], 'dots': {}, 'prev': []},
        'value': {'norms_sq': [], 'dots': {}, 'prev': []},
    }
    
    _sample_counts = []
    for _lid in range(_num_layouts):
        # Classify at the environment level first, then include all agents from
        # that environment in the family loss. batchify is agent-major, so
        # tiling reproduces the (agent_0 envs, agent_1 envs, ...) actor order.
        _env_mask = (_layout_ids == _lid).astype(jnp.float32)  # (T, NUM_ENVS)
        _mask = jnp.tile(_env_mask, (1, num_agents))       # (T, NUM_ACTORS)
        _cnt = _mask.sum() + 1e-8
        _sample_counts.append(_env_mask.sum())
    
        # Single forward pass per layout; 2 backward passes via vjp cotangents.
        def _fwd(p, mask=_mask, cnt=_cnt):
            _, pi, value = jax.checkpoint(network.apply)(
                p, initial_hstate,
                (_gc_traj.obs, _gc_traj.done, _gc_traj.agent_positions),
            )
            lp = pi.log_prob(_gc_traj.action)
            gae = (_gc_adv - _gc_adv_mean) / (_gc_adv_std + 1e-8)
            ratio = jnp.exp(lp - _gc_traj.log_prob)
            al = -(jnp.minimum(
                ratio * gae,
                jnp.clip(ratio, 1 - config["CLIP_EPS"], 1 + config["CLIP_EPS"]) * gae,
            ) * mask).sum() / cnt
            vpc = _gc_traj.value + (value - _gc_traj.value).clip(
                -config["CLIP_EPS"], config["CLIP_EPS"]
            )
            vl = 0.5 * (jnp.maximum(
                jnp.square(value - _gc_tgt), jnp.square(vpc - _gc_tgt)
            ) * mask).sum() / cnt
            return al, vl
    
        _, _vjp_fn = jax.vjp(_fwd, original_params)
        # cotangent (1, 0) → actor gradient; (0, 1) → value gradient
        _g_actor, = _vjp_fn((1.0, 0.0))
        _g_value, = _vjp_fn((0.0, 1.0))
    
        for _loss_type, _g in [('actor', _g_actor), ('value', _g_value)]:
            _s = _gc_state[_loss_type]
            _s['norms_sq'].append(_tnorm2(_g))
            for _prev_lid, _g_prev in enumerate(_s['prev']):
                _s['dots'][(_prev_lid, _lid)] = _tdot(_g_prev, _g)
            _s['prev'].append(_g)
    
    grad_conflict = {}
    _sample_counts_arr = jnp.stack(_sample_counts)
    _sample_total = _sample_counts_arr.sum() + 1e-8
    for _i in range(_num_layouts):
        grad_conflict[f"sample_share/{layout_names[_i]}"] = _sample_counts_arr[_i] / _sample_total
    
    for _loss_type, _s in _gc_state.items():
        _valid_layout = _sample_counts_arr > 0
        _norm_role = "actor" if _loss_type == "actor" else "critic"
        # per-layout gradient norms
        for _i in range(_num_layouts):
            grad_conflict[f"grad_norm_{_norm_role}/{layout_names[_i]}"] = (
                jnp.where(
                    _valid_layout[_i],
                    jnp.sqrt(_s['norms_sq'][_i]),
                    jnp.nan,
                )
            )
        # pairwise cosine similarities
        for _i in range(_num_layouts):
            for _j in range(_i + 1, _num_layouts):
                _pair_valid = _valid_layout[_i] & _valid_layout[_j]
                cos = jnp.where(_pair_valid, _s['dots'][(_i, _j)] / (
                    jnp.sqrt(_s['norms_sq'][_i] * _s['norms_sq'][_j]) + 1e-8
                ), jnp.nan)
                grad_conflict[
                    f"grad_conflict_{_loss_type}/{layout_names[_i]}_vs_{layout_names[_j]}"
                ] = cos
                # neg_dot_ij = max(0, -g_i · g_j): magnitude of conflict, 0 when aligned
                neg_dot = jnp.where(
                    _pair_valid,
                    jnp.maximum(0.0, -_s['dots'][(_i, _j)]),
                    jnp.nan,
                )
                grad_conflict[
                    f"grad_neg_dot_{_loss_type}/{layout_names[_i]}_vs_{layout_names[_j]}"
                ] = neg_dot
        # Count-weighted combined direction. Each g_i is a per-layout mean,
        # so count_i * g_i is proportional to that layout's contribution to
        # the combined GC-window gradient.
        _g_all = jax.tree.map(
            lambda *gs: sum(
                _sample_counts_arr[i] * g for i, g in enumerate(gs)
            ),
            *_s['prev'],
        )
        _norm_all_sq = _tnorm2(_g_all)
        # `_g_all` is the count-weighted sum of per-layout mean gradients.
        # Divide by the total count so this norm is directly comparable with
        # each per-layout mean-gradient norm above.
        grad_conflict[f"grad_norm_{_norm_role}/total"] = (
            jnp.sqrt(_norm_all_sq) / _sample_total
        )
        for _i in range(_num_layouts):
            _dot_i_all = _tdot(_s['prev'][_i], _g_all)
            _align = jnp.where(
                _valid_layout[_i],
                _dot_i_all / (
                    jnp.sqrt(_s['norms_sq'][_i] * _norm_all_sq) + 1e-8
                ),
                jnp.nan,
            )
            grad_conflict[f"grad_conflict_{_loss_type}/alignment/{layout_names[_i]}"] = _align
            # Two complementary layout-update signals:
            #   magnitude = sample_share_i * ||mean_gradient_i||
            #   signed = magnitude * cos(mean_gradient_i, combined_gradient)
            # The first ignores direction; the second is signed and measures
            # the component along the actual combined update direction.
            _sample_share_i = _sample_counts_arr[_i] / _sample_total
            _weighted_magnitude = jnp.where(
                _valid_layout[_i],
                _sample_share_i * jnp.sqrt(_s['norms_sq'][_i]),
                jnp.nan,
            )
            grad_conflict[
                f"grad_contribution_magnitude_{_norm_role}/{layout_names[_i]}"
            ] = _weighted_magnitude
            grad_conflict[
                f"grad_contribution_signed_{_norm_role}/{layout_names[_i]}"
            ] = jnp.where(
                _valid_layout[_i], _weighted_magnitude * _align, jnp.nan
            )
            # Remove the layout's full count-weighted contribution.
            _g_others = jax.tree.map(
                lambda ga, gi: ga - _sample_counts_arr[_i] * gi,
                _g_all,
                _s['prev'][_i],
            )
            _norm_others_sq = _tnorm2(_g_others)
            _dot_i_others = _tdot(_s['prev'][_i], _g_others)
            _others_valid = (_sample_total - _sample_counts_arr[_i]) > 1e-8
            _align_loo = jnp.where(
                _valid_layout[_i] & _others_valid,
                _dot_i_others / (
                    jnp.sqrt(_s['norms_sq'][_i] * _norm_others_sq) + 1e-8
                ),
                jnp.nan,
            )
            grad_conflict[f"grad_conflict_{_loss_type}/alignment_loo/{layout_names[_i]}"] = _align_loo
    # ── end gradient conflict ──────────────────────────────────────
    
    # ── sample-level (per-environment) gradient conflict ─────────────────────
    # One sample is one environment slot's GC-window batch slice containing all
    # agents. It may cross an episode reset; `done` correctly resets the RNN
    # state. Gradients are compared across NUM_ENVS without separating layouts.
    _VALUE_TRUNK_KEYS = tuple(value_trunk_keys)
    _ACTOR_TRUNK_KEYS = tuple(actor_trunk_keys)
    
    def _select_keys(g, keys):
        return {k: g['params'][k] for k in keys}
    
    def _restricted_normsq(g, keys):
        selected = _select_keys(g, keys)
        return sum(jnp.sum(x ** 2) for x in jax.tree_util.tree_leaves(selected))
    
    _other_value_params = {
        k: v for k, v in original_params['params'].items()
        if k not in _VALUE_TRUNK_KEYS
    }
    _value_trunk_params = _select_keys(original_params, _VALUE_TRUNK_KEYS)
    _other_actor_params = {
        k: v for k, v in original_params['params'].items()
        if k not in _ACTOR_TRUNK_KEYS
    }
    _actor_trunk_params = _select_keys(original_params, _ACTOR_TRUNK_KEYS)
    
    def _rebuild_value_params(vp):
        merged = dict(_other_value_params)
        merged.update(vp)
        return {'params': merged}
    
    def _rebuild_actor_params(ap):
        merged = dict(_other_actor_params)
        merged.update(ap)
        return {'params': merged}
    
    def _per_env_value_loss(vp, hstate_i, obs_i, done_i, pos_i, value_i, tgt_i):
        _, _, value = jax.checkpoint(network.apply)(
            _rebuild_value_params(vp), hstate_i, (obs_i, done_i, pos_i)
        )
        vpc = value_i + (value - value_i).clip(
            -config["CLIP_EPS"], config["CLIP_EPS"]
        )
        return 0.5 * jnp.maximum(
            jnp.square(value - tgt_i), jnp.square(vpc - tgt_i)
        ).mean()
    
    def _per_env_value_grad(vp, hstate_i, obs_i, done_i, pos_i, value_i, tgt_i):
        return jax.grad(_per_env_value_loss)(
            vp, hstate_i, obs_i, done_i, pos_i, value_i, tgt_i
        )
    
    def _per_env_actor_loss(
        ap, hstate_i, obs_i, done_i, pos_i, action_i, logprob_i, adv_i
    ):
        _, pi, _ = jax.checkpoint(network.apply)(
            _rebuild_actor_params(ap), hstate_i, (obs_i, done_i, pos_i)
        )
        ratio = jnp.exp(pi.log_prob(action_i) - logprob_i)
        gae_i = (adv_i - _gc_adv_mean) / (_gc_adv_std + 1e-8)
        return -jnp.minimum(
            ratio * gae_i,
            jnp.clip(
                ratio, 1 - config["CLIP_EPS"], 1 + config["CLIP_EPS"]
            ) * gae_i,
        ).mean()
    
    def _per_env_actor_grad(
        ap, hstate_i, obs_i, done_i, pos_i, action_i, logprob_i, adv_i
    ):
        return jax.grad(_per_env_actor_loss)(
            ap, hstate_i, obs_i, done_i, pos_i,
            action_i, logprob_i, adv_i,
        )
    
    # The global mean uses exactly the same loss definition as every env sample.
    def _global_sample_fwd(p):
        _, pi, value = jax.checkpoint(network.apply)(
            p, initial_hstate,
            (_gc_traj.obs, _gc_traj.done, _gc_traj.agent_positions),
        )
        ratio = jnp.exp(pi.log_prob(_gc_traj.action) - _gc_traj.log_prob)
        gae = (_gc_adv - _gc_adv_mean) / (_gc_adv_std + 1e-8)
        actor_loss = -jnp.minimum(
            ratio * gae,
            jnp.clip(
                ratio, 1 - config["CLIP_EPS"], 1 + config["CLIP_EPS"]
            ) * gae,
        ).mean()
        vpc = _gc_traj.value + (value - _gc_traj.value).clip(
            -config["CLIP_EPS"], config["CLIP_EPS"]
        )
        value_loss = 0.5 * jnp.maximum(
            jnp.square(value - _gc_tgt), jnp.square(vpc - _gc_tgt)
        ).mean()
        return actor_loss, value_loss
    
    _, _global_vjp = jax.vjp(_global_sample_fwd, original_params)
    _global_actor_grad, = _global_vjp((1.0, 0.0))
    _global_value_grad, = _global_vjp((0.0, 1.0))
    
    _CHUNK = config["GRAD_CONFLICT_CHUNK_SIZE"]
    _n_envs = config["NUM_ENVS"]
    _n_agents = num_agents
    _n_chunks = _n_envs // _CHUNK
    # Form a new random perfect matching at every diagnostic update.  After
    # permutation, adjacent slots make NUM_ENVS / 2 disjoint unordered pairs.
    # This keeps the streaming memory footprint while avoiding a fixed
    # environment-slot pairing bias.
    _env_permutation = jax.random.permutation(pairing_key, _n_envs)
    
    # batchify is agent-major:
    # (T, agents * envs, ...) -> (chunks, chunk, T, agents, ...)
    def _to_env_chunks(x):
        x = x.reshape(
            (x.shape[0], _n_agents, _n_envs) + x.shape[2:]
        )
        x = jnp.moveaxis(x, 2, 0)
        x = jnp.take(x, _env_permutation, axis=0)
        return x.reshape((_n_chunks, _CHUNK) + x.shape[1:])
    
    _obs_ec = _to_env_chunks(_gc_traj.obs)
    _done_ec = _to_env_chunks(_gc_traj.done)
    _pos_ec = _to_env_chunks(_gc_traj.agent_positions)
    _value_ec = _to_env_chunks(_gc_traj.value)
    _tgt_ec = _to_env_chunks(_gc_tgt)
    _action_ec = _to_env_chunks(_gc_traj.action)
    _logprob_ec = _to_env_chunks(_gc_traj.log_prob)
    _adv_ec = _to_env_chunks(_gc_adv)
    def _hstate_to_env_chunks(h):
        h = jnp.moveaxis(
            h.reshape((_n_agents, _n_envs) + h.shape[1:]), 1, 0
        )
        h = jnp.take(h, _env_permutation, axis=0)
        return h.reshape(
            (_n_chunks, _CHUNK, _n_agents) + h.shape[2:]
        )

    _hstate_ec = jax.tree.map(_hstate_to_env_chunks, initial_hstate)
    
    def _accumulate_env_gradients(carry, env_data):
        (
            _sum_unit_value, _sum_unit_actor,
            _sum_sqnorm_value, _sum_sqnorm_actor,
            _sum_unit_sqnorm_value, _sum_unit_sqnorm_actor,
            _pending_unit_value, _pending_unit_actor, _has_pending,
            _conflict_count_value, _conflict_count_actor,
            _negative_cosine_sum_value, _negative_cosine_sum_actor,
            _pair_count,
        ) = carry
        (
            hstate_i, obs_i, done_i, pos_i, value_i, tgt_i,
            action_i, logprob_i, adv_i,
        ) = env_data
        _g_value = _per_env_value_grad(
            _value_trunk_params, hstate_i, obs_i, done_i,
            pos_i, value_i, tgt_i,
        )
        _g_actor = _per_env_actor_grad(
            _actor_trunk_params, hstate_i, obs_i, done_i,
            pos_i, action_i, logprob_i, adv_i,
        )
    
        _sq_value = _tnorm2(_g_value)
        _sq_actor = _tnorm2(_g_actor)
        _denom_value = jnp.sqrt(_sq_value) + 1e-8
        _denom_actor = jnp.sqrt(_sq_actor) + 1e-8
        _unit_value = jax.tree.map(
            lambda g: g / _denom_value, _g_value
        )
        _unit_actor = jax.tree.map(
            lambda g: g / _denom_actor, _g_actor
        )
        _sum_unit_value = jax.tree.map(
            lambda total, unit: total + unit,
            _sum_unit_value,
            _unit_value,
        )
        _sum_unit_actor = jax.tree.map(
            lambda total, unit: total + unit,
            _sum_unit_actor,
            _unit_actor,
        )

        # Estimate sign-sensitive pair statistics without retaining all
        # NUM_ENVS gradients. Adjacent slots in the randomized order form a
        # perfect matching, giving NUM_ENVS / 2 pair samples per update.
        _pair_cosine_value = _tdot(_pending_unit_value, _unit_value)
        _pair_cosine_actor = _tdot(_pending_unit_actor, _unit_actor)
        _pair_weight = _has_pending.astype(jnp.float32)
        _conflict_count_value += _pair_weight * (
            _pair_cosine_value < 0
        ).astype(jnp.float32)
        _conflict_count_actor += _pair_weight * (
            _pair_cosine_actor < 0
        ).astype(jnp.float32)
        _negative_cosine_sum_value += _pair_weight * jnp.maximum(
            0.0, -_pair_cosine_value
        )
        _negative_cosine_sum_actor += _pair_weight * jnp.maximum(
            0.0, -_pair_cosine_actor
        )
        _pair_count += _pair_weight
        _pending_unit_value = jax.tree.map(
            lambda pending, unit: jnp.where(
                _has_pending, jnp.zeros_like(pending), unit
            ),
            _pending_unit_value,
            _unit_value,
        )
        _pending_unit_actor = jax.tree.map(
            lambda pending, unit: jnp.where(
                _has_pending, jnp.zeros_like(pending), unit
            ),
            _pending_unit_actor,
            _unit_actor,
        )
        return (
            _sum_unit_value,
            _sum_unit_actor,
            _sum_sqnorm_value + _sq_value,
            _sum_sqnorm_actor + _sq_actor,
            _sum_unit_sqnorm_value + _sq_value / (_denom_value ** 2),
            _sum_unit_sqnorm_actor + _sq_actor / (_denom_actor ** 2),
            _pending_unit_value,
            _pending_unit_actor,
            ~_has_pending,
            _conflict_count_value,
            _conflict_count_actor,
            _negative_cosine_sum_value,
            _negative_cosine_sum_actor,
            _pair_count,
        ), None
    
    def _chunk_body(carry, chunk_data):
        return jax.lax.scan(_accumulate_env_gradients, carry, chunk_data)[0], None
    
    _sample_accum_init = (
        jax.tree.map(jnp.zeros_like, _value_trunk_params),
        jax.tree.map(jnp.zeros_like, _actor_trunk_params),
        jnp.array(0.0),
        jnp.array(0.0),
        jnp.array(0.0),
        jnp.array(0.0),
        jax.tree.map(jnp.zeros_like, _value_trunk_params),
        jax.tree.map(jnp.zeros_like, _actor_trunk_params),
        jnp.array(False),
        jnp.array(0.0),
        jnp.array(0.0),
        jnp.array(0.0),
        jnp.array(0.0),
        jnp.array(0.0),
    )
    (
        _sum_unit_value, _sum_unit_actor,
        _sum_sqnorm_value, _sum_sqnorm_actor,
        _sum_unit_sqnorm_value, _sum_unit_sqnorm_actor,
        _pending_unit_value, _pending_unit_actor, _has_pending,
        _conflict_count_value, _conflict_count_actor,
        _negative_cosine_sum_value, _negative_cosine_sum_actor,
        _pair_count,
    ), _ = jax.lax.scan(
        _chunk_body,
        _sample_accum_init,
        (
            _hstate_ec, _obs_ec, _done_ec, _pos_ec, _value_ec,
            _tgt_ec, _action_ec, _logprob_ec, _adv_ec,
        ),
    )
    
    for (
        _loss_type, _sum_indiv_sqnorm, _sum_unit,
        _sum_unit_indiv_sqnorm, _global_grad, _keys,
        _conflict_count, _negative_cosine_sum,
    ) in (
        (
            'value', _sum_sqnorm_value, _sum_unit_value,
            _sum_unit_sqnorm_value, _global_value_grad, _VALUE_TRUNK_KEYS,
            _conflict_count_value, _negative_cosine_sum_value,
        ),
        (
            'actor', _sum_sqnorm_actor, _sum_unit_actor,
            _sum_unit_sqnorm_actor, _global_actor_grad, _ACTOR_TRUNK_KEYS,
            _conflict_count_actor, _negative_cosine_sum_actor,
        ),
    ):
        _mean_grad_normsq = _restricted_normsq(_global_grad, _keys)
        _sum_all_sqnorm = (_n_envs ** 2) * _mean_grad_normsq
        _alignment = (
            _sum_all_sqnorm / (_n_envs * _sum_indiv_sqnorm + 1e-8)
        )
        # For u_i = g_i / (||g_i|| + eps):
        # sum_{i != j} cos(g_i, g_j) = ||sum_i u_i||^2 - sum_i ||u_i||^2.
        _avg_pairwise_cosine = (
            (_tnorm2(_sum_unit) - _sum_unit_indiv_sqnorm)
            / (_n_envs * (_n_envs - 1) + 1e-8)
        )
        grad_conflict[
            f"grad_conflict_sample_{_loss_type}/avg_pairwise_cosine"
        ] = _avg_pairwise_cosine
        grad_conflict[
            f"grad_conflict_sample_{_loss_type}/conflict_rate"
        ] = _conflict_count / (_pair_count + 1e-8)
        grad_conflict[
            f"grad_conflict_sample_{_loss_type}/avg_negative_cosine"
        ] = _negative_cosine_sum / (_pair_count + 1e-8)
        grad_conflict[
            f"grad_conflict_sample_{_loss_type}/alignment"
        ] = _alignment
    # ── end sample-level gradient conflict ──────────────────────────
    
    return grad_conflict
