"""Shared gradient-conflict diagnostics for CEC IPPO variants."""

import jax
import jax.numpy as jnp


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
    value_trunk_keys,
    actor_trunk_keys,
):
    _num_layouts = len(layout_names)
    # subsample: use only the first _GC_STEPS steps to reduce activation memory
    _GC_STEPS = config["GRAD_CONFLICT_STEPS"]
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
        grad_conflict[
            f"grad_conflict/sample_count/{layout_names[_i]}"
        ] = _sample_counts_arr[_i]
        grad_conflict[f"sample_share/{layout_names[_i]}"] = _sample_counts_arr[_i] / _sample_total
    
    for _loss_type, _s in _gc_state.items():
        _valid_layout = _sample_counts_arr > 0
        # per-layout gradient norms
        for _i in range(_num_layouts):
            grad_conflict[f"grad_conflict_{_loss_type}/norm/{layout_names[_i]}"] = (
                jnp.where(
                    _valid_layout[_i],
                    jnp.sqrt(_s['norms_sq'][_i]),
                    jnp.nan,
                )
            )
        # gradient share p_f, dominance ratio D, norm CV
        _norms = jnp.stack([
            jnp.sqrt(_s['norms_sq'][_i]) for _i in range(_num_layouts)
        ])
        _norm_sum = _norms.sum() + 1e-8
        for _i in range(_num_layouts):
            grad_conflict[f"grad_share_{_loss_type}/{layout_names[_i]}"] = jnp.where(
                _valid_layout[_i], _norms[_i] / _norm_sum, jnp.nan
            )
        _valid_norms = jnp.where(_valid_layout, _norms, jnp.nan)
        grad_conflict[f"grad_dominance_{_loss_type}"] = (
            jnp.nanmax(_valid_norms) / (jnp.nanmedian(_valid_norms) + 1e-8)
        )
        grad_conflict[f"grad_norm_cv_{_loss_type}"] = (
            jnp.nanstd(_valid_norms) / (jnp.nanmean(_valid_norms) + 1e-8)
        )
    
        # sample-weighted gradient share: weights each layout's (per-sample-mean) norm
        # by its actual sample count, approximating its contribution to the real,
        # unmasked combined gradient (which averages over all samples, not per layout).
        _weighted_norms = _norms * _sample_counts_arr
        _weighted_norm_sum = _weighted_norms.sum() + 1e-8
        for _i in range(_num_layouts):
            grad_conflict[f"grad_share_weighted_{_loss_type}/{layout_names[_i]}"] = (
                jnp.where(
                    _valid_layout[_i],
                    _weighted_norms[_i] / _weighted_norm_sum,
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
    assert _n_envs % _CHUNK == 0, (
        "GRAD_CONFLICT_CHUNK_SIZE must evenly divide NUM_ENVS"
    )
    _n_chunks = _n_envs // _CHUNK
    
    # batchify is agent-major:
    # (T, agents * envs, ...) -> (chunks, chunk, T, agents, ...)
    def _to_env_chunks(x):
        x = x.reshape(
            (x.shape[0], _n_agents, _n_envs) + x.shape[2:]
        )
        x = jnp.moveaxis(x, 2, 0)
        return x.reshape((_n_chunks, _CHUNK) + x.shape[1:])
    
    _obs_ec = _to_env_chunks(_gc_traj.obs)
    _done_ec = _to_env_chunks(_gc_traj.done)
    _pos_ec = _to_env_chunks(_gc_traj.agent_positions)
    _value_ec = _to_env_chunks(_gc_traj.value)
    _tgt_ec = _to_env_chunks(_gc_tgt)
    _action_ec = _to_env_chunks(_gc_traj.action)
    _logprob_ec = _to_env_chunks(_gc_traj.log_prob)
    _adv_ec = _to_env_chunks(_gc_adv)
    _hstate_ec = jax.tree.map(
        lambda h: jnp.moveaxis(
            h.reshape((_n_agents, _n_envs) + h.shape[1:]), 1, 0
        ).reshape(
            (_n_chunks, _CHUNK, _n_agents) + h.shape[1:]
        ),
        initial_hstate,
    )
    
    def _accumulate_env_gradients(carry, env_data):
        (
            _sum_unit_value, _sum_unit_actor,
            _sum_sqnorm_value, _sum_sqnorm_actor,
            _sum_unit_sqnorm_value, _sum_unit_sqnorm_actor,
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
        return (
            _sum_unit_value,
            _sum_unit_actor,
            _sum_sqnorm_value + _sq_value,
            _sum_sqnorm_actor + _sq_actor,
            _sum_unit_sqnorm_value + _sq_value / (_denom_value ** 2),
            _sum_unit_sqnorm_actor + _sq_actor / (_denom_actor ** 2),
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
    )
    (
        _sum_unit_value, _sum_unit_actor,
        _sum_sqnorm_value, _sum_sqnorm_actor,
        _sum_unit_sqnorm_value, _sum_unit_sqnorm_actor,
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
    ) in (
        (
            'value', _sum_sqnorm_value, _sum_unit_value,
            _sum_unit_sqnorm_value, _global_value_grad, _VALUE_TRUNK_KEYS,
        ),
        (
            'actor', _sum_sqnorm_actor, _sum_unit_actor,
            _sum_unit_sqnorm_actor, _global_actor_grad, _ACTOR_TRUNK_KEYS,
        ),
    ):
        _mean_grad_normsq = _restricted_normsq(_global_grad, _keys)
        _sum_all_sqnorm = (_n_envs ** 2) * _mean_grad_normsq
        _avg_pairwise_dot = (
            (_sum_all_sqnorm - _sum_indiv_sqnorm)
            / (_n_envs * (_n_envs - 1) + 1e-8)
        )
        _alignment = (
            _sum_all_sqnorm / (_n_envs * _sum_indiv_sqnorm + 1e-8)
        )
        grad_conflict[
            f"grad_conflict_sample_{_loss_type}/avg_pairwise_dot"
        ] = _avg_pairwise_dot
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
            f"grad_conflict_sample_{_loss_type}/alignment"
        ] = _alignment
    # ── end sample-level gradient conflict ──────────────────────────
    
    return grad_conflict
