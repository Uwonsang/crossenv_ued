import jax
import jax.numpy as jnp
from jaxmarl.environments.overcooked.layouts import (
    make_asymm_advantages_9x9,
    make_coord_ring_9x9,
    make_counter_circuit_9x9,
    make_forced_coord_9x9,
    make_cramped_room_9x9,
)
from collections import namedtuple
from .ued_scores import (
    UEDScore,
    compute_episodic_stats,
    compute_max_mc,
    compute_ued_scores_no_vmap,
    plr_flatten_scores_to_env,
)
PLRUEDBatch = namedtuple("PLRUEDBatch", ["rewards", "dones", "values", "advantages"])
MAX_WALL_IDX_LEN = 75


def pad_wall_idx(layout, max_len: int = MAX_WALL_IDX_LEN):
    w = layout["wall_idx"]
    pad_amt = max_len - w.shape[0]
    padded = jnp.pad(w, (0, pad_amt), mode="edge")
    return layout.copy({"wall_idx": padded})


@jax.jit
def sample_layout_reset_all(key):

    def mk(fn):
        def f(k):
            k, sk = jax.random.split(k)
            layout = fn(sk, ik=True)
            return pad_wall_idx(layout)

        return f

    branches = (
        mk(make_asymm_advantages_9x9),
        mk(make_coord_ring_9x9),
        mk(make_counter_circuit_9x9),
        mk(make_forced_coord_9x9),
        mk(make_cramped_room_9x9),
    )
    idx = jax.random.randint(key, (), 0, 5)
    return jax.lax.switch(idx, branches, key)


def plr_batch_from_traj(traj_batch, advantages, num_steps, num_agents, num_envs):

    r_all = traj_batch.reward.reshape(num_steps, num_agents, num_envs)
    v_all = traj_batch.value.reshape(num_steps, num_agents, num_envs)
    gd_all = traj_batch.global_done.reshape(num_steps, num_agents, num_envs)
    adv_all = advantages.reshape(num_steps, num_agents, num_envs)
    return PLRUEDBatch(
        rewards=r_all.transpose(1, 0, 2),
        dones=gd_all.transpose(1, 0, 2),
        values=v_all.transpose(1, 0, 2),
        advantages=adv_all.transpose(1, 0, 2),
    )


def plr_ued_scores_and_info(plr_ued_score, batch, plr_buffer, level_idxs, num_envs):
    """
    Returns (ued_scores (ne,), update_info: None | {'max_returns': ...}).
    MAX_MC needs buffer prev max; other scores use _compute_ued_scores only.
    """
    if plr_ued_score == UEDScore.MAX_MC:
        safe_idx = jnp.maximum(level_idxs, 0)
        prev_max = jnp.where(
            level_idxs >= 0,
            plr_buffer.max_returns[safe_idx],
            jnp.full((num_envs,), -jnp.inf),
        )
        mean_scores, _, _ = compute_max_mc(batch, {"max_returns": prev_max})
        ued_scores = plr_flatten_scores_to_env(mean_scores, num_envs)
        _, max_ep = compute_episodic_stats(batch.rewards, batch.dones, time_average=False)
        merged = jnp.maximum(max_ep, prev_max)
        return ued_scores, {"max_returns": merged}
    
    mean_s, _, _ = compute_ued_scores_no_vmap(plr_ued_score, batch, None)
    ued_scores = plr_flatten_scores_to_env(mean_s, num_envs)
    
    return ued_scores, None