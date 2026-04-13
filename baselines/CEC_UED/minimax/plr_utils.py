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
    _compute_ued_scores,
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
    layout_dict = jax.lax.switch(idx, branches, key)
    return layout_dict


@jax.jit
def check_heldout_match_overcooked(state, held_out_goal, held_out_wall, held_out_pot):
    """Return True when a state matches one of held-out layouts."""
    goal_match = jax.vmap(lambda g_pos: jnp.all(g_pos == state.goal_pos))(held_out_goal)
    wall_match = jax.vmap(lambda w_map: jnp.all(w_map == state.wall_map))(held_out_wall)
    pot_match = jax.vmap(lambda p_pos: jnp.all(p_pos == state.pot_pos))(held_out_pot)
    return jnp.all(jnp.stack([goal_match, wall_match, pot_match], axis=0), axis=0).any()


def plr_batch_from_traj(traj_batch, advantages, num_steps, num_agents, num_envs):

    r_all = traj_batch.reward.reshape(num_steps, num_agents, num_envs)
    v_all = traj_batch.value.reshape(num_steps, num_agents, num_envs)
    gd_all = traj_batch.global_done.reshape(num_steps, num_agents, num_envs)
    adv_all = advantages.reshape(num_steps, num_agents, num_envs)

    return PLRUEDBatch(
        rewards=r_all.transpose(0, 2, 1), # (time, num_envs, num_agents)
        values=v_all.transpose(0, 2, 1), # (time, num_envs, num_agents)
        dones=gd_all.transpose(0, 2, 1), # (time, num_envs, num_agents)
        advantages=adv_all.transpose(0, 2, 1) # (time, num_envs, num_agents)
    )


def plr_ued_scores_and_info(plr_ued_score, batch, plr_buffer, level_idxs, num_envs):
    info = None
    prev_max = None
    if plr_ued_score == UEDScore.MAX_MC:
        safe_idx = jnp.maximum(level_idxs, 0)
        prev_max = jnp.where(
            level_idxs >= 0,
            plr_buffer.max_returns[safe_idx],
            jnp.full((num_envs,), -jnp.inf),
        )
        info = {"max_returns": prev_max}

    mean_s, _, score_info = _compute_ued_scores(plr_ued_score, batch, info)
    ued_scores = jnp.ravel(jnp.asarray(mean_s))

    if plr_ued_score == UEDScore.MAX_MC:
        merged_max_returns = jnp.maximum(score_info["max_returns"], prev_max)
        return ued_scores, {"max_returns": merged_max_returns}

    return ued_scores, None

def layout_comparator(a, b):
    keys = (
        "agent_idx",
        "goal_idx",
        "onion_pile_idx",
        "plate_pile_idx",
        "pot_idx",
        "wall_idx",
    )
    checks = [jnp.all(a[k] == b[k]) for k in keys]
    return jnp.all(jnp.stack(checks))