from typing import Tuple

import chex
import jax
import jax.numpy as jnp
from flax import struct


@struct.dataclass
class ReachabilityState:
    flood_input: chex.Array
    flood_count: chex.Array
    target_mask: chex.Array
    done: chex.Array


@struct.dataclass
class LayoutValidationResult:
    pot_count: chex.Array
    onion_count: chex.Array
    plate_count: chex.Array
    goal_count: chex.Array

    count_ok: chex.Array

    onion_reachable: chex.Array
    plate_reachable: chex.Array
    pot_reachable: chex.Array
    goal_reachable: chex.Array

    reachable_ok: chex.Array
    valid: chex.Array


def make_tile_mask(env_map: chex.Array, tile_idx: int) -> chex.Array:
    return env_map == tile_idx


def make_multi_tile_mask(env_map: chex.Array, tile_indices: Tuple[int, ...]) -> chex.Array:
    mask = jnp.zeros(env_map.shape, dtype=bool)

    for tile_idx in tile_indices:
        mask = jnp.logical_or(mask, env_map == tile_idx)

    return mask


def make_passable_mask(env_map: chex.Array, passable_tiles: Tuple[int, ...]) -> chex.Array:
    return make_multi_tile_mask(env_map, passable_tiles)


def shift_up(mask: chex.Array) -> chex.Array:
    shifted = jnp.zeros_like(mask)
    shifted = shifted.at[:-1, :].set(mask[1:, :])
    return shifted


def shift_down(mask: chex.Array) -> chex.Array:
    shifted = jnp.zeros_like(mask)
    shifted = shifted.at[1:, :].set(mask[:-1, :])
    return shifted


def shift_left(mask: chex.Array) -> chex.Array:
    shifted = jnp.zeros_like(mask)
    shifted = shifted.at[:, :-1].set(mask[:, 1:])
    return shifted


def shift_right(mask: chex.Array) -> chex.Array:
    shifted = jnp.zeros_like(mask)
    shifted = shifted.at[:, 1:].set(mask[:, :-1])
    return shifted


def make_adjacent_access_mask(
    env_map: chex.Array,
    object_tile_idx: int,
    passable_tiles: Tuple[int, ...],
) -> chex.Array:
    """
    object 타일 그 자체가 아니라,
    object 상하좌우 중 '서 있을 수 있는 칸(passable)'을 target으로 만든다.
    """
    object_mask = make_tile_mask(env_map, object_tile_idx)
    passable_mask = make_passable_mask(env_map, passable_tiles)

    adjacent_mask = shift_up(object_mask)
    adjacent_mask = jnp.logical_or(adjacent_mask, shift_down(object_mask))
    adjacent_mask = jnp.logical_or(adjacent_mask, shift_left(object_mask))
    adjacent_mask = jnp.logical_or(adjacent_mask, shift_right(object_mask))

    access_mask = jnp.logical_and(adjacent_mask, passable_mask)
    return access_mask

def is_not_done(state: ReachabilityState):
    return jnp.logical_not(state.done)


def _expand_flood_once(flood_input: chex.Array) -> chex.Array:
    """
    flood_path_net 없이 map_validate.py 내부에서 직접 flood 1-step 확장.
    flood_input[..., 0] = occupied_map (1이면 막힘)
    flood_input[..., 1] = current_flood
    """
    occupied_map = flood_input[..., 0]
    current_flood = flood_input[..., 1]

    up = shift_up(current_flood)
    down = shift_down(current_flood)
    left = shift_left(current_flood)
    right = shift_right(current_flood)

    expanded = jnp.logical_or(current_flood > 0, up > 0)
    expanded = jnp.logical_or(expanded, down > 0)
    expanded = jnp.logical_or(expanded, left > 0)
    expanded = jnp.logical_or(expanded, right > 0)

    # 벽 타일은 flood 불가
    expanded = jnp.logical_and(expanded, occupied_map == 0)

    expanded = expanded.astype(jnp.float32)
    flood_out = jnp.stack([occupied_map, expanded], axis=-1)
    return flood_out


def flood_step_until_target(state: ReachabilityState) -> ReachabilityState:
    flood_input = state.flood_input
    flood_count = state.flood_count
    target_mask = state.target_mask

    flood_out = _expand_flood_once(flood_input)
    new_flood_count = flood_count + flood_out[..., -1]

    reached_target = jnp.any(jnp.logical_and(new_flood_count > 0, target_mask))
    no_change = jnp.all(flood_input == flood_out)
    done = jnp.logical_or(reached_target, no_change)

    return ReachabilityState(
        flood_input=flood_out,
        flood_count=new_flood_count,
        target_mask=target_mask,
        done=done,
    )


def is_target_reachable(
    env_map: chex.Array,
    start_mask: chex.Array,
    target_mask: chex.Array,
    passable_tiles: Tuple[int, ...],
):
    """
    start_mask 에서 시작해서 target_mask 중 하나라도 reachable 한지 검사한다.
    """
    passable_mask = make_passable_mask(env_map, passable_tiles)
    occupied_map = jnp.logical_not(passable_mask).astype(jnp.float32)

    start_mask = jnp.logical_and(start_mask, passable_mask)
    start_mask_f = start_mask.astype(jnp.float32)

    flood_input = jnp.stack([occupied_map, start_mask_f], axis=-1)

    init_state = ReachabilityState(
        flood_input=flood_input,
        flood_count=start_mask_f,
        target_mask=target_mask,
        done=jnp.array(False),
    )

    final_state = jax.lax.while_loop(
        cond_fun=is_not_done,
        body_fun=flood_step_until_target,
        init_val=init_state,
    )

    reachable = jnp.any(jnp.logical_and(final_state.flood_count > 0, target_mask))
    return reachable


def is_not_done(state: ReachabilityState):
    return jnp.logical_not(state.done)


def count_required_objects(env_map: chex.Array, object_to_index: dict):
    pot_count = jnp.sum(env_map == object_to_index["pot"])
    onion_count = jnp.sum(env_map == object_to_index["onion_pile"])
    plate_count = jnp.sum(env_map == object_to_index["plate_pile"])
    goal_count = jnp.sum(env_map == object_to_index["goal"])

    count_ok = (
        (pot_count > 0) &
        (onion_count > 0) &
        (plate_count > 0) &
        (goal_count > 0)
    )

    return pot_count, onion_count, plate_count, goal_count, count_ok


def validate_layout(
    maze_map: chex.Array,
    passable_tiles: Tuple[int, ...],
    object_to_index: dict,
    player_tiles: Tuple[int, ...],
) -> LayoutValidationResult:
    """
    1. 필수 object counting 검사
    2. player 시작 위치에서 각 object 옆 칸까지 갈 수 있는지 검사
    """
    pot_count, onion_count, plate_count, goal_count, count_ok = count_required_objects(
        maze_map,
        object_to_index,
    )

    start_mask = make_multi_tile_mask(maze_map, player_tiles)

    onion_access_mask = make_adjacent_access_mask(
        maze_map,
        object_to_index["onion_pile"],
        passable_tiles,
    )
    plate_access_mask = make_adjacent_access_mask(
        maze_map,
        object_to_index["plate_pile"],
        passable_tiles,
    )
    pot_access_mask = make_adjacent_access_mask(
        maze_map,
        object_to_index["pot"],
        passable_tiles,
    )
    goal_access_mask = make_adjacent_access_mask(
        maze_map,
        object_to_index["goal"],
        passable_tiles,
    )

    onion_reachable = is_target_reachable(
        maze_map,
        start_mask,
        onion_access_mask,
        passable_tiles,
    )
    plate_reachable = is_target_reachable(
        maze_map,
        start_mask,
        plate_access_mask,
        passable_tiles,
    )
    pot_reachable = is_target_reachable(
        maze_map,
        start_mask,
        pot_access_mask,
        passable_tiles,
    )
    goal_reachable = is_target_reachable(
        maze_map,
        start_mask,
        goal_access_mask,
        passable_tiles,
    )

    reachable_ok = (
        onion_reachable &
        plate_reachable &
        pot_reachable &
        goal_reachable
    )

    valid = count_ok & reachable_ok

    return LayoutValidationResult(
        pot_count=pot_count,
        onion_count=onion_count,
        plate_count=plate_count,
        goal_count=goal_count,
        count_ok=count_ok,
        onion_reachable=onion_reachable,
        plate_reachable=plate_reachable,
        pot_reachable=pot_reachable,
        goal_reachable=goal_reachable,
        reachable_ok=reachable_ok,
        valid=valid,
    )



def debug_print_layout_result(result):
    jax.debug.print(
        "[layout check] pot={p}, onion={o}, plate={pl}, goal={g}, "
        "count_ok={co}, onion_reachable={or_}, plate_reachable={pr}, "
        "pot_reachable={por}, goal_reachable={gr}, reachable_ok={ro}, valid={v}",
        p=result.pot_count,
        o=result.onion_count,
        pl=result.plate_count,
        g=result.goal_count,
        co=result.count_ok,
        or_=result.onion_reachable,
        pr=result.plate_reachable,
        por=result.pot_reachable,
        gr=result.goal_reachable,
        ro=result.reachable_ok,
        v=result.valid,
    )