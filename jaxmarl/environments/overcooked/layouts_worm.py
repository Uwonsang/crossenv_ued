import jax.numpy as jnp
import jax
import numpy as np
from flax.core.frozen_dict import FrozenDict
import pdb
cramped_room = {
    "height" : 4,
    "width" : 5,
    "wall_idx" : jnp.array([0,1,2,3,4,
                            5,9,
                            10,14,
                            15,16,17,18,19]),
    "agent_idx" : jnp.array([6, 8]),
    "goal_idx" : jnp.array([18, 18]),
    "plate_pile_idx" : jnp.array([16, 16]),
    "onion_pile_idx" : jnp.array([5,9]),
    "pot_idx" : jnp.array([2, 2]),
    "layout_name": "cramped_room"
}

asymm_advantages = {
    "height" : 5,
    "width" : 9,
    "wall_idx" : jnp.array([0,1,2,3,4,5,6,7,8,
                            9,11,12,13,14,15,17,
                            18,22,26,
                            27,31,35,
                            36,37,38,39,40,41,42,43,44]),
    "agent_idx" : jnp.array([29, 32]),
    "goal_idx" : jnp.array([12,17]),
    "plate_pile_idx" : jnp.array([39,41]),
    "onion_pile_idx" : jnp.array([9,14]),
    "pot_idx" : jnp.array([22,31]),
    "layout_name": "asymm_advantages"
}
coord_ring = {
    "height" : 5,
    "width" : 5,
    "wall_idx" : jnp.array([0,1,2,3,4,
                            5,9,
                            10,12,14,
                            15,19,
                            20,21,22,23,24]),
    "agent_idx" : jnp.array([7, 11]),
    "goal_idx" : jnp.array([22, 22]),
    "plate_pile_idx" : jnp.array([10, 10]),
    "onion_pile_idx" : jnp.array([15,21]),
    "pot_idx" : jnp.array([3,9]),
    "layout_name": "coord_ring"
}
forced_coord = {
    "height" : 5,
    "width" : 5,
    "wall_idx" : jnp.array([0,1,2,3,4,
                            5,7,9,
                            10,12,14,
                            15,17,19,
                            20,21,22,23,24]),
    "agent_idx" : jnp.array([11,8]),
    "goal_idx" : jnp.array([23, 23]),
    "onion_pile_idx" : jnp.array([5,10]),
    "plate_pile_idx" : jnp.array([15, 15]),
    "pot_idx" : jnp.array([3,9]),
    "layout_name": "forced_coord"
}

# Example of layout provided as a grid
counter_circuit_grid = """
WWWPPWWX
W A    W
B WWWW X
W     AW
BWWOOWWW
"""

# squeezed_room_drawn = {
#     "height" : 7,
#     "width" : 7,
#     "wall_idx": jnp.array([0,1,2,3,4,5,6,
#                             7,8,9,10,11,12,13,
#                             14,15,16,17,18,19,20,
#                             21,22,23,24,25,26,27,
#                             28,29,30,31,32,33,34,
#                             35,36,37,38,39,40,41,
#                             42,43,44,45,46,47,48]),
#     "agent_idx": jnp.array([8, 40]),
#     "goal_idx": jnp.array([1,7]),
#     "plate_pile_idx": jnp.array([5,13]),
#     "onion_pile_idx": jnp.array([35,43]),
#     "pot_idx": jnp.array([41,47]),
#     "layout_name": "squeezed_room_drawn"
# }
squeezed_room_manual = {
    "height" : 7,
    "width" : 7,
    "wall_idx": jnp.array([0,1,2,3,4,5,6,
                            7,13,
                            14,20,
                            21,27,
                            28,34,
                            35,41,
                            42,43,44,45,46,47,48]),
    "agent_idx": jnp.array([8, 40]),
    "goal_idx": jnp.array([1,7]),
    "plate_pile_idx": jnp.array([5,13]),
    "onion_pile_idx": jnp.array([35,43]),
    "pot_idx": jnp.array([41,47]),
    "layout_name": "squeezed_room_drawn"
}

squeezed_room_drawn = """
WWWWWWW
WWWWWWW
WWWWWWW
WXWWWPW
XA W  P
O  W AB
WWOWBWW
"""





# cramped_room_padded = """
# WXWWWPW
# XA    P
# W     W
# W     W
# W     W
# O    AB
# WOWWWBW
# """

# counter_circuit_padded = """
# BWPWPWX
# WA    W
# B     X
# W WWW W
# W    AW
# W     W
# WWOWOWW
# """

counter_circuit_padded = """
XWWPWWP
W  A  W
W     W
B WWW X
W     W
W  A  W
BWWOWWO
"""

harder_counter_circuit_padded = """
XWWPWWP
W  A  W
W W W W
B WWW X
W W W W
W  A  W
BWWOWWO
"""

coord_ring_padded = """
WWPWPWW
WA A  W
B WWW W
W WWW W
W WWW W
B     W
WOOWXXW
"""

forced_coord_padded = """
WWWWWWP
OA W  P
O  W  W
W  W  W
B  W AX
B  W  X
WWWWWWW
"""

asymm_advantages_padded = """
WXWWWWW
W   A P
B     O
WWWWWWW
B     X
W A   P
WOWWWWW
"""


cramped_room_padded = """
WWWPWWP
OA    O
W     W
W     W
W     W
W    AW
BBWWWXX
"""



coord_ring_padded = """
BWWWWPW
WA A  P
W WWW W
B WWW W
W WWW W
O     W
WOWXWWX
"""
# coord_ring_padded = """
# BWWWWPW
# WA A  P
# W WWW W
# W WWW W
# B WWW W
# O     W
# WOWXWWX
# """
# coord_ring_padded = """
# BWWWWPW
# WA A  P
# W WWW W
# W WWW W
# B WWW W
# O     W
# WOWWXWX
# """
# coord_ring_padded = """
# WWWXWWW
# W     W
# W WWW W
# W WXW W
# W W W W
# BA   AB
# WOPWPOW
# """

forced_coord_padded = """
WWWWWWW
OA W  P
O  W  P
W  W  W
B  W AW
B  W  W
WWWWXXW
"""

asymm_advantages_padded = """
WXWWWWW
W   A W
B     O
WWWPWPW
B     X
W A   W
WOWWWWW
"""

figure_ates = """
WWWWWWX
W     W
OAOWW W
P     X
BABWW W
P     W
WWWWWWW
"""

hallway_halting = """
WWWXWWW
W     W
W WWW W
W WXW W
W W W W
BA   AB
WOPWPOW
"""

column_control = """
WWWWWWW
W  W  X
W  W  X
P  W  O
P AW  B
W   A O
WWWWWBW
"""


def layout_grid_to_dict(grid, layout_name="counter_circuit_grid"):
    """Assumes `grid` is string representation of the layout, with 1 line per row, and the following symbols:
    W: wall
    A: agent
    X: goal
    B: plate (bowl) pile
    O: onion pile
    P: pot location
    ' ' (space) : empty cell
    """

    rows = grid.split('\n')

    if len(rows[0]) == 0:
        rows = rows[1:]
    if len(rows[-1]) == 0:
        rows = rows[:-1]

    keys = ["wall_idx", "agent_idx", "goal_idx", "plate_pile_idx", "onion_pile_idx", "pot_idx"]
    symbol_to_key = {"W" : "wall_idx",
                     "A" : "agent_idx",
                     "X" : "goal_idx",
                     "B" : "plate_pile_idx",
                     "O" : "onion_pile_idx",
                     "P" : "pot_idx"}

    layout_dict = {key : [] for key in keys}
    layout_dict["height"] = len(rows)
    layout_dict["width"] = len(rows[0])
    width = len(rows[0])

    for i, row in enumerate(rows):
        for j, obj in enumerate(row):
            idx = width * i + j
            if obj in symbol_to_key.keys():
                # Add object
                layout_dict[symbol_to_key[obj]].append(idx)
            if obj in ["X", "B", "O", "P"]:
                # These objects are also walls technically
                layout_dict["wall_idx"].append(idx)
            elif obj == " ":
                # Empty cell
                continue

    for key in symbol_to_key.values():
        # Transform lists to arrays
        layout_dict[key] = jnp.array(layout_dict[key])
    layout_dict["layout_name"] = layout_name

    return FrozenDict(layout_dict)


def initialize_border(height, width):
    '''
    makes a border around the grid
    returns list of lists of shape (height, width)
    '''
    grid = [[' ']*width for _ in range(height)]
    for h in range(height):
        grid[h][0] = 'W'
        grid[h][width-1] = 'W'
    for w in range(width):
        grid[0][w] = 'W'
        grid[height-1][w] = 'W'
    return grid

def initialize_items(grid, items):
    '''
    assumes grid is a list of lists of shape (height, width), with only the border covered in walls
    items is a dictionary mapping symbol --> 
        max num_items --> int
        item prob --> list
    '''
    total_to_sample = 0
    symbol_array = []
    for item_symbol in items:  # aggregate number of symbols
        max_num_item, item_prob = items[item_symbol]['max_items'], items[item_symbol]['item_prob']
        if item_prob is not None:
            assert len(item_prob) == max_num_item, f"Provided probabilities are not the same as maximum number of items for item {item_symbol}"
        num_items = np.random.choice(np.arange(1, max_num_item + 1), p=item_prob)
        total_to_sample += num_items
        symbol_array += [item_symbol] * num_items
    
    wall_idxes = get_walls(grid)
    assert total_to_sample <= len(wall_idxes), f"total to sample {total_to_sample}; num_walls {len(wall_idxes)}"
    sampled_ids = np.random.choice(len(wall_idxes), size=total_to_sample, replace=False)
    for i, list_id in enumerate(sampled_ids):  # substitute symbols
        h, w = wall_idxes[list_id]
        grid[h][w] = symbol_array[i]
    return grid

def initialize_agents(grid, one_frozen=False):
    height, width = len(grid), len(grid[0])
    if not one_frozen:
        sampled_heights = np.random.choice(np.arange(1, height-1), size=2, replace=False)
        sampled_widths = np.random.choice(np.arange(1, width-1), size=2, replace=False)
        for (h, w) in zip(sampled_heights, sampled_widths):
            grid[h][w] = 'A'
    else:
        # h, w = np.random.choice(np.arange(1, height-1)), np.random.choice(np.arange(1, width-1))
        # grid[h][w] = 'A'
        # grid[0][0] = 'A'  # stick one agent in a corner for now
        sampled_heights = np.random.choice(np.arange(1, height-1), size=2, replace=False)
        sampled_widths = np.random.choice(np.arange(1, width-1), size=2, replace=False)
        for (h, w) in zip(sampled_heights, sampled_widths):
            grid[h][w] = 'A'
    return grid

def is_wall(grid, i, j):
    '''
    Returns true if the grid location is a wall
    '''
    return grid[i][j] == 'W'


def get_walls(grid):
    '''
    returns a set of all (h,w) pairs that have a wall
    '''
    height, width = len(grid), len(grid[0])
    wall_idxes = {(h, w) for h, row in enumerate(grid) for w, cell in enumerate(row) if cell == 'W'}
    corners = {(0,0), (0, width-1), (height-1, 0), (height-1, width-1)}
    wall_idxes = wall_idxes - corners

    return list(wall_idxes)


def print_grid(grid):
    for row in grid:
        print(''.join(row))


def init_item_dict(max_num_items, item_prob=None):
    return {'max_items': max_num_items, 'item_prob': item_prob}


def sample_overcooked_grid(min_height=7, max_height=7, min_width=7, max_width=7, one_frozen=False):
    height = np.random.choice(np.arange(min_height, max_height+1))
    width = np.random.choice(np.arange(min_width, max_width+1))
    grid = initialize_border(height, width)

    if not one_frozen:
        items_to_init = {
            'P': init_item_dict(max_num_items=2, item_prob=[0.67, 0.33]),
            'X': init_item_dict(max_num_items=2, item_prob=[0.8, 0.2]),
            'O': init_item_dict(max_num_items=3, item_prob=[0.0, 0.8, 0.2]),
            'B': init_item_dict(max_num_items=2, item_prob=[0.67, 0.33]),
        }
    else:
        items_to_init = {
            'P': init_item_dict(max_num_items=2, item_prob=[1.0, 0.0]),
            'X': init_item_dict(max_num_items=2, item_prob=[1.0, 0.0]),
            'O': init_item_dict(max_num_items=3, item_prob=[0.0, 1.0, 0.0]),
            'B': init_item_dict(max_num_items=2, item_prob=[1.0, 0.0]),
        }

    grid = initialize_items(grid, items_to_init)
    grid = initialize_agents(grid, one_frozen=False)
    
    grid_string = '\n'.join([''.join(row) for row in grid])

    return grid_string

def layout_array_to_dict(grid, layout_name="array_layout", num_pots=2, num_plates=2, num_onions=2, num_goals=2, num_agents=2, num_base_walls=None):
    """Converts a jax.numpy array representation of a layout to dictionary format.
    Assumes the following encoding:
    0: free space
    1: wall
    2: agent
    3: goal
    4: plate (bowl) pile
    5: onion pile
    6: pot location
    
    Args:
        grid: jnp.array of shape (height, width) containing integers 0-5
        layout_name: string name for the layout
    """
    height, width = grid.shape
    
    # Create indices array
    idx_grid = jnp.arange(height * width).reshape(height, width)
    
    # Get indices for each object type using jnp.where
    if num_base_walls is None:
        num_base_walls = (2*height + 2*width - 4) - num_pots - num_plates - num_onions - num_goals
    else:
        num_base_walls = num_base_walls
    
        
    wall_idx = jnp.ravel(idx_grid)[jnp.where(jnp.ravel(grid) == 1, size=num_base_walls)[0]]
    agent_idx = jnp.ravel(idx_grid)[jnp.where(jnp.ravel(grid) == 2, size=num_agents)[0]]
    goal_idx = jnp.ravel(idx_grid)[jnp.where(jnp.ravel(grid) == 3, size=num_goals)[0]]
    plate_pile_idx = jnp.ravel(idx_grid)[jnp.where(jnp.ravel(grid) == 4, size=num_plates)[0]]
    onion_pile_idx = jnp.ravel(idx_grid)[jnp.where(jnp.ravel(grid) == 5, size=num_onions)[0]]
    pot_idx = jnp.ravel(idx_grid)[jnp.where(jnp.ravel(grid) == 6, size=num_pots)[0]]
    
    # Add additional wall indices for objects that are also walls
    wall_idx = jnp.concatenate([wall_idx, goal_idx, plate_pile_idx, onion_pile_idx, pot_idx])

    corner_indices = jnp.array([
        [0, 0],
        [0, width-1],
        [height-1, 0],
        [height-1, width-1]
    ])
    
    layout_dict = {
        "height": height,
        "width": width,
        "wall_idx": wall_idx,
        "agent_idx": agent_idx,
        "goal_idx": goal_idx,
        "plate_pile_idx": plate_pile_idx,
        "onion_pile_idx": onion_pile_idx,
        "pot_idx": pot_idx
    }
    
    return FrozenDict(layout_dict)



def single_cramped_room():
    return layout_grid_to_dict(sample_overcooked_grid(one_frozen=True), layout_name="single_cramped_room")


def make_9x9_layout(rng, layout_grid, rotate=False, num_base_walls=None):
    base_layout = jnp.ones((9, 9))
    def sub_in_default_cramped_room(target_layout, base_layout, num_rotations, to_flip, flip_axis):
        layout_height, layout_width = target_layout.shape
        def flip_zero(x):
            return jnp.flip(x, axis=0)
        def flip_one(x):
            return jnp.flip(x, axis=1)
        flip_func = lambda x, y: jnp.where(y==0, flip_zero(x), flip_one(x))
        target_layout = jnp.where(to_flip==1, flip_func(target_layout, flip_axis), target_layout)
        updated_layout = base_layout.at[:layout_height, :layout_width].set(target_layout)
        frozen_dict = layout_array_to_dict(updated_layout, num_base_walls=num_base_walls)
        return updated_layout, frozen_dict
    def sub_in_90_degree_rotation(target_layout, base_layout, num_rotations, to_flip, flip_axis):
        def get_effect(target_layout, base_layout, num_rotations, to_flip, flip_axis):
            rot_layout = jnp.rot90(target_layout)
            def flip_zero(x):
                return jnp.flip(x, axis=0)
            def flip_one(x):
                return jnp.flip(x, axis=1)
            flip_func = lambda x, y: jnp.where(y==0, flip_zero(x), flip_one(x))
            rot_layout = jnp.where(to_flip==1, flip_func(rot_layout, flip_axis), rot_layout)
            layout_height, layout_width = rot_layout.shape
            updated_layout = base_layout.at[:layout_height, :layout_width].set(rot_layout)
            frozen_dict = layout_array_to_dict(updated_layout, num_base_walls=num_base_walls)
            return updated_layout, frozen_dict
        updated_layout, frozen_dict = jax.lax.cond(num_rotations == 1, get_effect, sub_in_180_degree_rotation, target_layout, base_layout, num_rotations, to_flip, flip_axis)
        return updated_layout, frozen_dict
    def sub_in_180_degree_rotation(target_layout, base_layout, num_rotations, to_flip, flip_axis):
        def get_effect(target_layout, base_layout, num_rotations, to_flip, flip_axis):
            rot_layout = jnp.rot90(target_layout, k=2)
            def flip_zero(x):
                return jnp.flip(x, axis=0)
            def flip_one(x):
                return jnp.flip(x, axis=1)
            flip_func = lambda x, y: jnp.where(y==0, flip_zero(x), flip_one(x))
            rot_layout = jnp.where(to_flip==1, flip_func(rot_layout, flip_axis), rot_layout)
            layout_height, layout_width = rot_layout.shape
            updated_layout = base_layout.at[:layout_height, :layout_width].set(rot_layout)
            frozen_dict = layout_array_to_dict(updated_layout, num_base_walls=num_base_walls)
            return updated_layout, frozen_dict
        updated_layout, frozen_dict = jax.lax.cond(num_rotations == 2, get_effect, sub_in_270_degree_rotation, target_layout, base_layout, num_rotations, to_flip, flip_axis)
        return updated_layout, frozen_dict
    def sub_in_270_degree_rotation(target_layout, base_layout, num_rotations, to_flip, flip_axis):
        rot_layout = jnp.rot90(target_layout, k=3)
        def flip_zero(x):
            return jnp.flip(x, axis=0)
        def flip_one(x):
            return jnp.flip(x, axis=1)
        flip_func = lambda x, y: jnp.where(y==0, flip_zero(x), flip_one(x))
        rot_layout = jnp.where(to_flip==1, flip_func(rot_layout, flip_axis), rot_layout)
        layout_height, layout_width = rot_layout.shape
        updated_layout = base_layout.at[:layout_height, :layout_width].set(rot_layout)
        frozen_dict = layout_array_to_dict(updated_layout, num_base_walls=num_base_walls)
        return updated_layout, frozen_dict

    num_rotations = jax.random.randint(rng, (), 0, 4)
    num_rotations = jnp.where(rotate, num_rotations, 0)
    rng, rng_sub = jax.random.split(rng)
    to_flip = jax.random.randint(rng_sub, (), 0, 2)
    to_flip = jnp.where(rotate, to_flip, 0)
    rng, rng_sub = jax.random.split(rng)
    flip_axis = jax.random.randint(rng_sub, (), 0, 2)
    flip_axis = jnp.where(rotate, flip_axis, 0)

    updated_layout, layout_dict = jax.lax.cond(num_rotations == 0, sub_in_default_cramped_room, sub_in_90_degree_rotation, layout_grid, base_layout, num_rotations, to_flip, flip_axis)
    return layout_dict


# --------------------------------------------------
# worm preset
# easy가 더 자주 나오게 확률도 같이 둠
# --------------------------------------------------
WORM_PRESETS = {
    "easy": {
        "worm_init_count": 1,
        "worm_energy": 3,
        "worm_separate_enable": 0,
        "branch_prob": 0.00,
    },
    "normal": {
        "worm_init_count": 2,
        "worm_energy": 3,
        "worm_separate_enable": 1,
        "branch_prob": 0.10,
    },
    "hard": {
        "worm_init_count": 3,
        "worm_energy": 3,
        "worm_separate_enable": 1,
        "branch_prob": 0.20,
    },
}

# easy / normal / hard
WORM_DIFFICULTY_PROBS = jnp.array([0.45, 0.45, 0.10])

# JAX 안에서 동적 index로 쓰기 쉽게 배열로도 따로 빼둠
WORM_INIT_COUNTS = jnp.array([
    WORM_PRESETS["easy"]["worm_init_count"],
    WORM_PRESETS["normal"]["worm_init_count"],
    WORM_PRESETS["hard"]["worm_init_count"],
])

WORM_ENERGIES = jnp.array([
    WORM_PRESETS["easy"]["worm_energy"],
    WORM_PRESETS["normal"]["worm_energy"],
    WORM_PRESETS["hard"]["worm_energy"],
])

WORM_SEPARATE_ENABLES = jnp.array([
    WORM_PRESETS["easy"]["worm_separate_enable"],
    WORM_PRESETS["normal"]["worm_separate_enable"],
    WORM_PRESETS["hard"]["worm_separate_enable"],
])

WORM_BRANCH_PROBS = jnp.array([
    WORM_PRESETS["easy"]["branch_prob"],
    WORM_PRESETS["normal"]["branch_prob"],
    WORM_PRESETS["hard"]["branch_prob"],
])

ITEM_COUNTS = jnp.array([2, 2, 2])   # easy, normal, hard

# 4종류 object × 2개 = 8개 필요
MIN_COUNTERS_FOR_OBJECTS = 8

CAN_EXTEND_CARVE = False
CAN_PLACE_OBJECT_BEYOND_LAYOUT = False


# GLOBAL_LAYOUT_CONFIG = {
#     "can_extend_carve": False,
#     "worm_steps": 20,
#     "max_worms": 12,
#     "turn_prob": 0.18,
# }

# def layout_to_skeleton(layout, height=9, width=9):
#     """
#     Overcooked layout에서 wall(1)만 유지하고,
#     나머지 오브젝트/바닥은 전부 open(0)으로 바꾼 뒤 9x9 skeleton으로 올림.
#     """
#     orig_h, orig_w = layout.shape
#     layout_skeleton = jnp.where(layout == 1, 1, 0).astype(jnp.int32)

#     base_skeleton = jnp.ones((height, width), dtype=jnp.int32)
#     skeleton = base_skeleton.at[:orig_h, :orig_w].set(layout_skeleton)
#     return skeleton
def layout_to_skeleton(layout, height=9, width=9):
    """
    counter 성격의 타일(벽, goal, plate, onion, pot)은 모두 1로 유지하고,
    floor/agent만 0으로 둔다.
    """
    orig_h, orig_w = layout.shape

    wall_like = (
        (layout == 1) |  # wall
        (layout == 3) |  # goal
        (layout == 4) |  # plate pile
        (layout == 5) |  # onion pile
        (layout == 6)    # pot
    )

    layout_skeleton = wall_like.astype(jnp.int32)

    base_skeleton = jnp.ones((height, width), dtype=jnp.int32)
    skeleton = base_skeleton.at[:orig_h, :orig_w].set(layout_skeleton)
    return skeleton


def get_carve_bounds(orig_h, orig_w, height, width, can_extend_carve):
    """
    can_extend_carve=False:
        원본 layout 크기까지만 carve 허용
    can_extend_carve=True:
        9x9 전체 carve 허용
    """
    if can_extend_carve:
        return 0, height, 0, width
    else:
        return 0, orig_h, 0, orig_w

def flood_from_seed(open_mask, seed_idx):
    height, width = open_mask.shape
    sx = seed_idx // width
    sy = seed_idx % width

    seed_valid = open_mask[sx, sy]
    reach = jnp.zeros((height, width), dtype=bool).at[sx, sy].set(seed_valid)

    def step_fn(reach, _):
        up = jnp.pad(reach[1:, :], ((0, 1), (0, 0)))
        down = jnp.pad(reach[:-1, :], ((1, 0), (0, 0)))
        left = jnp.pad(reach[:, 1:], ((0, 0), (0, 1)))
        right = jnp.pad(reach[:, :-1], ((0, 0), (1, 0)))
        expanded = reach | ((up | down | left | right) & open_mask)
        return expanded, None

    reach, _ = jax.lax.scan(step_fn, reach, xs=None, length=height * width)
    return reach


def largest_open_component_mask(open_mask):
    height, width = open_mask.shape
    flat_open = open_mask.reshape(-1)
    all_idx = jnp.arange(height * width)

    def one_component(idx):
        comp = flood_from_seed(open_mask, idx)
        size = jnp.sum(comp.astype(jnp.int32))
        size = jnp.where(flat_open[idx], size, -1)
        return comp, size

    comps, sizes = jax.vmap(one_component)(all_idx)
    best_idx = jnp.argmax(sizes)
    return comps[best_idx]
    
def compute_reachability_and_counters(skeleton, orig_h, orig_w):
    """
    skeleton에서
    - largest reachable floor
    - reachable floor에 인접한 counter mask
    - orig 범위 내부 restricted counter mask
    를 한 번에 계산
    """
    open_mask_2d = (skeleton == 0)
    reachable_floor = largest_open_component_mask(open_mask_2d)
    reachable_floor_i = reachable_floor.astype(jnp.int32)

    up = jnp.pad(reachable_floor_i[1:, :], ((0, 1), (0, 0)))
    down = jnp.pad(reachable_floor_i[:-1, :], ((1, 0), (0, 0)))
    left = jnp.pad(reachable_floor_i[:, 1:], ((0, 0), (0, 1)))
    right = jnp.pad(reachable_floor_i[:, :-1], ((0, 0), (1, 0)))
    adj_reachable = (up | down | left | right)

    counter_mask = (skeleton == 1) & (adj_reachable == 1)

    restricted_counter_mask = jnp.zeros_like(counter_mask, dtype=bool)
    restricted_counter_mask = restricted_counter_mask.at[:orig_h, :orig_w].set(
        counter_mask[:orig_h, :orig_w]
    )

    counter_count = jnp.sum(restricted_counter_mask.astype(jnp.int32))

    return reachable_floor, counter_mask, restricted_counter_mask, counter_count

def count_placeable_counters(skeleton, orig_h, orig_w):
    _, _, _, counter_count = compute_reachability_and_counters(skeleton, orig_h, orig_w)
    return counter_count

def place_overcooked_objects(rng, skeleton, difficulty_idx, orig_h=9, orig_w=9, can_extend_object_place=False):
    height, width = skeleton.shape

    # 현재 설정에서는 난이도와 무관하게 항상 2개씩
    item_count = 2
    required_counter_slots = 8

    if can_extend_object_place:
        obj_min_x, obj_max_x, obj_min_y, obj_max_y = 0, height, 0, width
    else:
        obj_min_x, obj_max_x, obj_min_y, obj_max_y = 0, orig_h, 0, orig_w

    reachable_floor, counter_base_mask, _, _ = compute_reachability_and_counters(
        skeleton, orig_h, orig_w
    )

    counter_range_mask = jnp.zeros((height, width), dtype=bool)
    counter_range_mask = counter_range_mask.at[obj_min_x:obj_max_x, obj_min_y:obj_max_y].set(
        counter_base_mask[obj_min_x:obj_max_x, obj_min_y:obj_max_y]
    )
    counter_mask = counter_range_mask.reshape(-1)

    agent_range_mask = jnp.zeros((height, width), dtype=bool)
    agent_range_mask = agent_range_mask.at[obj_min_x:obj_max_x, obj_min_y:obj_max_y].set(
        reachable_floor[obj_min_x:obj_max_x, obj_min_y:obj_max_y]
    )
    agent_mask = agent_range_mask.reshape(-1)

    rng, r_counter, r_agent = jax.random.split(rng, 3)

    counter_scores = jnp.where(
        counter_mask,
        jax.random.gumbel(r_counter, counter_mask.shape),
        -jnp.inf
    )
    _, counter_idx = jax.lax.top_k(counter_scores, 8)
    counter_idx = counter_idx.reshape(4, 2)

    agent_scores = jnp.where(
        agent_mask,
        jax.random.gumbel(r_agent, agent_mask.shape),
        -jnp.inf
    )
    _, agent_idx = jax.lax.top_k(agent_scores, 2)

    modified_layout = skeleton

    def set_one(arr, flat_idx, value):
        x = flat_idx // width
        y = flat_idx % width
        return arr.at[x, y].set(value)

    def set_many(arr, flat_indices, value, count):
        def body(i, a):
            flat_idx = flat_indices[i]
            x = flat_idx // width
            y = flat_idx % width
            return a.at[x, y].set(value)
        return jax.lax.fori_loop(0, count, body, arr)

    plate_idx_all = counter_idx[0]
    onion_idx_all = counter_idx[1]
    pot_idx_all   = counter_idx[2]
    goal_idx_all  = counter_idx[3]

    modified_layout = set_many(modified_layout, plate_idx_all, 4, 2)
    modified_layout = set_many(modified_layout, onion_idx_all, 5, 2)
    modified_layout = set_many(modified_layout, pot_idx_all, 6, 2)
    modified_layout = set_many(modified_layout, goal_idx_all, 3, 2)

    modified_layout = set_one(modified_layout, agent_idx[0], 2)
    modified_layout = set_one(modified_layout, agent_idx[1], 2)

    return rng, modified_layout

def apply_worm_carving(
    rng,
    skeleton,
    orig_h,
    orig_w,
    can_extend_carve=False,
    can_extend_object_place=False,
    min_counters=MIN_COUNTERS_FOR_OBJECTS,
    worm_steps=20,
    max_worms=12,
    turn_prob=0.18,
):
    """
    skeleton 위에서 worm carving 수행.
    can_extend_carve=False 이면 원본 layout 범위(orig_h x orig_w) 안에서만 carve.
    can_extend_carve=True 이면 9x9 전체 범위에서 carve.
    
    can_extend_object_place=False이고 min_counters가 설정되면,
    carving 중 counter 개수가 min_counters 이하로 떨어지지 않도록 보호.
    """
    height, width = skeleton.shape

    carve_min_x, carve_max_x, carve_min_y, carve_max_y = get_carve_bounds(
        orig_h, orig_w, height, width, can_extend_carve
    )

    # 1) 난도 샘플링
    rng, r_diff = jax.random.split(rng)
    difficulty_idx = jax.random.choice(
        r_diff,
        jnp.arange(3),
        p=WORM_DIFFICULTY_PROBS
    )

    worm_init_count = WORM_INIT_COUNTS[difficulty_idx]
    worm_energy = WORM_ENERGIES[difficulty_idx]
    worm_separate_enable = WORM_SEPARATE_ENABLES[difficulty_idx].astype(bool)
    branch_prob = WORM_BRANCH_PROBS[difficulty_idx]

    # 2) 시작점 후보도 carve 가능 범위 안의 open cell만 허용
    candidate_open_mask_2d = jnp.zeros((height, width), dtype=bool)
    candidate_open_mask_2d = candidate_open_mask_2d.at[
        carve_min_x:carve_max_x, carve_min_y:carve_max_y
    ].set(
        skeleton[carve_min_x:carve_max_x, carve_min_y:carve_max_y] == 0
    )

    open_mask = candidate_open_mask_2d.reshape(-1)

    rng, r_start, r_dir = jax.random.split(rng, 3)
    start_scores = jnp.where(
        open_mask,
        jax.random.gumbel(r_start, open_mask.shape),
        -jnp.inf
    )
    _, start_idx = jax.lax.top_k(start_scores, max_worms)

    start_x = start_idx // width
    start_y = start_idx % width

    num_valid_open = jnp.sum(open_mask.astype(jnp.int32))
    active_worm_count = jnp.minimum(worm_init_count, num_valid_open)

    active = (jnp.arange(max_worms) < active_worm_count)
    worm_x = start_x
    worm_y = start_y
    worm_dir = jax.random.randint(r_dir, shape=(max_worms,), minval=0, maxval=4)
    worm_e = jnp.where(active, worm_energy, 0)

    def step_fn(carry, _):
        skeleton, active, worm_x, worm_y, worm_dir, worm_e, rng = carry

        def worm_body(i, inner_carry):
            skeleton, active, worm_x, worm_y, worm_dir, worm_e, rng = inner_carry
            rng, r_turn, r_lr, r_branch, r_branch_lr = jax.random.split(rng, 5)

            is_active = active[i]

            def do_active(_):
                x = worm_x[i]
                y = worm_y[i]
                d = worm_dir[i]
                e = worm_e[i]

                do_turn = jax.random.bernoulli(r_turn, turn_prob)
                turn_left = (d + 3) % 4
                turn_right = (d + 1) % 4

                turned = jax.lax.cond(
                    jax.random.bernoulli(r_lr, 0.5),
                    lambda _: turn_left,
                    lambda _: turn_right,
                    operand=None
                )

                nd = jax.lax.cond(
                    do_turn,
                    lambda _: turned,
                    lambda _: d,
                    operand=None
                )

                dx = jnp.array([-1, 0, 1, 0])[nd]
                dy = jnp.array([0, 1, 0, -1])[nd]

                nx = x + dx
                ny = y + dy

                # 핵심: carve 가능 범위를 전역 옵션으로 제어
                valid = (
                    (nx >= carve_min_x) & (nx < carve_max_x) &
                    (ny >= carve_min_y) & (ny < carve_max_y)
                )

                # 먼저 carve가 벽을 대상으로 하는지 확인
                is_wall = (skeleton[nx, ny] == 1)
                
                # carve 후 counter 개수 계산 (보호 로직용)
                skeleton_after_carve = skeleton.at[nx, ny].set(0)
    
                
                #remaining_counters = count_placeable_counters(skeleton_after_carve, orig_h, orig_w)
                #should_protect = (~can_extend_object_place) & (remaining_counters < min_counters)
                should_protect = False

                can_carve = valid & is_wall & ~should_protect

                skeleton2 = jax.lax.cond(
                    can_carve,
                    lambda s: s.at[nx, ny].set(0),
                    lambda s: s,
                    skeleton
                )

                new_x = jax.lax.cond(can_carve, lambda _: nx, lambda _: x, operand=None)
                new_y = jax.lax.cond(can_carve, lambda _: ny, lambda _: y, operand=None)

                # carve 성공했을 때만 energy 감소
                new_e = jax.lax.cond(can_carve, lambda _: e - 1, lambda _: e, operand=None)
                still_active = new_e > 0

                active2 = active.at[i].set(still_active)
                worm_x2 = worm_x.at[i].set(new_x)
                worm_y2 = worm_y.at[i].set(new_y)
                worm_dir2 = worm_dir.at[i].set(nd)
                worm_e2 = worm_e.at[i].set(jnp.maximum(new_e, 0))

                # branch
                can_branch = worm_separate_enable & still_active & (new_e >= 3)
                do_branch = can_branch & jax.random.bernoulli(r_branch, branch_prob)

                inactive_mask = ~active2
                slot = jnp.argmax(inactive_mask.astype(jnp.int32))
                has_slot = jnp.any(inactive_mask)

                branch_dir = jax.lax.cond(
                    jax.random.bernoulli(r_branch_lr, 0.5),
                    lambda _: (nd + 3) % 4,
                    lambda _: (nd + 1) % 4,
                    operand=None
                )
                branch_e = jnp.maximum(new_e // 2, 2)

                def spawn_branch(args):
                    active2, worm_x2, worm_y2, worm_dir2, worm_e2 = args
                    active2 = active2.at[slot].set(True)
                    worm_x2 = worm_x2.at[slot].set(new_x)
                    worm_y2 = worm_y2.at[slot].set(new_y)
                    worm_dir2 = worm_dir2.at[slot].set(branch_dir)
                    worm_e2 = worm_e2.at[slot].set(branch_e)
                    return active2, worm_x2, worm_y2, worm_dir2, worm_e2

                active2, worm_x2, worm_y2, worm_dir2, worm_e2 = jax.lax.cond(
                    do_branch & has_slot,
                    spawn_branch,
                    lambda args: args,
                    (active2, worm_x2, worm_y2, worm_dir2, worm_e2)
                )

                return skeleton2, active2, worm_x2, worm_y2, worm_dir2, worm_e2, rng

            return jax.lax.cond(
                is_active,
                do_active,
                lambda _: (skeleton, active, worm_x, worm_y, worm_dir, worm_e, rng),
                operand=None
            )

        skeleton, active, worm_x, worm_y, worm_dir, worm_e, rng = jax.lax.fori_loop(
            0,
            max_worms,
            worm_body,
            (skeleton, active, worm_x, worm_y, worm_dir, worm_e, rng)
        )

        return (skeleton, active, worm_x, worm_y, worm_dir, worm_e, rng), None

    (skeleton, _, _, _, _, _, rng), _ = jax.lax.scan(
        step_fn,
        (skeleton, active, worm_x, worm_y, worm_dir, worm_e, rng),
        xs=jnp.arange(worm_steps)
    )

    return rng, skeleton, difficulty_idx





@jax.jit
def make_cramped_room_9x9(rng, ik=False, num_default_walls=67):
    cramped_room_array = jnp.array([
        [6, 1, 6, 1, 1],
        [5, 2, 0, 2, 5],
        [1, 0, 0, 0, 1],
        [4, 4, 1, 3, 3],
    ])


    def default_cramped_room(rng, layout=cramped_room_array):
        return make_9x9_layout(rng, layout, rotate=False, num_base_walls=num_default_walls)

    def ik_cramped_room(rng, layout=cramped_room_array):
        orig_h, orig_w = layout.shape

        skeleton = layout_to_skeleton(layout, height=9, width=9)

        rng, skeleton, difficulty_idx = apply_worm_carving(
            rng,
            skeleton,
            orig_h,
            orig_w,
            can_extend_carve=CAN_EXTEND_CARVE,
            can_extend_object_place=CAN_PLACE_OBJECT_BEYOND_LAYOUT,
            min_counters=MIN_COUNTERS_FOR_OBJECTS,
            worm_steps=20,
            max_worms=12,
            turn_prob=0.18,
        )

        rng, modified_layout = place_overcooked_objects(rng, skeleton, difficulty_idx, orig_h, orig_w, CAN_PLACE_OBJECT_BEYOND_LAYOUT)

        return make_9x9_layout(
            rng,
            modified_layout,
            rotate=True,
            num_base_walls=num_default_walls
        )

    return jax.lax.cond(ik, ik_cramped_room, default_cramped_room, rng, cramped_room_array)

def calc_num_walls(layout):
    num_walls =jnp.where(layout == 1, 1, 0).sum()
    return 81 - (layout.shape[0] * layout.shape[1] - num_walls)

@jax.jit
def make_asymm_advantages_9x9(rng, ik=False, num_default_walls=59):
    # 14 walls by default means num of walls in 9x9 is 81 - (36-14) = 51
    asymm_advantages_array = jnp.array([
        [5, 0, 1, 3, 1, 5, 1, 0, 3],
        [1, 0, 2, 0, 6, 0, 2, 0, 1],
        [1, 0, 0, 0, 6, 0, 0, 0, 1],
        [1, 1, 1, 4, 1, 4, 1, 1, 1],
    ])

    height, width = asymm_advantages_array.shape

    def default_asymm_advantages(rng, layout=asymm_advantages_array):
        return make_9x9_layout(rng, layout, rotate=False, num_base_walls=num_default_walls)

    def ik_asymm_advantages(rng, layout=asymm_advantages_array):
        orig_h, orig_w = layout.shape

        skeleton = layout_to_skeleton(layout, height=9, width=9)

        rng, skeleton, difficulty_idx = apply_worm_carving(
            rng,
            skeleton,
            orig_h,
            orig_w,
            can_extend_carve=CAN_EXTEND_CARVE,
            can_extend_object_place=CAN_PLACE_OBJECT_BEYOND_LAYOUT,
            min_counters=MIN_COUNTERS_FOR_OBJECTS,
            worm_steps=20,
            max_worms=12,
            turn_prob=0.18,
        )

        rng, modified_layout = place_overcooked_objects(rng, skeleton, difficulty_idx, orig_h, orig_w, CAN_PLACE_OBJECT_BEYOND_LAYOUT)

        return make_9x9_layout(
            rng,
            modified_layout,
            rotate=True,
            num_base_walls=num_default_walls
        )
    
    return jax.lax.cond(ik, ik_asymm_advantages, default_asymm_advantages, rng, asymm_advantages_array)


import jax
import jax.numpy as jnp

@jax.jit
def make_coord_ring_9x9(rng, ik=False, num_default_walls=65):
    coord_ring_array = jnp.array([
        [4,1,1,6,1],
        [1,0,2,0,6],
        [4,0,1,0,1],
        [5,0,2,0,1],
        [1,5,3,1,3],
    ])

    def default_coord_ring(rng, layout=coord_ring_array):
        return make_9x9_layout(rng, layout, rotate=False, num_base_walls=num_default_walls)

    def ik_coord_ring(rng, layout=coord_ring_array):
        orig_h, orig_w = layout.shape

        skeleton = layout_to_skeleton(layout, height=9, width=9)

        rng, skeleton, difficulty_idx = apply_worm_carving(
            rng,
            skeleton,
            orig_h,
            orig_w,
            can_extend_carve=CAN_EXTEND_CARVE,
            can_extend_object_place=CAN_PLACE_OBJECT_BEYOND_LAYOUT,
            min_counters=MIN_COUNTERS_FOR_OBJECTS,
            worm_steps=20,
            max_worms=12,
            turn_prob=0.18,
        )

        rng, modified_layout = place_overcooked_objects(rng, skeleton, difficulty_idx, orig_h, orig_w, CAN_PLACE_OBJECT_BEYOND_LAYOUT)

        return make_9x9_layout(
            rng,
            modified_layout,
            rotate=True,
            num_base_walls=num_default_walls
        )

    return jax.lax.cond(ik, ik_coord_ring, default_coord_ring, rng, coord_ring_array)

    

@jax.jit
def make_forced_coord_9x9(rng, ik=False, num_default_walls=67):
    forced_coord_array = jnp.array([
        [1,1,1,6,1],
        [5,0,1,0,6],
        [5,2,1,2,1],
        [4,0,1,0,1],
        [4,1,1,3,3],
    ])

    def default_forced_coord(rng, layout=forced_coord_array):
        return make_9x9_layout(rng, layout, rotate=False, num_base_walls=num_default_walls)

    def ik_forced_coord(rng, layout=forced_coord_array):
        orig_h, orig_w = layout.shape

        skeleton = layout_to_skeleton(layout, height=9, width=9)

        rng, skeleton, difficulty_idx = apply_worm_carving(
            rng,
            skeleton,
            orig_h,
            orig_w,
            can_extend_carve=CAN_EXTEND_CARVE,
            can_extend_object_place=CAN_PLACE_OBJECT_BEYOND_LAYOUT,
            min_counters=MIN_COUNTERS_FOR_OBJECTS,
            worm_steps=20,
            max_worms=12,
            turn_prob=0.18,
        )

        rng, modified_layout = place_overcooked_objects(rng, skeleton, difficulty_idx, orig_h, orig_w, CAN_PLACE_OBJECT_BEYOND_LAYOUT)

        return make_9x9_layout(
            rng,
            modified_layout,
            rotate=True,
            num_base_walls=num_default_walls
        )

    return jax.lax.cond(ik, ik_forced_coord, default_forced_coord, rng, forced_coord_array)


@jax.jit
def make_counter_circuit_9x9(rng, ik=False, num_default_walls=59):
    counter_circuit_array = jnp.array([
        [4,1,1,6,6,1,1,3],
        [1,0,0,0,0,0,0,1],
        [4,2,1,1,1,1,2,3],
        [1,0,0,0,0,0,0,1],
        [1,1,1,5,5,1,1,1],
    ])

    def default_counter_circuit(rng, layout=counter_circuit_array):
        return make_9x9_layout(rng, layout, rotate=False, num_base_walls=num_default_walls)

    def ik_counter_circuit(rng, layout=counter_circuit_array):
        orig_h, orig_w = layout.shape

        skeleton = layout_to_skeleton(layout, height=9, width=9)

        rng, skeleton, difficulty_idx = apply_worm_carving(
            rng,
            skeleton,
            orig_h,
            orig_w,
            can_extend_carve=CAN_EXTEND_CARVE,
            can_extend_object_place=CAN_PLACE_OBJECT_BEYOND_LAYOUT,
            min_counters=MIN_COUNTERS_FOR_OBJECTS,
            worm_steps=20,
            max_worms=12,
            turn_prob=0.18,
        )

        rng, modified_layout = place_overcooked_objects(rng, skeleton, difficulty_idx, orig_h, orig_w, CAN_PLACE_OBJECT_BEYOND_LAYOUT)

        return make_9x9_layout(
            rng,
            modified_layout,
            rotate=True,
            num_base_walls=num_default_walls
        )

    return jax.lax.cond(
        ik,
        ik_counter_circuit,
        default_counter_circuit,
        rng,
        counter_circuit_array
    )


overcooked_layouts = {
    "cramped_room" : FrozenDict(cramped_room),
    "squeezed_room_drawn" : layout_grid_to_dict(squeezed_room_drawn),
    "asymm_advantages" : FrozenDict(asymm_advantages),
    "coord_ring" : FrozenDict(coord_ring),
    "forced_coord" : FrozenDict(forced_coord),
    "counter_circuit" : layout_grid_to_dict(counter_circuit_grid),
    "cramped_room_padded" : layout_grid_to_dict(cramped_room_padded),
    "counter_circuit_padded" : layout_grid_to_dict(counter_circuit_padded),
    "forced_coord_padded" : layout_grid_to_dict(forced_coord_padded),
    "asymm_advantages_padded" : layout_grid_to_dict(asymm_advantages_padded),
    "coord_ring_padded" : layout_grid_to_dict(coord_ring_padded),
    "figure_ates" : layout_grid_to_dict(figure_ates),
    "hallway_halting" : layout_grid_to_dict(hallway_halting),
    "column_control" : layout_grid_to_dict(column_control),
    'harder_counter_circuit': layout_grid_to_dict(harder_counter_circuit_padded),
    'cramped_room_9': make_cramped_room_9x9(jax.random.PRNGKey(0), ik=False),
    'asymm_advantages_9': make_asymm_advantages_9x9(jax.random.PRNGKey(0), ik=False),
    # object 개수 고정하고 실험 시, 아래 오른 쪽 주석 풀고 사용
    'coord_ring_9': make_coord_ring_9x9(jax.random.PRNGKey(0), ik=False), #'coord_ring_9': make_coord_ring_custom_ljy_9x9(jax.random.PRNGKey(0), ik=False),  
    'counter_circuit_9': make_counter_circuit_9x9(jax.random.PRNGKey(0), ik=False),
    'forced_coord_9': make_forced_coord_9x9(jax.random.PRNGKey(0), ik=False),
}


if __name__ == "__main__":

    rng = jax.random.PRNGKey(0)
    layout_dict = make_counter_circuit_9x9(rng, ik=True)
    print(layout_dict)

    # coord_ring_array = jnp.array([
    #     [4,1,1,6,1],
    #     [1,0,2,0,6],
    #     [4,0,1,0,1],
    #     [5,0,2,0,1],
    #     [1,5,3,1,3],
    # ])
    # new_layout, new_layout_dict=make_9x9_layout(rng, coord_ring_array, rotate=False)
    # print(new_layout_dict)
    # print(new_layout)