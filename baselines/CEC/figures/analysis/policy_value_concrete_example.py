"""Build and evaluate visible policy-equivalent/value-distinct candidates.

Every state puts agent 0 directly in front of a pot containing two onions while
holding the third onion. A/B pairs share the ego orientation but otherwise come
from distinct maps. Checkpoints are evaluated only after pair construction.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict, deque
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "baselines/CEC"))

from environment_representation_probe import (  # noqa: E402
    MODEL_SPECS, checkpoint_path, load_config, load_params, parse_model_spec,
)


DIRECTIONS = ((0, -1), (0, 1), (1, 0), (-1, 0))
ACTION_NAMES = ("North", "South", "East", "West", "Stay", "Interact")
FAMILY_LABELS = {
    "asymm_advantages": "Asymmetric Advantages",
    "coord_ring": "Coordination Ring",
    "counter_circuit": "Counter Circuit",
    "forced_coord": "Forced Coordination",
    "cramped_room": "Cramped Room",
}


def js_divergence(left, right):
    midpoint = .5 * (left + right)
    left_mask, right_mask = left > 0, right > 0
    return float(.5 * (
        np.sum(left[left_mask] * np.log(left[left_mask] / midpoint[left_mask]))
        + np.sum(right[right_mask] * np.log(right[right_mask] / midpoint[right_mask]))
    ))


def neighbors(position, floor):
    x, y = position
    return [(x + dx, y + dy) for dx, dy in DIRECTIONS
            if (x + dx, y + dy) in floor]


def distances(floor, starts):
    result = {tuple(start): 0 for start in starts}
    queue = deque(result)
    while queue:
        current = queue.popleft()
        for nxt in neighbors(current, floor):
            if nxt not in result:
                result[nxt] = result[current] + 1
                queue.append(nxt)
    return result


def layout_record(layout, seed, family):
    return {
        "layout": {
            key: int(value) if key in ("height", "width")
            else np.asarray(value).tolist()
            for key, value in layout.items()
        },
        "map_seed": seed,
        "family": family,
    }


def instantiate(config, record, horizon):
    import jax
    import jax.numpy as jnp
    import jaxmarl
    from jaxmarl.environments.overcooked.common import (
        COLOR_TO_INDEX, DIR_TO_VEC, OBJECT_TO_INDEX,
    )

    layout = {
        key: value if key in ("height", "width") else jnp.asarray(value)
        for key, value in record["layout"].items()
    }
    kwargs = dict(config["ENV_KWARGS"])
    kwargs.update(layout=layout, random_reset=False, check_held_out=False,
                  shuffle_inv_and_pot=False, max_steps=horizon)
    env = jaxmarl.make("overcooked", **kwargs)
    _, state = env.custom_reset(
        jax.random.PRNGKey(record["map_seed"]), layout=layout,
        random_reset=False, shuffle_inv_and_pot=False,
    )
    floor = {
        (int(x), int(y)) for y, x in np.argwhere(~np.asarray(state.wall_map))
    }
    pot = tuple(map(int, np.asarray(state.pot_pos[0])))
    goals = [tuple(map(int, value)) for value in np.asarray(state.goal_pos)]
    pot_access = neighbors(pot, floor)
    goal_access = {cell for goal in goals for cell in neighbors(goal, floor)}
    if not pot_access or not goal_access:
        raise ValueError("Layout has no usable pot or serving access cell")

    ego = min(pot_access)
    direction_vector = (pot[0] - ego[0], pot[1] - ego[1])
    direction = DIRECTIONS.index(direction_vector)
    original_positions = [
        tuple(map(int, position)) for position in np.asarray(state.agent_pos)
    ]
    teammate_indices = [
        index for index, position in enumerate(original_positions)
        if position != ego
    ]
    if not teammate_indices:
        raise ValueError("No original teammate position distinct from ego")
    teammate_index = teammate_indices[0]
    teammate = original_positions[teammate_index]
    from_pot = distances(floor - {ego}, [cell for cell in pot_access if cell != ego])
    if teammate not in from_pot:
        raise ValueError("Original teammate position is not reachable from the pot")
    pot_to_goal = min(
        distances(floor, pot_access).get(cell, 10**6) for cell in goal_access
    )
    route_cost = int(from_pot[teammate] + pot_to_goal)

    pad = (state.maze_map.shape[0] - int(layout["height"])) // 2
    maze = state.maze_map
    empty = jnp.array([OBJECT_TO_INDEX["empty"], 0, 0], dtype=jnp.uint8)
    for x, y in np.asarray(state.agent_pos):
        maze = maze.at[pad + int(y), pad + int(x)].set(empty)
    # Preserve the environment state's exact dtypes. JAX scan requires every
    # carry leaf to have the same dtype before and after env.step_env().
    agent_positions = jnp.asarray(
        [ego, teammate], dtype=state.agent_pos.dtype
    )
    agent_directions = jnp.asarray([
        direction, state.agent_dir_idx[teammate_index]
    ], dtype=state.agent_dir_idx.dtype)
    for index, ((x, y), facing) in enumerate(zip(agent_positions, agent_directions)):
        agent = jnp.array([
            OBJECT_TO_INDEX["agent"], COLOR_TO_INDEX["red"] + index * 2,
            facing,
        ], dtype=jnp.uint8)
        maze = maze.at[pad + y, pad + x].set(agent)
    pot_x, pot_y = pot
    maze = maze.at[pad + pot_y, pad + pot_x, 2].set(
        jnp.asarray(21, dtype=maze.dtype)
    )  # two onions
    state = state.replace(
        agent_pos=agent_positions,
        agent_dir_idx=agent_directions,
        agent_dir=DIR_TO_VEC[agent_directions],
        agent_inv=jnp.asarray([
            OBJECT_TO_INDEX["onion"], state.agent_inv[teammate_index],
        ], dtype=state.agent_inv.dtype),
        maze_map=maze,
    )
    record = dict(record, ego=list(ego), teammate=list(teammate), pot=list(pot),
                  goals=[list(value) for value in goals], direction=direction,
                  route_cost=route_cost)
    return env, state, record


def make_policy_predictor(config, model, checkpoint):
    """Return a zero-history policy evaluator for controlled candidate states."""
    import jax
    import jax.numpy as jnp
    from actor_networks import ScannedRNN

    if model == "FCP":
        from actor_networks import ActorCriticRNN
        spec = {
            "value_network": ActorCriticRNN,
            "value_separate_hidden": False,
        }
    else:
        spec = MODEL_SPECS[model]
    network = spec["value_network"](6, config=config)
    params = load_params(checkpoint)
    hidden_dim = int(config["GRU_HIDDEN_DIM"])

    @jax.jit
    def predict(obs_batch, positions):
        hidden = ScannedRNN.initialize_carry(2, hidden_dim)
        if spec["value_separate_hidden"]:
            hidden = (hidden, hidden)
        _, policy, _ = network.apply(
            params, hidden,
            (obs_batch[None], jnp.zeros((1, 2), dtype=bool), positions[None]),
        )
        return policy.probs[0, 0]

    def predict_record(record, horizon):
        env, state, _ = instantiate(config, record, horizon)
        obs = env.get_obs(state)
        obs_batch = jnp.stack([obs[agent].reshape(-1) for agent in env.agents])
        return np.asarray(predict(obs_batch, state.agent_pos))

    return predict_record


def resolve_reference_checkpoint(args):
    if args.reference_checkpoint is not None:
        checkpoint = args.reference_checkpoint.expanduser()
    else:
        checkpoint = (
            args.model_root / "FCP" / f"{args.family}_9"
            / f"seed{args.reference_seed}"
            / f"fcp_seed{args.reference_seed}_best.pkl"
        )
    if not checkpoint.is_file():
        raise RuntimeError(f"FCP reference checkpoint was not found: {checkpoint}")
    return checkpoint


def generate_candidates(config, args):
    import jax
    from jaxmarl.environments.overcooked import layouts
    try:
        from tqdm.auto import tqdm
    except ImportError:
        def tqdm(iterable, **_kwargs):
            return iterable

    candidates = []
    generator = getattr(layouts, f"make_{args.family}_9x9")
    offsets = tqdm(
        range(args.map_candidates),
        total=args.map_candidates,
        desc=f"Generating {args.family} candidates",
        unit="map",
        dynamic_ncols=True,
    )
    for offset in offsets:
        seed = args.map_seed + offset
        base = layout_record(generator(jax.random.PRNGKey(seed), ik=True), seed, args.family)
        for variant in ("A", "B"):
            try:
                _, _, record = instantiate(
                    config, dict(base, variant=variant), args.horizon
                )
            except ValueError:
                continue
            candidates.append(record)
    if not candidates:
        raise RuntimeError("Could not construct a valid concrete state")
    return candidates


def choose_controlled_example(config, args):
    """Pair states that share only the requested immediate-state template."""
    candidates = generate_candidates(config, args)
    grouped = defaultdict(lambda: {"A": [], "B": []})
    for record in candidates:
        grouped[record["direction"]][record["variant"]].append(record)

    matches = []
    for direction, variants in grouped.items():
        states_a = sorted(variants["A"], key=lambda record: record["map_seed"])
        states_b = sorted(variants["B"], key=lambda record: record["map_seed"])
        if len(states_a) < 2 or len(states_b) < 2:
            continue
        for index, state_a in enumerate(states_a):
            state_b = states_b[(index + 1) % len(states_b)]
            if state_a["map_seed"] == state_b["map_seed"]:
                continue
            matches.append((
                state_a["map_seed"], state_b["map_seed"], direction,
                state_a, state_b,
            ))
    if not matches:
        raise RuntimeError(
            "No two maps produced the controlled state with the same ego "
            "orientation. Increase --map-candidates."
        )

    ordered = sorted(matches)
    selected_matches = []
    used_map_seeds = set()
    for match in ordered:
        seed_a, seed_b = match[0], match[1]
        if seed_a in used_map_seeds or seed_b in used_map_seeds:
            continue
        selected_matches.append(match)
        used_map_seeds.update((seed_a, seed_b))
        if len(selected_matches) == args.num_pairs:
            break
    if len(selected_matches) < args.num_pairs:
        selected_keys = {(match[0], match[1]) for match in selected_matches}
        for match in ordered:
            if (match[0], match[1]) in selected_keys:
                continue
            selected_matches.append(match)
            if len(selected_matches) == args.num_pairs:
                break

    selections = []
    for selected in selected_matches:
        state_a, state_b = selected[3], selected[4]
        selection = {
            "method": "controlled_semantic_state",
            "uses_model_outputs": False,
            "semantic_state": {
                "ego_inventory": "onion",
                "ego_adjacent_to_pot": True,
                "ego_facing_pot": True,
                "pot_onions": 2,
            },
            "same_orientation": state_a["direction"] == state_b["direction"],
            "route_cost_gap": state_b["route_cost"] - state_a["route_cost"],
        }
        selections.append(((state_a, state_b), selection))
    return selections


def choose_fcp_example(config, args):
    candidates = generate_candidates(config, args)

    predictor = make_policy_predictor(config, "FCP", args.reference_checkpoint)

    eligible = {"A": [], "B": []}
    interact_argmax_count = {"A": 0, "B": 0}
    max_interact_probability = {"A": 0.0, "B": 0.0}
    for record in candidates:
        probabilities = predictor(record, args.horizon)
        variant = record["variant"]
        interact_probability = float(probabilities[5])
        max_interact_probability[variant] = max(
            max_interact_probability[variant], interact_probability
        )
        if int(probabilities.argmax()) == 5:
            interact_argmax_count[variant] += 1
        if (int(probabilities.argmax()) == 5
                and float(probabilities[5])
                >= args.selection_min_interact_probability):
            eligible[variant].append((record, probabilities))

    matches = []
    for easy, easy_probs in eligible["A"]:
        for hard, hard_probs in eligible["B"]:
            if easy["map_seed"] == hard["map_seed"]:
                continue
            route_gap = hard["route_cost"] - easy["route_cost"]
            if route_gap < args.min_route_cost_gap:
                continue
            divergence = js_divergence(easy_probs, hard_probs)
            if divergence > args.selection_max_policy_js:
                continue
            matches.append((
                -route_gap, divergence,
                easy["map_seed"], hard["map_seed"],
                easy, hard, easy_probs, hard_probs,
            ))
    if not matches:
        raise RuntimeError(
            "No concrete pair satisfied the FCP reference policy filter. "
            f"Eligible individual states: A={len(eligible['A'])}, "
            f"B={len(eligible['B'])}. Interact-argmax states: "
            f"A={interact_argmax_count['A']}, B={interact_argmax_count['B']}; "
            f"maximum Interact probabilities: "
            f"A={max_interact_probability['A']:.4f}, "
            f"B={max_interact_probability['B']:.4f}. Increase "
            "--map-candidates, choose another --reference-seed, or set "
            "--selection-min-interact-probability no higher than the reported "
            "maximum. The Interact-argmax requirement remains active."
        )
    selections = []
    for selected in sorted(matches)[:args.num_pairs]:
        divergence = selected[1]
        easy, hard = selected[4], selected[5]
        easy_probs, hard_probs = selected[6], selected[7]
        selection = {
            "method": "fcp_policy_filter",
            "uses_model_outputs": True,
            "policy": "FCP",
            "seed": args.reference_seed,
            "checkpoint": str(args.reference_checkpoint),
            "max_policy_js": args.selection_max_policy_js,
            "min_interact_probability": args.selection_min_interact_probability,
            "min_route_cost_gap": args.min_route_cost_gap,
            "route_cost_gap": hard["route_cost"] - easy["route_cost"],
            "policy_metrics": {
                "probabilities_a": easy_probs.tolist(),
                "probabilities_b": hard_probs.tolist(),
                "interact_a": float(easy_probs[5]),
                "interact_b": float(hard_probs[5]),
                "js_nats": divergence,
            },
        }
        selections.append(((easy, hard), selection))
    return selections


def draw_example(records, output_dir, pair_id):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle, Rectangle

    figure, axes = plt.subplots(1, 2, figsize=(9.4, 4.5))
    for ax, record in zip(axes, records):
        layout = record["layout"]
        width, height = int(layout["width"]), int(layout["height"])
        def positions(key):
            return {(int(i) % width, int(i) // width) for i in layout[key]}
        walls = positions("wall_idx")
        for y in range(height):
            for x in range(width):
                ax.add_patch(Rectangle(
                    (x, y), 1, 1,
                    facecolor="#707070" if (x, y) in walls else "#f7f7f7",
                    edgecolor="#4a4a4a", linewidth=.45,
                ))
        for key, color, label in (
            ("goal_idx", "#20df36", "Serve"),
            ("onion_pile_idx", "#ffe600", "Onion"),
            ("plate_pile_idx", "white", "Plate"),
            ("pot_idx", "#1b1b1b", "Pot\n2/3"),
        ):
            for x, y in positions(key):
                ax.add_patch(Rectangle((x + .08, y + .08), .84, .84,
                    facecolor=color, edgecolor="black", linewidth=.6))
                ax.text(x + .5, y + .5, label, ha="center", va="center",
                        fontsize=6.5, color="white" if key == "pot_idx" else "black")
        for position, color, label in (
            (record["ego"], "#d62728", "Ego\nOnion"),
            (record["teammate"], "#2455d6", "Mate"),
        ):
            x, y = position
            ax.add_patch(Circle((x + .5, y + .5), .35, facecolor=color,
                                edgecolor="black", linewidth=.8, zorder=5))
            ax.text(x + .5, y + .5, label, ha="center", va="center",
                    fontsize=6.5, color="white", fontweight="bold", zorder=6)
        ego_x, ego_y = record["ego"]
        pot_x, pot_y = record["pot"]
        ax.annotate("", xy=(pot_x + .5, pot_y + .5),
                    xytext=(ego_x + .5, ego_y + .5),
                    arrowprops=dict(arrowstyle="->", color="#d62728", lw=2))
        ax.set(xlim=(0, width), ylim=(height, 0), aspect="equal")
        ax.axis("off")
        ax.set_title(
            f"Environment {record['variant']}\n"
            f"{FAMILY_LABELS.get(record['family'], record['family'])}, "
            f"route cost={record['route_cost']}",
            fontweight="bold",
        )
    figure.suptitle(
        "Controlled immediate semantic-state pair",
        fontweight="bold",
    )
    figure.tight_layout()
    figure.savefig(
        output_dir / f"concrete_state_pair_{pair_id:02d}.png",
        dpi=300, bbox_inches="tight"
    )
    figure.canvas.draw()
    state_image = np.asarray(figure.canvas.buffer_rgba()).copy()
    plt.close(figure)
    return state_image


def cosine_distance(left, right):
    denominator = np.linalg.norm(left) * np.linalg.norm(right)
    return float(1 - np.dot(left, right) / denominator) if denominator else float("nan")


def euclidean_distance(left, right):
    return float(np.linalg.norm(left - right))


def rms_euclidean_distance(left, right):
    """Euclidean distance normalized by the square root of feature count."""
    return euclidean_distance(left, right) / np.sqrt(left.size)


def linear_cka(left, right):
    """Linear centered-kernel alignment for two sample-aligned matrices."""
    left = left - left.mean(axis=0, keepdims=True)
    right = right - right.mean(axis=0, keepdims=True)
    cross = np.linalg.norm(left.T @ right, ord="fro") ** 2
    left_norm = np.linalg.norm(left.T @ left, ord="fro")
    right_norm = np.linalg.norm(right.T @ right, ord="fro")
    denominator = left_norm * right_norm
    return float(cross / denominator) if denominator > 0 else float("nan")


def average_ranks(values):
    """Return one-based average ranks, including ties, without SciPy."""
    values = np.asarray(values)
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=np.float64)
    start = 0
    while start < len(values):
        stop = start + 1
        while stop < len(values) and values[order[stop]] == values[order[start]]:
            stop += 1
        ranks[order[start:stop]] = .5 * (start + stop - 1) + 1
        start = stop
    return ranks


def spearman_rsa(left, right):
    """Spearman correlation between two vectorized dissimilarity matrices."""
    left, right = np.asarray(left), np.asarray(right)
    valid = np.isfinite(left) & np.isfinite(right)
    if valid.sum() < 2:
        return float("nan")
    left_rank = average_ranks(left[valid])
    right_rank = average_ranks(right[valid])
    if left_rank.std() == 0 or right_rank.std() == 0:
        return float("nan")
    return float(np.corrcoef(left_rank, right_rank)[0, 1])


def upper_triangle(matrix):
    indices = np.triu_indices(matrix.shape[0], k=1)
    return np.asarray(matrix)[indices]


def cosine_rdm(features):
    """Pairwise cosine-distance representational dissimilarity matrix."""
    features = np.asarray(features, dtype=np.float64)
    norms = np.linalg.norm(features, axis=1)
    denominator = np.outer(norms, norms)
    similarities = np.divide(
        features @ features.T, denominator,
        out=np.full_like(denominator, np.nan), where=denominator > 0,
    )
    result = 1 - np.clip(similarities, -1, 1)
    np.fill_diagonal(result, 0)
    return result


def zscore_features(features):
    features = np.asarray(features, dtype=np.float64)
    std = features.std(axis=0)
    safe_std = np.where(std > 1e-8, std, 1.0)
    return (features - features.mean(axis=0)) / safe_std


def rms_rdm(features):
    """Pairwise RMS Euclidean distance matrix for already standardized features."""
    features = np.asarray(features, dtype=np.float64)
    differences = features[:, None, :] - features[None, :, :]
    return np.sqrt(np.mean(np.square(differences), axis=-1))


def js_rdm(probabilities):
    """Pairwise Jensen-Shannon divergence matrix in nats."""
    probabilities = np.asarray(probabilities, dtype=np.float64)
    count = len(probabilities)
    result = np.zeros((count, count), dtype=np.float64)
    for left_index in range(count):
        for right_index in range(left_index + 1, count):
            distance = js_divergence(
                probabilities[left_index], probabilities[right_index]
            )
            result[left_index, right_index] = distance
            result[right_index, left_index] = distance
    return result


def compute_rsa_rows(rows):
    """Compute checkpoint-wise RSA using all fixed A/B states in a layout."""
    grouped = defaultdict(list)
    for row in rows:
        grouped[(row["model"], row["num_envs"], row["seed"])].append(row)

    summaries = []
    for (model, num_envs, seed), group in grouped.items():
        group = sorted(group, key=lambda row: row["pair_id"])
        policy_a = np.stack([row["_policy_rep_a"] for row in group])
        policy_b = np.stack([row["_policy_rep_b"] for row in group])
        value_a = np.stack([row["_value_rep_a"] for row in group])
        value_b = np.stack([row["_value_rep_b"] for row in group])
        policy = np.concatenate((policy_a, policy_b), axis=0)
        value = np.concatenate((value_a, value_b), axis=0)
        probabilities = np.asarray([
            [row[f"prob_{action.lower()}_a"] for action in ACTION_NAMES]
            for row in group
        ] + [
            [row[f"prob_{action.lower()}_b"] for action in ACTION_NAMES]
            for row in group
        ])
        returns = np.asarray(
            [row["mc_return_a"] for row in group]
            + [row["mc_return_b"] for row in group],
            dtype=np.float64,
        )
        behavior_vector = upper_triangle(js_rdm(probabilities))
        return_vector = upper_triangle(np.abs(returns[:, None] - returns[None, :]))

        z_policy = zscore_features(policy)
        z_value = zscore_features(value)
        count = len(group)
        rdms = {
            "cosine": (cosine_rdm(policy), cosine_rdm(value)),
            "zscored_rms": (rms_rdm(z_policy), rms_rdm(z_value)),
        }
        for distance_metric, (policy_rdm, value_rdm) in rdms.items():
            policy_vector = upper_triangle(policy_rdm)
            value_vector = upper_triangle(value_rdm)
            policy_behavior = spearman_rsa(policy_vector, behavior_vector)
            policy_return = spearman_rsa(policy_vector, return_vector)
            value_behavior = spearman_rsa(value_vector, behavior_vector)
            value_return = spearman_rsa(value_vector, return_vector)
            policy_selectivity = policy_behavior - policy_return
            value_selectivity = value_return - value_behavior
            summaries.append({
                "model": model,
                "num_envs": num_envs,
                "seed": seed,
                "pairs": count,
                "states": len(policy),
                "rdm_entries": len(policy_vector),
                "distance_metric": distance_metric,
                "correlation": "spearman",
                "behavior_distance": "jensen_shannon_nats",
                "return_distance": "absolute_mc_return_difference",
                "policy_value_rsa": spearman_rsa(policy_vector, value_vector),
                "policy_a_b_rsa": spearman_rsa(
                    upper_triangle(policy_rdm[:count, :count]),
                    upper_triangle(policy_rdm[count:, count:]),
                ),
                "value_a_b_rsa": spearman_rsa(
                    upper_triangle(value_rdm[:count, :count]),
                    upper_triangle(value_rdm[count:, count:]),
                ),
                "policy_behavior_rsa": policy_behavior,
                "policy_return_rsa": policy_return,
                "value_behavior_rsa": value_behavior,
                "value_return_rsa": value_return,
                "policy_behavior_minus_return": policy_selectivity,
                "value_return_minus_behavior": value_selectivity,
                "rsa_asymmetry_score": policy_selectivity + value_selectivity,
            })
    return summaries


def compute_cka_rows(rows):
    """Compute per-checkpoint CKA over the fixed, pair-aligned state set."""
    grouped = defaultdict(list)
    for row in rows:
        grouped[(row["model"], row["num_envs"], row["seed"])].append(row)

    summaries = []
    for (model, num_envs, seed), group in grouped.items():
        group = sorted(group, key=lambda row: row["pair_id"])
        policy_a = np.stack([row["_policy_rep_a"] for row in group])
        policy_b = np.stack([row["_policy_rep_b"] for row in group])
        value_a = np.stack([row["_value_rep_a"] for row in group])
        value_b = np.stack([row["_value_rep_b"] for row in group])
        policy = np.concatenate((policy_a, policy_b), axis=0)
        value = np.concatenate((value_a, value_b), axis=0)
        returns = np.asarray(
            [row["mc_return_a"] for row in group]
            + [row["mc_return_b"] for row in group]
        )[:, None]
        summaries.append({
            "model": model,
            "num_envs": num_envs,
            "seed": seed,
            "pairs": len(group),
            "states": len(policy),
            "policy_rep_dim": policy.shape[1],
            "value_rep_dim": value.shape[1],
            "policy_a_b_linear_cka": linear_cka(policy_a, policy_b),
            "value_a_b_linear_cka": linear_cka(value_a, value_b),
            "policy_value_linear_cka": linear_cka(policy, value),
            "policy_return_linear_cka": linear_cka(policy, returns),
            "value_return_linear_cka": linear_cka(value, returns),
        })
    return summaries


def add_dataset_normalized_distances(rows):
    """Add per-checkpoint z-scored distances and remove temporary features."""
    grouped = defaultdict(list)
    for row in rows:
        grouped[(row["model"], row["num_envs"], row["seed"])].append(row)

    for group in grouped.values():
        for prefix in ("policy", "value"):
            features = np.stack([
                row[f"_{prefix}_rep_{variant}"]
                for row in group for variant in ("a", "b")
            ])
            mean = features.mean(axis=0)
            std = features.std(axis=0)
            safe_std = np.where(std > 1e-8, std, 1.0)
            for row in group:
                normalized_a = (row[f"_{prefix}_rep_a"] - mean) / safe_std
                normalized_b = (row[f"_{prefix}_rep_b"] - mean) / safe_std
                row[f"{prefix}_rep_zscored_euclidean_distance"] = (
                    euclidean_distance(normalized_a, normalized_b)
                )
                row[f"{prefix}_rep_zscored_rms_distance"] = (
                    rms_euclidean_distance(normalized_a, normalized_b)
                )

    for row in rows:
        for prefix in ("policy", "value"):
            del row[f"_{prefix}_rep_a"]
            del row[f"_{prefix}_rep_b"]


def draw_checkpoint_reports(rows, output_dir, state_images):
    """Create one directly inspectable A/B report for every checkpoint."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    report_dir = output_dir / "checkpoint_visualizations"
    report_dir.mkdir(parents=True, exist_ok=True)
    action_x = np.arange(len(ACTION_NAMES))
    for row in rows:
        figure = plt.figure(figsize=(10.5, 8.0))
        grid = figure.add_gridspec(3, 2, height_ratios=(1.7, 1, .75), hspace=.38)
        map_axis = figure.add_subplot(grid[0, :])
        map_axis.imshow(state_images[row["pair_id"]])
        map_axis.axis("off")

        action_axis = figure.add_subplot(grid[1, 0])
        probabilities_a = np.asarray([
            row[f"prob_{name.lower()}_a"] for name in ACTION_NAMES
        ])
        probabilities_b = np.asarray([
            row[f"prob_{name.lower()}_b"] for name in ACTION_NAMES
        ])
        width = .38
        action_axis.bar(action_x - width / 2, probabilities_a, width,
                        label="Environment A", color="#56B4E9")
        action_axis.bar(action_x + width / 2, probabilities_b, width,
                        label="Environment B", color="#0072B2")
        action_axis.set_xticks(action_x)
        action_axis.set_xticklabels(ACTION_NAMES, rotation=30, ha="right")
        action_axis.set_ylim(0, 1)
        action_axis.set_ylabel("Action probability")
        action_axis.set_title("Policy at the concrete state", fontweight="bold")
        action_axis.grid(axis="y", alpha=.25)
        action_axis.legend(frameon=False, fontsize=8)

        value_axis = figure.add_subplot(grid[1, 1])
        positions = np.arange(2)
        value_axis.bar(positions - width / 2, [
            row["predicted_value_a"], row["predicted_value_b"]
        ], width, label="Predicted value", color="#e3a21a")
        value_axis.bar(positions + width / 2, [
            row["mc_return_a"], row["mc_return_b"]
        ], width, label="MC return", color="#117733")
        value_axis.set_xticks(positions)
        value_axis.set_xticklabels(("Environment A", "Environment B"))
        value_axis.set_title("Value and realized return", fontweight="bold")
        value_axis.grid(axis="y", alpha=.25)
        value_axis.legend(frameon=False, fontsize=8)

        text_axis = figure.add_subplot(grid[2, :])
        text_axis.axis("off")
        status = "PASS" if row["passes_concrete_example"] else "DOES NOT PASS"
        details = (
            f"Policy JS: {row['policy_js_nats']:.4f}   |   "
            f"argmax: A={row['argmax_a']}, B={row['argmax_b']}   |   "
            f"Interact: A={row['interact_probability_a']:.3f}, "
            f"B={row['interact_probability_b']:.3f}\n"
            f"Predicted value A−B: {row['predicted_value_delta_a_minus_b']:.3f}   |   "
            f"MC return A−B: {row['mc_return_delta_a_minus_b']:.3f} "
            f"± {row['mc_return_delta_sem']:.3f}   |   "
            f"Rep cosine (policy/value): "
            f"{row['policy_rep_cosine_distance']:.4f} / "
            f"{row['value_rep_cosine_distance']:.4f}\n"
            f"Rep raw L2 (policy/value): "
            f"{row['policy_rep_raw_euclidean_distance']:.4f} / "
            f"{row['value_rep_raw_euclidean_distance']:.4f}   |   "
            f"z-scored RMS (policy/value): "
            f"{row['policy_rep_zscored_rms_distance']:.4f} / "
            f"{row['value_rep_zscored_rms_distance']:.4f}\n"
            f"Automatic filter: {status}"
        )
        text_axis.text(
            .5, .5, details, ha="center", va="center", fontsize=10,
            bbox=dict(boxstyle="round,pad=.6", facecolor="#f2f2f2",
                      edgecolor="#555555"),
        )
        label = "DCEC" if row["model"] == "CEC_IDAAC" else row["model"]
        figure.suptitle(
            f"Pair {row['pair_id']:02d} · {label} ({row['num_envs']}) · "
            f"seed {row['seed']}",
            fontsize=15, fontweight="bold",
        )
        stem = report_dir / (
            f"pair{row['pair_id']:02d}_{label.lower()}_"
            f"{row['num_envs']}_seed{row['seed']}"
        )
        figure.savefig(stem.with_suffix(".png"), dpi=220, bbox_inches="tight")
        plt.close(figure)


def evaluate_checkpoint(
    config, records, pair_id, model, num_envs, seed, checkpoint, args
):
    import jax
    import jax.numpy as jnp
    from actor_networks import ScannedRNN

    spec = MODEL_SPECS[model]
    params = load_params(checkpoint)
    network = spec["value_network"](6, config=config)
    hidden_dim = int(config["GRU_HIDDEN_DIM"])

    def carry():
        hidden = ScannedRNN.initialize_carry(2, hidden_dim)
        return (hidden, hidden) if spec["value_separate_hidden"] else hidden

    evaluated = {}
    for record in records:
        env, initial, _ = instantiate(config, record, args.horizon)
        obs = env.get_obs(initial)
        obs_batch = jnp.stack([obs[a].reshape(-1) for a in env.agents])
        inputs = (obs_batch[None], jnp.zeros((1, 2), dtype=bool),
                  initial.agent_pos[None])
        (_, policy, value), captured = network.apply(
            params, carry(), inputs, mutable=["intermediates"]
        )
        probabilities = np.asarray(policy.probs[0, 0])
        intermediate = captured["intermediates"]
        policy_rep = np.asarray(
            intermediate["actor_penultimate"][0][0, 0]
        )
        value_rep = np.asarray(
            intermediate["critic_penultimate"][0][0, 0]
        )
        policy_source = "actor_penultimate"
        value_source = "critic_penultimate"

        def rollout(key):
            def step(carry_value, time):
                state, hidden, done, key = carry_value
                obs = env.get_obs(state)
                obs_batch = jnp.stack([obs[a].reshape(-1) for a in env.agents])
                inputs = (obs_batch[None], done[None], state.agent_pos[None])
                hidden, pi, _ = network.apply(params, hidden, inputs)
                key, action_key, step_key = jax.random.split(key, 3)
                actions = pi.sample(seed=action_key)[0]
                action_dict = {a: actions[i] for i, a in enumerate(env.agents)}
                _, next_state, reward, dones, _ = env.step_env(
                    step_key, state, action_dict
                )
                finished = jnp.all(done)
                next_done = jnp.asarray([dones[a] for a in env.agents])
                reward_value = jnp.where(finished, 0.0, reward["agent_0"])
                return (next_state, hidden, next_done, key), args.gamma**time * reward_value
            _, rewards = jax.lax.scan(
                step, (initial, carry(), jnp.zeros(2, dtype=bool), key),
                jnp.arange(args.horizon),
            )
            return rewards.sum()

        returns = np.asarray(jax.jit(jax.vmap(rollout))(
            jax.random.split(jax.random.PRNGKey(args.rollout_seed), args.rollouts)
        ))
        evaluated[record["variant"]] = {
            "probabilities": probabilities,
            "predicted_value": float(value[0, 0]),
            "policy_rep": policy_rep,
            "value_rep": value_rep,
            "policy_rep_source": policy_source,
            "value_rep_source": value_source,
            "mc_return": float(returns.mean()),
            "mc_returns": returns,
        }

    a, b = evaluated["A"], evaluated["B"]
    midpoint = .5 * (a["probabilities"] + b["probabilities"])
    def kl(probabilities):
        mask = probabilities > 0
        return np.sum(probabilities[mask] * np.log(probabilities[mask] / midpoint[mask]))
    delta = a["mc_returns"] - b["mc_returns"]
    row = {
        "pair_id": pair_id,
        "map_seed_a": records[0]["map_seed"],
        "map_seed_b": records[1]["map_seed"],
        "route_cost_a": records[0]["route_cost"],
        "route_cost_b": records[1]["route_cost"],
        "model": model, "num_envs": num_envs, "seed": seed,
        "checkpoint": str(checkpoint),
        "used_for_pair_selection": False,
        "policy_rep_source": a["policy_rep_source"],
        "value_rep_source": a["value_rep_source"],
        "policy_js_nats": float(.5 * (kl(a["probabilities"]) + kl(b["probabilities"]))),
        "argmax_a": ACTION_NAMES[int(a["probabilities"].argmax())],
        "argmax_b": ACTION_NAMES[int(b["probabilities"].argmax())],
        "argmax_same": bool(a["probabilities"].argmax() == b["probabilities"].argmax()),
        "interact_probability_a": float(a["probabilities"][5]),
        "interact_probability_b": float(b["probabilities"][5]),
        "predicted_value_a": a["predicted_value"],
        "predicted_value_b": b["predicted_value"],
        "predicted_value_delta_a_minus_b": a["predicted_value"] - b["predicted_value"],
        "mc_return_a": a["mc_return"], "mc_return_b": b["mc_return"],
        "mc_return_delta_a_minus_b": float(delta.mean()),
        "mc_return_delta_sem": float(delta.std(ddof=1) / np.sqrt(len(delta))),
        "policy_rep_cosine_distance": cosine_distance(a["policy_rep"], b["policy_rep"]),
        "value_rep_cosine_distance": cosine_distance(a["value_rep"], b["value_rep"]),
        "policy_rep_raw_euclidean_distance": euclidean_distance(
            a["policy_rep"], b["policy_rep"]
        ),
        "value_rep_raw_euclidean_distance": euclidean_distance(
            a["value_rep"], b["value_rep"]
        ),
        "policy_rep_raw_rms_distance": rms_euclidean_distance(
            a["policy_rep"], b["policy_rep"]
        ),
        "value_rep_raw_rms_distance": rms_euclidean_distance(
            a["value_rep"], b["value_rep"]
        ),
        "policy_rep_dim": int(a["policy_rep"].size),
        "value_rep_dim": int(a["value_rep"].size),
        "_policy_rep_a": a["policy_rep"],
        "_policy_rep_b": b["policy_rep"],
        "_value_rep_a": a["value_rep"],
        "_value_rep_b": b["value_rep"],
    }
    row["passes_policy_equivalence"] = bool(
        row["policy_js_nats"] < args.max_policy_js and row["argmax_same"]
    )
    row["passes_intended_interaction"] = bool(
        row["argmax_a"] == "Interact" and row["argmax_b"] == "Interact"
        and row["interact_probability_a"] >= args.min_interact_probability
        and row["interact_probability_b"] >= args.min_interact_probability
    )
    row["passes_return_distinction"] = bool(
        abs(row["mc_return_delta_a_minus_b"]) >= args.min_abs_return_delta
    )
    row["passes_concrete_example"] = bool(
        row["passes_policy_equivalence"]
        and row["passes_intended_interaction"]
        and row["passes_return_distinction"]
    )
    for variant, result in evaluated.items():
        for action, probability in zip(ACTION_NAMES, result["probabilities"]):
            row[f"prob_{action.lower()}_{variant.lower()}"] = float(probability)
    return row


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-root", type=Path, required=True)
    parser.add_argument("--models", nargs="+", type=parse_model_spec,
                        default=[("CEC", 64), ("CEC_IDAAC", 64)])
    parser.add_argument("--seeds", type=int, nargs="+", default=list(range(6)))
    parser.add_argument("--reference-checkpoint", type=Path,
                        help="FCP checkpoint; default is inferred from model root/family/seed")
    parser.add_argument("--reference-seed", type=int, default=0)
    parser.add_argument("--pair-selection", choices=("controlled", "fcp"),
                        default="controlled")
    parser.add_argument("--selection-max-policy-js", type=float, default=.2)
    parser.add_argument("--selection-min-interact-probability", type=float,
                        default=.6)
    parser.add_argument("--min-route-cost-gap", type=int, default=2)
    parser.add_argument("--num-pairs", type=int)
    parser.add_argument("--family", choices=tuple(FAMILY_LABELS),
                        default="counter_circuit")
    parser.add_argument("--map-seed", type=int, default=1701)
    parser.add_argument("--map-candidates", type=int)
    parser.add_argument("--large-scale", action="store_true")
    parser.add_argument("--horizon", type=int, default=200)
    parser.add_argument("--rollouts", type=int, default=100)
    parser.add_argument("--rollout-seed", type=int, default=2701)
    parser.add_argument("--gamma", type=float, default=.99)
    parser.add_argument("--max-policy-js", type=float, default=.15)
    parser.add_argument("--min-interact-probability", type=float, default=.7)
    parser.add_argument("--min-abs-return-delta", type=float, default=1.0)
    parser.add_argument(
        "--config", type=Path,
        default=ROOT / "baselines/CEC_UED/config/ippo_overcooked_CEC_gradient.yaml",
    )
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args()
    args.model_root = args.model_root.expanduser()
    if args.map_candidates is None:
        args.map_candidates = 3000 if args.large_scale else 1000
    if args.num_pairs is None:
        args.num_pairs = 50 if args.large_scale else 10
    if args.pair_selection == "fcp":
        args.reference_checkpoint = resolve_reference_checkpoint(args)
    default_output_name = (
        "policy_value_large_scale"
        if args.large_scale else "policy_value_concrete_example"
    )
    output_root = args.output_dir or (
        args.model_root / "analysis" / default_output_name
    )
    args.output_dir = output_root / args.family
    args.output_dir.mkdir(parents=True, exist_ok=True)
    config = load_config(args.config)
    if args.pair_selection == "controlled":
        selected_pairs = choose_controlled_example(config, args)
    else:
        selected_pairs = choose_fcp_example(config, args)
    print(f"Selected {len(selected_pairs)} pair(s)")

    serialized_pairs = []
    state_images = {}
    for pair_id, (records, pair_selection) in enumerate(selected_pairs):
        print(
            f"Pair {pair_id:02d}: A seed={records[0]['map_seed']}, "
            f"B seed={records[1]['map_seed']}, "
            f"route-cost gap={pair_selection['route_cost_gap']}"
        )
        if pair_selection["method"] == "fcp_policy_filter":
            policy = pair_selection["policy_metrics"]
            print(
                f"  FCP: JS={policy['js_nats']:.6f}, "
                f"Interact(A)={policy['interact_a']:.4f}, "
                f"Interact(B)={policy['interact_b']:.4f}"
            )
        serialized_pairs.append({
            "pair_id": pair_id,
            "states": records,
            "pair_selection": pair_selection,
        })
        state_images[pair_id] = draw_example(records, args.output_dir, pair_id)
    (args.output_dir / "concrete_state_pairs.json").write_text(
        json.dumps({"pairs": serialized_pairs}, indent=2),
        encoding="utf-8",
    )
    if args.prepare_only:
        print(f"Saved {len(selected_pairs)} concrete pairs to {args.output_dir}")
        return
    rows = []
    for model, num_envs in args.models:
        for seed in args.seeds:
            checkpoint = checkpoint_path(args.model_root, model, num_envs, seed)
            if checkpoint is None:
                print(f"Missing checkpoint: {model} {num_envs} seed {seed}")
                continue
            for pair_id, (records, _) in enumerate(selected_pairs):
                rows.append(evaluate_checkpoint(
                    config, records, pair_id, model, num_envs, seed,
                    checkpoint, args
                ))
            print(
                f"Evaluated {model} {num_envs} seed {seed} on "
                f"{len(selected_pairs)} pairs"
            )
    if not rows:
        raise RuntimeError("No usable checkpoint was found")
    cka_rows = compute_cka_rows(rows)
    rsa_rows = compute_rsa_rows(rows)
    add_dataset_normalized_distances(rows)
    with (args.output_dir / "concrete_example_metrics.csv").open(
        "w", newline="", encoding="utf-8"
    ) as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    with (args.output_dir / "concrete_example_cka.csv").open(
        "w", newline="", encoding="utf-8"
    ) as file:
        writer = csv.DictWriter(file, fieldnames=list(cka_rows[0]))
        writer.writeheader()
        writer.writerows(cka_rows)
    with (args.output_dir / "concrete_example_rsa.csv").open(
        "w", newline="", encoding="utf-8"
    ) as file:
        writer = csv.DictWriter(file, fieldnames=list(rsa_rows[0]))
        writer.writeheader()
        writer.writerows(rsa_rows)
    model_filtered = [row for row in rows if row["passes_concrete_example"]]
    with (args.output_dir / "concrete_example_model_filtered.csv").open(
        "w", newline="", encoding="utf-8"
    ) as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(model_filtered)

    requested_specs = set(args.models)
    joint_groups = defaultdict(list)
    for row in rows:
        joint_groups[(row["pair_id"], row["seed"])].append(row)
    filtered = []
    for group in joint_groups.values():
        by_spec = {(row["model"], row["num_envs"]): row for row in group}
        if not requested_specs.issubset(by_spec):
            continue
        if all(by_spec[spec]["passes_concrete_example"] for spec in requested_specs):
            filtered.extend(by_spec[spec] for spec in args.models)
    with (args.output_dir / "concrete_example_filtered.csv").open(
        "w", newline="", encoding="utf-8"
    ) as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(filtered)

    pair_groups = defaultdict(list)
    for row in rows:
        pair_groups[(row["model"], row["num_envs"], row["pair_id"])].append(row)
    pair_summary = []
    for (model, num_envs, pair_id), group in pair_groups.items():
        pair_summary.append({
            "model": model,
            "num_envs": num_envs,
            "pair_id": pair_id,
            "map_seed_a": group[0]["map_seed_a"],
            "map_seed_b": group[0]["map_seed_b"],
            "route_cost_gap": group[0]["route_cost_b"] - group[0]["route_cost_a"],
            "seeds": len(group),
            "policy_equivalence_rate": np.mean([
                row["passes_policy_equivalence"] for row in group
            ]),
            "intended_interaction_rate": np.mean([
                row["passes_intended_interaction"] for row in group
            ]),
            "return_distinction_rate": np.mean([
                row["passes_return_distinction"] for row in group
            ]),
            "full_example_pass_rate": np.mean([
                row["passes_concrete_example"] for row in group
            ]),
            "mean_policy_js_nats": np.mean([
                row["policy_js_nats"] for row in group
            ]),
            "mean_abs_mc_return_delta": np.mean([
                abs(row["mc_return_delta_a_minus_b"]) for row in group
            ]),
            "mean_policy_rep_distance": np.mean([
                row["policy_rep_cosine_distance"] for row in group
            ]),
            "mean_value_rep_distance": np.mean([
                row["value_rep_cosine_distance"] for row in group
            ]),
            "mean_policy_rep_raw_euclidean_distance": np.mean([
                row["policy_rep_raw_euclidean_distance"] for row in group
            ]),
            "mean_value_rep_raw_euclidean_distance": np.mean([
                row["value_rep_raw_euclidean_distance"] for row in group
            ]),
            "mean_policy_rep_zscored_euclidean_distance": np.mean([
                row["policy_rep_zscored_euclidean_distance"] for row in group
            ]),
            "mean_value_rep_zscored_euclidean_distance": np.mean([
                row["value_rep_zscored_euclidean_distance"] for row in group
            ]),
            "mean_policy_rep_zscored_rms_distance": np.mean([
                row["policy_rep_zscored_rms_distance"] for row in group
            ]),
            "mean_value_rep_zscored_rms_distance": np.mean([
                row["value_rep_zscored_rms_distance"] for row in group
            ]),
        })
    with (args.output_dir / "concrete_example_pair_summary.csv").open(
        "w", newline="", encoding="utf-8"
    ) as file:
        writer = csv.DictWriter(file, fieldnames=list(pair_summary[0]))
        writer.writeheader()
        writer.writerows(pair_summary)

    grouped = defaultdict(list)
    for row in rows:
        grouped[(row["model"], row["num_envs"])].append(row)
    summary = []
    for (model, num_envs), group in grouped.items():
        summary.append({
            "model": model,
            "num_envs": num_envs,
            "seeds": len({row["seed"] for row in group}),
            "pairs": len({row["pair_id"] for row in group}),
            "evaluations": len(group),
            "policy_equivalence_rate": np.mean([
                row["passes_policy_equivalence"] for row in group
            ]),
            "intended_interaction_rate": np.mean([
                row["passes_intended_interaction"] for row in group
            ]),
            "return_distinction_rate": np.mean([
                row["passes_return_distinction"] for row in group
            ]),
            "full_example_pass_rate": np.mean([
                row["passes_concrete_example"] for row in group
            ]),
            "mean_policy_js_nats": np.mean([
                row["policy_js_nats"] for row in group
            ]),
            "mean_abs_mc_return_delta": np.mean([
                abs(row["mc_return_delta_a_minus_b"]) for row in group
            ]),
            "mean_policy_rep_distance": np.mean([
                row["policy_rep_cosine_distance"] for row in group
            ]),
            "mean_value_rep_distance": np.mean([
                row["value_rep_cosine_distance"] for row in group
            ]),
            "mean_policy_rep_raw_euclidean_distance": np.mean([
                row["policy_rep_raw_euclidean_distance"] for row in group
            ]),
            "mean_value_rep_raw_euclidean_distance": np.mean([
                row["value_rep_raw_euclidean_distance"] for row in group
            ]),
            "mean_policy_rep_zscored_euclidean_distance": np.mean([
                row["policy_rep_zscored_euclidean_distance"] for row in group
            ]),
            "mean_value_rep_zscored_euclidean_distance": np.mean([
                row["value_rep_zscored_euclidean_distance"] for row in group
            ]),
            "mean_policy_rep_zscored_rms_distance": np.mean([
                row["policy_rep_zscored_rms_distance"] for row in group
            ]),
            "mean_value_rep_zscored_rms_distance": np.mean([
                row["value_rep_zscored_rms_distance"] for row in group
            ]),
        })
    if summary:
        with (args.output_dir / "concrete_example_summary.csv").open(
            "w", newline="", encoding="utf-8"
        ) as file:
            writer = csv.DictWriter(file, fieldnames=list(summary[0]))
            writer.writeheader()
            writer.writerows(summary)
    if not args.large_scale:
        draw_checkpoint_reports(rows, args.output_dir, state_images)
    else:
        print("Skipping checkpoint PNG visualizations in large-scale mode")
    print(f"Saved concrete pairs and metrics to {args.output_dir}")


if __name__ == "__main__":
    main()
