# Five-family variant asymmetry diagnostic

The workflow is split into three executable files:

- `policy_value_asymmetry_pairs.py`: generate, save, and visualize fixed state pairs.
- `policy_value_asymmetry.py`: evaluate checkpoints on those saved pairs.
- `policy_value_asymmetry_graph.py`: plot saved CSV results locally.

The rollout-based representation-geometry experiment uses:

- `policy_value_geometry.py`: collect reference trajectories, select
  policy-equivalent/value-distinct state pairs, and extract recurrent features.
- `policy_value_geometry_graph.py`: plot return-to-go difference against policy
  and value representation distance.
- `policy_value_concrete_example.py`: construct and visualize one simple
  onion-to-pot example, evaluate checkpoints, and filter successful results.

## Concrete visible example

This is the smallest diagnostic to run first. Both environments place agent 0
in front of a pot containing two onions while it holds the third onion. Agent 1
holds a plate. The maps are selected only from navigation geometry: A has a
short plate-to-pot-to-serving route and B has a long route. Model predictions
are not used to choose the maps.

```bash
python baselines/CEC/figures/analysis/policy_value_concrete_example.py \
  --model-root /mnt/nas/wonsang/crossenv_ued/models/ICRL \
  --models cec_64 dcec_64 \
  --seeds 0 1 2 3 4 5 \
  --family counter_circuit \
  --rollouts 100
```

The output directory is
`<model-root>/analysis/policy_value_concrete_example/`. It contains the A/B map
figure and JSON, all checkpoint metrics, rows passing the automatic filter, and
a seed-level model summary. The default filter requires policy JS at most 0.05,
Interact as both argmax actions with probability at least 0.5, and an absolute
Monte Carlo return gap of at least 1.0. These can be changed with
`--max-policy-js`, `--min-interact-probability`, and
`--min-abs-return-delta`.

The `checkpoint_visualizations/` subdirectory contains one PDF and PNG per
checkpoint. Each page shows the exact A/B maps, both six-action distributions,
predicted values, Monte Carlo returns, representation distances, and the
automatic filter decision. `concrete_state_pair.json` contains the complete
layout and controlled state coordinates for exact inspection or reconstruction.

This diagnostic uses the existing `make_*_9x9(..., ik=True)` generators for
Asymmetric Advantages, Coordination Ring, Counter Circuit, Forced Coordination,
and Cramped Room. It does not create a new corridor layout family.

## Prepare without checkpoints

Run from the repository root in the project's JAX environment:

```bash
python baselines/CEC/figures/analysis/policy_value_asymmetry_pairs.py \
  --candidates-per-family 500 \
  --pairs-per-family 5 \
  --output artifacts/policy_value_asymmetry/state_pairs.json
```

`state_pairs.json` stores the exact layouts, seeds, delivery plans, and matching
counts for every family. No checkpoint is loaded during preparation. The search
can find fewer pairs or zero pairs; missing families must be reported. Increasing
the candidate budget is possible, but do not silently weaken matching criteria.

Within a family, pairs have the same agent positions, partner position, fixed
orientation, and the same unique optimal first action for shortest delivery.
Their full `9x9x26` observations are intentionally different PCG layout
variants. The default required distance gap is two steps. These criteria are
independent of model predictions, and candidate states are not reused.

Agent 0 is explicitly given soup; agent 1 stays at its reset position. Empty
pots, time zero, and zero recurrent history are used. This is a controlled state
intervention, not a claim that states follow the training occupancy distribution.
The planner includes orientation and treats the partner as an obstacle. Its
objective is the first delivery, not the optimal full episode return.

## Generated-map visualization

Map visualization runs automatically immediately after `state_pairs.json` is
saved. The output directory defaults to a `map_visualizations` subdirectory
beside the JSON. For example:

```bash
python baselines/CEC/figures/analysis/policy_value_asymmetry_pairs.py \
  --candidates-per-family 500 \
  --pairs-per-family 5 \
  --output /app/nas/models/ICRL/analysis/policy_value_asymmetry/state_pairs.json
```

For this example, family-level PDF and PNG files are written under:

```text
/app/nas/models/ICRL/analysis/policy_value_asymmetry/map_visualizations/
```

Every row displays one short/long pair. It includes facilities, both agents,
the focal agent's soup marker, shortest route, path length, and map seed.

## Evaluate from a model root

The model root is expected to have the same structure used by the representation
probe: `MODEL/NUM_ENVS/seedN/seedN_ckpt*.pkl`. For example:

```bash
python baselines/CEC/figures/analysis/policy_value_asymmetry.py \
  --model-root /mnt/nas/wonsang/crossenv_ued/models/ICRL \
  --models cec_64 dcec_64 \
  --seeds 1 2 3 4 5 \
  --pairs artifacts/policy_value_asymmetry/state_pairs.json \
  --reference-model dcec_64 \
  --reference-seed 0 \
  --max-reference-policy-js 0.05 \
  --min-reference-mc-delta 0.5 \
  --rollouts 100
```

The latest checkpoint found for each requested seed is used. Results are saved
automatically under:

```text
<model-root>/analysis/policy_value_asymmetry/
```

Use `--output-dir` only when an explicit alternative is wanted.

## Evaluate with an explicit manifest

As an alternative, create a JSON manifest with checkpoint paths, e.g.:

```json
[
  {"model": "CEC", "num_envs": 64, "seed": 1, "checkpoint": "/path/to/cec/checkpoint.pkl"},
  {"model": "CEC_IDAAC", "num_envs": 64, "seed": 0, "checkpoint": "/path/to/reference/dcec/checkpoint.pkl"},
  {"model": "CEC_IDAAC", "num_envs": 64, "seed": 1, "checkpoint": "/path/to/dcec/checkpoint.pkl"}
]
```

```bash
python baselines/CEC/figures/analysis/policy_value_asymmetry.py \
  --pairs artifacts/policy_value_asymmetry/state_pairs.json \
  --checkpoints checkpoints.json --rollouts 100 \
  --output-dir artifacts/policy_value_asymmetry/evaluation
```

Use `--config` for the checkpoint-compatible architecture/configuration. Both
models must consume the same saved pairs. The default config is the existing
CEC gradient config. Before evaluation, each saved plan is checked against real
environment transitions and the expected discounted first-delivery reward.

Outputs:

- `paired_metrics_all.csv`: every geometry-matched candidate pair.
- `paired_metrics.csv`: only pairs passing the reference-policy JS and Monte
  Carlo return-difference thresholds; it contains policy JS divergence,
  oracle-action accuracy, value difference,
  full-horizon Monte Carlo return difference and paired rollout SEM,
  first-delivery oracle difference, and within-network actor/critic cosine
  distances across full-layout observations.
- `rollout_returns.csv`: individual full-horizon discounted sparse returns.
- `metadata.json`: settings, manifest, exact pair definitions, and caveats.

The reference checkpoint is loaded automatically from `--model-root` even when
its seed is omitted from `--seeds`. It remains in `paired_metrics_all.csv` for
auditing and is excluded from `paired_metrics.csv`, so use a held-out reference
seed as in the command above. With an explicit `--checkpoints` manifest, include
that reference checkpoint and provide `model`, `num_envs`, `seed`, and
`checkpoint` for every entry.

## Visualize locally

The graph script only needs pandas and Matplotlib; it does not load JAX or model
checkpoints. Copy `paired_metrics.csv` from the model root to a local artifacts
directory, then run:

```bash
python baselines/CEC/figures/analysis/policy_value_asymmetry_graph.py \
  --metrics artifacts/policy_value_asymmetry/paired_metrics.csv \
  --output-dir artifacts/policy_value_asymmetry/figures
```

This creates `policy_value_asymmetry_summary.pdf/png` and
`policy_value_asymmetry_pairs.pdf/png` beside the CSV. The summary first
averages state pairs within each training seed and uses variation across seeds
for error bars. With one seed, it draws no uncertainty bar. The pair plot shows
policy JS divergence against the predicted short-minus-long value difference.

The fourth summary panel measures how strongly actor and critic features change
between the two full layout observations. Greater cosine distance means stronger
layout-specific representation sensitivity. It should be interpreted together
with oracle-action accuracy: equal but incorrect actions are not evidence of a
useful invariant policy.

Small policy divergence with a reproducible return difference supports local
policy/value asymmetry under this intervention. It does not prove equality of
optimal episode policies or that value learning causes generalization failures.
Critic predictions were trained under potentially different partners and reward
shaping; they need not equal this stationary-partner sparse-return estimate.
The oracle's first-delivery return is distinct from full-horizon policy value.
Cosine distances across separately trained feature spaces are descriptive only.
Use independent training seeds for model comparisons; rollout SEM measures
sampling uncertainty for a single checkpoint, not training-seed uncertainty.

## Rollout-based representation geometry

This is the primary policy-equivalent/value-distinct analysis. A held-out DCEC
checkpoint collects trajectories from all five layout families. At every
timestep it records the full observation history, action distribution, and
realized discounted return-to-go. Pairs must come from different layout
episodes within the same family and agent index. They are retained when the
reference policy JS divergence is small and the return-to-go gap is large.

```bash
python baselines/CEC/figures/analysis/policy_value_geometry.py \
  --model-root /mnt/nas/wonsang/crossenv_ued/models/ICRL \
  --models cec_64 dcec_64 \
  --seeds 1 2 3 4 5 \
  --reference-model dcec_64 \
  --reference-seed 0 \
  --episodes-per-layout 20 \
  --steps 400 \
  --max-reference-policy-js 0.02 \
  --min-return-gap 1.0 \
  --pairs-per-family 100
```

Outputs are saved under `<model-root>/analysis/policy_value_geometry/`:

- `selected_state_pairs.csv`: the common state-pair definition and reference
  policy/return criteria.
- `representation_geometry.csv`: CEC/DCEC policy/value distances, model policy
  JS, predicted values, checkpoint, and seed for every selected pair.
- `metadata.json`: the reference checkpoint and all analysis definitions.

Plot a copied metrics CSV locally with:

```bash
python baselines/CEC/figures/analysis/policy_value_geometry_graph.py \
  --metrics artifacts/policy_value_geometry/representation_geometry.csv \
  --output artifacts/policy_value_geometry/representation_geometry \
  --max-model-policy-js 0.05
```

Because the networks are recurrent, all checkpoints are teacher-forced through
the same complete reference observation histories. The comparison therefore
does not reset the LSTM independently at each sampled state. For CEC, the
primary policy and value distances both refer to its shared recurrent trunk.
For DCEC, they refer to its separate policy and value trunks. Distances at the
actor and critic head inputs are also saved separately.

The graph contains two panels with `|delta G|` on the x-axis and cosine distance
on the y-axis. It plots only rows where the evaluated model itself also has low
policy JS. Spearman correlations are computed within each training seed and
written to `representation_geometry_correlations.csv` beside the figure,
together with the seed-level mean policy/value distance and their difference.
Evidence consistent with the hypothesis is weak
policy-distance correlation together with positive value-distance correlation
for DCEC. These are associations under a fixed reference-policy occupancy
distribution, not a causal proof that representation coupling harms return.
