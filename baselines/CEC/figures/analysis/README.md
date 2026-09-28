# figures/analysis/

Scripts that pull metrics directly from a wandb run and save a per-layout line plot to
`figures/results/<script_name>/`. Each script is standalone — run it directly with Python.

All scripts share the same CLI flags:

| Flag | Default | Description |
|---|---|---|
| `--entity` | `overcooked_ai` | wandb entity |
| `--project` | `crossenv_ued_gradient` | wandb project |
| `--run-id` | (per script) | wandb run ID to plot |
| `--smooth-window` | `50` | rolling-mean window, in samples |
| `--x-axis` | 대부분 `env_step` | `env_step` or `update_step` (`update_step = env_step / (NUM_ENVS * NUM_STEPS)`); `absolute_contribution_graph.py`만 기본값이 `update_step` |

Some scripts add an extra flag for selecting which logged variant to plot:

| Script | Extra flag | Choices |
|---|---|---|
| `td_error_graph.py` | `--td-metric` | `mean_abs`, `rmse` |
| `share_gradient_graph.py` | `--loss-type`, `--view` | `actor`/`value`, `deviation`/`raw` |
| `grad_norm_graph.py` | `--loss-type` | `actor`, `value` |
| `absolute_contribution_graph.py` | `--loss-type`, `--view` | `actor`/`value`, `line`/`stack` |

Output filenames encode the flags used plus the run's `model_name` (from wandb config),
e.g. `grad_norm_value_CEC_POP_5rwobcx9_env_step.png`.

## Scripts

- **`td_error_graph.py`** — TD error (`td_error/{metric}`), per layout.
- **`share_gradient_graph.py`** — weighted gradient share (`grad_share_weighted_{loss_type}/...`), per layout; `--view deviation` (default) plots each layout's deviation from the equal share (1/5), `--view raw` plots the raw share with an equal-share reference line.
- **`grad_norm_graph.py`** — gradient norm (`grad_conflict_{loss_type}/norm/...`), per layout.
- **`absolute_contribution_graph.py`** — absolute weighted gradient contribution per layout, `c_l = w_l * ||g_l||` where `w_l = sample_share/{layout}` (raw sample fraction) and `||g_l|| = grad_conflict_{loss_type}/norm/{layout}`; not a ratio, unlike `grad_share_weighted`. `--view line` (default) plots per-layout lines; `--view stack` stacks them (the stack sums to the batch-average gradient norm).
- **`target_raw_graph.py`** — raw value target (`target_raw/...`), per layout.
- **`target_popart_graph.py`** — PopArt-normalized value target (`target_popart/...`), per layout.
- **`train_returns_graph.py`** — training return (`train_returns/...`), per layout.
- **`eval_graph.py`** — eval return (`eval/...`), per layout.
- **`value_loss_generalization_graph.py`** — W&B의 `value_loss`와 held-out
  `eval/...` return 사이의 상관관계를 그린다. 동일 `NUM_ENVS`의 seed를 먼저
  평균하고, 각 점을 `NUM_ENVS` 설정 하나로 표시한다. 레이아웃별 Pearson
  correlation과 선형 회귀선을 포함하며 run-level/집계 CSV도 함께 저장한다.
  기본 출력 경로는 `artifacts/value_loss_generalization/`이다.
- **`generalization_gap_graph.py`** — 마지막 공통 학습 구간에서 레이아웃별
  `train_returns/* - eval/*` generalization gap을 계산한다. 양수는 생성된
  training layout의 return이 고정 evaluation layout보다 높다는 뜻이다.
  CEC/CEC-IDAAC의 환경 수별 seed 평균과 SEM을 비교하고 원자료/집계 CSV를
  `artifacts/generalization_gap/`에 저장한다.
- **`eval_xp_model_graph.py`** — `config.model_name`별 BC cross-play return
  (`eval_xp/mean`) 평균 곡선. 여러 run은 모델별로 묶는다. 기본 프로젝트는
  `crossenv_ICLR`이다. 환경 수가 다른 run도
  샘플 수 기준으로 비교할 수 있도록 x축을
  `update_step * NUM_ENVS * NUM_STEPS`로 다시 계산한다. 서로 다른 평가
  주기는 공통 x-grid로 보간한 뒤 집계하며, 기본적으로 해당 모델 run의
  절반 이상이 존재하는 구간만 표시한다. 결과는
  `figures/results/eval_xp_model_graph/`에 저장한다. `--num-envs`로 하나
  이상의 `NUM_ENVS` 값에 해당하는 run만 선택할 수 있다. 선은 해당
  model/NUM_ENVS run들의 평균이고 음영은 최소–최대 범위다. `--seeds`로
  그래프에 포함할 seed들을 지정할 수 있다.
- **`eval_xp_comparison_graphs.py`** — `NUM_ENVS=256`에서 generalization
  technique 비교 그래프만 생성한다. `--num-envs`와 `--seeds`로 포함할
  환경 수와 seed를 선택할 수 있다.
  BatchNorm은 현재 존재하는 seed 4/5를 사용한다. LayerNorm seed 0/1은
  현재 XP 평가가 각각 4개뿐이므로 그래프에 incomplete로 표시한다.
- **`eval_xp_scaling_graph.py`** — 선택한 seed의 `CEC`, `CEC_IDAAC_POP`에 대해
  `NUM_ENVS=32, 64, 128, 256` XP environment scaling curve를 두 개의
  subplot으로 생성한다. `--seeds 0 1`을 사용하면 선은 seed 평균, 음영은
  seed 최소–최대 범위를 나타낸다. 출력 파일명에는 선택한 seed가 자동으로
  포함된다.

## EGTA from cross-algorithm CSVs

`egta_analysis.py` reads the CSV schema produced by `baselines/CEC/cross_algo.py`.
It requires NumPy, pandas, SciPy and Matplotlib; no W&B connection or model
loading is needed. This analysis assumes a two-player shared-reward game with
uniform random seat assignment. It averages trajectories within each seed pair,
then seed pairs within each layout, then layouts equally. The role-averaged
payoff is `(M + M.T) / 2`. Missing cells (including diagonals), duplicate
evaluation keys and nonfinite rewards are errors. Inputs must share evaluation
settings/checkpoint selection; do not mix fixed-task and PCG datasets.

```bash
python baselines/CEC/figures/analysis/egta_analysis.py \
  --input /mnt/nas/wonsang/crossenv_ued/models/ICRL/xp_results_diff_algo \
  --models FCP CEC_envs64 CEC_IDAAC_envs256 \
  --xp-only --output-dir artifacts/egta/fixed_three
```

`--input` accepts one or more CSV files/directories. `--layouts` optionally
selects layouts. `--xp-only` excludes same-model, same-seed pairs; the default
retains them, matching whatever the evaluator collected. With XP-only, each
model needs cross-seed evaluations on the diagonal. Model names are raw CSV
identifiers. Use separate output directories for different model/task groups.

Outputs:

- `payoff.csv`: role-averaged, equal-layout payoff matrix in raw return units.
- `layout_payoffs.csv`, `seed_pair_summary.csv`: directional and role-averaged
  layout values, and seed-pair means/counts for auditing aggregation.
- `simplex.png`: vector field and trajectories, generated only for exactly
  three models. This represents the selected three-model subgame.
- `population.png`, `trajectories.csv`: dynamics for any number of models;
  the first start is uniform (solid lines), others are seeded Dirichlet draws
  (faint lines). Defaults: 16 starts, horizon 100, 501 time samples, seed 0.
- `endpoints.csv`: finite-time fractions, raw derivative norm, and symmetric
  Nash gap `max(A @ x) - x @ A @ x`.
- `pure_strategy_gaps.csv`: the same gap at each pure strategy. A zero
  replicator derivative at a vertex alone does not establish equilibrium.
- `metadata.json`: input paths, settings and assumptions.

Integration uses log population coordinates and one global payoff scale
`max(ptp(A), 1)`; this changes time units, not the dynamics' paths. No per-row,
per-column or per-layout score normalization is applied. Results are restricted
to the selected empirical model set. Endpoints are not certified equilibria,
and counts of endpoints are not certified attraction basin probabilities.
Sampling uncertainty/bootstrap confidence intervals are not estimated.
Asymmetric individual-payoff games and multiplayer payoff tensors are outside
this script's scope.

References: [Tuyls et al. (2018)](https://arxiv.org/abs/1803.06376),
[Serrino et al. (2019)](https://arxiv.org/abs/1906.02330),
[Wellman et al. (2024)](https://arxiv.org/abs/2403.04018).

Verification: `python -m unittest discover -s baselines/CEC/figures/analysis -p test_egta_analysis.py`.

## Current logging compatibility (training curves)

현재 `ippo_general_gradient.py` 계열의 새 run에는 `train_returns_graph.py`,
`eval_graph.py`, `target_raw_graph.py`가 그대로 동작한다.
`target_popart_graph.py`는 PopArt run에만 사용하고, `td_error_graph.py`는
`--td-metric rmse`로 실행해야 한다.

`grad_norm_graph.py`, `share_gradient_graph.py`,
`absolute_contribution_graph.py`는 과거 run에 남아 있는
`grad_conflict_*/norm/*`, `grad_share_weighted_*` key를 대상으로 한다. 현재
로깅은 `grad_norm_actor/*`, `grad_norm_critic/*`,
`grad_contribution_signed_*`를 사용하므로 이 세 스크립트는 새 run에 바로
적용되지 않는다.

## Examples

```bash
python baselines/CEC/figures/analysis/td_error_graph.py --run-id 9g9abvem --td-metric rmse
python baselines/CEC/figures/analysis/share_gradient_graph.py --run-id 5rwobcx9 --loss-type actor --view raw
python baselines/CEC/figures/analysis/grad_norm_graph.py --run-id 5rwobcx9 --loss-type value --x-axis update_step
python baselines/CEC/figures/analysis/absolute_contribution_graph.py --run-id 5rwobcx9 --loss-type value --view stack
python baselines/CEC/figures/analysis/target_raw_graph.py --run-id 5rwobcx9
python baselines/CEC/figures/analysis/target_popart_graph.py --run-id 5rwobcx9
python baselines/CEC/figures/analysis/train_returns_graph.py --run-id 5rwobcx9 --smooth-window 20
python baselines/CEC/figures/analysis/eval_graph.py --run-id 5rwobcx9
python baselines/CEC/figures/analysis/value_loss_generalization_graph.py \
  --project crossenv_ICLR \
  --target-env-steps 3000000000 \
  --model-names CEC CEC_IDAAC \
  --num-envs 32 64 128 256
python baselines/CEC/figures/analysis/generalization_gap_graph.py
python baselines/CEC/figures/analysis/eval_xp_model_graph.py \
  --model-names CEC_IDAAC CEC_POP CEC_IDAAC_POP CEC \
  --num-envs 256 \
  --seeds 0 1
python baselines/CEC/figures/analysis/eval_xp_comparison_graphs.py
python baselines/CEC/figures/analysis/eval_xp_scaling_graph.py
```

Run `python3 <script>.py --help` for the full flag list.

## Sparse baseline comparison panels

The simplex defaults to `--resolution 8` (45 grid positions) rather than 24
(325 positions). Resolution is unrelated to evaluation episode count.
Add `--show-trajectories` to overlay integration paths.

```bash
python baselines/CEC/figures/analysis/egta_panels.py \
  --payoff artifacts/egta/fixed_all/payoff.csv \
  --focal CEC_IDAAC_envs256 \
  --pair FCP IPPO --pair CEC_envs64 FCP --pair E3T IPPO \
  --title '5 original tasks | DCEC256' \
  --output-dir artifacts/egta/dcec256_baseline_panels
```

PNG/PDF panels use a shared speed scale normalized to [0, 1].
`baseline_checks.csv` records strict dominance margins within each subgame,
individual invasion gains, and focal pure-strategy Nash gaps. These are
empirical point estimates, not statistical significance tests. Missing focal
models are rejected; DCEC256 cannot substitute for DCEC128.

The evaluator accepts `CEC_IDAAC_envs128` explicitly through `MODEL_NAMES`.
Collect its missing data in a Python environment with the project's JAX
requirements, using the configured model root (override MODEL_PATH if needed):

```bash
for layout in asymm_advantages_9 coord_ring_9 counter_circuit_9 cramped_room_9 forced_coord_9; do
  python baselines/CEC/cross_algo.py \
    "ENV_KWARGS.layout=$layout" \
    'MODEL_NAMES=[IPPO,E3T,FCP,CEC_envs64,CEC_IDAAC_envs128]' \
    SAVE_PATH=artifacts/egta/dcec128_cross_play \
    TEST_KWARGS.num_trajs=10 XP_ONLY=True
done
```

Then run `egta_analysis.py` with that input directory and those five models,
and pass its payoff CSV to `egta_panels.py --focal CEC_IDAAC_envs128`.
PCG requires a separate complete cross-algorithm dataset; within-algorithm
cross-play scores do not provide the missing baseline cells.
