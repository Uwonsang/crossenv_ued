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
- **`gradient_diagnostics_num_envs_graph.py`** — `cec_stiffness_100m`에서
  `TOTAL_TIMESTEPS=300M`인 IPPO/IDAAC run의 `policy_value`, environment
  gradient cosine, effective rank, SNR을 가져온다. 알고리즘·`NUM_ENVS`별
  0M–300M 전체 구간과 마지막 30M-step 구간의 평균 그래프·집계 CSV를
  동시에 저장한다. 300M의 95%에 도달하지 못한 run은 run/history CSV에는
  남기고 두 요약에서는 제외한다. 학습 time-series는 W&B에서 직접 확인한다.

## Current logging compatibility

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
python baselines/CEC/figures/analysis/eval_xp_model_graph.py \
  --model-names CEC_IDAAC CEC_POP CEC_IDAAC_POP CEC \
  --num-envs 256 \
  --seeds 0 1
python baselines/CEC/figures/analysis/eval_xp_comparison_graphs.py
python baselines/CEC/figures/analysis/eval_xp_scaling_graph.py
python baselines/CEC/figures/analysis/gradient_diagnostics_num_envs_graph.py
```

Run `python3 <script>.py --help` for the full flag list.
