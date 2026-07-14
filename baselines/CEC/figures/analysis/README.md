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
| `--x-axis` | `env_step` | `env_step` or `update_step` (`update_step = env_step / (NUM_ENVS * NUM_STEPS)`) |

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

## Examples

```bash
python3 td_error_graph.py --run-id 9g9abvem --td-metric rmse
python3 share_gradient_graph.py --run-id 5rwobcx9 --loss-type actor --view raw
python3 grad_norm_graph.py --run-id 5rwobcx9 --loss-type value --x-axis update_step
python3 absolute_contribution_graph.py --run-id 5rwobcx9 --loss-type value --view stack
python3 target_raw_graph.py --run-id 5rwobcx9
python3 target_popart_graph.py --run-id 5rwobcx9
python3 train_returns_graph.py --run-id 5rwobcx9 --smooth-window 20
python3 eval_graph.py --run-id 5rwobcx9
```

Run `python3 <script>.py --help` for the full flag list.
