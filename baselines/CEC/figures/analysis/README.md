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
| `share_gradient_graph.py` | `--loss-type` | `actor`, `value` |
| `grad_norm_graph.py` | `--loss-type` | `actor`, `value` |

Output filenames encode the flags used, e.g. `grad_norm_value_5rwobcx9_env_step.png`.

## Scripts

- **`td_error_graph.py`** — TD error (`td_error/{metric}`), per layout + overall.
- **`share_gradient_graph.py`** — weighted gradient share (`grad_share_weighted_{loss_type}/...`), stacked area per layout.
- **`grad_norm_graph.py`** — gradient norm (`grad_conflict_{loss_type}/norm/...`), per layout.
- **`target_raw_graph.py`** — raw value target (`target_raw/...`), per layout + overall.
- **`train_returns_graph.py`** — training return (`train_returns/...`), per layout + overall.

## Examples

```bash
python3 td_error_graph.py --run-id 9g9abvem --td-metric rmse
python3 share_gradient_graph.py --run-id 5rwobcx9 --loss-type actor
python3 grad_norm_graph.py --run-id 5rwobcx9 --loss-type value --x-axis update_step
python3 target_raw_graph.py --run-id 5rwobcx9
python3 train_returns_graph.py --run-id 5rwobcx9 --smooth-window 20
```

Run `python3 <script>.py --help` for the full flag list.
