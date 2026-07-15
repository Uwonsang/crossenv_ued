# Human Proxy (Behaviour Cloning)

Trains a CNN policy via behaviour cloning on the Overcooked-AI human-human trial data,
producing a "human proxy" agent that can be dropped into the same evaluation pipeline
as the other Overcooked baselines (same 9×9 observation format as CEC/IPPO).

## Files

| File | Description |
|------|-------------|
| [layouts.py](layouts.py) | Maps raw human-data grids onto jaxmarl's fixed 9×9 `Overcooked` layout format |
| [preprocess.py](preprocess.py) | Converts a raw trial CSV into `(obs, action)` pairs, using `Overcooked.get_obs` directly so observations exactly match the live env |
| [bc_agent.py](bc_agent.py) | Flax CNN classifier trained with cross-entropy on human actions (Hydra + W&B) |
| [config/bc.yaml](config/bc.yaml) | Hydra config (hyperparams, data paths, W&B settings) |
| [data/](data/) | Raw `2019_hh_trials.csv` / `2020_hh_trials.csv` and `data/processed/` (preprocessed `.npz`) |

## Data & layout mapping

The raw CSVs are the original (unprocessed) Overcooked-AI web-experiment trial logs — one
row per game timestep, with the full JSON `state`, `joint_action`, and the layout's own
grid string. `jaxmarl`'s `Overcooked` env hardcodes a 9×9 board, but the raw layouts are
smaller (e.g. `cramped_room` is 4×5), so `layouts.py` embeds each raw grid **top-left**
into a 9×9 canvas and fills the rest with walls (zero coordinate offset needed as a
result). Only 5 of the 2019 layouts have a matching jaxmarl layout family:

| Raw `layout_name` | jaxmarl layout |
|---|---|
| `cramped_room` | `cramped_room` |
| `coordination_ring` | `coord_ring` |
| `asymmetric_advantages` | `asymm_advantages` |
| `random0` | `forced_coord` |
| `random3` | `counter_circuit` |

**2020 data is not supported.** All 8 of its layouts either exceed 9 columns (too wide
for the 9×9 grid) or use tomato as an actual order ingredient, which `jaxmarl`'s
`Overcooked` has no object type for (`common.py`'s `OBJECT_TO_INDEX` has no `tomato`
entry). Using it would require extending the shared env, not just this pipeline.

## Usage

**Preprocess the raw CSV into train/test `.npz`:**
```bash
python -m baselines.human_proxy.preprocess \
  --csv baselines/human_proxy/data/2019_hh_trials.csv \
  --out_dir baselines/human_proxy/data/processed
```

**Train:**
```bash
python -m baselines.human_proxy.bc_agent
```

Config overrides follow Hydra syntax, e.g.:
```bash
python -m baselines.human_proxy.bc_agent NUM_EPOCHS=100 BATCH_SIZE=512 WANDB_MODE=online
```

| Key | Default | Notes |
|-----|---------|-------|
| `DATA_DIR` / `DATA_STEM` | `baselines/human_proxy/data/processed` / `2019_hh_trials` | Which preprocessed split to load |
| `NUM_EPOCHS` | `50` | Full dataset passes |
| `BATCH_SIZE` | `256` | |
| `CKPT_DIR` | `baselines/human_proxy/checkpoints` | Where the trained params (pickled) are saved |
| `WANDB_MODE` | `disabled` | Set to `online` to log to W&B |

Both players' (obs, action) pairs from every trial are used as independent training
samples; train/test are split by `(trial_id, player_0_id, player_1_id)` so no single
human play session leaks across the split.
