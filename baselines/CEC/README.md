# CEC Baselines

Training and evaluation code for **Cross-Environment Coordination (CEC)** and comparison baselines on Overcooked (JAX/JaxMARL).

## Files

### Training

| File | Description |
|------|-------------|
| [ippo_general.py](ippo_general.py) | IPPO trained across 5 procedurally generated 9x9 Overcooked layouts (CEC training regime) |
| [ippo_general_population.py](ippo_general_population.py) | IPPO population training — trains a pool of agents for FCP partner generation |
| [fcp_general.py](fcp_general.py) | Fictitious Co-Play (FCP) — loads a population and trains a coordinator agent |
| [e3t.py](e3t.py) | E3T — trains with entropy-regularised teammate randomization across the 5 layouts |

### Evaluation

| File | Description |
|------|-------------|
| [test_general.py](test_general.py) | Cross-play evaluation: loads a checkpoint and rolls out against all partner types across layouts |
| [test_general_pcg.py](test_general_pcg.py) | Same as above but on procedurally generated (held-out) layouts |
| [test_general_check.py](test_general_check.py) | Sanity-check variant of `test_general.py` |
| [test_e3t.py](test_e3t.py) | Evaluation script specific to E3T checkpoints (includes t-SNE embedding) |
| [test_oracle.py](test_oracle.py) | Evaluates an oracle agent (trained per-layout) as an upper-bound baseline |
| [test_cross_env.py](test_cross_env.py) | Evaluates cross-environment generalization |
| [test_all_models_cross.py](test_all_models_cross.py) | Batch evaluation across all model types (IPPO, E3T, FCP, CEC) |
| [cross_algo.py](cross_algo.py) | Shared cross-play rollout logic used by eval scripts |

### Networks

| File | Description |
|------|-------------|
| [actor_networks.py](actor_networks.py) | `ScannedRNN` (LSTM), `ActorCriticRNN`, `ActorCriticE3T` — shared across all algorithms |

### Config & Scripts

| Path | Description |
|------|-------------|
| [config/ippo_final.yaml](config/ippo_final.yaml) | Default Hydra config (hyperparams, env kwargs, W&B settings) |
| [shell/bash_test_general.sh](shell/bash_test_general.sh) | Runs `test_general.py` for all layouts × all models |
| [shell/bash_e3t.sh](shell/bash_e3t.sh) | Training script for E3T |
| [shell/bash_fcp.sh](shell/bash_fcp.sh) | Training script for FCP |
| [shell/bash_ippo_pop.sh](shell/bash_ippo_pop.sh) | Training script for IPPO population |
| [sweep/](sweep/) | W&B sweep configs for hyperparameter search |
| [figures/](figures/) | Scripts for generating paper figures |

## Layouts

All training uses five 9×9 Overcooked layouts:

- `cramped_room_9`
- `asymm_advantages_9`
- `coord_ring_9`
- `counter_circuit_9`
- `forced_coord_9`

Set via `ENV_KWARGS.layout` in config or as a CLI override.

## Usage

**Train CEC (IPPO across environments):**
```bash
python baselines/CEC/ippo_general.py
```

**Evaluate all models across all layouts:**
```bash
bash baselines/CEC/shell/bash_test_general.sh <gpu_id>
```

**Evaluate a single model/layout:**
```bash
python baselines/CEC/test_general.py ENV_KWARGS.layout=cramped_room_9 model_name=CEC
```

Config overrides follow Hydra syntax. Key config fields:

| Key | Default | Notes |
|-----|---------|-------|
| `ENV_KWARGS.random_reset` | `False` | `True` = CEC regime, `False` = single-task |
| `TRAIN_KWARGS.ckpt_id` | `0` | Checkpoint index to save/load |
| `WANDB_MODE` | `disabled` | Set to `online` to log to W&B |
| `TEST_KWARGS.self_play` | `False` | `True` evaluates full cross-play matrix |
