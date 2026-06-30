# baselines/CEC_UED/

CEC extended with UED. Two variants:

- `ippo_general_minimax.py` — **Minimax**: CEC + PLR curriculum only. No VAE. PLR selects hard layouts from randomly generated ones.
- `ippo_general_vae.py` — **VAE**: CEC + VAE layout generator + PLR curriculum. VAE samples new layouts from latent space; PLR keeps hard ones and retrains VAE on them.
- `algo_utils.py` — HDF5 helpers + `EVAL_LAYOUTS_9` constant
- `regret_z_generator.py` — adversarial net that proposes hard layout latents

## minimax/ — PLR Curriculum

- **PLRBuffer**: ring buffer of `(layout, score, staleness)` entries.
- **UED score**: regret = max possible return − achieved return. High score = hard layout.
- **Staleness**: tracks how long since a level was last trained on; stale levels get refreshed.
- **Sampling**: mixes high-score replays and new random layouts via `rho` / `replay_prob` config params.

Public API imported by `ippo_general_minimax.py`: `PLRManager`, `UEDScore`, `plr_batch_from_traj`, `plr_ued_scores_and_info`, `sample_layout_reset_all`.

## VAE/ — Layout Generation

VAE that encodes Overcooked 9×9 layouts into a latent space for CEC_UED sampling.

Input: `(H, W, 26)` obs → only 5 static channels used: `[10=pot, 11=wall, 12=onion_pile, 14=plate_pile, 15=goal]` (`STATIC_TRAIN_CHANNELS` in `utils.py`).

**Data pipeline**: `ippo_general.py` saves env states → `dataset/` (HDF5) → `train_vae.py` trains VAE → checkpoint loaded by `ippo_general_vae.py`.
