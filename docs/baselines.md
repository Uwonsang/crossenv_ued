# baselines/

All training loops use `jax.lax.scan` (PureJaxRL-style, fully JIT-compiled). Config via Hydra (`config/*.yaml`). Logging via W&B.

**IPPO / MAPPO / QLearning** — baselines: 2 agents, 1 fixed map, weak generalisation.
**CEC** — paper core: 2 agent, ~100–1000s procedural maps, strong zero-shot generalisation.
**CEC_UED (minimax)** — extension: same as CEC but layout selection uses PLR curriculum (no VAE).
**CEC_UED (VAE)** — extension: layouts are generated from a VAE latent space + PLR curriculum.

Eval is always on 5 fixed 9×9 Overcooked layouts (`EVAL_LAYOUTS_9` in `CEC_UED/algo_utils.py`).
