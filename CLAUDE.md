# CLAUDE.md

## Project

**CrossEnv UED** — single agent trained across procedurally generated environments outperforms multi-agent baselines on zero-shot coordination (Overcooked).

Stack: JAX + JaxMARL, Hydra config, W&B logging, `jax.lax.scan` training loops.

## Reference Docs

Detailed specs in `docs/` — Claude reads these on-demand when relevant:

- `docs/baselines.md` — algorithm overview, shared conventions (scan, Hydra, W&B, eval layouts)
- `docs/cec.md` — CEC core idea, file roles, network architectures
- `docs/cec_ued.md` — VAE→PLR training loop, minimax PLR concepts, VAE encoding
- `docs/jaxmarl.md` — stateless env design, LogWrapper, Overcooked `custom_reset` API
- `PROJECT_STRUCTURE.md` — full directory map, algorithm comparison table
