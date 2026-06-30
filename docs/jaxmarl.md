# jaxmarl/

JAX multi-agent RL framework. All env logic is pure-functional (stateless), JIT-compilable, and vectorisable with `jax.vmap`. `reset`/`step` return new state objects — never mutate.

Primary env for this research: `environments/overcooked/` (2-agent cooperative cooking, 9×9 PCG layouts).
`LogWrapper` in `wrappers/baselines.py` is applied in every training script.

## environments/overcooked/

Non-obvious API: `env.custom_reset(key, layout=layout_dict, random_reset=False)` — resets to a specific layout dict (used in CEC eval and VAE pipeline).
Layout constructors live in `overcooked/layouts.py` (`make_*_9x9()`); the fixed eval set is `overcooked_layouts` dict.
