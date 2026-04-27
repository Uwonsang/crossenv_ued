# baselines/CEC/

**Core idea**: identical to IPPO (2 agents, standard PPO), but the Overcooked layout is randomly permuted every episode instead of fixed. No other algorithmic change — the diversity of layouts alone drives better zero-shot generalisation.

- `ippo_general.py` — CEC (IPPO + random layout reset per episode)
- `e3t.py` — E3T (ensemble variant)
- `fcp_general.py` — FCP variant
- `ippo_general_population.py` — CEC-FT (population fine-tuning)
- `actor_networks.py` — shared nets: `ScannedRNN`, `ActorCriticRNN`, `ActorCriticE3T`
- `test_*.py` — evaluation scripts; `test_all_models_cross.py` compares all methods
