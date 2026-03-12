import os
import numpy as np
import jax
import jax.numpy as jnp
import flax.linen as nn
from flax.training.train_state import TrainState
import optax
import distrax
import wandb
from typing import Tuple


class NormalGaussianZ:
    def __init__(self, args):
        self.z_dim = args.z_dim
        self.batch_size = args.n_rollout_threads

    def get_z(self, key: jax.Array):
        return jax.random.normal(key, (self.batch_size, self.z_dim))

class AdversaryNet(nn.Module):
    z_dim: int
    hidden_dim: int

    @nn.compact
    def __call__(self, z: jax.Array) -> Tuple[jax.Array, jax.Array]:
        x = nn.Dense(self.hidden_dim)(z)
        x = nn.relu(x)
        x = nn.Dense(self.hidden_dim)(x)
        x = nn.relu(x)
        x = nn.Dense(self.hidden_dim)(x)
        x = nn.relu(x)
        x = nn.Dense(2 * self.z_dim)(x)
        mean, log_std = jnp.split(x, 2, axis=-1)
        return mean, log_std


class AdversarialZ:
    def __init__(self, args, key: jax.Array, run_dir: str = None):
        self.z_dim        = args.z_dim
        self.batch_size   = args.n_rollout_threads
        self.hidden_dim   = args.hidden_dim
        self.use_wandb    = args.use_wandb

        # loss coefficients
        self.kl_coeff      = args.kl_coeff      if args.use_kl             else 0.0
        self.entropy_coeff = args.entropy_coeff  if args.use_entropy_bonus  else 0.0

        # gradient clipping
        self.use_grad_clip  = args.use_grad_clip
        self.grad_clip_norm = args.grad_clip_norm

        # mean / std scaling
        self.use_mean_clipping = args.use_mean_clipping
        self.clip_mean_min     = getattr(args, 'clip_mean_min', -0.5)
        self.clip_mean_max     = getattr(args, 'clip_mean_max',  0.5)
        self.use_std_scaling   = args.use_std_scaling
        self.scale_log_std     = getattr(args, 'scale_log_std',   1.0)
        self.scale_std_offset  = getattr(args, 'scale_std_offset', 0.0)

        # logging
        if not self.use_wandb and run_dir is not None:
            self.log_dir  = os.path.join(run_dir, "logs")
            self.save_dir = os.path.join(run_dir, "models")
            os.makedirs(self.log_dir,  exist_ok=True)
            os.makedirs(self.save_dir, exist_ok=True)

        # build network + optimizer
        self.net = AdversaryNet(z_dim=self.z_dim, hidden_dim=self.hidden_dim)

        tx = self._build_optimizer(args)

        dummy_z = jnp.zeros((1, self.z_dim))
        params  = self.net.init(key, dummy_z)
        self.state = TrainState.create(apply_fn=self.net.apply, params=params, tx=tx)

    def _build_optimizer(self, args) -> optax.GradientTransformation:
        lr = args.adversary_learning_rate
        wd = args.adversary_weight_decay

        if args.use_lr_scheduler:
            schedule = optax.linear_schedule(
                init_value=lr * args.start_factor,
                end_value=lr  * args.end_factor,
                transition_steps=args.total_episodes,
            )
        else:
            schedule = lr

        transforms = [optax.adamw(learning_rate=schedule, weight_decay=wd)]
        if args.use_grad_clip:
            transforms.insert(0, optax.clip_by_global_norm(args.grad_clip_norm))

        return optax.chain(*transforms)

    def _apply_net(self, params, z: jax.Array):
        mean, log_std = self.state.apply_fn(params, z)

        if self.use_mean_clipping:
            current_mean = jnp.mean(mean)
            shift = jnp.where(
                current_mean < self.clip_mean_min,
                self.clip_mean_min - current_mean,
                jnp.where(current_mean > self.clip_mean_max,
                           self.clip_mean_max - current_mean, 0.0)
            )
            mean = mean + shift

        if self.use_std_scaling:
            log_std = jnp.tanh(log_std) * self.scale_log_std + self.scale_std_offset

        return mean, log_std

    def get_z(self, key: jax.Array):
        """Returns (z_new, z_old, log_prob, current_dist, prior_dist)."""
        key, k1, k2 = jax.random.split(key, 3)

        # prior sample
        z_old = jax.random.normal(k1, (self.batch_size, self.z_dim))

        # posterior
        mean, log_std = self._apply_net(self.state.params, z_old)
        std           = jnp.exp(log_std) + 1e-6
        current_dist  = distrax.Normal(mean, std)
        prior_dist    = distrax.Normal(jnp.zeros_like(mean), jnp.ones_like(std))

        z_new    = current_dist.sample(seed=k2)
        log_prob = current_dist.log_prob(z_new)

        return z_new, z_old, log_prob, current_dist, prior_dist

    def train_step(
        self,
        key: jax.Array,
        episode_rewards_vae_vs_vae: jax.Array,
        episode_rewards_minimax:    jax.Array,
        z_new:        jax.Array,
        z_old:        jax.Array,
        log_probs:    jax.Array,
        current_dist: distrax.Distribution,
        old_dist:     distrax.Distribution,
    ):
        key, k_diag = jax.random.split(key)

        # snapshot params before update (for policy-change stats)
        old_params = self.state.params

        def loss_fn(params):
            # regret signal
            raw_regret = episode_rewards_vae_vs_vae - episode_rewards_minimax
            regret     = (raw_regret - raw_regret.mean()) / (raw_regret.std() + 1e-8)

            weighted_log_probs = log_probs.sum(axis=1) * regret

            # re-evaluate current dist under updated params for KL/entropy
            mean, log_std = self._apply_net(params, z_old)
            std  = jnp.exp(log_std) + 1e-6
            dist = distrax.Normal(mean, std)

            kl_div  = distrax.kl_divergence(dist, old_dist).mean()
            entropy = dist.entropy().mean()

            loss = (
                -weighted_log_probs.mean()
                + self.kl_coeff      * kl_div
                - self.entropy_coeff * entropy
            )
            return loss, (weighted_log_probs, kl_div, entropy)

        (loss, aux), grads = jax.value_and_grad(loss_fn, has_aux=True)(self.state.params)
        self.state = self.state.apply_gradients(grads=grads)

        weighted_log_probs, kl_div, entropy = aux

        # policy-change stats (diagonal gaussian)
        z_diag          = jax.random.normal(k_diag, (self.batch_size, self.z_dim))
        old_mean, old_ls = self._apply_net(old_params,       z_diag)
        new_mean, new_ls = self._apply_net(self.state.params, z_diag)
        old_d = distrax.Normal(old_mean, jnp.exp(old_ls) + 1e-6)
        new_d = distrax.Normal(new_mean, jnp.exp(new_ls) + 1e-6)
        entropy_diag = new_d.entropy().mean()
        kl_diag      = distrax.kl_divergence(old_d, new_d).mean()

        raw_regret = episode_rewards_vae_vs_vae - episode_rewards_minimax
        train_info = {
            'loss':                        float(loss),
            'lr':                          float(self.state.opt_state[-1].count) if hasattr(self.state.opt_state, '__len__') else 0,
            'episode_rewards_vae_vs_vae':  float(episode_rewards_vae_vs_vae.mean()),
            'episode_rewards_minimax':     float(episode_rewards_minimax.mean()),
            'regret':                      float(raw_regret.mean()),
            'weighted_log_probs':          float(weighted_log_probs.mean()),
            'kl_divergence':               float(kl_div),
            'policy_update':               float(kl_diag),
            'entropy':                     float(entropy_diag),
            'z_sample/mean':               float(z_new.mean()),
            'z_sample/std':                float(z_new.std()),
            'log_prob':                    float(log_probs.mean()),
        }
        return train_info

    def log_train_info(self, train_info: dict, steps: int):
        log_data = {
            'adversary/lr':                  train_info['lr'],
            'adversary/vae_vs_vae_reward':   train_info['episode_rewards_vae_vs_vae'],
            'adversary/minimax_reward':      train_info['episode_rewards_minimax'],
            'adversary/regret':              train_info['regret'],
            'adversary/loss':                train_info['loss'],
            'adversary/kl_divergence':       train_info['kl_divergence'],
            'adversary/entropy':             train_info['entropy'],
            'adversary/weighted_log_probs':  train_info['weighted_log_probs'],
            'adversary/z_sample/mean':       train_info['z_sample/mean'],
            'adversary/z_sample/std':        train_info['z_sample/std'],
            'adversary/z_sample/log_prob':   train_info['log_prob'],
            'adversary/policy_update':       train_info['policy_update'],
        }
        if self.use_wandb:
            wandb.log(log_data, step=steps)
        else:
            for k, v in log_data.items():
                self.writer.add_scalar(k, v, steps)


    def save(self, steps: int):
        import pickle
        path = os.path.join(self.save_dir, "adversary", f"adversary_{steps}.pkl")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'wb') as f:
            pickle.dump(self.state.params, f)

    def restore(self, steps: int):
        import pickle
        path = os.path.join(self.save_dir, "adversary", f"adversary_{steps}.pkl")
        with open(path, 'rb') as f:
            params = pickle.load(f)
        self.state = self.state.replace(params=params)

def get_z_generator(args, key: jax.Array, run_dir: str = None):
    if args.vae_z_generator == "gaussian":
        return NormalGaussianZ(args)
    elif args.vae_z_generator == "adversarial":
        return AdversarialZ(args, key, run_dir=run_dir)
    else:
        raise ValueError(f"Z generator {args.vae_z_generator} not defined")