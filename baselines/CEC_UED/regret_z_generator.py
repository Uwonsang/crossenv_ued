import os
import jax
import jax.numpy as jnp
import flax.linen as nn
from flax.training.train_state import TrainState
import optax
import distrax
import wandb
import pickle

class NormalGaussianZ:
    def __init__(self, config):
        self.z_dim = config["z_dim"]

    def get_z(self, z_gen_state, key):
        return jax.random.normal(key, (1, self.z_dim))

class AdversaryNet(nn.Module):
    z_dim: int
    hidden_dim: int

    @nn.compact
    def __call__(self, z):
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
    def __init__(self, config, key, run_dir: str = None):
        self.z_dim         = config["ENV_KWARGS"]['vae_config']['latent_dim']
        self.batch_size    = config["NUM_ENVS"]
        self.hidden_dim    = config["gen_hidden_dim"]
        self.use_wandb     = config["WANDB_MODE"] # have to fix
        self.kl_coeff      = config["kl_coeff"]
        self.entropy_coeff = config["entropy_coeff"]

        self.use_mean_clipping = config["use_mean_clipping"]
        self.clip_mean_min     = getattr(config, 'clip_mean_min', -0.5)
        self.clip_mean_max     = getattr(config, 'clip_mean_max',  0.5)
        self.use_std_scaling   = config["use_std_scaling"]
        self.scale_log_std     = getattr(config, 'scale_log_std',   1.0)
        self.scale_std_offset  = getattr(config, 'scale_std_offset', 0.0)

        if not self.use_wandb and run_dir is not None:
            self.log_dir  = os.path.join(run_dir, "logs")
            self.save_dir = os.path.join(run_dir, "models")
            os.makedirs(self.log_dir,  exist_ok=True)
            os.makedirs(self.save_dir, exist_ok=True)

        self.net = AdversaryNet(z_dim=self.z_dim, hidden_dim=self.hidden_dim)
        tx = self._build_optimizer(config)
        dummy_z = jnp.zeros((1, self.z_dim))
        params = self.net.init(key, dummy_z)
        self.init_state = TrainState.create(apply_fn=self.net.apply, params=params, tx=tx)

    def _build_optimizer(self, config):

        lr = config["adversary_learning_rate"]
        wd = config["adversary_weight_decay"]
        transforms = [optax.adamw(learning_rate=lr, weight_decay=wd)] # TODO: grad_clip, lr_scheduler
        
        return optax.chain(*transforms)

    def _apply_net(self, apply_fn, params, z):
        mean, log_std = apply_fn(params, z)
        
        if self.use_mean_clipping:
            current_mean = jnp.mean(mean)
            shift = jnp.where(current_mean < self.clip_mean_min,  self.clip_mean_min - current_mean,
                    jnp.where(current_mean > self.clip_mean_max, self.clip_mean_max - current_mean, 0.0))
            mean = mean + shift

        if self.use_std_scaling:
            log_std = jnp.tanh(log_std) * self.scale_log_std + self.scale_std_offset
        
        return mean, log_std

    def get_z(self, z_gen_state, key):
        """Returns (z_new, z_old, log_prob, prior_mean, prior_std).
        prior_mean/std returned for use in train_step KL calculation.
        """
        key, key1, key2 = jax.random.split(key, 3)
        z_prior = distrax.Normal(
            loc=jnp.zeros((self.z_dim,)),
            scale=jnp.ones((self.z_dim,)))

        z_sample_old = z_prior.sample(seed=key1)

        mean, log_std = self._apply_net(
            z_gen_state.apply_fn, z_gen_state.params, z_sample_old)
        std = jnp.exp(log_std) + 1e-6

        z_current = distrax.Normal(mean, std)
        z_sample_new = z_current.sample(seed=key2)
        log_prob = z_current.log_prob(z_sample_new)

        return z_sample_new, z_sample_old, log_prob, z_current, z_prior
        
    def compute_policy_stats(self, apply_fn, params, old_dist=None, key=None):
        """Compute policy distribution stats after update."""

        z_normal = jax.random.normal(key, (self.batch_size, self.z_dim))
        mean, log_std = self._apply_net(apply_fn, params, z_normal)
        std = jnp.exp(log_std) # TODO: check add the number [jnp.exp(log_std) + 1e-6]

        new_dist = distrax.Normal(mean, std)
        entropy = new_dist.entropy().mean()

        kl_div = 0.0
        if old_dist is not None:
            kl_div = self.kl_divergence(old_dist, new_dist).mean()

        return entropy, kl_div

    def train_step(self, z_gen_state, key, current_reward, trained_reward, 
                         z_new, z_old, z_log_probs, z_current, z_prior):

        # No grad
        key, k_diag = jax.random.split(key)
        z_normal_diag = jax.random.normal(k_diag, (self.batch_size, self.z_dim))
        old_mean_diag, old_log_std_diag = self._apply_net(z_gen_state.apply_fn, z_gen_state.params, z_normal_diag)
        old_std_diag = jnp.exp(old_log_std_diag)  # TODO: check add the number [jnp.exp(old_log_std_diag) + 1e-6]
        old_dist_diag = distrax.Normal(old_mean_diag, old_std_diag)

        def loss_fn(params):
            raw_regret = trained_reward - current_reward
            regret = (raw_regret - raw_regret.mean()) / (raw_regret.std() + 1e-8)
            weighted_log_probs = z_log_probs.sum(axis=1) * regret 

            kl_div = self.kl_divergence(z_current, old_dist_diag)
            entropy = z_current.entropy().mean()

            loss = (
                -weighted_log_probs.mean()
                + self.kl_coeff * kl_div
                - self.entropy_coeff * entropy
            )
            
            return loss, (weighted_log_probs, kl_div, entropy, raw_regret)

        (loss, aux), grads = jax.value_and_grad(loss_fn, has_aux=True)(z_gen_state.params)
        new_z_gen_state = z_gen_state.apply_gradients(grads=grads)
        weighted_log_probs, kl_div, entropy, raw_regret = aux
        
        key, k_stats = jax.random.split(key)
        entropy_diag, kl_diag = self.compute_policy_stats(apply_fn=new_z_gen_state.apply_fn, params=new_z_gen_state.params, 
                                            old_dist=old_dist_diag, key=k_stats)
        
        train_info = {
            "loss": loss,
            "episode_rewards_current": current_reward.mean(),
            "episode_rewards_trained": trained_reward.mean(),
            "regret": raw_regret.mean(),
            "weighted_log_probs": weighted_log_probs.mean(),
            "kl_divergence": kl_div,
            "policy_update": kl_diag,
            "entropy": entropy_diag}

        return new_z_gen_state, train_info


    def log_train_info(self, train_info, steps):
        log_data = {
            'adversary/before_reward': float(train_info['before_reward']),
            'adversary/after_reward': float(train_info['after_reward']),
            'adversary/regret': float(train_info['regret']),
            'adversary/loss': float(train_info['loss']),
            'adversary/kl_divergence': float(train_info['kl_divergence']),
            'adversary/entropy': float(train_info['entropy']),
            'adversary/weighted_log_probs': float(train_info['weighted_log_probs']),
            'adversary/z_sample/log_prob': float(train_info['log_prob']),
            'adversary/policy_update': float(train_info['policy_update']),
        }
        if self.use_wandb:
            wandb.log(log_data, step=steps)
        else:
            for k, v in log_data.items():
                self.writer.add_scalar(k, v, steps)

    def save(self, z_gen_state, steps):
        path = os.path.join(self.save_dir, "adversary", f"adversary_{steps}.pkl")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'wb') as f:
            pickle.dump(z_gen_state.params, f)

    def restore(self, z_gen_state, steps):
        path = os.path.join(self.save_dir, "adversary", f"adversary_{steps}.pkl")
        with open(path, 'rb') as f:
            params = pickle.load(f)
        return z_gen_state.replace(params=params)

    def kl_divergence(self, z_current, old_dist_diag):
        # KL(N1 || N0) analytic formula
        mean1 = z_current.loc
        std1  = z_current.scale
        mean0 = old_dist_diag.loc
        std0  = old_dist_diag.scale

        var1 = std1 ** 2
        var0 = std0 ** 2

        kl_per_dim = jnp.log(std0 / std1) + (var1 + (mean1 - mean0) ** 2) / (2.0 * var0) - 0.5
        kl_div = kl_per_dim.sum(axis=-1).mean()
        
        return kl_div