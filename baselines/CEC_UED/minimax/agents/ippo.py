"""
Copyright (c) Meta Platforms, Inc. and affiliates.
All rights reserved.

This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.
"""

from functools import partial
from collections import OrderedDict

import jax
import jax.numpy as jnp
import optax
from flax.training.train_state import TrainState
from tensorflow_probability.substrates import jax as tfp

from .agent import Agent


class IPPOAgent(Agent):
    def __init__(
            self,
            model,
            config,
            obs_dim,
            n_devices=1):

        self.model = model
        self.n_epochs = config["UPDATE_EPOCHS"]
        self.n_minibatches = config["NUM_MINIBATCHES"]
        self.value_loss_coef = config["VF_COEF"]
        self.entropy_coef = config["ENT_COEF"]
        self.clip_eps = config["CLIP_EPS"]
        self.n_unroll_update = config["n_unroll_update"]
        self.num_envs = config["NUM_ENVS"]
        self.gpu_hidden_dim = config["GRU_HIDDEN_DIM"]
        self.n_devices = n_devices

        self.grad_fn = jax.value_and_grad(self._loss, has_aux=True)

    def init_params(self, rng, obs):
        """
        Returns initialized parameters and RNN hidden state for a specific
        observation shape.
        """
        rng, subrng = jax.random.split(rng)
        batch_size = jax.tree_util.tree_leaves(obs)[0].shape[1]

        # get flattened obs dim #TODO: fix from obs
        # flattened_obs_dim = 1
        # for dim in env.observation_space(env.agents[0]).shape:
        #     flattened_obs_dim *= dim
       
        flattened_obs_dim = 2106
        init_x = (
            jnp.zeros(
                (1, self.num_envs, flattened_obs_dim)
            ),
            jnp.zeros((1, self.num_envs)),
            jnp.zeros((1, self.num_envs, 2, 2)).astype(jnp.int32)
        )
        init_hstate = self.model.initialize_carry(self.num_envs, self.gpu_hidden_dim)
        params = self.model.init(subrng, init_hstate, init_x)

        return params

    def init_carry(self, rng, batch_dims=(1,)):
        return self.model.initialize_carry(rng=rng, batch_dims=batch_dims)

    @partial(jax.jit, static_argnums=(0,))
    def act(self, params, obs, carry=None, reset=None):
        value, logits, carry = self.model.apply(params, obs, carry, reset)

        return value, logits, carry

    @partial(jax.jit, static_argnums=(0,))
    def get_value(self, params, obs, carry=None, reset=None):
        value, _, carry = self.model.apply(params, obs, carry, reset)
        return value, carry

    @partial(jax.jit, static_argnums=(0,))
    def evaluate(self, params, action, obs, carry=None, reset=None):
        value, dist_params, carry = self.model.apply(params, obs, carry, reset)
        dist = self.get_action_dist(dist_params, dtype=action.dtype)
        log_prob = dist.log_prob(action)
        entropy = dist.entropy()

        return value.squeeze(), \
            log_prob.squeeze(), \
            entropy.squeeze(), \
            carry

    def get_action_dist(self, dist_params, dtype=jnp.uint8):
        return tfp.distributions.Categorical(logits=dist_params, dtype=dtype)

    @partial(jax.jit, static_argnums=(0,))
    def update(self, rng, train_state, batch):
        rngs = jax.random.split(rng, self.n_epochs)

        def _scan_epoch(carry, rng):
            brng, urng = jax.random.split(rng)
            batch, train_state = carry
            minibatches = self._get_minibatches(brng, batch)
            train_state, stats = \
                self._update_epoch(
                    urng, train_state, minibatches)

            return (batch, train_state), stats

        (_, train_state), stats = jax.lax.scan(
            _scan_epoch,
            (batch, train_state),
            rngs,
            length=len(rngs)
        )

        stats = jax.tree_util.tree_map(lambda x: x.mean(), stats)
        train_state = train_state.increment_updates()

        return train_state, stats

    @partial(jax.jit, static_argnums=(0,))
    def get_empty_update_stats(self):
        keys = ['total_loss',
                'actor_loss',
                'value_loss',
                'entropy',
                'mean_value',
                'mean_target',
                'mean_gae',
                'grad_norm']

        return OrderedDict({k: -jnp.inf for k in keys})

    @partial(jax.jit, static_argnums=(0,))
    def _update_epoch(
            self,
            rng,
            train_state: TrainState,
            minibatches):

        def _update_minibatch(carry, step):
            rng, minibatch = step
            train_state = carry

            (loss, aux_info), grads = self.grad_fn(
                train_state.params,
                train_state.apply_fn,
                minibatch,
                rng,
            )

            loss_info = (loss,) + aux_info
            loss_info = loss_info + (optax.global_norm(grads),)

            if self.n_devices > 1:
                loss_info = jax.tree_map(
                    lambda x: jax.lax.pmean(x, 'device'), loss_info)
                grads = jax.tree_map(
                    lambda x: jax.lax.pmean(x, 'device'), grads)

            train_state = train_state.apply_gradients(grads=grads)

            stats_def = jax.tree_util.tree_structure(OrderedDict({
                k: 0 for k in [
                    'total_loss',
                    'actor_loss',
                    'value_loss',
                    'entropy',
                    'mean_value',
                    'mean_target',
                    'mean_gae',
                    'grad_norm',
                ]}))

            loss_stats = jax.tree_util.tree_unflatten(
                stats_def, jax.tree_util.tree_leaves(loss_info))

            return train_state, loss_stats

        rngs = jax.random.split(rng, self.n_minibatches)
        train_state, loss_stats = jax.lax.scan(
            _update_minibatch,
            train_state,
            (rngs, minibatches),
            length=self.n_minibatches,
            unroll=self.n_unroll_update
        )

        loss_stats = jax.tree_util.tree_map(
            lambda x: x.mean(axis=0), loss_stats)

        return train_state, loss_stats

    @partial(jax.jit, static_argnums=(0, 2, 4))
    def _loss(self, params, apply_fn, batch, rng=None):

        """
        Elements have batch shape of n_rollout_steps x n_envs//n_minibatches
        """
        carry = jax.tree_util.tree_map(lambda x: x[0, :], batch.carry)
        obs, action, rewards, dones, log_pi_old, value_old, target, gae, carry_old = batch

        dones = dones.at[1:, :].set(dones[:-1, :])
        dones = dones.at[0, :].set(False)
        _batch = batch._replace(dones=dones)

        # Returns LxB and LxBxH tensors
        obs, action, _, done, _, _, _, _, _ = _batch
        value, log_pi, entropy, carry = apply_fn(
            params, action, obs, carry, done)

        # CALCULATE VALUE LOSS
        value_pred_clipped = value_old + (value - value_old).clip(
            -self.clip_eps, self.clip_eps
        )
        value_losses = jnp.square(value - target)
        value_losses_clipped = jnp.square(value_pred_clipped - target)
        value_loss = 0.5 * \
            jnp.maximum(value_losses, value_losses_clipped).mean()


        if self.model.value_ensemble_size > 1:
            gae = gae.at[..., 0].get()

        # CALCULATE ACTOR LOSS
        ratio = jnp.exp(log_pi - log_pi_old)
        norm_gae = (gae - gae.mean()) / (gae.std() + 1e-5)
        loss_actor1 = ratio * norm_gae
        loss_actor2 = jnp.clip(ratio, 1.0 - self.clip_eps,
                               1.0 + self.clip_eps) * norm_gae
        loss_actor = -jnp.minimum(loss_actor1, loss_actor2).mean()

        entropy = entropy.mean()

        total_loss = (
            loss_actor + self.value_loss_coef*value_loss - self.entropy_coef*entropy
        )

        return total_loss, (
            loss_actor,
            value_loss,
            entropy,
            value.mean(),
            target.mean(),
            gae.mean()
        )

    @partial(jax.jit, static_argnums=0)
    def _get_minibatches(self, rng, batch):
        # get dims based on dones
        n_rollout_steps, n_envs = batch.dones.shape[0:2]
        """
        Reshape elements into a batch shape of 
        n_minibatches x n_envs//n_minibatches x n_rollout_steps.
        """
        assert n_envs % self.n_minibatches == 0, \
            'Number of environments must be divisible into number of minibatches.'

        n_env_per_minibatch = n_envs//self.n_minibatches
        shuffled_idx = jax.random.permutation(rng, jnp.arange(n_envs))

        shuffled_batch = jax.tree_util.tree_map(
            lambda x: jnp.take(x, shuffled_idx, axis=1), batch)

        minibatches = jax.tree_util.tree_map(
            lambda x: x.swapaxes(0, 1).reshape(
                self.n_minibatches,
                n_env_per_minibatch,
                n_rollout_steps,
                *x.shape[2:]
            ).swapaxes(1, 2), shuffled_batch)

        return minibatches
