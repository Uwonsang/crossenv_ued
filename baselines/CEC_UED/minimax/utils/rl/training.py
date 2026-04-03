"""
Copyright (c) Meta Platforms, Inc. and affiliates.
All rights reserved.

This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.
"""

from typing import Any, Callable

import jax
import jax.numpy as jnp
from flax import core
from flax import struct
import optax
import chex

from .plr import PLRBuffer


class VmapMAPPOTrainState(struct.PyTreeNode):
    n_iters: chex.Array
    n_updates: chex.Array  # per agent
    n_grad_updates: chex.Array  # per agent
    actor_apply_fn: Callable = struct.field(pytree_node=False)
    actor_params: core.FrozenDict[str, Any]
    actor_tx: optax.GradientTransformation = struct.field(pytree_node=False)
    actor_opt_state: optax.OptState

    critic_apply_fn: Callable = struct.field(pytree_node=False)
    critic_params: core.FrozenDict[str, Any]
    critic_tx: optax.GradientTransformation = struct.field(pytree_node=False)
    critic_opt_state: optax.OptState

    shaped_reward_coeff: float = 0.0

    plr_buffer: PLRBuffer = None

    def apply_gradients(self, *, actor_grads, critic_grads, **kwargs):
        # Actor update
        actor_updates, new_actor_opt_state = self.actor_tx.update(
            actor_grads, self.actor_opt_state, self.actor_params)
        new_actor_params = optax.apply_updates(
            self.actor_params, actor_updates)

        # Critic update
        critic_updates, new_critic_opt_state = self.critic_tx.update(
            critic_grads, self.critic_opt_state, self.critic_params)
        new_critic_params = optax.apply_updates(
            self.critic_params, critic_updates)

        return self.replace(
            n_grad_updates=self.n_updates + 1,
            actor_params=new_actor_params,
            actor_opt_state=new_actor_opt_state,
            critic_params=new_critic_params,
            critic_opt_state=new_critic_opt_state,
            **kwargs,
        )

    @classmethod
    def create(
        cls, *,
        actor_apply_fn,
        actor_params,
        actor_tx,
        critic_apply_fn,
        critic_params,
        critic_tx,
        **kwargs
    ):
        actor_opt_state = jax.vmap(actor_tx.init)(actor_params)
        critic_opt_state = jax.vmap(critic_tx.init)(critic_params)
        return cls(
            n_iters=jnp.array(jax.vmap(lambda x: 0)(
                actor_params), dtype=jnp.uint32),
            n_updates=jnp.array(jax.vmap(lambda x: 0)
                                (actor_params), dtype=jnp.uint32),
            n_grad_updates=jnp.array(
                jax.vmap(lambda x: 0)(actor_params), dtype=jnp.uint32),
            actor_apply_fn=actor_apply_fn,
            actor_params=actor_params,
            actor_tx=actor_tx,
            actor_opt_state=actor_opt_state,
            critic_apply_fn=critic_apply_fn,
            critic_params=critic_params,
            critic_tx=critic_tx,
            critic_opt_state=critic_opt_state,
            **kwargs,
        )

    def increment(self):
        return self.replace(
            n_iters=self.n_iters + 1,
        )

    def increment_updates(self):
        return self.replace(
            n_updates=self.n_updates + 1,
        )

    @property
    def state_dict(self):
        return dict(
            n_iters=self.n_iters,
            n_updates=self.n_updates,
            n_grad_updates=self.n_grad_updates,
            actor_params=self.actor_params,
            actor_opt_state=self.actor_opt_state,
            critic_params=self.critic_params,
            critic_opt_state=self.critic_opt_state,
        )
    
    def set_new_shaped_reward_coeff(self, new_coeff):
        return self.replace(
            shaped_reward_coeff=new_coeff
        )

    def load_state_dict(self, state):
        return self.replace(
            n_iters=state['n_iters'],
            n_updates=state['n_updates'],
            n_grad_updates=state['n_grad_updates'],
            actor_params=state['actor_params'],
            actor_opt_state=state['actor_opt_state'],
            critic_params=state['critic_params'],
            critic_opt_state=state['critic_opt_state'],
        )


class VmapTrainState(struct.PyTreeNode):
    n_iters: chex.Array
    n_updates: chex.Array  # per agent
    n_grad_updates: chex.Array  # per agent
    apply_fn: Callable = struct.field(pytree_node=False)
    params: core.FrozenDict[str, Any]
    tx: optax.GradientTransformation = struct.field(pytree_node=False)
    opt_state: optax.OptState
    plr_buffer: PLRBuffer = None

    def apply_gradients(self, *, grads, **kwargs):
        updates, new_opt_state = self.tx.update(
            grads, self.opt_state, self.params)
        new_params = optax.apply_updates(self.params, updates)

        return self.replace(
            n_grad_updates=self.n_updates + 1,
            params=new_params,
            opt_state=new_opt_state,
            **kwargs,
        )

    @classmethod
    def create(cls, *,
               apply_fn,
               params,
               tx,
               **kwargs
               ):
        opt_state = jax.vmap(tx.init)(params)
        return cls(
            n_iters=jnp.array(jax.vmap(lambda x: 0)(params), dtype=jnp.uint32),
            n_updates=jnp.array(jax.vmap(lambda x: 0)
                                (params), dtype=jnp.uint32),
            n_grad_updates=jnp.array(
                jax.vmap(lambda x: 0)(params), dtype=jnp.uint32),
            apply_fn=apply_fn,
            params=params,
            tx=tx,
            opt_state=opt_state,
            **kwargs,
        )

    def increment(self):
        return self.replace(
            n_iters=self.n_iters + 1,
        )

    def increment_updates(self):
        return self.replace(
            n_updates=self.n_updates + 1,
        )

    @property
    def state_dict(self):
        return dict(
            n_iters=self.n_iters,
            n_updates=self.n_updates,
            n_grad_updates=self.n_grad_updates,
            params=self.params,
            opt_state=self.opt_state
        )

    def load_state_dict(self, state):
        return self.replace(
            n_iters=state['n_iters'],
            n_updates=state['n_updates'],
            n_grad_updates=state['n_grad_updates'],
            params=state['params'],
            opt_state=state['opt_state']
        )
