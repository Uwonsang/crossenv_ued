import jax
import jax.numpy as jnp

import flax.linen as nn
from flax.linen.initializers import constant, orthogonal

import numpy as np
import optax

class Encoder(nn.Module):

    @nn.compact
    def __call__(self, x):

        x = nn.Conv(
            features=32,
            kernel_size=(3, 3),
            strides=(1, 1),
            kernel_init=orthogonal(np.sqrt(2)),
            bias_init=constant(0.0),
        )(x)
        x = nn.relu(x)

        x = nn.Conv(
            features=64,
            kernel_size=(3, 3),
            strides=(2, 2),
            kernel_init=orthogonal(np.sqrt(2)),
            bias_init=constant(0.0),
        )(x)
        x = nn.relu(x)

        x = nn.Conv(
            features=64,
            kernel_size=(3, 3),
            strides=(2, 2),
            kernel_init=orthogonal(np.sqrt(2)),
            bias_init=constant(0.0),
        )(x)
        x = nn.relu(x)
        x = x.reshape((x.shape[0], -1))
        
        mean = nn.Dense(16, kernel_init=orthogonal(1.0), bias_init=constant(0.0))(x)
        logvar = nn.Dense(16, kernel_init=orthogonal(1.0), bias_init=constant(0.0))(x)
        return mean, logvar

class Decoder(nn.Module):
    
    @nn.compact
    def __call__(self, z):
        z = nn.Dense(3 * 3 * 64, kernel_init=orthogonal(1.0), bias_init=constant(0.0))(z)
        z = nn.relu(z)
        z = z.reshape((z.shape[0], 3, 3, 64)) # (B,3,3,8)

        z = nn.ConvTranspose(
            features=64,
            kernel_size=(5, 5),
            strides=(2, 2),
            padding='VALID',              
            kernel_init=orthogonal(np.sqrt(2)),
            bias_init=constant(0.0),
        )(z)
        z = nn.relu(z) # (B,5,5,64)

        z = nn.ConvTranspose(
            features=15,
            kernel_size=(3, 3),
            strides=(1, 1),
            padding='SAME',
            kernel_init=orthogonal(np.sqrt(2)),
            bias_init=constant(0.0),
        )(z)
        
        return z

class VAE(nn.Module):
    image_shape: int
    config: dict

    @nn.compact
    def __call__(self, x, rng):
        # x: (B,9,9,26)
        mean, logvar = Encoder()(x)
        std = jnp.exp(0.5 * logvar)

        rng, _rng = jax.random.split(rng)
        z = mean + std * jax.random.normal(_rng, mean.shape)

        recon = Decoder()(z) 

        return recon, mean, logvar

def vae_loss(params, apply_fn, x, rng, beta):
    logits, mean, logvar = apply_fn(params, x, rng)
    x_flat      = x.reshape((x.shape[0], -1))
    logits_flat = logits.reshape((logits.shape[0], -1))
    recon_loss = jnp.mean(jnp.sum(optax.sigmoid_binary_cross_entropy(logits_flat, x_flat), axis=-1)) 
    kl_loss     = jnp.mean(-0.5 * jnp.sum(1.0 + logvar - mean**2 - jnp.exp(logvar), axis=-1))
    loss = recon_loss + beta * kl_loss
    return loss, (recon_loss, kl_loss)