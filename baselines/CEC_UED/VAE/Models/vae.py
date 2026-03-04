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
            features=16,
            kernel_size=(3, 3),
            strides=(1, 1),
            kernel_init=orthogonal(np.sqrt(2)),
            bias_init=constant(0.0),
        )(x)
        x = nn.relu(x)

        x = nn.Conv(
            features=12,
            kernel_size=(3, 3),
            strides=(2, 2),
            kernel_init=orthogonal(np.sqrt(2)),
            bias_init=constant(0.0),
        )(x)
        x = nn.relu(x)

        x = nn.Conv(
            features=8,
            kernel_size=(3, 3),
            strides=(2, 2),
            kernel_init=orthogonal(np.sqrt(2)),
            bias_init=constant(0.0),
        )(x)
        x = nn.relu(x)

        x = x.reshape((x.shape[0], -1))
        mean = nn.Dense(8, kernel_init=orthogonal(1.0), bias_init=constant(0.0))(x)
        log_std = nn.Dense(8, kernel_init=orthogonal(1.0), bias_init=constant(0.0))(x)
        return mean, log_std

class Decoder(nn.Module):
    
    @nn.compact
    def __call__(self, z):
        z = nn.Dense(3 * 3 * 8, kernel_init=orthogonal(1.0), bias_init=constant(0.0))(z)
        z = nn.relu(z)
        z = z.reshape((z.shape[0], 3, 3, 8)) # (B,3,3,8)

        z = nn.ConvTranspose(
            features=12,
            kernel_size=(5, 5),
            strides=(2, 2),
            padding='VALID',              
            kernel_init=orthogonal(np.sqrt(2)),
            bias_init=constant(0.0),
        )(z)
        z = nn.relu(z) # (B,9,9,12)

        z = nn.ConvTranspose(
            features=16,
            kernel_size=(3, 3),
            strides=(1, 1),
            padding='SAME',
            kernel_init=orthogonal(np.sqrt(2)),
            bias_init=constant(0.0),
        )(z)
        z = nn.relu(z) # (B,9,9,16)

        z = nn.Conv(
            features=26,
            kernel_size=(3, 3),
            strides=(1, 1),
            padding='SAME',
            kernel_init=orthogonal(1.0),
            bias_init=constant(0.0),
        )(z) # (B,9,9,26)
        
        return z

class VAE(nn.Module):
    image_shape: int
    config: dict

    @nn.compact
    def __call__(self, x, rng):
        # x: (B,9,9,26)
        mean, log_std = Encoder()(x)
        std = jnp.exp(log_std)

        rng, _rng = jax.random.split(rng)
        z = mean + std * jax.random.normal(_rng, mean.shape)

        recon = Decoder()(z) 

        return recon, mean, std

def vae_loss(params, apply_fn, x, rng, beta):
    logits, mean, std = apply_fn(params, x, rng)
    x_flat      = x.reshape((x.shape[0], -1))
    logits_flat = logits.reshape((logits.shape[0], -1))
    recon_loss  = jnp.mean(optax.sigmoid_binary_cross_entropy(logits_flat, x_flat))
    kl_loss     = jnp.mean(0.5 * jnp.mean(-2 * jnp.log(std) - 1.0 + std**2 + mean**2, axis=-1))
    return recon_loss + beta * kl_loss, (recon_loss, kl_loss)