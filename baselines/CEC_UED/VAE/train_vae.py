import jax
import jax.numpy as jnp
import flax.linen as nn
import numpy as np
import optax
from flax.training.train_state import TrainState
from flax.linen.initializers import constant, orthogonal
import hydra
from omegaconf import OmegaConf


import wandb
from jax_tqdm import scan_tqdm
import yaml
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
import os


class Encoder(nn.Module):
    hidden_size: int
    latent_size: int

    @nn.compact
    def __call__(self, x):
        x = nn.Dense(self.hidden_size, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(x)
        x = nn.relu(x)
        mean = nn.Dense(self.latent_size, kernel_init=orthogonal(1.0), bias_init=constant(0.0))(x)
        log_std = nn.Dense(self.latent_size, kernel_init=orthogonal(1.0), bias_init=constant(0.0))(x)
        return mean, log_std


class Decoder(nn.Module):
    hidden_size: int
    output_size: int

    @nn.compact
    def __call__(self, z):
        z = nn.Dense(self.hidden_size, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(z)
        z = nn.relu(z)
        logits = nn.Dense(self.output_size, kernel_init=orthogonal(1.0), bias_init=constant(0.0))(z)
        return logits


class VAE(nn.Module):
    image_shape: int
    config: int

    @nn.compact
    def __call__(self, x, rng):
        input_size = int(np.prod(self.image_shape))
        x = x.reshape((x.shape[0], -1))

        mean, log_std = Encoder(self.config['HIDDEN_SIZE'], self.config['LATENT_SIZE'])(x)
        std = jnp.exp(log_std)

        rng, _rng = jax.random.split(rng)
        z = mean + std * jax.random.normal(_rng, mean.shape)
       
        logits = Decoder(self.config['HIDDEN_SIZE'], input_size)(z)

        return logits, mean, std


def make_train(config, train_data, test_data):
    """data: jnp.array of shape (N, input_size)"""

    n_samples = train_data.shape[0]
    input_shape = train_data.shape[1:]
    batch_size = config["BATCH_SIZE"]
    num_updates = config["NUM_EPOCHS"]
    
    def train(rng):
        # INIT NETWORK
        model = VAE(input_shape, config=config)
        
        rng, _rng = jax.random.split(rng)
        init_x = jnp.zeros((1, *input_shape))
        init_rng = jax.random.PRNGKey(0)
        params = model.init(_rng, init_x, init_rng)

        tx = optax.adam(config["LR"])
        train_state = TrainState.create(apply_fn=model.apply, params=params, tx=tx)

        @scan_tqdm(num_updates)
        def _update_step(runner_state, update_idx):
            train_state, rng = runner_state

            rng, _rng_batch, _rng_vae = jax.random.split(rng, 3)
            batch_idx = jax.random.randint(_rng_batch, (batch_size,), 0, n_samples)
            batch = train_data[batch_idx]

            def loss_fn(params):
                def vae_loss(params, apply_fn, x, rng):
                    logits, mean, std = apply_fn(params, x, rng)
                    
                    x_flat = x.reshape((x.shape[0], -1))
                    reconstruction_loss = jnp.mean(
                        optax.sigmoid_binary_cross_entropy(logits, x_flat)
                    )
                    kl_loss = jnp.mean(
                        0.5 * jnp.mean(-2 * jnp.log(std) - 1.0 + std ** 2 + mean ** 2, axis=-1)
                    )
                    return reconstruction_loss + 0.1 * kl_loss, (reconstruction_loss, kl_loss)
                return vae_loss(params, train_state.apply_fn, batch, _rng_vae)

            (loss, (recon_loss, kl_loss)), grads = jax.value_and_grad(loss_fn, has_aux=True)(train_state.params)
            train_state = train_state.apply_gradients(grads=grads)

            # evlauation on test data
            rng, _rng_test = jax.random.split(rng)
            test_logits, test_mean, test_std = train_state.apply_fn(train_state.params, test_data, _rng_test)
            test_x_flat = test_data.reshape((test_data.shape[0], -1))
            test_recon = jnp.mean(optax.sigmoid_binary_cross_entropy(test_logits, test_x_flat))
            test_kl = jnp.mean(0.5 * jnp.mean(-2 * jnp.log(test_std) - 1.0 + test_std**2 + test_mean**2, axis=-1))
            test_loss = test_recon + 0.1 * test_kl

            metric = {
                "loss": loss,
                "recon_loss": recon_loss,
                "kl_loss": kl_loss,
                "update_step": update_idx,
                "test_loss": test_loss,
                "test_recon_loss": test_recon,
                "test_kl_loss": test_kl,
            }

            def callback(metric):
                wandb.log({k: float(v) for k, v in metric.items()})
                # Only log every N steps
                PRINT_INTERVAL = 100
                if int(metric["update_step"]) % PRINT_INTERVAL == 0:
                    print(
                        f"Step: {int(metric['update_step'])}, "
                        f"Loss: {metric['loss']:.4f}, "
                        f"Recon Loss: {metric['recon_loss']:.4f}, "
                        f"KL Loss: {metric['kl_loss']:.4f}, "
                        f"Test Loss: {metric['test_loss']:.4f}, "
                        f"Test Recon Loss: {metric['test_recon_loss']:.4f}, "
                        f"Test KL Loss: {metric['test_kl_loss']:.4f}"
                    )

            jax.debug.callback(callback, metric)

            return (train_state, rng), metric

        rng, _rng = jax.random.split(rng)
        runner_state = (train_state, _rng)
        runner_state, metrics = jax.lax.scan(
            _update_step, runner_state, jnp.arange(num_updates), num_updates
        )

        return {"runner_state": runner_state, "metrics": metrics}

    return train

@hydra.main(version_base=None, config_path="../config", config_name="vae_config")
def main(config):
    config = OmegaConf.to_container(config)

    if config["WANDB_MODE"] == "online":
        with open("private.yaml") as f:
            private_info = yaml.load(f, Loader=yaml.FullLoader)
        wandb.login(key=private_info["wandb_key"])
    
    wandb.init(
        entity=config["ENTITY"],
        project=config["PROJECT"],
        tags=["IPPO", "RNN", "SP"],
        config=config,
        mode=config["WANDB_MODE"],
        name=f"VAE_seed{config['SEED']}"
    )

    digits = load_digits()
    images_normed = (digits.images / 16) > 0.5 #불러온 이미지 정규화
    splits = train_test_split(images_normed, random_state=0)
    images_train, images_test = map(jnp.asarray, splits)

    rng = jax.random.PRNGKey(config["SEED"])
    train_fn = jax.jit(make_train(config, images_train, images_test))
    out = train_fn(rng)
    print("Done.")

    # visualize results
    train_state, _ = out["runner_state"]
    model = VAE(images_test.shape[1:], config=config)
    rng_vis = jax.random.PRNGKey(0)
    logits, mean, std = model.apply(train_state.params, images_test, rng_vis)
    images_pred = jax.nn.sigmoid(logits).reshape(-1, *images_test.shape[1:])

    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(2, 10, figsize=(6, 1.5),
                    subplot_kw={'xticks':[], 'yticks':[]},
                    gridspec_kw=dict(hspace=0.1, wspace=0.1))
    for i in range(10):
        ax[0, i].imshow(images_test[i], cmap='binary', interpolation='gaussian')
        ax[1, i].imshow(images_pred[i], cmap='binary', interpolation='gaussian')
    os.makedirs('vae_results', exist_ok=True)
    plt.savefig('vae_results/vae_results.png')


if __name__ == "__main__":
    main()