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
import h5py

def load_h5(path, config):
    with h5py.File(path, "r") as f:
        return {k: f[k][:1000].reshape(-1, *f[k].shape[2:]) for k in f.keys()}

def split_dataset(dataset, validation_ratio, rng):
    num_data = len(dataset)
    random_indices = jax.random.permutation(rng, jnp.arange(num_data))
    validation_split = int(validation_ratio * num_data)
    train_indices = random_indices[:-validation_split]
    validation_indices = random_indices[-validation_split:]
    return dataset[train_indices], dataset[validation_indices]

class Encoder(nn.Module):
    latent_size: int

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

        x = x.reshape((x.shape[0], -1))  # (B, 72)
        mean    = nn.Dense(self.latent_size, kernel_init=orthogonal(1.0), bias_init=constant(0.0))(x)
        log_std = nn.Dense(self.latent_size, kernel_init=orthogonal(1.0), bias_init=constant(0.0))(x)
        return mean, log_std

class Decoder(nn.Module):
    @nn.compact
    def __call__(self, z):
        z = nn.Dense(3 * 3 * 8, kernel_init=orthogonal(1.0), bias_init=constant(0.0))(z)
        z = nn.relu(z)
        z = z.reshape((z.shape[0], 3, 3, 8))           # (B,3,3,8)

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
        mean, log_std = Encoder(self.config['LATENT_SIZE'])(x)
        std = jnp.exp(log_std)

        rng, _rng = jax.random.split(rng)
        z = mean + std * jax.random.normal(_rng, mean.shape)

        recon = Decoder()(z) 

        return recon, mean, std


def make_train(config, train_data, test_data):

    n_train_samples = train_data.shape[0]
    n_test_samples = test_data.shape[0]
    input_shape = train_data.shape[1:]
    train_batch_size = config["BATCH_SIZE_TRAIN"]
    test_batch_size = config["BATCH_SIZE_TEST"]
    num_epochs = config["NUM_EPOCHS"]
    
    def train(rng):
        # INIT NETWORK
        model = VAE(input_shape, config=config)
        
        rng, _rng = jax.random.split(rng)
        init_x = jnp.zeros((1, *input_shape))
        init_rng = jax.random.PRNGKey(0)
        params = model.init(_rng, init_x, init_rng)

        tx = optax.adam(config["LR"])
        train_state = TrainState.create(apply_fn=model.apply, params=params, tx=tx)

        @scan_tqdm(num_epochs)
        def _update_step(runner_state, epoch_idx):
            train_state, rng = runner_state

            rng, _rng_train_batch, _rng_vae = jax.random.split(rng, 3)
            train_batch_idx = jax.random.randint(_rng_train_batch, (train_batch_size,), 0, n_train_samples) 
            train_batch = train_data[train_batch_idx]

            def vae_loss(params, apply_fn, x, rng):
                logits, mean, std = apply_fn(params, x, rng)

                x_flat = x.reshape((x.shape[0], -1))
                logits_flat = logits.reshape((logits.shape[0], -1))

                reconstruction_loss = jnp.mean(
                    optax.sigmoid_binary_cross_entropy(logits_flat, x_flat)
                )
                kl_loss = jnp.mean(
                    0.5 * jnp.mean(-2 * jnp.log(std) - 1.0 + std**2 + mean**2, axis=-1)
                )
                return reconstruction_loss + 0.1 * kl_loss, (reconstruction_loss, kl_loss)

            (loss, (recon_loss, kl_loss)), grads = jax.value_and_grad(vae_loss, has_aux=True)(
                train_state.params, train_state.apply_fn, train_batch, _rng_vae
                )
            train_state = train_state.apply_gradients(grads=grads)

            # evlauation on test data
            rng, _rng_test_batch, _rng_test_vae = jax.random.split(rng, 3)
            test_batch_idx = jax.random.randint(_rng_test_batch, (test_batch_size,), 0, n_test_samples) 
            test_batch = test_data[test_batch_idx]

            test_logits, test_mean, test_std = train_state.apply_fn(train_state.params, test_batch, _rng_test_vae)
            test_x_flat = test_batch.reshape((test_batch.shape[0], -1))
            test_logits_flat = test_logits.reshape((test_logits.shape[0], -1))  # 추가
            test_recon = jnp.mean(optax.sigmoid_binary_cross_entropy(test_logits_flat, test_x_flat))
            test_kl = jnp.mean(0.5 * jnp.mean(-2 * jnp.log(test_std) - 1.0 + test_std**2 + test_mean**2, axis=-1))
            test_loss = test_recon + 0.1 * test_kl

            metric = {
                "loss": loss,
                "recon_loss": recon_loss,
                "kl_loss": kl_loss,
                "epoch_idx": epoch_idx,
                "test_loss": test_loss,
                "test_recon_loss": test_recon,
                "test_kl_loss": test_kl,
            }

            def callback(metric):
                wandb.log({k: float(v) for k, v in metric.items()})
                # Only log every N steps
                PRINT_INTERVAL = 100
                if int(metric["epoch_idx"]) % PRINT_INTERVAL == 0:
                    print(
                        f"Step: {int(metric['epoch_idx'])}, "
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
            _update_step, runner_state, jnp.arange(num_epochs), num_epochs
        )

        return {"runner_state": runner_state, "metrics": metrics}

    return train

@hydra.main(version_base=None, config_path="config", config_name="vae_config")
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

    rng = jax.random.PRNGKey(config["SEED"])
    rng_data, rng_train = jax.random.split(rng)

    data = load_h5(config["LAYOUT_DATA_PATH"], config)
    full_dataset = jnp.asarray(data["agent_0"])
    images_train, images_test = split_dataset(
        full_dataset, config["VALIDATION_RATIO"], rng=rng_data)

    train_fn = jax.jit(make_train(config, images_train, images_test))
    out = train_fn(rng_train)
    print("Done.")

    # visualize results
    # train_state, _ = out["runner_state"]
    # model = VAE(images_test.shape[1:], config=config)
    # rng_vis = jax.random.PRNGKey(0)
    # logits, mean, std = model.apply(train_state.params, images_test, rng_vis)
    # images_pred = jax.nn.sigmoid(logits).reshape(-1, *images_test.shape[1:])


if __name__ == "__main__":
    main()