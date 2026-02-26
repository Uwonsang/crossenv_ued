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
import yaml
import h5py
from tqdm import tqdm
import pickle

def load_h5(path, config):
    with h5py.File(path, "r") as f:
        return {k: f[k][:].reshape(-1, *f[k].shape[2:]) for k in f.keys()}

def split_dataset(dataset, validation_ratio):
    num_data = len(dataset)
    random_indices = np.random.permutation(np.arange(num_data))
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
    steps_per_epoch = n_train_samples // train_batch_size

    def train(rng):
        model = VAE(input_shape, config=config)
        rng, _rng = jax.random.split(rng)
        params = model.init(_rng, jnp.zeros((1, *input_shape)), jax.random.PRNGKey(0))
        train_state = TrainState.create(apply_fn=model.apply, params=params, tx=optax.adam(config["LR"]))

        def vae_loss(params, apply_fn, x, rng):
            logits, mean, std = apply_fn(params, x, rng)
            x_flat      = x.reshape((x.shape[0], -1))
            logits_flat = logits.reshape((logits.shape[0], -1))
            recon_loss  = jnp.mean(optax.sigmoid_binary_cross_entropy(logits_flat, x_flat))
            kl_loss     = jnp.mean(0.5 * jnp.mean(-2 * jnp.log(std) - 1.0 + std**2 + mean**2, axis=-1))
            return recon_loss + 0.1 * kl_loss, (recon_loss, kl_loss)

        jit_update = jax.jit(jax.value_and_grad(vae_loss, has_aux=True), static_argnums=(1,))
        jit_eval   = jax.jit(vae_loss, static_argnums=(1,))

        for epoch in tqdm(range(num_epochs)):
            perm = np.random.permutation(n_train_samples)
            epoch_losses, epoch_recons, epoch_kls = [], [], []

            for step in tqdm(range(steps_per_epoch)):
                idx = perm[step * train_batch_size : (step + 1) * train_batch_size]
                train_batch = jnp.array(train_data[idx])

                rng, _rng_vae = jax.random.split(rng)
                (loss, (recon_loss, kl_loss)), grads = jit_update(
                    train_state.params, train_state.apply_fn, train_batch, _rng_vae
                )
                train_state = train_state.apply_gradients(grads=grads)
                epoch_losses.append(float(loss))
                epoch_recons.append(float(recon_loss))
                epoch_kls.append(float(kl_loss))

            # test eval (epoch당 1회, 전체 test 순회)
            # test_steps = n_test_samples // test_batch_size
            # test_perm  = np.random.permutation(n_test_samples)
            # test_losses, test_recons, test_kls = [], [], []

            # for step in range(test_steps):
            #     idx = test_perm[step * test_batch_size : (step + 1) * test_batch_size]
            #     test_batch = jnp.array(test_data[idx])
            #     rng, _rng_test = jax.random.split(rng)
            #     test_loss, (test_recon, test_kl) = jit_eval(
            #         train_state.params, test_batch, _rng_test
            #     )
            #     test_losses.append(float(test_loss))
            #     test_recons.append(float(test_recon))
            #     test_kls.append(float(test_kl))

            wandb.log({
                "loss": np.mean(epoch_losses), "recon_loss": np.mean(epoch_recons), "kl_loss": np.mean(epoch_kls),
                # "test_loss": np.mean(test_losses), "test_recon_loss": np.mean(test_recons), "test_kl_loss": np.mean(test_kls),
                "epoch": epoch,
            })

            if epoch % 10 == 0:
                print(
                    f"Epoch {epoch} | Loss: {np.mean(epoch_losses):.4f} | Recon: {np.mean(epoch_recons):.4f} | "
                    f"KL: {np.mean(epoch_kls):.4f}"
                )
                # | Test Loss: {np.mean(test_losses):.4f

        return {"train_state": train_state, "key": rng}

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
    rng, rng_train = jax.random.split(rng)

    data = load_h5(config["LAYOUT_DATA_PATH"], config)
    images_train, images_test = split_dataset(
        data["agent_0"], config["VALIDATION_RATIO"])

    train_fn = make_train(config, images_train, images_test)
    out = train_fn(rng_train)
    
    # after training, save the model
    os.makedirs("baselines/CEC_UED/VAE/checkpoints", exist_ok=True)
    with open(f"baselines/CEC_UED/VAE/checkpoints/vae_seed{config['SEED']}.pkl", "wb") as f:
        pickle.dump({'params': out["train_state"], 'key': out["key"]}, f)
    print("Done.")

if __name__ == "__main__":
    main()