import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax.training.train_state import TrainState
import hydra
from omegaconf import OmegaConf

import wandb
import yaml
from tqdm import tqdm
import pickle
import os
import jaxmarl

from jaxmarl.viz.overcooked_visualizer import OvercookedVisualizer
from utils import restore_from_obs, split_dataset, DataLoader, concat_images_with_labels, load_h5
from map_viz import FilteredState
from Models.vqvae import VQVAE, vqvae_loss


def make_train(config, train_data, test_data):
    # for viz
    env = jaxmarl.make(config["ENV_NAME"], **config["ENV_KWARGS"])
    agent_view_size = env.agent_view_size
    viz = OvercookedVisualizer()

    n_train_samples = train_data.shape[0]
    n_test_samples = test_data.shape[0]
    input_shape = train_data.shape[1:]
    num_epochs = config["NUM_EPOCHS"]

    train_loader = DataLoader(train_data, config["BATCH_SIZE_TRAIN"], shuffle=True)
    test_loader = DataLoader(test_data, config["BATCH_SIZE_TEST"], shuffle=True)

    def train(rng):
        model = VQVAE(config=config)
        rng, _rng = jax.random.split(rng)
        params = model.init(_rng, jnp.zeros((1, *input_shape)), jax.random.PRNGKey(0))
        train_state = TrainState.create(apply_fn=model.apply, params=params, tx=optax.adam(config["LR"]))

        jit_update = jax.jit(jax.value_and_grad(vqvae_loss, has_aux=True), static_argnums=(1,))
        jit_eval   = jax.jit(vqvae_loss, static_argnums=(1,))

        for epoch in tqdm(range(num_epochs), desc="Epochs"):
            epoch_losses, epoch_recons, epoch_kls = [], [], []
            epoch_test_losses, epoch_test_recons, epoch_test_kls = [], [], []

            for train_batch in tqdm(train_loader, desc="Training"):
                rng, _rng_vae = jax.random.split(rng)
                (loss, (recon_loss, kl_loss)), grads = jit_update(
                    train_state.params, train_state.apply_fn, train_batch, _rng_vae)

                train_state = train_state.apply_gradients(grads=grads)
                epoch_losses.append(float(loss))
                epoch_recons.append(float(recon_loss))
                epoch_kls.append(float(kl_loss))
            
            wandb.log({
                "loss": np.mean(epoch_losses), "recon_loss": np.mean(epoch_recons), "kl_loss": np.mean(epoch_kls)}, step=epoch)

            if epoch % config["validation_freq"] == 0 and epoch != 0:
                for test_idx, (test_batch) in enumerate(tqdm(test_loader, desc="Testing")):
                    rng, _rng_test = jax.random.split(rng)
                    test_loss, (test_recon, test_kl) = jit_eval(train_state.params, train_state.apply_fn, test_batch, _rng_test)
                    epoch_test_losses.append(float(test_loss))
                    epoch_test_recons.append(float(test_recon))
                    epoch_test_kls.append(float(test_kl))


                    if epoch % config["render_freq"] == 0 and epoch != 0 and test_idx == 0:
                        restore_from_obs_batch = jax.vmap(restore_from_obs, in_axes=0)
                        
                        restored_gt = restore_from_obs_batch(test_batch)
                        test_states = FilteredState(**restored_gt)
                        test_frame = [
                            viz.custom_get_frame(jax.tree_map(lambda x: x[step], test_states), agent_view_size)
                            for step in range(config["n_render_samples"])]
                        
                        # VAE_render
                        vae_render, _, _ = train_state.apply_fn(train_state.params, test_batch, _rng_test)
                        vae_render = (jax.nn.sigmoid(vae_render) > 0.5).astype(jnp.uint8)
                        restored_pred = restore_from_obs_batch(vae_render)
                        test_states_pred = FilteredState(**restored_pred)
                        
                        vae_frame = [
                            viz.custom_get_frame(jax.tree_map(lambda x: x[step], test_states_pred), agent_view_size)
                            for step in range(config["n_render_samples"])]

                        combined = concat_images_with_labels([test_frame, vae_frame])
                                            
                        for i, canvas in enumerate(combined):
                            wandb.log({f"comparison_{i:03d}": wandb.Image(canvas)}, step=epoch)
                
                wandb.log({
                "test_loss": np.mean(epoch_test_losses), "test_recon_loss": np.mean(epoch_test_recons), 
                            "test_kl_loss": np.mean(epoch_test_kls)}, step=epoch)


        return {"train_state": train_state, "key": rng}

    return train

@hydra.main(version_base=None, config_path="config", config_name="vqvae_config")
def main(config):
    config = OmegaConf.to_container(config)

    if config["WANDB_MODE"] == "online":
        with open("private.yaml") as f:
            private_info = yaml.load(f, Loader=yaml.FullLoader)
        wandb.login(key=private_info["wandb_key"])

    name = f"VQVAE_layout_{config['LAYOUT_DATA_FILE'].replace('.h5', '').split('layout_dataset_')[1]}_seed{config['SEED']}"
    
    wandb.init(
        entity=config["ENTITY"],
        project=config["PROJECT"],
        tags=["IPPO", "RNN", "SP"],
        config=config,
        mode=config["WANDB_MODE"],
        name=name
    )

    rng = jax.random.PRNGKey(config["SEED"])
    rng, rng_train = jax.random.split(rng)

    data_path = os.path.join(config["LAYOUT_DATA_PATH"], config["LAYOUT_DATA_FILE"])
    data = load_h5(data_path)
    images_train, images_test = split_dataset(
        data["agent_0"], config["VALIDATION_RATIO"])

    train_fn = make_train(config, images_train, images_test)
    out = train_fn(rng_train)
    
    # after training, save the model
    os.makedirs("/app/baselines/CEC_UED/VAE/checkpoints", exist_ok=True)
    with open(f"/app/baselines/CEC_UED/VAE/checkpoints/vae_seed{config['SEED']}.pkl", "wb") as f:
        pickle.dump({'params': out["train_state"].params, 'config': config, 'key': out["key"]}, f)
    print("Done.")

if __name__ == "__main__":
    main()