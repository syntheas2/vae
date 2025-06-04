import _init # Your local package structure if needed
from typing import List, Dict, Any, Tuple, Annotated
import torch
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
from zenml import step, get_step_context # Keep ZenML decorator and context
from zenml.logger import get_logger # ZenML's logger
import mlflow
import mlflow.pytorch
from tqdm import tqdm
from zenml.client import Client
import time
import numpy as np
from pathlib import Path
import os
from zenml import register_artifact
from pythelpers.ml.mlflow import load_latest_checkpoint2, load_best_model, rotate_checkpoints, remove_all_from_artifact_dir, rotate_bestmodels
import tempfile # For temporary file handling

# Assuming these are correctly importable from your project structure
from vae_syntheas.model import Model_VAE , Encoder_model, Decoder_model
from vae_syntheas.main import compute_loss # Assuming this is your loss function
from utils_train import TabularDataset # Ensure this is findable
from pipelines.train_vae_args import VAEArgs # Your configuration class

logger = get_logger(__name__) # Use ZenML's logger for the step

ReturnType = Tuple[
    Annotated[Model_VAE, "best_model"],
    Annotated[torch.nn.Module, "encoder"],
    Annotated[torch.nn.Module, "decoder"],
    Annotated[np.ndarray, "train_z_latents"],
]

@step(enable_cache=False) # ZenML step decorator
def train_evaluate_vae(
    X_train_num: torch.Tensor,
    X_train_cat: torch.Tensor,
    X_val_num: torch.Tensor,
    X_val_cat: torch.Tensor,
    d_numerical: int,
    categories: List[int],
    config: VAEArgs, # Your configuration class
) -> ReturnType: # Returns path to the best model's state_dict (ZenML output)
    """
    Initializes, trains, and evaluates the VAE model.
    Saves resumable model checkpoints per epoch to MLflow artifacts with rotation.
    Saves the best model state_dict as a ZenML artifact.
    """
    context = get_step_context()
    current_run = mlflow.active_run()
    if not current_run:
        # This can happen if autolog starts a run implicitly and we try to access it too soon
        # or if running outside a `with mlflow.start_run():` block when not relying on autolog to create it.
        # For robust explicit run management, you'd wrap the core logic in `with mlflow.start_run() as run:`.
        # However, ZenML + autolog usually handles run creation. If this becomes an issue, explicit run start is needed.
        logger.info("No active MLflow run found initially, will rely on autolog or subsequent calls to create/get it.")
        # Attempt to get it again, as autolog might initialize it.
        # If this is still None later when needed, it's an issue.
        # For now, we assume autolog handles it or it's available when `load_latest_checkpoint_from_mlflow` is called.
    
    run_id = current_run.info.run_id if current_run else None


    mlflow.pytorch.autolog(
        log_models=True, 
        checkpoint=True, 
        disable_for_unsupported_versions=True,
        registered_model_name=None
    )

    logger.info(f"Starting VAE training on device: {config.device}. MLflow autologging enabled.")
    logger.info(f"ZenML step logs can be found in your ZenML dashboard or console output for this step.")

    # Log all config parameters manually
    mlflow.log_params(config.model_dump())

    X_val_num = X_val_num.to(config.device)
    X_val_cat = X_val_cat.to(config.device)
    train_dataset = TabularDataset(X_train_num, X_train_cat)
    train_loader = DataLoader(
        train_dataset, batch_size=config.batch_size, shuffle=True,
        num_workers=config.num_workers, pin_memory=(config.device == "cuda")
    )

    model = Model_VAE(
        num_layers=config.num_layers, d_numerical=d_numerical, categories=categories,
        d_token=config.d_token, n_head=config.n_head, factor=config.factor, bias=config.token_bias
    ).to(config.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=config.scheduler_factor, patience=config.scheduler_patience)

    start_epoch = 0
    best_val_loss = float('inf')
    current_patience = 0
    beta = config.max_beta
    
    manual_checkpoint_subdir = "manual_model_checkpoints" # Define consistently
    best_model_data_to_save = None

    if config.load_from_checkpoint:
        ckp_run_id = config.load_ckp_from_run_id
        if ckp_run_id:
            logger.info(f"Attempting to load checkpoint from MLflow run '{ckp_run_id}', subdir '{manual_checkpoint_subdir}'...")
            checkpoint_data = load_latest_checkpoint2(
                run_id=ckp_run_id,
                artifact_subdir=manual_checkpoint_subdir,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                device=config.device
            )
            if checkpoint_data:
                start_epoch = checkpoint_data.get('epoch', -1) + 1 # Resume from NEXT epoch
                best_val_loss = checkpoint_data.get('best_val_loss', float('inf'))
                current_patience = checkpoint_data.get('current_patience', 0)
                beta = checkpoint_data.get('beta', config.max_beta)
                logger.info(f"Successfully loaded checkpoint. Resuming training from epoch {start_epoch}.")
                best_model_data_to_save = load_best_model(
                    run_id=config.bestmodels_runid,
                    artifact_subdir=config.manual_bestmodel_subdir,
                    metric='loss',
                    device=config.device
                )
                logger.info(f"Successfully loaded best model")
                # Model, optimizer, scheduler states are loaded by the helper
            else:
                logger.info("No suitable checkpoint found. Starting training from scratch.")
        else:
            logger.warning("MLflow run ID not available, cannot attempt to load checkpoint.")

    pre_encoder = Encoder_model(config.num_layers, d_numerical, categories, d_token=config.d_token, n_head = config.n_head, factor = config.factor).to(config.device)
    pre_decoder = Decoder_model(config.num_layers, d_numerical, categories, d_token=config.d_token, n_head = config.n_head, factor = config.factor).to(config.device)

    pre_encoder.eval()
    pre_decoder.eval()

    logger.info(f"Training for {config.num_epochs} epochs, starting from epoch {start_epoch}.")
    training_start_time = time.time()
    best_model_changed = False

    for epoch in range(start_epoch, config.num_epochs):
        model.train()
        # ... (rest of your training loop for one epoch: pbar, loss calculation, optimizer.step)
        epoch_loss_mse, epoch_loss_ce, epoch_loss_kld = 0.0, 0.0, 0.0
        epoch_train_acc_sum = 0.0
        processed_samples = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{config.num_epochs}")
        for batch_num, batch_cat in pbar:
            batch_num = batch_num.to(config.device)
            batch_cat = batch_cat.to(config.device)
            
            optimizer.zero_grad()
            Recon_X_num, Recon_X_cat, mu_z, std_z = model(batch_num, batch_cat)
            loss_mse, loss_ce, loss_kld, current_batch_train_acc = compute_loss(
                batch_num, batch_cat, Recon_X_num, Recon_X_cat, mu_z, std_z
            )
            
            total_loss = loss_mse + loss_ce + beta * loss_kld
            total_loss.backward()
            optimizer.step()

            batch_len = batch_num.shape[0]
            epoch_loss_mse += loss_mse.item() * batch_len
            epoch_loss_ce += loss_ce.item() * batch_len
            epoch_loss_kld += loss_kld.item() * batch_len
            epoch_train_acc_sum += current_batch_train_acc.item() * batch_len
            processed_samples += batch_len
            pbar.set_postfix({
                "MSE": loss_mse.item(), "CE": loss_ce.item(), "KL": loss_kld.item(), 
                "Beta*KL": (beta * loss_kld).item(), "TrainAcc": current_batch_train_acc.item()
            })

        avg_epoch_loss_mse = epoch_loss_mse / processed_samples if processed_samples > 0 else 0
        avg_epoch_loss_ce = epoch_loss_ce / processed_samples if processed_samples > 0 else 0
        avg_epoch_loss_kld = epoch_loss_kld / processed_samples if processed_samples > 0 else 0
        avg_epoch_train_acc = epoch_train_acc_sum / processed_samples if processed_samples > 0 else 0

        model.eval()
        with torch.no_grad():
            Recon_X_val_num, Recon_X_val_cat, mu_val_z, std_val_z = model(X_val_num, X_val_cat)
            val_mse, val_ce, val_kl, val_acc = compute_loss(
                X_val_num, X_val_cat, Recon_X_val_num, Recon_X_val_cat, mu_val_z, std_val_z
            )
            current_val_loss = val_ce.item() 

        scheduler.step(current_val_loss)
        
        logger.info(
            f"Epoch {epoch+1}: Train MSE: {avg_epoch_loss_mse:.4f}, CE: {avg_epoch_loss_ce:.4f}, KL: {avg_epoch_loss_kld:.4f} | "
            f"Val MSE: {val_mse.item():.4f}, CE: {val_ce.item():.4f}, KL: {val_kl.item():.4f} | "
            f"Train ACC: {avg_epoch_train_acc:.4f}, Val ACC: {val_acc.item():.4f} | Beta: {beta:.6f}"
        )
        
        mlflow.log_metrics({
            # ... your metrics ...
            "train_mse_loss_epoch": avg_epoch_loss_mse, "train_ce_loss_epoch": avg_epoch_loss_ce,
            "train_kl_divergence_epoch": avg_epoch_loss_kld, "train_accuracy_epoch": avg_epoch_train_acc,
            "val_mse_loss_epoch": val_mse.item(), "val_ce_loss_epoch": val_ce.item(),
            "val_kl_divergence_epoch": val_kl.item(), "val_accuracy_epoch": val_acc.item(),
            "beta_value": beta, "learning_rate": optimizer.param_groups[0]['lr']
        }, step=epoch)

        # --- Best Model Logic (for ZenML output) ---
        if current_val_loss < best_val_loss:
            best_val_loss = current_val_loss
            current_patience = 0
            
            best_model_data_to_save = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'validation_loss': current_val_loss,
                # Add any other specific metadata you want for the "best model" artifact
            }
            best_model_changed = True
        else:
            current_patience += 1
            if current_patience >= config.early_stopping_patience:
                if beta > config.min_beta:
                    beta = max(beta * config.lambd_beta_decay, config.min_beta)
                    logger.info(f"Patience threshold hit, reducing beta to: {beta:.6f}")
                    current_patience = 0 
                else:
                   logger.info(f"Early stopping criteria met. Stopping training.")
                   break 

        # --- Save Full Resumable Checkpoint (Manual) ---
        save_this_epoch = (epoch % config.checkpoint_save_interval == 0) or \
                          (epoch == config.num_epochs - 1)
        if save_this_epoch and run_id:
            checkpoint_state: Dict[str, Any] = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_val_loss': best_val_loss,
                'current_patience': current_patience,
                'beta': beta,
                'config_dump': config.model_dump() # For reference
            }
            
            ckpt_filename = f"model_checkpoint_epoch_{epoch:04d}.pt" # Padded epoch for sorting
            with tempfile.TemporaryDirectory() as tmpdir:
                local_tmp_checkpoint_path = Path(tmpdir) / ckpt_filename
                torch.save(checkpoint_state, local_tmp_checkpoint_path)
                
                mlflow.log_artifact(
                    str(local_tmp_checkpoint_path), 
                    artifact_path=f"{manual_checkpoint_subdir}" 
                )
            logger.info(f"Saved manual checkpoint to MLflow: {manual_checkpoint_subdir}")

            # --- Checkpoint Rotation ---
            rotate_checkpoints(
                run_id=run_id,
                artifact_subdir=manual_checkpoint_subdir,
                max_checkpoints=config.max_checkpoints_to_keep
            )

            if best_model_changed:
                bestmodel_filename = f"model_loss_{best_val_loss:.4f}.pt" # Padded epoch for sorting
                with tempfile.TemporaryDirectory() as tmpdir:
                    local_tmp_checkpoint_path = Path(tmpdir) / bestmodel_filename
                    torch.save(best_model_data_to_save, local_tmp_checkpoint_path)
                    
                    mlflow.log_artifact(
                        str(local_tmp_checkpoint_path), 
                        artifact_path=f"{config.manual_bestmodel_subdir}",
                        run_id=config.bestmodels_runid 
                    )
                            # --- Checkpoint Rotation ---
                rotate_bestmodels(
                    run_id=config.bestmodels_runid,
                    metric='loss',
                    artifact_subdir=config.manual_bestmodel_subdir,
                    max=config.max_checkpoints_to_keep
                )
                logger.info(f"Saved manual best model to MLflow: {config.manual_bestmodel_subdir}")
                best_model_changed = False
    
    training_duration_min = (time.time() - training_start_time) / 60
    logger.info(f"Total training time for this run: {training_duration_min:.2f} minutes.")
    mlflow.log_metric("total_training_time_min", training_duration_min)
    mlflow.log_metric("final_best_validation_loss", best_val_loss)

    model_to_extract_parts_from = model
    if best_model_data_to_save:
        model_to_extract_parts_from.load_state_dict(best_model_data_to_save['model_state_dict'])
        # Assuming your Model_VAE has 'encoder' and 'decoder' attributes
        # If not, you need to adapt how you get these state_dicts.
        # Saving latent embeddings
        with torch.no_grad():
            pre_encoder.load_weights(model_to_extract_parts_from)
            pre_decoder.load_weights(model_to_extract_parts_from)

            X_train_num = X_train_num.to(config.device)
            X_train_cat = X_train_cat.to(config.device)
            train_z = pre_encoder(X_train_num, X_train_cat).detach().cpu().numpy()

    mlflow.pytorch.autolog(disable=True)
    return model_to_extract_parts_from, pre_encoder, pre_decoder, train_z



    