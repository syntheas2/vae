import __init__
from typing import Tuple
import torch
import numpy as np
from zenml import step
from zenml.logger import get_logger
import mlflow

from pipelines.train_vae_args import VAEArgs # Assuming config.py is in the same directory or PYTHONPATH

logger = get_logger(__name__)

@step(enable_cache=True)
def prepare_pytorch_data(
    X_num_input: Tuple[np.ndarray, np.ndarray],
    X_cat_input: Tuple[np.ndarray, np.ndarray],
    config: VAEArgs,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Converts input numpy arrays to PyTorch tensors.
    X_num_input: (X_train_num_np, X_val_num_np)
    X_cat_input: (X_train_cat_np, X_val_cat_np)
    """
    logger.info("Preparing PyTorch data...")
    X_train_num_np, X_val_num_np = X_num_input
    X_train_cat_np, X_val_cat_np = X_cat_input

    X_train_num_pt = torch.tensor(X_train_num_np).float()
    X_val_num_pt = torch.tensor(X_val_num_np).float().to(config.device)
    
    X_train_cat_pt = torch.tensor(X_train_cat_np) # Usually long for categorical indices
    X_val_cat_pt = torch.tensor(X_val_cat_np).to(config.device)

    logger.info(f"X_train_num shape: {X_train_num_pt.shape}, X_train_cat shape: {X_train_cat_pt.shape}")
    logger.info(f"X_val_num shape: {X_val_num_pt.shape}, X_val_cat shape: {X_val_cat_pt.shape}")

    # Log dataset shapes to MLflow
    mlflow.log_param("X_train_num_rows", X_train_num_pt.shape[0])
    mlflow.log_param("X_train_num_cols", X_train_num_pt.shape[1])
    mlflow.log_param("X_train_cat_rows", X_train_cat_pt.shape[0])
    mlflow.log_param("X_train_cat_cols", X_train_cat_pt.shape[1] if X_train_cat_pt.ndim > 1 else 0)
    
    return X_train_num_pt, X_train_cat_pt, X_val_num_pt, X_val_cat_pt