from pydantic import BaseModel
import torch

class VAEArgs(BaseModel):
    # Data and Execution
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    batch_size: int = 4096
    num_workers: int = 4

    # Model Hyperparameters (from your global constants)
    num_layers: int = 2
    d_token: int = 4
    n_head: int = 1
    factor: int = 32
    token_bias: bool = True # TOKEN_BIAS was True

    # Training Hyperparameters
    learning_rate: float = 1e-3 # LR
    weight_decay: float = 0.0   # WD
    num_epochs: int = 600 # Example: make this configurable, was 1 in snippet
    max_beta: float = 0.01
    min_beta: float = 1e-5
    lambd_beta_decay: float = 0.99 # 'lambd' in your code for beta decay
    scheduler_patience: int = 10
    scheduler_factor: float = 0.95
    early_stopping_patience: int = 10 # 'patience' for beta adjustment and potential early stopping

    # MLflow
    mlflow_experiment_name: str = "VAE_Training_Experiment"
    load_ckp_from_run_id: str = "2d33e0d6241949cf987c54c3330f85d1"
    manual_bestmodel_subdir: str = "model_best"
    bestmodels_runid: str | None = "94356be5f3924963a84ec2333ff7aea8"
    load_from_checkpoint: bool = True
    max_checkpoints_to_keep: int = 5
    checkpoint_save_interval: int = 20