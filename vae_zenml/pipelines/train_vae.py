import _init # noqa: F401
from zenml import pipeline
import mlflow  # Import the mlflow library
from datetime import datetime # For the run description
import torch
from steps.load_data import load_data_step
from steps.prepare_data import prepare_pytorch_data
from steps.train_vae import train_evaluate_vae
from pipelines.train_vae_args import VAEArgs


@pipeline
def train_vae_pipeline():
    args = VAEArgs()

    # --- Set the MLflow run name using a tag ---
    now = datetime.now() # Use current time for uniqueness
    timestamp_str = now.strftime("%Y-%m-%d_%H-%M-%S-%f")[:-3]
    mlflow.set_experiment(args.mlflow_experiment_name)
    mlflow.set_tag("mlflow.runName", f"{args.mlflow_run_name}_{timestamp_str}")
    if not args.bestmodels_runid:
        args.bestmodels_runid = mlflow.active_run().info.run_id
    
    if torch.cuda.is_available():
        args.device = f'cuda:{args.gpu}'
    else:
        args.device = 'cpu'

    X_num, X_cat, categories, d_numerical, num_inverse, cat_inverse, all_columns, column_metadata = load_data_step()

    # Step 1: Prepare data for PyTorch
    X_train_num_pt, X_train_cat_pt, X_val_num_pt, X_val_cat_pt = prepare_pytorch_data(
        X_num_input=X_num,
        X_cat_input=X_cat,
        config=args
    )

    # Step 2: Train and evaluate the VAE model
    model_to_extract_parts_from, pre_encoder, pre_decoder, train_z = train_evaluate_vae(
        X_train_num=X_train_num_pt,
        X_train_cat=X_train_cat_pt,
        X_val_num=X_val_num_pt,
        X_val_cat=X_val_cat_pt,
        d_numerical=d_numerical,
        categories=categories,
        config=args
    )

    return model_to_extract_parts_from, pre_encoder, pre_decoder, train_z


if __name__ == "__main__":
    train_vae_pipeline.with_options(
        enable_cache=False  
    )()