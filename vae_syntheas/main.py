import torch
import torch.nn as nn

import argparse
import warnings
import numpy as np



warnings.filterwarnings('ignore')

def transform_preprocessed_data(train_df, test_df):
    """
    Load preprocessed CSV files and format them to match the output of the preprocess function.
    
    Args:
        task_type (str): Type of task ('binclass', 'multiclass', or 'regression')
        
    Returns:
        tuple: (X_num, X_cat, categories, d_numerical) where:
            - X_num is a tuple of (train_numerical_features, test_numerical_features)
            - X_cat is a tuple of (train_categorical_features, test_categorical_features)
            - categories is a list of sizes for each categorical variable
            - d_numerical is the number of numerical features
    """
    # Remove excluded columns only (keep the target 'impact')
    if 'combined_tks' in train_df.columns:
        train_df = train_df.drop(columns=['combined_tks', 'id'])
        test_df = test_df.drop(columns=['combined_tks', 'id'])
    
     # Identify numerical and categorical columns
    all_columns = train_df.columns.tolist()
    
    # Group categorical columns by their prefix to identify the categorical features
    category_prefixes = ['category_', 'sub_category1_', 'sub_category2_', 'ticket_type_', 'business_service_']
    
    # Dictionary to store grouped categorical columns
    categorical_groups = {}
    
    # Group the categorical columns by their prefix
    for prefix in category_prefixes:
        cols = [col for col in all_columns if col.startswith(prefix)]
        if cols:
            prefix_clean = prefix.rstrip('_')  # Remove trailing underscore
            categorical_groups[prefix_clean] = sorted(cols)
    
    # All other columns except combined_tks are considered numerical
    num_cols = [col for col in all_columns if not any(col.startswith(prefix) for prefix in category_prefixes) 
                and col != 'combined_tks']
    
    # Extract numerical features
    X_train_num = train_df[num_cols].values
    X_test_num = test_df[num_cols].values
    
    # Process categorical features - convert one-hot encoding to indices
    X_train_cat_indices = []
    X_test_cat_indices = []
    categories = []
    
    for group_name, group_cols in categorical_groups.items():
        # Extract one-hot encoded columns for this group
        train_group = train_df[group_cols].values
        test_group = test_df[group_cols].values
        
        # Convert one-hot encoding to indices (argmax)
        # If a row is all zeros, we'll use the last category as default
        train_indices = np.argmax(train_group, axis=1)
        test_indices = np.argmax(test_group, axis=1)
        
        # Handle case where all values in a row are 0 (no category selected)
        # Set to a special index (num_categories)
        all_zeros_train = (train_group.sum(axis=1) == 0)
        all_zeros_test = (test_group.sum(axis=1) == 0)
        
        if np.any(all_zeros_train) or np.any(all_zeros_test):
            # Add an extra category for "none selected"
            num_categories = len(group_cols) + 1
            train_indices[all_zeros_train] = len(group_cols)
            test_indices[all_zeros_test] = len(group_cols)
        else:
            num_categories = len(group_cols)
        
        X_train_cat_indices.append(train_indices)
        X_test_cat_indices.append(test_indices)
        categories.append(num_categories)
    
    # Stack all categorical features into a single matrix
    X_train_cat = np.column_stack(X_train_cat_indices) if X_train_cat_indices else np.empty((len(X_train_num), 0), dtype=np.int64)
    X_test_cat = np.column_stack(X_test_cat_indices) if X_test_cat_indices else np.empty((len(X_test_num), 0), dtype=np.int64)
    
    # Number of numerical features
    d_numerical = X_train_num.shape[1]
    
    # Return in the same format as preprocess
    return (X_train_num, X_test_num), (X_train_cat, X_test_cat), categories, d_numerical




def compute_loss(X_num, X_cat, Recon_X_num, Recon_X_cat, mu_z, logvar_z):
    ce_loss_fn = nn.CrossEntropyLoss()
    mse_loss = (X_num - Recon_X_num).pow(2).mean()
    ce_loss = 0
    acc = 0
    total_num = 0

    for idx, x_cat in enumerate(Recon_X_cat):
        if x_cat is not None:
            ce_loss += ce_loss_fn(x_cat, X_cat[:, idx])
            x_hat = x_cat.argmax(dim = -1)
        acc += (x_hat == X_cat[:,idx]).float().sum()
        total_num += x_hat.shape[0]
    
    ce_loss /= (idx + 1)
    acc /= total_num
    # loss = mse_loss + ce_loss

    temp = 1 + logvar_z - mu_z.pow(2) - logvar_z.exp()

    loss_kld = -0.5 * torch.mean(temp.mean(-1).mean())
    return mse_loss, ce_loss, loss_kld, acc

    parser = argparse.ArgumentParser(description='Variational Autoencoder')

    parser.add_argument('--dataname', type=str, default='adult', help='Name of dataset.')
    parser.add_argument('--gpu', type=int, default=0, help='GPU index.')
    parser.add_argument('--max_beta', type=float, default=1e-2, help='Initial Beta.')
    parser.add_argument('--min_beta', type=float, default=1e-5, help='Minimum Beta.')
    parser.add_argument('--lambd', type=float, default=0.7, help='Decay of Beta.')

    args = parser.parse_args()

    # check cuda
    if args.gpu != -1 and torch.cuda.is_available():
        args.device = 'cuda:{}'.format(args.gpu)
    else:
        args.device = 'cpu'