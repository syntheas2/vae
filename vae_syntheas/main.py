import torch
import torch.nn as nn

import argparse
import warnings
import numpy as np
import pandas as pd


warnings.filterwarnings('ignore')

def transform_preprocessed_data(train_df, test_df, metadata, excluded_cols, inverse=False):
    """
    Transform dataframes with non-one-hot categorical columns into a format suitable for modeling.
    Uses metadata to infer categorical and numerical columns.

    Args:
        train_df (pd.DataFrame): Training dataframe.
        test_df (pd.DataFrame): Test dataframe.
        metadata (dict): Metadata describing columns (type, unique values, etc).
        inverse (bool): Whether to return inverse transformation functions.

    Returns:
        (X_num, X_cat, categories, d_numerical) or (+inverse funcs, all_columns, column_metadata)
    """
    # Remove excluded columns
    train_df = train_df.drop(columns=[col for col in excluded_cols if col in train_df.columns], errors='ignore')
    test_df  = test_df.drop(columns=[col for col in excluded_cols if col in test_df.columns], errors='ignore')
    
    # Use metadata to extract categorical and numerical columns
    categorical_columns = []
    numerical_columns = []

    for col_meta in metadata['columns']:
        col_name = col_meta['name']
        if col_name in excluded_cols:
            continue
        if col_meta.get('type') == 'categorical':
            categorical_columns.append(col_name)
        elif col_meta.get('type') in ('float', 'int', 'numeric', 'number'):
            numerical_columns.append(col_name)
        else:
            # fallback: treat as numerical if dtype is numeric, otherwise skip
            if pd.api.types.is_numeric_dtype(train_df[col_name]):
                numerical_columns.append(col_name)

    # Extract numerical features
    X_train_num = train_df[numerical_columns].values.astype(np.float32) if numerical_columns else np.empty((train_df.shape[0], 0))
    X_test_num  = test_df[numerical_columns].values.astype(np.float32) if numerical_columns else np.empty((test_df.shape[0], 0))
    
    # Extract categorical features as indices
    X_train_cat_indices = []
    X_test_cat_indices = []
    categories = []
    cat_value_maps = {}  # map col_name -> [class0, class1, ...] for inverse

    for col_meta in metadata['columns']:
        col_name = col_meta['name']
        if col_name in categorical_columns:
            unique_values = col_meta.get('unique_values')
            if unique_values is None:
                # fallback: compute from train/test
                unique_values = sorted(pd.concat([train_df[col_name], test_df[col_name]]).dropna().unique())
            # Map to indices
            value_to_idx = {val: idx for idx, val in enumerate(unique_values)}
            idx_to_value = {idx: val for idx, val in enumerate(unique_values)}
            cat_value_maps[col_name] = idx_to_value
            # Map train and test
            train_indices = train_df[col_name].map(value_to_idx).fillna(-1).astype(int).values
            test_indices  = test_df[col_name].map(value_to_idx).fillna(-1).astype(int).values

            n_cat = len(unique_values)
            categories.append(n_cat)
            X_train_cat_indices.append(train_indices)
            X_test_cat_indices.append(test_indices)

    if X_train_cat_indices:
        X_train_cat = np.column_stack(X_train_cat_indices)
        X_test_cat  = np.column_stack(X_test_cat_indices)
    else:
        X_train_cat = np.empty((len(X_train_num), 0), dtype=np.int64)
        X_test_cat  = np.empty((len(X_test_num), 0), dtype=np.int64)

    d_numerical = X_train_num.shape[1]

    # Build metadata for inverse
    column_metadata = {
        'categorical_columns': categorical_columns,
        'numerical_columns': numerical_columns,
        'cat_value_maps': cat_value_maps,
    }
    all_columns = train_df.columns.tolist()

    if inverse:
        # Identity for numericals
        def num_inverse(numerical_data):
            """Identity transform for numerical data"""
            return numerical_data

        # Inverse for categoricals: indices -> dataframe with original values
        def cat_inverse(categorical_indices):
            """
            categorical_indices: np.ndarray of shape [batch, n_cat_cols]
            Returns pd.DataFrame with original column names and values
            """
            result = {}
            for i, col_name in enumerate(categorical_columns):
                idx_to_value = cat_value_maps[col_name]
                arr = categorical_indices[:, i]
                # If index is last (unseen/missing), set to np.nan or a special value
                result[col_name] = [idx_to_value.get(idx, np.nan) for idx in arr]
            return pd.DataFrame(result)

        return (X_train_num, X_test_num), (X_train_cat, X_test_cat), categories, d_numerical, num_inverse, cat_inverse, all_columns, column_metadata
    else:
        return (X_train_num, X_test_num), (X_train_cat, X_test_cat), categories, d_numerical




def compute_loss(X_num, X_cat, Recon_X_num, Recon_X_cat, mu_z, logvar_z):
    ce_loss_fn = nn.CrossEntropyLoss()
    if len(X_num[0]) > 0:
        num_mse_loss = (X_num - Recon_X_num).pow(2).mean()
    else:
        num_mse_loss = torch.tensor(0)

    ce_loss = 0
    acc = 0
    total_num = 0

    for idx, x_cat in enumerate(Recon_X_cat):
        if x_cat is not None:
            ce_loss += ce_loss_fn(x_cat, X_cat[:, idx].long())
            x_hat = x_cat.argmax(dim = -1)
        acc += (x_hat == X_cat[:,idx]).float().sum()
        total_num += x_hat.shape[0]
    
    ce_loss /= (idx + 1)
    acc /= total_num
    # loss = mse_loss + ce_loss

    temp = 1 + logvar_z - mu_z.pow(2) - logvar_z.exp()

    loss_kld = -0.5 * torch.mean(temp.mean(-1).mean())
    return num_mse_loss, ce_loss, loss_kld, acc

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