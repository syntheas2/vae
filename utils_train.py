import numpy as np
import os

import src._init as src
from torch.utils.data import Dataset


class TabularDataset(Dataset):
    def __init__(self, X_num, X_cat):
        self.X_num = X_num
        self.X_cat = X_cat

    def __getitem__(self, index):
        this_num = self.X_num[index]
        this_cat = self.X_cat[index]

        sample = (this_num, this_cat)

        return sample

    def __len__(self):
        return self.X_num.shape[0]

def preprocess(dataset_path, task_type = 'binclass', inverse = False, cat_encoding = None, concat = True):
    
    T_dict = {}

    T_dict['normalization'] = "quantile"
    T_dict['num_nan_policy'] = 'mean'
    T_dict['cat_nan_policy'] =  None
    T_dict['cat_min_frequency'] = None
    T_dict['cat_encoding'] = cat_encoding
    T_dict['y_policy'] = "default"

    T = src.Transformations(**T_dict)

    dataset = make_dataset(
        data_path = dataset_path,
        T = T,
        task_type = task_type,
        change_val = False,
        concat = concat
    )

    if cat_encoding is None:
        X_num = dataset.X_num
        X_cat = dataset.X_cat

        X_train_num, X_test_num = X_num['train'], X_num['test']
        X_train_cat, X_test_cat = X_cat['train'], X_cat['test']
        
        categories = src.get_categories(X_train_cat)
        d_numerical = X_train_num.shape[1]

        X_num = (X_train_num, X_test_num)
        X_cat = (X_train_cat, X_test_cat)


        if inverse:
            num_inverse = dataset.num_transform.inverse_transform
            cat_inverse = dataset.cat_transform.inverse_transform

            return X_num, X_cat, categories, d_numerical, num_inverse, cat_inverse
        else:
            return X_num, X_cat, categories, d_numerical
    else:
        return dataset
    

def load_preprocessed_csv(trainpath, testpath, inverse=False):
    """
    Load preprocessed CSV files without using an info object.
    
    Args:
        trainpath (str): Path to the training CSV file
        testpath (str): Path to the test CSV file
        inverse (bool): Whether to return inverse transformation functions
        
    Returns:
        tuple: Data needed for training or sampling
    """
    import pandas as pd
    import numpy as np
    
    # Load train and test data
    train_df = pd.read_csv(trainpath)
    test_df = pd.read_csv(testpath)
    

    
    # Remove excluded columns
    excluded_cols = ['combined_tks', 'id']
    for col in excluded_cols:
        if col in train_df.columns:
            train_df = train_df.drop(columns=[col])
            test_df = test_df.drop(columns=[col])

    
    # Identify columns after exclusion
    all_columns = train_df.columns.tolist()
    
    # Group categorical columns by their prefix
    category_prefixes = ['category_', 'sub_category1_', 'sub_category2_', 'ticket_type_', 'business_service_']
    
    # Store column metadata for custom transformations
    column_metadata = {
        'categorical_groups': [],
        'numerical_columns': [],
        'target_column': 'impact' if 'impact' in all_columns else None
    }
    
    # Identify categorical columns and group them
    for prefix in category_prefixes:
        cols = [col for col in all_columns if col.startswith(prefix)]
        if cols:
            prefix_clean = prefix.rstrip('_')
            column_metadata['categorical_groups'].append((prefix_clean, sorted(cols)))
    
    # Identify numerical columns (all columns that aren't categorical)
    num_cols = [col for col in all_columns if not any(col.startswith(prefix) for prefix in category_prefixes)]
    column_metadata['numerical_columns'] = num_cols
    
    # Extract numerical features
    X_train_num = train_df[num_cols].values
    X_test_num = test_df[num_cols].values
    
    # Process categorical features - convert one-hot encoding to indices
    X_train_cat_indices = []
    X_test_cat_indices = []
    categories = []
    
    for group_name, group_cols in column_metadata['categorical_groups']:
        # Extract one-hot encoded columns for this group
        train_group = train_df[group_cols].values
        test_group = test_df[group_cols].values
        
        # Convert one-hot encoding to indices (argmax)
        train_indices = np.argmax(train_group, axis=1)
        test_indices = np.argmax(test_group, axis=1)
        
        # Handle all-zero rows
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
    
    # Create inverse transformation functions if required
    if inverse:
        # Create numerical inverse transformation (identity function)
        def num_inverse(numerical_data):
            """Identity transform for numerical data"""
            return numerical_data
        
        # Create categorical inverse transformation function that operates on raw indices
        def cat_inverse(categorical_indices):
            """Transform categorical indices back to one-hot encoding"""
            batch_size = categorical_indices.shape[0]
            result = {}
            
            # Process each categorical group
            for group_idx, (group_name, group_cols) in enumerate(column_metadata['categorical_groups']):
                # Get indices for this group
                indices = categorical_indices[:, group_idx].astype(int)
                
                # Create one-hot encoding for this group
                one_hot = np.zeros((batch_size, len(group_cols)))
                
                # Set 1s for valid indices (less than number of columns)
                for row_idx, col_idx in enumerate(indices):
                    if col_idx < len(group_cols):
                        one_hot[row_idx, col_idx] = 1.0
                
                # Store one-hot encoding with column names
                for col_idx, col_name in enumerate(group_cols):
                    result[col_name] = one_hot[:, col_idx]
            
            # Return the categorical data in DataFrame format
            return pd.DataFrame(result)
        
        return (X_train_num, X_test_num), (X_train_cat, X_test_cat), categories, d_numerical, num_inverse, cat_inverse, all_columns, column_metadata
    else:
        return (X_train_num, X_test_num), (X_train_cat, X_test_cat), categories, d_numerical


def update_ema(target_params, source_params, rate=0.999):
    """
    Update target parameters to be closer to those of source parameters using
    an exponential moving average.
    :param target_params: the target parameter sequence.
    :param source_params: the source parameter sequence.
    :param rate: the EMA rate (closer to 1 means slower).
    """
    for target, source in zip(target_params, source_params):
        target.detach().mul_(rate).add_(source.detach(), alpha=1 - rate)



def concat_y_to_X(X, y):
    if X is None:
        return y.reshape(-1, 1)
    return np.concatenate([y.reshape(-1, 1), X], axis=1)


def make_dataset(
    data_path: str,
    T: src.Transformations,
    task_type,
    change_val: bool,
    concat = True,
):

    # classification
    if task_type == 'binclass' or task_type == 'multiclass':
        X_cat = {} if os.path.exists(os.path.join(data_path, 'X_cat_train.npy'))  else None
        X_num = {} if os.path.exists(os.path.join(data_path, 'X_num_train.npy')) else None
        y = {} if os.path.exists(os.path.join(data_path, 'y_train.npy')) else None

        for split in ['train', 'test']:
            X_num_t, X_cat_t, y_t = src.read_pure_data(data_path, split)
            if X_num is not None:
                X_num[split] = X_num_t
            if X_cat is not None:
                if concat:
                    X_cat_t = concat_y_to_X(X_cat_t, y_t)
                X_cat[split] = X_cat_t  
            if y is not None:
                y[split] = y_t
    else:
        # regression
        X_cat = {} if os.path.exists(os.path.join(data_path, 'X_cat_train.npy')) else None
        X_num = {} if os.path.exists(os.path.join(data_path, 'X_num_train.npy')) else None
        y = {} if os.path.exists(os.path.join(data_path, 'y_train.npy')) else None

        for split in ['train', 'test']:
            X_num_t, X_cat_t, y_t = src.read_pure_data(data_path, split)

            if X_num is not None:
                if concat:
                    X_num_t = concat_y_to_X(X_num_t, y_t)
                X_num[split] = X_num_t
            if X_cat is not None:
                X_cat[split] = X_cat_t
            if y is not None:
                y[split] = y_t

    info = src.load_json(os.path.join(data_path, 'info.json'))

    D = src.Dataset(
        X_num,
        X_cat,
        y,
        y_info={},
        task_type=src.TaskType(info['task_type']),
        n_classes=info.get('n_classes')
    )

    if change_val:
        D = src.change_val(D)

    # def categorical_to_idx(feature):
    #     unique_categories = np.unique(feature)
    #     idx_mapping = {category: index for index, category in enumerate(unique_categories)}
    #     idx_feature = np.array([idx_mapping[category] for category in feature])
    #     return idx_feature

    # for split in ['train', 'val', 'test']:
    # D.y[split] = categorical_to_idx(D.y[split].squeeze(1))

    return src.transform_dataset(D, T, None)