import _init # noqa: F401
from scipy.sparse import csr_matrix
from typing import Tuple, Annotated
import pandas as pd
from zenml import step
from typing import Annotated, List, Any
from zenml.client import Client
import torch
from utils import get_args
# at first try train tabsyn
from pathlib import Path
import numpy as np
from vae_syntheas.main import transform_preprocessed_data
from syntheas_zenml_mgmt.data import get_df_test, get_df_train, get_df_val



@step
def load_data_step() -> Tuple[
    Annotated[Tuple[np.ndarray, np.ndarray], "X_num"],
    Annotated[Tuple[np.ndarray, np.ndarray], "X_cat"],
    Annotated[List[int], "categories"],
    Annotated[int, "d_numerical"],
    Annotated[Any, "num_inverse"],
    Annotated[Any, "cat_inverse"],
    Annotated[Any, "all_columns"],
    Annotated[Any, "column_metadata"],
]:
    df_train = get_df_train()
    df_train.impact = df_train.impact.astype(int)
    df_val = get_df_val()
    df_val.impact = df_val.impact.astype(int)

    X_num, X_cat, categories, d_numerical, num_inverse, cat_inverse, all_columns, column_metadata = transform_preprocessed_data(df_train, df_val, inverse=True)

    return X_num, X_cat, categories, d_numerical, num_inverse, cat_inverse, all_columns, column_metadata