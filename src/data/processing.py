import numpy as np
import pandas as pd
from pandas import DataFrame
from sklearn.model_selection import train_test_split

from src.utils.vars import CHUNK_COL


def create_sequences(
        df: pd.DataFrame,
        group_cols: list[str],
        feature_cols: list[str],
        target_col: str,
        sequence_length: int,
        info_cols: list[str] = None
) -> tuple[np.ndarray, np.ndarray] | tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Groups processed and creates sliding window sequences

    :param df: Dataframe to process
    :param group_cols: Cols for df.groupby(). Order sensitive. Sequence generation resets between groups
    :param feature_cols: Names of features
    :param target_col: Target feature name (label). Use the last value in the window
    :param sequence_length: Width of a sliding window. Number of rows in one sample
    :param info_cols: (optional) Additional info, that can describe a whole window with single value (e.g. movement direction, surface type). Use the last value in the window

    :return: A tuple containing numpy arrays
        - X: Input sequences with shape (n_samples, sequence_length, n_features)
        - y: Target labels with shape (n_samples,)
        - info: (Optional) Metadata with shape (n_samples, n_info_cols) if info_cols is provided
    """
    sequences = []
    labels = []
    infos = []

    grouped = df.groupby(group_cols)

    for _, group in grouped:
        # Skip if too short
        if len(group) < sequence_length:
            continue

        feature_data = group[feature_cols].values
        label_data = group[target_col].values
        info_data = group[info_cols].values if info_cols else None

        # Manual sliding window
        for i in range(len(group) - sequence_length + 1):
            seq = feature_data[i: i + sequence_length]
            label = label_data[i + sequence_length - 1]

            sequences.append(seq)
            labels.append(label)

            if info_cols:
                info = info_data[i + sequence_length - 1]
                infos.append(info)

    if info_cols:
        return np.array(sequences), np.array(labels), np.array(infos)
    else:
        return np.array(sequences), np.array(labels)


def chunk_split(
        df: DataFrame,
        group_cols: list[str],
        target_col: str,
        **kwargs) -> tuple[DataFrame, DataFrame]:

    df = df.copy()

    # Prepare group cols
    if CHUNK_COL not in group_cols:
        group_cols = group_cols + [CHUNK_COL]
    if target_col not in group_cols:
        group_cols = [target_col] + group_cols

    # Split
    chunk_metadata = df[group_cols].drop_duplicates()
    train_ids, test_ids = train_test_split(
        chunk_metadata,
        stratify=chunk_metadata[target_col],
        **kwargs
    )

    # Compose dataframes
    train_df = df.merge(train_ids, on=group_cols, how='inner')
    test_df = df.merge(test_ids, on=group_cols, how='inner')

    return train_df, test_df
