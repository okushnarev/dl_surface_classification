from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import LabelEncoder
from torch.nn import Module
from torch.utils.data import DataLoader, TensorDataset

from src.data.processing import create_sequences


@dataclass
class ModelWrapper:
    name: str
    net_type: str  # e.g., RNN, CNN
    filter_type: str  # e.g., kalman, no_filter
    dataset: str  # e.g., type_1, type_2
    model: Module = field(repr=False)
    features: list[str] = field(repr=False)


def top_sorted_dict(d: dict[str, float], top_n: int, nets: list[str]) -> dict[str, float]:
    if top_n < 0:
        return d

    cnt = dict()
    new_dict = dict()
    nets_names_ordered = sorted(nets, key=len, reverse=True)
    for name, acc in d.items():
        # find net type
        net_type = ''
        for n in nets_names_ordered:
            if n in name:
                net_type = n
                break

        # leave top elements only
        if net_type not in cnt:
            cnt[net_type] = 0
        if cnt[net_type] < top_n:
            new_dict[name] = acc
        cnt[net_type] += 1
    return new_dict


def run_inference(
        model_wrapper: ModelWrapper,
        df: pd.DataFrame,
        group_cols: list[str],
        feature_cols: list[str],
        target_col: str,
        info_cols: list[str],
        sequence_len: int,
        batch_size: int,
        scaler: Any,
        label_encoder: LabelEncoder,
        device: str,
) -> pd.DataFrame:
    """
    Runs inference pipeline: Scaling -> Sequence Generation -> Prediction -> Inverse Transform
    """
    # Prepare Data
    df_copy = df.copy()

    _, _, info_arr = create_sequences(
        df_copy,
        group_cols,
        feature_cols,
        target_col,
        sequence_len,
        info_cols
    )

    # Scale and encode features
    df_copy[feature_cols] = scaler.transform(df_copy[feature_cols])
    df_copy[target_col] = label_encoder.transform(df_copy[target_col])

    # Create Sequences
    X, y = create_sequences(
        df_copy,
        group_cols,
        feature_cols,
        target_col,
        sequence_len,
    )

    # Create loader
    X = torch.tensor(X, dtype=torch.bfloat16)
    y = torch.tensor(y, dtype=torch.long)
    dataset = TensorDataset(X, y)

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    # Inference
    model = model_wrapper.model
    model.eval()

    all_predictions = []
    with torch.no_grad():
        for seqs, _ in loader:
            seqs = seqs.to(device)

            outputs = model(seqs)
            predictions = torch.argmax(outputs, dim=1)
            all_predictions.append(predictions.cpu().numpy())

    all_predictions = np.concatenate(all_predictions)

    # Build results dataframe
    res_df = pd.DataFrame(info_arr, columns=info_cols)

    # Add predicitons
    res_df['prediction'] = label_encoder.inverse_transform(all_predictions)

    return res_df
