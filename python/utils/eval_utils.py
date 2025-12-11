import pickle
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import DataLoader, TensorDataset

from python.utils.dataset import create_sequences
from python.utils.model_hash import get_model_hash
from python.utils.save_load import read_csv_and_metadata


@dataclass
class Model:
    net_type: str  # e.g., RNN, CNN
    filter_type: str  # e.g., kalman, no_filter
    dataset: str  # e.g., type_1, type_2
    model: torch.nn.Module = field(repr=False)
    features: list[str] = field(repr=False)

    @property
    def name(self):
        net_type = self.net_type.upper()
        filter_type = self.filter_type.replace('_', ' ')
        dataset = self.dataset.replace('_', ' ')
        return f'{net_type} {filter_type} {dataset}'


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
        model_name: str,
        model_wrapper: Model,
        df: pd.DataFrame,
        group_cols: str | list[str],
        target_col: str,
        sequence_len: int,
        info_cols: list[str],
        label_encoder: LabelEncoder,
        device: str,
        batch_size: int,
        exps_to_unscale: list[str],
        unscale_cols: list[str]
) -> pd.DataFrame:
    """
    Runs inference for a specific model on a dataframe.
    Handles scaling, sequence creation, batched inference, and inverse transformation.
    """
    print(f'Running {model_name}')

    # Scaler
    ckpt_path = Path('data/runs') / model_name
    scaler_path = ckpt_path / 'scaler.pkl'
    with open(scaler_path, 'rb') as f:
        scaler = pickle.load(f)

    # Features
    feature_cols = model_wrapper.features

    # DataFrame prep
    df_copy = df.copy()
    df_copy[feature_cols] = scaler.transform(df_copy[feature_cols])

    # Create sequences
    X, y, info_arr = create_sequences(df_copy, group_cols, feature_cols, target_col, sequence_len, info_cols)

    info = pd.DataFrame(data=info_arr, columns=info_cols)
    info['surf'] = info['surf'].astype(np.uint8)
    info['surf'] = label_encoder.inverse_transform(info['surf'])

    # Create loader
    X = torch.tensor(X, dtype=torch.float32)
    y = torch.tensor(y, dtype=torch.long)
    eval_dataset = TensorDataset(X, y)

    eval_loader = DataLoader(eval_dataset, batch_size=batch_size, shuffle=False)

    # Run tests
    model = model_wrapper.model
    predictions = np.array([], dtype=np.uint8)
    model.eval()
    with torch.no_grad():
        for sequences, labels in eval_loader:
            sequences = sequences.to(device, non_blocking=True)
            outputs = model(sequences)
            predicted = torch.argmax(outputs.data, 1)

            predictions = np.hstack([predictions, predicted.cpu().numpy()])

    info['predictions'] = label_encoder.inverse_transform(predictions)

    # Unscale certain columns if needed
    if model_name in exps_to_unscale:
        # Create placeholder matrix with the shape that Scaler expects
        dummy_data = np.zeros((len(info), len(feature_cols)))
        # Copy data_to_unscale to its corresponding position in the placeholder matrix
        for _col in unscale_cols:
            dummy_data[:, feature_cols.index(_col)] = info[_col]
        dummy_data = scaler.inverse_transform(dummy_data)
        # Take out only chosen unscaled data from the placeholder matrix
        for _col in unscale_cols:
            info[_col] = dummy_data[:, feature_cols.index(_col)]

    return info


def compose_metadata(model: torch.nn.Module, model_name: str, columns: list[str]) -> dict:
    return dict(
        model_name=model_name,
        hash=get_model_hash(model),
        columns=columns,
        timestamp=datetime.now().isoformat(),
    )


def check_for_cache(path: Path, model: torch.nn.Module, columns: list[str]) -> tuple[bool, pd.DataFrame | None]:
    use_chache = False
    info = None
    if path.exists():
        info, metadata = read_csv_and_metadata(path)
        if metadata is not None:
            model_hash = get_model_hash(model)
            use_chache = (model_hash == metadata['hash']
                          and sorted(metadata['columns']) == sorted(columns))
        else:
            print(f'Lacking metadata at path: {path}')

    return use_chache, info
