import hashlib
import json
from datetime import datetime
from pathlib import Path

import pandas as pd
import torch
from torch.nn import Module


def get_model_hash(model: torch.nn.Module) -> str:
    """
    Generates a hash based on the model's state dictionary (architecture and weights)
    """
    hasher = hashlib.sha256()

    state_dict = model.state_dict()

    # Sort to maintain order
    for key in sorted(state_dict.keys()):
        tensor = state_dict[key]

        # Update hash with the key name
        hasher.update(key.encode('utf-8'))

        # Update hash with tensor values
        chunk = tensor.cpu().numpy().tobytes()
        hasher.update(chunk)

    return hasher.hexdigest()


def compose_metadata(model: Module, model_name: str, columns: list[str]) -> dict:
    return dict(
        model_name=model_name,
        hash=get_model_hash(model),
        columns=columns,
        timestamp=datetime.now().isoformat(),
    )


def check_cache(path: Path, model: Module, columns: list[str]) -> tuple[bool, pd.DataFrame | None]:
    is_cache_valid = False
    if path.exists():
        with open(path, 'r') as f:
            metadata = json.load(f)
        model_hash = get_model_hash(model)
        is_cache_valid = (model_hash == metadata.get('hash', None)
                      and sorted(metadata.get('columns', [])) == sorted(columns))

    return is_cache_valid
