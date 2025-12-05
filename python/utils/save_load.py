import os
import pandas as pd
from pathlib import Path
import torch
import json


def save_checkpoint(model, optimizer, epoch, loss, filepath):
    """Saves a checkpoint for resuming training."""

    print(f'Saving checkpoint to {filepath}')

    checkpoint = {
        'epoch':                epoch + 1,  # Save the next epoch to start from
        'model_state_dict':     model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss':                 loss,
    }

    torch.save(checkpoint, filepath)


def load_checkpoint(model, optimizer, filepath, device='cpu'):
    """Loads a checkpoint to resume training or for inference."""

    # Check if the checkpoint file exists
    if not os.path.isfile(filepath):
        print(f'No checkpoint found at {filepath}')
        return 0  # Return epoch 0 to start training from scratch

    print(f'Loading checkpoint from {filepath}')

    checkpoint = torch.load(filepath, map_location=torch.device(device))

    # Load the states into model and optimizer
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    return checkpoint


def save_csv_and_metadata(df: pd.DataFrame, metadata: dict, filepath: Path, **kwargs):
    """
    Saves a dataframe and metadata to csv and json
    :param df: Dataframe to save in csv format
    :param metadata: Dict metadata to save in json format
    :param filepath: Path to save csv and json. Pass .csv full path. JSON path will be handlen automatically
    :param kwargs: Additional arguments to pass to DataFrame.to_csv()
    """
    df.to_csv(filepath, **kwargs)

    with open(filepath.with_suffix('.json'), 'w') as file:
        json.dump(metadata, file, indent=2)

def read_csv_and_metadata(filepath: Path, **kwargs) -> tuple[pd.DataFrame, dict | None]:
    """
    Reads a csv file with the corresponding json metadata
    :param filepath: Path to csv file
    :param kwargs: Additional arguments to pass to pandas.read_csv()
    :return: Dataframe and dict metadata. Metadata would be None if json file is not found
    """
    df = pd.read_csv(filepath, **kwargs)

    metadata = None
    if (json_path := filepath.with_suffix('.json')).exists():
        with open(json_path) as file:
            metadata = json.load(file)


    return df, metadata
