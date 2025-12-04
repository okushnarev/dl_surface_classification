import os

import torch


def save_checkpoint(model, optimizer, epoch, loss, filepath):
    '''Saves a checkpoint for resuming training.'''

    print(f'=> Saving checkpoint to {filepath}')

    # Create a dictionary containing all the necessary states
    checkpoint = {
        'epoch':                epoch + 1,  # Save the next epoch to start from
        'model_state_dict':     model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss':                 loss,
    }

    # Save the dictionary to the specified file
    torch.save(checkpoint, filepath)


def load_checkpoint(model, optimizer, filepath, device='cpu'):
    '''Loads a checkpoint to resume training or for inference.'''

    # Check if the checkpoint file exists
    if not os.path.isfile(filepath):
        print(f'=> No checkpoint found at {filepath}')
        return 0  # Return epoch 0 to start training from scratch

    print(f'=> Loading checkpoint from {filepath}')

    # Load the checkpoint dictionary. It's good practice to load to the CPU first
    # to avoid GPU memory issues if the model was saved on a different device.
    checkpoint = torch.load(filepath, map_location=torch.device(device))

    # Load the states into your model and optimizer
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    return checkpoint
