import hashlib
import torch


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
