from typing import Any, Callable, Tuple


def get_model_components(net_type: str) -> Tuple[Callable, Any]:
    """
    Returns the config preparation function and the Model class
    based on the network type string.
    """
    match net_type:
        case 'rnn':
            from src.models.nets.rnn import prep_cfg
            from src.models.nets.rnn import TabularRNN
            return prep_cfg, TabularRNN

        case 'cnn' | 'cnn_manual':
            from src.models.nets.cnn import prep_cfg
            from src.models.nets.cnn import CNN
            return prep_cfg, CNN

        case 'transformer':
            from src.models.nets.transformer import prep_cfg
            from src.models.nets.transformer import Transformer
            return prep_cfg, Transformer

        case 'transformer_cross_attn':
            from src.models.nets.transformer_cross_attn import prep_cfg
            from src.models.nets.transformer_cross_attn import TransformerCrossAttn
            return prep_cfg, TransformerCrossAttn

        case 'transformer_full_seq':
            from src.models.nets.transformer_full_seq import prep_cfg
            from src.models.nets.transformer_full_seq import TransformerFullSeq
            return prep_cfg, TransformerFullSeq

        case 'mamba':
            from src.models.nets.mamba import prep_cfg
            from src.models.nets.mamba import MambaClassifier
            return prep_cfg, MambaClassifier

        case _:
            raise ValueError(f'Unknown net {net_type}')
