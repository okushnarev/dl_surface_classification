from typing import Any, Dict


def get_model_components(net_type: str) -> Dict[str, Any]:
    """
    Returns a dictionary containing the model components based on the network type string

    :return:
    Dictionary:
    - 'class': The nn.Module class
    - 'prep_config': Callable(path, input_dim, ...) -> dict (For Training)
    - 'optuna_params': Callable(trial) -> dict (For Optimization)
    """
    match net_type:
        case 'rnn':
            from src.models.nets.rnn import prep_cfg, get_optuna_params
            from src.models.nets.rnn import TabularRNN
            return {
                'class':         TabularRNN,
                'prep_config':   prep_cfg,
                'optuna_params': get_optuna_params
            }

        case 'cnn' | 'cnn_manual':
            from src.models.nets.cnn import prep_cfg, get_optuna_params
            from src.models.nets.cnn import CNN
            return {
                'class':         CNN,
                'prep_config':   prep_cfg,
                'optuna_params': get_optuna_params
            }

        case 'transformer':
            from src.models.nets.transformer import prep_cfg, get_optuna_params
            from src.models.nets.transformer import Transformer
            return {
                'class':         Transformer,
                'prep_config':   prep_cfg,
                'optuna_params': get_optuna_params
            }

        case 'transformer_cross_attn':
            from src.models.nets.transformer_cross_attn import prep_cfg, get_optuna_params
            from src.models.nets.transformer_cross_attn import TransformerCrossAttn
            return {
                'class':         TransformerCrossAttn,
                'prep_config':   prep_cfg,
                'optuna_params': get_optuna_params
            }

        case 'transformer_full_seq':
            from src.models.nets.transformer_full_seq import prep_cfg, get_optuna_params
            from src.models.nets.transformer_full_seq import TransformerFullSeq
            return {
                'class':         TransformerFullSeq,
                'prep_config':   prep_cfg,
                'optuna_params': get_optuna_params
            }

        case 'mamba':
            from src.models.nets.mamba import prep_cfg, get_optuna_params
            from src.models.nets.mamba import MambaClassifier
            return {
                'class':         MambaClassifier,
                'prep_config':   prep_cfg,
                'optuna_params': get_optuna_params
            }

        case _:
            raise ValueError(f'Unknown net type: {net_type}')