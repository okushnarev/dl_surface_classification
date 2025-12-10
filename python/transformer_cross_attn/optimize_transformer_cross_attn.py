from pathlib import Path

from torch.utils.data import DataLoader

from models.cnn import MLPLayerConfig
from models.configs.transformer_cross_attn_config import TransformerCrossAttnConfig
from models.transformer_cross_attn import TransformerCrossAttn
from python.optimize_seq import optimize
from python.utils.optimization_utils import run_training_loop


def transformer_cross_attn_objective(trial, val_dataset, input_dim, num_steps, num_classes, batch_size, device, epochs,
                                     seed):
    """Defines a single trial using a fixed train/validation split."""

    # Suggest Hyperparameters
    lr = trial.suggest_float('lr', low=1e-4, high=1e-2, log=True)
    embedding_dim = 2 ** trial.suggest_int('embedding_dim_pow', low=4, high=8)

    num_transformer_heads = 2 ** trial.suggest_int('num_transformer_heads_pow', low=0, high=2)
    num_transformer_layers = trial.suggest_int('num_transformer_layers', low=1, high=4)

    num_cross_attn_heads = 2 ** trial.suggest_int('num_cross_attn_heads_pow', low=0, high=2)
    num_cross_attn_layers = trial.suggest_int('num_cross_attn_layers', low=1, high=4)

    encoder_n_layers = trial.suggest_int('encoder_n_layers', 1, 4)
    encoder_dims = [2 ** trial.suggest_int(f'encoder_dim_{i}_pow', low=4, high=8) for i in range(encoder_n_layers)]

    classification_n_layers = trial.suggest_int('classification_n_layers', 1, 4)
    classification_dims = [2 ** trial.suggest_int(f'classification_dim_{i}_pow', low=4, high=8) for i in
                           range(classification_n_layers)]

    cross_attn_n_layers = trial.suggest_int('cross_attn_n_layers', 1, 4)
    cross_attn_dims = [2 ** trial.suggest_int(f'cross_attn_dim_{i}_pow', low=4, high=8) for i in
                       range(cross_attn_n_layers)]

    encoder_layers = [
        MLPLayerConfig(out_dim=d, dropout=0.2)
        for d in encoder_dims
    ]
    classification_layers = [
        MLPLayerConfig(out_dim=d, dropout=0.2)
        for d in classification_dims
    ]

    cross_attn_layers = [
        MLPLayerConfig(out_dim=d, dropout=0.2)
        for d in cross_attn_dims
    ]

    model_cfg = TransformerCrossAttnConfig(
        input_dim=input_dim,
        sequence_length=num_steps,
        embedding_dim=embedding_dim,

        encoder_layers=encoder_layers,

        num_transformer_heads=num_transformer_heads,
        num_transformer_layers=num_transformer_layers,

        num_cross_attn_heads=num_cross_attn_heads,
        num_cross_attn_layers=num_cross_attn_layers,
        cross_attn_ffn_config=cross_attn_layers,

        classification_layers=classification_layers,
        num_classes=num_classes,
    )

    # Instantiate Model and Optimizer
    model = TransformerCrossAttn(model_cfg).to(device)

    # Create DataLoaders for this trial
    val_loader = DataLoader(val_dataset, batch_size=batch_size, pin_memory=True)

    accuracy = run_training_loop(
        trial,
        model,
        val_loader,
        epochs,
        lr,
        device
    )

    return accuracy


def main():
    params_path = Path(f'data/params/transformer_cross_attn_optim')
    params_path.mkdir(parents=True, exist_ok=True)
    optimize(transformer_cross_attn_objective, params_path)


if __name__ == '__main__':
    main()
