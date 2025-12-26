import argparse
import json
import sys
from pathlib import Path

import joblib
import pandas as pd
import torch
import yaml
from sklearn.preprocessing import LabelEncoder

from src.utils.hashing import check_cache, compose_metadata

# Add project root to PATH
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

from src.utils.paths import ProjectPaths
from src.utils.io import save_csv_and_metadata
from src.models.factory import get_model_components
from src.engine.evaluator import ModelWrapper, run_inference


def parse_args():
    parser = argparse.ArgumentParser(description='Evaluation Runner')

    parser.add_argument('--dataset', type=str, default='main', help='Dataset scope (e.g., main, boreal)')
    parser.add_argument('--ds_path', type=str, default=None, help='Path to evaluation dataset')
    parser.add_argument('--subset', type=str, default='full', choices=['test', 'val', 'train', 'full'],
                        help='Data subset to evaluate on')

    parser.add_argument('--nets', nargs='+', default=['rnn'], help='List of networks to evaluate (e.g. rnn cnn)')
    parser.add_argument('--cache_only_nets', nargs='+', default=[],
                        help='List of networks to use cache straightforward without validation '
                             '(helpful for nets that require different accelerator than yours, e.g. mamba runs on GPU only)')
    parser.add_argument('--config_name', type=str, default='main', help='Name of the experiment YAML file')

    parser.add_argument('--ckpt_type', default='best', choices=['best', 'last'], help='Checkpoint to load')
    parser.add_argument('--batch_size', type=int, default=4096)

    return parser.parse_args()


def main():
    args = parse_args()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'--- Starting Evaluation on \'{args.dataset}/{args.subset}\' ---')

    # Load data config
    ds_config_path = ProjectPaths.get_dataset_config_path(args.dataset)
    with open(ds_config_path, 'r') as f:
        ds_config = json.load(f)

    group_cols = ds_config['metadata']['group_cols']
    info_cols = group_cols
    target_col = ds_config['metadata']['target_col']
    features_map = ds_config['features']

    # Find dataset
    if args.ds_path is not None:
        csv_path = Path(args.ds_path)
        args.subset = csv_path.stem
    elif args.subset in ['test', 'val', 'train']:
        data_dir = ProjectPaths.get_processed_data_dir(args.dataset)
        csv_path = data_dir / f'{args.subset}.csv'
    elif args.subset == 'full':
        data_dir = ProjectPaths.get_raw_data_dir()
        csv_path = data_dir / f'{args.dataset}.csv'
    else:
        print(f'Error: no instruction passed to find evaluation dataset')
        sys.exit(1)

    if not csv_path.exists():
        print(f'Error: Data file not found at {csv_path}')
        sys.exit(1)

    print(f'Loading data from {csv_path}...')
    df = pd.read_csv(csv_path)

    # Define output directory
    results_dir = ProjectPaths.get_evaluation_dir(args.dataset, args.subset)
    results_dir.mkdir(parents=True, exist_ok=True)

    # Ensure nets are in the same order
    nets = sorted(args.nets, key=len, reverse=True)
    # Loop over nets
    for net_name in nets:
        # Load experiment config
        exp_config_path = ProjectPaths.get_experiment_config_path(net_name, args.config_name)
        if not exp_config_path.exists():
            print(f'Skipping {net_name}: Config not found at {exp_config_path}')
            continue

        with open(exp_config_path, 'r') as f:
            yaml_content = yaml.safe_load(f)

        defaults = yaml_content.get('defaults', {})
        experiments = yaml_content.get('experiments', [])

        print(f'\nEvaluating {len(experiments)} experiments for {net_name}:')

        # Loop over experiments
        for exp in experiments:
            exp_name = exp.get('name')
            print(f'\nProcessing {exp_name}')

            # Prepare experiment args
            exp_args = defaults.get('common', {}).copy()
            exp_args |= exp.get('common', {})

            filter_type = exp_args.get('filter', 'no_filter')
            ds_type = exp_args.get('ds_type', 'type_1')

            # Identify features
            try:
                feature_cols = features_map[filter_type][ds_type]
            except KeyError:
                print(f'Error: Could not find features for {filter_type}/{ds_type} in dataset config.')
                continue

            # Define output path
            result_file = results_dir / f'{exp_name}.csv'

            # Skip for cache-only nets with existent results
            if net_name in args.cache_only_nets:
                if result_file.exists():
                    print(f'\tSkipping {exp_name} (Cache Only Model)')
                else:
                    print(f'\tSkipping {exp_name} (Cache Only Model, No file found: {result_file})')
                continue

            # Checkpoint, scaler, label encoder
            exp_dataset_scope = exp.get('dataset', defaults.get('dataset', args.dataset))
            run_dir = ProjectPaths.get_run_dir(exp_dataset_scope, exp_name)

            ckpt_path = run_dir / f'{args.ckpt_type}.pt'
            scaler_path = run_dir / 'scaler.joblib'
            label_encoder_path = run_dir / 'label_encoder.joblib'

            if not ckpt_path.exists():
                print(f'\tCheckpoint not found: {ckpt_path}. Skipping')
                continue

            # Load Scaler
            if not scaler_path.exists():
                print(f'\tScaler not found: {scaler_path}. Skipping')
                continue
            else:
                scaler = joblib.load(scaler_path)

            # Load label encoder
            if not label_encoder_path.exists():
                print(f'\tLabelEncoder not found: {label_encoder_path}. Creating from existing dataset')
                label_encoder = LabelEncoder()
                label_encoder.fit(df[target_col])
            else:
                label_encoder = joblib.load(label_encoder_path)

            num_classes = len(label_encoder.classes_)

            # Resolve Params File
            train_args = defaults.get('train', {}).copy()
            train_args |= exp.get('train', {})

            param_file = train_args.get('param_file')
            if param_file:
                cfg_path = Path(param_file)
            else:
                cfg_path = ProjectPaths.get_params_path(net_name, exp_dataset_scope, exp_name)

            # Prepare config
            seq_len = exp_args.get('seq_len', 10)

            # Create model
            components = get_model_components(net_name)
            ModelClass = components['class']
            prep_cfg = components['prep_config']

            model_cfg = prep_cfg(
                cfg_path,
                input_dim=len(feature_cols),
                num_classes=num_classes,
                sequence_length=seq_len
            )
            model = ModelClass(**model_cfg['model_kwargs']).to(device)

            # Load Weights
            checkpoint = torch.load(ckpt_path, map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])

            # Check if there is a cached result for this model
            is_cache_valid = check_cache(result_file.with_suffix('.json'), model, info_cols + ['prediction'])
            if is_cache_valid:
                print(f'\tSkipping {exp_name} (Cached)')
                continue

            # In case there is no cache, we have to calculate results
            print(f'\tNo valid cache found. Evaluating...')

            # Wrap model
            model_wrapper = ModelWrapper(
                name=exp_name,
                model=model,
                features=feature_cols,
                net_type=net_name,
                filter_type=filter_type,
                dataset=ds_type
            )

            # Run inference

            results_df = run_inference(
                model_wrapper=model_wrapper,
                df=df,
                group_cols=group_cols,
                target_col=target_col,
                feature_cols=feature_cols,
                sequence_len=seq_len,
                info_cols=info_cols,
                scaler=scaler,
                label_encoder=label_encoder,
                device=device,
                batch_size=args.batch_size
            )

            # Save
            metadata = compose_metadata(model, exp_name, list(results_df.columns))
            save_csv_and_metadata(results_df, metadata, result_file, index=False)
            print(f'\tSaved to {result_file}')

        print('\nEvaluation Complete')


if __name__ == '__main__':
    main()
