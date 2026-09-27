import json
import sys
from argparse import ArgumentParser
from pathlib import Path

import codegreen
import joblib
import numpy as np
import onnx_tool
import pandas as pd
import torch
import yaml

# Add project root to PATH
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

from src.utils.paths import ProjectPaths
from src.models.factory import get_model_components
from src.performance.inference_time import profile_inference_time, setup_ort_session
from src.performance.onnx_export import export_model_to_onnx


def parse_args():
    parser = ArgumentParser('Script for inference profiling')
    parser.add_argument('--nets', nargs='+', default=['rnn'], help='Networks to include in report')
    parser.add_argument('--configs', nargs='+', default=['belyaev_kushnarev'], help='Experiment YAML filename')
    parser.add_argument('--ckpt-type', type=str, choices=['last', 'best'],
                        default='last', help='Model\'s checkpoint type to load')
    parser.add_argument('--output_name', type=str, default=None, help='Name of output file to overwrite default')
    return parser.parse_args()


def main():
    args = parse_args()
    nets = sorted(args.nets, key=len, reverse=True)
    device = 'cpu'

    output_name = args.output_name or '_'.join(args.configs)
    output_path = ProjectPaths.get_tables_dir('profiling', args.ckpt_type) / f'{output_name}.csv'
    output_path.parent.mkdir(exist_ok=True, parents=True)

    results = []
    for config_name in args.configs:
        for net in nets:
            exp_cfg_path = ProjectPaths.get_experiment_config_path(net, config_name)
            if not exp_cfg_path.exists():
                print(f'Skipping {net}: Config not found at {exp_cfg_path}')
                continue

            with open(exp_cfg_path, 'r') as f:
                yaml_content = yaml.safe_load(f)

            defaults = yaml_content.get('defaults', {})
            experiments = yaml_content.get('experiments', [])

            # Load data config
            dataset = defaults['dataset']
            ds_config_path = ProjectPaths.get_dataset_config_path(dataset)
            with open(ds_config_path, 'r') as f:
                ds_config = json.load(f)
            features_map = ds_config['features']

            # Loop over experiments
            for exp in experiments:
                exp_name = exp.get('name')
                print(f'\nProcessing {config_name} – {exp_name}')

                # Prepare experiment args
                exp_args = defaults.get('common', {}).copy()
                exp_args |= exp.get('common', {})

                filter_type = exp_args.get('filter', 'no_filter')
                feature_set = exp_args.get('feature_set', 'type_1')

                # Identify features
                try:
                    feature_cols = features_map[filter_type][feature_set]
                except KeyError:
                    print(f'Error: Could not find features for {filter_type}/{feature_set} in dataset config.')
                    continue

                # Checkpoint, scaler, label encoder
                run_dir = ProjectPaths.get_run_dir(config_name, exp_name)

                ckpt_path = run_dir / f'{args.ckpt_type}.pt'
                scaler_path = run_dir / 'scaler.joblib'
                label_encoder_path = run_dir / 'label_encoder.joblib'

                if not ckpt_path.exists():
                    print(f'\tCheckpoint not found: {ckpt_path}. Skipping')
                    continue

                # Load label encoder
                if not label_encoder_path.exists():
                    print(f'\tLabelEncoder not found: {label_encoder_path}. Implying output shape from dataset config')
                    num_classes = len(ds_config['metadata']['class_colors'])
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
                    cfg_path = ProjectPaths.get_params_path(net, config_name, exp_name)

                # Prepare config
                seq_len: int = exp_args.get('seq_len', 10)

                # Create model
                components = get_model_components(net)
                ModelClass = components['class']
                prep_cfg = components['prep_config']

                model_cfg = prep_cfg(
                    cfg_path,
                    input_dim=len(feature_cols),
                    num_classes=num_classes,
                    sequence_length=seq_len
                )
                model = ModelClass(**model_cfg['model']).to(device)

                # Load Weights
                checkpoint = torch.load(ckpt_path, map_location=device)
                try:
                    model.load_state_dict(checkpoint['model_state_dict'])
                except Exception as e:
                    print(f'Exception occured during weights loading. Skipping.\n{e}')
                    continue

                onnx_path = run_dir / 'model.onnx'
                export_model_to_onnx(model, onnx_path, (1, seq_len, len(feature_cols)))

                # Inference time profiling
                n_runs = 100
                print(f'  Starting inference time profiling with {n_runs} runs')
                ort_session = setup_ort_session(onnx_path)
                elapsed_time = np.array(profile_inference_time(ort_session, n_runs))

                # MACs profiling
                print(f'  Starting MAC and Params count')
                model = onnx_tool.Model(onnx_path)
                model.graph.shape_infer()
                model.graph.profile()
                total_macs = model.graph.macs[0]
                total_params = model.graph.params

                # Energy profiling
                n_energy_runs = 10_000
                print(f'  Starting energy profiling with {n_energy_runs} runs')
                task_name = 'forward_pass'
                with codegreen.Session('onnx_inference', save_to_file=False) as s:
                    with s.task(task_name):
                        profile_inference_time(ort_session, n_energy_runs, 0)

                energy_per_run = None
                for task in s.tasks:
                    if task.name == task_name:
                        energy_per_run = task.energy_j / n_energy_runs

                results.append({
                    'config':              config_name,
                    'exp_name':            exp_name,
                    'net':                 net,
                    'filter_type':         filter_type,
                    'feature_set':         feature_set,
                    'inference_time_mean': elapsed_time.mean(),
                    'inference_time_std':  elapsed_time.std(),
                    'macs':                total_macs,
                    'params':              total_params,
                    'energy_per_run':      energy_per_run,
                })
    df_res = pd.DataFrame(results)
    df_res.to_csv(output_path, index=False)


if __name__ == '__main__':
    main()
