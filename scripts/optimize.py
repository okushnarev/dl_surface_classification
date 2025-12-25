import argparse
import subprocess
import sys
from pathlib import Path

import yaml

# Add project root to PATH
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

from src.utils.paths import ProjectPaths
from src.engine.optimizer import add_optimizer_args, run_optimization


def parse_args():
    parser = argparse.ArgumentParser(description='Hyperparameter Optimization Runner')

    # Singe mod to run individual experiments without config file
    parser.add_argument('--single_run', action='store_true', help='Run a single optimization task via CLI arguments')

    # Runner args
    parser.add_argument('--nn_name', type=str, help='Name of the model (folder in configs/experiments)')
    parser.add_argument('--config_name', type=str, default='main', help='YAML config filename')

    # Optimizer args from engine/optimizer
    # Adds --n_trials, --n_jobs, --epochs, --filter, etc.
    parser = add_optimizer_args(parser)

    # Ensure compulsory args exist in the parser for Manager Mode usage
    if not any(action.dest == 'nn_name' for action in parser._actions):
        parser.add_argument('--nn_name', type=str, required=True)
    if not any(action.dest == 'dataset' for action in parser._actions):
        parser.add_argument('--dataset', type=str, default='main')

    return parser.parse_args()


def run_batch_mode(args):
    """
    Reads YAML and runs optimization experiments sequentially.
    """
    # Load config with ProjectPaths
    config_path = ProjectPaths.get_experiment_config_path(args.nn_name, args.config_name)

    if not config_path.exists():
        print(f'Error: Config not found at {config_path}')
        sys.exit(1)

    with open(config_path, 'r') as f:
        yaml_config = yaml.safe_load(f)

    defaults = yaml_config.get('defaults', {})
    experiments = yaml_config.get('experiments', [])

    print(f'--- Loaded {len(experiments)} optimization tasks from {config_path.name} ---')

    for idx, exp in enumerate(experiments):
        exp_name = exp.get('name', f'exp_{idx}')

        if exp.get('skip', False):
            print(f'Skipping {exp_name}...')
            continue

        print(f'\nRunning Optimization {idx + 1}/{len(experiments)}: {exp_name}')

        # Merge arguments. Priority: Experiment > Defaults
        run_config = defaults.copy()

        # Merge sub-sections (common, optimization)
        for section in ['common', 'optimization']:
            if section in defaults:
                run_config.update(defaults[section])
            if section in exp:
                run_config.update(exp[section])

        # Set required args
        run_config['nn_name'] = args.nn_name

        # Compose CLI command
        # Call THIS script again, but in --single_run mode
        cmd = [sys.executable, str(Path(__file__).resolve()), '--single_run']

        for key, value in run_config.items():
            if value is None or value == 'None': continue

            # Handle Booleans (Flags)
            if isinstance(value, bool):
                if value is True:
                    cmd.append(f'--{key}')
            else:
                # Handle standard values
                cmd.extend([f'--{key}', str(value)])

        try:
            subprocess.run(cmd, check=True)
        except subprocess.CalledProcessError:
            print(f'Error optimizing {exp_name}')
            continue


def run_worker_mode(args):
    """
    Worker Mode: Calls the actual optimization engine.
    """
    run_optimization(args)


if __name__ == '__main__':
    args = parse_args()

    if args.single_run:
        run_worker_mode(args)
    else:
        run_batch_mode(args)
