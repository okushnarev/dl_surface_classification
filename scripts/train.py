import argparse
import subprocess
import sys
import time
from pathlib import Path

import yaml

# Add project root to PATH
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

from src.utils.paths import ProjectPaths
from src.engine.trainer import add_trainer_args, train_model


def parse_args():
    parser = argparse.ArgumentParser(description='Training Runner')

    # Singe mod to run individual experiments without config file
    parser.add_argument('--single_run', action='store_true', help='Run a single experiment via CLI arguments')

    # Runner args
    parser.add_argument('--nn_name', type=str, help='Name of the model (folder in configs/experiments)')
    parser.add_argument('--config_name', type=str, default='main', help='Name of the YAML config file (without .yaml)')
    parser.add_argument('--n_jobs', type=int, default=1, help='Number of parallel experiments to run')

    # Trainer args from engine/trainer
    # This adds --epochs, --batch_size, --lr, etc. to the parser automatically
    parser = add_trainer_args(parser)

    # Handle possibly conflicting arguments
    if not any(action.dest == 'nn_name' for action in parser._actions):
        parser.add_argument('--nn_name', type=str, required=True)
    if not any(action.dest == 'dataset' for action in parser._actions):
        parser.add_argument('--dataset', type=str, default='main')

    return parser.parse_args()


def run_batch_mode(args):
    """
    Reads YAML and spawns subprocesses for each experiment
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

    print(f'--- Loaded {len(experiments)} experiments from {config_path.name} ---')

    # Track processes
    active_processes = []  # [(process_handle, exp_name, log_file_handle)]
    # Queue of experiments to run
    queue = list(enumerate(experiments))

    print(f'--- Starting: {len(experiments)} experiments, {args.jobs} parallel jobs ---')

    while len(queue) > 0 or len(active_processes) > 0:
        # Check for finished processes
        # Iterate in reverse since the .pop() works from the end of the list
        for i in range(len(active_processes) - 1, -1, -1):
            proc, name, log_f = active_processes[i]

            # poll() returns None if running
            exit_code = proc.poll()

            if exit_code is not None:
                # Process finished
                log_f.close()
                active_processes.pop(i)

                if exit_code == 0:
                    print(f'Finished: {name}')
                else:
                    print(f'Failed: {name} (Check logs)')

        # Launch new processes if there are free slots
        while len(active_processes) < args.jobs and len(queue) > 0:
            idx, exp = queue.pop(0)  # Get the next experiment
            exp_name = exp.get('name', f'exp_{idx}')

            if exp.get('skip', False):
                print(f"Skipping {exp_name}...")
                continue

            # Merge arguments. Priority: Experiment > Defaults
            run_config = defaults.copy()

            # Merge sub-sections (common, train)
            for section in ['common', 'train']:
                if section in defaults:
                    run_config |= defaults[section]
                if section in exp:
                    run_config |= exp[section]

            # Add explicit experiment overrides
            for k, v in exp.items():
                if k not in ['common', 'train', 'skip', 'name']:
                    run_config[k] = v

            # Set required args
            run_config['nn_name'] = args.nn_name
            run_config['exp_name'] = exp_name

            # Compose CLI command
            # Call THIS script again, but in --single_run mode
            cmd = [sys.executable, str(Path(__file__).resolve()), '--single_run']

            for key, value in run_config.items():
                if key in ['common', 'train']:
                    # Processed already
                    continue

                if value is None:
                    continue

                # Handle Booleans (Flags)
                if isinstance(value, bool):
                    if value is True:
                        cmd.append(f'--{key}')
                else:
                    # Handle standard values
                    cmd.extend([f'--{key}', str(value)])

            # Compose a path to store logs
            dataset_scope = args.config_name
            log_dir = ProjectPaths.get_run_dir(dataset_scope, exp_name)
            log_dir.mkdir(parents=True, exist_ok=True)
            log_path = log_dir / 'training_log.txt'

            # Non-blocking launch
            print(f'Launching: {exp_name} -> Logs: {log_path}')
            # Redirect stdout and stderr to file
            f_out = open(log_path, 'w')
            proc = subprocess.Popen(cmd, stdout=f_out, stderr=subprocess.STDOUT)

            active_processes.append((proc, exp_name, f_out))
        time.sleep(1)


def run_worker_mode(args):
    """
    Runs the training code
    """
    # Call the engine
    train_model(args)


if __name__ == '__main__':
    args = parse_args()

    if args.single_run:
        run_worker_mode(args)
    else:
        run_batch_mode(args)
