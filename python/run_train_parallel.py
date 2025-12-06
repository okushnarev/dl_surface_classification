import argparse
import subprocess
import sys
from multiprocessing import Pool
from pathlib import Path

import yaml


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--nn_name', type=str, default='rnn', required=True, help='Name of neural network')
    parser.add_argument('--processes', type=int, default=1, help='Number of processes to use')
    return parser.parse_args()

def run_command(command, name):
    print(f'Running experiment: {name}')
    print(' '.join([str(c) for c in command]))
    subprocess.run(command)

if __name__ == '__main__':
    args = parse_args()
    nn_name = args.nn_name

    directory = Path(f'python/{nn_name}/')
    # Load experiments from the config file
    with open(directory / 'configs' / 'experiments.yaml', 'r') as f:
        experiments = yaml.safe_load(f)

    commands = []
    names = []
    for idx, config in enumerate(experiments):
        # print(f'Running experiment {idx + 1}/{len(experiments)}: {config["name"]}')
        names.append(config['name'])
        if 'skip' in config and config['skip']:
            print('Skipping experiment due to config param "skip"')
            print('-' * 50)
            continue

        # Build the command
        command = [sys.executable, directory / f'train_{nn_name}.py']
        for key, value in config.items():
            if key == 'name':
                command.extend([f'--exp_name', str(value)])

                suffix = value.replace(nn_name, '').strip('_')
                param_file_name = f'best_params_{suffix}'
                if (p := Path(f'data/params/{nn_name}_optim/{param_file_name}.json')).exists():
                    command.extend([f'--config', str(p)])
            elif key in ('train', 'common'):
                # Parse inner dictionaries
                for k, v in value.items():
                    command.extend([f'--{k}', str(v)] if v != 'None' else [f'--{k}'])

        commands.append(command)

    with Pool(processes=args.processes) as pool:
        pool.starmap(run_command, zip(commands, names))

    print('\nAll experiments completed!')
