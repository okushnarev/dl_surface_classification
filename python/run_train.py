import argparse
import subprocess
import sys
from pathlib import Path

import yaml


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--nn_name', type=str, default='rnn', required=True, help='Name of neural network')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    nn_name = args.nn_name

    directory = Path(f'python/{nn_name}/')
    # Load experiments from the config file
    with open(directory / 'configs' / 'experiments.yaml', 'r') as f:
        experiments = yaml.safe_load(f)

    # The rest of the script is almost identical to Option 1
    for idx, config in enumerate(experiments):
        print('-' * 50)
        print(f'Running experiment {idx + 1}/{len(experiments)}: {config["name"]}')
        if 'skip' in config and config['skip']:
            print('Skipping experiment due to config param "skip"')
            print('-' * 50)
            continue

        # Build the command
        command = [sys.executable, directory / f'train_{nn_name}.py']
        for key, value in config.items():
            if key == 'name':
                command.extend([f'--exp_name', str(value)])

                param_file_name = '_'.join(['best', 'params'] + value.split('_')[1:])
                command.extend([f'--config', f'data/params/{nn_name}_optim/{param_file_name}.json'])
            elif key in ('train', 'common'):
                # Parse inner dictionaries
                for k, v in value.items():
                    command.extend([f'--{k}', str(v)] if v != 'None' else [f'--{k}'])

        print(' '.join([str(c) for c in command]))
        print('-' * 50)
        try:
            subprocess.run(command, check=True)
        except subprocess.CalledProcessError as e:
            print(f'Error running experiment {config['name']}. Stopping.')
            break

    print('\nAll experiments completed!')
