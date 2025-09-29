import subprocess
import sys
import yaml
from pathlib import Path

if __name__ == '__main__':
    directory = Path('python/rnn/')
    # Load experiments from the config file
    with open(directory / 'configs' / 'experiments.yaml', 'r') as f:
        experiments = yaml.safe_load(f)

    # The rest of the script is almost identical to Option 1
    for idx, config in enumerate(experiments):
        print('-' * 50)
        print(f'Running experiment {idx + 1}/{len(experiments)}: {config["name"]}')


        # Build the command
        command = [sys.executable, directory / 'optimize_rnn.py']
        for key, value in config.items():
            if key in ('optimization', 'common'):
                # Parse inner dictionaries
                for k, v in value.items():
                    command.extend([f'--{k}', str(v)])

        print(' '.join([str(c) for c in command]))
        print('-' * 50)
        try:
            subprocess.run(command, check=True)
        except subprocess.CalledProcessError as e:
            print(f'Error running experiment {config['name']}. Stopping.')
            break

    print('\nAll experiments completed!')
