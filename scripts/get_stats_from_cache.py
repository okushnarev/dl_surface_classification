import argparse
import sys
from pathlib import Path

import pandas as pd
import yaml
from sklearn.metrics import accuracy_score

# Add project root to PATH
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

from src.utils.paths import ProjectPaths


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--nets', '-n', nargs='+', default=['rnn'], help='Nets to test')
    parser.add_argument('--ds', default='full', choices=['full', 'test'], help='Dataset to use')
    return parser.parse_args()


def main():
    args = parse_args()

    # Ensure same nets' names order
    nets = sorted(args.nets, key=len, reverse=True)

    # Data path
    cache_path = ProjectPaths.get_evaluation_dir('main', args.ds).parent
    data_prefix = '_'.join(nets)
    results_path = cache_path / 'csv' / data_prefix
    results_path.mkdir(parents=True, exist_ok=True)


    # Gather processed
    raw_results = {
        args.ds: {},
        'square': {},
        'circle': {},
    }
    for net in nets:
        exp_cfg_path = ProjectPaths.get_experiment_config_path(net, 'main')
        with open(exp_cfg_path, 'r') as f:
            config = yaml.safe_load(f)

        experiments = config['experiments']
        for exp in experiments:
            model_name = exp['name']
            print(f'Loading experiment {model_name}')

            filter_type = exp['common']['filter']
            ds_type = exp['common']['ds_type']

            for trajectory_type in raw_results.keys():
                cache_info_path = cache_path / trajectory_type / f'{model_name}.csv'
                if cache_info_path.exists():
                    metadata = {
                        'net':         net,
                        'filter_type': filter_type,
                        'ds_type':     ds_type,
                    }
                    raw_results[trajectory_type][model_name] = (metadata, pd.read_csv(cache_info_path))
                else:
                    print(f'No cached results for: {trajectory_type} {model_name}')

    # Process results
    for trajectory_type in raw_results.keys():
        acc_df = pd.DataFrame([
            {**metadata, 'accuracy': accuracy_score(df['surf'], df['prediction'])}
            for metadata, df in raw_results[trajectory_type].values()
        ])

        acc_df.to_csv(results_path / f'{trajectory_type}.csv', index=False)

        try:
            pivot_acc_df = acc_df.pivot(
                index=['net', 'filter_type'],
                columns='ds_type',
                values='accuracy'
            )
            pivot_acc_df.to_csv(results_path / f'pivot_{trajectory_type}.csv', index=True)
        except Exception as e:
            print(f'Could not save pivot results because {e}')


if __name__ == '__main__':
    main()
