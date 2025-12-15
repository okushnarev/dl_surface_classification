import json
from pathlib import Path


def main():
    output_path = Path('data/datasets')

    datasets = {}
    filters = {
        'no_filter': '',
        'kalman':    '_corr',
    }
    for filter, suffix in filters.items():
        in_data = {
            'type_1': [
                f'm1cur{suffix}', f'm2cur{suffix}', f'm3cur{suffix}',
                f'm1vel{suffix}', f'm2vel{suffix}', f'm3vel{suffix}',
            ],
            'type_2': [
                f'rpower{suffix}', f'Ke1{suffix}',
            ],
        }
        in_data['type_3'] = in_data['type_1'] + in_data['type_2']
        if filter == 'kalman':
            in_data['type_4'] = in_data['type_2'] + ['movedir']
            in_data['type_5'] = ['movedir', 'speedamp', f'Ke1{suffix}']
        datasets[filter] = in_data


    with (output_path / 'datasets_features.json').open('w') as f:
        json.dump(datasets, f, indent=2)


if __name__ == '__main__':
    main()
