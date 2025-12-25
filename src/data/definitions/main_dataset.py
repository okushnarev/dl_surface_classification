import json

from src.utils.paths import ProjectPaths


def main():
    metadata = {
        'group_cols':   ['surf', 'movedir', 'speedamp'],
        'target_col':   'surf',
        'class_colors': {
            'gray':  '#b6b6b6',
            'green': '#4fc54c',
            'table': '#9f6a4d',
            'brown': '#ad3024',
        }
    }

    features = {}
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
        features[filter] = in_data

    output = dict(
        metadata=metadata,
        features=features,
    )
    output_path = ProjectPaths.get_dataset_config_path('main')
    with output_path.open('w') as f:
        json.dump(output, f, indent=2)


if __name__ == '__main__':
    main()
