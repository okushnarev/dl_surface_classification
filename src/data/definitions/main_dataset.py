import json

from sympy.physics.units import energy

from src.utils.paths import ProjectPaths


def main():
    metadata = {
        'group_cols':   ['exp_idx'],
        'info_cols':    ['surf', 'movedir', 'speedamp'],
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
        in_data = dict()

        in_data['type_1'] = [
            f'm1cur{suffix}', f'm2cur{suffix}', f'm3cur{suffix}',
            f'm1vel{suffix}', f'm2vel{suffix}', f'm3vel{suffix}',
        ]

        energy = [f'Ke1{suffix}']
        in_data['type_2'] = in_data['type_1'] + energy
        in_data['type_3'] = ['movedir'] + energy

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
