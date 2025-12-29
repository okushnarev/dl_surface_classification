import json

from src.utils.paths import ProjectPaths


def main():
    # Metadata
    metadata = {
        'group_cols':   ['surf', 'run_idx'],
        'info_cols':    ['surf', 'run_idx'],
        'target_col':   'surf',
        'class_colors': {
            'asphalt':    '#2d2d2d',
            'flooring':   '#c8a2c8',
            'ice':        '#89dee2',
            'sandy_loam': '#d6c68b',  # silty_loam in the article
            'snow':       '#f0f8ff',
        }
    }

    # Features
    features = {}
    filters = {
        'no_filter': '',
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

        imu = ['wx', 'wy', 'wz', 'ax', 'ay', 'az']
        in_data['type_4'] = [f'm1cur{suffix}', f'm2cur{suffix}', f'm3cur{suffix}'] + imu
        in_data['type_5'] = in_data['type_1'] + imu
        in_data['type_6'] = in_data['type_2'] + imu

        features[filter] = in_data

    # Output
    output = dict(
        metadata=metadata,
        features=features,
    )
    output_path = ProjectPaths.get_dataset_config_path('boreal')
    with output_path.open('w') as f:
        json.dump(output, f, indent=2)


if __name__ == '__main__':
    main()
