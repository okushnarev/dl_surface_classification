import json

from src.utils.paths import ProjectPaths


def main():
    # Metadata
    metadata = {
        'group_cols':   ['surf', 'run_idx'],
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
        in_data = {
            'type_1': [
                f'curL{suffix}', f'curR{suffix}',
                f'velL{suffix}', f'velR{suffix}',
            ],
            'type_2': [
                f'Ke1{suffix}',
            ],
        }
        in_data['type_3'] = in_data['type_1'] + in_data['type_2']
        in_data['type_4'] = in_data['type_2'] + ['movedir']
        in_data['type_5'] = ['movedir', 'speedamp', f'Ke1{suffix}']
        in_data['type_6'] = in_data['type_3'] + ['wx', 'wy', 'wz', 'ax', 'ay', 'az', ]  # IMU and Power data
        in_data['type_7'] = [f'velL{suffix}', f'velR{suffix}', 'wx', 'wy', 'wz', 'ax', 'ay',
                             'az']  # IMU + velocities only
        in_data['type_8'] = in_data['type_1'] + ['wx', 'wy', 'wz', 'ax', 'ay', 'az']  # Original feature set
        in_data['type_9'] = [f'velL{suffix}', f'velR{suffix}', f'Ke1{suffix}']

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
