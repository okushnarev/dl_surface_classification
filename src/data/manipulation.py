from typing import Iterable

import pandas as pd
import yaml

from src.utils.paths import ProjectPaths


def get_results(
        nets: list[str],
        configs: list[str],
        ckpt_type: str,
        subset: str
) -> dict[str, pd.DataFrame]:
    """
    Load results for specified networks, configurations, and subset

    :param ckpt_type:
    :param nets: List of network names
    :param configs: List of experiment configuration names
    :param subset: Data subset to evaluate ('test', 'train', or 'full')
    :returns: Dictionary mapping model names to (metadata dict, DataFrame) tuples
    """
    results = {}
    for net in nets:
        for config_name in configs:
            exp_cfg_path = ProjectPaths.get_experiment_config_path(net, config_name)
            cache_path = ProjectPaths.get_evaluation_dir(config_name, ckpt_type, subset)

            if not exp_cfg_path.exists():
                print(f'\tNo experiment config file at {exp_cfg_path}')
                continue

            with open(exp_cfg_path, 'r') as f:
                config = yaml.safe_load(f)
            experiments = config['experiments']

            model_names = set()
            for exp in experiments:
                model_name = exp['name']
                print(f'Loading experiment {model_name}')
                if model_name in model_names:
                    model_name += config_name
                    print(f'\tThe name is already taken, changing to {model_name}')
                model_names.add(model_name)

                filter_type = exp['common']['filter']
                ds_type = exp['common']['ds_type']

                # Check for usual cache
                cache_info_path = cache_path / f'{model_name}.csv'
                if cache_info_path.exists():
                    metadata = {
                        'net':         net,
                        'filter_type': filter_type,
                        'ds_type':     ds_type,
                    }
                    results[model_name] = (metadata, pd.read_csv(cache_info_path))
                else:
                    print(f'\tNo cached results for: linear {model_name}')
    return results


def align_dfs(
        dfs: Iterable[pd.DataFrame],
        index_col: str
) -> list[pd.DataFrame]:
    """
    Align multiple DataFrames to have the same index and columns

    :param dfs: Iterable of DataFrames to align
    :param index_col: Name of the column to use as index for alignment
    :returns: List of aligned DataFrames with common index and columns
    """
    dfs = [df.copy() for df in dfs]

    cols = set(dfs[0].columns)
    index = set(dfs[0][index_col])
    for idx, df in enumerate(dfs):
        dfs[idx] = df.set_index(index_col)
        cols &= set(dfs[idx].columns)
        index &= set(dfs[idx].index)

    cols = list(cols)
    index = list(index)
    for idx, df in enumerate(dfs):
        dfs[idx] = df.loc[index, cols]
    return dfs


def match_original_indices(
        df_orig: pd.DataFrame,
        coordinates: list[tuple[str, str]],
        index_column: str
) -> list[tuple[int, int]]:
    """
    Convert string-based coordinates to numeric indices in original DataFrame

    :param df_orig: Original DataFrame to use for index mapping
    :param coordinates: List of (row_label, column_label) tuples
    :param index_column: Name of the column containing row labels
    :returns: List of (row_index, column_index) integer tuples
    """
    rows = df_orig[index_column].tolist()
    columns = df_orig.columns.tolist()
    result = []
    for coord in coordinates:
        _r, _c = coord
        orig_row = rows.index(_r)
        orig_col = columns.index(_c)
        result.append((orig_row, orig_col))
    return result
