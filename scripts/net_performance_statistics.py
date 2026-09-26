import itertools
from argparse import ArgumentParser
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import pandas as pd
from joblib import Memory

# Add project root to PATH
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

from src.utils.paths import ProjectPaths
from src.data.manipulation import get_results
from src.utils.excel import SheetStyles, Style, convert_to_wide_format, extract_stats_from_results, find_better_values, \
    parse_baseline, prepare_paths, write_df_with_style

memory = Memory(project_root / '.math_cache', verbose=0)


def parse_args():
    parser = ArgumentParser('Data aggregation script')
    parser.add_argument('--nets', nargs='+', default=['rnn'], help='Networks to include in report')
    parser.add_argument('--configs', nargs='+', default=['belyaev_kushnarev'], help='Experiment YAML filename')
    parser.add_argument('--subset', type=str, default='test', choices=['test', 'train', 'full'],
                        help='Data subset to evaluate on')
    parser.add_argument('--baseline_df', type=str, default=None, help='Name of classification report df')
    parser.add_argument('--output_name', type=str, default=None, help='Name of output file to overwrite default')
    return parser.parse_args()


def main():
    args = parse_args()
    args.ckpt_type = 'stats'
    nets = sorted(args.nets, key=len, reverse=True)

    baseline_path, output_path = prepare_paths(args)

    # Process data
    long_sheet_name = 'Main'
    wide_sheet_name = 'Main_Wide'
    print('Loading results')
    main_df_rows, metrics_dfs = extract_multi_seed_stats(
        nets,
        args.configs,
        args.subset,
        long_sheet_name,
        wide_sheet_name,
    )

    # Prep main df
    main_df = pd.DataFrame(list(itertools.chain(*main_df_rows)))
    main_df_unique_cols = main_df[['Net', 'Feature set', 'Stats']].drop_duplicates()
    main_df_stats = (
        main_df
        .groupby(['Net', 'Feature set'])
        .agg(
            Accuracy=('Accuracy', 'mean'),
            STD=('Accuracy', 'std'),
        )
        .reset_index()
    )
    main_df_stats = main_df_stats.merge(main_df_unique_cols, on=['Net', 'Feature set'], how='left')
    wide_main_df_stats = convert_to_wide_format(main_df_stats)

    # Prep metrics
    metrics_df_stats = {}
    for k in metrics_dfs[0].keys():
        df = pd.concat([d[k] for d in metrics_dfs], ignore_index=True)
        df = df.groupby('Surface')[['Precision', 'Recall', 'F1-score']].agg(['mean', 'std']).reset_index()
        df.columns = ['_'.join(col).strip('_ ') for col in df.columns.values]
        df = df.merge(metrics_dfs[0][k][['Surface', 'Back to main']], on='Surface', how='left')
        metrics_df_stats[k] = df

    baseline_accuracy_value, baseline_stats_df = parse_baseline(baseline_path)
    # Prep baseline accuracy dfs
    base_acc_long = None
    base_acc_wide = None
    if baseline_accuracy_value:
        base_acc_long = main_df_stats[['Stats', 'Accuracy']].copy()
        base_acc_long['Accuracy'] = baseline_accuracy_value

        base_acc_wide = wide_main_df_stats.copy()
        base_acc_wide.iloc[0:, 1:] = baseline_accuracy_value

    with pd.ExcelWriter(output_path, engine='xlsxwriter') as writer:
        # Base style
        base_style = Style()

        # Set font params to entire workbook for proper auto-width computation
        workbook = writer.book
        workbook.formats[0].set_font_name(base_style['font_name'])
        workbook.formats[0].set_font_size(base_style['font_size'])

        # Other styles
        header_style = base_style.copy()
        header_style.set(bottom=1, bold=1)

        better_stats_style = base_style.copy()
        better_stats_style.set(bold=1)

        link_style = base_style.copy()
        link_style.set(font_color='blue', underline=1)

        separator_style = base_style.copy()
        separator_style.set(top=1)

        sheet_style = SheetStyles(
            base=base_style,
            header=header_style,
            link=link_style,
            better_stats=better_stats_style,
            separator=separator_style,
        )

        # Write long Main df
        write_df_with_style(
            writer=writer,
            sheet_name=long_sheet_name,
            df=main_df_stats,
            sheet_style=sheet_style,
            link_cols='Stats',
            better_stats_idx=find_better_values(main_df_stats, base_acc_long, 'Stats'),
            index_col='Stats',
        )
    print(f'Saving results to: {output_path}')


@memory.cache
def extract_multi_seed_stats(
        nets: list[str],
        configs: list[str],
        subset: str,
        long_sheet_name: str = 'Main',
        wide_sheet_name: str = 'Main_Wide'
) -> tuple[list[dict[str, Any]], list[dict[str, pd.DataFrame]]]:
    first_raw_results = get_results(nets, configs, 'last', subset)
    # Seeded results
    seeded_configs = {cfg: defaultdict(list) for cfg in configs}
    for cfg in configs:
        for net in nets:
            cfg_paths = list(ProjectPaths.get_experiment_config_path(net, cfg).parent.glob(f'{cfg}_s*'))
            cfg_names = [p.stem for p in cfg_paths]
            seeded_configs[cfg][net].extend(cfg_names)
    seeded_configs = {cfg: list(zip(*l.values())) for cfg, l in seeded_configs.items()}
    seeded_configs = [list(itertools.chain(*item)) for item in zip(*seeded_configs.values())]
    seeded_raw_results = [get_results(nets, cfg, 'best', subset) for cfg in seeded_configs]
    print()
    # Stats
    main_df_rows, metrics_dfs = zip(*(
        extract_stats_from_results(_r, [long_sheet_name, wide_sheet_name])
        for _r in [first_raw_results] + seeded_raw_results
    ))
    return main_df_rows, metrics_dfs


if __name__ == '__main__':
    main()
