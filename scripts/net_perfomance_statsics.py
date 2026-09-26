import itertools
from argparse import ArgumentParser
import sys
from collections import defaultdict
from pathlib import Path
import pandas as pd

# Add project root to PATH
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

from src.utils.paths import ProjectPaths
from src.data.manipulation import get_results
from src.utils.excel import extract_stats_from_results,  prepare_paths


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
    print('Loading results')
    first_raw_results = get_results(nets, args.configs, 'last', args.subset)

    # Seeded results
    seeded_configs = {cfg: defaultdict(list) for cfg in args.configs}
    for cfg in args.configs:
        for net in nets:
            cfg_paths = list(ProjectPaths.get_experiment_config_path(net, cfg).parent.glob(f'{cfg}_s*'))
            cfg_names = [p.stem for p in cfg_paths]
            seeded_configs[cfg][net].extend(cfg_names)

    seeded_configs = {cfg: list(zip(*l.values())) for cfg, l in seeded_configs.items()}
    seeded_configs = [list(itertools.chain(*item)) for item in zip(*seeded_configs.values())]

    seeded_raw_results = [get_results(nets, cfg, 'best', args.subset) for cfg in seeded_configs]
    print()

    # Stats
    long_sheet_name = 'Main'
    wide_sheet_name = 'Main_Wide'
    main_df_rows, metrics_dfs = zip(*(
        extract_stats_from_results(_r, [long_sheet_name, wide_sheet_name])
        for _r in [first_raw_results] + seeded_raw_results
    ))

    # Prep main df
    main_df = pd.DataFrame(list(itertools.chain(*main_df_rows)))
    main_df_unique_cols = main_df[['Net', 'Feature set', 'Stats']].drop_duplicates()
    main_df_stats = main_df.groupby(['Net', 'Feature set'])['Accuracy'].agg(['mean', 'std']).reset_index()
    main_df_stats = main_df_stats.merge(main_df_unique_cols, on=['Net', 'Feature set'], how='left')

if __name__ == '__main__':
    main()
