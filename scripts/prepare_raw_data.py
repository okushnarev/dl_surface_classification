import argparse
import json
import sys
from pathlib import Path

import pandas as pd

# Add project root to PATH
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

from src.utils.paths import ProjectPaths
from src.data.processing import chunk_split
from src.utils.vars import CHUNK_COL


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='main')
    parser.add_argument('--full_ds_name', type=str, default='main')
    parser.add_argument('--chunk_size', type=int, default=300)
    parser.add_argument('--proxy_size', type=float, default=0.4,
                        help='Part of the train set that will be used as proxy dataset')
    parser.add_argument('--seed', type=int, default=69, help='Random seed')
    return parser.parse_args()


def main():
    args = parse_args()
    input_path = ProjectPaths.get_raw_data_dir()

    df = pd.read_csv(input_path / f'{args.full_ds_name}.csv')

    with open(ProjectPaths.get_dataset_config_path(args.dataset), 'r') as f:
        datasets_cfg = json.load(f)

    group_cols = datasets_cfg['metadata']['group_cols']
    target_col = datasets_cfg['metadata']['target_col']

    # Create chunk col in full dataset
    df[CHUNK_COL] = df.groupby(group_cols).cumcount() // args.chunk_size

    df_train, df_test = chunk_split(df,
                                    group_cols=group_cols,
                                    target_col=target_col,
                                    # kwargs
                                    train_size=0.8,
                                    random_state=args.seed)
    df_proxy, _ = chunk_split(df_train,
                              group_cols=group_cols,
                              target_col=target_col,
                              # kwargs
                              train_size=args.proxy_size,
                              random_state=args.seed)

    output_path = ProjectPaths.get_processed_data_dir(args.dataset)
    output_path.mkdir(parents=True, exist_ok=True)

    df_train.to_csv(output_path / 'train.csv', index=False)
    df_test.to_csv(output_path / 'test.csv', index=False)
    df_proxy.to_csv(output_path / 'proxy.csv', index=False)


if __name__ == '__main__':
    main()
