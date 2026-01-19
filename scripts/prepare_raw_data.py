import argparse
import sys
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

# Add project root to PATH
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

from src.utils.paths import ProjectPaths


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='main')
    parser.add_argument('--full_ds_name', type=str, default='main')
    parser.add_argument('--seed', type=int, default=69, help='Random seed')
    return parser.parse_args()


def main():
    args = parse_args()
    input_path = ProjectPaths.get_raw_data_dir()

    df = pd.read_csv(input_path / f'{args.full_ds_name}.csv')

    df_train, df_test = train_test_split(df,
                                         train_size=0.8,
                                         shuffle=True,
                                         stratify=df['surf'],
                                         random_state=args.seed)
    _, df_proxy = train_test_split(df_train,
                                        test_size=0.4,
                                        shuffle=True,
                                        stratify=df_train['surf'],
                                        random_state=args.seed)
    df_train = df_train.sort_index()
    df_test = df_test.sort_index()
    df_proxy = df_proxy.sort_index()

    output_path = ProjectPaths.get_processed_data_dir(args.dataset)
    output_path.mkdir(parents=True, exist_ok=True)

    df_train.to_csv(output_path / 'train.csv', index=False)
    df_test.to_csv(output_path / 'test.csv', index=False)
    df_proxy.to_csv(output_path / 'proxy.csv', index=False)


if __name__ == '__main__':
    main()
