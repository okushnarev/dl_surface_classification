import argparse
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=69, help='Random seed')
    return parser.parse_args()


def main():
    args = parse_args()
    random_seed = args.seed

    input_path = Path('processed/raw')
    output_path = Path('processed/processed')

    df = pd.read_csv(input_path / 'concat_noavg_kalman.csv')

    # Warning! Split shuffles the processed
    # For robot processed I assume that the order is not important
    # Reimplement if order should be conserved
    df_train, df_test = train_test_split(df,
                                         train_size=0.8,
                                         shuffle=True,
                                         stratify=df['surf'],
                                         random_state=random_seed)
    df_train, df_val = train_test_split(df_train,
                                        train_size=0.875,
                                        shuffle=True,
                                        stratify=df_train['surf'],
                                        random_state=random_seed)

    df_train.to_csv(output_path / 'train.csv', index=False)
    df_test.to_csv(output_path / 'test.csv', index=False)
    df_val.to_csv(output_path / 'val.csv', index=False)

if __name__ == '__main__':
    main()
