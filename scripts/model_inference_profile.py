from argparse import ArgumentParser


def parse_args():
    parser = ArgumentParser('Script for inference profiling')
    parser.add_argument('--nets', nargs='+', default=['rnn'], help='Networks to include in report')
    parser.add_argument('--configs', nargs='+', default=['belyaev_kushnarev'], help='Experiment YAML filename')
    parser.add_argument('--output_name', type=str, default=None, help='Name of output file to overwrite default')
    parser.add_argument('--column-format', type=str, choices=['separate', 'combined'], default='separate',
                        help='Whether to store mean and std data in separate columns or combined with ±. '
                             'Converted to string. Number of decimals is set with `--decimals`')
    parser.add_argument('--decimals', type=int, default=2,
                        help='Number of decimals to show in ± annotation. Works when `--column-format` is set to `combined`')
    return parser.parse_args()


def main():
    pass


if __name__ == '__main__':
    main()
