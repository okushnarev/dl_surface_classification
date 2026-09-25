from argparse import ArgumentParser


def parse_args():
    parser = ArgumentParser('Data aggregation script')
    parser.add_argument('--nets', nargs='+', default=['rnn'], help='Networks to include in report')
    parser.add_argument('--configs', nargs='+', default=['belyaev_kushnarev'], help='Experiment YAML filename')
    parser.add_argument('--subset', type=str, default='test', choices=['test', 'train', 'full'],
                        help='Data subset to evaluate on')
    return parser.parse_args()