from argparse import ArgumentParser
import sys
from pathlib import Path
# Add project root to PATH
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))
from src.data.manipulation import get_results
from src.utils.excel import prepare_paths


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
