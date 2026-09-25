import argparse
import sys
from dataclasses import replace
from pathlib import Path

import pandas as pd

# Add project root to PATH
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

from src.data.manipulation import get_results
from src.utils.excel import SheetStyles, Style, convert_to_wide_format, extract_stats_from_results, find_better_values, \
    parse_baseline, prepare_paths, write_df_with_style


def parse_args():
    parser = argparse.ArgumentParser('Data aggregation script')
    parser.add_argument('--nets', nargs='+', default=['rnn'], help='Networks to include in plots')
    parser.add_argument('--configs', nargs='+', default=['belyaev_kushnarev'], help='Experiment YAML filename')
    parser.add_argument('--subset', type=str, default='test', choices=['test', 'train', 'full'],
                        help='Data subset to evaluate on')
    parser.add_argument('--baseline_df', type=str, default=None, help='Name of classification report df')
    parser.add_argument('--output_name', type=str, default=None, help='Name of output file to overwrite default')
    parser.add_argument('--ckpt_type', default='best', choices=['best', 'last'], help='Checkpoint to load')
    return parser.parse_args()


def main():
    args = parse_args()
    nets = sorted(args.nets, key=len, reverse=True)

    baseline_path, output_path = prepare_paths(args)

    # Process data
    print('Loading results')
    raw_results = get_results(nets, args.configs, args.ckpt_type, args.subset)
    print()

    long_sheet_name = 'Main'
    wide_sheet_name = 'Main_Wide'
    main_df_rows, stats_dfs = extract_stats_from_results(
        raw_results,
        [long_sheet_name, wide_sheet_name]
    )

    df_long = pd.DataFrame(main_df_rows)
    df_wide = convert_to_wide_format(df_long)

    baseline_accuracy_value, baseline_stats_df = parse_baseline(baseline_path)
    # Prep baseline accuracy dfs
    base_acc_long = None
    base_acc_wide = None
    if baseline_accuracy_value:
        base_acc_long = df_long[['Stats', 'Accuracy']].copy()
        base_acc_long['Accuracy'] = baseline_accuracy_value

        base_acc_wide = df_wide.copy()
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
            df=df_long,
            sheet_style=sheet_style,
            link_cols='Stats',
            better_stats_idx=find_better_values(df_long, base_acc_long, 'Stats'),
            index_col='Stats',
        )

        # Find better_stats_idx for wide format
        numeric_df_wide = convert_to_wide_format(df_long, 'Accuracy')
        better_stats_idx = find_better_values(numeric_df_wide, base_acc_wide, 'Net')
        better_stats_style_wide = better_stats_style + link_style
        wide_sheet_style = replace(sheet_style, better_stats=better_stats_style_wide)

        # Write wide Main df
        write_df_with_style(
            writer=writer,
            sheet_name=wide_sheet_name,
            df=df_wide,
            sheet_style=wide_sheet_style,
            link_cols=df_wide.columns.tolist()[1:],  # Ignoring 'Net' column (non-numeric)
            better_stats_idx=better_stats_idx,
        )

        # Write numeric wide main df
        write_df_with_style(
            writer=writer,
            sheet_name=f'{wide_sheet_name}_raw',
            df=numeric_df_wide,
            sheet_style=sheet_style,
            better_stats_idx=better_stats_idx,
        )

        for sheet_name, _df in stats_dfs.items():
            write_df_with_style(
                writer=writer,
                sheet_name=sheet_name,
                df=_df,
                sheet_style=sheet_style,
                link_cols='Back to main',
                better_stats_idx=find_better_values(_df, baseline_stats_df, 'Surface'),
                index_col='Surface',
            )
    print(f'Saving results to: {output_path}')


if __name__ == '__main__':
    main()
