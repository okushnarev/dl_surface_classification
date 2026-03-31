import argparse
import sys
from dataclasses import fields, replace
from pathlib import Path
from typing import Iterable

import pandas as pd
from sklearn.metrics import classification_report

# Add project root to PATH
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

from src.data.manipulation import align_dfs, get_results, match_original_indices
from src.utils.paths import ProjectPaths
from src.utils.excel import SheetStyles, Style


def parse_args():
    parser = argparse.ArgumentParser('Data aggregation script')
    parser.add_argument('--nets', nargs='+', default=['rnn'], help='Networks to include in plots')
    parser.add_argument('--configs', nargs='+', default=['main'], help='Experiment YAML filename')
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


def write_df_with_style(
        writer: pd.ExcelWriter,
        sheet_name: str,
        df: pd.DataFrame,
        sheet_style: SheetStyles,
        link_cols: list[str] | str = None,
        better_stats_idx: Iterable[tuple[int, int]] = [],
        index_col: str = None,
) -> None:
    """
    Write dataframe to Excel file with styling applied

    :param writer: ExcelWriter object to write to
    :param sheet_name: Name of the sheet to write
    :param df: DataFrame to write
    :param base_style: Dictionary of base style properties for formatting
    :param link_cols: Column name(s) to format as hyperlinks
    :param better_stats_idx: Optional dataframe coordinates to apply better_stats_style
    :param better_stats_style: Dictionary of better_stats style properties for formatting
    :param index_col: Name of the column to use as index for baseline comparison
    :returns: None
    """
    df.to_excel(writer, sheet_name=sheet_name, startrow=1, header=False, index=False)

    # Base variables
    workbook = writer.book
    worksheet = writer.sheets[sheet_name]

    # Formats
    formats = {}
    for field in fields(sheet_style):
        formats[field.name] = workbook.add_format(getattr(sheet_style, field.name).style)

    # Apply formats
    # Base format
    worksheet.set_column(0, len(df.columns) - 1, None, formats['base'])

    # Header format
    write_header(writer, sheet_name, df, formats['header'])

    # Better stats format
    for idx in better_stats_idx:
        worksheet.write(idx[0] + 1, idx[1], df.iloc[*idx], formats['better_stats'])

    # Link format
    if type(link_cols) is not list:
        link_cols = [link_cols]
    if link_cols:
        for link_col in link_cols:
            _col_idx = df.columns.get_loc(link_col)
            worksheet.set_column(_col_idx, _col_idx, len(link_col), formats['link'])

    # Add bottom border before macro/weighted average
    if index_col and (_p := 'macro avg') in (_l := df[index_col].tolist()):
        average_stats_row = _l.index(_p)
        for col_num, value in enumerate(df.iloc[average_stats_row]):
            worksheet.write(average_stats_row + 1, col_num, value, formats['separator'])

    # Cell size format
    worksheet.autofit()
    for row_num in range(len(df) + 1):
        worksheet.set_row(row_num, 25)


def write_header(writer, sheet_name, df, format):
    """
    Write header row to Excel sheet with specified format

    :param writer: ExcelWriter object
    :param sheet_name: Name of the sheet
    :param df: DataFrame whose columns are used as header
    :param format: Format object to apply to header cells
    :returns: None
    """
    worksheet = writer.sheets[sheet_name]
    for col_num, value in enumerate(df.columns.values):
        worksheet.write(0, col_num, value, format)


def parse_baseline(baseline_path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Parse baseline CSV file and prepare DataFrames for comparison
    Baseline DataFrame should be the output of sklearn's classification report

    :param baseline_path: Path to baseline CSV file
    :returns: Tuple of (baseline_accuracy_df, baseline_stats_df)
    """
    baseline_stats_df = None
    baseline_accuracy_value = None
    if baseline_path:
        baseline_accuracy_value = 1
        baseline_df = pd.read_csv(baseline_path, header=0, index_col=0)
        if (_n := 'support') in baseline_df.index:
            baseline_df = baseline_df.drop(index=[_n])
        baseline_df = baseline_df.T
        if (_n := 'accuracy') in baseline_df.index:
            baseline_accuracy_value = baseline_df.loc[_n].max()
            baseline_df = baseline_df.drop(index=[_n])
        baseline_stats_df = baseline_df.reset_index(names=['Surface'])
    return baseline_accuracy_value, baseline_stats_df


def extract_stats_from_results(
        raw_results: dict[str, tuple[dict[str, str], pd.DataFrame]],
        backlink_sheet_names: Iterable[str]
):
    """
    Extract accuracy and classification report statistics from raw results

    :param raw_results: Dictionary mapping experiment names to (metadata, DataFrame) tuples
    :param backlink_sheet_names: Name of the main sheets for hyperlink reference
    :returns: Tuple of (list of main sheet rows, dictionary of stats DataFrames)
    """
    # Find the number of filter types
    n_filters = len({v[0]['filter_type'] for v in raw_results.values()})

    main_df_rows = []
    stats_dfs = {}
    for exp_name, (meta, _df) in raw_results.items():
        _df['is_correct'] = _df['surf'] == _df['prediction']
        accuracy = _df['is_correct'].mean()
        ds_row = meta['ds_type'] + (f"_{meta['filter_type']}" if n_filters > 1 else '')
        main_df_rows.append({
            'Net':      meta['net'],
            'Dataset':  ds_row,
            'Accuracy': accuracy,
            'Stats':    f'=HYPERLINK("#{exp_name}!A1", "Link")'
        })

        stats = classification_report(_df['surf'], _df['prediction'], output_dict=True)
        stats = pd.DataFrame(stats)
        stats = stats.drop(index=['support'], columns=['accuracy'])
        stats = stats.T.reset_index(names=['Surface'])
        stats = stats.rename(columns=str.capitalize)
        stats['Back to main'] = None
        for idx, name in enumerate(backlink_sheet_names):
            stats.loc[idx, 'Back to main'] = f'=HYPERLINK("#{name}!A1", "{name}")'
        stats_dfs[exp_name] = stats
    return main_df_rows, stats_dfs


def prepare_paths(args):
    """
    Prepare baseline path and output path based on command line arguments

    :param args: Parsed command line arguments
    :returns: Tuple of (baseline_path, output_path)
    """
    baseline_path = None
    if args.baseline_df:
        if (_p := ProjectPaths.get_baseline_dfs_dir() / f'{args.baseline_df}.csv').exists():
            baseline_path = _p
            print(f'Loading baseline data from: {baseline_path}\n')
        else:
            print(f'Baseline path do not exist: {_p}')
            print('Skipping baseline analysis\n')

    combined_config_name = '_'.join(args.configs)
    baseline_name = f'_baseline_{baseline_path.stem}' if baseline_path else ''
    ProjectPaths.get_tables_dir(combined_config_name, 'best').mkdir(parents=True, exist_ok=True)
    output_name = args.output_name if args.output_name else f'results_{args.subset}{baseline_name}'
    output_path = ProjectPaths.get_tables_dir(combined_config_name, args.ckpt_type) / f'{output_name}.xlsx'
    return baseline_path, output_path


def find_better_values(
        df: pd.DataFrame,
        baseline_df: pd.DataFrame,
        index_col: str
) -> list[tuple[int, int]]:
    """
    Find indices where values in df are greater than corresponding baseline values

    :param df: DataFrame to compare
    :param baseline_df: DataFrame with baseline values
    :param index_col: Name of the column to use as index for alignment
    :returns: List of (row, column) integer indices where df exceeds baseline
    """
    _df, _baseline_df = align_dfs((df, baseline_df), index_col)
    compare_df = _df > _baseline_df
    better_stats_idx = compare_df.stack()[compare_df.stack()].index.tolist()
    better_stats_idx = match_original_indices(df, better_stats_idx, index_col)
    return better_stats_idx


def convert_to_wide_format(df: pd.DataFrame, value_col: str = 'Stats_Acc') -> pd.DataFrame:
    df = df.copy()
    df['Stats_Acc'] = df.apply(lambda row: row['Stats'].replace('Link', f"{row['Accuracy']:.4f}"), axis=1)
    df = df.pivot(index='Net', columns='Dataset', values=value_col).reset_index()
    return df


if __name__ == '__main__':
    main()
