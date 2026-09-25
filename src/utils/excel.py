from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any, Iterable, Self

import pandas as pd
from sklearn.metrics import classification_report

from src.data.manipulation import align_dfs, match_original_indices
from src.utils.paths import ProjectPaths


class Style:
    def __init__(self, base_style: dict[str, Any] = None):
        if base_style is None:
            base_style = {
                'font_name':  'Calibri',
                'font_size':  15,
                'valign':     'vcenter',
                'align':      'left',
                'num_format': '0.0000',
            }

        self.style = base_style

    def __add__(self, other: Self | dict[str, Any]) -> Self:
        if not isinstance(other, dict):
            other = other.style
        return Style(self.style | other)

    def __getitem__(self, key: str) -> Any:
        return self.style[key]

    def __copy__(self) -> Self:
        return Style(self.style.copy())

    def copy(self) -> Self:
        return self.__copy__()

    def set(self, **kwargs: Any):
        for key, value in kwargs.items():
            self.style[key] = value


@dataclass
class SheetStyles:
    base: Style
    header: Style = None
    link: Style = None
    better_stats: Style = None
    separator: Style = None

    def __post_init__(self):
        for field in fields(self):
            if field.name != 'base' and getattr(self, field.name) is None:
                setattr(self, field.name, self.base)


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
    if link_cols:
        if type(link_cols) is not list:
            link_cols = [link_cols]
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
        ds_row = meta['feature_set'] + (f"_{meta['filter_type']}" if n_filters > 1 else '')
        main_df_rows.append({
            'Net':      meta['net'],
            'Feature set':  ds_row,
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
    ProjectPaths.get_tables_dir(combined_config_name, args.ckpt_type).mkdir(parents=True, exist_ok=True)
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
    df = df.pivot(index='Net', columns='Feature set', values=value_col).reset_index()
    return df
