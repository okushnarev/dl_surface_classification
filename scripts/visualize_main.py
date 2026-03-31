import argparse
import json
import sys
from pathlib import Path
from typing import Any, Collection, Optional

import numpy as np
import pandas as pd
import plotly.express as px
import yaml
from pandas import DataFrame
from plotly.graph_objs import Figure
from plotly.subplots import make_subplots
from sklearn.metrics import accuracy_score

# Add project root to PATH
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

from src.utils.paths import ProjectPaths
from src.visualization.tools import bar_plot


def format_model_name(net, filter_type, ds_type):
    """Formats model metadata into a plot-friendly string (HTML allowed)"""
    return '<br>'.join([
        net.upper(),
        filter_type.replace('_', ' '),
        ds_type.replace('_', ' ')
    ])


def load_memory_baseline(dataset_name):
    """Loads the memory classifier baseline if available"""
    # Assuming this file sits in raw data
    mem_file = ProjectPaths.get_raw_data_dir() / 'cls_mem_res.csv'

    if not mem_file.exists():
        print(f'Warning: Memory baseline file not found at {mem_file}')
        return None

    df = pd.read_csv(mem_file)
    # Ensure columns match expectations
    if 'cls_memory' in df.columns:
        df['prediction'] = df['cls_memory']
    return df


def get_experiment_metadata(nets, config_name):
    """
    Parses YAML configs to get a mapping of {exp_name: (net, filter, ds_type)}
    This ensures we only plot experiments defined in the config.
    """
    metadata_map = {}

    for net in nets:
        cfg_path = ProjectPaths.get_experiment_config_path(net, config_name)
        if not cfg_path.exists():
            continue

        with open(cfg_path, 'r') as f:
            data = yaml.safe_load(f)

        defaults = data.get('defaults', {})
        experiments = data.get('experiments', [])

        for exp in experiments:
            name = exp.get('name')
            if not name: continue

            # Merge common args to find filter/type
            common = defaults.get('common', {}).copy()
            common.update(exp.get('common', {}))

            filter_type = common.get('filter', 'no_filter')
            ds_type = common.get('ds_type', 'type_1')

            metadata_map[name] = (net, filter_type, ds_type)

    return metadata_map


def radial_barplot(
        acc_summaries: dict[str, DataFrame],
        baseline_df: DataFrame,
        surfs: Collection[str],
        color_palette: dict[str, str]
) -> Optional[Figure]:
    """
    Creates the radial bar plot
    """
    print('Creating radial plots')

    # Prepare data
    # Combine memory baseline with models
    plot_data = {}

    # Process Baseline
    if baseline_df is not None:
        baseline_df['is_correct'] = baseline_df['surf'] == baseline_df['prediction']
        mem_acc = baseline_df.groupby(['surf', 'movedir'])['is_correct'].mean().reset_index(name='accuracy')
        plot_data['Mem'] = mem_acc

    # Process models
    plot_data.update(acc_summaries)

    if not plot_data:
        print('No data for radial plots.')
        return None

    # Setup Layout
    surfs = list(color_palette.keys())  # Assumes keys in config match 'surf' column values
    rows = len(plot_data)
    cols = len(surfs)

    fig = make_subplots(
        rows=rows,
        cols=cols,
        subplot_titles=surfs,
        row_titles=list(plot_data.keys()),
        specs=[[{'type': 'polar'} for _ in range(cols)] for _ in range(rows)]
    )

    # Add traces
    for row_idx, (name, df) in enumerate(plot_data.items()):
        for col_idx, surf in enumerate(surfs):

            color = color_palette.get(surf, '#000000')
            subset = df.query('surf == @surf')

            if subset.empty:
                continue

            trace = px.bar_polar(
                subset,
                r='accuracy',
                theta='movedir',
                hover_data=['surf'],
                direction='counterclockwise',
                color_discrete_sequence=[color]
            ).data[0]

            fig.add_trace(trace, row=row_idx + 1, col=col_idx + 1)

    # Styling
    ticks = np.arange(0, 360, 45)
    polar_layout = dict(
        radialaxis=dict(showticklabels=False, range=(0, 1)),
        angularaxis=dict(
            tickvals=ticks,
            ticktext=[str(x) for x in ticks],
            direction='counterclockwise',
            rotation=90,
        )
    )

    # Apply to all polar subplots
    layout_updates = {
        f'polar{i if i > 1 else ""}': polar_layout
        for i in range(1, (rows * cols) + 1)
    }

    fig.update_layout(
        template='plotly_white',
        height=250 * rows,
        **layout_updates
    )

    # Adjust annotations
    for annotation in fig.layout.annotations:
        if annotation.text in surfs:
            annotation.y += 0.02
        elif annotation.text in plot_data.keys():
            annotation.x = -0.04
            annotation.textangle = 270

    return fig


def mean_acc_barplot(
        model_accuracies: dict[str, DataFrame],
        baseline_df: DataFrame,
) -> Optional[Figure]:
    """
    Generates the Mean Accuracy comparison bar chart (Linear Motion).
    Compares Models against the Memory Baseline.
    """
    print('Creating Linear Motion Comparison Bar Plot')

    rows = []

    # Process Memory Baseline
    if baseline_df is not None:
        mem_acc = accuracy_score(baseline_df['surf'], baseline_df['prediction'])
        rows.append({'Classifier': 'Mem', 'accuracy': mem_acc, 'is_mem': True})

    # Process Models
    for name, acc in model_accuracies.items():
        rows.append({'Classifier': name, 'accuracy': acc, 'is_mem': False})

    if not rows:
        print('No results found')
        return None

    df_bar = pd.DataFrame(rows)
    # Accuracy stats
    min_acc = df_bar['accuracy'].min()

    # Plot
    fig = bar_plot(
        df_bar,
        threshold=8,
        x='Classifier',
        y='accuracy',
        text='accuracy',
        color='is_mem',
        color_discrete_map={True: '#EF553B', False: '#636EFA'},
        template='plotly_white',
        range_y=[min_acc - 0.02, 1.01],
        title='Linear Motion (Mean Accuracy)'
    )
    fig.update_traces(texttemplate='%{text:.2f}', textposition='outside')

    return fig


def grouped_stats_barplot(
        stats_df: DataFrame,
        acc_order: list[str],
        model_order: list[str],
        palette: dict[str, str],
        stat_name: str,
) -> Optional[Figure]:
    """
    Generates the Stats  bar chart (Recall, Precision, F1 Score).
    """
    print(f'Creating {stat_name.capitalize()} Bar Plot')

    # Min stat
    min_stat = stats_df[stat_name].min()

    # Plot
    fig = px.bar(
        stats_df,
        x='model',
        y=stat_name,
        color='surf',
        barmode='group',
        range_y=[min_stat - 0.02, 1],
        category_orders={'surf': acc_order, 'model': model_order},
        color_discrete_map=palette,
        text_auto='.4f',
        text=stat_name,
        template='plotly_white',
        title=f'Linear Motion ({stat_name})'
    )

    return fig


def trajectory_comparison_barplot(
        exp_metadata: dict[str, tuple[Any]],
        cache_dir: Path,
        top_n: int = -1) -> Optional[Figure]:
    """
    Generates comparison bar chart for Square vs Circle trajectories.
    Reads from results/main/square/ and results/main/circle/.
    """
    print('Creating bar plot for trajectory comparison: square and circle')

    trajectory_types = ['circle', 'square']
    rows = []

    for exp_name, (net, f_type, ds_type) in exp_metadata.items():
        pretty_name = format_model_name(net, f_type, ds_type)

        for traj in trajectory_types:
            # Locate result file: results/main/circle/<exp_name>.csv
            csv_path = cache_dir.parent / traj / f'{exp_name}.csv'

            if csv_path.exists():
                try:
                    df = pd.read_csv(csv_path)
                    acc = accuracy_score(df['surf'], df['prediction'])
                    rows.append({
                        'name':     pretty_name,
                        'ds_type':  traj,
                        'accuracy': acc
                    })
                except Exception as e:
                    print(f'Error reading {csv_path}: {e}')

    if not rows:
        print('No trajectory results found')
        return None

    df_acc = pd.DataFrame(rows)

    # Filter top n models
    if top_n > 0:
        mean_acc = df_acc.groupby('name')['accuracy'].mean().sort_values(ascending=False)
        top_names = mean_acc.head(top_n).index
        df_acc = df_acc[df_acc['name'].isin(top_names)]

    # Add memory baseline
    baseline = pd.DataFrame(dict(
        name=['Mem', 'Mem'],
        ds_type=['circle', 'square'],
        accuracy=[0.836631, 0.759980],
    ))
    df_acc = pd.concat([df_acc, baseline], ignore_index=True)

    # Accuracy stats
    min_acc = df_acc['accuracy'].min()

    # Sorting
    df_acc['mean_acc'] = df_acc.groupby('name')['accuracy'].transform('mean')
    df_acc = df_acc.sort_values('mean_acc', ascending=False)
    sorted_names = df_acc['name'].unique()

    fig = bar_plot(
        df_acc,
        threshold=8 * 2,
        x='name',
        y='accuracy',
        text='accuracy',
        color='ds_type',
        template='plotly_white',
        barmode='group',
        range_y=[min_acc - 0.05, 1.01],
        category_orders={'name': list(sorted_names)},
        title='Square and Circle Trajectories (Mean Accuracy)'
    )
    fig.update_traces(texttemplate='%{text:.2f}', textposition='outside')

    return fig


def get_stats_from_df(df: pd.DataFrame):
    df['is_correct'] = df['surf'] == df['prediction']
    _acc_by_dir = df.groupby(['surf', 'movedir'])['is_correct'].mean().reset_index(name='accuracy')
    _recall = dict(df.groupby('surf')['is_correct'].mean())
    _precision = dict(df.groupby('prediction')['is_correct'].mean())
    _mean_acc = df['is_correct'].mean()
    return _acc_by_dir, _mean_acc, _precision, _recall


def write_image(fig, path, name, raster_out: bool = False):
    output_path = path / f'{name}.html'
    fig.write_html(output_path, include_plotlyjs='cdn')
    print(f'Saved: {output_path}')
    if raster_out:
        output_path = path / f'{name}.png'
        fig.write_image(output_path, width=1100, height=600, scale=3)
        print(f'Saved: {output_path}')


def parse_args():
    parser = argparse.ArgumentParser(description='Visualization for Main Dataset')
    parser.add_argument('--nets', nargs='+', default=['rnn'], help='Networks to include in plots')
    parser.add_argument('--config_name', type=str, default='main', help='Experiment YAML filename')
    parser.add_argument('--subset', type=str, default='full', choices=['test', 'val', 'train', 'full'],
                        help='Data subset to evaluate on')
    parser.add_argument('--top', type=int, default=-1, help='Top N results to show')
    parser.add_argument('--raster_out', action='store_true', help='Output raster image along with html')
    parser.add_argument('--ckpt_type', default='best', choices=['best', 'last'], help='Checkpoint to load')
    return parser.parse_args()


def main():
    args = parse_args()
    dataset = 'main'  # This script is specific to Main

    # Load baseline df
    mem_df = load_memory_baseline(dataset)

    # Load Configs
    ds_cfg_path = ProjectPaths.get_dataset_config_path(dataset)
    with open(ds_cfg_path) as f:
        ds_config = json.load(f)

    colors = ds_config['metadata'].get('class_colors', {})

    # Identify experiments
    # dict[exp_name, (net, filter, type)]
    exp_meta = get_experiment_metadata(args.nets, args.config_name)

    # Load results
    results_dir = ProjectPaths.get_evaluation_dir(args.config_name, args.ckpt_type, args.subset)

    # Aggregate data
    accuracy_by_dir = {}
    mean_accuracy = {}
    recall_by_surf = {}
    precision_by_surf = {}
    surfs = set()
    for exp_name, (net, f_type, ds_type) in exp_meta.items():
        csv_path = results_dir / f'{exp_name}.csv'

        if csv_path.exists():
            df = pd.read_csv(csv_path)
            surfs |= set(df['surf'])

            _acc_by_dir, _mean_acc, _precision, _recall = get_stats_from_df(df)

            pretty_name = format_model_name(net, f_type, ds_type)
            accuracy_by_dir[pretty_name] = _acc_by_dir
            mean_accuracy[pretty_name] = _mean_acc
            recall_by_surf[pretty_name] = _recall
            precision_by_surf[pretty_name] = _precision
    surfs = list(surfs)

    # Get stats from baseline df
    mem_acc_by_dir, mem_mean_acc, mem_precision, mem_recall = get_stats_from_df(mem_df)
    recall_by_surf['Mem'] = mem_recall
    precision_by_surf['Mem'] = mem_precision

    # Sort keys by accuracy descending
    # Filter top n if args.top > 0
    model_order = sorted(mean_accuracy, key=mean_accuracy.get, reverse=True)[:args.top if args.top > 0 else None]

    accuracy_by_dir = {k: accuracy_by_dir[k] for k in model_order}
    mean_accuracy = {k: mean_accuracy[k] for k in model_order}

    # Process stats
    # Recall
    recall_df = pd.DataFrame(recall_by_surf).T.reset_index(names=['model'])
    recall_df = pd.melt(
        recall_df,
        id_vars=['model'],
        var_name='surf',
        value_name='recall',
    )
    recall_df['surf'] = recall_df['surf'].apply(str.lower)
    recall_order = list(recall_df.groupby('surf')['recall'].mean().sort_values(ascending=False).index)

    # Precision
    precision_df = pd.DataFrame(precision_by_surf).T.reset_index(names=['model'])
    precision_df = pd.melt(
        precision_df,
        id_vars=['model'],
        var_name='surf',
        value_name='precision',
    )
    precision_df['surf'] = precision_df['surf'].apply(str.lower)
    precision_order = list(precision_df.groupby('surf')['precision'].mean().sort_values(ascending=False).index)

    # Paths
    prefix = '_'.join(sorted(args.nets))
    figure_dir = ProjectPaths.get_figures_dir(args.config_name) / prefix
    figure_dir.mkdir(parents=True, exist_ok=True)

    # Radial plots
    radial_fig = radial_barplot(
        accuracy_by_dir,
        mem_df,
        surfs,
        colors,
    )
    if radial_fig:
        write_image(
            radial_fig,
            path=figure_dir,
            name='radial',
            raster_out=False,
        )

    # Bar plots
    bar_fig = mean_acc_barplot(
        mean_accuracy,
        mem_df,
    )
    if bar_fig:
        write_image(
            bar_fig,
            path=figure_dir,
            name='bar',
            raster_out=args.raster_out,
        )

    # Recall Plots
    recall_barplot_fig = grouped_stats_barplot(
        recall_df,
        recall_order,
        model_order,
        colors,
        'recall'
    )

    if recall_barplot_fig:
        write_image(
            recall_barplot_fig,
            path=figure_dir,
            name='recall',
            raster_out=args.raster_out,
        )

    # Precision Plots
    precision_barplot_fig = grouped_stats_barplot(
        precision_df,
        precision_order,
        model_order,
        colors,
        'precision'
    )

    if precision_barplot_fig:
        write_image(
            precision_barplot_fig,
            path=figure_dir,
            name='precision',
            raster_out=args.raster_out,
        )

    # Square circle
    square_circle_fig = trajectory_comparison_barplot(
        exp_meta,
        results_dir,
        top_n=args.top
    )
    if square_circle_fig:
        write_image(
            square_circle_fig,
            path=figure_dir,
            name='bar_square_circle',
            raster_out=args.raster_out,
        )


if __name__ == '__main__':
    main()
