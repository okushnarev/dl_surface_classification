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


def parse_args():
    parser = argparse.ArgumentParser(description='Visualization for Main Dataset')
    parser.add_argument('--nets', nargs='+', default=['rnn'], help='Networks to include in plots')
    parser.add_argument('--config_name', type=str, default='main', help='Experiment YAML filename')
    parser.add_argument('--subset', type=str, default='full', choices=['test', 'val', 'train', 'full'],
                        help='Data subset to evaluate on')
    parser.add_argument('--top', type=int, default=-1, help='Top N results to show')
    return parser.parse_args()


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
        df['predictions'] = df['cls_memory']
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


def create_radial_plot(
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
        baseline_df['is_correct'] = baseline_df['surf'] == baseline_df['predictions']
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
        horizontal_spacing=0.05,
        vertical_spacing=min(0.12, 1 / (rows - 1) if rows > 1 else 1),
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


def create_bar_plot(
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
        mem_acc = accuracy_score(baseline_df['surf'], baseline_df['predictions'])
        rows.append({'Classifier': 'Mem', 'accuracy': mem_acc, 'is_mem': True})

    # Process Models
    for name, acc in model_accuracies.items():
        rows.append({'Classifier': name, 'accuracy': acc, 'is_mem': False})

    if not rows:
        print('No results found')
        return None

    df_bar = pd.DataFrame(rows)

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
        range_y=[0, 1.01],
        title='Linear Motion (Mean Accuracy)'
    )
    fig.update_traces(texttemplate='%{text:.2f}', textposition='outside')

    return fig



def create_trajectory_comparison(
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

    # Sorting
    df_acc['mean_acc'] = df_acc.groupby('name')['accuracy'].transform('mean')
    df_acc = df_acc.sort_values('mean_acc', ascending=False)
    sorted_names = df_acc['name'].unique()

    fig = bar_plot(
        df_acc,
        threshold=8,
        x='name',
        y='accuracy',
        text='accuracy',
        color='ds_type',
        template='plotly_white',
        barmode='group',
        range_y=[0, 1.01],
        category_orders={'name': list(sorted_names)},
        title='Square and Circle Trajectories (Mean Accuracy)'
    )
    fig.update_traces(texttemplate='%{text:.2f}', textposition='outside')

    return fig



def main():
    args = parse_args()
    dataset = 'main'  # This script is specific to Main

    # Load Configs
    ds_cfg_path = ProjectPaths.get_dataset_config_path(dataset)
    with open(ds_cfg_path) as f:
        ds_config = json.load(f)

    colors = ds_config['metadata'].get('class_colors', {})

    # Identify experiments
    # dict[exp_name, (net, filter, type)]
    exp_meta = get_experiment_metadata(args.nets, args.config_name)

    # Load results
    results_dir = ProjectPaths.get_evaluation_dir(args.config_name, args.subset)

    # Aggregate data
    accuracy_by_dir = {}
    mean_accuracy = {}
    surfs = set()
    for exp_name, (net, f_type, ds_type) in exp_meta.items():
        csv_path = results_dir / f'{exp_name}.csv'

        if csv_path.exists():
            df = pd.read_csv(csv_path)
            surfs |= set(df['surf'])

            df['is_correct'] = df['surf'] == df['prediction']
            _acc_by_dir = df.groupby(['surf', 'movedir'])['is_correct'].mean().reset_index(name='accuracy')
            _mean_acc = df['is_correct'].mean()

            pretty_name = format_model_name(net, f_type, ds_type)
            accuracy_by_dir[pretty_name] = _acc_by_dir
            mean_accuracy[pretty_name] = _mean_acc
    surfs = list(surfs)

    # Filter top n
    if args.top > 0:
        # Sort keys by accuracy descending
        sorted_keys = sorted(mean_accuracy, key=mean_accuracy.get, reverse=True)[:args.top]

        accuracy_by_dir = {k: accuracy_by_dir[k] for k in sorted_keys}
        mean_accuracy = {k: mean_accuracy[k] for k in sorted_keys}

    # Paths
    prefix = '_'.join(sorted(args.nets))
    figure_dir = ProjectPaths.get_figures_dir(args.config_name) / prefix
    figure_dir.mkdir(parents=True, exist_ok=True)

    # Radial plots
    mem_df = load_memory_baseline(dataset)
    radial_fig = create_radial_plot(
        accuracy_by_dir,
        mem_df,
        surfs,
        colors,
    )
    if radial_fig:
        output_path = figure_dir / f'radial.html'
        radial_fig.write_html(output_path, include_plotlyjs='cdn')
        print(f'Saved: {output_path}')

    # Bar plots
    bar_fig = create_bar_plot(
        mean_accuracy,
        mem_df,
    )
    if bar_fig:
        output_path = figure_dir / f'bar.html'
        bar_fig.write_html(output_path, include_plotlyjs='cdn')
        print(f'Saved: {output_path}')

    # Square circle
    square_circle_fig = create_trajectory_comparison(
        exp_meta,
        results_dir,
        top_n=args.top
    )
    output_path = figure_dir / f'bar_square_circle.html'
    if square_circle_fig:
        square_circle_fig.write_html(output_path, include_plotlyjs='cdn')
        print(f'Saved: {output_path}')


if __name__ == '__main__':
    main()
