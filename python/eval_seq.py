import argparse
import json
import pickle
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import torch
import yaml
from plotly.subplots import make_subplots
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import DataLoader

from python.utils.dataset import SequentialTabularDataset, create_sequences


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--nets', '-n', nargs='*', default='rnn', help='Nets to test')
    parser.add_argument('--top', default=-1, type=int, help='Top N results to show for each net. Use -1 to show all')
    parser.add_argument('--seq_len', default=10, type=int, help='Amount of consecutive data points to use')
    parser.add_argument('--ds', default='full', choices=['full', 'test'], help='Dataset to use')
    parser.add_argument('--ckpt_type', default='best', choices=['best', 'last'], help='Checkpoint type to load models')
    return parser.parse_args()


@dataclass
class Model:
    net_type: str  # e.g., RNN, CNN
    filter_type: str  # e.g., kalman, no_filter
    dataset: str  # e.g., type_1, type_2
    model: torch.nn.Module = field(repr=False)
    features: list[str] = field(repr=False)

    @property
    def name(self):
        net_type = self.net_type.upper()
        filter_type = self.filter_type.replace('_', ' ')
        dataset = self.dataset.replace('_', ' ')
        return f'{net_type} {filter_type} {dataset}'


def top_sorted_dict(d: dict[str, float], top_n: int, nets: list[str]) -> dict[str, float]:
    if top_n < 0:
        return d

    cnt = dict()
    new_dict = dict()
    nets_names_ordered = sorted(nets, key=len, reverse=True)
    for name, acc in d.items():
        # find net type
        net_type = ''
        for n in nets_names_ordered:
            if n in name:
                net_type = n
                break

        # leave top elements only
        if net_type not in cnt:
            cnt[net_type] = 0
        if cnt[net_type] < top_n:
            new_dict[name] = acc
        cnt[net_type] += 1
    return new_dict


def prep_name_plotly(model: Model):
    net_type = model.net_type.upper()
    filter_type = model.filter_type.replace('_', ' ')
    dataset = model.dataset.replace('_', ' ')
    return '<br>'.join([net_type, filter_type, dataset])


def run_inference(
        model_name: str,
        model_wrapper: Model,
        df: pd.DataFrame,
        group_cols: str | list[str],
        target_col: str,
        sequence_len: int,
        info_cols: list[str],
        label_encoder: LabelEncoder,
        device: str,
        batch_size: int,
        exps_to_unscale: list[str],
        unscale_cols: list[str]
) -> pd.DataFrame:
    """
    Runs inference for a specific model on a dataframe.
    Handles scaling, sequence creation, batched inference, and inverse transformation.
    """
    print(f'Running {model_name}')

    # Scaler
    ckpt_path = Path('data/runs') / model_name
    scaler_path = ckpt_path / 'scaler.pkl'
    with open(scaler_path, 'rb') as f:
        scaler = pickle.load(f)

    # Features
    feature_cols = model_wrapper.features

    # DataFrame prep
    df_copy = df.copy()
    df_copy[feature_cols] = scaler.transform(df_copy[feature_cols])

    # Create sequences
    X, y, info_arr = create_sequences(df_copy, group_cols, feature_cols, target_col, sequence_len, info_cols)

    info = pd.DataFrame(data=info_arr, columns=info_cols)
    info['surf'] = info['surf'].astype(np.uint8)
    info['surf'] = label_encoder.inverse_transform(info['surf'])

    # Create loader
    eval_dataset = SequentialTabularDataset(X, y)
    eval_loader = DataLoader(eval_dataset, batch_size=batch_size, shuffle=False)

    # Run tests
    model = model_wrapper.model
    predictions = np.array([], dtype=np.uint8)
    model.eval()
    with torch.no_grad():
        for sequences, labels in eval_loader:
            sequences = sequences.to(device, non_blocking=True)
            outputs = model(sequences)
            _, predicted = torch.max(outputs.data, 1)

            predictions = np.hstack([predictions, predicted.cpu().numpy()])

    info['predictions'] = label_encoder.inverse_transform(predictions)

    # Unscale certain columns if needed
    if model_name in exps_to_unscale:
        # Create placeholder matrix with the shape that Scaler expects
        dummy_data = np.zeros((len(info), len(feature_cols)))
        # Copy data_to_unscale to its corresponding position in the placeholder matrix
        for _col in unscale_cols:
            dummy_data[:, feature_cols.index(_col)] = info[_col]
        dummy_data = scaler.inverse_transform(dummy_data)
        # Take out only chosen unscaled data from the placeholder matrix
        for _col in unscale_cols:
            info[_col] = dummy_data[:, feature_cols.index(_col)]

    return info


def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    args = parse_args()

    # Params
    sequence_len = args.seq_len
    target_col = 'surf'

    # Load linear datasets
    ds_path = Path('data/input')
    data_path = Path('data/datasets')
    match args.ds:
        case 'full':
            df = pd.read_csv(ds_path / 'concat_noavg_kalman.csv')
        case 'test':
            df = pd.read_csv(data_path / 'test.csv')
        case _:
            raise ValueError(f'Unknown dataset: {args.ds}')

    figure_path = Path('figures') / 'evaluation' / args.ds
    figure_path.mkdir(parents=True, exist_ok=True)

    # Labels
    label_encoder = LabelEncoder()
    df[target_col] = label_encoder.fit_transform(df[target_col])
    num_classes = len(label_encoder.classes_)

    # Load memory cls results
    df_mem_cls = pd.read_csv(ds_path / 'cls_mem_res.csv')
    df_mem_cls['predictions'] = df_mem_cls['cls_memory']
    res_df_mem = df_mem_cls.groupby(['surf', 'movedir'])[['surf', 'predictions']].apply(
        lambda x: accuracy_score(x['surf'], x['predictions'])
    ).reset_index(name='accuracy')

    # Load models
    features_path = data_path / 'datasets_features.json'
    with open(features_path) as f:
        ds_features = json.load(f)

    # Ensure same nets' names order
    nets = sorted(args.nets, key=len, reverse=True)
    ckpt_type = args.ckpt_type
    models: dict[str, Model] = {}

    for net in nets:
        exp_cfg_path = Path(f'python/{net}/configs/experiments.yaml')
        with open(exp_cfg_path, 'r') as f:
            experiments = yaml.safe_load(f)

        for exp in experiments:
            print(f'Loading experiment {exp["name"]}')
            temp_name = exp['name'].replace(net, '').strip('_')
            cfg_path = Path(f'data/params/{net}_optim/best_params_{temp_name}.json')
            ckpt_path = Path('data/runs') / exp['name'] / f'{ckpt_type}.pt'

            filter_type = exp['common']['filter']
            ds_type = exp['common']['ds_type']
            features = ds_features[filter_type][ds_type]

            # Hyperparams
            input_dim = len(features)

            match net:
                case 'rnn':
                    from python.rnn.train_rnn import prep_rnn_cfg
                    from models.rnn import TabularRNN

                    prep_cfg = prep_rnn_cfg
                    prep_model = TabularRNN

                case 'cnn' | 'cnn_manual':
                    from python.cnn.train_cnn import prep_cnn_cfg
                    from python.cnn.train_cnn import CNNTrainWrapper

                    prep_cfg = prep_cnn_cfg
                    prep_model = CNNTrainWrapper

                case 'transfromer':
                    from python.transformer.train_transformer import prep_transformer_cfg
                    from models.transformer import Transformer

                    prep_cfg = prep_transformer_cfg
                    prep_model = Transformer

                case _:
                    raise ValueError(f'Unknown net {net}')

            cfg = prep_cfg(
                cfg_path,
                input_dim,
                num_classes,
                sequence_len
            )
            model = prep_model(**cfg['model']).to(device)

            # load model
            checkpoint = torch.load(ckpt_path, map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])
            _model = Model(
                net_type=net,
                filter_type=filter_type,
                dataset=ds_type,
                model=model,
                features=features,
            )
            models[exp['name']] = _model

    # Process data
    batch_size = 2 ** 15

    group_cols = ['surf', 'movedir', 'speedamp']
    info_cols = ['surf', 'movedir']

    unscale_cols = ['movedir']
    exps_to_unscale = [f'{n}_kalman_type_4' for n in nets]

    raw_results: dict[str, pd.DataFrame] = {}

    for model_name, model_wrapper in models.items():
        info = run_inference(
            model_name=model_name,
            model_wrapper=model_wrapper,
            df=df,
            group_cols=group_cols,
            target_col=target_col,
            sequence_len=sequence_len,
            info_cols=info_cols,
            label_encoder=label_encoder,
            device=device,
            batch_size=batch_size,
            exps_to_unscale=exps_to_unscale,
            unscale_cols=unscale_cols
        )
        raw_results[model_name] = info

    # Create accuracy summaries
    acc_summaries: dict[str, pd.DataFrame] = {}
    for model_name, raw_df in raw_results.items():
        _res_df = raw_df.groupby(['surf', 'movedir'])[['surf', 'predictions']].apply(
            lambda x: accuracy_score(x['surf'], x['predictions'])
        ).reset_index(name='accuracy')
        acc_summaries[model_name] = _res_df

    # Order names by accuracy
    names_by_accuracy: dict[str, float] = {
        _n: accuracy_score(_df['surf'], _df['predictions']) for _n, _df in raw_results.items()
    }
    names_by_accuracy = dict(sorted(names_by_accuracy.items(), key=lambda item: item[1], reverse=True))

    top_n = args.top
    names_by_accuracy = top_sorted_dict(names_by_accuracy, top_n, nets)

    # Radial plots
    acc_summaries_pretty_names = {
        prep_name_plotly(models[_n]): acc_summaries[_n] for _n in names_by_accuracy
    }
    res_dfs = {
        'Mem': res_df_mem,
        **acc_summaries_pretty_names,
    }
    surfs = ['gray', 'green', 'table', 'brown']
    my_pal = {
        'gray':  '#b6b6b6',
        'green': '#4fc54c',
        'table': '#9f6a4d',
        'brown': '#ad3024'
    }

    rows = len(res_dfs)
    cols = num_classes

    fig = make_subplots(
        cols=cols,
        rows=rows,
        subplot_titles=surfs,
        row_titles=list(res_dfs.keys()),
        horizontal_spacing=0.05,
        vertical_spacing=min(0.055, 1 / (rows - 1)),
        specs=[[{"type": "barpolar"} for _ in range(cols)] for _ in range(rows)],
    )

    for row_idx, (df_name, _df) in enumerate(res_dfs.items()):
        for col_idx, surf in enumerate(surfs):
            # Create a single figure for the specific df and surf
            figure = px.bar_polar(
                _df.query(f'surf == @surf'),
                r='accuracy',
                theta='movedir',
                hover_data='surf',
                direction='counterclockwise',
                color_discrete_sequence=[my_pal[surf]]
            )

            # Add the traces from this single figure to the correct subplot
            for trace in figure.data:
                fig.add_trace(trace, row=row_idx + 1, col=col_idx + 1)

    ticks = np.arange(0, 360, 45)
    polar = dict(
        radialaxis=dict(showticklabels=False, range=(0, 1)),
        angularaxis=dict(
            tickvals=ticks,
            ticktext=[f'{x}' for x in ticks],
            direction='counterclockwise',
            rotation=90,
        )
    )

    polar_keys = []
    for i in range(1, (rows * cols) + 1):
        key = f'polar{i}' if i > 1 else 'polar'
        polar_keys.append(key)

    fig.update_layout(
        {key: polar for key in polar_keys},
        template='plotly_white',
        height=250 * len(res_dfs),
    )

    # For HTML file
    for annotation in fig.layout.annotations:
        if annotation.text in surfs:
            annotation.y += 0.02

        elif annotation.text in res_dfs.keys():
            annotation.x = -0.03
            annotation.textangle = 0
    fig.write_html(figure_path / f'{"_".join(nets)}_radial.html', include_plotlyjs='cdn')

    # Bar plot mean accuracies
    df_bar_mean_acc = pd.DataFrame(
        {
            'Mem': accuracy_score(df_mem_cls.surf, df_mem_cls.predictions),
            **{prep_name_plotly(models[_n]): names_by_accuracy[_n] for _n in names_by_accuracy},
        },
        index=['accuracy']
    ).T.reset_index(names=['Classifier'])
    df_bar_mean_acc['is_mem'] = df_bar_mean_acc['Classifier'] == 'Mem'
    # TODO: For bar plots add auto detection of excessive amount of models to transfrom from horizontal to vertical. Do not forget to reverse accuracy data for vertical
    fig = px.bar(
        df_bar_mean_acc,
        x='Classifier',
        y='accuracy',
        text='accuracy',
        color='is_mem',
        color_discrete_map={True: '#EF553B', False: '#636EFA'},
        template='plotly_white',
        range_y=[0, 1.01],
        title='Linear Motion (Mean Accuracy)',

    )
    fig.update_traces(texttemplate='%{text:.2f}', textposition='outside')
    fig.write_html(figure_path / f'{"_".join(nets)}_bar.html', include_plotlyjs='cdn')

    # Process circle square datasets
    df_circle = pd.read_csv(ds_path / 'circle_kalman.csv')
    df_square = pd.read_csv(ds_path / 'square_kalman.csv')

    df_names = ('circle', 'square')
    df_list = (df_circle, df_square)

    dfs = dict(zip(df_names, df_list))

    raw_names = []
    raw_ds_types = []
    raw_dfs = []
    for df_name, _df in dfs.items():
        if 'speedamp' not in _df.columns:
            _df['speedamp'] = (_df['xsetspeed'].pow(2) + _df['ysetspeed'].pow(2)).pow(0.5).abs().round(3)

        _df['group'] = 1

        # Encode label
        _df[target_col] = label_encoder.transform(_df[target_col])

        for model_name, model_wrapper in models.items():
            info = run_inference(
                model_name=model_name,
                model_wrapper=model_wrapper,
                df=_df,
                group_cols='group',
                target_col=target_col,
                sequence_len=sequence_len,
                info_cols=info_cols,
                label_encoder=label_encoder,
                device=device,
                batch_size=batch_size,
                exps_to_unscale=exps_to_unscale,
                unscale_cols=unscale_cols
            )

            raw_names.append(model_name)
            raw_ds_types.append(df_name)
            raw_dfs.append(info)

    # Main acc dataframe
    accuracies = pd.DataFrame(dict(
        name=[prep_name_plotly(models[_n]) for _n in raw_names],
        ds_type=raw_ds_types,
        accuracy=[accuracy_score(_df['surf'], _df['predictions']) for _df in raw_dfs],
    ))

    # Add mem classifier's results
    temp_df = pd.DataFrame(dict(
        name=['Mem', 'Mem'],
        ds_type=['circle', 'square'],
        accuracy=[0.836631, 0.759980],
    ))
    accuracies = pd.concat([accuracies, temp_df])

    # Cut top n for every method
    _acc_dict = accuracies.groupby('name')['accuracy'].mean().sort_values(ascending=False).to_dict()
    _names = list(set([k.split('<br>')[0] if '<br>' in k else k for k in _acc_dict.keys()]))
    top_acc = top_sorted_dict(_acc_dict, top_n, _names)
    accuracies = accuracies.query('name in @top_acc')

    accuracies['mean_acc'] = accuracies.groupby('name')['accuracy'].transform('mean')

    # Sort the DataFrame by mean_acc in descending order
    accuracies_sorted = accuracies.sort_values('mean_acc', ascending=False)

    # Bar plot for square circle
    sorted_names = accuracies_sorted['name'].unique()
    fig = px.bar(
        accuracies,
        x='name',
        y='accuracy',
        text='accuracy',
        color='ds_type',
        template='plotly_white',
        barmode='group',
        range_y=[0, 1.01],
        category_orders={'name': list(sorted_names)},
        title='Square Circle'
    )
    fig.update_traces(texttemplate='%{text:.2f}', textposition='outside')

    fig.write_html(figure_path / f'{"_".join(nets)}_bar_square_circle.html', include_plotlyjs='cdn')


if __name__ == '__main__':
    main()
