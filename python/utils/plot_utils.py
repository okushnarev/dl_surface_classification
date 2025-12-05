from plotly import express as px

from python.utils.eval_utils import Model


def prep_name_plotly(model: Model):
    net_type = model.net_type.upper()
    filter_type = model.filter_type.replace('_', ' ')
    dataset = model.dataset.replace('_', ' ')
    return '<br>'.join([net_type, filter_type, dataset])


def bar_plot(df, threshold, scale=1, **kwargs):
    if len(df) > threshold * scale:
        df = df.sort_index(ascending=False)
        kwargs['x'], kwargs['y'] = kwargs['y'], kwargs['x']
        if 'range_y' in kwargs:
            kwargs['range_x'] = kwargs['range_y']
            del kwargs['range_y']

    return px.bar(df, **kwargs)
