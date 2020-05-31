import dash_core_components as dcc
import pandas as pd
import yaml

with open("columns_names.yaml", 'r') as stream:
    out = yaml.load(stream)
    orig_data_columns = out['LABELS']['orig_data_columns']
    trend_labels = out['LABELS']['trend_labels']

delta_data_columns = ['daily_' + item for item in orig_data_columns]
percent_delta_data_columns = ['%daily_' + item for item in orig_data_columns]


def get_options(labels):
    dict_list = []
    for i in labels:
        dict_list.append({'label': i, 'value': i})
    return dict_list


def get_trend_controls(df_reg):
    controls = [
        dcc.Dropdown(
            id='region',
            options=get_options(df_reg.keys()),
            value='Italy',
            clearable=False,
            multi=False
        ),
        dcc.Dropdown(
            id='label',
            options=get_options(trend_labels),
            value='nuovi_positivi',
            clearable=False,
            multi=False
        ),
        dcc.DatePickerSingle(
            id='start_fit',
            min_date_allowed=pd.to_datetime(
                df_reg['Italy'].index[0]),
            max_date_allowed=pd.to_datetime(
                df_reg['Italy'].index[-1]),
            date=pd.to_datetime(df_reg['Italy'].index[0])
        ),
        dcc.DatePickerSingle(
            id='end_fit',
            min_date_allowed=pd.to_datetime(df_reg['Italy'].index[0]),
            max_date_allowed=pd.to_datetime(df_reg['Italy'].index[-1]),
            date=pd.to_datetime(df_reg['Italy'].index[-1])
        ),
        dcc.Input(
            id='forecast_periods',
            value=90,
            # type= 'int',
            placeholder='forecasting period',
        ),
        dcc.Input(
            id='smoothing',
            value=1,
            # type='int',
            placeholder='smoothing days',
        )
    ]
    return controls
