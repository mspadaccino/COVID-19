import dash_core_components as dcc
from utils import get_options
import pandas as pd

def get_trend_controls(df_reg):
    controls = [
                dcc.Dropdown(
                    id='region',
                    options=get_options(df_reg.keys()),
                    # [{'label': value, 'value': value} for value in df_reg.keys()],
                    value='Italy',
                    clearable=False,
                    multi=False
                ),
                dcc.Dropdown(
                    id='label',
                    options=[{'label': value, 'value': value}
                             for value
                             in df_reg['Italy'].columns],
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