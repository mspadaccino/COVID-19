import dash
import dash_core_components as dcc
import dash_html_components as html
import pandas as pd
import numpy as np
import plotly.graph_objs as go
from dash.dependencies import Input, Output
from src.data_downloader import DATA_REPOS, download_from_repo, get_dataframes
import plotly.graph_objects as go
from fbprophet import Prophet
from controls import get_trend_controls


dest='../data'

UPDATE_REPO = False
if UPDATE_REPO:
    print('updating datasets from repos...')
    print('downloading Italian data')
    download_from_repo(DATA_REPOS['italy']['url'], filenames=DATA_REPOS['italy']['streams'], dest=dest)
    print('downloading world data')
    download_from_repo(DATA_REPOS['world']['url'], filenames=DATA_REPOS['world']['streams'], dest=dest)

df_naz, reg, prov, df_reg, df_prov, df_world_confirmed, df_world_deaths, \
df_world_recovered, populations, ita_populations, df_comuni_sett = get_dataframes(dest)



external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

app.layout = html.Div(children=[
                                html.Div(
                                    className='row',  # Define the row element
                                    children=[
                                        html.Div(
                                            className='four columns div-user-controls',
                                            children=[
                                                html.H1('COVID19 Dashboard'),
                                                html.P('''Monitoring SARS-COVID19 desease across Italy and the World'''),
                                                html.P('''Select one of the analysis from below.'''),
                                                # Dividing the dashboard in tabs
                                                dcc.Tabs(
                                                    id="main-tabs",
                                                    children=[# Defining the layout of the first Tab
                                                        dcc.Tab(
                                                            label='Italy',
                                                            children=[
                                                                dcc.Tabs(
                                                                    id="italy-tabs",
                                                                    children=[
                                                                        dcc.Tab(
                                                                            label='trend',
                                                                            children=get_trend_controls(df_reg)
                                                                        )
                                                                    ]
                                                                )
                                                            ]
                                                        ),
                                                        dcc.Tab(
                                                            label='World',
                                                        )
                                                    ]
                                                )
                                            ]
                                        ),  # Define the left element
                                        html.Div(
                                            className='eight columns div-for-charts bg-grey',
                                            children=[
                                                dcc.Graph(
                                                    id='prophet-chart',
                                                ),
                                            ]
                                        )  # Define the right element
                                    ])
                                ])

# app.layout = html.Div([
#     dcc.Dropdown(
#         id='region',
#         options=[{'label': value, 'value': value} for value in df_reg.keys()],
#         value='Italy',
#         clearable=False,
#         multi=False
#     ),
#     dcc.Dropdown(
#         id='label',
#         options=[{'label': value, 'value': value} for value in df_reg['Italy'].columns],
#         value='nuovi_positivi',
#         clearable=False,
#         multi=False
#     ),
#     dcc.DatePickerSingle(
#         id='start_fit',
#         min_date_allowed=pd.to_datetime(df_naz.index[0]),
#         max_date_allowed=pd.to_datetime(df_naz.index[-1]),
#         date=pd.to_datetime(df_naz.index[0])
#     ),
#     dcc.DatePickerSingle(
#         id='end_fit',
#         min_date_allowed=pd.to_datetime(df_naz.index[0]),
#         max_date_allowed=pd.to_datetime(df_naz.index[-1]),
#         date=pd.to_datetime(df_naz.index[-1])
#     ),
#     dcc.Input(
#         id='forecast_periods',
#         value=90,
#         # type= 'int',
#         placeholder='forecasting period',
#     ),
#     dcc.Input(
#         id='smoothing',
#         value=1,
#         # type='int',
#         placeholder='smoothing days',
#     ),
#     dcc.Graph(
#         id='prophet-chart',
#     ),
# ])


@app.callback(
    Output(component_id='prophet-chart', component_property='figure'),
    [
        Input(component_id='region', component_property='value'),
        Input(component_id='label', component_property='value'),
        Input(component_id='start_fit', component_property='date'),
        Input(component_id='end_fit', component_property='date'),
        Input(component_id='forecast_periods', component_property='value'),
        Input(component_id='smoothing', component_property='value')
    ]
)
def get_forecast(region, label, start_fit, end_fit, forecast_periods, smoothing):
    if smoothing == '': smoothing=1
    if forecast_periods == '': forecast_periods=1
    smoothing = int(smoothing)
    forecast_periods = int(forecast_periods)
    df = df_reg[region][label].rolling(smoothing).mean()
    y = label
    train_data = pd.DataFrame()
    train_data['ds']=pd.to_datetime(df.index)
    train_data['y']=np.log1p(df.reset_index(drop=True).values)
    train_data['floor'] = 0.
    m = Prophet(growth='linear', daily_seasonality=True, weekly_seasonality=True, yearly_seasonality=False)
    m.fit(train_data.set_index('ds').loc[start_fit:end_fit].reset_index())
    future = m.make_future_dataframe(periods=forecast_periods)
    future['floor'] = train_data['floor']
    forecast = m.predict(future)
    forecast['yhat'] = np.expm1(forecast['yhat'])
    forecast['yhat_lower'] = np.expm1(forecast['yhat_lower'])
    forecast['yhat_upper'] = np.expm1(forecast['yhat_upper'])
    train_data['y'] = np.expm1(train_data['y'])
    df = pd.merge(left=train_data, right=forecast, on='ds', how='outer').set_index('ds')
    df.index = pd.to_datetime(df.index)
    traces = [[
        go.Scatter(x=df.index, y=df['yhat_lower'], fill=None, mode='lines', line_color='lightgrey',name='confidence_lev_down'),
        go.Scatter(x=df.index, y=df['yhat_upper'], fill='tonexty', mode='lines', line_color='lightgrey',name='confidence_lev_up'),
        go.Scatter(x=df.index, y=df['y'], name='{}'.format(y), mode='lines+markers', marker=dict(size=5)),
        go.Scatter(x=df.loc[start_fit:end_fit].index, y=df['yhat'].loc[start_fit:end_fit], line_color='goldenrod',mode='lines', name='model fit'),
        go.Scatter(x=df.loc[end_fit:].index, y=df['yhat'].loc[end_fit:], line_color='darkblue', mode='markers',marker=dict(size=2), name='forecast')
    ]]
    data = [val for sublist in traces for val in sublist]
    figure = {'data': data,
              'layout': go.Layout(
                  title={'text': label.replace('_', ' ') + ' for ' + region, 'xanchor': 'left'}
              ),
    }

    return figure


if __name__ == '__main__':
    app.run_server(debug=True)