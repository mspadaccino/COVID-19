import dash
import dash_core_components as dcc
import dash_html_components as html
import pandas as pd
import numpy as np
import plotly.graph_objs as go
from dash.dependencies import Input, Output
from src.data_downloader import get_dataframes, update_repos
import plotly.graph_objects as go
from fbprophet import Prophet
from controls import get_trend_controls, get_evo_controls

dest = '../data'

UPDATE_REPO = False
if UPDATE_REPO: update_repos(dest)

df_naz, reg, prov, df_reg, df_prov, df_world_confirmed, df_world_deaths, \
df_world_recovered, populations, ita_populations, df_comuni_sett = get_dataframes(dest)

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

app.layout = html.Div(children=[
    html.Div(
        className='row',  # Define the row element
        children=[
            html.Div(

                # className='eight columns div-for-charts bg-grey',
                children=[
                    html.H1('COVID19 Dashboard'),
                    html.P('''Monitoring SARS-COVID19 desease across Italy and the World'''),
                    # html.P('''Select one of the analysis from below.'''),
                    # Dividing the dashboard in tabs
                    dcc.Tabs(
                        id="main-tabs",
                        children=[  # Defining the layout of the first Tab
                            dcc.Tab(
                                label='Italy',
                                children=[
                                    dcc.Tabs(
                                        id="italy-tabs",
                                        children=[
                                            dcc.Tab(
                                                label='Data Evolution',
                                                children=get_evo_controls(df_reg),
                                            ),
                                            dcc.Tab(
                                                label='Forecast',
                                                children=get_trend_controls(df_reg)
                                            ),
                                            dcc.Tab(
                                                label='Daily updates',
                                                children=[]
                                            )
                                        ]
                                    )
                                ]
                            ),
                            dcc.Tab(
                                label='World',
                            )
                        ]
                    ),
                ]
            )  # Define the right element
        ])
])


@app.callback(Output(component_id='updatemode-output-container', component_property='children'),
              [Input(component_id='forecast_periods', component_property='value')])
def display_value(forecast_periods):
    return 'forecasting days: {}'.format(forecast_periods)


@app.callback(
    Output(component_id='evo-chart', component_property='figure'),
    [
        Input(component_id='regions', component_property='value'),
        Input(component_id='labels', component_property='value'),
        Input(component_id='log', component_property='value'),
    ]
)
def get_evo(regions, labels, log):
    if log == 'linear':
        log = False
    relative_dates = False
    cases_per_mln_people = False
    plot_bars = True
    aggregate = False
    show_trend = True

    mult = 1.
    trace = []
    for item in labels:
        if aggregate:
            if cases_per_mln_people:
                mult = 1e06 / ita_populations.loc[regions, 'Popolazione'].sum()
            temp = df_reg[regions[0]][item].copy()
            for region in regions[1:]:
                temp = temp.add(df_reg[region][item])
            if item == 'Rth':
                if (len(regions[1:])) > 0:
                    temp = temp / len(regions[1:])
            temp = pd.DataFrame(temp)
            if relative_dates: temp = temp.loc[~(temp[item] == 0)].reset_index(drop=True).iloc[:-1]
            temp['mov_avg_7d'] = temp.rolling(7).mean().shift(-0)
            if plot_bars:
                trace.append(go.Bar(x=temp.index, y=temp[item] * mult, name=item + '_' + '-'.join(regions)))
            else:
                trace.append(go.Scatter(x=temp.index, y=temp[item] * mult, name=item + '_' + '-'.join(regions)))
            if show_trend:
                trace.append(go.Scatter(x=pd.date_range(start=temp.index.min(), end=temp.index.max()),
                                        y=temp['mov_avg_7d'].values, name=item + '_wkly_trend', mode='lines',
                                        line=dict(color='black', width=.45, dash='dot')))
        else:
            for region in regions:
                if cases_per_mln_people:
                    mult = 1e06 / ita_populations.loc[region, 'Popolazione']
                df_reg[region].index = pd.to_datetime(df_reg[region].index)
                temp = df_reg[region]
                temp['mov_avg_7d'] = temp[item].rolling(7).mean().shift(-0)
                if relative_dates: temp = temp.loc[~(temp[item] == 0)].reset_index(drop=True).iloc[:-1]

                if plot_bars:
                    trace.append(go.Bar(x=temp.index, y=temp[item] * mult, name=item + '_' + region))
                else:
                    trace.append(go.Scatter(x=temp.index, y=temp[item] * mult, name=item + '_' + region))
                if show_trend:
                    trace.append(go.Scatter(x=pd.date_range(start=temp.index.min(), end=temp.index.max()),
                                            y=temp['mov_avg_7d'].values, name=item + '_wkly_trend', mode='lines',
                                            line=dict(color='black', width=.45, dash='dot')))

    traces = [trace]
    data = [val for sublist in traces for val in sublist]
    figure = {'data': data,
              'layout': go.Layout(
                  title={'text': 'test2', 'xanchor': 'left'},
                  showlegend=True, paper_bgcolor='rgba(0,0,0,0)', font=dict(color='lightgray'),
                  plot_bgcolor='rgba(0,0,0,0)',
                  yaxis_type="log" if log else None
              ),
              }
    return figure


@app.callback(
    Output(component_id='trend-chart', component_property='figure'),
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
    if smoothing == '': smoothing = 1
    if forecast_periods == '': forecast_periods = 1
    smoothing = int(smoothing)
    forecast_periods = int(forecast_periods)
    df = df_reg[region][label].rolling(smoothing).mean()
    y = label
    train_data = pd.DataFrame()
    train_data['ds'] = pd.to_datetime(df.index)
    train_data['y'] = np.log1p(df.reset_index(drop=True).values)
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
        go.Scatter(x=df.index, y=df['yhat_lower'], fill=None, mode='lines', line_color='lightgrey',
                   name='confidence_lev_down'),
        go.Scatter(x=df.index, y=df['yhat_upper'], fill='tonexty', mode='lines', line_color='lightgrey',
                   name='confidence_lev_up'),
        go.Scatter(x=df.index, y=df['y'], name='{}'.format(y), mode='lines+markers', marker=dict(size=5)),
        go.Scatter(x=df.loc[start_fit:end_fit].index, y=df['yhat'].loc[start_fit:end_fit], line_color='goldenrod',
                   mode='lines', name='model fit'),
        go.Scatter(x=df.loc[end_fit:].index, y=df['yhat'].loc[end_fit:], line_color='darkblue', mode='markers',
                   marker=dict(size=2), name='forecast')
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
