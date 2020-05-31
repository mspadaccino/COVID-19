import dash
import dash_core_components as dcc
import dash_html_components as html
import pandas as pd
import numpy as np
import plotly.graph_objs as go
from dash.dependencies import Input, Output
from src.data_downloader import DATA_REPOS, download_from_repo, get_dataframes

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

app.layout = html.Div([
    # Setting the main title of the Dashboard
    html.H1("Facebook Data Analysis", style={"textAlign": "center"}),
    # Dividing the dashboard in tabs
    dcc.Tabs(id="tabs", children=[
        # Defining the layout of the first Tab
        dcc.Tab(label='Stock Prices', children=[
            html.Div([
                html.H1("Facebook Stocks High vs Lows",
                        style={'textAlign': 'center'}),
                # Adding the first dropdown menu and the subsequent time-series graph
                dcc.Dropdown(id='my-dropdown',
                             options=[{'label': 'Tesla', 'value': 'TSLA'},
                                      {'label': 'Apple','value': 'AAPL'},
                                      {'label': 'Facebook', 'value': 'FB'},
                                      {'label': 'Microsoft','value': 'MSFT'}],
                             multi=True,value=['FB'],
                             style={"display": "block", "margin-left": "auto",
                                    "margin-right": "auto", "width": "60%"}),
                dcc.Graph(id='highlow'),
                html.H1("Facebook Market Volume", style={'textAlign': 'center'}),
                # Adding the second dropdown menu and the subsequent time-series graph
                dcc.Dropdown(id='my-dropdown2',
                             options=[{'label': 'Tesla', 'value': 'TSLA'},
                                      {'label': 'Apple','value': 'AAPL'},
                                      {'label': 'Facebook', 'value': 'FB'},
                                      {'label': 'Microsoft','value': 'MSFT'}],
                             multi=True,value=['FB'],
                             style={"display": "block", "margin-left": "auto",
                                    "margin-right": "auto", "width": "60%"}),
                dcc.Graph(id='volume')
            ], className="container"),
        ]),
        # Defining the layout of the second tab
        dcc.Tab(label='Performance Metrics', children=[
            html.H1("Facebook Metrics Distributions",
                    style={"textAlign": "center"}),
            # Adding a dropdown menu and the subsequent histogram graph
            html.Div([
                      html.Div([dcc.Dropdown(id='feature-selected1',
                      options=[{'label': i.title(),
                                'value': i} for i in df_naz.columns.values[1:]],
                                 value="Type")],
                                 className="twelve columns",
                                 style={"display": "block", "margin-left": "auto",
                                        "margin-right": "auto", "width": "60%"}),
                    ], className="row",
                    style={"padding": 50, "width": "60%",
                           "margin-left": "auto", "margin-right": "auto"}),
                    dcc.Graph(id='my-graph2'),
        ])
    ])
])


@app.callback(Output('highlow', 'figure'),
              [Input('my-dropdown', 'value')])
def update_graph(selected_dropdown):
    dropdown = {"TSLA": "Tesla","AAPL": "Apple","FB": "Facebook","MSFT": "Microsoft",}
    trace1 = []
    trace2 = []
    for stock in selected_dropdown:
        trace1.append(
          go.Scatter(x=df[df["Stock"] == stock]["Date"],
                     y=df[df["Stock"] == stock]["High"],
                     mode='lines', opacity=0.7,
                     name=f'High {dropdown[stock]}',textposition='bottom center'))
        trace2.append(
          go.Scatter(x=df[df["Stock"] == stock]["Date"],
                     y=df[df["Stock"] == stock]["Low"],
                     mode='lines', opacity=0.6,
                     name=f'Low {dropdown[stock]}',textposition='bottom center'))
    traces = [trace1, trace2]
    data = [val for sublist in traces for val in sublist]
    figure = {'data': data,
              'layout': go.Layout(colorway=["#5E0DAC", '#FF4F00', '#375CB1',
                                            '#FF7400', '#FFF400', '#FF0056'],
            height=600,
            title=f"High and Low Prices for {', '.join(str(dropdown[i]) for i in selected_dropdown)} Over Time",
            xaxis={"title":"Date",
                   'rangeselector': {'buttons': list([{'count': 1, 'label': '1M',
                                                       'step': 'month',
                                                       'stepmode': 'backward'},
                                                      {'count': 6, 'label': '6M',
                                                       'step': 'month',
                                                       'stepmode': 'backward'},
                                                      {'step': 'all'}])},
                   'rangeslider': {'visible': True}, 'type': 'date'},
             yaxis={"title":"Price (USD)"})}
    return figure


@app.callback(Output('volume', 'figure'),
              [Input('my-dropdown2', 'value')])
def update_graph(selected_dropdown_value):
    dropdown = {"TSLA": "Tesla","AAPL": "Apple","FB": "Facebook","MSFT": "Microsoft",}
    trace1 = []
    for stock in selected_dropdown_value:
        trace1.append(
          go.Scatter(x=df[df["Stock"] == stock]["Date"],
                     y=df[df["Stock"] == stock]["Volume"],
                     mode='lines', opacity=0.7,
                     name=f'Volume {dropdown[stock]}', textposition='bottom center'))
    traces = [trace1]
    data = [val for sublist in traces for val in sublist]
    figure = {'data': data,
              'layout': go.Layout(colorway=["#5E0DAC", '#FF4F00', '#375CB1',
                                            '#FF7400', '#FFF400', '#FF0056'],
            height=600,
            title=f"Market Volume for {', '.join(str(dropdown[i]) for i in selected_dropdown_value)} Over Time",
            xaxis={"title":"Date",
                   'rangeselector': {'buttons': list([{'count': 1, 'label': '1M',
                                                       'step': 'month',
                                                       'stepmode': 'backward'},
                                                      {'count': 6, 'label': '6M',
                                                       'step': 'month',
                                                       'stepmode': 'backward'},
                                                      {'step': 'all'}])},
                   'rangeslider': {'visible': True}, 'type': 'date'},
             yaxis={"title":"Transactions Volume"})}
    return figure


@app.callback(
    dash.dependencies.Output('my-graph2', 'figure'),
    [dash.dependencies.Input('feature-selected1', 'value')])
def update_graph(selected_feature1):
    if selected_feature1 == None:
        selected_feature1 = 'Type'
        trace = go.Histogram(x= df2.Type,
                             marker=dict(color='rgb(0, 0, 100)'))
    else:
        trace = go.Histogram(x=df2[selected_feature1],
                         marker=dict(color='rgb(0, 0, 100)'))

    return {
        'data': [trace],
        'layout': go.Layout(title=f'Metrics considered: {selected_feature1.title()}',
                            colorway=["#EF963B", "#EF533B"], hovermode="closest",
                            xaxis={'title': "Distribution",
                                   'titlefont': {'color': 'black', 'size': 14},
                                   'tickfont': {'size': 14, 'color': 'black'}},
                            yaxis={'title': "Frequency",
                                   'titlefont': {'color': 'black', 'size': 14, },
                                   'tickfont': {'color': 'black'}})}


if __name__ == '__main__':
    app.run_server(debug=True)