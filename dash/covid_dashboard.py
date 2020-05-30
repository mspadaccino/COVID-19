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
    html.H1("Covid19 Data Monitor", style={"textAlign": "center"}),
    # Dividing the dashboard in tabs
    dcc.Tabs(id="tabs", children=[
        # Defining the layout of the first Tab
        dcc.Tab(label='Italy and regions', children=[
            html.Div([dcc.Graph(id='graph'),
                      dcc.Slider(
                          id='year-slider',
                          min=df['year'].min(),
                          max=df['year'].max(),
                          value=df['year'].min(),
                          marks={str(year): str(year) for year in df['year'].unique()},
                          step=None
                      )
            ], className="container"),
        ]),
        # Defining the layout of the second tab
        dcc.Tab(label='World', children=[
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




if __name__ == '__main__':
    app.run_server(debug=True)