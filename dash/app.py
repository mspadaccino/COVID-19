import dash
import dash_core_components as dcc
import dash_html_components as html
import pandas as pd
import numpy as np
import plotly.graph_objs as go
from dash.dependencies import Input, Output
from src.data_downloader import DATA_REPOS, download_from_repo
from src.tools import add_extra_features, calculate_Rth
import os

app = dash.Dash()
server = app.server

dest='../data'
# print('updating datasets from repos...')
# print('downloading Italian data')
# download_from_repo(DATA_REPOS['italy']['url'], filenames=DATA_REPOS['italy']['streams'], dest=dest)
# print('downloading world data')
# download_from_repo(DATA_REPOS['world']['url'], filenames=DATA_REPOS['world']['streams'], dest=dest)


############################ loading datasets ####################################################
df_naz = pd.read_csv(os.path.join(dest,'dpc-covid19-ita-andamento-nazionale.csv')).drop('stato',1)
print('last available date for Italy data',df_naz['data'].iloc[-1])
reg = pd.read_csv(os.path.join(dest,'dpc-covid19-ita-regioni.csv'))
prov = pd.read_csv(os.path.join(dest,'dpc-covid19-ita-province.csv')).drop('stato',1)
df_naz.index = pd.to_datetime(df_naz.index)
reg['data'] = pd.to_datetime(reg['data'])
prov['data'] = pd.to_datetime(prov['data'])
df_comuni_sett = pd.read_csv(os.path.join(dest, 'df_comuni_sett.csv'))
df_world_confirmed = pd.read_csv(os.path.join(dest,'time_series_covid19_confirmed_global.csv'))
df_world_deaths = pd.read_csv(os.path.join(dest,'time_series_covid19_deaths_global.csv'))
df_world_recovered = pd.read_csv(os.path.join(dest,'time_series_covid19_recovered_global.csv'))
populations = pd.read_csv(os.path.join(dest,'API_SP.POP.TOTL_DS2_en_csv_v2.csv'), skiprows=4, engine='python').set_index('Country Name')['2018']
ita_populations = pd.read_csv(os.path.join(dest,'popitaregions.csv'))
df_world_confirmed['pop'] = df_world_confirmed['Country/Region'].map(populations)
df_world_deaths['pop'] = df_world_deaths['Country/Region'].map(populations)
df_world_recovered['pop'] = df_world_recovered['Country/Region'].map(populations)
print('last available date for World data',df_world_confirmed.columns[-2])
df_naz = add_extra_features(df_naz)
regions = reg.groupby('denominazione_regione')
df_reg = {}
df_reg['Italy'] = df_naz
for item in regions.groups:
    df_reg[item] = add_extra_features(regions.get_group(item)).replace((np.inf, np.nan), 0)
for data in df_reg.keys():
    df_reg[data]['Rth'] = calculate_Rth(df_reg[data],npt_rth=7)
provinces = prov.groupby('sigla_provincia')
df_prov = pd.DataFrame()
for item in provinces.groups:
    df_prov = pd.concat((df_prov,add_extra_features(provinces.get_group(item)).replace((np.inf, np.nan), 0)),0)
# fixing country and province different names from different datasets
pop_replace = [('US', 'United States'), ('Korea, South', 'Korea, Rep.'),
               ('Venezuela', 'Venezuela, RB'), ('Bahamas', 'Bahamas, The'),
               ('Iran', 'Iran, Islamic Rep.'), ('Russia', 'Russian Federation'),
               ('Egypt', 'Egypt, Arab Rep.'), ('Syria', 'Syrian Arab Republic'),
               ('Slovakia', 'Slovak Republic'), ('Czechia', 'Czech Republic'),
               ('Congo (Brazzaville)', 'Congo, Rep.'),
               ('Congo (Kinshasa)', 'Congo, Dem. Rep.'), ('Kyrgyzstan', 'Kyrgyz Republic'),
               ('Laos', 'Lao PDR'), ('Brunei', 'Brunei Darussalam'),
               ('Gambia', 'Gambia, The')]
for item in pop_replace:
    try:
        populations.loc[item[0]] = populations.loc[item[1]]
        del populations[item[1]]
    except Exception as e:
        print(e)

pops = ita_populations.loc[
           ita_populations['Regione'] == 'Trentino-Alto Adige', ['Popolazione', 'Superficie sqkm', 'ab/sqkm',
                                                                 'Numero_comuni', 'Numero_province']].values / 2
newdf = pd.DataFrame(index=['P.A. Trento', 'P.A. Bolzano', 'Italy'],
                     columns=ita_populations.set_index('Regione').columns)
newdf.loc['P.A. Trento'] = pops[0]
newdf.loc['P.A. Bolzano'] = pops[0]
newdf.loc['Italy'] = [populations.loc['Italy'], 0., 0., 0., 0.]
newdf.reset_index(inplace=True)
newdf.rename(columns={'index': 'Regione'}, inplace=True)
ita_populations = pd.concat((ita_populations, newdf)).set_index('Regione')
##################################################################################################

app.layout = html.Div([
    # Setting the main title of the Dashboard
    html.H1("COVID19 Monitor", style={"textAlign": "center"}),
    # Dividing the dashboard in tabs
    dcc.Tabs(id="tabs", children=[
        # Defining the layout of the first Tab
        dcc.Tab(label='National analysis', children=[
            html.Div([
                html.H1("Test1",
                        style={'textAlign': 'center'}),
                # Adding the first dropdown menu and the subsequent time-series graph
                dcc.Dropdown(id='my-dropdown',
                             options=[{'label': label, 'value': label} for label in df_naz.columns],
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

if __name__ == '__main__':
    app.run_server(debug=True)