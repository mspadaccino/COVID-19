import numpy as np
import pandas as pd
import stat
import os
import git
import shutil
from gitinfo import get_git_info
from src.tools import add_extra_features, calculate_Rth

def update_repos(dest):
    print('updating datasets from repos...')
    print('downloading Italian data')
    download_from_repo(DATA_REPOS['italy']['url'], filenames=DATA_REPOS['italy']['streams'], dest=dest)
    print('downloading world data')
    download_from_repo(DATA_REPOS['world']['url'], filenames=DATA_REPOS['world']['streams'], dest=dest)

def rmtree(top):
    for root, dirs, files in os.walk(top, topdown=False):
        for name in files:
            filename = os.path.join(root, name)
            os.chmod(filename, stat.S_IWUSR)
            os.remove(filename)
        for name in dirs:
            os.rmdir(os.path.join(root, name))
    os.rmdir(top)

DATA_REPOS = {
    "world": {
        "url": "https://github.com/CSSEGISandData/COVID-19",
        "streams": ["/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_deaths_global.csv",
                    "/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_confirmed_global.csv",
                    "/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_recovered_global.csv"]
    },
    "italy": {
        "url": 'https://github.com/pcm-dpc/COVID-19',
        "streams": ["/dati-andamento-nazionale/dpc-covid19-ita-andamento-nazionale.csv",
                    "/dati-regioni/dpc-covid19-ita-regioni.csv",
                    "/dati-province/dpc-covid19-ita-province.csv"]
    },
}



def download_from_repo(url, filenames, dest):
    # Create temporary dir
    t = os.path.join(dest,'temp')
    if os.path.exists(t):
        rmtree(t)
    # Clone into temporary dir
    repo = git.Repo.clone_from(url, t, branch='master', depth=1)

    # Copy desired file from temporary dir
    for filename in filenames:
        # print(t +'/'+ filename)
        shutil.move(t +'/'+ filename, os.path.join(dest, filename.split('/')[-1]))
        print('updated ', filename)
    # get repo info (last commit)
    try:
        info = get_git_info(t)
        print('last commit ', info['author_date'])
    except Exception as e:
        print('could not retrieve repo infos, ',e)
    # Remove temporary dir
    rmtree(t)
    return


def reformat(path):
    raw_data = pd.read_csv(path)
    lines = []
    dates = [np.datetime64('20{2}-{0:02d}-{1:02d}'.format(*map(int, d.split('/')))) for d in raw_data.columns[4:]]
    for i, record in raw_data.iterrows():
        for i, d in enumerate(record[4:]):
            location = record['Country/Region'].strip()
            if isinstance(record['Province/State'], str):
                location += ' - ' + record['Province/State'].strip()
            if d > 0:
                lines.append({
                    'location': location,
                    'country': record['Country/Region'],
                    'deaths': d,
                    'date': dates[i]
                })

    return pd.DataFrame(lines).set_index('date')

# uncomment this to change xls to faster loading csv for istat data
def convert_istat(dest):
    df_comuni_sett = pd.read_excel(os.path.join(dest, 'comuni_settimana.xlsx'))
    df_comuni_sett.to_csv('df_comuni_sett.csv')

def get_dataframes(dest, npt_rth=5, smooth=True):
    ############################ loading datasets ####################################################
    df_naz = pd.read_csv(
        'https://raw.githubusercontent.com/pcm-dpc/COVID-19/master/dati-andamento-nazionale/dpc-covid19-ita-andamento-nazionale.csv').drop('stato',1)
    reg = pd.read_csv(
        'https://raw.githubusercontent.com/pcm-dpc/COVID-19/master/dati-regioni/dpc-covid19-ita-regioni.csv')
    prov = pd.read_csv(
        'https://raw.githubusercontent.com/pcm-dpc/COVID-19/master/dati-province/dpc-covid19-ita-province.csv').drop('stato',1)
    # df_naz = pd.read_csv(os.path.join(dest,'dpc-covid19-ita-andamento-nazionale.csv')).drop('stato',1)
    print('last available date for Italy data',df_naz['data'].iloc[-1])
    # reg = pd.read_csv(os.path.join(dest,'dpc-covid19-ita-regioni.csv'))
    # prov = pd.read_csv(os.path.join(dest,'dpc-covid19-ita-province.csv')).drop('stato',1)
    df_naz.index = pd.to_datetime(df_naz.index)
    reg['data'] = pd.to_datetime(reg['data'])
    prov['data'] = pd.to_datetime(prov['data'])
    df_comuni_sett = pd.read_csv(os.path.join(dest, 'df_comuni_sett.csv'))
    # df_world_confirmed = pd.read_csv(os.path.join(dest,'time_series_covid19_confirmed_global.csv'))
    df_world_confirmed = pd.read_csv(
        'https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_confirmed_global.csv')
    df_world_deaths = pd.read_csv(
        'https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_deaths_global.csv')
    # df_world_deaths = pd.read_csv(os.path.join(dest,'time_series_covid19_deaths_global.csv'))
    # df_world_recovered = pd.read_csv(os.path.join(dest,'time_series_covid19_recovered_global.csv'))
    df_world_recovered = pd.read_csv('https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_recovered_global.csv')

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
        df_reg[data]['Rth'] = calculate_Rth(df_reg[data],npt_rth=npt_rth, version=1)
        df_reg[data]['Rth_v2'] = calculate_Rth(df_reg[data], npt_rth=npt_rth, version=2, smooth=smooth)
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
    return df_naz, reg, prov, df_reg, df_prov, df_world_confirmed, df_world_deaths, df_world_recovered, populations, ita_populations, df_comuni_sett

