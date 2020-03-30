import numpy as np
import pandas as pd
import stat
import os
import git
import shutil
from gitinfo import get_git_info

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
    # print(filenames)
    t = os.path.join(dest,'temp')
    if os.path.exists(t):
        rmtree(t)
    os.makedirs(t, exist_ok=True)
    os.chmod(t, stat.S_IWUSR)
    # Clone into temporary dir
    repo = git.Repo.clone_from(url, t, branch='master', depth=1)
    # Copy desired file from temporary dir
    try:
        info = get_git_info(t)
        print('last commit ', info['author_date'])
    except Exception as e:
        print('could not retrieve repo infos, ',e)

    for filename in filenames:
        # print(t +'/'+ filename)
        shutil.move(t +'/'+ filename, os.path.join(dest, filename.split('/')[-1]))
    # Remove temporary dir
    rmtree(t)
    return None
#

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