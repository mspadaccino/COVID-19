### taken from https://github.com/alexamici/covid-19-notebooks
import pathlib

import numpy as np
import pandas as pd
import requests
import os
import git
import shutil
import tempfile

DATA_REPOS = {
    "world": {
        "url": "https://github.com/CSSEGISandData/COVID-19",
        "streams": {
            "deaths": "/csse_covid_19_data/csse_covid_19_time_series/time_series_19-covid-Deaths.csv",
            "confirmed": "/csse_covid_19_data/csse_covid_19_time_series/time_series_19-covid-Confirmed.csv",
            "recovered": "/csse_covid_19_data/csse_covid_19_time_series/time_series_19-covid-Recovered.csv"
        },
    },
    "italy": {
        "url": 'https://github.com/pcm-dpc/COVID-19',
        "streams": {
            "andamento-nazionale": "/dati-andamento-nazionale/dpc-covid19-ita-andamento-nazionale.csv",
            "regioni": "/dati-regioni/dpc-covid19-ita-regioni.csv",
            "province": "/dati-province/dpc-covid19-ita-province.csv",
        },
    },
}



def download_from_repo(url, filename, dest):
    # Create temporary dir
    t = tempfile.mkdtemp()
    # Clone into temporary dir
    git.Repo.clone_from(url, t, branch='master', depth=1)
    # Copy desired file from temporary dir

    shutil.move(t + filename, os.path.join(dest, filename.split('/')[-1]))
    # Remove temporary dir
    shutil.rmtree(t)
    return None


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