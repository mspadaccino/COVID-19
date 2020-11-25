import pandas as pd
import inspect
import numpy as np
import matplotlib.pyplot as plt
import scipy
from scipy.optimize import curve_fit
from sklearn.linear_model import LinearRegression


def add_extra_features(df_orig):
    df = df_orig.copy()
    colslist = ['ricoverati_con_sintomi','terapia_intensiva','totale_ospedalizzati','isolamento_domiciliare',
                'totale_positivi','variazione_totale_positivi','nuovi_positivi',
                'dimessi_guariti','deceduti','totale_casi','tamponi','casi_testati']
    for col in colslist:
        if col in df.columns:
            if col == 'totale_casi':
                df['growth_factor'] = df['totale_casi'].diff() / df['totale_casi'].shift().diff()
            if col == 'totale_positivi':
                df['terapia_intensiva_su_totale_positivi'] = df['terapia_intensiva'] / df['totale_positivi']
            if col == 'deceduti':
                df['deceduti_su_tot'] = df['deceduti'] / df['totale_casi']
                df['deceduti_su_dimessi'] = df['deceduti'] / df['dimessi_guariti']
                df['deceduti_con_TI'] = df['deceduti'] + df['terapia_intensiva']
            if col == 'casi_testati':
                df['deceduti_su_casi_testati'] = df['deceduti']/df['casi_testati']
                df['totale_casi_su_casi_testati'] = df['totale_casi']/df['casi_testati']
                df['totale_ospedalizzati_su_casi_testati'] = df['totale_ospedalizzati'] /df['casi_testati']
            if col == 'tamponi':
                df['totale_ospedalizzati_su_tamponi'] = df['totale_ospedalizzati'] / df['tamponi']
                df['totale_casi_su_tamponi'] = df['totale_casi'] / df['tamponi']
                df['deceduti_su_tamponi'] = df['deceduti'] / df['tamponi']
                df['casi_testati_su_tamponi'] = df['casi_testati'] / df['tamponi']
            df['daily_'+col] = df[col].diff()
            df['%daily_'+ col] = df[col].diff()/df[col].shift()
            if col == 'deceduti':
                df["daily_casi_gravi"] = df['deceduti'].diff() + df['terapia_intensiva'].diff()

    if 'nuovi_positivi' in df.columns:
        df['nuovi_positivi_su_daily_casi_testati'] = df['nuovi_positivi'] / df['daily_casi_testati']
        df['nuovi_positivi_su_daily_tamponi'] = df['nuovi_positivi'] / df['daily_tamponi']
        df['nuovi_positivi_su_daily_terapia_intensiva'] = df['nuovi_positivi'] / df['daily_terapia_intensiva']
        df['daily_deceduti_su_nuovi_positivi@t-20'] = df['daily_deceduti'] / df['nuovi_positivi'].shift(20)
    # if 'terapia_intensiva' in df.columns:
    #     df['deceduti_su_terapia_intensiva@t-20'] = df['deceduti'] / df['terapia_intensiva'].shift(20)
    #     df['deceduti_su_totale_ospedalizzati@t-20'] = df['deceduti'] / df['totale_ospedalizzati'].shift(20)


    if 'data' in df.columns:
        df['data'] = pd.to_datetime(df['data']).dt.strftime('%m/%d/%Y')
        df = df.set_index('data')


    return df

def calculate_Rth(data, npt_rth = 5, version=1, smooth=True):
    if version == 1:
        days_rth = list(range(npt_rth))
        data['totale_positivi_slope'] = np.nan
        data['dimessi_guariti_slope'] = np.nan
        data['deceduti_slope'] = np.nan
        data['r_th'] = np.nan
        for i in range(npt_rth - 1, data.shape[0]):
            i1 = i - npt_rth + 1
            i2 = i + 1
            data.loc[data.index[i],'totale_positivi_slope'] = scipy.stats.linregress(days_rth,data['totale_positivi'].iloc[i1:i2]).slope
            data.loc[data.index[i],'dimessi_guariti_slope'] = scipy.stats.linregress(days_rth,data['dimessi_guariti'].iloc[i1:i2]).slope
            data.loc[data.index[i],'deceduti_slope'] = scipy.stats.linregress(days_rth, data['deceduti'].iloc[i1:i2]).slope
            data['r_th'] = ((data['totale_positivi_slope'] + data['dimessi_guariti_slope'] + data['deceduti_slope']) / (
                        data['dimessi_guariti_slope'] + data['deceduti_slope'])).clip(0, 1000)

    else:
        if smooth:
            tempseries = data['nuovi_positivi'].rolling(npt_rth).mean().rolling(npt_rth).sum()
        else:
            tempseries = data['nuovi_positivi'].rolling(npt_rth).sum()
        data['r_th'] = 1+np.log(tempseries / tempseries.shift(npt_rth))
    # data['r_th'] = 1 + data['totale_positivi'] / npt_rth * data['nuovi_positivi']
    return data['r_th'].replace(1000,np.nan).fillna(method='ffill')


def plot_model(df, col, backward_fit=-1, backward_fit_gomp=-1, forward_look=5, stdev=1,
               plotlimit=False, plotdifferential=True, plotloglinear=True,
               show_log=True, show_exp=False, show_gomp=True, show_dgomp=False, show_pol=False, label=''):
    y = df[col].values
    ndays = df.shape[0]
    x = np.linspace(1, ndays, ndays)
    max_fit = len(x)
    # x_fit = x[:max_fit-backward_fit]
    # y_fit = y[:max_fit-backward_fit]
    x_fit = x[:backward_fit]
    x_fit_gomp = x[:backward_fit_gomp]
    y_fit = y[:backward_fit]
    y_fit_gomp = y[:backward_fit_gomp]

    x_pred = np.linspace(1, ndays + forward_look, (ndays + forward_look))

    plt.figure(figsize=(12, 12))
    plt.subplot(311)
    plt.grid()
    plt.title(label+' '+col+' model calibrated from day ' + str(backward_fit) + ' days')
    plt.plot(x, y, 'ko', label="Original Data")

    if show_exp:
        popt_exp, pcov_exp = curve_fit(func_exp, x_fit, y_fit, method='lm', maxfev=10000)
        y_pred_exp = func_exp(x_pred, *popt_exp)
        exp_rmse = np.sqrt(np.mean((y_fit - func_exp(x_fit, *popt_exp))**2))
        plt.plot(x_pred, y_pred_exp, 'r-', label="exp model rmse %i"%int(exp_rmse))
        perr = np.sqrt(np.diag(pcov_exp))
        popt_exp_up = popt_exp + perr * stdev
        popt_exp_down = popt_exp - perr * stdev
        plt.fill_between(x_pred, func_exp(x_pred, *popt_exp_down), func_exp(x_pred, *popt_exp_up), alpha=0.2)

    if show_pol:
        popt_pol, pcov_pol = curve_fit(func_pol, x_fit, y_fit)
        y_pred_pol = func_pol(x_pred, *popt_pol)
        pol_rmse = np.sqrt(np.mean((y_fit - func_pol(x_fit, *popt_pol)) ** 2))
        plt.plot(x_pred, y_pred_pol, 'b-', label="pol model rmse %i"%int(pol_rmse))
        perr = np.sqrt(np.diag(pcov_pol))
        popt_pol_up = popt_pol + perr * stdev
        popt_pol_down = popt_pol - perr * stdev
        plt.fill_between(x_pred, func_pol(x_pred, *popt_pol_down), func_pol(x_pred, *popt_pol_up), alpha=0.2)
        if plotlimit:
            pollimit = func_pol(50, *popt_pol)
            pol_limit = np.array([pollimit for i in range(len(x_pred))])
            plt.plot(x_pred, pol_limit, 'y--', label='pol limit: %i' % (pollimit))

    if show_log:
        sig = inspect.signature(func_log)
        n_params = len(sig.parameters.items()) -1
        popt_log, pcov_log = curve_fit(func_log, x_fit, y_fit, bounds=([0. for item in range(n_params)], [np.inf for item in range(n_params)]))
        y_pred_log = func_log(x_pred, *popt_log)
        log_rmse = np.sqrt(np.mean((y_fit - func_log(x_fit, *popt_log)) ** 2))
        plt.plot(x_pred, y_pred_log, 'g-', label="log model rmse %i" % int(log_rmse))
        perr = np.sqrt(np.diag(pcov_log))
        popt_log_up = popt_log + perr * stdev
        popt_log_down = popt_log - perr * stdev
        plt.fill_between(x_pred, func_log(x_pred, *popt_log_down), func_log(x_pred, *popt_log_up), alpha=0.2)
        plt.fill_between(x_pred, func_log(x_pred, *popt_log_down), func_log(x_pred, *popt_log_up), alpha=0.2)
        plt.axvspan(backward_fit, x_pred[-1], alpha=0.1, color='green')
        if plotlimit:
            loglimit = func_log(50, *popt_log)
            log_limit = np.array([loglimit for i in range(len(x_pred))])
            plt.plot(x_pred, log_limit, 'g--', label='log limit: %i' % (loglimit))
            plt.ylim(0, loglimit * 1.2)
        plt.legend(loc='upper left')

    if show_gomp:
        sig = inspect.signature(func_gomp)
        n_params = len(sig.parameters.items()) - 1
        popt_gomp, pcov_gomp = curve_fit(func_gomp, x_fit_gomp, y_fit_gomp,  bounds=([0. for item in range(n_params)], [np.inf for item in range(n_params)]))
        y_pred_gomp = func_gomp(x_pred, *popt_gomp)
        gomp_rmse = np.sqrt(np.mean((y_fit_gomp - func_gomp(x_fit_gomp, *popt_gomp)) ** 2))
        plt.plot(x_pred, y_pred_gomp, 'c-', label="gomp model rmse %i" % int(gomp_rmse))
        perr = np.sqrt(np.diag(pcov_gomp))
        popt_gomp_up = popt_gomp + perr * stdev
        popt_gomp_down = popt_gomp - perr * stdev
        plt.fill_between(x_pred, func_gomp(x_pred, *popt_gomp_down), func_gomp(x_pred, *popt_gomp_up), alpha=0.2, color='cyan')
        plt.fill_between(x_pred, func_gomp(x_pred, *popt_gomp_down), func_gomp(x_pred, *popt_gomp_up), alpha=0.2, color='cyan')
        plt.axvspan(backward_fit_gomp, x_pred[-1], alpha=0.1, color='cyan')
        if plotlimit:
            gomplimit = popt_gomp[0]
            gomp_limit = np.array([gomplimit for i in range(len(x_pred))])
            plt.plot(x_pred, gomp_limit, 'c--', label='gompertz limit: %i' % (gomplimit))
            #plt.ylim(0, gomplimit*1.2)
        plt.legend(loc='upper left')

    if show_dgomp:
        sig = inspect.signature(func_dgomp)
        n_params = len(sig.parameters.items()) - 1
        popt_dgomp, pcov_dgomp = curve_fit(func_gomp, x_fit_gomp, y_fit_gomp,  bounds=([0. for item in range(n_params)], [np.inf for item in range(n_params)]))
        y_pred_gomp = func_dgomp(x_pred, *popt_dgomp)
        gomp_rmse = np.sqrt(np.mean((y_fit_gomp - func_dgomp(x_fit_gomp, *popt_dgomp)) ** 2))
        plt.plot(x_pred, y_pred_gomp, 'c-', label="dgomp model rmse %i" % int(gomp_rmse))
        perr = np.sqrt(np.diag(pcov_dgomp))
        popt_gomp_up = popt_dgomp + perr * stdev
        popt_gomp_down = popt_dgomp - perr * stdev
        plt.fill_between(x_pred, func_dgomp(x_pred, *popt_gomp_down), func_dgomp(x_pred, *popt_gomp_up), alpha=0.2, color='cyan')
        plt.fill_between(x_pred, func_dgomp(x_pred, *popt_gomp_down), func_dgomp(x_pred, *popt_gomp_up), alpha=0.2, color='cyan')
        plt.axvspan(backward_fit_gomp, x_pred[-1], alpha=0.1, color='cyan')
        plt.legend(loc='upper left')
        print(popt_dgomp)

    plt.subplot(312)
    if plotdifferential:
        plt.subplot(312)
        if show_log: plt.plot(x_pred, dfunc(x_pred, func_log, *popt_log), 'g-', label='logistic daily variations')
        if show_gomp: plt.plot(x_pred, dfunc(x_pred, func_gomp, *popt_gomp), 'c-', label='gompertz daily variations')
        plt.bar(x, np.diff(y, prepend=0), alpha=0.4)
        plt.legend(loc='lower left')
    if 'growth_factor' in df.columns:
        plt.grid()
        plt.twinx()
        growth_threshold_line = np.array([1 for i in range(len(x_pred))])
        plt.plot(x_pred, growth_threshold_line, 'b--', alpha=0.1, label='growth factor threshold')
        plt.fill_between(x_pred, 0, growth_threshold_line, alpha=0.1)
        plt.plot(x, df['growth_factor'], label='growth factor', alpha=.3)
    if '%delta_' + col in df.columns:
        plt.bar(x, df['%delta_' + col], color='m', label='daily % increase', alpha=.15)
        #plt.legend(loc='lower left')
    plt.legend(loc='upper left')

    if plotloglinear:
        model = LinearRegression()
        model.fit(x_fit.reshape(-1, 1), np.log(y_fit))
        r_sq = model.score(x_fit.reshape(-1, 1), np.log(y_fit))

        plt.subplot(313)
        plt.title('log-linear model fit R2: %f' % r_sq)
        plt.plot(x, np.log(df[col].values), 'b-*')
        plt.plot(x_pred, model.predict(x_pred.reshape(-1, 1)), 'r--')
        plt.show()

    if show_log:
        next_day_prediction_log = func_log(x[-1] + 1, *popt_log)
        print('next day prediction for log model: ', int(next_day_prediction_log))
    if show_gomp:
        next_day_prediction_gomp = func_gomp(x[-1] + 1, *popt_gomp)
        print('next day prediction for gomp model: ', int(next_day_prediction_gomp))
    if show_exp:
        next_day_prediction_exp = func_exp(x[-1] + 1, *popt_exp)
        print('next day prediction for exp model: ', int(next_day_prediction_exp))
    if show_pol:
        next_day_prediction_pol = func_pol(x[-1] + 1, *popt_pol)
        print('next day prediction for pol model: ', int(next_day_prediction_pol))

    print(df[col].tail())


