import ipywidgets as widgets
from ipywidgets import interact, interact_manual
import sys
import os
sys.path.append(os.path.expanduser(os.path.join('~','Documents', 'projects', 'coronavirus')))
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import plotly.graph_objects as go
import src.tools as tools
import plotly.express as px
from scipy.integrate import odeint
from scipy.optimize import curve_fit
from scipy.optimize import differential_evolution
import scipy.integrate as integrate
from src.data_downloader import DATA_REPOS, download_from_repo

def SEIIIRDModel(t, S0, E0, I10, I20, I30, R0, D0, beta1, beta2, beta3,
                 IncubPeriod, DurMildInf, FracSevere, FracCritical,
                 DurHosp, DurICU, CFR):
    # SEIR model differential equations
    def SEIIIRD_deriv(y, t, beta1, beta2, beta3, IncubPeriod, DurMildInf,
                      FracSevere, FracCritical,
                      DurHosp, DurICU, CFR):
        S, E, I1, I2, I3, R, D = y
        N = S + E + I1 + I2 + I3 + R + D
        FracMild = 1. - FracSevere - FracCritical
        alpha = 1 / IncubPeriod
        gamma1 = (1 / DurMildInf) * FracMild
        p1 = (1 / DurMildInf) - gamma1
        p2 = (1 / DurHosp) * (FracCritical / (FracSevere + FracCritical))
        gamma2 = (1 / DurHosp) - p2
        mu = (1 / DurICU) * (CFR / FracCritical)
        gamma3 = (1 / DurICU) - mu
        dSdt = -(beta1 * I1 + beta2 * I2 + beta3 * I3) * S / N
        dEdt = (beta1 * I1 + beta2 * I2 + beta3 * I3) * S / N - alpha * E
        dI1dt = alpha * E - (gamma1 + p1) * I1
        dI2dt = p1 * I1 - (gamma2 + p2) * I2
        dI3dt = p2 * I2 - (gamma3 + mu) * I3
        dRdt = gamma1 * I1 + gamma2 * I2 + gamma3 * I3
        dDdt = mu * I3
        return dSdt, dEdt, dI1dt, dI2dt, dI3dt, dRdt, dDdt

    # Initial conditions vector
    y0 = S0, E0, I10, I20, I30, R0, D0
    # Integrate the SIR equations over time t
    ret = odeint(SEIIIRD_deriv, y0, t, args=(beta1, beta2, beta3, IncubPeriod,
                                             DurMildInf, FracSevere, FracCritical,
                                             DurHosp, DurICU, CFR))
    S, E, I1, I2, I3, R, D = ret.T

    return S, E, I1, I2, I3, R, D


def plot_SEIIIRD(time_range, S0, E0, I10, I20, I30,
                 R0, D0, beta1, beta2, beta3,
                 IncubPeriod, DurMildInf,
                 FracSevere, FracCritical,
                 DurHosp, DurICU, CFR,
                 plot_output=['I', 'D'],
):

    S, E, I1, I2, I3, R, D = SEIIIRDModel(time_range, S0, E0, I10, I20, I30,
                                          R0, D0, beta1, beta2, beta3,
                                          IncubPeriod, DurMildInf,
                                          FracSevere, FracCritical,
                                          DurHosp, DurICU, CFR)
    I = I1 + I2 + I3
    FracMild = 1. - FracSevere - FracCritical
    alpha = 1 / IncubPeriod
    gamma1 = (1 / DurMildInf) * FracMild
    p1 = (1 / DurMildInf) - gamma1
    p2 = (1 / DurHosp) * (FracCritical / (FracSevere + FracCritical))
    gamma2 = (1 / DurHosp) - p2
    mu = (1 / DurICU) * (CFR / FracCritical)
    gamma3 = (1 / DurICU) - mu
    r0 = 1 / (p1 + gamma1) * (beta1 + p1 / (p2 + gamma2) * (beta2 + beta3 * p2 / (mu * gamma3)))
    print('max number of infections ', int(np.max(I)))
    print('peak of infections ', np.argmax(I))
    print('max number of exposed ', int(np.max(E)))
    print('max number of recovered ', int(np.max(R)))
    print('max number of deaths ', int(np.max(D)))
    print('r0 ', r0)
    cases = {'I': I, 'I1': I1, 'I2': I2, 'I3': I3, 'S': S, 'E': E, 'R': R, 'D': D}

    fig = go.Figure()
    if 'S' in plot_output: fig.add_trace(go.Scatter(x=list(time_range), y=S, mode='lines', name='Susceptibles'))
    if 'I' in plot_output: fig.add_trace(go.Scatter(x=list(time_range), y=I, mode='lines', name='Total Infections'))
    if 'I1' in plot_output: fig.add_trace(go.Scatter(x=list(time_range), y=I1, mode='lines', name='Mild infections'))
    if 'I2' in plot_output: fig.add_trace(go.Scatter(x=list(time_range), y=I2, mode='lines', name='Severe Infections'))
    if 'I3' in plot_output: fig.add_trace(go.Scatter(x=list(time_range), y=I3, mode='lines', name='Critical Infections'))
    if 'R' in plot_output: fig.add_trace(go.Scatter(x=list(time_range), y=R, mode='lines', name='Recovered'))
    if 'E' in plot_output: fig.add_trace(go.Scatter(x=list(time_range), y=E, mode='lines', name='Exposed'))
    if 'D' in plot_output: fig.add_trace(go.Scatter(x=list(time_range), y=D, mode='lines', name='Deaths'))
    fig.update_layout(
        title="extended SEIRD Model",
        xaxis_title="days of spreading",
        yaxis_title="population",
        font=dict(
            family="Courier New, monospace",
            size=12,
            color="#7f7f7f"
        )
    )
    fig.show()
    return cases

def calibrate_SEIIIRD(
        ENERGY,
        y,
        fit_range,
        S0,
        I10,
        I20=0,
        I30=0,
        R0=0,
        E0=0,
        D0=0,
        bounds=None,
        use_differential_evolution=False,
        plot_range = 100,
        plot_output = None):
    # helper functions
    def SEIIIRD_differential(y, t, beta1, beta2, beta3, IncubPeriod, DurMildInf,
                             FracSevere, FracCritical, DurHosp, DurICU, CFR):

        S, E, I1, I2, I3, R, D = y
        N = S + E + I1 + I2 + I3 + R + D
        FracMild = 1. - FracSevere - FracCritical
        alpha = 1 / IncubPeriod
        gamma1 = (1 / DurMildInf) * FracMild
        p1 = (1 / DurMildInf) - gamma1
        p2 = (1 / DurHosp) * (FracCritical / (FracSevere + FracCritical))
        gamma2 = (1 / DurHosp) - p2
        mu = (1 / DurICU) * (CFR / FracCritical)
        gamma3 = (1 / DurICU) - mu
        dSdt = -(beta1 * I1 + beta2 * I2 + beta3 * I3) * S / N
        dEdt = (beta1 * I1 + beta2 * I2 + beta3 * I3) * S / N - alpha * E
        dI1dt = alpha * E - (gamma1 + p1) * I1
        dI2dt = p1 * I1 - (gamma2 + p2) * I2
        dI3dt = p2 * I2 - (gamma3 + mu) * I3
        dRdt = gamma1 * I1 + gamma2 * I2 + gamma3 * I3
        dDdt = mu * I3
        return dSdt, dEdt, dI1dt, dI2dt, dI3dt, dRdt, dDdt

    def SEIIIRDModel_solver(t, beta1, beta2, beta3, IncubPeriod, DurMildInf,
                            FracSevere, FracCritical, DurHosp, DurICU, CFR):
        # Initial conditions vector
        y0 = S0, E0, I10, I20, I30, R0, D0
        # Integrate the SEIIIRD equations over time t
        ret = odeint(SEIIIRD_differential, y0, t, args=(beta1, beta2, beta3,
                                                        IncubPeriod, DurMildInf,
                                                        FracSevere, FracCritical,
                                                        DurHosp, DurICU, CFR))
        series = {}
        series['S'] = ret.T[0]
        series['E'] = ret.T[1]
        series['I'] = ret.T[2] + ret.T[3] + ret.T[4]
        series['I1'] = ret.T[2]
        series['I2'] = ret.T[3]
        series['I3'] = ret.T[4]
        series['R'] = ret.T[5]
        series['D'] = ret.T[6]
        return series[ENERGY]

    def cost_fun(params):
        beta1, beta2, beta3, IncubPeriod, DurMildInf, FracSevere, FracCritical, DurHosp, DurICU, CFR = params
        return np.mean(np.abs(SEIIIRDModel_solver(x_fit, beta1, beta2, beta3, IncubPeriod,
                                                  DurMildInf, FracSevere, FracCritical, DurHosp, DurICU, CFR) - y_fit))

    y_fit = y[fit_range[0]:fit_range[1]]
    x_fit = range(len(y_fit))#np.linspace(0, len(y_fit), len(y_fit))

    if use_differential_evolution:
        print('calibrating with genetic algorithm...')
        optimization = differential_evolution(cost_fun, bounds, popsize=150, maxiter=10000)
        params = optimization.x
    else:
        if bounds is not None:
            lo_bound = []
            up_bound = []
            for bound in bounds:
                lo_bound.append(bound[0])
                up_bound.append(bound[1])
        else:
            lo_bound = None
            up_bound = None
        params, _ = curve_fit(f=SEIIIRDModel_solver, xdata=x_fit, ydata=y_fit,
                              method='trf', bounds=(lo_bound, up_bound))
    beta1, beta2, beta3, IncubPeriod, DurMildInf, FracSevere, FracCritical, DurHosp, DurICU, CFR = params
    time_range = range(plot_range)
    S, E, I1, I2, I3, R, D = SEIIIRDModel(time_range, S0, E0, I10, I20, I30,
                                          R0, D0, beta1, beta2, beta3,
                                          IncubPeriod, DurMildInf,
                                          FracSevere, FracCritical,
                                          DurHosp, DurICU, CFR)
    I = I1 + I2 + I3
    FracMild = 1. - FracSevere - FracCritical
    alpha = 1 / IncubPeriod
    gamma1 = (1 / DurMildInf) * FracMild
    p1 = (1 / DurMildInf) - gamma1
    p2 = (1 / DurHosp) * (FracCritical / (FracSevere + FracCritical))
    gamma2 = (1 / DurHosp) - p2
    mu = (1 / DurICU) * (CFR / FracCritical)
    gamma3 = (1 / DurICU) - mu
    r0 = 1 / (p1 + gamma1) * (beta1 + p1 / (p2 + gamma2) * (beta2 + beta3 * p2 / (mu * gamma3)))

    print('beta1', beta1)
    print('beta2', beta2)
    print('beta3', beta3)
    print('IncubPeriod', IncubPeriod)
    print('DurMildInf', DurMildInf)
    print('FracSevere', FracSevere)
    print('FracCritical', FracCritical)
    print('DurHosp', DurHosp)
    print('DurICU', DurICU)
    print('CFR', CFR)
    print('r0', r0)
    y_pred = SEIIIRDModel_solver(x_fit, *params)
    print('mae ', int(np.mean(np.abs(y_fit - y_pred))))
    plt.plot(y_pred)
#    cases = {'I': I, 'I1': I1, 'I2': I2, 'I3': I3, 'S': S, 'E': E, 'R': R, 'D': D}
    if plot_output is not None:
        fig = go.Figure()
        if 'S' in plot_output: fig.add_trace(go.Scatter(x=list(time_range), y=S, mode='lines', name='Susceptibles'))
        if 'I' in plot_output: fig.add_trace(go.Scatter(x=list(time_range), y=I, mode='lines', name='Total Infections'))
        if 'I1' in plot_output: fig.add_trace(go.Scatter(x=list(time_range), y=I1, mode='lines', name='Mild infections'))
        if 'I2' in plot_output: fig.add_trace(go.Scatter(x=list(time_range), y=I2, mode='lines', name='Severe Infections'))
        if 'I3' in plot_output: fig.add_trace(go.Scatter(x=list(time_range), y=I3, mode='lines', name='Critical Infections'))
        if 'R' in plot_output: fig.add_trace(go.Scatter(x=list(time_range), y=R, mode='lines', name='Recovered'))
        if 'E' in plot_output: fig.add_trace(go.Scatter(x=list(time_range), y=E, mode='lines', name='Exposed'))
        if 'D' in plot_output: fig.add_trace(go.Scatter(x=list(time_range), y=D, mode='lines', name='Deaths'))
        fig.add_trace(go.Scatter(x=list(range(len(x_fit))), y=y_fit, mode='lines', name='Actual'))
        fig.update_layout(
            title="extended SEIRD Model",
            xaxis_title="days of spreading",
            yaxis_title="population",
            font=dict(
                family="Courier New, monospace",
                size=12,
                color="#7f7f7f"
            )
        )
        fig.show()

    plt.plot(y_pred)
    plt.plot(y_fit, 'xr')

    return params

if __name__ == '__main__':
    dest = '/home/maurizio/Documents/projects/coronavirus/data'
    # download_from_repo(DATA_REPOS['italy']['url'], filenames=DATA_REPOS['italy']['streams'], dest=dest)
    ENERGY = 'I'
    cases_multiplier = 1
    deaths_multiplier = 1
    exposed_multiplier = 0
    start_day = 0
    end_day = 15
    df = pd.read_csv('../data/dpc-covid19-ita-andamento-nazionale.csv').drop('stato', 1)
    df['D'] = deaths_multiplier * df['deceduti']
    df['I'] = cases_multiplier * df['totale_positivi']
    df['I1'] = df['isolamento_domiciliare']  # mild infected
    df['I2'] = df['ricoverati_con_sintomi']  # severe infected
    df['I3'] = df['terapia_intensiva']  # critical infected
    df['R'] = df['dimessi_guariti']  # recovered
    N = 60431283  # population
    y_fit = df[ENERGY].iloc[start_day:end_day].values  # variable to fit
    I10 = df['I1'].iloc[start_day]
    I20 = df['I2'].iloc[start_day]
    I30 = df['I3'].iloc[start_day]
    R0 = df['R'].iloc[start_day]
    E0 = exposed_multiplier * df['I'].iloc[start_day]
    D0 = df['D'].iloc[start_day]
    S0 = N - E0 - I10 - I20 - I30 - R0 - D0
    time_range = 300
    beta1, beta2, beta3, IncubPeriod, DurMildInf, FracSevere, FracCritical, DurHosp, DurICU, CFR = calibrate_SEIIIRD(
        y_fit=y_fit,
        S0=S0,
        I10=I10,
        I20=I20,
        I30=I30,
        R0=R0,
        E0=E0,
        D0=D0,
        #time_horizon=time_range,
        bounds=[
            (0, .4),  # beta1
            (0, .4),  # beta2
            (0, .4),  # beta3
            (1, 30),  # IncubPeriod
            (1, 30),  # DurMildInf
            (0, 0.3),  # FracSevere
            (0, 0.1),  # FracCritical
            (1, 30),  # DurHosp
            (1, 30),  # DurICU
            (0, 0.1),  # CFR
        ],
        use_differential_evolution=False)
    cases = plot_SEIIIRD(range(time_range), N, E0, I10, I20, I30,
                         R0, D0, beta1, beta2, beta3,
                         IncubPeriod, DurMildInf, FracSevere, FracCritical, DurHosp,
                         DurICU, CFR, y_actual=y_fit, output=['I', 'D'])