import requests
import pandas as pd
from datetime import datetime
import numpy as np
import datetime as dt
import pandas_datareader as pdr
from pandas_datareader import data as pdr
from scipy.stats import norm, t
import matplotlib.pyplot as plt
import yfinance as yf
import seaborn as sns
from creds import stlouis_api_key
yf.pdr_override() # <== that's all it takes :-)

def get_intr_rate():
    api_key= stlouis_api_key
    series_id = 'FEDFUNDS'  # Federal Funds Rate
    url = f'https://api.stlouisfed.org/fred/series/observations?series_id={series_id}&api_key={api_key}&file_type=json'
    response = requests.get(url)
    data = response.json()
    return float(data['observations'][-1]['value'])/100

# def get_variables(tickers):    
#     var_type = type(tickers)
#     if var_type == str:
#         tickers = yf.Tickers(tickers)
#     elif var_type == list:
#         tickersstr = ' '.join(tickers)
#         tickers = yf.Tickers(tickersstr)
#     k = get_options_strike(tickers)
#     T = get_options_timeto_expertion(tickers)
#     S0 = getstockprice(tickers)
#     r = get_intr_rate()
#     u = 1 + get_average_daily_volatility(tickers)
#     d = 1/u
#     options = pd.DataFrame(get_options_price(tickers))
#     return k, T, S0, r, u, d, options

def get_options_price(tickers):
    var_type = type(tickers)
    if var_type == str:
        tickers = yf.Tickers(tickers)
    elif var_type == list:
        tickersstr = ' '.join(tickers)
        tickers = yf.Tickers(tickersstr)
    # k = get_options_strike(tickers)
    # T = get_options_timeto_expertion(tickers)
    S0 = getstockprice(tickers)
    r = get_intr_rate()
    u = get_average_daily_volatility(tickers)

    options = {}
    columns = ['contractSymbol', 'bid', 'ask', 'lastPrice', 'strike']
    additional_columns = ['option_type', 'expirationindays', 'underlying', 'up']
    options_prices = {i:[] for i in columns+additional_columns}
    for i in tickers.tickers:
        str_exp = tickers.tickers[i].options[0:5]
        for idate in str_exp:
            date_i = datetime.strptime(idate, '%Y-%m-%d')
            today = datetime.today()
            delta = date_i - today
            if delta.days < 0:
                continue
            options[f"{i}_{idate}_C"] = tickers.tickers[i].option_chain(idate).calls[columns]
            for index, row in options[f"{i}_{idate}_C"].iterrows():
                for col in options_prices.keys():
                    if col in additional_columns:
                        if col == 'option_type':
                            options_prices[col].append('C')
                        elif col == 'expirationindays':
                            options_prices[col].append(delta.days)
                        elif col == 'underlying':
                            options_prices[col].append(i)
                        elif col == 'up':
                            options_prices[col].append(u.loc[i, 0])
                        continue
                    options_prices[col].append(row[col])
            options[f"{i}_{idate}_P"] = tickers.tickers[i].option_chain(idate).puts[columns]
            for index, row in options[f"{i}_{idate}_P"].iterrows():
                for col in options_prices.keys():
                    if col in additional_columns:
                        if col == 'option_type':
                            options_prices[col].append('P')
                        elif col == 'expirationindays':
                            options_prices[col].append(delta.days)
                        elif col == 'underlying':
                            options_prices[col].append(i)
                        elif col == 'up':
                            options_prices[col].append(u.loc[i, 0])
                        continue
                    options_prices[col].append(row[col])
    options_prices_df = pd.DataFrame(options_prices)
    return S0, r, u, options_prices_df


def get_options_strike(tickers):
    K = {}
    for i in tickers.tickers:
        str_exp = tickers.tickers[i].options[0:5]
        for idate in str_exp:
            date_i = datetime.strptime(idate, '%Y-%m-%d')
            today = datetime.today()
            delta = date_i - today
            K[f"{i}_{idate}_C"] = tickers.tickers[i].option_chain(idate).calls['strike'].tolist()
            K[f"{i}_{idate}_P"] = tickers.tickers[i].option_chain(idate).puts['strike'].tolist()
    return K

def get_options_timeto_expertion(tickers):
    T = {}
    for i in tickers.tickers:
        str_exp = tickers.tickers[i].options[0:5]
        for idate in str_exp:
            date_i = datetime.strptime(idate, '%Y-%m-%d')
            today = datetime.today()
            delta = date_i - today
            # date_i = ''.join(idate.split('-'))
            T[f"{i}_{idate}"] = delta.days
    return T

def getstockprice(tickers):
    var_type = type(tickers)
    if var_type == str:
        tickers = yf.Tickers(tickers)
    elif var_type == list:
        tickersstr = ' '.join(tickers)
        tickers = yf.Tickers(tickersstr)
    S0 = {i: round(tickers.tickers[i].get_fast_info().last_price, 2) for i in tickers.tickers}
    return S0

def american_fast_tree(K,T,S0,r,N,u,d,opttype='P'):
    # k = strike price
    # T = time to expiration
    # S0 = stock price
    # r = risk free rate
    # N = number of time steps
    # u = up factor
    # d = down factor
    # opttype = 'C' for call, 'P' for put
    #precompute values
    dt = T/N
    q = (np.exp(r*dt) - d)/(u-d)
    disc = np.exp(-r*dt)

    # initialise stock prices at maturity
    S = S0 * d**(np.arange(N,-1,-1)) * u**(np.arange(0,N+1,1))

    # option payoff
    if opttype == 'P':
        C = np.maximum(0, K - S)
    else:
        C = np.maximum(0, S - K)

    # backward recursion through the tree
    for i in np.arange(N-1,-1,-1):
        S = S0 * d**(np.arange(i,-1,-1)) * u**(np.arange(0,i+1,1))
        C[:i+1] = disc * ( q*C[1:i+2] + (1-q)*C[0:i+1] )
        C = C[:-1]
        if opttype == 'P':
            C = np.maximum(C, K - S)
        else:
            C = np.maximum(C, S - K)

    return C[0]

def binomial_tree_fast(K,T,S0,r,N,u,d,opttype='C'):
    #precompute constants
    dt = T/N
    q = (np.exp(r*dt) - d) / (u-d)
    disc = np.exp(-r*dt)

    # initialise asset prices at maturity - Time step N
    C = S0 * d ** (np.arange(N,-1,-1)) * u ** (np.arange(0,N+1,1))

    # initialise option values at maturity
    C = np.maximum( C - K , np.zeros(N+1) )

    # step backwards through tree
    for i in np.arange(N,0,-1):
        C = disc * ( q * C[1:i+1] + (1-q) * C[0:i] )

    return C[0]

def get_average_daily_volatility(tickers):
    # return per stock
    volatilities = {i:0 for i in tickers.tickers}
    for i in tickers.tickers:
        prices = tickers.tickers[i].history(period='1Y', interval='1D')['Close']
        prices = prices.dropna()
        returns = prices.pct_change().dropna()
        volatility = returns.std()
        volatilities[i] = volatility
    volatilities = pd.DataFrame(volatilities.values(), index=volatilities.keys())
    volatilities = volatilities + 1

    return volatilities