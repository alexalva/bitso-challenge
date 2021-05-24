# https://github.com/bashtage/arch

# from data_client import DataClient
# pair_list = DataClient().get_binance_pairs(base_currencies=['USDT'],quote_currencies=['BTC'])
# print(pair_list)
# store_data = DataClient().kline_data(pair_list,'12h',start_date='06/01/2019',end_date='06/05/2019',storage=['csv','kline_data/'],progress_statements=True)


# Raw Package
import numpy as np
import pandas as pd

#Data Source
import yfinance as yf

#Data viz
import plotly.graph_objs as go

import statsmodels.api as sm
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.stats.diagnostic import het_arch



df = yf.download(tickers='BTC-USD', start="2015-01-01", end="2020-12-31", interval = '1d')
# df['log return'] = np.log(df['Close']).diff()

df['log return'] = np.log(df['Close']/df['Close'].shift(1))

df['log Close'] = np.log(df['Close'])

serie = np.correlate(df['log Close'], df['log Close'], mode='full')

serie = pd.Series(sm.tsa.acf(df['log Close'], nlags=5))


df['log Close'] = np.log(df['Close'])

plot_acf(df['log Close'])
plot_acf(df['log return'])
