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

from arch.unitroot import engle_granger



df = yf.download(tickers='BTC-USD', start="2015-01-01", end="2020-12-31", interval = '1d')
# df['log return'] = np.log(df['Close']).diff()

df['log Close'] = np.log(df['Close'])
df['log return'] = np.log(df['Close']/df['Close'].shift(1)).fillna(0)

logrtn = np.log(df['Close']/df['Close'].shift(1)).fillna(0)
logrtn.plot()

# serie = np.correlate(df['log Close'], df['log Close'], mode='full')
# serie = pd.Series(sm.tsa.acf(df['log Close'], nlags=5))


# plot_acf(df['log Close'])

#Jungle-Box Text for Serial Corelation
sm.stats.acorr_ljungbox(logrtn, lags=[5], return_df=True)

#Null

plot_acf(logrtn)

eg_test = engle_granger(logrtn, logrtn, trend="n")
print(eg_test)


res = sm.tsa.ARMA(logrtn, (1,1)).fit(disp=-1)
print(sm.stats.diagnostic.het_arch(res.resid, nlags=12))

plot_acf(logrtn)