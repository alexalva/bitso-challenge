# https://github.com/bashtage/arch
# https://github.com/0xboz/forecast_cryptocurrencies_volatility_garch_variants/blob/master/GARCH.ipynb

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
from matplotlib import pyplot

import statsmodels.api as sm
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.stats.diagnostic import het_arch

from arch.unitroot import engle_granger
from arch import arch_model

import pylab as py


df = yf.download(tickers='BTC-USD', start="2015-01-01", end="2020-12-31", interval = '1d')
df['log_price'] = np.log(df.price)
df['return'] = df.price.pct_change().dropna()
df['log_return'] = np.log(df['Close']/df['Close'].shift(1)).fillna(0)
df['squared_log_return'] = np.power(df['log_return'], 2)

# Scale up 100x
df['return_100x'] = np.multiply(df['return'], 100)
df['log_return_100x'] = np.multiply(df['log_return'], 100)

df.head()

logrtn = np.log(df['Close']/df['Close'].shift(1)).fillna(0)
# logrtn.index = df['Close']
logrtn.plot()

#Jungle-Box Text for Serial Corelation
seriestest = sm.stats.acorr_ljungbox(logrtn, return_df=True)
print(seriestest)


#Null

plot_acf(logrtn)

#Fit Arma-Garch

am = arch_model(logrtn, p=1, o=0, q=1, rescale= True)
res = am.fit(update_freq=5, disp="off")
print(res.summary())

#Show QQ-Plot

sm.qqplot(res.resid, line ='45')
py.show()

#Machine Learning

