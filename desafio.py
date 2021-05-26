
# Raw Package

import numpy as np
import pandas as pd

#Data Source
import yfinance as yf

#Data Viz
import matplotlib.pyplot as plt
import seaborn as sns


df = yf.download(tickers='BTC-USD', start="2015-01-01", end="2020-12-31", interval = '1d')
df['log_close'] = np.log(df.Close)
df['return'] = df.Close.pct_change().dropna()
df['log_return'] = np.log(df['Close']/df['Close'].shift(1)).fillna(0)
df['squared_log_return'] = np.power(df['log_return'], 2)

# Scale up 100x
df['return_100x'] = np.multiply(df['return'], 100)
df['log_return_100x'] = np.multiply(df['log_return'], 100)

df.head()

import statsmodels.api as sm

logrtn = np.log(df['Close']/df['Close'].shift(1)).fillna(0)

#ljunge-Box Text for Serial Corelation
seriestest = sm.stats.acorr_ljungbox(logrtn, return_df=True)
print(seriestest)

from statsmodels.graphics.tsaplots import plot_acf
_ = plot_acf(logrtn, lags=40, title='Daily Log Return ACF')

from arch import arch_model

logrtn100 = np.multiply(logrtn, 100)
#Fit Arma-Garch
am = arch_model(logrtn, p=1, o=0, q=1, rescale= True, vol="Garch")
res = am.fit(update_freq=5, disp="off")
print(res.summary())

#Show QQ-Plot
sm.qqplot(res.resid, line ='45')

#EGARCH proposal
am = arch_model(logrtn100, p=1, o=1, q=1, rescale= True, vol="EGarch")
res = am.fit(update_freq=5, disp="off")
print(res.summary())


