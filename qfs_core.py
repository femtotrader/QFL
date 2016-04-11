import pandas as pd
import datetime as dt
import pandas_datareader.data as pdata
import matplotlib.pyplot as plt
import numpy as np

data = pdata.get_data_yahoo(symbols="^GSPC",
                            start="01/01/2010",
                            end="01/01/2015")

plt.figure()
data['Adj Close'].plot()
plt.show()

returns = data['Adj Close'] / data['Adj Close'].shift(1) - 1
plt.figure()
plt.hist(returns[np.isfinite(returns)])