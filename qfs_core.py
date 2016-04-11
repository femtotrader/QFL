import pandas as pd
import datetime as dt
import pandas_datareader.data as pdata
import matplotlib.pyplot as plt

data = pdata.get_data_yahoo(symbols="^GSPC",
                            start="01/01/2010",
                            end="01/01/2015")

plt.figure()
data['Adj Close'].plot()
plt.show()


