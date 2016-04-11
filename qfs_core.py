import pandas as pd
import datetime as dt
import pandas_datareader.data as pdata
%matplotlib inline

yahoo = pdata.get_data_yahoo("YHOO", "01/01/2010", dt.datetime.today)
yahoo.plot()
