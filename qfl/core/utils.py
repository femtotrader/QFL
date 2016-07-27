import datetime as dt
import pandas as pd
from pandas.tseries.offsets import BDay


def closest_business_day_in_past(date=None):
    if date is None:
        date = dt.datetime.today()
    return date + BDay(1) - BDay(1)


