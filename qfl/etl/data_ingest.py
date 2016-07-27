"""
This module should ideally be ONLY tasked with the automation and triggering of
ETL tasks and should rely on other modules to actually implement those tasks?
EG, we shouldn't need references here to YahooApi...
"""

import pandas as pd
import datetime as dt
import numpy as np

from qfl.core.data_interfaces import YahooApi, QuandlApi, FigiApi
from qfl.core.data_interfaces import DatabaseInterface as db
import qfl.core.utils as utils


# Default start date for historical data ingests
default_start_date = dt.datetime(1990, 1, 1)


def initialize_data_environment():
    db.initialize()


def test_airflow():

    df = pd.DataFrame(np.random.randn(10, 5))
    # df.to_csv('test' + dt.datetime.today().__str__() + '.csv')
    print('successfully ran airflow test...')
    return True


def test_airflow_awesome():

    df = pd.DataFrame(np.random.randn(10, 5))
    # df.to_csv('test' + dt.datetime.today().__str__() + '.csv')
    print('test CSV is printing...')
    return True


def daily_equity_price_ingest():

    initialize_data_environment()
    data_source = 'yahoo'
    update_equity_prices(ids=None,
                         data_source=data_source,
                         _db=db)


def historical_equity_price_ingest():

    initialize_data_environment()
    date = utils.closest_business_day_in_past()
    data_source = 'yahoo'
    load_historical_equity_prices(ids=None,
                                  start_date=default_start_date,
                                  end_date=date,
                                  data_source=data_source,
                                  _db=db)


def historical_dividends_ingest():

    initialize_data_environment()
    load_historical_dividends(ids=None,
                              start_date=default_start_date,
                              end_date=dt.datetime.today(),
                              data_source='yahoo')


def _prep_equity_price_load(ids=None, _db=None):

    # Grab full universe
    equities = _db.get_data(table_name='equities', index_table=True)
    equity_prices_table = _db.get_table(table_name='equity_prices')

    # Default is everything
    if ids is None:
        ids = equities.index.tolist()

    rows = equities.loc[ids].reset_index()
    equity_tickers = rows['ticker'].tolist()

    # handle potential unicode weirdness
    equity_tickers = [str(ticker) for ticker in equity_tickers]

    return equities, equity_prices_table, ids, equity_tickers, rows


def _update_option_attrs(raw_data=None, _db=None):

    options_table = _db.get_table(table_name='equity_options')
    option_attributes = pd.DataFrame(
        raw_data.index.get_level_values(level='Symbol'))
    option_attributes = option_attributes.rename(
        columns={'Symbol': 'ticker'})
    option_attributes['option_type'] = raw_data.index.get_level_values(
        level='Type')
    option_attributes['strike_price'] = raw_data.index.get_level_values(
        level='Strike')
    option_attributes['maturity_date'] = raw_data.index.get_level_values(
        level='Expiry')
    option_attributes['underlying_id'] = _db.get_equity_ids(
        equity_tickers=raw_data['Underlying'])

    _db.execute_db_save(df=option_attributes,
                        table=options_table,
                        use_column_as_key='ticker')

    # Get their ID's
    t = tuple([str(ticker) for ticker in option_attributes['ticker']])
    q = 'select ticker, id from equity_options where ticker in {0}'.format(t)
    ticker_id_map = _db.read_sql(query=q)
    ticker_id_map.index = ticker_id_map['ticker']

    return ticker_id_map, option_attributes['ticker']


def update_option_prices(ids=None,
                         data_source='yahoo',
                         _db=None):
    for id_ in ids:
        update_option_prices_one(id_=id_,
                                 data_source=data_source,
                                 _db=_db)


def update_option_prices_one(id_=None,
                             data_source='yahoo',
                             _db=None):

    ticker = _db.get_equity_tickers(ids=[id_])[0]

    if data_source == 'yahoo':

        # Get raw data
        tmp = YahooApi.retrieve_options_data(ticker)
        raw_data = tmp[0]

        # Update option universe
        ticker_id_map, tickers = _update_option_attrs(raw_data, _db)

        option_prices_table = db.get_table(table_name='equity_option_prices')

        # Load prices
        option_prices = pd.DataFrame(columns=['date'])
        option_prices['date'] = raw_data['Quote_Time'].dt.date
        option_prices['quote_time'] = raw_data['Quote_Time'].dt.time
        option_prices['last_price'] = raw_data['Last']
        option_prices['bid_price'] = raw_data['Bid']
        option_prices['ask_price'] = raw_data['Ask']
        option_prices['iv'] = raw_data['IV']
        option_prices['volume'] = raw_data['Vol']
        option_prices['open_interest'] = raw_data['Open_Int']
        option_prices['spot_price'] = raw_data['Underlying_Price']
        option_prices['iv'] = option_prices['iv'].str.replace('%', '')
        option_prices['iv'] = option_prices['iv'].astype(float) / 100.0

        ids = ticker_id_map.loc[tickers, 'id']
        option_prices = option_prices.reset_index()
        option_prices['id'] = ids.values

        db.execute_db_save(df=option_prices,
                           table=option_prices_table)

    else:
        raise NotImplementedError


def update_equity_prices(ids=None,
                         data_source='yahoo',
                         _db=None):

    date = utils.closest_business_day_in_past(dt.datetime.today())

    load_historical_equity_prices(ids=ids,
                                  start_date=date,
                                  end_date=date,
                                  data_source=data_source,
                                  _db=_db,
                                  batch_ones=False)


def _load_historical_equity_prices(ids=None,
                                   start_date=default_start_date,
                                   end_date=dt.datetime.today(),
                                   data_source='yahoo',
                                   _db=None):
    # Prep
    equities, equity_prices_table, ids, equity_tickers, rows = \
        _prep_equity_price_load(ids, _db)

    if data_source == 'yahoo':

        prices = YahooApi.retrieve_prices(equity_tickers, start_date, end_date)
        prices_df = prices.to_frame()

        yahoo_fields = ['id', 'date', 'Open', 'High', 'Low',
                        'Close', 'Volume', 'Adj Close']

        db_fields = ['id', 'date', 'open_price', 'high_price', 'low_price',
                     'last_price', 'volume', 'adj_close']

        # Remove indices to prepare for database
        prices_df.index.names = ['date', 'ticker']
        prices_df = prices_df.reset_index()

        # Merge with ID's
        mapped_prices = pd.merge(left=prices_df,
                                 right=rows,
                                 on='ticker',
                                 how='inner')

    else:
        raise NotImplementedError

    # Map to database column structure
    equity_prices_data = pd.DataFrame(index=mapped_prices.index,
                                      columns=equity_prices_table.columns.keys())
    for i in range(0, len(yahoo_fields)):
        equity_prices_data[db_fields[i]] = mapped_prices[yahoo_fields[i]]

    _db.execute_db_save(equity_prices_data, equity_prices_table)


def load_historical_equity_prices(ids=None,
                                  start_date=default_start_date,
                                  end_date=dt.datetime.today(),
                                  data_source='yahoo',
                                  _db=None,
                                  batch_ones=True):

    if batch_ones:
        for _id in ids:
            _load_historical_equity_prices([_id],
                                           start_date,
                                           end_date,
                                           data_source,
                                           _db)
    else:
        _load_historical_equity_prices(ids,
                                       start_date,
                                       end_date,
                                       data_source,
                                       _db)


def load_historical_dividends(ids=None,
                              start_date=None,
                              end_date=None,
                              data_source='yahoo',
                              _db=None):

    if end_date is None:
        end_date = dt.datetime.today()

    if start_date is None:
        start_date = default_start_date

    # Use existing routine to get tickers and id defaults
    equities, equity_prices_table, ids, tickers, rows = \
        _prep_equity_price_load(ids)

    if data_source == 'yahoo':

        dividends = YahooApi.retrieve_dividends(equity_tickers=tickers,
                                                start_date=start_date,
                                                end_date=end_date)

        schedules_table = _db.get_table(table_name='equity_schedules')
        equities = _db.get_data(table_name='equities')

        schedules_data = pd.DataFrame(columns=schedules_table.columns.keys())
        schedules_data['ticker'] = dividends['Ticker']
        schedules_data['date'] = dividends['Date']
        schedules_data['value'] = dividends['Dividend']
        schedules_data['schedule_type'] = 'dividend'
        del schedules_data['id']

        schedules_data = pd.merge(left=schedules_data,
                                  right=equities,
                                  on='ticker')

    else:
        raise NotImplementedError

    _db.execute_db_save(df=schedules_data, table=schedules_table)


def add_equities_from_index(ticker=None, method='quandl', _db=None):

    # right now this uses quandl
    tickers = list()
    if method == 'quandl':
        if ticker not in QuandlApi.get_equity_index_universe():
            raise NotImplementedError
        tickers = QuandlApi.get_equity_universe(ticker)
    else:
        raise NotImplementedError

    # Add the equities
    add_equities_from_list(tickers=tickers)

    # Make sure string format is normal
    tickers = [str(ticker) for ticker in tickers]

    # Get the equities we just created
    equities_table = db.get_table(table_name='equities')
    equities_table_data = _db.get_data(table_name='equities')

    # Find the index mapping
    indices = _db.get_table(table_name='equity_indices')
    index_id = indices[indices['ticker'] == ticker]['index_id'].values[0]

    # Get index members table
    index_members_table = _db.get_table(table_name='equity_index_members')
    index_membership_data = pd.DataFrame(
        columns=index_members_table.columns.keys())
    index_membership_data['equity_id'] = equities_table_data['id']
    index_membership_data['valid_date'] = dt.date.today()
    index_membership_data['index_id'] = index_id

    # Update equity index membership table
    db.execute_db_save(df=index_membership_data, table=index_members_table)


def add_equities_from_list(tickers=None, _db=None):
    tickers_df = pd.DataFrame(data=tickers, columns=['ticker'])
    equities_table = _db.get_table(table_name='equities')
    db.execute_db_save(df=tickers_df, table=equities_table)

