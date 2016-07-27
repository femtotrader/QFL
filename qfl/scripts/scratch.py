
import pandas as pd
import datetime as dt
import pandas_datareader.data as pdata
import matplotlib.pyplot as plt
import numpy as np
import pyfolio
import urllib
import qfl.core.data_interfaces as data_interfaces
from qfl.core.data_interfaces import DatabaseInterface as db
import qfl.etl.data_ingest as etl
import qfl.core.calcs as lib
import pymc
import requests
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter, A4
import sqlalchemy as sa
from sqlalchemy.sql.expression import Insert, Select, and_, or_, bindparam
import qfl.core.utils as utils
from qfl.core.data_interfaces import YahooApi


# AWS





reload(etl)
reload(data_interfaces)
import qfl.core.data_interfaces as data_interfaces
from qfl.core.data_interfaces import DatabaseInterface as db

# Database stuff
db.initialize()

etl.daily_equity_price_ingest()

date = utils.closest_business_day_in_past()
data_source = 'yahoo'

etl.load_historical_equity_prices(ids=None,
                                  start_date=etl.default_start_date,
                                  end_date=date,
                                  data_source=data_source,
                                  _db=db)

equities, equity_prices_table, ids, equity_tickers, rows = \
    etl._prep_equity_price_load(ids=None, _db=db)


etl.update_equity_prices(ids=None, data_source='yahoo', _db=db)
print('done!')



_id = ids[0]
etl._load_historical_equity_prices(_id,
                                   start_date=date,
                                   end_date=date,
                                   data_source='yahoo',
                                   _db=db)



id_ = 3642
etl.update_option_prices_one(id_=id_, db=db)

ids = [3642, 3643]
etl.update_option_prices(ids, 'yahoo', db)

etl._load_historical_equity_prices(_id=3642,
                                   start_date=None,
                                   end_date=None,
                                   data_source='yahoo',
                                   _db=db)

# Test load of options prices from yahoo
start_date = dt.datetime(1990, 1, 1)
end_date = dt.datetime.today()

options_table = db.get_table(table_name='equity_options')

tmp = YahooApi.retrieve_options_data('ABBV')
raw_data = tmp[0]
unique_symbols = np.unique(raw_data.index.get_level_values(level='Symbol'))
unique_symbols = [str(symbol) for symbol in unique_symbols]

# Load attributes
option_attributes = pd.DataFrame(raw_data.index.get_level_values(level='Symbol'))
option_attributes = option_attributes.rename(columns={'Symbol': 'ticker'})
option_attributes['option_type'] = raw_data.index.get_level_values(
    level='Type')
option_attributes['strike_price'] = raw_data.index.get_level_values(
    level='Strike')
option_attributes['maturity_date'] = raw_data.index.get_level_values(
    level='Expiry')
option_attributes['underlying_id'] = db.get_equity_ids(
    equity_tickers=raw_data['Underlying'])

db.execute_db_save(df=option_attributes,
                   table=options_table,
                   use_column_as_key='ticker')


# Get their ID's
t = tuple([str(ticker) for ticker in option_attributes['ticker']])
q = 'select ticker, id from equity_options where ticker in {0}'.format(t)
ticker_id_map = db.read_sql(query=q)
ticker_id_map.index = ticker_id_map['ticker']

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

ids = ticker_id_map.loc[option_attributes['ticker'], 'id']
option_prices = option_prices.reset_index()
option_prices['id'] = ids.values

db.execute_db_save(df=option_prices,
                   table=option_prices_table)







# Testing archive of historical prices

equities = db.get_data(table_name='equities', index_table=True)
ids = equities.index.tolist()

rows = equities.loc[ids].reset_index()
equity_tickers = rows['ticker'].tolist()
equity_tickers = [str(ticker) for ticker in equity_tickers]

equity_tickers = equity_tickers[0:1]

ids = db.get_equity_ids(equity_tickers)

date = utils.closest_business_day_in_past()

expiry_dates, links = pdata.YahooOptions('ABBV') \
    ._get_expiry_dates_and_links()

data = pdata.YahooOptions('ABBV').get_near_stock_price(
    above_below=20, expiry=expiry_dates[0])

def hello(c):
    c.drawString(100, 100, "Hello World")

# Create a canvas
c = canvas.Canvas(filename="hello.pdf",
                  pagesize=letter,
                  bottomup=1,
                  pageCompression=0,
                  verbosity=0,
                  encrypt=None)
width, height = letter

# Draw some stuff


hello(c)
c.showPage()
c.save()


# FIGI identifiers - cool!
api_key = "471197f1-50fe-429b-9e11-a6828980e213"
req_data = [{"idType":"TICKER","idValue":"YHOO","exchCode":"US"}]
r = requests.post('https://api.openfigi.com/v1/mapping',
                  headers={"Content-Type": "text/json",
                           "X-OPENFIGI-APIKEY": api_key},
                  json=req_data)




lib.plot_invgamma(a=12, b=1)

# Bayesian normal regression example
# NOTE: the linear regression model we're trying to solve for is
# given by:
# y = b0 + b1(x) + error
# where b0 is the intercept term, b1 is the slope, and error is
# the error

float_df = pd.DataFrame()
float_df['weight'] = np.random.normal(0, 1, 100)
float_df['mpg'] = 0.5 * float_df['weight'] + np.random.normal(0, 0.5, 100)

# model the intercept/slope terms of our model as
# normal random variables with comically large variances
b0 = pymc.Normal("b0", 0, 0.0003)
b1 = pymc.Normal("b1", 0, 0.0003)

# model our error term as a uniform random variable
err = pymc.Uniform("err", 0, 500)

# "model" the observed x values as a normal random variable
# in reality, because x is observed, it doesn't actually matter
# how we choose to model x -- PyMC isn't going to change x's values
x_weight = pymc.Normal("weight", 0, 1, value=np.array(float_df["weight"]), observed=True)

# this is the heart of our model: given our b0, b1 and our x observations, we want
# to predict y
@pymc.deterministic
def pred(b0=b0, b1=b1, x=x_weight):
    return b0 + b1*x

# "model" the observed y values: again, I reiterate that PyMC treats y as
# evidence -- as fixed; it's going to use this as evidence in updating our belief
# about the "unobserved" parameters (b0, b1, and err), which are the
# things we're interested in inferring after all
y = pymc.Normal("y", pred, err, value=np.array(float_df["mpg"]), observed=True)

# put everything we've modeled into a PyMC model
model = pymc.Model([pred, b0, b1, y, err, x_weight])

mc = pymc.MCMC(model)
mc.sample(10000)
print np.mean(mc.trace('b1')[:])
plt.hist(mc.trace('b1')[:], bins=50)

print(__doc__)

# Authors: Alexandre Gramfort
#          Denis A. Engemann
# License: BSD 3 clause

import numpy as np
import matplotlib.pyplot as plt
from scipy import linalg

from sklearn.decomposition import PCA, FactorAnalysis
from sklearn.covariance import ShrunkCovariance, LedoitWolf
from sklearn.cross_validation import cross_val_score
from sklearn.grid_search import GridSearchCV

###############################################################################
# Create the data

n_samples, n_features, rank = 1000, 50, 10
sigma = 1.
rng = np.random.RandomState(42)
U, _, _ = linalg.svd(rng.randn(n_features, n_features))
X = np.dot(rng.randn(n_samples, rank), U[:, :rank].T)

# Adding homoscedastic noise
X_homo = X + sigma * rng.randn(n_samples, n_features)

# Adding heteroscedastic noise
sigmas = sigma * rng.rand(n_features) + sigma / 2.
X_hetero = X + rng.randn(n_samples, n_features) * sigmas

###############################################################################
# Fit the models

n_components = np.arange(0, n_features, 5)  # options for n_components


def compute_scores(X):
    pca = PCA()
    fa = FactorAnalysis()

    pca_scores, fa_scores = [], []
    for n in n_components:
        pca.n_components = n
        fa.n_components = n
        pca_scores.append(np.mean(cross_val_score(pca, X)))
        fa_scores.append(np.mean(cross_val_score(fa, X)))

    return pca_scores, fa_scores


def shrunk_cov_score(X):
    shrinkages = np.logspace(-2, 0, 30)
    cv = GridSearchCV(ShrunkCovariance(), {'shrinkage': shrinkages})
    return np.mean(cross_val_score(cv.fit(X).best_estimator_, X))


def lw_score(X):
    return np.mean(cross_val_score(LedoitWolf(), X))


for X, title in [(X_homo, 'Homoscedastic Noise'),
                 (X_hetero, 'Heteroscedastic Noise')]:
    pca_scores, fa_scores = compute_scores(X)
    n_components_pca = n_components[np.argmax(pca_scores)]
    n_components_fa = n_components[np.argmax(fa_scores)]

    pca = PCA(n_components='mle')
    pca.fit(X)
    n_components_pca_mle = pca.n_components_

    print("best n_components by PCA CV = %d" % n_components_pca)
    print("best n_components by FactorAnalysis CV = %d" % n_components_fa)
    print("best n_components by PCA MLE = %d" % n_components_pca_mle)

    plt.figure()
    plt.plot(n_components, pca_scores, 'b', label='PCA scores')
    plt.plot(n_components, fa_scores, 'r', label='FA scores')
    plt.axvline(rank, color='g', label='TRUTH: %d' % rank, linestyle='-')
    plt.axvline(n_components_pca, color='b',
                label='PCA CV: %d' % n_components_pca, linestyle='--')
    plt.axvline(n_components_fa, color='r',
                label='FactorAnalysis CV: %d' % n_components_fa, linestyle='--')
    plt.axvline(n_components_pca_mle, color='k',
                label='PCA MLE: %d' % n_components_pca_mle, linestyle='--')

    # compare with other covariance estimators
    plt.axhline(shrunk_cov_score(X), color='violet',
                label='Shrunk Covariance MLE', linestyle='-.')
    plt.axhline(lw_score(X), color='orange',
                label='LedoitWolf MLE' % n_components_pca_mle, linestyle='-.')

    plt.xlabel('nb of components')
    plt.ylabel('CV scores')
    plt.legend(loc='lower right')
    plt.title(title)

plt.show()






# Dividend retrieval
ticker = 'GE'
start_date = dt.datetime(1990, 01, 01)
end_date = dt.datetime.today()
dividends = db.YahooApi.retrieve_dividends(equity_ticker=ticker, start_date=start_date, end_date=end_date)

# Options retrieval
above_below = 20
test_data = pdata.YahooOptions(ticker).get_all_data()

expiry_dates, links = pdata.YahooOptions(ticker)._get_expiry_dates_and_links()
expiry_dates = [date for date in expiry_dates if date >= dt.datetime.today().date()]


options_data = pdata.YahooOptions(ticker).get_near_stock_price(above_below=above_below, expiry=expiry_dates[1])
print('done')

options_data = db.YahooApi.retrieve_options_data('GOOGL')
print('done')


data = pdata.get_data_yahoo(symbols='^RUT',
                            start="01/01/2010",
                            end="05/01/2015")


returns = data['Adj Close'] / data['Adj Close'].shift(1) - 1

returns.index = returns.index.normalize()
if returns.index.tzinfo is None:
    returns.index = returns.index.tz_localize('UTC')

pyfolio.create_returns_tear_sheet(returns=returns, return_fig=True)





test_options = pdata.YahooOptions('AAPL').get_all_data()

plt.figure()
data['Adj Close'].plot()
plt.show()

opener = urllib.URLopener()
target = 'https://s3.amazonaws.com/static.quandl.com/tickers/SP500.csv'
opener.retrieve(target, 'SP500.csv')
tickers_file = pd.read_csv('SP500.csv')
tickers = tickers_file['ticker'].values

plt.figure()
plt.hist(returns[np.isfinite(returns)])