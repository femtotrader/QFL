import pandas as pd
import datetime as dt
import numpy as np
from scipy.stats import norm, expon, gamma, invgamma, beta, nct
from scipy.stats import multivariate_normal, wishart, invwishart, dirichlet
from scipy.stats import bernoulli
import pymc
import matplotlib.pyplot as plt


def _ensure_float(arr=None):
    output = [float(x) for x in arr]
    return output


def plot_normal(mu=None, sigma=None, x=None):
    mu = float(mu)
    sigma = float(sigma)
    if x == None:
        points = 5
        x = np.array(range(-3 * points, 3 * points)) * sigma / points + mu
    y = norm.pdf(x, mu, sigma)
    plt.plot(y, x)
    return pd.DataFrame(data=y, index=x, columns=['density'])


def compose_normals(mu1=None, mu2=None, sigma1=None, sigma2=None, x=None):
    [mu1, mu2, sigma1, sigma2] = _ensure_float([mu1, mu2, sigma1, sigma2])
    x1 = norm.pdf(x, mu1, sigma1)
    x2 = norm.pdf(x, mu2, sigma2)
    y = x1 * x2
    output = pd.DataFrame(index=x, data=y)
    output['x1'] = x1
    output['x2'] = x2
    output = output / output.sum() / (np.max(x) - np.min(x))
    return output


def plot_invgamma(a, b, x=None):
    a = float(a)
    b = float(b)
    if x == None:
        mean = b / (a - 1)
        stdev = (b ** 2 / ((a - 1) ** 2 * (a - 2))) ** 0.5
        points = 5
        x = np.array(range(-3 * points, 3 * points)) * stdev / points + mean
    y = invgamma.pdf(x, a, scale=b)
    plt.plot(y, x)
    return pd.DataFrame(data=y, index=x, columns=['density']), mean, stdev


def calc_forward_from_options(strikes, expiry_dates, option_types, option_prices):

    """
    :strikes:
    :expiry_dates:
    :option_types:
    :option_prices:
    :return:
    """

    # put-call parity: call + pv(cash) = put + stock + divs


def linear_setup(df, ind_cols, dep_col):
    '''
        Inputs: pandas Data Frame, list of strings for the independent variables,
        single string for the dependent variable
        Output: PyMC Model
    '''

    # model our intercept and error term as above
    b0 = pymc.Normal("b0", 0, 0.0001)
    err = pymc.Uniform("err", 0, 500)

    # initialize a NumPy array to hold our betas
    # and our observed x values
    b = np.empty(len(ind_cols), dtype=object)
    x = np.empty(len(ind_cols), dtype=object)

    # loop through b, and make our ith beta
    # a normal random variable, as in the single variable case
    for i in range(len(b)):
        b[i] = pymc.Normal("b" + str(i + 1), 0, 0.0001)

    # loop through x, and inform our model about the observed
    # x values that correspond to the ith position
    for i, col in enumerate(ind_cols):
        x[i] = pymc.Normal("x" + str(i + 1), 0, 1, value=np.array(df[col]),
                           observed=True)

    # as above, but use .dot() for 2D array (i.e., matrix) multiplication
    @pymc.deterministic
    def y_pred(b0=b0, b=b, x=x):
        return b0 + b.dot(x)

    # finally, "model" our observed y values as above
    y = pymc.Normal("y", y_pred, err, value=np.array(df[dep_col]),
                    observed=True)

    return pymc.Model(
        [b0, pymc.Container(b), err, pymc.Container(x), y, y_pred])