# Module containing utility functions for the library

import numpy as np
import math
import scipy
import os
import pandas as pd
from itertools import chain, combinations
from linearmodels.panel import PanelOLS #temporary solution to get PanelOLS estimates


def space_size(iterable) -> int:
    n = len(iterable)
    n_per_iter = list(map(lambda x:
                          math.factorial(n) / math.factorial(x)
                          / math.factorial(n-x),
                          range(0, n + 1)))
    return round((sum(n_per_iter)))


def all_subsets(ss):
    return chain(*map(lambda x: combinations(ss, x),
                      range(0, len(ss) + 1)))


def simple_ols(y, x) -> dict:
    x = np.asarray(x)
    y = np.asarray(y)
    if x.size == 0 or y.size == 0:
        raise ValueError("Inputs must not be empty.")

    inv_xx = np.linalg.inv(np.dot(x.T, x))
    xy = np.dot(x.T, y)
    b = np.dot(inv_xx, xy)  # estimate coefficients
    nobs = y.shape[0]  # number of observations
    ncoef = x.shape[1]  # number of coef.
    df_e = nobs - ncoef  # degrees of freedom, error
    df_r = ncoef - 1  # degrees of freedom, regression
    e = y - np.dot(x, b)  # residuals
    sse = np.dot(e.T, e) / df_e  # SSE
    se = np.sqrt(np.diagonal(sse * inv_xx))  # coef. standard errors
    t = b / se  # coef. t-statistics
    p = (1 - scipy.stats.t.cdf(abs(t), df_e)) * 2  # coef. p-values
    R2 = 1 - e.var() / y.var()  # model R-squared
    R2adj = 1 - (1 - R2) * ((nobs - 1) / (nobs - ncoef))  # adjusted R-square
    ll = (-(nobs * 1 / 2) * (1 + np.log(2 * np.pi)) - (nobs / 2)
          * np.log(np.dot(e.T, e) / nobs))
    aic = (-2 * ll) + (2 * ncoef)
    bic = (-2 * ll) + (ncoef * np.log(nobs))
    return {'b': b,
            'p': p,
            'll': ll,
            'aic': aic,
            'bic': bic}


def scipy_ols(y, x):
    pass


def panel_ols(y, x):
    if np.asarray(x).size == 0 or np.asarray(y).size == 0:
        raise ValueError("Inputs must not be empty.")
    mod = PanelOLS(y,
                   x,
                   entity_effects=True,
                   drop_absorbed=True)
    res = mod.fit(cov_type='clustered',
                  cluster_entity=True)
    nobs = y.shape[0]  # number of observations
    ncoef = x.shape[1]  # number of coef.
    ll = res.loglik
    aic = (-2 * ll) + (2 * ncoef)
    bic = (-2 * ll) + (ncoef * np.log(nobs))
    p = res.pvalues
    b = res.params
    return {'b': b,
            'p': p,
            'll': ll,
            'aic': aic,
            'bic': bic}


def save_myrobust(beta, p, aic, bic, example_path):
    if os.path.exists(example_path) is False:
        os.mkdir(example_path)
    np.savetxt(os.path.join(example_path, 'betas.csv'),
               beta, delimiter=",")
    np.savetxt(os.path.join(example_path, 'p.csv'),
               p, delimiter=",")
    np.savetxt(os.path.join(example_path, 'aic.csv'),
               aic, delimiter=",")
    np.savetxt(os.path.join(example_path, 'bic.csv'),
               bic, delimiter=",")


def load_myrobust(d_path):
    beta = pd.read_csv(os.path.join(d_path, 'betas.csv'), header=None)
    p = pd.read_csv(os.path.join(d_path, 'p.csv'), header=None)
    aic = pd.read_csv(os.path.join(d_path, 'aic.csv'), header=None)
    bic = pd.read_csv(os.path.join(d_path, 'bic.csv'), header=None)
    list_df = []
    summary_df = pd.DataFrame(columns=['beta_med',
                                       'beta_max',
                                       'beta_min',
                                       'beta_std'])
    for strap in range(0, beta.shape[0]):
        new_df = pd.DataFrame()
        new_df['betas'] = beta.values[strap]
        new_df['p'] = p.values[strap]
        new_df['aic'] = aic.values[strap]
        new_df['bic'] = bic.values[strap]
        list_df.append(new_df)
        summary_df.at[strap, 'beta_med'] = np.median(beta.values[strap])
        summary_df.at[strap, 'beta_min'] = np.min(beta.values[strap])
        summary_df.at[strap, 'beta_max'] = np.max(beta.values[strap])
        summary_df.at[strap, 'beta_std'] = np.std(beta.values[strap])
        summary_df.at[strap, 'beta_std_plus'] = np.median(beta.values[strap]) + np.std(beta.values[strap])
        summary_df.at[strap, 'beta_std_minus'] = np.median(beta.values[strap]) - np.std(beta.values[strap])
    return beta, summary_df, list_df
