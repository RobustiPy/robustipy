# Module containing utility functions for the library

import numpy as np
import pandas as pd
import scipy
from linearmodels.panel import PanelOLS #temporary solution to get PanelOLS estimates

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
