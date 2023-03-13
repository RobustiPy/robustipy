# Module containing utility functions for the library

import numpy as np
import scipy
import os
import matplotlib
import pandas as pd
from itertools import chain, combinations
# temporary solution to get PanelOLS estimates
from linearmodels.panel import PanelOLS


def space_size(iterable) -> int:
    n = len(iterable)
    return int(2**n)


def all_subsets(ss):
    return chain(*map(lambda x: combinations(ss, x),
                      range(0, len(ss) + 1)))


def simple_ols(y, x) -> dict:
    x = np.asarray(x)
    y = np.asarray(y)
    if x.size == 0 or y.size == 0:
        raise ValueError("Inputs must not be empty.")
    try:
        inv_xx = np.linalg.inv(np.dot(x.T, x))
    except np.linalg.LinAlgError:
        inv_xx = np.linalg.pinv(np.dot(x.T, x))
    xy = np.dot(x.T, y)
    b = np.dot(inv_xx, xy)  # estimate coefficients
    nobs = y.shape[0]  # number of observations
    ncoef = x.shape[1]  # number of coef.
    df_e = nobs - ncoef  # degrees of freedom, error
    #df_r = ncoef - 1  # degrees of freedom, regression
    e = y - np.dot(x, b)  # residuals
    sse = np.dot(e.T, e) / df_e  # SSE
    se = np.sqrt(np.diagonal(sse * inv_xx))  # coef. standard errors
    t = b / se  # coef. t-statistics
    p = (1 - scipy.stats.t.cdf(abs(t), df_e)) * 2  # coef. p-values
    #R2 = 1 - e.var() / y.var()  # model R-squared
    #R2adj = 1 - (1 - R2) * ((nobs - 1) / (nobs - ncoef))  # adjusted R-square
    ll = (-(nobs * 1 / 2) * (1 + np.log(2 * np.pi)) - (nobs / 2)
          * np.log(abs(np.dot(e.T, e) / nobs)))
    aic = (-2.0 * ll) + (2 * ncoef)
    bic = (-2.0 * ll) + (ncoef * np.log(nobs))
    hqic = (-2.0 * ll) + 2 * np.log(np.log(nobs)) * ncoef
    return {'b': b,
            'p': p,
            'll': ll,
            'aic': aic,
            'bic': bic,
            'hqic': hqic}


def group_demean(x, group=None):
    data = x.copy()
    if group is None:
        return data - np.mean(data)
    data_gm = data.groupby([group]).transform(np.mean)
    return data.drop(columns=group) - data_gm


def simple_panel_ols(y, x, group):
    if np.asarray(x).size == 0 or np.asarray(y).size == 0:
        raise ValueError("Inputs must not be empty.")
    y_c = group_demean(y, group)
    x_c = group_demean(x, group)
    return simple_ols(x_c, y_c)


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


def save_spec(b_spec, p_spec, aic_spec, bic_spec, example_path):
    np.savetxt(os.path.join(example_path, 'b_spec.csv'),
               b_spec, delimiter=",")
    np.savetxt(os.path.join(example_path, 'p_spec.csv'),
               p_spec, delimiter=",")
    np.savetxt(os.path.join(example_path, 'aic_spec.csv'),
               aic_spec, delimiter=",")
    np.savetxt(os.path.join(example_path, 'bic_spec.csv'),
               bic_spec, delimiter=",")


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
    # the operation performed with this loop can be done with built-in pandas
    # functions for better performance. See function compute_summary bellow.
    # Use the decorator_timer to measure the execution time.
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


def load_spec(example_path):
    beta = pd.read_csv(os.path.join(example_path, 'b_spec.csv'), header=None)
    p = pd.read_csv(os.path.join(example_path, 'p_spec.csv'), header=None)
    aic = pd.read_csv(os.path.join(example_path, 'aic_spec.csv'), header=None)
    bic = pd.read_csv(os.path.join(example_path, 'bic_spec.csv'), header=None)
    return beta, p, aic, bic


def decorator_timer(some_function):
    from time import time

    def wrapper(*args, **kwargs):
        t1 = time()
        result = some_function(*args, **kwargs)
        end = time()-t1
        return result, end
    return wrapper


def compute_summary(data):
    out = pd.DataFrame()
    out['median'] = data.median(axis=1)
    out['max'] = data.max(axis=1)
    out['min'] = data.min(axis=1)
    out['ci_up'] = data.quantile(q=0.975, axis=1, interpolation='nearest')
    out['ci_down'] = data.quantile(q=0.025, axis=1, interpolation='nearest')
    return out


def get_selection_key(specs):
    if all(isinstance(ele, list) for ele in specs):
        target = [frozenset(x) for x in specs]
        return target
    else:
        raise ValueError('Argument `specs` must be a list of list.')


def get_colors(specs, color_set_name=None):
    if color_set_name is None:
        color_set_name = 'Set1'
    if all(isinstance(ele, list) for ele in specs):
        colorset = matplotlib.colormaps[color_set_name]
        colorset = colorset.resampled(len(specs))
        return colorset.colors
    else:
        raise ValueError('Argument `specs` must be a list of list.')


def panel_ols(y, x):
    if np.asarray(x).size == 0 or np.asarray(y).size == 0:
        raise ValueError("Inputs must not be empty.")
    try:
        mod = PanelOLS(y,
                       x,
                       entity_effects=True,
                       drop_absorbed=True,
                       # necessary because we dont always have
                       # full rank on some resamples...
                       # @TODO: a better way to handle this?
                      #                    check_rank=False
                                          )
        res = mod.fit(cov_type='clustered',
                      cluster_entity=True)
        nobs = y.shape[0]  # number of observations
        ncoef = x.shape[1]  # number of coef.
        ll = res.loglik
        aic = (-2 * ll) + (2 * ncoef)
        bic = (-2 * ll) + (ncoef * np.log(nobs))
        try:
        # @TODO: necessary due to a singular matrix error
        # due to resampling of the ASC data. Not sure what
        # else to do.
            p = res.pvalues
        except:
            p = np.nan
        b = res.params
        return {'b': b,
                'p': p,
                'll': ll,
                'aic': aic,
                'bic': bic}
    except ValueError:
        return {'b': np.nan,
                'p': np.nan,
                'll': np.nan,
                'aic': np.nan,
                'bic': np.nan}


def scipy_ols(y, x) -> dict:
    x = np.asarray(x)
    y = np.asarray(y)
    if x.size == 0 or y.size == 0:
        raise ValueError("Inputs must not be empty.")
    b, res, rnk, s = scipy.linalg.lstsq(x,
                                        y,
                                        lapack_driver='gelsy',
                                        check_finite=False)
    return {'b': b,
            'p': p,
            'll': ll,
            'aic': aic,
            'bic': bic}
