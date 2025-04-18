# Module containing utility functions for the library

import numpy as np
import random
import scipy
import matplotlib
import pandas as pd
from itertools import chain, combinations
import statsmodels.api as sm
import matplotlib.pyplot as plt

def space_size(iterable) -> int:
    """
    Calculate the size of the power set of the given iterable.

    Parameters
    ----------
    iterable: iterable
        Input iterable.

    Returns
    ----------
    int:
        Size of the power set of the input iterable.
    """
    n = len(iterable)
    return int(2**n)


def all_subsets(ss):
    """
    Generate all subsets of a given iterable.

    Parameters
    ----------
    ss: iterable
        Input iterable.

    Returns
    ----------
    itertools.chain:
        A chain object containing all subsets of the input iterable.
    """
    return chain(*map(lambda x: combinations(ss, x),
                      range(0, len(ss) + 1)))


def make_aic(ll, n):
    return (-2.0 * ll) + (2 * n)


def make_bic(ll, n, k):
    return (-2.0 * ll) + (k * np.log(n))


def make_hqic(ll, n, k):
    return -2 * ll + 2 * k * np.log(np.log(n))


def logistic_regression_sm(y, x) -> dict:
    """
    Perform logistic regression based on statsmodels.Logit.

    Parameters
    ----------
    y : array-like
        Dependent variable values.
    x : array-like
        Independent variable values. The matrix should be shaped as (number of observations, number of independent variables).

    Returns
    ----------
    dict: Dictionary containing regression results, including coefficients, p-values, log-likelihood,
          AIC, BIC, and HQIC.
    """
    X_const = sm.add_constant(x, prepend=False)
    model = sm.Logit(y, X_const)
    result = model.fit(method='newton', tol=1e-8, disp=0)
    n = result.nobs
    k = result.df_model + 1
    ll = result.llf

    null_model = sm.Logit(y, np.ones_like(y))  # only intercept
    result_null = null_model.fit(method='newton', tol=1e-8, disp=0)
    ll_null = result_null.llf
    r2 = 1 - (ll / ll_null)
    return {'b': [[x] for x in result.params.values],
            'p': [[x] for x in result.pvalues.values],
            'r2': r2,
            'll': ll,
            'aic': make_aic(ll, n),
            'bic': make_bic(ll, n, k),
            'hqic': make_hqic(ll, n, k)
            }


def logistic_regression_sm_stripped(y, x) -> dict:
    """
    Perform logistic regression using statsmodels with stripped output.

    Parameters
    ----------
    y : array-like
        Dependent variable values.
    x: array-like
        Independent variable values. The matrix should be shaped as
                   (number of observations, number of independent variables).

    Returns
    ----------
    dict: A dictionary containing regression coefficients ('b') and corresponding
          p-values ('p') for each independent variable.
    """
    X_const = sm.add_constant(x, prepend=False)
    model = sm.Logit(y, X_const)
    result = model.fit(method='newton', tol=1e-8, disp=0)
#    n = result.nobs
#    k = result.df_model + 1
    ll = result.llf
    null_model = sm.Logit(y, np.ones_like(y))  # only intercept
    result_null = null_model.fit(method='newton', tol=1e-8, disp=0)
    ll_null = result_null.llf
    r2 = 1 - (ll / ll_null)
    return {'b': [[x] for x in result.params.values],
            'p': [[x] for x in result.pvalues.values],
            'r2': r2,
#            'll': ll#,
#            'aic': make_aic(ll, n),
#            'bic': make_bic(ll, n, k),
#            'hqic': make_hqic(ll, n, k)
            }



def simple_ols(y, x) -> dict:
    """
    Perform simple ordinary least squares regression.

    Parameters
    ----------
    y : array-like
        Dependent variable.
    x : array-like
        Independent variables.

    Returns
    ----------
    dict: Dictionary containing regression results, including coefficients, p-values, log-likelihood,
          AIC, BIC, and HQIC.
    """
    x['const'] = 1
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
    # df_r = ncoef - 1  # degrees of freedom, regression
    e = y - np.dot(x, b)  # residuals
    sse = np.dot(e.T, e) / df_e  # SSE
    se = np.sqrt(np.diagonal(sse * inv_xx))  # coef. standard errors
    t = b / se  # coef. t-statistics
    p = (1 - scipy.stats.t.cdf(abs(t), df_e)) * 2  # coef. p-values
    R2 = 1 - e.var() / y.var()  # model R-squared
    R2adj = 1 - (1 - R2) * ((nobs - 1) / (nobs - ncoef))  # adjusted R-square
    ll = (-(nobs * 1 / 2) * (1 + np.log(2 * np.pi)) - (nobs / 2)
          * np.log(abs(np.dot(e.T, e) / nobs)))
    return {'b': b,
            'p': p,
            'r2': R2adj,
            'll': ll,
            'aic': make_aic(ll, nobs),
            'bic': make_bic(ll, nobs, ncoef),
            'hqic': make_hqic(ll, nobs, ncoef)
            }


def group_demean(x, group=None):
    """
    Demean the input data within groups.

    Parameters
    ----------
    x:  pd.DataFrame
        Input DataFrame.
    group : str, optional
        Column name for grouping. Default is None.

    Returns
    ----------
    pd.DataFrame: Demeaned DataFrame.
    """
    data = x.copy()
    if group is None:
        return data - np.mean(data)
    data_gm = data.groupby([group]).transform('mean')
    out = data.drop(columns=group) - data_gm
    return pd.concat([out, data[group]], axis=1)


def decorator_timer(some_function):
    """
    Decorator function to measure the execution time of a function.

    Parameters
    ----------
    some_function : function
        Input function to be timed.

    Returns
    ----------
    function: Wrapped function with timing functionality.
    """
    from time import time

    def wrapper(*args, **kwargs):
        t1 = time()
        result = some_function(*args, **kwargs)
        end = time()-t1
        return result, end
    return wrapper


def get_selection_key(specs):
    """
    Generate selection keys for specifications.

    Parameters
    ----------
    specs: list
        List of lists containing specifications.

    Returns
    ----------
    list: List of frozen sets representing selection keys.
    """
    if all(isinstance(ele, list) for ele in specs):
        target = [frozenset(x) for x in specs]
        return target
    else:
        raise ValueError('Argument `specs` must be a list of list.')


def get_colormap_colors(colormap_name, num_colors=3, brightness_threshold=0.7):
    # Get the colormap by name
    colormap = plt.get_cmap(colormap_name)

    # Generate evenly spaced intervals between 0 and 1
    indices = np.linspace(0, 1, num_colors)

    # Extract the corresponding colors from the colormap
    colors = []
    for i in indices:
        color = colormap(i)

        # Calculate brightness (using a simple average of the RGB values)
        brightness = sum(color[:3]) / 3

        # If brightness is too high, move to a darker color
        while brightness > brightness_threshold and i > 0:
            i -= 0.05  # Move to a slightly darker color in the colormap
            color = colormap(i)
            brightness = sum(color[:3]) / 3

        colors.append(color)

    return colors


def get_colors(specs, color_set_name=None):
    """
    Generate colors for visualizing specifications.

    Parameters
    ----------
    specs: list
        List of lists containing specifications.
    color_set_name (str, optional): Name of the colormap. Default is 'Set1'.

    Returns
    ----------
    list: List of colors based on the specified colormap.
    """
    if color_set_name is None:
        color_set_name = 'Set1'
    if all(isinstance(ele, list) for ele in specs):
        colorset = matplotlib.colormaps[color_set_name]
        colorset = colorset.resampled(len(specs))
        return colorset.colors
    else:
        raise ValueError('Argument `specs` must be a list of list.')


def join_sig_test(*,
                  results_target,
                  results_shuffled,
                  sig_level,
                  positive):
    """
    Calculate joint significance test for the entire specification curve.

    Parameters
    ----------
    results_target : OLSResult
        Results object from the original analysis.
    results_shuffled : OLSResult
        Results object from shuffled analysis.
    sig_level : float
        Significance level threshold for specifications.
    positive : bool
        Direction of the joint significance test.

    Returns
    ----------
    float: Estimated p-value for the joint significance test.
    """
    if positive:
        target_sig_n = sum(results_target.summary_df.ci_down > 0)
        n_draws = []
        for col in results_shuffled.estimates:
            idx = results_shuffled.estimates[col] > 0
            n_draw = sum(results_shuffled.p_values[col][idx] < sig_level)
            n_draws.append(n_draw)
        shuffle_sig_n = sum(np.array(n_draws) >= target_sig_n)
        return shuffle_sig_n/results_shuffled.estimates.shape[1]
    else:
        target_sig_n = sum(results_target.summary_df.ci_up < 0)
        n_draws = []
        for col in results_shuffled.estimates:
            idx = results_shuffled.estimates[col] < 0
            n_draw = sum(results_shuffled.p_values[col][idx] < sig_level)
            n_draws.append(n_draw)
        shuffle_sig_n = sum(np.array(n_draws) >= target_sig_n)
        return shuffle_sig_n/results_shuffled.estimates.shape[1]


def prepare_union(path_to_union):
    """
    Prepare data for union example.

    Reads a Stata file from the given path, processes the data, and prepares it for
    regression analysis. The function creates binary indicators for categorical variables,
    augments the input data with a log-transformed wage variable, and handles missing
    values.

    Parameters
    ----------
    path_to_union : str
        File path to the Stata file containing union data.

    Returns
    ----------
    tuple: A tuple containing the dependent variable ('y'), list of control variables ('c'),
           independent variable ('x'), and the prepared DataFrame ('final_data').

    Raises
    ----------
    FileNotFoundError: If the file specified in 'path_to_union' does not exist.
    """
    union_df = pd.read_stata(path_to_union)
    union_df[['smsa','collgrad','married','union']]= union_df[['smsa','collgrad','married','union']].astype('str')
    union_df.loc[:, 'log_wage'] = np.log(union_df['wage'].copy()) * 100
    union_df = union_df[union_df['union'].notnull()].copy()
    union_df.loc[:, 'union'] = np.where(union_df['union'] == 'union', 1, 0)
    union_df.loc[:, 'married'] = np.where(union_df['married'] == 'married', 1, 0)
    union_df.loc[:, 'collgrad'] = np.where(union_df['collgrad'] == 'college grad', 1, 0)
    union_df.loc[:, 'smsa'] = np.where(union_df['smsa'] == 'SMSA', 1, 0)
    union_df[['smsa', 'collgrad', 'married', 'union']] = union_df[['smsa', 'collgrad', 'married', 'union']].astype('int')
    indep_list = ['hours',
                  'age',
                  'grade',
                  'collgrad',
                  'married',
                  'south',
                  'smsa',
                  'c_city',
                  'ttl_exp',
                  'tenure']
    for var in indep_list:
        union_df = union_df[union_df[var].notnull()]
    y = 'log_wage'
    c = indep_list
    x = 'union'
    final_data = pd.merge(union_df[y],
                          union_df[x],
                          how='left',
                          left_index=True,
                          right_index=True)
    final_data = pd.merge(final_data,
                          union_df[indep_list],
                          how='left',
                          left_index=True,
                          right_index=True)
    final_data = final_data.reset_index(drop=True)
    return y, c, x, final_data


def prepare_asc(asc_path):
    """
    Prepare data for ASC example.

    Reads a Stata file from the given path, processes the data, creates binary indicators
    for categorical variables, and handles missing values. One-hot encodes the 'year' column
    and computes interaction terms for specific columns.

    Parameters:
    asc_path (str): File path to the Stata file containing ASC data.

    Returns:
    tuple: A tuple containing the dependent variable ('y'), list of continuous variables ('x'),
           list of control variables ('c'), grouping variable ('group'), and the prepared DataFrame.

    Raises:
    FileNotFoundError: If the file specified in 'asc_path' does not exist.
    """
    ASC_df = pd.read_stata(asc_path, convert_categoricals=False)
    one_hot = pd.get_dummies(ASC_df['year'])
    ASC_df = ASC_df.join(one_hot)
    ASC_df['dcareNew*c.lrealgs'] = ASC_df['dcareNew'] * ASC_df['lrealgs']
    #ASC_df['constant'] = 1
    ASC_df = ASC_df[['wellbeing_kikert', 'lrealgs', 'dcareNew*c.lrealgs', 'dcareNew',
                     'DR', 'lgva', 'Mtotp', 'ddgree', 'age',
                     2005, 2006.0, 2007.0, 2009.0,
                     2010.0, 2011.0, 2012.0, 2013.0, 2014.0,
                     2015.0, 2016.0, 2017.0, 2018.0,
                     'married', 'widowed', 'disable', 'lrealtinc_m',
                     'house_ownership', 'hhsize', 'work', 'retired',
                     #'constant'
                     'pidp'
                     ]]
    ASC_df = ASC_df.dropna()
    y = 'wellbeing_kikert'
    x = ['lrealgs', 'dcareNew*c.lrealgs', 'dcareNew',
         'DR', 'lgva', 'Mtotp', 'ddgree', 'age',
         2005, 2006.0, 2007.0, 2009.0,
         2010.0, 2011.0, 2012.0, 2013.0, 2014.0,
         2015.0, 2016.0, 2017.0, 2018.0]
    c = ['married', 'widowed', 'disable', 'lrealtinc_m',
         'house_ownership', 'hhsize', 'work', 'retired'
         ]
    group = 'pidp'
    ASC_df['pidp'] = ASC_df['pidp'].astype(int)
    return y, c, x, group, ASC_df


def reservoir_sampling(generator, k):
    reservoir = []
    for i, item in enumerate(generator):
        if i < k:
            reservoir.append(item)
        else:
            j = random.randint(0, i)  # Randomly choose an index from 0 to i
            if j < k:
                reservoir[j] = item  # Replace element at the chosen index with new item
    return reservoir
