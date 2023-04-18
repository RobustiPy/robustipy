import numpy as np
import scipy
np.seterr(divide='ignore', invalid='ignore')


def stripped_ols(y, x) -> dict:
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
    b = np.dot(inv_xx, xy)
    nobs = y.shape[0]  # number of observations
    ncoef = x.shape[1]  # number of coef.
    df_e = nobs - ncoef  # degrees of freedom, error
    e = y - np.dot(x, b)  # residuals
    sse = np.dot(e.T, e) / df_e  # SSE
    se = np.sqrt(np.diagonal(sse * inv_xx))  # coef. standard errors
    t = b / se  # coef. t-statistics
    p = (1 - scipy.stats.t.cdf(abs(t), df_e)) * 2  # coef. p-values
    return {'b': b,
            'p': p}
