import numpy as np
import scipy
np.seterr(divide='ignore', invalid='ignore')


def stripped_ols(y, x) -> dict:
    """
    Perform Ordinary Least Squares (OLS) regression analysis with stripped output.

    Parameters:
    y (array-like): Dependent variable values.
    x (array-like): Independent variable values. The matrix should be shaped as 
                   (number of observations, number of independent variables).

    Returns:
    dict: A dictionary containing regression coefficients ('b') and corresponding 
          p-values ('p') for each independent variable.

    Raises:
    ValueError: If inputs `x` or `y` are empty.

    Notes:
    - Missing values in `x` or `y` are not handled, and the function may produce
      unexpected results if there are missing values in the input data.
    - The function internally adds a constant column to the independent variables 
      matrix `x` to represent the intercept term in the regression equation.
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
