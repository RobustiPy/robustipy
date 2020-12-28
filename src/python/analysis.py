import statsmodels.api as sm
import scipy
import pandas as pd
import numpy as np
from itertools import chain, combinations
from tqdm import tqdm
from joblib import Parallel, delayed


class Ols:
    """
    Doc: Class for multi-variate regression using OLS
    Input:
        y = dependent variable
        x = independent variables
    Output:
        self = an object containing the key metrics

    """
    def __init__(self, y, x):
        """Initializing the ols class."""
        self.y = y
        self.x = x
        self.estimate()

    def estimate(self):
        """Estimating coefficients, and basic stats."""
        self.inv_xx = np.linalg.inv(np.dot(self.x.T, self.x))
        xy = np.dot(self.x.T, self.y)
        self.b = np.dot(self.inv_xx, xy)  # estimate coefficients
        self.nobs = self.y.shape[0]  # number of observations
        self.ncoef = self.x.shape[1]  # number of coef.
        self.df_e = self.nobs - self.ncoef  # degrees of freedom, error
        self.df_r = self.ncoef - 1  # degrees of freedom, regression
        self.e = self.y - np.dot(self.x, self.b)  # residuals
        self.sse = np.dot(self.e, self.e) / self.df_e  # SSE
        self.se = np.sqrt(np.diagonal(self.sse * self.inv_xx))  # coef. standard errors
        self.t = self.b / self.se  # coef. t-statistics
        self.p = (1 - scipy.stats.t.cdf(abs(self.t), self.df_e)) * 2  # coef. p-values
        self.R2 = 1 - self.e.var() / self.y.var()  # model R-squared
        self.R2adj = 1 - (1 - self.R2) * ((self.nobs - 1) / (self.nobs - self.ncoef))  # adjusted R-square
        self.ll = -(self.nobs * 1 / 2) * (1 + np.log(2 * np.pi)) - (self.nobs / 2) * np.log(np.dot(self.e, self.e) / self.nobs)
        self.aic = -2 * self.ll / self.nobs + (2 * self.ncoef / self.nobs)
        self.bic = -2 * self.ll / self.nobs + (self.ncoef * np.log(self.nobs)) / self.nobs
        return self


def strap(comb_var):
    samp_df = comb_var[np.random.choice(comb_var.shape[0], 800, replace=True)]
    # @TODO generalize the frac to the function call
    results = Ols(samp_df[:, 0], samp_df[:, 1:])
    return results.b[0], results.p[0], results.aic, results.bic


def get_mspace(varnames) -> list:
    model_space = []

    def all_subsets(ss):
        return chain(*map(lambda x: combinations(ss, x),
                          range(0, len(ss) + 1)))

    for subset in all_subsets(varnames):
        model_space.append(subset)
    return model_space


def run_ols_sm(y, x):
    x.loc[:, 'constant'] = 1
    model = sm.OLS(y, x, hasconst=True).fit()
    print(model.summary())
    return model.params[0]


def make_robust(y, x, c, space, info='bic', s=1000):
    beta = np.empty([len(space), s])
    p = np.empty([len(space), s])
    aic = np.empty([len(space), s])
    bic = np.empty([len(space), s])
    for spec in tqdm(range(len(space))):
        if spec == 0:
            comb = x
        else:
            comb = pd.merge(x, c[list(space[spec])],
                            how='left', left_index=True,
                            right_index=True)
        comb = pd.merge(y, comb, how='left', left_index=True,
                        right_index=True)
        comb.loc[:, 'constant'] = 1
        comb = comb.to_numpy()
        b_list, p_list, aic_list, bic_list = zip(*Parallel(n_jobs=-1)(delayed(strap)(comb) for i in range(0, s)))
        beta[spec, :] = b_list
        p[spec, :] = p_list
        aic[spec, :] = aic_list
        bic[spec, :] = bic_list
    return beta, p, aic, bic
