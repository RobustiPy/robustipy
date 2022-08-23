from .prototypes import Protomodel
import statsmodels.api as sm
import scipy
import pandas as pd
import numpy as np
from itertools import chain, combinations
from tqdm import tqdm
from joblib import Parallel, delayed
from .figures import main_figure


class OLSRobust(Protomodel):
    """
    Class for multi-variate regression using OLS
    Input: y = dependent variable
           x = independent variables
    Output:
           self = an object containing the key metrics
    """

    def __init__(self, y, x):
        super().__init__()
        self.y = y
        self.x = x
        self.results = None

    def fit(self, c, space, info='bic', s=1000):
        beta = np.empty([len(space), s])
        p = np.empty([len(space), s])
        aic = np.empty([len(space), s])
        bic = np.empty([len(space), s])
        for spec in tqdm(range(len(space))):
            if spec == 0:
                comb = self.x
            else:
                comb = pd.merge(self.x, c[list(space[spec])],
                                how='left', left_index=True,
                                right_index=True)
            comb = pd.merge(self.y, comb, how='left', left_index=True,
                            right_index=True)
            comb.loc[:, 'constant'] = 1
            comb = comb.to_numpy()
            b_list, p_list, aic_list, bic_list = (zip(*Parallel(n_jobs=-1)(delayed(self._strap)(comb)
                                                                           for i in range(0, s))))
            beta[spec, :] = b_list
            p[spec, :] = p_list
            aic[spec, :] = aic_list
            bic[spec, :] = bic_list
        return beta, p, aic, bic
    
    def _estimate(self, y, x):
        # Internal method for stimation
        # TODO: Review if this shouod in seprate module
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
        return b, p, aic, bic 

    def _strap(self, comb_var):
        # Internal method for boostraing
        # TODO Review if put in another module
        samp_df = comb_var[np.random.choice(comb_var.shape[0], 800, replace=True)]
        # @TODO generalize the frac to the function call
        b, p, aic, bic = self._estimate(samp_df[:, 0], samp_df[:, 1:])
        return b[0], p[0], aic, bic

    def _all_subsets(self, ss):
        return chain(*map(lambda x: combinations(ss, x),
                          range(0, len(ss) + 1)))
    
    def _get_mspace(self, varnames) -> list:
        model_space = []
        for subset in self.__all_subsets(varnames):
            model_space.append(subset)
        return model_space

