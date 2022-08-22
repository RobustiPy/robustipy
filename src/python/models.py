from prototypes import Protomodel
import statsmodels.api as sm
import scipy
import pandas as pd
import numpy as np
from itertools import chain, combinations
from tqdm import tqdm
from joblib import Parallel, delayed
from figures import main_figure


class OLSRobust(Protomodel):
    """
    Class for multi-variate regression using OLS
    Input: y = dependent variable
           x = independent variables
    Output:
           self = an object containing the key metrics
    """

    def __init__(self, y, x):
        self.y = None
        self.x = None
        self.results = None

    def fit(self, c, space, info='bic', s=1000):
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
            b_list, p_list, aic_list, bic_list = (zip(*Parallel(n_jobs=-1)(delayed(__strap)(comb)
                                                                           for i in range(0, s))))
            beta[spec, :] = b_list
            p[spec, :] = p_list
            aic[spec, :] = aic_list
            bic[spec, :] = bic_list
        return beta, p, aic, bic
    
    def __estimate(self):
        # Internal method for stimation
        # TODO: Review if this shouod in seprate module
        self.inv_xx = np.linalg.inv(np.dot(self.x.T, self.x))
        xy = np.dot(self.x.T, self.y)
        self.b = np.dot(self.inv_xx, xy)  # estimate coefficients
        self.nobs = self.y.shape[0]  # number of observations
        self.ncoef = self.x.shape[1]  # number of coef.
        self.df_e = self.nobs - self.ncoef  # degrees of freedom, error
        self.df_r = self.ncoef - 1  # degrees of freedom, regression
        self.e = self.y - np.dot(self.x, self.b)  # residuals
        self.sse = np.dot(self.e.T, self.e) / self.df_e  # SSE
        self.se = np.sqrt(np.diagonal(self.sse * self.inv_xx))  # coef. standard errors
        self.t = self.b / self.se  # coef. t-statistics
        self.p = (1 - scipy.stats.t.cdf(abs(self.t), self.df_e)) * 2  # coef. p-values
        self.R2 = 1 - self.e.var() / self.y.var()  # model R-squared
        self.R2adj = 1 - (1 - self.R2) * ((self.nobs - 1) / (self.nobs - self.ncoef))  # adjusted R-square
        self.ll = (-(self.nobs * 1 / 2) * (1 + np.log(2 * np.pi)) - (self.nobs / 2)
                   * np.log(np.dot(self.e.T, self.e) / self.nobs))
        self.aic = (-2 * self.ll) + (2 * self.ncoef)
        self.bic = (-2 * self.ll) + (self.ncoef * np.log(self.nobs))
        return self

    def __strap(self, comb_var):
        # Internal method for boostraing
        # TODO Review if put in another module
        samp_df = comb_var[np.random.choice(comb_var.shape[0], 800, replace=True)]
        # @TODO generalize the frac to the function call
        self.__estimate(samp_df[:, 0], samp_df[:, 1:])
        return self.b[0], self.p[0], self.aic, self.bic

    def summary(self):
        pass

    def plot(self):
        main_figure(results)
