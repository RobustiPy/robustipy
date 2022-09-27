from .prototypes import Protomodel
import statsmodels.api as sm
import scipy
import pandas as pd
import numpy as np
from itertools import chain, combinations
from tqdm import tqdm
from joblib import Parallel, delayed
from .figures import main_figure
from .utils import simple_ols, panel_ols


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

    def fit(self, c, space, info='bic', s=1000, mode='simple'):
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
            #comb = comb.to_numpy()
            b_list, p_list, aic_list, bic_list = (zip(*Parallel(n_jobs=-1)(delayed(self._strap)(comb, mode)
                                                                           for i in range(0, s))))
            beta[spec, :] = b_list
            p[spec, :] = p_list
            aic[spec, :] = aic_list
            bic[spec, :] = bic_list
        return beta, p, aic, bic
    
    def _estimate(self, y, x, mode):      
        if mode == 'simple':
            output = simple_ols(y, x)
        elif mode == 'panel':
            output = panel_ols(y, x)
        b = output['b']
        p = output['p']
        aic = output['aic']
        bic = output['bic']
        return b, p, aic, bic

    def _strap(self, comb_var, mode):
        # Internal method for boostraing
        # TODO Review if put in another module
        #samp_df = comb_var[np.random.choice(comb_var.shape[0], 800, replace=True)]
        samp_df = comb_var.sample(5000, replace=True)
        # @TODO generalize the frac to the function call
        b, p, aic, bic = self._estimate(samp_df.iloc[:, 0],
                                        samp_df.iloc[:, 1:],
                                        mode)
        return b[0], p[0], aic, bic

