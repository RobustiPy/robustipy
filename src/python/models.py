from src.python.prototypes import Protomodel
import statsmodels.api as sm
import scipy
import pandas as pd
import numpy as np
from tqdm import tqdm
from joblib import Parallel, delayed
from src.python.utils import simple_ols, panel_ols, space_size, all_subsets


class OLSRobust(Protomodel):
    """
    Class for multi-variate regression using OLS

    Parameters
    ----------
    y : Array like object
     Array like object containing the data for the dependent variable.
    x : Array like object
     Array like object containing the data for the independent variable(s).

    Returns
    -------
    self : Object
        An object containing the key metrics.
    """

    def __init__(self, y, x):
        super().__init__()
        self.y = y
        self.x = x
        self.results = None

    def fit(self,
            controls,
            info='bic',
            samples=1000,
            mode='simple'):

        """
        Fit the OLS models into the specification space
        as well as over the bootstrapped samples.

        Parameters
        ----------
        controls : Data.Frame
                Pandas data frame containing the all the possible
                control variables of the model.
        info : str
            Type of information criteria to be included in the
            output. Defaults to 'bic'.
        samples : int
              Number of bootstrap samples to collect.
        mode : str
            Estimation method. Curretly only supporting simple OLS
            and panel OLS.

        Returns
        -------
        beta : Array
            Numpy array contaning the estimates of the independent variable x
            for each specification of control variables across all the
            bootstrap samples.
        p : Array
         Numpy array containing the p values of all the betas.
        aic : Array
           Numpy array containing aic values for estimated models.
        bic : Array
           Numpy array containing aic values for estimated models.
        """

        vars_names = list(controls.columns.values)
        space_n = space_size(vars_names)
        beta = np.empty([space_n, samples])
        p = np.empty([space_n, samples])
        aic = np.empty([space_n, samples])
        bic = np.empty([space_n, samples])

        for spec, _ in zip(all_subsets(vars_names), tqdm(range(space_n))):
            if len(spec) == 0:
                comb = self.x
            else:
                comb = pd.merge(self.x, controls[list(spec)],
                                how='left', left_index=True,
                                right_index=True)
            comb = pd.merge(self.y, comb, how='left', left_index=True,
                            right_index=True)
            comb.loc[:, 'constant'] = 1

            b_list, p_list, aic_list, bic_list = (zip(*Parallel(n_jobs=-1)
                                                      (delayed(self._strap)(comb, mode)
                                                       for i in range(0, samples))))
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
