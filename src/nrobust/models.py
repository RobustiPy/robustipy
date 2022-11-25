from nrobust.prototypes import Protomodel
from nrobust.prototypes import Protoresult
import pandas as pd
import numpy as np
import warnings
from tqdm import tqdm
from joblib import Parallel, delayed
from nrobust.utils import simple_ols
from nrobust.utils import panel_ols
from nrobust.utils import space_size
from nrobust.utils import all_subsets
from nrobust.utils import compute_summary
from nrobust.figures import plot_results


class OLSResult(Protoresult):
    def __init__(self, *,
                 specs,
                 estimates,
                 p_values,
                 aic_array,
                 bic_array):
        super().__init__()
        self.specs_names = pd.Series(specs)
        self.estimates = pd.DataFrame(estimates)
        self.p_values = pd.DataFrame(p_values)
        self.summary_df = compute_summary(self.estimates)
        self.summary_df['aic'] = pd.Series(aic_array)
        self.summary_df['bic'] = pd.Series(bic_array)
        self.summary_df['spec_name'] = self.specs_names

    def summary(self):
        pass

    def plot(self,
             specs=None,
             colormap=None,
             colorset=None,
             figsize=(12, 6)):
        return plot_results(results_object=self,
                            specs=specs,
                            colormap=colormap,
                            colorset=colorset,
                            figsize=figsize)


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

    def __init__(self, *, y, x):
        super().__init__()
        self.y = y
        self.x = x
        self.results = None

    def get_results(self):
        return self.results

    def fit(self,
            *,
            controls,
            info='bic',
            mode='simple',
            draws=1000,
            sample_size=1000,
            replace=False):

        '''
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
        '''

        vars_names = list(controls.columns.values)
        space_n = space_size(vars_names)
        specs = []
        b_array = np.empty([space_n, draws])
        p_array = np.empty([space_n, draws])
        aic_array = np.empty([space_n])
        bic_array = np.empty([space_n])

        for spec, index in zip(all_subsets(vars_names),
                               tqdm(range(0, space_n))):
            if len(spec) == 0:
                comb = self.x
            else:
                comb = pd.merge(self.x, controls[list(spec)],
                                how='left', left_index=True,
                                right_index=True)
            comb = pd.merge(self.y, comb, how='left', left_index=True,
                            right_index=True)

            b_discard, p_discard, aic_i, bic_i = self._full_sample(comb, mode)

            b_list, p_list = (zip(*Parallel(n_jobs=-1)(delayed(self._strap)
                                                       (comb,
                                                        mode,
                                                        sample_size,
                                                        replace)
                                                       for i in range(0,
                                                                      draws))))

            specs.append(frozenset(spec))
            b_array[index, :] = b_list
            p_array[index, :] = p_list
            aic_array[index] = aic_i
            bic_array[index] = bic_i

        results = OLSResult(specs=specs,
                            estimates=b_array,
                            p_values=p_array,
                            aic_array=aic_array,
                            bic_array=bic_array)

        self.results = results

    def _estimate(self, y, x, mode):

        '''
        This method calls utils estimation functions
        depending of mode parameter. This method intentionally
        returns raw outputs from the estimation methods calls.

        Parameters
        ----------
        y : Array
          1D array like object (pandas dataframe of numpy array)
          contaning the data for y.
        x : Array
          ND array like object (pandas dataframe of numpy array)
          contaning the data for x and controls.
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
        '''

        if mode == 'simple':
            output = simple_ols(y, x)
        elif mode == 'panel':
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                output = panel_ols(y, x)
        b = output['b']
        p = output['p']
        aic = output['aic']
        bic = output['bic']
        return b, p, aic, bic

    def _full_sample(self, comb_var, mode):
        y = comb_var.iloc[:, :1]
        x = comb_var.iloc[:, 1:]
        if mode == 'simple':
            b, p, aic, bic = self._estimate(y=y,
                                            x=x,
                                            mode=mode)
            return b[0][0], p[0][0], aic[0][0], bic[0][0]
        elif mode == 'panel':
            b, p, aic, bic = self._estimate(y=y,
                                            x=x,
                                            mode=mode)
            # @TODO better error handling:
            # TypeError: 'float' object is not subscriptable
            try:
                b = b[0]
            except:
                b = b
            try:
                p = p[0]
            except:
                p = p
            return b, p, aic, bic
        else:
            raise ValueError(' "mode" argument not found')

    def _strap(self, comb_var, mode, sample_size, replace):

        '''
        This method calls self._estimate() over a random sample
        of the data contaning y, x and controls. Returns a single
        value for each returning variable.

        Parameters
        ----------
        comb_var : Array
                ND array like object (pandas dataframe of numpy array)
                contaning the data for y, x, and controls.
        mode : str
            Estimation method. Curretly only supporting simple OLS
            and panel OLS.

        Returns
        -------
        beta : float
            Estimate for x.
        p : float
         P value for x.
        aic : float
           Akaike Information Criteria for the model.
        bic : float
           Bayesian Information Criteria for the model.
        '''

        samp_df = comb_var.sample(n=sample_size, replace=replace)
        # @TODO generalize the frac to the function call
        y = samp_df.iloc[:, :1]
        x = samp_df.iloc[:, 1:]
        if mode == 'simple':
            b, p, aic, bic = self._estimate(y=y,
                                            x=x,
                                            mode=mode)
            return b[0][0], p[0][0]
        elif mode == 'panel':
            b, p, aic, bic = self._estimate(y=y,
                                            x=x,
                                            mode=mode)
            # @TODO better error handling:
            # TypeError: 'float' object is not subscriptable
            try:
                b = b[0]
            except:
                b = b
            try:
                p = p[0]
            except:
                p = p
            return b, p, aic, bic
        else:
            raise ValueError(' "mode" argument not found')
