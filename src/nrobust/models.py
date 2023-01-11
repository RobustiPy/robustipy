from nrobust.prototypes import Protomodel
from nrobust.prototypes import Protoresult
import pandas as pd
import numpy as np
import warnings
from tqdm import tqdm
from joblib import Parallel, delayed
from nrobust.utils import simple_ols
from nrobust.utils import simple_panel_ols
from nrobust.bootstrap_utils import stripped_ols
from nrobust.bootstrap_utils import stripped_panel_ols
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
    y : str
     Name of the dependent variable.
    x : str or list<str>
     Names of names of the independent variable(s).
    data : DataFrame
     DataFrame contaning all the data to be used in the model.

    Returns
    -------
    self : Object
        An object containing the key metrics.
    """

    def __init__(self, *, y, x, data):
        super().__init__()
        if data.isnull().values.any():
            raise ValueError('NaNs are not supported. NaN values found in data')
        self.y = y
        self.x = x
        self.data = data
        self.results = None

    def get_results(self):
        return self.results

    def fit(self,
            *,
            controls,
            info='bic',
            group: str = None,
            draws=1000,
            sample_size=1000,
            replace=False):

        '''
        Fit the OLS models into the specification space
        as well as over the bootstrapped samples.

        Parameters
        ----------
        controls : list<str>
                List containing all the names of the  possible
                control variables of the model.
        info : str
            Type of information criteria to be included in the
            output. Defaults to 'bic'.
        samples : int
              Number of bootstrap samples to collect.
        group : str
            Grouping variable. If provided a Fixed Effects model is estimated.

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

        space_n = space_size(controls)
        specs = []
        b_array = np.empty([space_n, draws])
        p_array = np.empty([space_n, draws])
        aic_array = np.empty([space_n])
        bic_array = np.empty([space_n])

        for spec, index in zip(all_subsets(controls),
                               tqdm(range(0, space_n))):
            if len(spec) == 0:
                comb = self.data[self.x]
            else:
                comb = pd.merge(self.data[self.x],
                                self.data[list(spec)],
                                how='left',
                                left_index=True,
                                right_index=True)
            comb = pd.merge(self.data[self.y],
                            comb,
                            how='left',
                            left_index=True,
                            right_index=True)
            if group:
                comb = pd.merge(self.data[[group]],
                                comb,
                                how='left',
                                left_index=True,
                                right_index=True)

            b_discard, p_discard, aic_i, bic_i = self._full_est(comb, group)

            b_list, p_list = (zip(*Parallel(n_jobs=-1)(delayed(self._strap_est)
                                                       (comb,
                                                        group,
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

    def _full_est(self, comb_var, group):
        if group is None:
            y = comb_var.loc[:, :self.y]
            x = comb_var.drop(columns=self.y)
            out = simple_ols(y=y,
                             x=x)
            return out['b'][0][0], out['p'][0][0], out['aic'][0][0], out['bic'][0][0]
        else:
            y = comb_var.loc[:, [self.y, group]]
            x = comb_var.drop(columns=self.y)
            out = simple_panel_ols(y=y,
                                   x=x,
                                   group=group)
            return out['b'][0][0], out['p'][0][0], out['aic'][0][0], out['bic'][0][0]

    def _strap_est(self, comb_var, group, sample_size, replace):

        '''
        This method calls self._estimate() over a random sample
        of the data contaning y, x and controls. Returns a single
        value for each returning variable.

        Parameters
        ----------
        comb_var : Array
                ND array like object (pandas dataframe of numpy array)
                contaning the data for y, x, and controls.
        group : str
            Grouping variable. If provided a Fixed Effects model is estimated.


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

        if group is None:
            samp_df = comb_var.sample(n=sample_size, replace=replace)
            # @TODO generalize the frac to the function call
            y = samp_df.loc[:, :self.y]
            x = samp_df.drop(columns=self.y)
            output = stripped_ols(y, x)
            b = output['b']
            p = output['p']
            return b[0][0], p[0][0]
        else:
            #samp_df = comb_var.groupby(group).sample(frac=0.3, replace=replace)
            # @TODO generalize the frac to the function call
            idx = np.random.choice(comb_var[group].unique(), sample_size)
            select = comb_var[comb_var[group].isin(idx)]
            no_singleton = select[select.groupby(group).transform('size') > 1]
            y = no_singleton.loc[:, [self.y, group]]
            x = no_singleton.drop(columns=self.y)
            output = stripped_panel_ols(y, x, group)
            b = output['b']
            p = output['p']
            return b[0][0], p[0][0]
