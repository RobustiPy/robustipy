from nrobust.prototypes import Protomodel
from nrobust.prototypes import Protoresult
import pandas as pd
import numpy as np
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
from nrobust.prototypes import MissingValueWarning
import warnings


class OLSResult(Protoresult):
    def __init__(self, *,
                 y,
                 specs,
                 estimates,
                 p_values,
                 aic_array,
                 bic_array,
                 hqic_array):
        super().__init__()
        self.y_name = y
        self.specs_names = pd.Series(specs)
        self.estimates = pd.DataFrame(estimates)
        self.p_values = pd.DataFrame(p_values)
        self.summary_df = compute_summary(self.estimates)
        self.summary_df['aic'] = pd.Series(aic_array)
        self.summary_df['bic'] = pd.Series(bic_array)
        self.summary_df['hqic'] = pd.Series(hqic_array)
        self.summary_df['spec_name'] = self.specs_names
        self.summary_df['y'] = self.y_name

    def summary(self):
        pass

    def plot(self,
             specs=None,
             ic=None,
             colormap=None,
             colorset=None,
             figsize=(12, 6)):
        return plot_results(results_object=self,
                            specs=specs,
                            ic=ic,
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
     List of names of the independent variable(s).
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
            warnings.warn('Missing values found in data. Listwise deletion will be applied',
                          MissingValueWarning)
        self.y = y
        self.x = x
        self.data = data
        self.results = None

    def get_results(self):
        return self.results

    def multiple_y(self):
        self.y_specs = []
        self.y_composites = []
        print("Calculating Composite Ys")
        for spec, index in zip(all_subsets(self.y),
                               tqdm(range(0, space_size(self.y)))):
            if len(spec) > 1:
                subset = self.data[list(spec)]
                subset = (subset-subset.mean())/subset.std()
                self.y_composites.append(subset.mean(axis=1))
                self.y_specs.append(spec)

    def fit(self,
            *,
            controls,
            info='bic',
            group: str = None,
            bootstrap=True,
            draws=500,
            sample_size=None,
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

        if sample_size is None:
            sample_size = self.data.shape[0]
            
        if len(self.y) > 1:
            self.multiple_y()
            list_b_array = []
            list_p_array = []
            list_aic_array = []
            list_bic_array = []
            list_hqic_array = []
            y_names = []
            specs = []
            for y, y_name in zip(self.y_composites,
                                 self.y_specs):
                space_n = space_size(controls)
                b_array = np.empty([space_n, draws])
                p_array = np.empty([space_n, draws])
                aic_array = np.empty([space_n])
                bic_array = np.empty([space_n])
                hqic_array = np.empty([space_n])

                for spec, index in zip(all_subsets(controls),
                                       tqdm(range(0, space_n))):
                    if len(spec) == 0:
                        comb = self.data[self.x]
                    else:
                        comb = self.data[self.x + list(spec)]
                    if group:
                        comb = self.data[self.x + [group] + list(spec)]

                    comb = pd.concat([y, comb], axis=1)
                    comb = comb.dropna()
                    (b_discard, p_discard,
                     aic_i, bic_i, hqic_i) = self._full_est(comb,
                                                            group)
                    b_list, p_list = (zip(*Parallel(n_jobs=-1)
                                          (delayed(self._strap_est)
                                           (comb,
                                            group,
                                            sample_size,
                                            replace)
                                           for i in range(0,
                                                          draws))))
                    y_names.append(y_name)
                    specs.append(frozenset(spec))
                    b_array[index, :] = b_list
                    p_array[index, :] = p_list
                    aic_array[index] = aic_i
                    bic_array[index] = bic_i
                    hqic_array[index] = hqic_i

                list_b_array.append(b_array)
                list_p_array.append(p_array)
                list_aic_array.append(aic_array)
                list_bic_array.append(bic_array)
                list_hqic_array.append(hqic_array)

            results = OLSResult(y=y_names,
                                specs=specs,
                                estimates=np.vstack(list_b_array),
                                p_values=np.vstack(list_p_array),
                                aic_array=np.hstack(list_aic_array),
                                bic_array=np.hstack(list_bic_array),
                                hqic_array=np.hstack(list_hqic_array))

            self.results = results

        else:
            space_n = space_size(controls)
            specs = []
            b_array = np.empty([space_n, draws])
            p_array = np.empty([space_n, draws])
            aic_array = np.empty([space_n])
            bic_array = np.empty([space_n])
            hqic_array = np.empty([space_n])
            for spec, index in zip(all_subsets(controls),
                                   tqdm(range(0, space_n))):
                if len(spec) == 0:
                    comb = self.data[self.y + self.x]
                else:
                    comb = self.data[self.y + self.x + list(spec)]
                if group:
                    comb = self.data[self.y + self.x + [group] + list(spec)]

                comb = comb.dropna()
                (b_discard, p_discard,
                 aic_i, bic_i, hqic_i) = self._full_est(comb,
                                                        group)
                b_list, p_list = (zip(*Parallel(n_jobs=-1)
                                      (delayed(self._strap_est)
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
                hqic_array[index] = hqic_i

            results = OLSResult(y=self.y[0],
                                specs=specs,
                                estimates=b_array,
                                p_values=p_array,
                                aic_array=aic_array,
                                bic_array=bic_array,
                                hqic_array=hqic_array)

            self.results = results

    def _full_est(self, comb_var, group):
        if group is None:
            y = comb_var.iloc[:, [0]]
            x = comb_var.drop(comb_var.columns[0], axis=1)
            out = simple_ols(y=y,
                             x=x)
            return (out['b'][0][0],
                    out['p'][0][0],
                    out['aic'][0][0],
                    out['bic'][0][0],
                    out['hqic'][0][0])
        else:
            g = comb_var.loc[:, [group]]
            y = comb_var.iloc[:, [0]]
            y_g = pd.concat([y, g], axis=1)
            x_g = comb_var.drop(comb_var.columns[0], axis=1)
            out = simple_panel_ols(y=y_g,
                                   x=x_g,
                                   group=group)
            return (out['b'][0][0],
                    out['p'][0][0],
                    out['aic'][0][0],
                    out['bic'][0][0],
                    out['hqic'][0][0])

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
            y = samp_df.iloc[:, [0]]
            x = samp_df.drop(samp_df.columns[0], axis=1)
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
            g = no_singleton.loc[:, [group]]
            y = no_singleton.iloc[:, [0]]
            y_g = pd.concat([y, g], axis=1)
            x_g = no_singleton.drop(no_singleton.columns[0], axis=1)
            output = stripped_panel_ols(y_g, x_g, group)
            b = output['b']
            p = output['p']
            return b[0][0], p[0][0]
