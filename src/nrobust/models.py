from nrobust.prototypes import Protomodel
from nrobust.prototypes import Protoresult
import pandas as pd
import numpy as np
from tqdm import tqdm
from joblib import Parallel, delayed
from nrobust.utils import simple_ols
from nrobust.bootstrap_utils import stripped_ols
from nrobust.utils import space_size
from nrobust.utils import all_subsets
from nrobust.figures import plot_results
from nrobust.utils import group_demean
from nrobust.prototypes import MissingValueWarning
import _pickle
import warnings


class OLSResult(Protoresult):
    def __init__(self, *,
                 y,
                 specs,
                 all_predictors,
                 controls,
                 draws,
                 estimates,
                 all_b,
                 all_p,
                 p_values,
                 ll_array,
                 aic_array,
                 bic_array,
                 hqic_array):
        super().__init__()
        self.y_name = y
        self.specs_names = pd.Series(specs)
        self.all_predictors = all_predictors
        self.controls = controls
        self.draws = draws
        self.estimates = pd.DataFrame(estimates)
        self.p_values = pd.DataFrame(p_values)
        self.all_b = all_b
        self.all_p = all_p
        self.summary_df = self._compute_summary()
        self.summary_df['ll'] = pd.Series(ll_array)
        self.summary_df['aic'] = pd.Series(aic_array)
        self.summary_df['bic'] = pd.Series(bic_array)
        self.summary_df['hqic'] = pd.Series(hqic_array)
        self.summary_df['spec_name'] = self.specs_names
        self.summary_df['y'] = self.y_name

    def save(self, filename):
        with open(filename, 'wb') as f:
            _pickle.dump(self, f, -1)

    @classmethod
    def load(cls, filename):
        with open(filename, 'rb') as f:
            return _pickle.load(f)

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

    def _compute_summary(self):
        data = self.estimates.copy()
        out = pd.DataFrame()
        out['median'] = data.median(axis=1)
        out['max'] = data.max(axis=1)
        out['min'] = data.min(axis=1)
        out['ci_up'] = data.quantile(q=0.975, axis=1, interpolation='nearest')
        out['ci_down'] = data.quantile(q=0.025, axis=1, interpolation='nearest')
        return out

    def compute_bma(self):
        """
        Bayesian model averaging using BIC implied priors
        """
        likelihood_per_var = []
        weigthed_coefs = []
        max_ll = np.max(-self.summary_df.bic/2)
        shifted_ll = (-self.summary_df.bic/2) - max_ll
        models_likelihood = np.exp(shifted_ll)
        sum_likelihoods = np.nansum(models_likelihood)
        coefs = [[i[0] for i in x] for x in self.all_b]
        coefs = [i for sl in coefs for i in sl]
        var_names = [i for sl in self.all_predictors for i in sl]
        coefs_df = pd.DataFrame({'coef': coefs, 'var_name': var_names})
        for ele in self.controls:
            idx = []
            for spec in self.specs_names:
                idx.append(ele in spec)
            likelihood_per_var.append(np.nansum(models_likelihood[idx]))
            coefs = coefs_df[coefs_df.var_name == ele].coef.to_numpy()
            likelihood = models_likelihood[idx]
            weigthed_coef = coefs * likelihood
            weigthed_coefs.append(np.nansum(weigthed_coef))
        probs = likelihood_per_var / sum_likelihoods
        final_coefs = weigthed_coefs / sum_likelihoods
        summary_bma = pd.DataFrame({
            'control_var': self.controls,
            'probs': probs,
            'average_coefs': final_coefs
        })
        return summary_bma


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
            group: str = None,
            draws=500,
            sample_size=None,
            replace=False):

        """
        Fit the OLS models into the specification space
        as well as over the bootstrapped samples.

        Parameters
        ----------
        controls : list<str>
                List containing all the names of the  possible
                control variables of the model.
        sample_size : int
              Number of bootstrap samples to collect.
        group : str
            Grouping variable. If provided a Fixed Effects model is estimated.

        Returns
        -------
        self : Object
             Object class OLSRobust containing the fitted estimators.
        """

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

                    if group:
                        comb = group_demean(comb, group=group)
                    (b_all, p_all,
                     aic_i, bic_i, hqic_i) = self._full_sample_OLS(comb)
                    b_list, p_list = (zip(*Parallel(n_jobs=-1)
                                          (delayed(self._strap_OLS)
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
                                draws=draws,
                                all_b=b_all,
                                all_p=p_all,
                                estimates=np.vstack(list_b_array),
                                p_values=np.vstack(list_p_array),
                                aic_array=np.hstack(list_aic_array),
                                bic_array=np.hstack(list_bic_array),
                                hqic_array=np.hstack(list_hqic_array))

            self.results = results

        else:
            space_n = space_size(controls)
            specs = []
            all_predictors = []
            b_all_list = []
            p_all_list = []
            b_array = np.empty([space_n, draws])
            p_array = np.empty([space_n, draws])
            ll_array = np.empty([space_n])
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

                if group:
                    comb = group_demean(comb, group=group)
                (b_all, p_all, ll_i,
                 aic_i, bic_i, hqic_i) = self._full_sample_OLS(comb)
                b_list, p_list = (zip(*Parallel(n_jobs=-1)
                                      (delayed(self._strap_OLS)
                                       (comb,
                                        group,
                                        sample_size,
                                        replace)
                                       for i in range(0,
                                                      draws))))

                specs.append(frozenset(spec))
                all_predictors.append(self.x + list(spec) + ['const'])
                b_array[index, :] = b_list
                p_array[index, :] = p_list
                ll_array[index] = ll_i
                aic_array[index] = aic_i
                bic_array[index] = bic_i
                hqic_array[index] = hqic_i
                b_all_list.append(b_all)
                p_all_list.append(p_all)

            results = OLSResult(y=self.y[0],
                                specs=specs,
                                all_predictors=all_predictors,
                                controls=controls,
                                draws=draws,
                                all_b=b_all_list,
                                all_p=p_all_list,
                                estimates=b_array,
                                p_values=p_array,
                                ll_array=ll_array,
                                aic_array=aic_array,
                                bic_array=bic_array,
                                hqic_array=hqic_array)

            self.results = results

    def _full_sample_OLS(self, comb_var):
        """
        This method calls stripped_ols()
        over the full data contaning y, x and controls.
        Returns a single value for each returning variable.

        Parameters
        ----------
        comb_var : Array
                ND array like object (pandas dataframe of numpy array)
                contaning the data for y, x, and controls.

        Returns
        -------
        beta : float
            Estimate for x.
        p : float
         P value for x.
        AIC : float
          Akaike information criteria value for the model.
        BIC : float
          Bayesian information criteria value for the model.
        HQIC : float
          Hannan-Quinn information criteria value for the model.
        """
        y = comb_var.iloc[:, [0]]
        x = comb_var.drop(comb_var.columns[0], axis=1)
        out = simple_ols(y=y,
                         x=x)
        return (out['b'],
                out['p'],
                out['ll'][0][0],
                out['aic'][0][0],
                out['bic'][0][0],
                out['hqic'][0][0])

    def _strap_OLS(self, comb_var, group, sample_size, replace):

        """
        This method calls stripped_ols() over a random sample
        of the data contaning y, x and controls.
        Returns a single value for each returning variable.

        Parameters
        ----------
        comb_var : Array
                ND array like object (pandas dataframe of numpy array)
                contaning the data for y, x, and controls.
        group : str
            Grouping variable. If provided sampling is performed over
            the group variable.
        sample_size : int
                  Optional: Sample size to use in the bootstrap. If not
                  provided, sample size is obtained from the length
                  of the self.data.
        replace : bool
              Whether to use replace on sampling.

        Returns
        -------
        beta : float
            Estimate for x.
        p : float
         P value for x.
        """

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
            # samp_df = comb_var.groupby(group).sample(frac=0.3,
            # replace=replace)
            # @TODO generalize the frac to the function call

            idx = np.random.choice(comb_var[group].unique(), sample_size)
            select = comb_var[comb_var[group].isin(idx)]
            no_singleton = select[select.groupby(group).transform('size') > 1]
            no_singleton = no_singleton.drop(columns=[group])
            y = no_singleton.iloc[:, [0]]
            x = no_singleton.drop(no_singleton.columns[0], axis=1)
            output = stripped_ols(y, x)
            b = output['b']
            p = output['p']
            return b[0][0], p[0][0]
