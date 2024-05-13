from robustipy.prototypes import Protomodel
from robustipy.prototypes import Protoresult
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from rich.progress import track
from joblib import Parallel, delayed
from robustipy.utils import simple_ols, logistic_regression_sm, logistic_regression_sm_stripped,logistic_regression_sk,logistic_regression_sk_stripped
from robustipy.bootstrap_utils import stripped_ols
from robustipy.utils import space_size
from robustipy.utils import all_subsets
from robustipy.figures import plot_results, plot_curve
from robustipy.utils import group_demean
from robustipy.prototypes import MissingValueWarning
import _pickle
import warnings
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error


class MergedResult(Protoresult):
    def __init__(self, *,
                 y,
                 specs,
                 estimates,
                 p_values, ):
        super().__init__()
        self.y_name = y
        self.specs_names = pd.Series(specs)
        self.estimates = pd.DataFrame(estimates)
        self.p_values = pd.DataFrame(p_values)
        self.summary_df = self._compute_summary()
        self.summary_df['spec_name'] = self.specs_names

    def summary(self):
        """
        Generates a summary of the regression results (not implemented).
        """
        pass

    def _compute_summary(self):
        """
        Computes summary statistics based on coefficient estimates.

        Returns:
            pd.DataFrame: DataFrame containing summary statistics.
        """
        data = self.estimates.copy()
        out = pd.DataFrame()
        out['median'] = data.median(axis=1)
        out['max'] = data.max(axis=1)
        out['min'] = data.min(axis=1)
        out['ci_up'] = data.quantile(q=0.975, axis=1,
                                     interpolation='nearest')
        out['ci_down'] = data.quantile(q=0.025, axis=1,
                                       interpolation='nearest')
        return out

    def plot(self,
             specs=None,
             colormap=None,
             colorset=None,
             figsize=(12, 6)):

        fig, ax = plt.subplots(figsize=figsize)

        if specs is not None:
            if not all(isinstance(l, list) for l in specs):
                raise TypeError("'specs' must be a list of lists.")

            if not all(frozenset(spec) in self.specs_names.to_list() for spec in specs):
                raise TypeError("All specifications in 'spec' must be in the valid computed specifications.")

        plot_curve(results_object=self,
                   specs=specs,
                   ax=ax,
                   colormap=colormap,
                   colorset=colorset)
        return fig

    def merge(self, result_obj, left_prefix, right_prefix):
        """
        Merges two OLSResult objects into one.

        Args:
            result_obj (OLSResult): OLSResult object to be merged.
            left_prefix (str): Prefix for the orignal result object.
            right_prefix (str): Prefix fort the new result object.

        Raises:
            TypeError: If the input object is not an instance of OLSResult.
        """
        if not isinstance(result_obj, OLSResult):
            raise TypeError("'result_obj' must be an instance of OLSResult.")

        if not isinstance(left_prefix, str) or not isinstance(right_prefix, str):
            raise TypeError("'prefixes' must be of type 'str.'")

        if self.y_name != result_obj.y_name:
            raise ValueError('Dependent variable names must match.')

        specs_original = [frozenset(list(s) + [left_prefix]) for s in self.specs_names]
        specs_new = [frozenset(list(s) + [right_prefix]) for s in result_obj.specs_names]
        y = self.y_name
        specs = specs_original + specs_new
        estimates = pd.concat([self.estimates, result_obj.estimates], ignore_index=True)
        p_values = pd.concat([self.p_values, result_obj.p_values], ignore_index=True)

        return MergedResult(
            y=y,
            specs=specs,
            estimates=estimates,
            p_values=p_values
        )


class OLSResult(Protoresult):
    """
    Result class containing the output of the OLSRobust class.

    Parameters:
        y (str): The name of the dependent variable.
        specs (list of str): List of specification names.
        all_predictors (list of lists of str): List of predictor variable names for each specification.
        controls (list of str): List of control variable names.
        draws (int): Number of draws in the analysis.
        estimates (pd.DataFrame): DataFrame containing regression coefficient estimates.
        all_b (list of lists): List of coefficient estimates for each specification and draw.
        all_p (list of lists): List of p-values for each specification and draw.
        p_values (pd.DataFrame): DataFrame containing p-values for coefficient estimates.
        ll_array (list): List of log-likelihood values for each specification.
        aic_array (list): List of AIC values for each specification.
        bic_array (list): List of BIC values for each specification.
        hqic_array (list): List of HQIC values for each specification.
        av_k_metric_array (list, optional): List of average Kullback-Leibler divergence metrics.

    Methods:
        save(filename):
            Save the OLSResult object to a file using pickle.

        load(filename):
            Load an OLSResult object from a file using pickle.

        summary():
            Placeholder for a method to generate a summary of the OLS results.

        plot(specs=None, ic=None, colormap=None, colorset=None, figsize=(12, 6)):
            Generate plots of the OLS results.

        compute_bma():
            Perform Bayesian model averaging using BIC implied priors and return the results.

        merge(result_obj, prefix):
            Merge two OLSResult objects into one.

    Attributes:
        y_name (str): The name of the dependent variable.
        specs_names (pd.Series): Series containing specification names.
        all_predictors (list of lists of str): List of predictor variable names for each specification.
        controls (list of str): List of control variable names.
        draws (int): Number of draws in the analysis.
        estimates (pd.DataFrame): DataFrame containing regression coefficient estimates.
        p_values (pd.DataFrame): DataFrame containing p-values for coefficient estimates.
        all_b (list of lists): List of coefficient estimates for each specification and draw.
        all_p (list of lists): List of p-values for each specification and draw.
        summary_df (pd.DataFrame): DataFrame containing summary statistics of coefficient estimates.
        summary_bma (pd.DataFrame, optional): DataFrame containing Bayesian model averaging results.
    """

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
                 hqic_array,
                 av_k_metric_array=None):
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
        self.summary_df['av_k_metric'] = pd.Series(av_k_metric_array)
        self.summary_df['spec_name'] = self.specs_names
        self.summary_df['y'] = self.y_name

    def save(self, filename):
        """
        Saves the OLSResult object to a binary file.

        Args:
            filename (str): Name of the file to which the object will be saved.
        """
        with open(filename, 'wb') as f:
            _pickle.dump(self, f, -1)

    @classmethod
    def load(cls, filename):
        """
        Loads an OLSResult object from a binary file.

        Args:
            filename (str): Name of the file from which the object will be loaded.

        Returns:
            OLSResult: Loaded OLSResult object.
        """
        with open(filename, 'rb') as f:
            return _pickle.load(f)

    def summary(self):
        """
        Generates a summary of the regression results (not implemented).
        """
        pass

    def plot(self,
             specs=None,
             ic='aic',
             colormap=None,
             colorset=None,
             figsize=(12, 6)):
        """
        Plots the regression results using specified options.

        Args:
            specs (list, optional): List of list of specification names to include in the plot.
            ic (str, optional): Information criterion to use for model selection (e.g., 'bic', 'aic').
            colormap (str, optional): Colormap to use for the plot.
            colorset (list, optional): List of colors to use for different specifications.
            figsize (tuple, optional): Size of the figure (width, height) in inches.

        Returns:
            matplotlib.figure.Figure: Plot showing the regression results.
        """

        valid_ic = ['bic', 'aic', 'hqic']

        if specs is not None:
            if not all(isinstance(l, list) for l in specs):
                raise TypeError("'specs' must be a list of lists.")

            if not all(frozenset(spec) in self.specs_names.to_list() for spec in specs):
                raise TypeError("All specifications in 'spec' must be in the valid computed specifications.")

        if ic not in valid_ic:
            raise ValueError(f"'ic' must be one of the following: {valid_ic}")

        return plot_results(results_object=self,
                            specs=specs,
                            ic=ic,
                            colormap=colormap,
                            colorset=colorset,
                            figsize=figsize)

    def _compute_summary(self):
        """
        Computes summary statistics based on coefficient estimates.

        Returns:
            pd.DataFrame: DataFrame containing summary statistics.
        """
        data = self.estimates.copy()
        out = pd.DataFrame()
        out['median'] = data.median(axis=1)
        out['max'] = data.max(axis=1)
        out['min'] = data.min(axis=1)
        out['ci_up'] = data.quantile(q=0.975, axis=1,
                                     interpolation='nearest')
        out['ci_down'] = data.quantile(q=0.025, axis=1,
                                       interpolation='nearest')
        return out

    def compute_bma(self):
        """
        Performs Bayesian Model Averaging (BMA) using BIC implied priors.

        Returns:
            pd.DataFrame: DataFrame containing BMA results.
        """
        likelihood_per_var = []
        weigthed_coefs = []
        max_ll = np.max(-self.summary_df.bic / 2)
        shifted_ll = (-self.summary_df.bic / 2) - max_ll
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

    def merge(self, result_obj, left_prefix, right_prefix) -> MergedResult:
        """
        Merges two OLSResult objects into one.

        Args:
            result_obj (OLSResult): OLSResult object to be merged.
            left_prefix (str): Prefix for the orignal result object.
            right_prefix (str): Prefix fort the new result object.

        Raises:
            TypeError: If the input object is not an instance of OLSResult.
        """
        if not isinstance(result_obj, OLSResult):
            raise TypeError("'result_obj' must be an instance of OLSResult.")

        if not isinstance(left_prefix, str) or not isinstance(right_prefix, str):
            raise TypeError("'prefixes' must be of type 'str.'")

        if self.y_name != result_obj.y_name:
            raise ValueError('Dependent variable names must match.')

        specs_original = [frozenset(list(s) + [left_prefix]) for s in self.specs_names]
        specs_new = [frozenset(list(s) + [right_prefix]) for s in result_obj.specs_names]
        y = self.y_name
        specs = specs_original + specs_new
        estimates = pd.concat([self.estimates, result_obj.estimates], ignore_index=True)
        p_values = pd.concat([self.p_values, result_obj.p_values], ignore_index=True)

        return MergedResult(
            y=y,
            specs=specs,
            estimates=estimates,
            p_values=p_values
        )

    def save_to_csv(self, path: str):
        """
        Function to save summary dataframe to a csv
        """
        self.summary_df.to_csv(path)


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
        """
        Initialize the OLSRobust object.

        Parameters
        ----------
        y : list<str>
            Name of the dependent variable.
        x : list<str>
            List of names of the independent variable(s).
        data : DataFrame
            DataFrame containing all the data to be used in the model.
        """
        super().__init__()
        if not isinstance(y, list) or not isinstance(x, list):
            raise TypeError("'y' and 'x' must be lists.")

        if not isinstance(data, pd.DataFrame):
            raise TypeError("'data' must be a pandas DataFrame.")

        all_vars = set(data.columns)
        if not all(var in all_vars for var in y) or not all(var in all_vars for var in x):
            raise ValueError("Variable names in 'y' and 'x' must exist in the provided DataFrame 'data'.")

        if data.isnull().values.any():
            warnings.warn('Missing values found in data. Listwise deletion will be applied',
                          MissingValueWarning)
        self.y = y
        self.x = x
        self.data = data
        self.results = None

    def get_results(self):
        """
        Get the results of the OLS regression.

        Returns
        -------
        results : OLSResult
            Object containing the regression results.
        """
        return self.results

    def multiple_y(self):
        """
        Cumputes composite y based on multiple indicators provided.
        """
        self.y_specs = []
        self.y_composites = []
        print("Calculating Composite Ys")
        for spec, index in track(zip(all_subsets(self.y),
                                     range(0, space_size(self.y))), total=space_size(self.y)):
            if len(spec) > 0:
                subset = self.data[list(spec)]
                subset = (subset - subset.mean()) / subset.std()
                self.y_composites.append(subset.mean(axis=1))
                self.y_specs.append(spec)

    def fit(self,
            *,
            controls,
            group=None,
            draws=500,
            kfold=None,
            shuffle=False):
        """
        Fit the OLS models into the specification space as well as over the bootstrapped samples.

        Parameters
        ----------
        controls : list<str>
            List containing all the names of the possible control variables of the model.
        group : str
            Grouping variable. If provided, a Fixed Effects model is estimated.
        draws : int, optional
            Number of draws for bootstrapping. Default is 500.
        kfold : int, optional
            Number of folds for k-fold cross-validation. Default is None.
        shuffle : bool, optional
            Whether to shuffle y variable to estimate joint significance test. Default is False.

        Returns
        -------
        self : Object
            Object class OLSRobust containing the fitted estimators.
        """
        if not isinstance(controls, list):
            raise TypeError("'controls' must be a list.")

        all_vars = set(self.data.columns)
        if not all(var in all_vars for var in controls):
            raise ValueError("Variable names in 'controls' must exist in the provided DataFrame 'data'.")

        if group is not None:
            if not isinstance(group,str) or not group in all_vars:
                raise ValueError("'group' variable must exist in the provided DataFrame 'data'.")

        sample_size = self.data.shape[0]

        if len(self.y) > 1:
            self.multiple_y()
            list_all_predictors = []
            list_b_array = []
            list_p_array = []
            list_ll_array = []
            list_aic_array = []
            list_bic_array = []
            list_hqic_array = []
            list_av_k_metric_array = []
            y_names = []
            specs = []
            for y, y_name in zip(self.y_composites,
                                 self.y_specs):
                space_n = space_size(controls)
                b_array = np.empty([space_n, draws])
                p_array = np.empty([space_n, draws])
                ll_array = np.empty([space_n])
                aic_array = np.empty([space_n])
                bic_array = np.empty([space_n])
                hqic_array = np.empty([space_n])
                all_predictors = []
                av_k_metric_array = np.empty([space_n])

                for spec, index in track(zip(all_subsets(controls), range(0, space_n)), total=space_n):
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
                    (b_all, p_all, ll_i,
                     aic_i, bic_i, hqic_i,
                     av_k_metric_i) = self._full_sample_OLS(comb,
                                                            kfold=kfold)
                    b_list, p_list = (zip(*Parallel(n_jobs=-1)
                    (delayed(self._strap_OLS)
                     (comb,
                      group,
                      sample_size,
                      shuffle)
                     for i in range(0,
                                    draws))))
                    y_names.append(y_name)
                    specs.append(frozenset(list(y_name) + list(spec)))
                    all_predictors.append(self.x + list(spec) + ['const'])
                    b_array[index, :] = b_list
                    p_array[index, :] = p_list
                    ll_array[index] = ll_i
                    aic_array[index] = aic_i
                    bic_array[index] = bic_i
                    hqic_array[index] = hqic_i
                    av_k_metric_array[index] = av_k_metric_i

                list_all_predictors.append(all_predictors)
                list_b_array.append(b_array)
                list_p_array.append(p_array)
                list_ll_array.append(ll_array)
                list_aic_array.append(aic_array)
                list_bic_array.append(bic_array)
                list_hqic_array.append(hqic_array)
                list_av_k_metric_array.append(av_k_metric_array)

            results = OLSResult(
                y=y_names,
                specs=specs,
                all_predictors=list_all_predictors,
                controls=controls,
                draws=draws,
                all_b=b_all,
                all_p=p_all,
                estimates=np.vstack(list_b_array),
                p_values=np.vstack(list_p_array),
                ll_array=np.hstack(list_ll_array),
                aic_array=np.hstack(list_aic_array),
                bic_array=np.hstack(list_bic_array),
                hqic_array=np.hstack(list_hqic_array),
                av_k_metric_array=np.hstack(list_av_k_metric_array)
            )

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
            av_k_metric_array = np.empty([space_n])
            for spec, index in track(zip(all_subsets(controls), range(0, space_n)), total=space_n):
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
                 aic_i, bic_i, hqic_i,
                 av_k_metric_i) = self._full_sample_OLS(comb,
                                                        kfold=kfold)
                b_list, p_list = (zip(*Parallel(n_jobs=-1)
                (delayed(self._strap_OLS)
                 (comb,
                  group,
                  sample_size,
                  shuffle)
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
                av_k_metric_array[index] = av_k_metric_i
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
                                hqic_array=hqic_array,
                                av_k_metric_array=av_k_metric_array)

            self.results = results

    def _predict(self, x_test, betas):
        """
        Predict the dependent variable based on the test data and coefficients.

        Parameters
        ----------
        x_test : array-like
            Test data for independent variables.
        betas : array-like
            Coefficients obtained from the regression.

        Returns
        -------
        y_pred : array
            Predicted values for the dependent variable.
        """
        return np.dot(x_test, betas)

    def _full_sample_OLS(self,
                         comb_var,
                         kfold):
        """
        Call stripped_ols() over the full data containing y, x, and controls.

        Parameters
        ----------
        comb_var : Array
            ND array-like object containing the data for y, x, and controls.
        kfold : Boolean
            Whether or not to calculate k-fold cross-validation.

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
        av_k_metric = None
        if kfold:
            k_fold = KFold(kfold)
            metrics = []
            for k, (train, test) in enumerate(k_fold.split(x, y)):
                out_k = simple_ols(y=y.loc[train],
                                   x=x.loc[train])
                y_pred = self._predict(x.loc[test], out_k['b'])
                y_true = y.loc[test]
                k_rmse = mean_squared_error(y_true, y_pred, squared=False)
                metrics.append(k_rmse)
            av_k_metric = np.mean(metrics)
        return (out['b'],
                out['p'],
                out['ll'][0][0],
                out['aic'][0][0],
                out['bic'][0][0],
                out['hqic'][0][0],
                av_k_metric)

    def _strap_OLS(self,
                   comb_var,
                   group,
                   sample_size,
                   shuffle):
        """
        Call stripped_ols() over a random sample of the data containing y, x, and controls.

        Parameters
        ----------
        comb_var : Array
            ND array-like object containing the data for y, x, and controls.
        group : str
            Grouping variable. If provided, sampling is performed over the group variable.
        sample_size : int
            Sample size to use in the bootstrap.
        shuffle : bool
            Whether to shuffle y var to estimate joint significant test.

        Returns
        -------
        beta : float
            Estimate for x.
        p : float
            P value for x.
        """
        temp_data = comb_var.copy()

        if shuffle:
            y = temp_data.iloc[:, [0]]
            idx_y = np.random.permutation(y.index)
            y = pd.DataFrame(y.iloc[idx_y]).reset_index(drop=True)
            x = temp_data.drop(temp_data.columns[0], axis=1)
            temp_data = pd.concat([y, x], axis=1)

        if group is None:
            samp_df = temp_data.sample(n=sample_size, replace=True)
            # @TODO generalize the frac to the function call
            y = samp_df.iloc[:, [0]]
            x = samp_df.drop(samp_df.columns[0], axis=1)
            output = stripped_ols(y, x)
            b = output['b']
            p = output['p']
            return b[0][0], p[0][0]
        else:
            idx = np.random.choice(temp_data[group].unique(), sample_size)
            select = temp_data[temp_data[group].isin(idx)]
            no_singleton = select[select.groupby(group).transform('size') > 1]
            no_singleton = no_singleton.drop(columns=[group])
            y = no_singleton.iloc[:, [0]]
            x = no_singleton.drop(no_singleton.columns[0], axis=1)
            output = stripped_ols(y, x)
            b = output['b']
            p = output['p']
            return b[0][0], p[0][0]


class LRobust_sm(Protomodel):
    """
    A class to perform robust logistic regression analysis.

    Parameters
    ----------
    y : array-like
        Dependent variable values.
    x : array-like
        Independent variable values. The matrix should be shaped as
        (number of observations, number of independent variables).
    data : DataFrame
        A pandas DataFrame containing the variables in the model.

    Attributes
    ----------
    y : array-like
        Dependent variable values.
    x : array-like
        Independent variable values.
    data : DataFrame
        A pandas DataFrame containing the variables in the model.
    results : dict
        A dictionary containing regression coefficients ('b') and corresponding
        p-values ('p') for each independent variable.
    """

    def __init__(self, *, y, x, data, model_name='LR_sm'):  # same as OLSRobust
        """
        Initialize the LRobust object.

        Parameters
        ----------
        y : str
            Name of the dependent variable.
        x : str or list<str>
            List of names of the independent variable(s).
        data : DataFrame
            DataFrame containing all the data to be used in the model.
        """
        super().__init__()
        if not isinstance(y, list) or not isinstance(x, list):
            raise TypeError("'y' and 'x' must be lists.")

        if not isinstance(data, pd.DataFrame):
            raise TypeError("'data' must be a pandas DataFrame.")

        all_vars = set(data.columns)
        if not all(var in all_vars for var in y) or not all(var in all_vars for var in x):
            raise ValueError("Variable names in 'y' and 'x' must exist in the provided DataFrame 'data'.")

        if data.isnull().values.any():
            warnings.warn('Missing values found in data. Listwise deletion will be applied',
                          MissingValueWarning)
        self.y = y
        self.x = x
        self.data = data
        self.results = None  # same as OLO
        self.model_name = model_name

    def get_results(self):
        """
        Get the results of the OLS regression.

        Returns
        -------
        results : OLSResult
            Object containing the regression results.
        """
        return self.results

    def multiple_y(self):
        raise NotImplementedError("Not implemented yet")

    def _full_sample(self, comb_var, kfold):
        """
        Call stripped_ols() over the full data containing y, x, and controls.

        Parameters
        ----------
        comb_var : Array
            ND array-like object containing the data for y, x, and controls.
        kfold : Boolean
            Whether or not to calculate k-fold cross-validation.

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

        out = logistic_regression_sm(y=y, x=x)
        av_k_metric = None
        if kfold:
            k_fold = KFold(kfold)
            metrics = []
            for k, (train, test) in enumerate(k_fold.split(x, y)):
                out_k = logistic_regression_sm(y=y.loc[train], x=x.loc[train])
                y_pred = self._predict_LR(x.loc[test], out_k['b'])
                y_true = y.loc[test]
                k_rmse = mean_squared_error(y_true, y_pred, squared=False)
                metrics.append(k_rmse)
            av_k_metric = np.mean(metrics)
        return (out['b'],
                out['p'],
                out['ll'],  # TODO: check is this correct?
                out['aic'],
                out['bic'],
                out['hqic'],
                av_k_metric)

    def _predict_LR(self, x_test, betas):
        """
        Predict the dependent variable using the estimated coefficients.
        """
        return 1 / (1 + np.exp(-x_test.dot(betas)))
    def fit(self,
            *,
            controls,
            group=None,
            draws=500,
            sample_size=None,
            kfold=None,
            shuffle=False):
        if not isinstance(controls, list):
            raise TypeError("'controls' must be a list.")

        all_vars = set(self.data.columns)
        if not all(var in all_vars for var in controls):
            raise ValueError("Variable names in 'controls' must exist in the provided DataFrame 'data'.")

        if group is not None:
            if not group in all_vars:
                raise ValueError("'group' variable must exist in the provided DataFrame 'data'.")

        if sample_size is None:
            sample_size = self.data.shape[0]
        if len(self.y) > 1:
            raise NotImplementedError("Not implemented yet for logistic regression")
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
            av_k_metric_array = np.empty([space_n])

            for spec, index in track(zip(all_subsets(controls), range(0, space_n)), total=space_n):

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
                 aic_i, bic_i, hqic_i,
                 av_k_metric_i) = self._full_sample(comb, kfold=kfold)

                b_list, p_list = (zip(*Parallel(n_jobs=-1)
                (delayed(self._strap_regression)
                 (comb,
                  group,
                  sample_size,
                  shuffle)
                 for i in range(0, draws))))

                specs.append(frozenset(spec))
                all_predictors.append(self.x + list(spec) + ['const'])
                b_array[index, :] = b_list
                p_array[index, :] = p_list
                ll_array[index] = ll_i
                aic_array[index] = aic_i
                bic_array[index] = bic_i
                hqic_array[index] = hqic_i
                av_k_metric_array[index] = av_k_metric_i
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
                                       hqic_array=hqic_array,
                                       av_k_metric_array=av_k_metric_array)
            self.results = results

    def _strap_regression(self,
                          comb_var,
                          group,
                          sample_size,
                          shuffle):
        temp_data = comb_var.copy()

        if shuffle:
            y = temp_data.iloc[:, [0]]
            idx_y = np.random.permutation(y.index)
            y = pd.DataFrame(y.iloc[idx_y]).reset_index(drop=True)
            x = temp_data.drop(temp_data.columns[0], axis=1)
            temp_data = pd.concat([y, x], axis=1)

        if group is None:
            samp_df = temp_data.sample(n=sample_size, replace=True)
            y = samp_df.iloc[:, [0]]
            x = samp_df.drop(samp_df.columns[0], axis=1)
            output = logistic_regression_sm_stripped(y, x)
            return output['b'][0][0], output['p'][0][0]
        else:
            idx = np.random.choice(temp_data[group].unique(), sample_size)
            select = temp_data[temp_data[group].isin(idx)]
            no_singleton = select[select.groupby(group).transform('size') > 1]
            no_singleton = no_singleton.drop(columns=[group])
            y = no_singleton.iloc[:, [0]]
            x = no_singleton.drop(no_singleton.columns[0], axis=1)
            output = logistic_regression_sm(y, x)
            return output['b'][0][0], output['p'][0][0]


class LRobust_sklearn(Protomodel):
    """
    A class to perform robust logistic regression analysis.

    Parameters
    ----------
    y : array-like
        Dependent variable values.
    x : array-like
        Independent variable values. The matrix should be shaped as
        (number of observations, number of independent variables).
    data : DataFrame
        A pandas DataFrame containing the variables in the model.

    Attributes
    ----------
    y : array-like
        Dependent variable values.
    x : array-like
        Independent variable values.
    data : DataFrame
        A pandas DataFrame containing the variables in the model.
    results : dict
        A dictionary containing regression coefficients ('b') and corresponding
        p-values ('p') for each independent variable.
    """

    def __init__(self, *, y, x, data, model_name='LR_sm'):  # same as OLSRobust
        """
        Initialize the LRobust object.

        Parameters
        ----------
        y : str
            Name of the dependent variable.
        x : str or list<str>
            List of names of the independent variable(s).
        data : DataFrame
            DataFrame containing all the data to be used in the model.
        """
        super().__init__()
        if not isinstance(y, list) or not isinstance(x, list):
            raise TypeError("'y' and 'x' must be lists.")

        if not isinstance(data, pd.DataFrame):
            raise TypeError("'data' must be a pandas DataFrame.")

        all_vars = set(data.columns)
        if not all(var in all_vars for var in y) or not all(var in all_vars for var in x):
            raise ValueError("Variable names in 'y' and 'x' must exist in the provided DataFrame 'data'.")

        if data.isnull().values.any():
            warnings.warn('Missing values found in data. Listwise deletion will be applied',
                          MissingValueWarning)
        self.y = y
        self.x = x
        self.data = data
        self.results = None  # same as OLO
        self.model_name = model_name

    def get_results(self):
        """
        Get the results of the OLS regression.

        Returns
        -------
        results : OLSResult
            Object containing the regression results.
        """
        return self.results

    def multiple_y(self):
        raise NotImplementedError("Not implemented yet")

    def _full_sample(self, comb_var, kfold):
        """
        Call stripped_ols() over the full data containing y, x, and controls.

        Parameters
        ----------
        comb_var : Array
            ND array-like object containing the data for y, x, and controls.
        kfold : Boolean
            Whether or not to calculate k-fold cross-validation.

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

        out = logistic_regression_sk(y=y, x=x)
        av_k_metric = None
        if kfold:
            k_fold = KFold(kfold)
            metrics = []
            for k, (train, test) in enumerate(k_fold.split(x, y)):
                out_k = logistic_regression_sk(y=y.loc[train], x=x.loc[train])
                y_pred = self._predict_LR(x.loc[test], out_k['b'])
                y_true = y.loc[test]
                k_rmse = mean_squared_error(y_true, y_pred, squared=False)
                metrics.append(k_rmse)
            av_k_metric = np.mean(metrics)
        return (out['b'],
                out['p'],
                out['ll'],  # TODO: check is this correct?
                out['aic'],
                out['bic'],
                out['hqic'],
                av_k_metric)

    def _predict_LR(self, x_test, betas):
        """
        Predict the dependent variable using the estimated coefficients.
        """
        return 1 / (1 + np.exp(-x_test.dot(betas)))
    def fit(self,
            *,
            controls,
            group=None,
            draws=500,
            sample_size=None,
            kfold=None,
            shuffle=False):
        if not isinstance(controls, list):
            raise TypeError("'controls' must be a list.")

        all_vars = set(self.data.columns)
        if not all(var in all_vars for var in controls):
            raise ValueError("Variable names in 'controls' must exist in the provided DataFrame 'data'.")

        if group is not None:
            if not group in all_vars:
                raise ValueError("'group' variable must exist in the provided DataFrame 'data'.")

        if sample_size is None:
            sample_size = self.data.shape[0]
        if len(self.y) > 1:
            raise NotImplementedError("Not implemented yet for logistic regression")
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
            av_k_metric_array = np.empty([space_n])

            for spec, index in track(zip(all_subsets(controls), range(0, space_n)), total=space_n):

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
                 aic_i, bic_i, hqic_i,
                 av_k_metric_i) = self._full_sample(comb, kfold=kfold)

                b_list, p_list = (zip(*Parallel(n_jobs=-1)
                (delayed(self._strap_regression)
                 (comb,
                  group,
                  sample_size,
                  shuffle)
                 for i in range(0, draws))))

                specs.append(frozenset(spec))
                all_predictors.append(self.x + list(spec) + ['const'])
                b_array[index, :] = b_list
                p_array[index, :] = p_list
                ll_array[index] = ll_i
                aic_array[index] = aic_i
                bic_array[index] = bic_i
                hqic_array[index] = hqic_i
                av_k_metric_array[index] = av_k_metric_i
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
                                       hqic_array=hqic_array,
                                       av_k_metric_array=av_k_metric_array)
            self.results = results

    def _strap_regression(self,
                          comb_var,
                          group,
                          sample_size,
                          shuffle):
        temp_data = comb_var.copy()

        if shuffle:
            y = temp_data.iloc[:, [0]]
            idx_y = np.random.permutation(y.index)
            y = pd.DataFrame(y.iloc[idx_y]).reset_index(drop=True)
            x = temp_data.drop(temp_data.columns[0], axis=1)
            temp_data = pd.concat([y, x], axis=1)

        if group is None:
            samp_df = temp_data.sample(n=sample_size, replace=True)
            y = samp_df.iloc[:, [0]]
            x = samp_df.drop(samp_df.columns[0], axis=1)
            output = logistic_regression_sk_stripped(y, x)
            return output['b'][0][0], output['p'][0][0]
        else:
            idx = np.random.choice(temp_data[group].unique(), sample_size)
            select = temp_data[temp_data[group].isin(idx)]
            no_singleton = select[select.groupby(group).transform('size') > 1]
            no_singleton = no_singleton.drop(columns=[group])
            y = no_singleton.iloc[:, [0]]
            x = no_singleton.drop(no_singleton.columns[0], axis=1)
            output = logistic_regression_sk_stripped(y, x)
            return output['b'][0][0], output['p'][0][0]
