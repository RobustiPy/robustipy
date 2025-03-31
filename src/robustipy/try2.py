"""
robustipy.models (Improved)

This module implements multivariate regression classes for Robust Inference.
It includes classes for OLS (OLSRobust and OLSResult) and logistic regression (LRobust)
analysis, along with utilities for model merging, plotting, and Bayesian model averaging.

Refactored to remove duplicate code by moving repeated logic into helper functions.
"""

import warnings
import _pickle
from multiprocessing import cpu_count

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm
from scipy import stats
import sklearn
import shap
from rich.progress import track
from joblib import Parallel, delayed
from statsmodels.tools.tools import add_constant
from sklearn.model_selection import train_test_split, KFold, GroupKFold
from sklearn.metrics import root_mean_squared_error, log_loss, r2_score

from robustipy.prototypes import Protomodel, Protoresult, MissingValueWarning
from robustipy.utils import (simple_ols, logistic_regression_sm, 
                             logistic_regression_sm_stripped, space_size, all_subsets, group_demean)
from robustipy.bootstrap_utils import stripped_ols
from robustipy.figures import plot_results

###############################################################################
#                         HELPER / UTILITY FUNCTIONS                          #
###############################################################################
def _check_numeric_columns(data, cols):
    """Check that all specified columns in the DataFrame are numeric."""
    non_numeric = data[cols].select_dtypes(exclude=[np.number]).columns.tolist()
    if non_numeric:
        raise ValueError(f"The following columns are not numeric and must be converted before fitting: {non_numeric}")

def _get_combination(data, y, x, spec, group=None):
    """
    Prepare a dataframe with the dependent variables (y), independent variables (x)
    and a specification subset (spec). If group is provided, it is inserted appropriately.
    """
    if group:
        cols = list(y) + list(x) + [group] + list(spec)
    else:
        cols = list(y) + list(x) + list(spec)
    return data[cols]

def _get_independent_data(data, x, spec, group=None):
    """
    Prepare a dataframe with independent variables only.
    Used when the dependent variable is provided externally (e.g. composite y).
    """
    if group:
        cols = list(x) + [group] + list(spec)
    else:
        cols = list(x) + list(spec)
    return data[cols]

def _get_bootstrap_sample(data, sample_size, seed, group=None, extra_col=None):
    """
    Return a bootstrap sample of the data.
    
    If extra_col is provided (e.g., "y_star"), it will be kept separately.
    """
    if group is None:
        samp_df = data.sample(n=sample_size, replace=True, random_state=seed)
    else:
        np.random.seed(seed)
        unique_groups = data[group].unique()
        selected_groups = np.random.choice(unique_groups, sample_size)
        select = data[data[group].isin(selected_groups)]
        # Remove groups with only one observation.
        samp_df = select[select.groupby(group).transform('size') > 1].drop(columns=[group])
    
    if extra_col is not None:
        y = samp_df.iloc[:, [0]]
        extra = samp_df[[extra_col]]
        # Drop the first column (dependent variable) and the extra column from independents.
        x = samp_df.drop(columns=[samp_df.columns[0], extra_col])
        return y, x, extra
    else:
        y = samp_df.iloc[:, [0]]
        x = samp_df.drop(samp_df.columns[0], axis=1)
        return y, x

def _evaluate_full_sample_ols(comb_var, kfold, group, oos_metric_name, predict_fn):
    """
    Run simple OLS on the full sample and compute a k-fold evaluation metric.
    """
    y = comb_var.iloc[:, [0]]
    x_temp = comb_var.drop(columns=comb_var.columns[0])
    x = x_temp.drop(columns=[group]) if group in x_temp.columns else x_temp
    out = simple_ols(y=y, x=x)
    
    # k-fold evaluation
    av_k_metric = None
    if kfold:
        if group:
            k_fold = GroupKFold(n_splits=kfold)
            groups = x_temp[group]
        else:
            k_fold = KFold(n_splits=kfold)
            groups = None
        metrics = []
        for train, test in k_fold.split(x, y, groups):
            out_k = simple_ols(y=y.loc[train], x=x.loc[train])
            y_pred = predict_fn(x.loc[test], out_k['b'])
            y_true = y.loc[test]
            if oos_metric_name == 'rmse':
                metrics.append(root_mean_squared_error(y_true, y_pred))
            elif oos_metric_name == 'r-squared':
                metrics.append(r2_score(y_true, y_pred))
            else:
                raise ValueError('No valid OOS metric provided.')
        av_k_metric = np.mean(metrics)
    
    return (out['b'], out['p'], out['r2'],
            out['ll'][0][0], out['aic'][0][0],
            out['bic'][0][0], out['hqic'][0][0],
            av_k_metric)

###############################################################################
#                             CODES                        
###############################################################################
def stouffer_method(p_values, weights=None):
    z_scores = norm.isf(p_values)  # Inverse survival function: Φ⁻¹(1 - p)
    if weights is None:
        Z = np.sum(z_scores) / np.sqrt(len(p_values))
    else:
        weights = np.asarray(weights)
        Z = np.dot(weights, z_scores) / np.sqrt(np.sum(weights**2))
    combined_p = norm.sf(Z)  # Survival function: 1 - Φ(Z)
    return Z, combined_p

###############################################################################
#                             MergedResult Class
###############################################################################
class MergedResult(Protoresult):
    """
    MergedResult: A class for merged OLS results from multiple specifications.
    """
    def __init__(self, *, y, specs, estimates, p_values, r2_values):
        super().__init__()
        self.y_name = y
        self.specs_names = pd.Series(specs)
        self.estimates = pd.DataFrame(estimates)
        self.p_values = pd.DataFrame(p_values)
        self.r2_values = pd.DataFrame(r2_values)
        self.summary_df = self._compute_summary()
        self.summary_df['spec_name'] = self.specs_names

    def summary(self):
        # (Not implemented – you can extend this as needed.)
        pass

    def _compute_summary(self):
        data = self.estimates.copy()
        out = pd.DataFrame()
        out['median'] = data.median(axis=1)
        out['max'] = data.max(axis=1)
        out['min'] = data.min(axis=1)
        out['ci_up'] = data.quantile(q=0.975, axis=1, interpolation='nearest')
        out['ci_down'] = data.quantile(q=0.025, axis=1, interpolation='nearest')
        return out

    def plot(self, loess=True, specs=None, colormap='Spectral_r', figsize=(16, 14),
             ext='pdf', project_name='no_project_name'):
        fig, ax = plt.subplots(figsize=figsize)
        if specs is not None:
            if not all(isinstance(spec, list) for spec in specs):
                raise TypeError("'specs' must be a list of lists.")
            if len(specs) > 3:
                raise ValueError("The max number of specifications to highlight is 3")
            if not all(frozenset(spec) in self.specs_names.to_list() for spec in specs):
                raise TypeError("All specifications in 'specs' must be in the valid computed specifications.")
        
        plot_results(
            results_object=self,
            loess=loess,
            specs=specs,
            ax=ax,
            colormap=colormap,
            ext=ext,
            project_name=project_name
        )
        return fig

    def merge(self, result_obj, left_prefix, right_prefix):
        if not isinstance(result_obj, OLSResult):
            raise TypeError("'result_obj' must be an instance of OLSResult.")
        if not isinstance(left_prefix, str) or not isinstance(right_prefix, str):
            raise TypeError("'prefixes' must be of type 'str'.")
        if self.y_name != result_obj.y_name:
            raise ValueError('Dependent variable names must match.')

        specs_original = [frozenset(list(s) + [left_prefix]) for s in self.specs_names]
        specs_new = [frozenset(list(s) + [right_prefix]) for s in result_obj.specs_names]
        y = self.y_name
        specs = specs_original + specs_new
        estimates = pd.concat([self.estimates, result_obj.estimates], ignore_index=True)
        p_values = pd.concat([self.p_values, result_obj.p_values], ignore_index=True)
        r2_values = pd.concat([self.r2_values, result_obj.r2_values], ignore_index=True)

        return MergedResult(
            y=y,
            specs=specs,
            estimates=estimates,
            p_values=p_values,
            r2_values=r2_values
        )

###############################################################################
#                             OLSResult Class
###############################################################################
class OLSResult(Protoresult):
    """
    OLSResult holds the outputs of the OLSRobust analysis.
    """
    def __init__(self, *, y, x, data, specs, all_predictors, controls, draws, kfold,
                 estimates, estimates_ystar, all_b, all_p, p_values, p_values_ystar,
                 r2_values, r2i_array, ll_array, aic_array, bic_array, hqic_array,
                 av_k_metric_array=None, model_name, name_av_k_metric=None, shap_return=None):
        super().__init__()
        self.y_name = y
        self.x_name = x
        self.data = data
        self.specs_names = pd.Series(specs)
        self.all_predictors = all_predictors
        self.controls = controls
        self.draws = draws
        self.kfold = kfold
        self.estimates = pd.DataFrame(estimates)
        self.p_values = pd.DataFrame(p_values)
        self.estimates_ystar = pd.DataFrame(estimates_ystar)
        self.p_values_ystar = pd.DataFrame(p_values_ystar)
        self.r2_values = pd.DataFrame(r2_values)
        self.all_b = all_b
        self.all_p = all_p
        self.summary_df = self._compute_summary()
        self._compute_inference()
        self.summary_df['r2'] = pd.Series(r2i_array)
        self.summary_df['ll'] = pd.Series(ll_array)
        self.summary_df['aic'] = pd.Series(aic_array)
        self.summary_df['bic'] = pd.Series(bic_array)
        self.summary_df['hqic'] = pd.Series(hqic_array)
        self.summary_df['av_k_metric'] = pd.Series(av_k_metric_array)
        self.summary_df['spec_name'] = self.specs_names
        self.summary_df['y'] = self.y_name
        self.model_name = model_name
        self.name_av_k_metric = name_av_k_metric
        self.shap_return = shap_return

    def save(self, filename):
        with open(filename, 'wb') as f:
            _pickle.dump(self, f, -1)

    @classmethod
    def load(cls, filename):
        with open(filename, 'rb') as f:
            return _pickle.load(f)

    def _compute_inference(self):
        df_model_result = pd.DataFrame({
            'betas': [b[0][0] for b in self.all_b],
            'p_values': [p[0][0] for p in self.all_p],
        })
        inference = not self.estimates_ystar.isna().all().all()
        df_model_result['positive_beta'] = df_model_result['betas'].apply(lambda x: 1 if x > 0 else 0)
        df_model_result['negative_beta'] = df_model_result['betas'].apply(lambda x: 1 if x < 0 else 0)
        df_model_result['significant'] = df_model_result['p_values'].apply(lambda x: 1 if x < 0.05 else 0)
        self.inference = {
            'median_ns': df_model_result['betas'].median(),
            'median': self.estimates.stack().median(),
            'median_p': np.nan if not inference else (((self.estimates_ystar.median(axis=0) > df_model_result['betas'].median()).sum()) / self.estimates_ystar.shape[1]),
            'min_ns': df_model_result['betas'].min(),
            'min': self.estimates.min().min(),
            'max_ns': df_model_result['betas'].max(),
            'max': self.estimates.max().max(),
            'pos_ns': df_model_result['positive_beta'].sum(),
            'pos_prop_ns': df_model_result['positive_beta'].mean(),
            'pos': (self.estimates > 0.0).sum().sum(),
            'pos_prop': (self.estimates > 0.0).mean().mean(),
            'pos_p': np.nan if not inference else ((((self.estimates_ystar > 0.0).sum(axis=0) > df_model_result['positive_beta'].sum()).sum()) / self.estimates_ystar.shape[1]),
            'neg_ns': df_model_result['negative_beta'].sum(),
            'neg_prop_ns': df_model_result['negative_beta'].mean(),
            'neg': (self.estimates < 0.0).sum().sum(),
            'neg_prop': (self.estimates < 0.0).mean().mean(),
            'neg_p': np.nan if not inference else ((((self.estimates_ystar < 0.0).sum(axis=0) > df_model_result['negative_beta'].sum()).sum()) / self.estimates_ystar.shape[1]),
            'sig_ns': df_model_result['significant'].sum(),
            'sig_prop_ns': df_model_result['significant'].mean(),
            'sig': (self.p_values < 0.05).sum().sum(),
            'sig_prop': (self.p_values < 0.05).mean().mean(),
            'sig_p': np.nan if not inference else ((((self.p_values_ystar < 0.05).sum(axis=0) > df_model_result['significant'].sum()).sum()) / self.p_values_ystar.shape[1]),
            'pos_sig_ns': (df_model_result['positive_beta'] & df_model_result['significant']).sum(),
            'pos_sig_prop_ns': (df_model_result['positive_beta'] & df_model_result['significant']).mean(),
            'pos_sig': ((self.estimates > 0.0) & (self.p_values < 0.05)).sum().sum(),
            'pos_sig_prop': ((self.estimates > 0.0) & (self.p_values < 0.05)).mean().mean(),
            'pos_sig_p': np.nan if not inference else ((((self.estimates_ystar > 0.0) & (self.p_values_ystar < 0.05)).sum(axis=0) > ((df_model_result['positive_beta'] & df_model_result['significant']).sum())).sum() / self.estimates_ystar.shape[1]),
            'neg_sig_ns': (df_model_result['negative_beta'] & df_model_result['significant']).sum(),
            'neg_sig_prop_ns': (df_model_result['negative_beta'] & df_model_result['significant']).mean(),
            'neg_sig': ((self.estimates < 0.0) & (self.p_values < 0.05)).sum().sum(),
            'neg_sig_prop': ((self.estimates < 0.0) & (self.p_values < 0.05)).mean().mean(),
            'neg_sig_p': np.nan if not inference else ((((self.estimates_ystar < 0.0) & (self.p_values_ystar < 0.05)).sum(axis=0) > ((df_model_result['negative_beta'] & df_model_result['significant']).sum())).sum() / self.estimates_ystar.shape[1]),
            'Stouffers': stouffer_method(df_model_result['p_values'])
        }

    def summary(self):
        def print_separator(title=None):
            if title:
                print('=' * 30)
                print(title)
                print('=' * 30)
            else:
                print('=' * 30)
        
        print_separator("1. Model Summary")
        print(f"Model: {self.model_name}")
        print('Inference Tests: Yes' if not self.estimates_ystar.isna().all().all() else 'Inference Tests: No')
        print(f"Dependent variable: {self.y_name}")
        print(f"Independent variable: {self.x_name}")
        print(f"Number of possible controls: {len(self.controls)}")
        print(f"Number of draws: {self.draws}")
        print(f"Number of folds: {self.kfold}")
        print(f"Number of specifications: {len(self.specs_names)}")
        
        print_separator("2.Model Robustness Metrics")
        print('2.1 Inference Metrics')
        print_separator()
        if np.isnan(self.inference['median_p']):
            print(f"Median beta (all specifications, no resampling): {self.inference['median_ns']}")
        else:
            print(f"Median beta (all specifications, no resampling): {self.inference['median_ns']} (p-value: {self.inference['median_p']})")
        print(f"Median beta (all bootstraps and specifications): {self.inference['median']}")
        # ... (similar prints for other metrics)
        print(f"Stouffer's Z-score test: {self.inference['Stouffers'][0]}, {self.inference['Stouffers'][1]}")
        
        print_separator()
        print('2.1 In-Sample Metrics (Full Sample)')
        print_separator()
        print(f"Min AIC: {self.summary_df['aic'].min()}, Specs: {list(self.summary_df['spec_name'].loc[self.summary_df['aic'].idxmin()])}")
        print(f"Min BIC: {self.summary_df['bic'].min()}, Specs: {list(self.summary_df['spec_name'].loc[self.summary_df['bic'].idxmin()])}")
        print(f"Min HQIC: {self.summary_df['hqic'].min()}, Specs: {list(self.summary_df['spec_name'].loc[self.summary_df['hqic'].idxmin()])}")
        print(f"Max Log Likelihood: {self.summary_df['ll'].max()}, Specs: {list(self.summary_df['spec_name'].loc[self.summary_df['ll'].idxmax()])}")
        print(f"Min Log Likelihood: {self.summary_df['ll'].min()}, Specs: {list(self.summary_df['spec_name'].loc[self.summary_df['ll'].idxmin()])}")
        print(f"Max R2: {self.summary_df['r2'].max()}, Specs: {list(self.summary_df['spec_name'].loc[self.summary_df['r2'].idxmax()])}")
        print(f"Min R2: {self.summary_df['r2'].min()}, Specs: {list(self.summary_df['spec_name'].loc[self.summary_df['r2'].idxmin()])}")
        
        print_separator()
        print(f"2.2 Out-Of-Sample Metrics ({self.name_av_k_metric} averaged across folds)")
        print_separator()
        oos_max_row = self.summary_df.loc[self.summary_df['av_k_metric'].idxmax(),]
        print(f'Max Average: {oos_max_row["av_k_metric"]}, Specs: {list(oos_max_row["spec_name"])} ')
        oos_min_row = self.summary_df.loc[self.summary_df['av_k_metric'].idxmin(),]
        print(f'Min Average: {oos_min_row["av_k_metric"]}, Specs: {list(oos_min_row["spec_name"])} ')
        print(f"Mean Average: {self.summary_df['av_k_metric'].mean()}")
        print(f"Median Average: {self.summary_df['av_k_metric'].median()}")

    def plot(self, loess=True, specs=None, ic='aic', colormap='Spectral_r', figsize=(12, 6),
             ext='pdf', project_name='no_project_name'):
        valid_ic = ['bic', 'aic', 'hqic']
        if specs is not None:
            if not all(isinstance(l, list) for l in specs):
                raise TypeError("'specs' must be a list of lists.")
            if len(specs) > 3:
                raise ValueError("The max number of specifications to highlight is 3") 
            if not all(frozenset(spec) in self.specs_names.to_list() for spec in specs):
                raise TypeError("All specifications in 'spec' must be in the valid computed specifications.")
        if ic not in valid_ic:
            raise ValueError(f"'ic' must be one of the following: {valid_ic}")
        return plot_results(results_object=self,
                            loess=loess,
                            specs=specs,
                            ic=ic,
                            colormap=colormap,
                            figsize=figsize,
                            ext=ext,
                            project_name=project_name)

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
            idx = [ele in spec for spec in self.specs_names]
            likelihood_per_var.append(np.nansum(models_likelihood[idx]))
            coefs_ele = coefs_df.loc[coefs_df.var_name == ele, 'coef'].to_numpy()
            likelihood = models_likelihood[idx]
            weigthed_coefs.append(np.nansum(coefs_ele * likelihood))
        probs = likelihood_per_var / sum_likelihoods
        final_coefs = weigthed_coefs / sum_likelihoods
        summary_bma = pd.DataFrame({
            'control_var': self.controls,
            'probs': probs,
            'average_coefs': final_coefs
        })
        return summary_bma

    def merge(self, result_obj, left_prefix, right_prefix) -> MergedResult:
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
        r2_values = pd.concat([self.r2_values, result_obj.r2_values], ignore_index=True)
        return MergedResult(y=y, specs=specs, estimates=estimates, p_values=p_values, r2_values=r2_values)

    def save_to_csv(self, path: str):
        self.summary_df.to_csv(path)

###############################################################################
#                             OLSRobust Class
###############################################################################
class OLSRobust(Protomodel):
    """
    Class for multivariate regression using OLS.
    """
    def __init__(self, *, y, x, data, model_name='OLS Robust'):
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
        self.model_name = model_name
        self.parameters = {'y': self.y, 'x': self.x}

    def get_results(self):
        return self.results

    def multiple_y(self):
        """
        Computes composite y based on multiple indicators provided.
        """
        self.y_specs = []
        self.y_composites = []
        print("Calculating Composite Ys")
        for spec, _ in track(zip(all_subsets(self.y), range(space_size(self.y))), total=space_size(self.y)):
            if spec:
                subset = self.data[list(spec)]
                subset = (subset - subset.mean()) / subset.std()
                self.y_composites.append(subset.mean(axis=1))
                self.y_specs.append(spec)
        self.parameters['y_specs'] = self.y_specs
        self.parameters['y_composites'] = self.y_composites

    def _strap_OLS(self, comb_var, group, sample_size, shuffle, seed, y_star):
        """
        Perform a bootstrap iteration using stripped_ols for both y and y_star.
        """
        temp_data = comb_var.copy()
        temp_data['y_star'] = y_star  # attach the computed y_star column
        # Get bootstrap sample (using helper that returns y, x and extra column)
        if group is None:
            y_sample, x_sample, y_star_sample = _get_bootstrap_sample(temp_data, sample_size, seed, extra_col='y_star')
        else:
            y_sample, x_sample, y_star_sample = _get_bootstrap_sample(temp_data, sample_size, seed, group, extra_col='y_star')
        output = stripped_ols(y=y_sample, x=x_sample)
        output_ystar = stripped_ols(y=y_star_sample, x=x_sample)
        return output['b'][0][0], output['p'][0][0], output['r2'], output_ystar['b'][0][0], output_ystar['p'][0][0]

    def _predict(self, x_test, betas):
        return np.dot(x_test, betas)

    def _full_sample_OLS(self, comb_var, kfold, group, oos_metric_name):
        # Use helper _evaluate_full_sample_ols with our _predict function.
        return _evaluate_full_sample_ols(comb_var, kfold, group, oos_metric_name, self._predict)

    def fit(self, *, controls, group=None, draws=500, kfold=5, shuffle=False,
            oos_metric='r-squared', n_cpu=None, seed=None):
        """
        Fit the OLS models over the specification space and bootstrapped samples.
        """
        if not isinstance(controls, list):
            raise TypeError("'controls' must be a list.")
        all_vars = set(self.data.columns)
        if not all(var in all_vars for var in controls):
            raise ValueError("Variable names in 'controls' must exist in the provided DataFrame 'data'.")
        if group is not None:
            if group not in all_vars:
                raise ValueError("'group' variable must exist in the provided DataFrame 'data'.")
            if not isinstance(group, str):
                raise TypeError("'group' must be a string.")
        if kfold < 2:
            raise ValueError(f"kfold values must be 2 or above, current value is {kfold}.")
        if draws < 1:
            raise ValueError(f"Draws value must be 1 or above, current value is {draws}.")
        valid_oos_metric = ['r-squared', 'rmse']
        if oos_metric not in valid_oos_metric:
            raise ValueError(f"OOS Metric must be one of {valid_oos_metric}.")
        if n_cpu is None:
            n_cpu = cpu_count()
        if not isinstance(n_cpu, int):
            raise TypeError("n_cpu must be an integer")
        if seed is not None:
            if not isinstance(seed, int):
                raise TypeError("seed must be an integer")
            np.random.seed(seed)
        
        cols_to_check = self.y + self.x + ([group] if group else []) + controls
        _check_numeric_columns(self.data, cols_to_check)
        sample_size = self.data.shape[0]
        self.oos_metric_name = oos_metric

        # Containers to store results for all specifications.
        specs = []
        all_predictors = []
        list_b_array = []
        list_p_array = []
        list_b_array_ystar = []
        list_p_array_ystar = []
        list_r2_array = []
        list_r2i_array = []
        list_ll_array = []
        list_aic_array = []
        list_bic_array = []
        list_hqic_array = []
        list_av_k_metric_array = []
        y_names = []

        if len(self.y) > 1:
            # When multiple dependent variables are provided.
            self.multiple_y()
            for comp_y, spec_y in zip(self.y_composites, self.y_specs):
                space_n = space_size(controls)
                b_array = np.empty([space_n, draws])
                p_array = np.empty([space_n, draws])
                b_array_ystar = np.empty([space_n, draws])
                p_array_ystar = np.empty([space_n, draws])
                r2_array = np.empty([space_n, draws])
                r2i_array = np.empty(space_n)
                ll_array = np.empty(space_n)
                aic_array = np.empty(space_n)
                bic_array = np.empty(space_n)
                hqic_array = np.empty(space_n)
                av_k_metric_array = np.empty(space_n)
                predictors_list = []
                for idx, spec in enumerate(track(list(all_subsets(controls)), total=space_size(controls))):
                    # Prepare independent variables from data.
                    comb_x = _get_independent_data(self.data, self.x, spec, group)
                    # Combine the externally computed composite y with x.
                    comb = pd.concat([comp_y, comb_x], axis=1)
                    # Run full sample OLS.
                    (b_all, p_all, r2_i, ll_i, aic_i, bic_i, hqic_i, av_k_metric_i) = \
                        self._full_sample_OLS(comb, kfold, group, self.oos_metric_name)
                    # Compute y_star for bootstrap using the first coefficient as a placeholder.
                    y_star = comb.iloc[:, [0]] - np.dot(comb.iloc[:, [1]], b_all[0][0])
                    seeds = np.random.randint(0, 2 ** 32 - 1, size=draws)
                    bootstrap_results = Parallel(n_jobs=n_cpu)(
                        delayed(self._strap_OLS)(comb, group, sample_size, shuffle, seed, y_star)
                        for seed in seeds)
                    # Unpack bootstrap results.
                    b_array[idx, :], p_array[idx, :], r2_array[idx, :], b_array_ystar[idx, :], p_array_ystar[idx, :] = zip(*bootstrap_results)
                    r2i_array[idx] = r2_i
                    ll_array[idx] = ll_i
                    aic_array[idx] = aic_i
                    bic_array[idx] = bic_i
                    hqic_array[idx] = hqic_i
                    av_k_metric_array[idx] = av_k_metric_i
                    specs.append(frozenset(list(spec_y) + list(spec)))
                    predictors_list.append(self.x + list(spec) + ['const'])
                    y_names.append(spec_y)
                all_predictors.append(predictors_list)
                list_b_array.append(b_array)
                list_p_array.append(p_array)
                list_b_array_ystar.append(b_array_ystar)
                list_p_array_ystar.append(p_array_ystar)
                list_r2_array.append(r2_array)
                list_r2i_array.append(r2i_array)
                list_ll_array.append(ll_array)
                list_aic_array.append(aic_array)
                list_bic_array.append(bic_array)
                list_hqic_array.append(hqic_array)
                list_av_k_metric_array.append(av_k_metric_array)
            results = OLSResult(
                y=y_names,
                x=self.x,
                data=self.data,
                specs=specs,
                all_predictors=[p for sublist in all_predictors for p in sublist],
                controls=controls,
                draws=draws,
                kfold=kfold,
                all_b=list_b_array,
                all_p=list_p_array,
                estimates=np.vstack(list_b_array),
                p_values=np.vstack(list_p_array),
                estimates_ystar=np.vstack(list_b_array_ystar),
                p_values_ystar=np.vstack(list_p_array_ystar),
                r2_values=np.vstack(list_r2_array),
                r2i_array=np.hstack(list_r2i_array),
                ll_array=np.hstack(list_ll_array),
                aic_array=np.hstack(list_aic_array),
                bic_array=np.hstack(list_bic_array),
                hqic_array=np.hstack(list_hqic_array),
                av_k_metric_array=np.hstack(list_av_k_metric_array),
                model_name=self.model_name,
                name_av_k_metric=self.oos_metric_name,
                shap_return=None
            )
            self.results = results
        else:
            # When only one dependent variable is provided.
            space_n = space_size(controls)
            specs = []
            all_predictors = []
            b_all_list = []
            p_all_list = []
            b_array = np.empty([space_n, draws])
            p_array = np.empty([space_n, draws])
            b_array_ystar = np.empty([space_n, draws])
            p_array_ystar = np.empty([space_n, draws])
            r2_array = np.empty([space_n, draws])
            r2i_array = np.empty(space_n)
            ll_array = np.empty(space_n)
            aic_array = np.empty(space_n)
            bic_array = np.empty(space_n)
            hqic_array = np.empty(space_n)
            av_k_metric_array = np.empty(space_n)
            
            # Prepare data for SHAP (optional)
            if group:
                shap_comb = group_demean(self.data[self.y + self.x + [group] + controls], group=group)
            else:
                shap_comb = self.data[self.y + self.x + controls]
            shap_comb = shap_comb.dropna().reset_index(drop=True)
            x_train, x_test, y_train, _ = train_test_split(shap_comb[self.x + controls],
                                                           shap_comb[self.y],
                                                           test_size=0.2,
                                                           random_state=seed)
            model = sklearn.linear_model.LinearRegression()
            model.fit(x_train, y_train)
            explainer = shap.LinearExplainer(model, x_train)
            shap_return = [explainer.shap_values(x_test), x_test]
            
            for idx, spec in enumerate(track(list(all_subsets(controls)), total=space_n)):
                comb = _get_combination(self.data, self.y, self.x, spec, group)
                comb = comb.dropna().reset_index(drop=True)
                if group:
                    comb = group_demean(comb, group=group)
                (b_all, p_all, r2_i, ll_i, aic_i, bic_i, hqic_i, av_k_metric_i) = \
                    self._full_sample_OLS(comb, kfold, group, self.oos_metric_name)
                y_star = comb.iloc[:, [0]] - np.dot(comb.iloc[:, [1]], b_all[0][0])
                seeds = np.random.randint(0, 2 ** 32 - 1, size=draws)
                bootstrap_results = Parallel(n_jobs=n_cpu)(
                    delayed(self._strap_OLS)(comb, group, sample_size, shuffle, seed, y_star)
                    for seed in seeds)
                b_array[idx, :], p_array[idx, :], r2_array[idx, :], b_array_ystar[idx, :], p_array_ystar[idx, :] = zip(*bootstrap_results)
                specs.append(frozenset(spec))
                all_predictors.append(self.x + list(spec) + ['const'])
                b_all_list.append(b_all)
                p_all_list.append(p_all)
                r2i_array[idx] = r2_i
                ll_array[idx] = ll_i
                aic_array[idx] = aic_i
                bic_array[idx] = bic_i
                hqic_array[idx] = hqic_i
                av_k_metric_array[idx] = av_k_metric_i
            results = OLSResult(
                y=self.y[0],
                x=self.x[0],
                data=self.data,
                specs=specs,
                all_predictors=all_predictors,
                controls=controls,
                draws=draws,
                kfold=kfold,
                all_b=b_all_list,
                all_p=p_all_list,
                estimates=b_array,
                p_values=p_array,
                estimates_ystar=b_array_ystar,
                p_values_ystar=p_array_ystar,
                r2_values=r2_array,
                r2i_array=r2i_array,
                ll_array=ll_array,
                aic_array=aic_array,
                bic_array=bic_array,
                hqic_array=hqic_array,
                av_k_metric_array=av_k_metric_array,
                model_name=self.model_name,
                name_av_k_metric=self.oos_metric_name,
                shap_return=shap_return
            )
            self.results = results

###############################################################################
#                             LRobust Class
###############################################################################
class LRobust(Protomodel):
    """
    A class to perform robust logistic regression analysis.
    """
    def __init__(self, *, y, x, data, model_name='Logistic Regression Robust'):
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
        self.model_name = model_name

    def get_results(self):
        return self.results

    def multiple_y(self):
        raise NotImplementedError("Not implemented yet")

    def _full_sample(self, comb_var, kfold, group, oos_metric_name, predict_fn):
        y = comb_var.iloc[:, [0]]
        x = comb_var.drop(columns=comb_var.columns[0])
        out = logistic_regression_sm(y=y, x=x)
        av_k_metric = None
        if kfold:
            k_fold = KFold(n_splits=kfold)
            metrics = []
            for train, test in k_fold.split(x, y):
                out_k = logistic_regression_sm(y=y.loc[train], x=x.loc[train])
                y_pred = predict_fn(x.loc[test], out_k['b'])
                y_true = y.loc[test]
                if oos_metric_name == 'rmse':
                    metrics.append(root_mean_squared_error(y_true, y_pred))
                elif oos_metric_name == 'r-squared':
                    metrics.append(r2_score(y_true, y_pred))
                elif oos_metric_name == 'cross-entropy':
                    metrics.append(log_loss(y_true, y_pred))
                else:
                    raise ValueError('No valid OOS metric provided.')
            av_k_metric = np.mean(metrics)
        return (out['b'], out['p'], out['r2'], out['ll'], out['aic'], out['bic'], out['hqic'], av_k_metric)

    def _predict_LR(self, x_test, betas):
        x_test = add_constant(x_test, prepend=False)
        return 1 / (1 + np.exp(-x_test.dot(betas)))

    def _strap_regression(self, comb_var, group, sample_size, shuffle, seed):
        temp_data = comb_var.copy()
        if group is None:
            y_sample, x_sample = _get_bootstrap_sample(temp_data, sample_size, seed)
        else:
            y_sample, x_sample = _get_bootstrap_sample(temp_data, sample_size, seed, group)
        output = logistic_regression_sm(y_sample, x_sample)
        return output['b'][0][0], output['p'][0][0], output['r2']

    def fit(self, *, controls, group=None, draws=500, sample_size=None, kfold=5,
            shuffle=False, oos_metric='r-squared', n_cpu=None, seed=None):
        if not isinstance(controls, list):
            raise TypeError("'controls' must be a list.")
        all_vars = set(self.data.columns)
        if not all(var in all_vars for var in controls):
            raise ValueError("Variable names in 'controls' must exist in the provided DataFrame 'data'.")
        if group is not None and group not in all_vars:
            raise ValueError("'group' variable must exist in the provided DataFrame 'data'.")
        if kfold < 2:
            raise ValueError(f"kfold values must be 2 or above, current value is {kfold}.")
        valid_oos_metric = ['r-squared', 'rmse', 'cross-entropy']
        if oos_metric not in valid_oos_metric:
            raise ValueError(f"OOS Metric must be one of {valid_oos_metric}.")
        if n_cpu is None:
            n_cpu = cpu_count()
        if not isinstance(n_cpu, int):
            raise TypeError("n_cpu must be an integer")
        if seed is not None:
            if not isinstance(seed, int):
                raise TypeError("seed must be an integer")
            np.random.seed(seed)

        cols_to_check = self.y + self.x + ([group] if group else []) + controls
        _check_numeric_columns(self.data, cols_to_check)
        sample_size = self.data.shape[0] if sample_size is None else sample_size
        self.oos_metric_name = oos_metric

        # Prepare SHAP values for model interpretability
        if group:
            shap_comb = group_demean(self.data[self.y + self.x + [group] + controls], group=group)
        else:
            shap_comb = self.data[self.y + self.x + controls]
        shap_comb = shap_comb.dropna().reset_index(drop=True)
        x_train, x_test, y_train, _ = train_test_split(shap_comb[self.x + controls],
                                                       shap_comb[self.y],
                                                       test_size=0.2,
                                                       random_state=seed)
        model = sklearn.linear_model.LogisticRegression(penalty="l2", C=0.1)
        model.fit(x_train, y_train.squeeze())
        explainer = shap.LinearExplainer(model, x_train)
        shap_return = [explainer.shap_values(x_test), x_test]

        space_n = space_size(controls)
        specs = []
        all_predictors = []
        b_all_list = []
        p_all_list = []
        b_array = np.empty([space_n, draws])
        p_array = np.empty([space_n, draws])
        r2_array = np.empty([space_n, draws])
        r2i_array = np.empty(space_n)
        ll_array = np.empty(space_n)
        aic_array = np.empty(space_n)
        bic_array = np.empty(space_n)
        hqic_array = np.empty(space_n)
        av_k_metric_array = np.empty(space_n)
        for idx, spec in enumerate(track(list(all_subsets(controls)), total=space_n)):
            if spec:
                comb = _get_combination(self.data, self.y, self.x, spec, group)
            else:
                comb = self.data[self.y + self.x]
            if group:
                comb = self.data[self.y + self.x + [group] + list(spec)]
            comb = comb.dropna().reset_index(drop=True)
            if group:
                comb = group_demean(comb, group=group)
            (b_all, p_all, r2_i, ll_i, aic_i, bic_i, hqic_i, av_k_metric_i) = \
                self._full_sample(comb, kfold, group, self.oos_metric_name, self._predict_LR)
            seeds = np.random.randint(0, 2 ** 32 - 1, size=draws)
            bootstrap_results = Parallel(n_jobs=n_cpu)(
                delayed(self._strap_regression)(comb, group, sample_size, shuffle, seed)
                for seed in seeds)
            b_array[idx, :], p_array[idx, :], r2_array[idx, :] = zip(*bootstrap_results)
            specs.append(frozenset(spec))
            all_predictors.append(self.x + list(spec) + ['const'])
            b_all_list.append(b_all)
            p_all_list.append(p_all)
            r2i_array[idx] = r2_i
            ll_array[idx] = ll_i
            aic_array[idx] = aic_i
            bic_array[idx] = bic_i
            hqic_array[idx] = hqic_i
            av_k_metric_array[idx] = av_k_metric_i
        results = OLSResult(
            y=self.y[0],
            x=self.x,
            data=self.data,
            specs=specs,
            all_predictors=all_predictors,
            controls=controls,
            draws=draws,
            kfold=kfold,
            all_b=b_all_list,
            all_p=p_all_list,
            estimates=b_array,
            p_values=p_array,
            estimates_ystar=np.nan * b_array,  # Not applicable for logistic regression
            p_values_ystar=np.nan * p_array,
            r2_values=r2_array,
            r2i_array=r2i_array,
            ll_array=ll_array,
            aic_array=aic_array,
            bic_array=bic_array,
            hqic_array=hqic_array,
            av_k_metric_array=av_k_metric_array,
            model_name=self.model_name,
            name_av_k_metric=self.oos_metric_name,
            shap_return=shap_return
        )
        self.results = results
