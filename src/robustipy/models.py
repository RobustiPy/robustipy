"""robustipy.models

This module implements multivariate regression classes for Robust Inference.
It includes classes for OLS (OLSRobust and OLSResult) and logistic regression (LRobust)
analysis, along with utilities for model merging, plotting, and Bayesian model averaging.
"""

import _pickle
import warnings
from multiprocessing import cpu_count

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
import sklearn
from joblib import Parallel, delayed
from rich.progress import track
from scipy.stats import norm
from sklearn.metrics import log_loss, r2_score, root_mean_squared_error
from sklearn.model_selection import GroupKFold, KFold, train_test_split
from statsmodels.tools.tools import add_constant

from robustipy.bootstrap_utils import stripped_ols
from robustipy.figures import plot_results
from robustipy.prototypes import MissingValueWarning, Protomodel, Protoresult
from robustipy.utils import (
    all_subsets,
    group_demean,
    logistic_regression_sm,
    simple_ols,
    space_size,
)


def _check_numeric_columns(data, cols):
    """Check that all specified columns in the DataFrame are numeric."""
    non_numeric = data[cols].select_dtypes(exclude=[np.number]).columns.tolist()
    if non_numeric:
        raise ValueError(f"The following columns are not numeric and must be converted before fitting: {non_numeric}")

def stouffer_method(p_values, weights=None):
    """Combine p-values using Stouffer's method."""
    z_scores = norm.isf(p_values)  # Inverse survival function: Φ⁻¹(1 - p)
    if weights is None:
        Z = np.sum(z_scores) / np.sqrt(len(p_values))
    else:
        weights = np.asarray(weights)
        Z = np.dot(weights, z_scores) / np.sqrt(np.sum(weights**2))
    combined_p = norm.sf(Z)  # Survival function: 1 - Φ(Z)
    return Z, combined_p

class MergedResult(Protoresult):
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
        """
        Generates a summary of the regression results (not implemented).
        """
        pass

    def _compute_summary(self):
        """
        Computes summary statistics based on coefficient estimates.

        Returns:
            pd.DataFrame: DataFrame containing median, min, max, and quantiles.
        """
        # TODO: use pandas describe
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
        """
        Plots the regression results using specified options.

        Returns
        -------
        matplotlib.figure.Figure:
            Plot showing the regression results.
        """
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
        """
        Merges two OLSResult objects into one.

        Parameters
        -------
            result_obj (OLSResult): OLSResult object to be merged.
            left_prefix (str): Prefix for the orignal result object.
            right_prefix (str): Prefix fort the new result object.

        Raises
        -------
            TypeError: If the input object is not an instance of OLSResult.
        """
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
        r2_values (pd.DataFrame): DataFrame containing r2 for each specification and draw.
        r2i_array (pd.DataFrame): DataFrame containing r2 for each specification (no straps).
        ll_array (list): List of log-likelihood values for each specification.
        aic_array (list): List of AIC values for each specification.
        bic_array (list): List of BIC values for each specification.
        hqic_array (list): List of HQIC values for each specification.
        av_k_metric_array (list, optional): List of average metrics.

    Methods:
        save(filename):
            Save the OLSResult object to a file using pickle.

        load(filename):
            Load an OLSResult object from a file using pickle.

        summary():
            Placeholder for a method to generate a summary of the results.

        plot(specs=None, ic=None, colormap=None, figsize=(12, 6)):
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


    def _compute_inference(self):
        """
        Compute various inference statistics and store in self.inference.
        """
        df_model_result = pd.DataFrame({
            'betas': [b[0][0] for b in self.all_b],
            'p_values': [p[0][0] for p in self.all_p],
        })
        inference = not self.estimates_ystar.isna().all().all()
        df_model_result['positive_beta'] = df_model_result['betas'].apply(lambda x: 1 if x > 0 else 0)
        df_model_result['negative_beta'] = df_model_result['betas'].apply(lambda x: 1 if x < 0 else 0)
        df_model_result['significant'] = df_model_result['p_values'].apply(lambda x: 1 if x < 0.05 else 0)
        self.inference = {}
        self.inference['median_ns'] = df_model_result['betas'].median() # note: ns for 'no sampling'
        self.inference['median'] = self.estimates.stack().median()
        self.inference['median_p'] = (
            np.nan if not inference else 
            (self.estimates_ystar.median(axis=0) > df_model_result['betas'].median()).mean()
        )
        self.inference['min_ns'] = df_model_result['betas'].min()
        self.inference['min'] = self.estimates.min().min()
        self.inference['max_ns'] = df_model_result['betas'].max()
        self.inference['max'] = self.estimates.min().max()

        self.inference['pos_ns'] = df_model_result['positive_beta'].sum()
        self.inference['pos_prop_ns'] = df_model_result['positive_beta'].mean()
        self.inference['pos'] = (self.estimates > 0.0).sum().sum()
        self.inference['pos_prop'] = (self.estimates > 0.0).mean().mean()
        self.inference['pos_p'] = (
            np.nan if not inference else
            ((self.estimates_ystar.gt(0).sum(axis=0) > df_model_result['positive_beta'].sum()).mean())
        )
        self.inference['neg_ns'] = df_model_result['negative_beta'].sum()
        self.inference['neg_prop_ns'] = df_model_result['negative_beta'].mean()
        self.inference['neg'] = (self.estimates < 0.0).sum().sum()
        self.inference['neg_prop'] = (self.estimates < 0.0).mean().mean()
        self.inference['neg_p'] = (
            np.nan if not inference else
            (self.estimates_ystar.lt(0).sum(axis=0) > df_model_result['negative_beta'].sum()).mean()
        )
        self.inference['sig_ns'] = df_model_result['significant'].sum()
        self.inference['sig_prop_ns'] = df_model_result['significant'].mean()
        self.inference['sig'] = (self.p_values.stack() < 0.05).sum().sum()
        self.inference['sig_prop'] = (self.p_values.stack() < 0.05).mean().mean()
        self.inference['sig_p'] = (
            np.nan if not inference else
            (self.p_values_ystar.lt(0.05).sum(axis=0) > df_model_result['significant'].sum()).mean()
        )
        self.inference['pos_sig_ns'] = (df_model_result['positive_beta'] &
                                     df_model_result['significant']).sum()
        self.inference['pos_sig_prop_ns'] = (df_model_result['positive_beta'] &
                                          df_model_result['significant']).mean()
        self.inference['pos_sig'] = ((self.estimates > 0.0) &
                                     (self.p_values < 0.05)).sum().sum()
        self.inference['pos_sig_prop'] = ((self.estimates > 0.0) &
                                          (self.p_values < 0.05)).mean().mean()
        self.inference['pos_sig_p'] = (
            np.nan if not inference else
            ((self.estimates_ystar.gt(0) & self.p_values_ystar.lt(0.05)).sum(axis=0) >
            (df_model_result['positive_beta'] & df_model_result['significant']).sum()).mean()
        )
        self.inference['neg_sig_ns'] = (df_model_result['negative_beta'] &
                                        df_model_result['significant']).sum()
        self.inference['neg_sig_prop_ns'] = (df_model_result['negative_beta'] &
                                             df_model_result['significant']).mean()
        self.inference['neg_sig'] = ((self.estimates < 0.0) &
                                     (self.p_values < 0.05)).sum().sum()
        self.inference['neg_sig_prop'] = ((self.estimates < 0.0) &
                                          (self.p_values < 0.05)).mean().mean()
        self.inference['neg_sig_p'] = (
            np.nan if not inference else
            ((self.estimates_ystar.lt(0) & self.p_values_ystar.lt(0.05)).sum(axis=0) >
            (df_model_result['negative_beta'] & df_model_result['significant']).sum()).mean()
        )
        self.inference['Stouffers'] = stouffer_method(df_model_result['p_values'])


    def summary(self):
        """Prints a summary of the model including basic configuration and robustness metrics."""
        def print_separator(title=None):
            print("=" * 30)
            if title:
                print(title)
                print("=" * 30)

        inference = not self.estimates_ystar.isna().all().all()
        # Display basic model information
        print_separator("1. Model Summary")
        print(f"Model: {self.model_name}")
        print("Inference Tests:", "Yes" if inference else "No")
        print(f"Dependent variable: {self.y_name}")
        print(f"Independent variable: {self.x_name}")
        print(f"Number of possible controls: {len(self.controls)}")
        print(f"Number of draws: {self.draws}")
        print(f"Number of folds: {self.kfold}")
        print(f"Number of specifications: {len(self.specs_names)}")

        # Print model robustness metrics
        print_separator("2.Model Robustness Metrics")
        print('2.1 Inference Metrics')
        print_separator()
        if inference is False:
            print(f"Median beta (all specifications, no resampling): {self.inference['median_ns']}")
        else:
            print(f"Median beta (all specifications, no resampling): {self.inference['median_ns']} (p-value: {self.inference['median_p']})")
        print(f"Median beta (all bootstraps and specifications): {self.inference['median']}")

        print(f"Min beta (all specifications, no resampling): {self.inference['min_ns']}")
        print(f"Min beta (all bootstraps and specifications): {self.inference['min']}")

        print(f"Max beta (all specifications, no resampling): {self.inference['max_ns']}")
        print(f"Max beta (all bootstraps and specifications): {self.inference['max']}")

        if inference is False:
            print(f"Significant portion of beta (all specifications, no resampling): {self.inference['sig_prop_ns']}")
        else:
            print(f"Significant portion of beta (all specifications, no resampling): {self.inference['sig_prop_ns']} (p-value: {self.inference['sig_p']})")
        print(f"Significant portion of beta (all bootstraps and specifications): {self.inference['sig_prop']}")
        if inference is False:
            print(f"Positive portion of beta (all specifications, no resampling): {self.inference['pos_prop_ns']}")
        else:
            print(f"Positive portion of beta (all specifications, no resampling): {self.inference['pos_prop_ns']} (p-value: {self.inference['pos_p']})")
        print(f"Positive portion of beta (all bootstraps and specifications): {self.inference['pos_prop']}")

        if inference is False:
            print(f"Negative portion of beta (all specifications, no resampling): {self.inference['neg_prop_ns']}")
        else:
            print(f"Negative portion of beta (all specifications, no resampling): {self.inference['neg_prop_ns']} (p-value: {self.inference['neg_p']})")
        print(f"Negative portion of beta (all bootstraps and specifications): {self.inference['neg_prop']}")

        if inference is False:
            print(f"Positive and Significant portion of beta (all specifications, no resampling): {self.inference['pos_sig_prop_ns']}")
        else:
            print(f"Positive and Significant portion of beta (all specifications, no resampling): {self.inference['pos_sig_prop_ns']} (p-value: {self.inference['pos_sig_p']})")
        print(f"Positive and Significant portion of beta (all bootstraps and specifications): {self.inference['pos_sig_prop']}")

        if inference is False:
            print(f"Negative and Significant portion of beta (all specifications, no resampling): {self.inference['neg_sig_prop_ns']}")
        else:
            print(f"Negative and Significant portion of beta (all specifications, no resampling): {self.inference['neg_sig_prop_ns']} (p-value: {self.inference['neg_sig_p']})")
        print(f"Negative and Significant portion of beta (all bootstraps and specifications): {self.inference['neg_sig_prop']}")

        print(f"Stouffer's Z-score test: {self.inference['Stouffers'][0]}, {self.inference['Stouffers'][1]}")

        print_separator()
        print('2.2 In-Sample Metrics (Full Sample)')
        print_separator()
        print(f"Min AIC: {self.summary_df['aic'].min()},Specs: {list(self.summary_df['spec_name'].loc[self.summary_df['aic'].idxmin()])}")
        print(f"Min BIC: {self.summary_df['bic'].min()}, Specs: {list(self.summary_df['spec_name'].loc[self.summary_df['bic'].idxmin()])}")
        print(f"Min HQIC: {self.summary_df['hqic'].min()}, Specs: {list(self.summary_df['spec_name'].loc[self.summary_df['hqic'].idxmin()])}")
        print(
            f"Max Log Likelihood: {self.summary_df['ll'].max()}, Specs: {list(self.summary_df['spec_name'].loc[self.summary_df['ll'].idxmax()])}")
        print(
            f"Min Log Likelihood: {self.summary_df['ll'].min()}, Specs: {list(self.summary_df['spec_name'].loc[self.summary_df['ll'].idxmin()])}")
        print(
            f"Max R2: {self.summary_df['r2'].max()}, Specs: {list(self.summary_df['spec_name'].loc[self.summary_df['r2'].idxmax()])}")
        print(
            f"Min R2: {self.summary_df['r2'].min()}, Specs: {list(self.summary_df['spec_name'].loc[self.summary_df['r2'].idxmin()])}")

        print_separator()
        print(f'2.3 Out-Of-Sample Metrics ({self.name_av_k_metric} averaged across folds)')
        print_separator()
        oos_max_row = self.summary_df.loc[self.summary_df['av_k_metric'].idxmax(),]
        print(f'Max Average: {oos_max_row["av_k_metric"]}, Specs: {list(oos_max_row["spec_name"])} ')
        oos_min_row = self.summary_df.loc[self.summary_df['av_k_metric'].idxmin(),]
        print(f'Min Average: {oos_min_row["av_k_metric"]}, Specs: {list(oos_min_row["spec_name"])} ')
        print(f"Mean Average: {self.summary_df['av_k_metric'].mean()}")
        print(f"Median Average: {self.summary_df['av_k_metric'].median()}")


    def plot(self,
             loess=True,
             specs=None,
             ic='aic',
             colormap='Spectral_r',
             figsize=(12, 6),
             ext='pdf',
             project_name='no_project_name'):
        """
        Plots the regression results using specified options.

        Args:
            specs (list, optional): List of list of specification names to include in the plot.
            ic (str, optional): Information criterion to use for model selection (e.g., 'bic', 'aic').
            colormap (str, optional): Colormap to use for the plot.
            figsize (tuple, optional): Size of the figure (width, height) in inches.

        Returns:
            matplotlib.figure.Figure: Plot showing the regression results.
        """
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
            idx = [ele in spec for spec in self.specs_names]
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
        r2_values = pd.concat([self.r2_values, result_obj.r2_values], ignore_index=True)

        return MergedResult(y=y, specs=specs, estimates=estimates, p_values=p_values, r2_values=r2_values)

    def save_to_csv(self, path: str):
        """Function to save summary dataframe to a csv"""
        self.summary_df.to_csv(path)


class BaseRobust(Protomodel):
    """
    A base class factoring out the repeated logic in OLSRobust and LRobust:
      - Basic validation (controls, group, etc.)
      - multiple_y support
      - parallel bootstrapping loops
      - SHAP logic
    """

    def __init__(self, *, y, x, data, model_name="BaseRobust"):
        super().__init__()
        # Basic validations
        if not isinstance(y, list) or not isinstance(x, list):
            raise TypeError("'y' and 'x' must be lists.")
        if not isinstance(data, pd.DataFrame):
            raise TypeError("'data' must be a pandas DataFrame.")
        all_vars = set(data.columns)
        if not all(var in all_vars for var in y) or not all(var in all_vars for var in x):
            raise ValueError("Variable names in 'y' and 'x' must exist in the provided DataFrame 'data'.")

        # Warn if missing data found
        if data.isnull().values.any():
            warnings.warn(
                "Missing values found in data. Listwise deletion will be applied.",
                MissingValueWarning
            )

        self.y = y
        self.x = x
        self.data = data
        self.model_name = model_name
        self.results = None
        self.parameters = {"y": self.y, "x": self.x}
    def get_results(self):
        raise NotImplementedError("This method should be implemented in subclasses.")
    def multiple_y(self):
        """
        Computes composite y based on multiple indicators provided.
        """
        self.y_specs = []
        self.y_composites = []
        print("Calculating Composite Ys")
        for spec, index in track(
            zip(all_subsets(self.y), range(0, space_size(self.y))),
            total=space_size(self.y)
        ):
            if len(spec) > 0:
                subset = self.data[list(spec)]
                subset = (subset - subset.mean()) / subset.std()  # standardize
                self.y_composites.append(subset.mean(axis=1))    # composite average
                self.y_specs.append(spec)
                self.parameters['y_specs'] = self.y_specs
                self.parameters['y_composites'] = self.y_composites
    
    def fit(self,
            *,
            controls,
            group=None,
            draws=500,
            kfold=5,
            oos_metric='r-squared',
            n_cpu=None,
            seed=None
            ):
        raise NotImplementedError("This method should be implemented in subclasses.")
    
    def _warn_if_large_draws(self, draws, controls, threshold=500):
        """
        Issues a warning if 'draws' * #specs is large.
        """
        est_specs = space_size(controls)
        total_models = est_specs * draws
        if draws > threshold:
            warnings.warn(
                f"You've requested {draws} bootstrap draws across {est_specs} specifications, "
                f"which is roughly {total_models:,} total model runs.\n\n"
                "This might lead to extended runtime or high memory usage.",
                UserWarning,
                stacklevel=2
            )

    def _check_numeric_columns_for_fit(self, controls, group):
        """
        Ensure columns are numeric.
        """
        cols_to_check = self.y + self.x + ( [group] if group else [] ) + controls
        _check_numeric_columns(self.data, cols_to_check)

    def _validate_fit_args(self,
                           controls,
                           group,
                           draws,
                           kfold,
                           oos_metric,
                           n_cpu,
                           seed,
                           valid_oos_metrics):
        """
        A shared validation method for the 'fit()' arguments used by both OLSRobust & LRobust.
        """
        # 1. Check controls type
        if not isinstance(controls, list):
            raise TypeError("'controls' must be a list.")

        # 2. Check that all controls exist in data
        all_vars = set(self.data.columns)
        if not all(var in all_vars for var in controls):
            raise ValueError("Variable names in 'controls' must exist in the provided DataFrame 'data'.")

        # 3. Group validation
        if group is not None:
            if group not in all_vars:
                raise ValueError("'group' variable must exist in the provided DataFrame 'data'.")
            if not isinstance(group, str):
                raise TypeError("'group' must be a string.")

        # 4. K-fold & draws
        if kfold < 2:
            raise ValueError(f"kfold values must be 2 or above, current value is {kfold}.")
        if draws < 1:
            raise ValueError(f"Draws value must be 1 or above, current value is {draws}.")

        # 5. OOS metric
        if oos_metric not in valid_oos_metrics:
            raise ValueError(f"OOS Metric must be one of {valid_oos_metrics}.")

        # 6. n_cpu
        if n_cpu is None:
            n_cpu = max(1, cpu_count()-1)
        else:
            if not isinstance(n_cpu, int):
                raise TypeError("n_cpu must be an integer")
            else:
                if (n_cpu <= 0) or (n_cpu > cpu_count()):
                    raise ValueError(f"n_cpu not in a valid range: pick between 0 and {cpu_count()}.")

        if seed is not None:
            if not isinstance(seed, int):
                raise TypeError("seed must be an integer")
            np.random.seed(seed)

        # 8. numeric columns check
        cols_to_check = self.y + self.x + ([group] if group else []) + controls
        _check_numeric_columns(self.data, cols_to_check)

        # 9. warn if large draws
        self._warn_if_large_draws(draws, controls, threshold=500)

        if len(self.x) == 0:
            raise ValueError("No independent variables (x) provided.")
        
        if len(set(controls) & set(self.x)) > 0:
            raise ValueError("Some control variables overlap with independent variables (x). Please ensure 'x' and 'controls' are disjoint sets.")

        if any(var in self.y for var in self.x):
            raise ValueError("Dependent variable(s) must not be included in the independent variables (x).")
        
        if len(self.x) == 0:
            raise ValueError("No independent variables (x) provided.")
        return n_cpu


class OLSRobust(BaseRobust):
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

    def __init__(self, *, y, x, data, model_name='OLS Robust'):
        super().__init__(y=y, x=x, data=data, model_name=model_name)

    def get_results(self):
        """
        Return the OLSResult object once .fit() has been called.
        """
        return self.results
    
    def fit(self,
            *,
            controls,
            group=None,
            draws=500,
            kfold=5,
            oos_metric='r-squared',
            n_cpu=None,
            seed=None
            ):
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

        Returns
        -------
        self : Object
            Object class OLSRobust containing the fitted estimators.
        """
        n_cpu = self._validate_fit_args(
            controls=controls,
            group=group,
            draws=draws,
            kfold=kfold,
            oos_metric=oos_metric,
            n_cpu=n_cpu,
            seed=seed,
            valid_oos_metrics=['r-squared','rmse']
        )
        print(f'[OLSRobust] Running with n_cpu={n_cpu}, draws={draws}')

        sample_size = self.data.shape[0]
        self.oos_metric_name = oos_metric

        if len(self.y) > 1:
            self.multiple_y()
            list_all_predictors = []
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
            specs = []
            for y, y_name in track(zip(self.y_composites,
                                 self.y_specs), total=len(self.y_composites)):
                space_n = space_size(controls)
                b_array = np.empty([space_n, draws])
                p_array = np.empty([space_n, draws])
                b_array_ystar = np.empty([space_n, draws])
                p_array_ystar = np.empty([space_n, draws])
                r2_array = np.empty([space_n, draws])
                r2i_array = np.empty([space_n])
                ll_array = np.empty([space_n])
                aic_array = np.empty([space_n])
                bic_array = np.empty([space_n])
                hqic_array = np.empty([space_n])
                all_predictors = []
                b_all_list = []
                p_all_list = []
                av_k_metric_array = np.empty([space_n])
                for spec, index in zip(all_subsets(controls), range(0, space_n)):
                    if len(spec) == 0:
                        comb = self.data[self.x]
                    else:
                        comb = self.data[self.x + list(spec)]
                    if group:
                        comb = self.data[self.x + [group] + list(spec)]

                    comb = pd.concat([y, comb], axis=1)
                    comb = comb.dropna()
                    comb = comb.reset_index(drop=True).copy()

                    if group:
                        comb = group_demean(comb, group=group)
                    (b_all, p_all, r2_i, ll_i,
                     aic_i, bic_i, hqic_i,
                     av_k_metric_i) = self._full_sample_OLS(comb,
                                                            kfold=kfold,
                                                            group=group,
                                                            oos_metric_name=self.oos_metric_name)
                    y_star = comb.iloc[:, [0]] - np.dot(comb.iloc[:, [1]], b_all[0][0])
                    seeds = np.random.randint(0, 2**31, size=draws)
                    b_list, p_list, r2_list, b_list_ystar, p_list_ystar = zip(*Parallel(n_jobs=n_cpu)
                    (delayed(self._strap_OLS)
                     (comb,
                      group,
                      sample_size,
                      seed,
                      y_star
                      )
                     for seed in seeds))
                    y_names.append(y_name)
                    specs.append(frozenset(list(y_name) + list(spec)))
                    all_predictors.append(self.x + list(spec) + ['const'])
                    b_array[index, :] = b_list
                    p_array[index, :] = p_list
                    b_array_ystar[index, :] = b_list_ystar
                    p_array_ystar[index, :] = p_list_ystar
                    r2_array[index, :] = r2_list
                    r2i_array[index] = r2_i
                    ll_array[index] = ll_i
                    aic_array[index] = aic_i
                    bic_array[index] = bic_i
                    hqic_array[index] = hqic_i
                    av_k_metric_array[index] = av_k_metric_i
                    b_all_list.append(b_all)
                    p_all_list.append(p_all)

                list_all_predictors.append(all_predictors)
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
                all_predictors=list_all_predictors,
                controls=controls,
                draws=draws,
                kfold=kfold,
                all_b=b_all_list,
                all_p=p_all_list,
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
            r2i_array = np.empty([space_n])
            ll_array = np.empty([space_n])
            aic_array = np.empty([space_n])
            bic_array = np.empty([space_n])
            hqic_array = np.empty([space_n])
            av_k_metric_array = np.empty([space_n])
            if group:
                SHAP_comb = self.data[self.y + self.x + [group] + controls]
                SHAP_comb = group_demean(SHAP_comb, group=group)
            else:
                SHAP_comb = self.data[self.y + self.x + controls]
            SHAP_comb = SHAP_comb.dropna()
            SHAP_comb = SHAP_comb.reset_index(drop=True).copy()
            x_train, x_test, y_train, _ = train_test_split(SHAP_comb[self.x + controls],
                                                           SHAP_comb[self.y],
                                                           test_size=0.2,
                                                           random_state=seed
                                                           )
            model = sklearn.linear_model.LinearRegression()
            model.fit(x_train, y_train)
            explainer = shap.LinearExplainer(model, x_train)
            shap_return = [explainer.shap_values(x_test), x_test]
            for spec, index in track(zip(all_subsets(controls), range(0, space_n)), total=space_n):
                if 0 == len(spec):
                    comb = self.data[self.y + self.x]
                else:
                    comb = self.data[self.y + self.x + list(spec)]
                if group:
                    comb = self.data[self.y + self.x + [group] + list(spec)]

                comb = comb.dropna()
                comb = comb.reset_index(drop=True).copy()

                if group:
                    comb = group_demean(comb, group=group)
                (b_all, p_all, r2_i, ll_i,
                 aic_i, bic_i, hqic_i,
                 av_k_metric_i) = self._full_sample_OLS(comb,
                                                        kfold=kfold,
                                                        group=group,
                                                        oos_metric_name=self.oos_metric_name)
                y_star = comb.iloc[:, [0]] - np.dot(comb.iloc[:, [1]], b_all[0][0])
                seeds = np.random.randint(0, 2 ** 32 - 1, size=draws)
                b_list, p_list, r2_list, b_list_ystar, p_list_ystar  = zip(*Parallel(n_jobs=n_cpu)
                (delayed(self._strap_OLS)
                 (comb,
                  group,
                  sample_size,
                  seed,
                  y_star)
                 for seed in seeds))

                specs.append(frozenset(spec))
                all_predictors.append(self.x + list(spec) + ['const'])
                b_array[index, :] = b_list
                p_array[index, :] = p_list
                b_array_ystar[index, :] = b_list_ystar
                p_array_ystar[index, :] = p_list_ystar
                r2_array[index, :] = r2_list
                r2i_array[index] = r2_i
                ll_array[index] = ll_i
                aic_array[index] = aic_i
                bic_array[index] = bic_i
                hqic_array[index] = hqic_i
                av_k_metric_array[index] = av_k_metric_i
                b_all_list.append(b_all)
                p_all_list.append(p_all)
            results = OLSResult(y=self.y[0],
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
                                shap_return=shap_return)
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
                         kfold,
                         group,
                         oos_metric_name):
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
        r2: float
            R2 value for the model
        AIC : float
            Akaike information criteria value for the model.
        BIC : float
            Bayesian information criteria value for the model.
        HQIC : float
            Hannan-Quinn information criteria value for the model.
        """
        y = comb_var.iloc[:, [0]]
        x_temp = comb_var.drop(comb_var.columns[0], axis=1)
        if group:
            x = x_temp.drop(columns=group)
        else:
            x = x_temp
        out = simple_ols(y=y,
                         x=x)
        av_k_metric = None
        if kfold:
            if group:
                k_fold = GroupKFold(kfold)
                metric = []
                for k, (train, test) in enumerate(k_fold.split(x, y, groups=x_temp[group])):
                    out_k = simple_ols(y=y.loc[train],
                                       x=x.loc[train])
                    y_pred = self._predict(x.loc[test], out_k['b'])
                    y_true = y.loc[test]
                    if oos_metric_name == 'rmse':
                        k_rmse = root_mean_squared_error(y_true, y_pred)
                        metric.append(k_rmse)
                    elif oos_metric_name == 'r-squared':
                        k_r2 = r2_score(y_true, y_pred)
                        metric.append(k_r2)
                    else:
                        raise ValueError('No valid OOS metric provided.')
                av_k_metric = np.mean(metric)
            else:
                k_fold = KFold(kfold)
                metric = []
                for k, (train, test) in enumerate(k_fold.split(x, y)):
                    out_k = simple_ols(y=y.loc[train],
                                       x=x.loc[train])
                    y_pred = self._predict(x.loc[test], out_k['b'])
                    y_true = y.loc[test]
                    if oos_metric_name == 'rmse':
                        k_rmse = root_mean_squared_error(y_true, y_pred)
                        metric.append(k_rmse)
                    elif oos_metric_name == 'r-squared':
                        k_r2 = r2_score(y_true, y_pred)
                        metric.append(k_r2)
                    else:
                        raise ValueError('No valid OOS metric provided.')
                av_k_metric = np.mean(metric)
        return (out['b'],
                out['p'],
                out['r2'],
                out['ll'][0][0],
                out['aic'][0][0],
                out['bic'][0][0],
                out['hqic'][0][0],
                av_k_metric)

    def _strap_OLS(self,
                   comb_var,
                   group,
                   sample_size,
                   seed,
                   y_star):
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

        Returns
        -------
        beta : float
            Estimate for x.
        p : float
            P value for x.
        """
        temp_data = comb_var.copy()
        temp_data['y_star'] = y_star

        if group is None:
            samp_df = temp_data.sample(n=sample_size, replace=True, random_state=seed)
            # @TODO generalize the frac to the function call
            y = samp_df.iloc[:, [0]]
            y_star = samp_df.iloc[:, [-1]]
            x = samp_df.drop('y_star', axis=1)
            x = x.drop(samp_df.columns[0], axis=1)
        else:
            np.random.seed(seed)
            idx = np.random.choice(temp_data[group].unique(), sample_size)
            select = temp_data[temp_data[group].isin(idx)]
            no_singleton = select[select.groupby(group).transform('size') > 1]
            if len(no_singleton) < 5:
                warnings.warn(
                    f"Bootstrap sample size is only {len(no_singleton)} after removing singleton groups "
                    f"(groups with a single observation). This may lead to unstable or unreliable estimates.",
                    UserWarning
                )
            no_singleton = no_singleton.drop(columns=[group])
            y = no_singleton.iloc[:, [0]]
            y_star = no_singleton.iloc[:, no_singleton.columns.get_loc('y_star')]
            y_star = y_star.to_frame()
            x = no_singleton.drop('y_star', axis=1)
            x = x.drop(no_singleton.columns[0], axis=1)
        output = stripped_ols(y=y, x=x)
        output_ystar = stripped_ols(y=y_star, x=x)
        b = output['b']
        p = output['p']
        r2 = output['r2']
        b_ystar = output_ystar['b']
        p_ystar = output_ystar['p']
        return b[0][0], p[0][0], r2, b_ystar[0][0], p_ystar[0][0]

class LRobust(BaseRobust):
    """
    A class to perform logistic regression analysis, underlying lr package = statsmodel

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

    def __init__(self, *, y, x, data, model_name='Logistic Regression Robust'):
        super().__init__(y=y, x=x, data=data, model_name=model_name)

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

    def _full_sample(self, comb_var, kfold, group, oos_metric_name):
        """
        Call logistic_regression_sm_stripped() over the full data containing y, x, and controls.

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
        r2: float
            r2 value for the full model
        AIC : float
            Akaike information criteria value for the model.
        BIC : float
            Bayesian information criteria value for the model.
        HQIC : float
            Hannan-Quinn information criteria value for the model.
        """
        # TODO Fixed effects Logistic Regression?
        y = comb_var.iloc[:, [0]]
        x = comb_var.drop(comb_var.columns[0], axis=1)
        out = logistic_regression_sm(y=y, x=x)
        av_k_metric = None

        if kfold:
            k_fold = KFold(kfold)
            metric = []
            for k, (train, test) in enumerate(k_fold.split(x, y)):
                out_k = logistic_regression_sm(y=y.loc[train], x=x.loc[train])
                y_pred = self._predict_LR(x.loc[test], out_k['b'])
                y_true = y.loc[test]
                if oos_metric_name == 'rmse':
                    k_rmse = root_mean_squared_error(y_true, y_pred)
                    metric.append(k_rmse)
                elif oos_metric_name == 'r-squared':
                    k_r2 = r2_score(y_true, y_pred)
                    metric.append(k_r2)
                elif oos_metric_name == 'cross-entropy':
                    k_cross_entropy = log_loss(y_true, y_pred)
                    metric.append(k_cross_entropy)
                else:
                    raise ValueError('No valid OOS metric provided.')
            av_k_metric = np.mean(metric)
        return (out['b'],
                out['p'],
                out['r2'],
                out['ll'],
                out['aic'],
                out['bic'],
                out['hqic'],
                av_k_metric)

    def _predict_LR(self, x_test, betas):
        """
        Predict the dependent variable using the estimated coefficients.
        """
        x_test = add_constant(x_test, prepend=False)
        return 1 / (1 + np.exp(-x_test.dot(betas)))

    def fit(self,
            *,
            controls,
            group=None,
            draws=500,
            sample_size=None,
            kfold=5,
            oos_metric='r-squared',
            n_cpu=None,
            seed=None):
        """Fit the logistic regression models over the specification space and bootstrap samples."""
        n_cpu = self._validate_fit_args(
            controls=controls,
            group=group,
            draws=draws,
            kfold=kfold,
            oos_metric=oos_metric,
            n_cpu=n_cpu,
            seed=seed,
            valid_oos_metrics=['r-squared','rmse','cross-entropy']
        )
        print(f'[LRobust] Running with n_cpu={n_cpu}, draws={draws}')


        self.oos_metric_name = oos_metric

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
            b_array_ystar = np.empty([space_n, draws])
            p_array_ystar = np.empty([space_n, draws])
            r2_array = np.empty([space_n, draws])
            r2i_array = np.empty([space_n])
            ll_array = np.empty([space_n])
            aic_array = np.empty([space_n])
            bic_array = np.empty([space_n])
            hqic_array = np.empty([space_n])
            av_k_metric_array = np.empty([space_n])
            if group:
                SHAP_comb = self.data[self.y + self.x + [group] + controls]
                SHAP_comb = group_demean(SHAP_comb, group=group)
            else:
                SHAP_comb = self.data[self.y + self.x + controls]
            SHAP_comb = SHAP_comb.dropna()
            SHAP_comb = SHAP_comb.reset_index(drop=True).copy()

            x_train, x_test, y_train, _ = train_test_split(SHAP_comb[self.x + controls],
                                                           SHAP_comb[self.y],
                                                           test_size=0.2,
                                                           random_state=seed
                                                           )
            model = sklearn.linear_model.LogisticRegression(penalty="l2", C=0.1)
            model.fit(x_train, y_train.squeeze())
            explainer = shap.LinearExplainer(model, x_train)
            shap_return = [explainer.shap_values(x_test), x_test]
            for spec, index in track(zip(all_subsets(controls), range(0, space_n)), total=space_n):

                if len(spec) == 0:
                    comb = self.data[self.y + self.x]
                else:
                    comb = self.data[self.y + self.x + list(spec)]

                if group:
                    comb = self.data[self.y + self.x + [group] + list(spec)]

                comb = comb.dropna()
                comb = comb.reset_index(drop=True).copy()

                if group:
                    comb = group_demean(comb, group=group)
                (b_all, p_all, r2_i, ll_i,
                 aic_i, bic_i, hqic_i,
                 av_k_metric_i) = self._full_sample(comb, kfold=kfold, group=group, oos_metric_name=self.oos_metric_name)
                seeds = np.random.randint(0, 2 ** 32 - 1, size=draws)
                (b_list, p_list, r2_list,
                 )= zip(*Parallel(n_jobs=n_cpu)
                (delayed(self._strap_regression)
                 (comb,
                  group,
                  sample_size,
                  seed
                  )
                 for seed in seeds))

                specs.append(frozenset(spec))
                all_predictors.append(self.x + list(spec) + ['const'])
                b_array[index, :] = b_list
                p_array[index, :] = p_list
                b_array_ystar[index, :] = np.nan*len(b_list)
                p_array_ystar[index, :] = np.nan*len(p_list)
                r2_array[index, :] = r2_list
                r2i_array[index] = r2_i
                ll_array[index] = ll_i
                aic_array[index] = aic_i
                bic_array[index] = bic_i
                hqic_array[index] = hqic_i
                av_k_metric_array[index] = av_k_metric_i
                b_all_list.append(b_all)
                p_all_list.append(p_all)

            self.results = OLSResult(y=self.y[0],
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
                                shap_return = shap_return
                                )

    def _strap_regression(self, comb_var, group, sample_size, seed):
        """Run bootstrap logistic regression on a random sample of the data."""
        temp_data = comb_var.copy()
        if group is None:
            samp_df = temp_data.sample(n=sample_size, replace=True, random_state=seed)
            y = samp_df.iloc[:, [0]]
            x = samp_df.drop(samp_df.columns[0], axis=1)
        else:
            np.random.seed(seed)
            idx = np.random.choice(temp_data[group].unique(), sample_size)
            select = temp_data[temp_data[group].isin(idx)]
            no_singleton = select[select.groupby(group).transform('size') > 1]
            no_singleton = no_singleton.drop(columns=[group])
            y = no_singleton.iloc[:, [0]]
            x = no_singleton.drop(no_singleton.columns[0], axis=1)
        output = logistic_regression_sm(y, x)
        return output['b'][0][0], output['p'][0][0], output['r2']