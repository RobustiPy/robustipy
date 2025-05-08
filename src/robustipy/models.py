"""robustipy.models

This module implements multivariate regression classes for Robust Inference.
It includes classes for OLS (OLSRobust and OLSResult) and logistic regression (LRobust)
analysis, along with utilities for model merging, plotting, and Bayesian model averaging.
"""

from __future__ import annotations
import _pickle
import warnings
from typing import Any, Optional, Sequence, List, Tuple, Union

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
from robustipy.prototypes import Protoresult, BaseRobust
from robustipy.utils import (
    all_subsets,
    group_demean,
    logistic_regression_sm,
    simple_ols,
    space_size,
    mcfadden_r2,
    calculate_imv_score
)
def stouffer_method(
    p_values: Sequence[float],
    weights: Optional[Sequence[float]] = None,
    eps: Optional[float] = None,
) -> Tuple[float, float]:
    """
    Combine p-values using Stouffer's Z-method, guarding against ±∞.

    Parameters
    ----------
    p_values : sequence of float
        Iterable of individual p-values.
    weights : sequence of float, optional
        Stouffer weights. If None, performs unweighted combination.
    eps : float, optional
        Positive clipping constant.  Default is the smallest positive
        normal IEEE-754 double (≈2.22e-308).

    Returns
    -------
    Z : float
        Combined Z-score.
    p_combined : float
        Combined p-value.
    """
    p = np.asarray(p_values, dtype=float)

    if eps is None:
        eps = np.finfo(float).eps  # 2**-52 ≈ 2.22e-16, big enough for safety
    p = np.clip(p, eps, 1.0 - eps)

    z = norm.isf(p)

    if weights is None:
        Z = z.sum() / np.sqrt(len(z))
    else:
        w = np.asarray(weights, dtype=float)
        if w.shape != z.shape:
            raise ValueError("weights and p_values must have same length")
        Z = np.dot(w, z) / np.linalg.norm(w)

    p_combined = norm.sf(Z)
    return Z, p_combined


class MergedResult(Protoresult):

    """
    Combine and summarize results exclusively from one or more OLSResult runs.

    Parameters
    ----------
    y : str
        Dependent variable name shared by all merged results.
    specs : Sequence[Sequence[str]]
        List of specifications; each inner sequence names the controls defining one spec.
    estimates : array-like or pandas.DataFrame
        Coefficient estimates for each spec (rows) and bootstrap draw (columns).
    p_values : array-like or pandas.DataFrame
        Corresponding p-values for each estimate.
    r2_values : array-like or pandas.DataFrame
        R² values for each spec and draw.

    Attributes
    ----------
    y_name : str
        Name of the dependent variable.
    specs_names : pandas.Series[frozenset]
        Each entry is the set of control variables defining a spec.
    estimates : pandas.DataFrame
        Coefficient estimates by spec and draw.
    p_values : pandas.DataFrame
        P-values by spec and draw.
    r2_values : pandas.DataFrame
        R² values by spec and draw.
    summary_df : pandas.DataFrame
        Per-spec summary with median, min, max, and 95% confidence intervals.
    """
    def __init__(
        self,
        *,
        y: str,
        specs: Sequence[Sequence[str]],
        estimates: Union[np.ndarray, pd.DataFrame],
        p_values: Union[np.ndarray, pd.DataFrame],
        r2_values: Union[np.ndarray, pd.DataFrame],
    ) -> None:
        super().__init__()
        self.y_name = y
        self.specs_names = pd.Series(specs)
        self.estimates = pd.DataFrame(estimates)
        self.p_values = pd.DataFrame(p_values)
        self.r2_values = pd.DataFrame(r2_values)
        self.summary_df = self._compute_summary()
        self.summary_df['spec_name'] = self.specs_names

    def summary(self) -> None:
        """
        Generates a summary of the regression results (not implemented).
        """
        pass

    def _compute_summary(self) -> pd.DataFrame:
        """
        Computes summary statistics based on coefficient estimates.

        Returns:
            pd.DataFrame: DataFrame containing median, min, max, and quantiles.
        """
        data = self.estimates.copy()
        out = pd.DataFrame()
        out['median'] = data.median(axis=1)
        out['max'] = data.max(axis=1)
        out['min'] = data.min(axis=1)
        out['ci_up'] = data.quantile(q=0.975, axis=1, interpolation='nearest')
        out['ci_down'] = data.quantile(q=0.025, axis=1, interpolation='nearest')
        return out

    def plot(
        self,
        loess: bool = True,
        specs: Optional[List[List[str]]] = None,
        colormap: str = 'Spectral_r',
        figsize: Tuple[int, int] = (16, 14),
        ext: str = 'pdf',
        project_name: str = 'no_project_name',
    ) -> plt.Figure:
        """
        Plot specification results highlighting up to three specs.

        Parameters
        ----------
        loess : bool
            Whether to apply LOESS smoothing to confidence intervals.
        specs : list of list of str, optional
            Specifications to highlight.
        colormap : str
            Matplotlib colormap name.
        figsize : tuple
            Figure size (width, height).
        ext : str
            File extension for saving.
        project_name : str
            Prefix for saved figure.

        Returns
        -------
        matplotlib.figure.Figure:
            Plot showing the regression results.
        """
        fig, ax = plt.subplots(figsize=figsize)

        if specs is not None and len(specs) == 0:
            specs = None
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
            ci=ci,
            specs=specs,
            ax=ax,
            colormap=colormap,
            ext=ext,
            project_name=project_name
        )
        return fig

    def merge(
        self,
        result_obj: OLSResult,
        left_prefix: str,
        right_prefix: str,
    ) -> MergedResult:
        """
        Merge the current OLSResult object with another, tagging each specification
        with a prefix to indicate origin.

        Parameters
        ----------
        result_obj : OLSResult
            Another OLSResult object to merge.
        left_prefix : str
            Prefix to tag the specifications from the current object.
        right_prefix : str
            Prefix to tag the specifications from the result_obj object.

        Returns
        -------
        MergedResult
            A new MergedResult object containing combined estimates and metadata.

        Raises
        ------
        TypeError
            If result_obj is not an instance of OLSResult or prefixes are not strings.
        ValueError
            If the dependent variable names do not match between the two objects.
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
    Encapsulates the results of an OLSRobust run

    Attributes
    ----------
    y_name : str
        Dependent variable name.
    x_name : str
        Main predictor name.
    data : pd.DataFrame
        Original DataFrame used for all fits.
    specs_names : pd.Series[frozenset[str]]
        Specification sets (which controls are included, etc.).
    all_predictors : list[list[str]]
        List of predictor+control sets for each specification.
    controls : list[str]
        Pool of all control variables considered.
    draws : int
        Number of bootstrap draws.
    kfold : int
        Number of folds for out-of-sample evaluation.
    estimates : pd.DataFrame
        Shape (n_specs, draws), bootstrap estimates of β₁.
    p_values : pd.DataFrame
        Same shape, bootstrap p-values for β₁.
    estimates_ystar : pd.DataFrame
        Bootstrap estimates under the null (for joint inference).
    p_values_ystar : pd.DataFrame
        Bootstrap p-values under the null.
    r2_values : pd.DataFrame
        Shape (n_specs, draws), bootstrapped R².
    summary_df : pd.DataFrame
        Per-spec summary (median, CI, info criteria, cross-val metric).
    inference : dict[str, Any]
        Aggregated inference statistics (proportions, Stouffer’s Z, etc.).
    shap_return : tuple[np.ndarray, pd.DataFrame] | None
        Optional SHAP values and the matrix they came from.
    """

    def __init__(
        self,
        *,
        y: str,
        x: str,
        data: pd.DataFrame,
        specs: list[frozenset[str]],
        all_predictors: list[list[str]],
        controls: list[str],
        draws: int,
        kfold: int,
        estimates: np.ndarray | pd.DataFrame,
        estimates_ystar: np.ndarray | pd.DataFrame,
        all_b: list[np.ndarray],
        all_p: list[np.ndarray],
        p_values: np.ndarray | pd.DataFrame,
        p_values_ystar: np.ndarray | pd.DataFrame,
        r2_values: np.ndarray | pd.DataFrame,
        r2i_array: list[float],
        ll_array: list[float],
        aic_array: list[float],
        bic_array: list[float],
        hqic_array: list[float],
        av_k_metric_array: list[float] | None = None,
        model_name: str,
        name_av_k_metric: str | None = None,
        shap_return: Any = None,
    ) -> None:
        """
        Initialize an OLSResult container.

        Parameters
        ----------
        y
            Dependent variable name.
        x
            Main predictor name.
        data
            Original DataFrame for all model fits.
        specs
            A list of frozensets indicating which controls each spec includes.
        all_predictors
            For each spec, the full list of predictors (x + controls).
        controls
            Pool of all candidate controls.
        draws
            Number of bootstrap draws.
        kfold
            Number of folds for cross-validation.
        estimates
            Bootstrap coefficient estimates (shape: n_specs × draws).
        estimates_ystar
            Bootstrap estimates under null (same shape).
        all_b
            Raw (non-resampled) β₁, one per spec.
        all_p
            Raw (non-resampled) p-values, one per spec.
        p_values
            Bootstrap p-values for β₁.
        p_values_ystar
            Bootstrap p-values under null.
        r2_values
            Bootstrapped R² (n_specs × draws).
        r2i_array
            In-sample R² for each spec.
        ll_array
            Log-likelihood for each spec.
        aic_array, bic_array, hqic_array
            Information criteria per spec.
        av_k_metric_array
            Out-of-sample metric (e.g. CV R²) per spec.
        model_name
            Human-readable name of the model (“OLS Robust”).
        name_av_k_metric
            Label for the CV metric (e.g. “r-squared”).
        shap_return
            Optional return of (shap_values, input_matrix).
        """
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
        self.summary_df['r2'] = pd.Series(r2i_array)
        self.summary_df['ll'] = pd.Series(ll_array)
        self.summary_df['aic'] = pd.Series(aic_array)
        self.summary_df['bic'] = pd.Series(bic_array)
        self.summary_df['hqic'] = pd.Series(hqic_array)
        self._compute_inference()
        self.summary_df['av_k_metric'] = pd.Series(av_k_metric_array)
        self.summary_df['spec_name'] = self.specs_names
        self.summary_df['y'] = self.y_name
        self.model_name = model_name
        self.name_av_k_metric = name_av_k_metric
        self.shap_return = shap_return


    def save(self, filename: str) -> None:
        """
        Pickle this OLSResult to disk.

        Parameters
        ----------
        filename
            Destination path for the pickle file.
        """
        with open(filename, 'wb') as f:
            _pickle.dump(self, f, -1)

    @classmethod
    def load(cls, filename: str) -> OLSResult:
        """
        Loads an OLSResult object from a pickle file.

        Parameters
        ----------
        filename
            Path to the pickle file.

        Returns
        -------
        OLSResult
        """
        with open(filename, 'rb') as f:
            return _pickle.load(f)


    def _compute_inference(self) -> pd.DataFrame:
        """
        Compute summary statistics of the bootstrap coefficient distribution.

        For each model specification, returns the median, minimum, maximum,
        and the 2.5%/97.5% quantile bounds of the estimated coefficient
        across all bootstrap draws.

        Returns
        -------
        summary_df : pandas.DataFrame
            A DataFrame indexed by specification, with columns:

            - median : float
                Median coefficient across bootstrap draws.
            - min : float
                Minimum coefficient across bootstrap draws.
            - max : float
                Maximum coefficient across bootstrap draws.
            - ci_down : float
                2.5th percentile coefficient (lower 95% bound).
            - ci_up : float
                97.5th percentile coefficient (upper 95% bound).
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
        for ic in ['aic', 'bic', 'hqic']:
            if max(len(t) for t in self.y_name) == 1:
                ic_array = np.array(self.summary_df[ic].to_list())
                all_b = [arr[0] for arr in self.all_b]
                coef_mat = np.vstack(all_b)
                delta = ic_array - ic_array.min()
                w = np.exp(-0.5 * delta)
                w /= w.sum()
                beta_avg = w @ coef_mat
                self.inference[ic + '_average'] = beta_avg[0]
            else:
                self.inference[ic + '_average'] = np.nan

        self.inference['median_p'] = (
            np.nan if not inference else
            (self.estimates_ystar.median(axis=0) > df_model_result['betas'].median()).mean()
        )
        self.inference['min_ns'] = df_model_result['betas'].min()
        self.inference['min'] = self.estimates.min().min()
        self.inference['max_ns'] = df_model_result['betas'].max()
        self.inference['max'] = self.estimates.max().max()

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


    def summary(self, digits=3) -> None:
        """
        Print a comprehensive textual summary of the fitted model.
        """

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
        if not inference:
            print(f"Median beta (all specifications, no resampling): {round(self.inference['median_ns'], digits)}")
        else:
            print(f"Median beta (all specifications, no resampling): {round(self.inference['median_ns'], digits)} (p-value: {round(self.inference['median_p'], digits)})")
        print(f"Median beta (all bootstraps and specifications): {round(self.inference['median'], digits)}")

        print(f"Min beta (all specifications, no resampling): {round(self.inference['min_ns'], digits)}")
        print(f"Min beta (all bootstraps and specifications): {round(self.inference['min'], digits)}")

        print(f"Max beta (all specifications, no resampling): {round(self.inference['max_ns'], digits)}")
        print(f"Max beta (all bootstraps and specifications): {round(self.inference['max'], digits)}")
        if self.inference['aic_average'] is not np.nan:
            print(f"AIC-weighted beta (all specifications, no resampling): {round(self.inference['aic_average'], digits)}")
        if self.inference['bic_average'] is not np.nan:
            print(f"BIC-weighted beta (all specifications, no resampling): {round(self.inference['bic_average'], digits)}")
        if self.inference['hqic_average'] is not np.nan:
            print(f"HQIC-weighted beta (all specifications, no resampling): {round(self.inference['hqic_average'], digits)}")

        if not inference:
            print(f"Significant portion of beta (all specifications, no resampling): {round(self.inference['sig_prop_ns'], digits)}")
        else:
            print(f"Significant portion of beta (all specifications, no resampling): {round(self.inference['sig_prop_ns'], digits)} (p-value: {round(self.inference['sig_p'], digits)})")
        print(f"Significant portion of beta (all bootstraps and specifications): {round(self.inference['sig_prop'], digits)}")

        if not inference:
            print(f"Positive portion of beta (all specifications, no resampling): {round(self.inference['pos_prop_ns'], digits)}")
        else:
            print(f"Positive portion of beta (all specifications, no resampling): {round(self.inference['pos_prop_ns'], digits)} (p-value: {round(self.inference['pos_p'], digits)})")
        print(f"Positive portion of beta (all bootstraps and specifications): {round(self.inference['pos_prop'], digits)}")

        if not inference:
            print(f"Negative portion of beta (all specifications, no resampling): {round(self.inference['neg_prop_ns'], digits)}")
        else:
            print(f"Negative portion of beta (all specifications, no resampling): {round(self.inference['neg_prop_ns'], digits)} (p-value: {round(self.inference['neg_p'], digits)})")
        print(f"Negative portion of beta (all bootstraps and specifications): {round(self.inference['neg_prop'], digits)}")

        if not inference:
            print(f"Positive and Significant portion of beta (all specifications, no resampling): {round(self.inference['pos_sig_prop_ns'], digits)}")
        else:
            print(f"Positive and Significant portion of beta (all specifications, no resampling): {round(self.inference['pos_sig_prop_ns'], digits)} (p-value: {round(self.inference['pos_sig_p'], digits)})")
        print(f"Positive and Significant portion of beta (all bootstraps and specifications): {round(self.inference['pos_sig_prop'], digits)}")

        if not inference:
            print(f"Negative and Significant portion of beta (all specifications, no resampling): {round(self.inference['neg_sig_prop_ns'], digits)}")
        else:
            print(f"Negative and Significant portion of beta (all specifications, no resampling): {round(self.inference['neg_sig_prop_ns'], digits)} (p-value: {round(self.inference['neg_sig_p'], digits)})")
        print(f"Negative and Significant portion of beta (all bootstraps and specifications): {round(self.inference['neg_sig_prop'], digits)}")

        print(f"Stouffer's Z-score test: {round(self.inference['Stouffers'][0], digits)}, {round(self.inference['Stouffers'][1], digits)}")

        print_separator()
        print('2.2 In-Sample Metrics (Full Sample)')
        print_separator()
        print(f"Min AIC: {round(self.summary_df['aic'].min(), digits)}, Specs: {list(self.summary_df['spec_name'].loc[self.summary_df['aic'].idxmin()])}")
        print(f"Min BIC: {round(self.summary_df['bic'].min(), digits)}, Specs: {list(self.summary_df['spec_name'].loc[self.summary_df['bic'].idxmin()])}")
        print(f"Min HQIC: {round(self.summary_df['hqic'].min(), digits)}, Specs: {list(self.summary_df['spec_name'].loc[self.summary_df['hqic'].idxmin()])}")
        print(f"Max Log Likelihood: {round(self.summary_df['ll'].max(), digits)}, Specs: {list(self.summary_df['spec_name'].loc[self.summary_df['ll'].idxmax()])}")
        print(f"Min Log Likelihood: {round(self.summary_df['ll'].min(), digits)}, Specs: {list(self.summary_df['spec_name'].loc[self.summary_df['ll'].idxmin()])}")
        print(f"Max { 'Adj-' if "OLS" in self.model_name else 'Pseudo'} R2: {round(self.summary_df['r2'].max(), digits)}, Specs: {list(self.summary_df['spec_name'].loc[self.summary_df['r2'].idxmax()])}")
        print(f"Min { 'Adj-' if "OLS" in self.model_name else 'Pseudo'} R2: {round(self.summary_df['r2'].min(), digits)}, Specs: {list(self.summary_df['spec_name'].loc[self.summary_df['r2'].idxmin()])}")

        print_separator()
        print(f'2.3 Out-Of-Sample Metrics ({self.name_av_k_metric} averaged across folds)')
        print_separator()
        oos_max_row = self.summary_df.loc[self.summary_df['av_k_metric'].idxmax(),]
        print(f'Max Average: {round(oos_max_row["av_k_metric"], digits)}, Specs: {list(oos_max_row["spec_name"])} ')
        oos_min_row = self.summary_df.loc[self.summary_df['av_k_metric'].idxmin(),]
        print(f'Min Average: {round(oos_min_row["av_k_metric"], digits)}, Specs: {list(oos_min_row["spec_name"])} ')
        print(f"Mean Average: {round(self.summary_df['av_k_metric'].mean(), digits)}")
        print(f"Median Average: {round(self.summary_df['av_k_metric'].median(), digits)}")


    def plot(self,
             loess: bool = True,
             specs: Optional[List[List[str]]] = None,
             ic: str = 'aic',
             ci: float = 1,
             colormap: str = 'Spectral_r',
             figsize: Tuple[int, int] = (12, 6),
             ext: str = 'pdf',
             project_name: str = 'no_project_name'
             ) -> plt.Figure:
        """
        Plots the regression results using specified options.

        Parameters
        ----------
        loess : bool, default=True
            Whether to add a LOESS smoothed trend line.
        specs : list of list of str, optional
            Up to three specific model specifications to highlight.
        ic : {'bic', 'aic', 'hqic'}, default='aic'
            Which information criterion to display.
        ci: float, default=1
            confidence interval.
        colormap : str, default='Spectral_r'
            Name of the matplotlib colormap for the plot.
        figsize : tuple of int, default=(12, 6)
            Figure width and height in inches.
        ext : str, default='pdf'
            File extension if saving the figure (unused if not saving).
        project_name : str, default='no_project_name'
            Project identifier used in saved filename (unused if not saving).

        Returns
        -------
        fig : matplotlib.figure.Figure
            The figure object containing the plot.

        Raises
        ------
        ValueError
            If `ic` is not one of {'bic', 'aic', 'hqic'}.
        TypeError
            If `specs` is provided but is not a list of lists of str,
            or if more than three specs are given,
            or any spec is not in the computed specifications.
        """

        valid_ic = ['bic', 'aic', 'hqic']
        if ic not in valid_ic:
            raise ValueError(
                f"Unsupported information criterion: expected one of {valid_ic}, "
                f"received: '{ic}'."
            )


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
                            ci=ci,
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

    def compute_bma(self) -> pd.DataFrame:
        """
        Performs Bayesian Model Averaging (BMA) using BIC-implied priors.

        Returnsnt
        -------
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

    def merge(self,
              result_obj: "OLSResult",
              left_prefix: str,
              right_prefix: str
              ) -> MergedResult:
        """
        Merge this OLSResult with another, tagging each spec by prefix.

        Parameters
        ----------
        result_obj : OLSResult
            Another result object with the same dependent variable.
        left_prefix : str
            Tag to append to this object's specifications.
        right_prefix : str
            Tag to append to the other object's specifications.

        Returns
        -------
        merged : MergedResult
            A new MergedResult containing all specs, estimates,
            p_values, and r2_values from both.

        Raises
        ------
        TypeError
            If `result_obj` is not an OLSResult, or prefixes are not strings.
        ValueError
            If the dependent variable names do not match.
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

    def save_to_csv(self, path: str) -> None:
        """Function to save summary dataframe to a csv"""
        self.summary_df.to_csv(path)


class OLSRobust(BaseRobust):
    """
    Class for multi-variate regression using OLS

    Parameters
    ----------
    y : str or list of str
        Name(s) of the dependent variable(s). If multiple, runs separate analyses.
    x : str or list of str
        Name(s) of the primary predictor(s) of interest.
    data : pandas.DataFrame
        The full dataset containing `y`, `x`, and any controls.
    model_name : str, default='OLS Robust'
        A custom label for this model run, used in outputs and plots

    Attributes
    ----------
    results : OLSResult
        Populated after calling `.fit()`, contains all estimates and diagnostics.
    """

    def __init__(
        self,
        *,
        y: Union[str, List[str]],
        x: Union[str, List[str]],
        data: pd.DataFrame,
        model_name: str = 'OLS Robust'
    ) -> None:
        super().__init__(y=y, x=x, data=data, model_name=model_name)

    def get_results(self) -> 'OLSResult':
        """
        Return the OLSResult object once .fit() has been called.

        Returns
        -------
        OLSResult
            The result object encapsulating all analysis outputs.
        """
        return self.results

    def fit(
        self,
        *,
        controls: List[str],
        group: Optional[str] = None,
        draws: int = 500,
        kfold: int = 5,
        oos_metric: str = 'r-squared',
        n_cpu: Optional[int] = None,
        seed: Optional[int] = None,
<<<<<<< HEAD
        threshold: int = 10_000,
        composite_sample: Optional[int] = None
=======
        threshold: int = 1000000
>>>>>>> d8c6e16301666144ff400a427cb5231ede7424db
    ) -> 'OLSRobust':
        """
        Fit the OLS models into the specification space as well as over the bootstrapped samples.

        Parameters
        ----------
        controls : list of str
            Candidate control variables to include in model specifications.
        group : str, optional
            Column name for grouping fixed effects; de-meaned if provided.
        draws : int, default=500
            Number of bootstrap resamples per specification.
        kfold : int, default=5
            Number of folds for out-of-sample evaluation; requires `oos_metric`.
        oos_metric : {'r-squared','rmse'}, default='r-squared'
            Metric for cross-validated performance.
        n_cpu : int, optional
            Number of parallel jobs; defaults to all available cores minus one.
        seed : int, optional
            Random seed for reproducibility.
        threshold : int, default=1000000
            Warn if total model runs exceed this number.

        Returns
        -------
        self : OLSRobust
            The fitted model instance, with `.results` attached.
        """
        self.composite_sample = composite_sample      # int or None
        self.seed             = seed
        if len(self.y) > 1:
            self.multiple_y()
        n_cpu = self._validate_fit_args(
            controls=controls,
            group=group,
            draws=draws,
            kfold=kfold,
            oos_metric=oos_metric,
            n_cpu=n_cpu,
            seed=seed,
            valid_oos_metrics=['r-squared','rmse'],
            threshold=threshold
        )
        print(f'[OLSRobust] Running with n_cpu={n_cpu}, draws={draws}')

        sample_size = self.data.shape[0]
        self.oos_metric_name = oos_metric

        if len(self.y) > 1:
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


    def _predict(
        self,
        x_test: np.ndarray,
        betas: np.ndarray
    ) -> np.ndarray:
        """
        Predict the dependent variable based on the test data and coefficients.

        Parameters
        ----------
        x_test : array-like, shape=(n_obs, n_features)
            Independent variables.
        betas : array-like, shape=(n_features,)
            Regression coefficients.

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
        comb_var : pd.DataFrame
            Combined y, x, and controls dataset (cleaned).
        kfold : int
            Number of CV folds; if 0 or None, skips CV.
        group : str or None
            Grouping column for fixed effects (dropped before fit).
        oos_metric_name : str
            Out-of-sample metric ('r-squared' or 'rmse')..

        Returns
        -------
        b : np.ndarray
            Full-sample coefficient estimates.
        p : np.ndarray
            Full-sample p-values.
        r2 : float
            In-sample R².
        ll : float
            Log-likelihood.
        aic : float
            Akaike Information Criterion.
        bic : float
            Bayesian Information Criterion.
        hqic : float
            Hannan-Quinn Information Criterion.
        av_k_metric : float
            Average out-of-sample metric across folds.
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

    def _strap_OLS(
        self,
        comb_var: pd.DataFrame,
        group: Optional[str],
        sample_size: int,
        seed: int,
        y_star: np.ndarray
    ) -> tuple:
        """
        Call stripped_ols() over a random sample of the data containing y, x, and controls.

        Parameters
        ----------
        comb_var : pd.DataFrame
            Combined y, x, and controls dataset.
        group : str or None
            Grouping column for fixed effects; sampling respects groups if provided.
        sample_size : int
            Number of observations in bootstrap sample.
        seed : int
            Random seed for resampling.
        y_star : array-like
            Pseudo-outcome residuals for stratified bootstrap.

        Returns
        -------
        beta : float
            Bootstrap coefficient estimate for focal predictor.
        p : float
            Associated p-value.
        r2 : float
            In-sample R² for the bootstrap sample.
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
    y : str or List[str]
        Name(s) of the dependent binary variable(s). If multiple,
        runs separate analyses for each.
    x : str or List[str]
        Name(s) of the primary predictor(s) of interest.
    data : pandas.DataFrame
        The dataset containing `y`, `x`, and any optional controls.
    model_name : str, default='Logistic Regression Robust'
        Custom label for this model, used in outputs and plots.

    Attributes
    ----------
    results : OLSResult
        Populated after calling `.fit()`. Contains all coefficient,
        p-value, and metric outputs.
    """

    def __init__(
        self,
        *,
        y: Union[str, List[str]],
        x: Union[str, List[str]],
        data: pd.DataFrame,
        model_name: str = 'Logistic Regression Robust'
    ) -> None:
        """
        Initialize the logistic robust analysis.

        Parameters
        ----------
        y : str or list of str
            Dependent variable name(s).
        x : str or list of str
            Predictor variable name(s).
        data : pandas.DataFrame
            Input dataset.
        model_name : str, optional
            Descriptive name for the model. Default 'Logistic Regression Robust'.
        """
        super().__init__(y=y, x=x, data=data, model_name=model_name)

    def get_results(self) -> Any:
        """
        Get the results of the OLS regression.

        Returns
        -------
        results : OLSResult
            Object containing the regression results.
        """
        return self.results

    def multiple_y(self) -> None:
        raise NotImplementedError("Not implemented yet")

    def _full_sample(
        self,
        comb_var: pd.DataFrame,
        kfold: int,
        group: Optional[str],
        oos_metric_name: str
    ) -> Tuple[np.ndarray, np.ndarray, float, float, float, float, float, Optional[float]]:
        """
        Call logistic_regression_sm_stripped() over the full data containing y, x, and controls.

        Parameters
        ----------
        comb_var : pandas.DataFrame
            DataFrame with columns [y, x, controls...].
        kfold : int
            Number of folds for cross-validation.
        group : str or None
            Grouping variable name (for future FE support).
        oos_metric_name : str
            Out-of-sample metric: 'r-squared', 'rmse', or 'cross-entropy'.


        Returns
        -------
        beta : ndarray
            Coefficient estimates.
        p : ndarray
            P-value estimates.
        r2 : float
            In-sample pseudo-R².
        ll : float
            Log-likelihood.
        aic : float
            Akaike Information Criterion.
        bic : float
            Bayesian Information Criterion.
        hqic : float
            Hannan-Quinn Information Criterion.
        av_k_metric : float or None
            Average cross-validation metric, if `kfold>1`, else None.
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
                    k_r2 = mcfadden_r2(y_true, y_pred)
                    metric.append(k_r2)
                elif oos_metric_name == 'cross-entropy':
                    k_cross_entropy = log_loss(y_true, y_pred)
                    metric.append(k_cross_entropy)
                elif oos_metric_name == 'imv':
                    imv = calculate_imv_score(y_true, y_pred)
                    metric.append(imv)
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

    def _predict_LR(
        self,
        x_test: pd.DataFrame,
        betas: np.ndarray
    ) -> np.ndarray:
        """
        Predict the dependent variable using the estimated coefficients.

        Parameters
        ----------
        x_test : pandas.DataFrame
            Test set covariates (no constant column).
        betas : ndarray
            Fitted coefficients (including intercept at end).

        Returns
        -------
        ndarray
            Predicted probabilities for the positive class.
        """
        x_test = add_constant(x_test, prepend=False)
        return 1 / (1 + np.exp(-x_test.dot(betas)))


    def fit(
        self,
        *,
        controls: List[str],
        group: Optional[str] = None,
        draws: int = 500,
        sample_size: Optional[int] = None,
        kfold: int = 5,
        oos_metric: str = 'r-squared',
        n_cpu: Optional[int] = None,
        seed: Optional[int] = None,
        threshold: int = 1000000
    ) -> 'LRobust':
        """
        Fit the logistic regression models over the specification space and bootstrap samples.

        Parameters
        ----------
        controls : list of str
            Names of optional control variables to include in every spec.
        group : str, optional
            *Currently ignored.* Placeholder for future fixed-effect demeaning.
        draws : int, default=500
            Number of bootstrap resamples per specification.
        sample_size : int, optional
            Number of observations per bootstrap draw; defaults to full dataset.
        kfold : int, default=5
            Folds for out-of-sample CV; set to 0 to disable.
        oos_metric : {'r-squared','rmse','cross-entropy'}, default='r-squared'
            Metric to compute on held-out folds.
        n_cpu : int, optional
            Number of parallel jobs; defaults to all available.
        seed : int, optional
            Random seed for reproducibility.
        threshold : int, default=1000000
            Warn if `draws * n_specs` exceeds this.

        Returns
        -------
        self : LRobust
            Self, with `.results` populated as an `OLSResult`.
        """
        n_cpu = self._validate_fit_args(
            controls=controls,
            group=group,
            draws=draws,
            kfold=kfold,
            oos_metric=oos_metric,
            n_cpu=n_cpu,
            seed=seed,
            valid_oos_metrics=['r-squared','rmse','cross-entropy','imv'],
            threshold=threshold
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

    def _strap_regression(
        self,
        comb_var: pd.DataFrame,
        group: Optional[str],
        sample_size: int,
        seed: int
    ) -> Tuple[float, float, float]:
        """
        Perform one bootstrap draw of logistic regression.

        Parameters
        ----------
        comb_var : pandas.DataFrame
            DataFrame with `y` in first column and features in remaining columns.
        group : str, optional
            *Ignored.* Placeholder for future grouping logic.
        sample_size : int
            Number of rows to sample with replacement.
        seed : int
            Random seed for sampling.

        Returns
        -------
        beta : float
            Estimated coefficient for the primary predictor.
        p : float
            Corresponding p-value.
        r2 : float
            In-sample pseudo-R² of this bootstrap sample.
        """
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

