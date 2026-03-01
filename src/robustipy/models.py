"""robustipy.models

This module implements multivariate regression classes for Robust Inference.
It includes classes for OLS (OLSRobust and OLSResult) and logistic regression (LRobust)
analysis, along with utilities for model merging, plotting, and Bayesian model averaging.
"""

from __future__ import annotations
import _pickle
import warnings
import inquirer
from inquirer.themes import GreenPassion
from inquirer.errors import ValidationError
from typing import Any, Optional, Sequence, List, Tuple, Union
from multiprocessing import cpu_count

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
import sys
import sklearn
from joblib import Parallel, delayed
from rich.progress import track
from scipy.stats import norm
from sklearn.metrics import log_loss, root_mean_squared_error
from sklearn.model_selection import GroupKFold, KFold, train_test_split
from statsmodels.tools.tools import add_constant

from robustipy.figures import plot_results
from robustipy.prototypes import Protoresult, BaseRobust
from robustipy.utils import (
    all_subsets,
    group_demean,
    logistic_regression_sm,
    simple_ols,
    space_size,
    mcfadden_r2,
    pseudo_r2,
    calculate_imv_score,
    sample_z_masks,
    _ensure_single_constant,
    _find_group_invariant_columns,
    rescale,
    make_inquiry,
    stripped_ols
)
import warnings
from typing import List, Tuple
import pandas as pd


import warnings
import pandas as pd
from typing import List, Tuple


def stouffer_method(
    p_values,
    *,
    two_sided=True,
    betas=None,                 # required if two_sided=True (for direction)
    weights=None,               # optional Stouffer weights
    rho=None,                   # optional common correlation among z's
    Sigma=None,                 # optional full covariance of z; overrides rho
    clip_floor=1e-300,          # must be >= np.finfo(float).tiny
    na_action="omit",
    warn: bool = True
):
    """
    Combine p-values via Stouffer's Z-method with numerical safeguards.
    Returns (Z, p_one_sided) where p_one_sided = sf(Z).
    """
    import numpy as np
    from scipy.stats import norm

    # ---- 0) inputs to arrays
    p_all = np.asarray(p_values, dtype=np.float64)

    # ---- 1) validate p and optionally omit invalid
    valid = np.isfinite(p_all) & (p_all >= 0.0) & (p_all <= 1.0)
    if not valid.all():
        if na_action == "raise":
            bad = np.where(~valid)[0].tolist()
            raise ValueError(f"Invalid p-values at indices {bad}.")
        p = p_all[valid]
        if p.size == 0:
            raise ValueError("No valid p-values after filtering.")
    else:
        p = p_all
    m = p.size

    # ---- 2) prepare aligned betas / weights (after omission)
    if two_sided:
        if betas is None:
            raise ValueError("`betas` must be provided when two_sided=True.")
        b_all = np.asarray(betas, dtype=np.float64)
        b = b_all[valid] if b_all.shape != p_all.shape else b_all
        if b.shape != p.shape:
            raise ValueError("`betas` length must match the number of valid p-values.")
        signs = np.sign(b)  # zeros give sign 0 -> z=0
    else:
        signs = None

    if weights is None:
        w = np.ones(m, dtype=np.float64)
    else:
        w_all = np.asarray(weights, dtype=np.float64)
        w = w_all[valid] if w_all.shape != p_all.shape else w_all
        if w.shape != (m,):
            raise ValueError("`weights` must have the same length as valid p-values.")
        if not np.all(np.isfinite(w)):
            raise ValueError("`weights` must be finite.")
        if not np.any(w != 0.0):
            raise ValueError("At least one weight must be nonzero.")

    # ---- 3) clip p away from {0,1} at a safe floor/ceiling
    floor_min = np.finfo(float).tiny
    if clip_floor is None or not np.isfinite(clip_floor) or clip_floor <= 0.0:
        clip_floor = floor_min
    else:
        clip_floor = max(float(clip_floor), floor_min)
    clip_ceiling = 1.0 - 1e-16

    n_zeros = int(np.sum(p == 0.0))
    n_ones  = int(np.sum(p == 1.0))
    if warn and (n_zeros or n_ones):
        print(f"WARNING: clipping {n_zeros} p=0 and {n_ones} p=1 to "
              f"[{clip_floor:g}, {clip_ceiling:g}].")
    p = np.clip(p, clip_floor, clip_ceiling)

    # ---- 4) map to z-scores (two-sided with direction vs one-sided)
    if two_sided:
        z = signs * norm.isf(p / 2.0)
    else:
        z = norm.isf(p)

    # Guard against any residual inf/-inf due to extreme p
    if not np.all(np.isfinite(z)):
        # Numerical cap: Φ^{-1}(1 - 5e-300) ≈ 37
        Z_MAX = 37.0
        if warn:
            print("WARNING: non-finite z encountered; clipping z to ±37.")
        z = np.clip(z, -Z_MAX, Z_MAX)

    # ---- 5) variance denominator under dependence
    if Sigma is not None:
        Sigma = np.asarray(Sigma, dtype=np.float64)
        if Sigma.shape != (m, m):
            raise ValueError("`Sigma` must be an (m×m) covariance for the valid tests.")
        denom2 = float(w @ (Sigma @ w))
    elif rho is not None:
        rho = float(rho)
        if not (-1.0/(m-1) < rho < 1.0):
            raise ValueError(f"`rho` must lie in (-1/(m-1), 1); got {rho}.")
        # Equal-correlation Σ = (1-ρ)I + ρ 11^T
        denom2 = (1.0 - rho) * float(w @ w) + rho * float(np.sum(w))**2
    else:
        denom2 = float(w @ w)

    if not np.isfinite(denom2) or denom2 <= 0.0:
        raise ValueError("Nonpositive or nonfinite variance under supplied Σ/ρ/weights.")

    # ---- 6) combined statistic and one-sided p
    Z = float(np.dot(w, z) / np.sqrt(denom2))
    p_combined = float(norm.sf(Z))

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
        colormap: str = 'viridis',
        figsize: Tuple[int, int] = (16, 14),
        ext: str = 'pdf',
        figpath: str = None,
        highlights: bool = True,
        project_name: str = None,
        oddsratio: bool = False,
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
        figpath : str or Path, optional
            Directory in which to save outputs; if None, uses current working dir.
        project_name : str
            Prefix for saved figure.
        bighlights: bool
            Whether to highlight specs
        oddsratio bool, default=False
            Whether to exponentiate the coefficients (e.g. for odds ratios).
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
            figpath=figpath,
            project_name=project_name,
            highlights=highlights,
            oddsratio=oddsratio
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

    @staticmethod
    def _empirical_p_two_sided(null_samples: np.ndarray, observed: float) -> float:
        t = np.asarray(null_samples, dtype=float)
        t = t[np.isfinite(t)]
        if t.size == 0:
            return np.nan
        n = t.size
        p_hi = (np.sum(t >= observed) + 1) / (n + 1)
        p_lo = (np.sum(t <= observed) + 1) / (n + 1)
        return float(min(1.0, 2.0 * min(p_hi, p_lo)))


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
            Label for the CV metric.
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

        obs_med = float(df_model_result['betas'].median())
        null_meds = self.estimates_ystar.median(axis=0).to_numpy()

        # two-sided via absolute value
        self.inference['median_p'] = float(np.mean(np.abs(null_meds) >= abs(obs_med)))
        
        # Estimate rho from bootstrap p-values
        # Convert p-values to z-scores across bootstrap draws
        p_matrix = self.p_values.to_numpy()  # shape: (n_specs, n_draws)
        if p_matrix.shape[1] > 1:  # Need multiple draws to estimate correlation
            # Convert two-sided p-values to z-scores
            z_matrix = np.zeros_like(p_matrix)
            for i in range(p_matrix.shape[0]):
                signs = np.sign(self.estimates.iloc[i, :].to_numpy())
                z_matrix[i, :] = signs * norm.isf(np.clip(p_matrix[i, :], 1e-300, 1 - 1e-16) / 2.0)
            
            # Calculate correlation matrix of z-scores across specifications.
            # Use safe manual correlation computation that handles zero-variance rows
            # without emitting platform-dependent warnings.
            z_centered = z_matrix - z_matrix.mean(axis=1, keepdims=True)
            z_std = np.std(z_matrix, axis=1, keepdims=True)
            # Avoid divide-by-zero: replace zero std with 1 (correlation undefined, set to NaN)
            z_std_safe = np.where(z_std > 1e-14, z_std, 1.0)
            z_normalized = z_centered / z_std_safe
            z_corr = np.dot(z_normalized, z_normalized.T) / z_matrix.shape[1]
            # Set correlation to NaN where either row has zero variance
            zero_var_rows = (z_std[:, 0] <= 1e-14)
            z_corr[zero_var_rows, :] = np.nan
            z_corr[:, zero_var_rows] = np.nan
            
            # Extract mean off-diagonal correlation as rho estimate
            n_specs = z_corr.shape[0]
            if n_specs > 1:
                off_diag_sum = np.sum(z_corr) - np.trace(z_corr)
                rho_estimate = off_diag_sum / (n_specs * (n_specs - 1))
                # Clip to valid range
                rho_estimate = np.clip(rho_estimate, -1.0 / (n_specs - 1) + 0.001, 0.999)
            else:
                rho_estimate = None
        else:
            rho_estimate = None
            z_corr = None

        self.inference['min_ns'] = df_model_result['betas'].min()
        self.inference['min'] = self.estimates.min().min()
        self.inference['max_ns'] = df_model_result['betas'].max()
        self.inference['max'] = self.estimates.max().max()

        self.inference['pos_ns'] = df_model_result['positive_beta'].sum()
        self.inference['pos_prop_ns'] = df_model_result['positive_beta'].mean()
        self.inference['pos'] = (self.estimates > 0.0).sum().sum()
        self.inference['pos_prop'] = (self.estimates > 0.0).mean().mean()
        self.inference['neg_ns'] = df_model_result['negative_beta'].sum()
        self.inference['neg_prop_ns'] = df_model_result['negative_beta'].mean()
        self.inference['neg'] = (self.estimates < 0.0).sum().sum()
        self.inference['neg_prop'] = (self.estimates < 0.0).mean().mean()

        # observed counts
        obs_pos = int((df_model_result['betas'] > 0).sum())
        obs_neg = int((df_model_result['betas'] < 0).sum())

        # null counts per null draw (across specs)
        null_pos = self.estimates_ystar.gt(0).sum(axis=0).to_numpy()
        null_neg = self.estimates_ystar.lt(0).sum(axis=0).to_numpy()

        self.inference['pos_p'] = self.__class__._empirical_p_two_sided(null_pos, obs_pos)
        self.inference['neg_p'] = self.__class__._empirical_p_two_sided(null_neg, obs_neg)
        self.inference['sig_ns'] = df_model_result['significant'].sum()
        self.inference['sig_prop_ns'] = df_model_result['significant'].mean()
        self.inference['sig'] = (self.p_values.stack() < 0.05).sum().sum()
        self.inference['sig_prop'] = (self.p_values.stack() < 0.05).mean().mean()

        alpha = 0.05
        obs_sig = int((self.p_values.stack() < alpha).sum())
        null_sig = self.p_values_ystar.lt(alpha).sum(axis=0).to_numpy()
        self.inference['sig_p'] = self.__class__._empirical_p_two_sided(null_sig, obs_sig)

        self.inference['pos_sig_ns'] = (df_model_result['positive_beta'] &
                                     df_model_result['significant']).sum()
        self.inference['pos_sig_prop_ns'] = (df_model_result['positive_beta'] &
                                          df_model_result['significant']).mean()
        self.inference['pos_sig'] = ((self.estimates > 0.0) &
                                     (self.p_values < 0.05)).sum().sum()
        self.inference['pos_sig_prop'] = ((self.estimates > 0.0) &
                                          (self.p_values < 0.05)).mean().mean()
        self.inference['neg_sig_ns'] = (df_model_result['negative_beta'] &
                                        df_model_result['significant']).sum()
        self.inference['neg_sig_prop_ns'] = (df_model_result['negative_beta'] &
                                             df_model_result['significant']).mean()
        self.inference['neg_sig'] = ((self.estimates < 0.0) &
                                     (self.p_values < 0.05)).sum().sum()
        self.inference['neg_sig_prop'] = ((self.estimates < 0.0) &
                                          (self.p_values < 0.05)).mean().mean()

        obs_pos_sig = int(((self.estimates > 0.0) & (self.p_values < alpha)).sum().sum())
        obs_neg_sig = int(((self.estimates < 0.0) & (self.p_values < alpha)).sum().sum())
        null_pos_sig = ((self.estimates_ystar.gt(0) & self.p_values_ystar.lt(alpha))
                        .sum(axis=0).to_numpy())
        null_neg_sig = ((self.estimates_ystar.lt(0) & self.p_values_ystar.lt(alpha))
                        .sum(axis=0).to_numpy())
        self.inference['pos_sig_p'] = self.__class__._empirical_p_two_sided(null_pos_sig, obs_pos_sig)
        self.inference['neg_sig_p'] = self.__class__._empirical_p_two_sided(null_neg_sig, obs_neg_sig)

        # Estimate rho from bootstrap p-values
        # Convert p-values to z-scores across bootstrap draws
        p_matrix = self.p_values.to_numpy()  # shape: (n_specs, n_draws)
        if p_matrix.shape[1] > 1:  # Need multiple draws to estimate correlation
            # Convert two-sided p-values to z-scores
            z_matrix = np.zeros_like(p_matrix)
            for i in range(p_matrix.shape[0]):
                signs = np.sign(self.estimates.iloc[i, :].to_numpy())
                z_matrix[i, :] = signs * norm.isf(np.clip(p_matrix[i, :], 1e-300, 1 - 1e-16) / 2.0)
            
            # Calculate correlation matrix of z-scores across specifications.

            # Suppress numpy RuntimeWarnings emitted by np.corrcoef (divide/invalid)
            # when any row has near-zero variance.
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                with np.errstate(divide="ignore", invalid="ignore"):
                    z_corr = np.corrcoef(z_matrix)
                    z_corr = np.nan_to_num(z_corr, nan=0.0)

            
            # Extract mean off-diagonal correlation as rho estimate
            n_specs = z_corr.shape[0]
            if n_specs > 1:
                off_diag_sum = np.sum(z_corr) - np.trace(z_corr)
                rho_estimate = off_diag_sum / (n_specs * (n_specs - 1))
                # Clip to valid range
                rho_estimate = np.clip(rho_estimate, -1.0 / (n_specs - 1) + 0.001, 0.999)
            else:
                rho_estimate = None
        else:
            rho_estimate = None
            z_corr = None

        # Stouffer's method with estimated covariance
        # Use full correlation matrix (Sigma) to account for heterogeneous correlations
        if z_corr is not None and p_matrix.shape[1] > 1:
            # Attempt Stouffer with the raw estimated correlation matrix.
            z_corr = np.asarray(z_corr, dtype=np.float64)
            try:
                Z, p_combined = stouffer_method(
                    df_model_result['p_values'].to_numpy(),
                    two_sided=True,
                    betas=df_model_result['betas'].to_numpy(),
                    Sigma=z_corr,
                )

                # Store results with covariance structure
                self.inference['Stouffers'] = (Z, p_combined)
                self.inference['correlation_matrix'] = z_corr
                self.inference['mean_correlation'] = rho_estimate
            except ValueError as e:
                # If Σ leads to nonpositive/nonfinite variance, fall back to independence
                # Use project-style WARNING lines (consistent with other messages).
                print(f"WARNING: Stouffer Σ path failed ({e}); falling back to independence assumption.")
                Z, p_comb = stouffer_method(
                    df_model_result['p_values'].to_numpy(),
                    two_sided=True,
                    betas=df_model_result['betas'].to_numpy(),
                    warn=False,
                )
                log_p_two = np.log(2.0) + norm.logsf(abs(Z))
                p_two = float(np.exp(log_p_two))
                self.inference['Stouffers'] = (Z, p_two)
                self.inference['correlation_matrix'] = None
                self.inference['mean_correlation'] = None
        else:
            # Fall back to independence assumption
            Z, p_comb = stouffer_method(
                df_model_result['p_values'].to_numpy(),
                two_sided=True,
                betas=df_model_result['betas'].to_numpy(),
            )
            log_p_two = np.log(2.0) + norm.logsf(abs(Z))
            p_two = float(np.exp(log_p_two))
            self.inference['Stouffers'] = (Z, p_two)
            self.inference['correlation_matrix'] = None
            self.inference['mean_correlation'] = None


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
        if max(len(t) for t in self.y_name) > 1:
            print(f"Dependent variable: {set().union(*self.y_name)}")
        else:
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

        # Medians / extrema
        if not inference:
            print(f"Median β (specs, no resampling): {self.inference['median_ns']:.{digits}f}")
        else:
            print(f"Median β (specs, no resampling): {self.inference['median_ns']:.{digits}f} "
                  f"(null-calibrated p: {self.inference['median_p']:.{digits}g})")
        print(f"Median β (bootstraps × specs): {self.inference['median']:.{digits}f}")

        print(f"Min β (specs, no resampling): {self.inference['min_ns']:.{digits}f}")
        print(f"Min β (bootstraps × specs): {self.inference['min']:.{digits}f}")

        print(f"Max β (specs, no resampling): {self.inference['max_ns']:.{digits}f}")
        print(f"Max β (bootstraps × specs): {self.inference['max']:.{digits}f}")

        # IC-weighted averages (fix NaN checks)
        if not np.isnan(self.inference.get('aic_average', np.nan)):
            print(f"AIC-weighted β (specs, no resampling): {self.inference['aic_average']:.{digits}f}")
        if not np.isnan(self.inference.get('bic_average', np.nan)):
            print(f"BIC-weighted β (specs, no resampling): {self.inference['bic_average']:.{digits}f}")
        if not np.isnan(self.inference.get('hqic_average', np.nan)):
            print(f"HQIC-weighted β (specs, no resampling): {self.inference['hqic_average']:.{digits}f}")

        # Descriptive shares + null-calibrated tests
        label_share = lambda s: f"{s:.{digits}f}"
        label_p = lambda p: f"{p:.{digits}g}"

        # Significant share
        print(f"Share significant (specs, no resampling) [descriptive]: "
              f"{label_share(self.inference['sig_prop_ns'])}")
        print(f"Share significant (bootstraps × specs) [descriptive]: "
              f"{label_share(self.inference['sig_prop'])}"
              + ("" if not inference else f"  |  null-calibrated p: {label_p(self.inference['sig_p'])}"))

        # Positive share
        print(f"Share β>0 (specs, no resampling) [descriptive]: "
              f"{label_share(self.inference['pos_prop_ns'])}")
        print(f"Share β>0 (bootstraps × specs) [descriptive]: "
              f"{label_share(self.inference['pos_prop'])}"
              + ("" if not inference else f"  |  null-calibrated p: {label_p(self.inference['pos_p'])}"))

        # Negative share
        print(f"Share β<0 (specs, no resampling) [descriptive]: "
              f"{label_share(self.inference['neg_prop_ns'])}")
        print(f"Share β<0 (bootstraps × specs) [descriptive]: "
              f"{label_share(self.inference['neg_prop'])}"
              + ("" if not inference else f"  |  null-calibrated p: {label_p(self.inference['neg_p'])}"))

        # Positive & significant share
        print(f"Share β>0 & significant (specs, no resampling) [descriptive]: "
              f"{label_share(self.inference['pos_sig_prop_ns'])}")
        print(f"Share β>0 & significant (bootstraps × specs) [descriptive]: "
              f"{label_share(self.inference['pos_sig_prop'])}"
              + ("" if not inference else f"  |  null-calibrated p: {label_p(self.inference['pos_sig_p'])}"))

        # Negative & significant share
        print(f"Share β<0 & significant (specs, no resampling) [descriptive]: "
              f"{label_share(self.inference['neg_sig_prop_ns'])}")
        print(f"Share β<0 & significant (bootstraps × specs) [descriptive]: "
              f"{label_share(self.inference['neg_sig_prop'])}"
              + ("" if not inference else f"  |  null-calibrated p: {label_p(self.inference['neg_sig_p'])}"))

        # Stouffer’s
        Z, p_two = self.inference['Stouffers']
        print(f"Stouffer’s Z = {Z:.{digits}f}  (two-sided p = {p_two:.{digits}g})")

        print_separator()

        # print in-sample metrics
        if max(len(t) for t in self.y_name) == 1:
            print('2.2 In-Sample Metrics (Full Sample)')
            print_separator()
            print(f"Min AIC: {round(self.summary_df['aic'].min(), digits)}, Specs: {list(self.summary_df['spec_name'].loc[self.summary_df['aic'].idxmin()])}")
            print(f"Min BIC: {round(self.summary_df['bic'].min(), digits)}, Specs: {list(self.summary_df['spec_name'].loc[self.summary_df['bic'].idxmin()])}")
            print(f"Min HQIC: {round(self.summary_df['hqic'].min(), digits)}, Specs: {list(self.summary_df['spec_name'].loc[self.summary_df['hqic'].idxmin()])}")
            print(f"Max Log Likelihood: {round(self.summary_df['ll'].max(), digits)}, Specs: {list(self.summary_df['spec_name'].loc[self.summary_df['ll'].idxmax()])}")
            print(
                f"Min Log Likelihood: {self.summary_df['ll'].min():.{digits}f}, "
                f"Specs: {list(self.summary_df['spec_name'].loc[self.summary_df['ll'].idxmin()])}"
            )
            print(
                f"Max {'Adj-' if 'OLS' in self.model_name else 'Pseudo-'}R²: "
                f"{self.summary_df['r2'].max():.{digits}f}, "
                f"Specs: {list(self.summary_df['spec_name'].loc[self.summary_df['r2'].idxmax()])}"
            )
            print(
                f"Min {'Adj-' if 'OLS' in self.model_name else 'Pseudo-'}R²: "
                f"{self.summary_df['r2'].min():.{digits}f}, "
                f"Specs: {list(self.summary_df['spec_name'].loc[self.summary_df['r2'].idxmin()])}"
            )

            print_separator()
        if max(len(t) for t in self.y_name) == 1:
            print(f'2.3 Out-Of-Sample Metrics ({self.name_av_k_metric} averaged across folds)')
        else:
            print(f'2.2 Out-Of-Sample Metrics ({self.name_av_k_metric} averaged across folds)')
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
             colormap: str = 'viridis',
             figsize: Tuple[int, int] = (12, 6),
             ext: str = '   pdf',
             figpath = None,
             project_name: str = 'no_project_name',
             highlights: bool = True,
             oddsratio: bool = False
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
            Confidence interval.
        colormap : str, default='viridis'
            Name of the matplotlib colormap for the plot.
        figpath : str or Path, optional
            Directory in which to save outputs; if None, uses current working dir.
        figsize : tuple of int, default=(12, 6)
            Figure width and height in inches.
        ext : str, default='pdf'
            File extension if saving the figure (unused if not saving).
        project_name : str, default='no_project_name'
            Project identifier used in saved filename (unused if not saving).
        highlights bool, default = True
            Whether to highlight individual plots.
        oddsratio bool, default=False
            Whether to exponentiate the coefficients (e.g. for odds ratios).
        Returns
        -------
        fig : matplotlib.figure.Figure
            The figure object containing the plot.

        Raises
        ------
        ValueError
            If `ic` is not one of {'bic', 'aic', 'hqic'}.
            or if more than three specs are given,
        TypeError
            If `specs` is provided but is not a list of lists of str,
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
                            figpath=figpath,
                            project_name=project_name,
                            highlights=highlights,
                            oddsratio=oddsratio)

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

        Returns
        -------
        pd.DataFrame
            DataFrame containing BMA results with control variable inclusion
            probabilities and average coefficients.
        """
        likelihood_per_var = []
        weighted_coefs = []

        # Compute model likelihoods using BIC-implied log posterior
        max_ll = np.max(-self.summary_df.bic / 2)
        shifted_ll = (-self.summary_df.bic / 2) - max_ll
        models_likelihood = np.exp(shifted_ll)
        sum_likelihoods = np.nansum(models_likelihood)

        # Flatten list of coefficient arrays and variable names
        coefs = [[i[0] for i in x] for x in self.all_b]
        coefs = [i for sl in coefs for i in sl]
        var_names = [i for sl in self.all_predictors for i in sl]
        coefs_df = pd.DataFrame({'coef': coefs, 'var_name': var_names})

        # Compute weighted likelihood and average coefficients for each control variable
        for ele in self.controls:
            idx = [ele in spec for spec in self.specs_names]
            likelihood_per_var.append(np.nansum(models_likelihood[idx]))

            coefs = coefs_df[coefs_df.var_name == ele].coef.to_numpy()
            likelihood = models_likelihood[idx]
            weighted_coef = coefs * likelihood
            weighted_coefs.append(np.nansum(weighted_coef))

        # Normalize to get posterior inclusion probabilities and BMA-weighted coefficients
        probs = likelihood_per_var / sum_likelihoods
        final_coefs = weighted_coefs / sum_likelihoods

        # Return summary DataFrame
        summary_bma = pd.DataFrame({
            'control_var': self.controls,
            'probs': probs,
            'average_coefs': final_coefs
        })

        return summary_bma

    def merge(
        self,
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

        # Validate input types
        if not isinstance(result_obj, OLSResult):
            raise TypeError("'result_obj' must be an instance of OLSResult.")
        
        if not isinstance(left_prefix, str) or not isinstance(right_prefix, str):
            raise TypeError("'prefixes' must be of type 'str.'")

        # Ensure dependent variables match before merging
        if self.y_name != result_obj.y_name:
            raise ValueError('Dependent variable names must match.')

        # Apply prefixes to spec sets
        specs_original = [frozenset(list(s) + [left_prefix]) for s in self.specs_names]
        specs_new = [frozenset(list(s) + [right_prefix]) for s in result_obj.specs_names]

        # Merge all relevant fields
        y = self.y_name
        specs = specs_original + specs_new
        estimates = pd.concat([self.estimates, result_obj.estimates], ignore_index=True)
        p_values = pd.concat([self.p_values, result_obj.p_values], ignore_index=True)
        r2_values = pd.concat([self.r2_values, result_obj.r2_values], ignore_index=True)

        # Return a new MergedResult object
        return MergedResult(
            y=y,
            specs=specs,
            estimates=estimates,
            p_values=p_values,
            r2_values=r2_values
        )
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
    
    def sample_z_specs(self, controls, z_specs_sample_size):
        """
        Generate a sample of z specifications by randomly selecting subsets of control variables.

        This method creates a set of binary masks to determine which subsets of the given `controls`
        should be included in each z specification. The number of subsets sampled is controlled by
        `z_specs_sample_size`.

        Parameters
        ----------
        controls : list of str
            A list of control variable names from which to build z specifications.
        z_specs_sample_size : int
            The number of z specifications to sample. Must be a positive integer.

        Returns
        -------
        space_n_sample : int
            The number of sampled z specifications.
        z_specs_sample : list of tuple of str
            A list where each element is a tuple of selected control variable names
            representing one sampled z specification.

        Notes
        -----
        - The method uses `sample_z_masks` to generate binary inclusion masks.
        - If `self.seed` is defined, it is passed to `sample_z_masks` to ensure reproducibility.
        - Each sampled mask determines a unique subset of `controls`.
        """
        z_cols = controls
        n_z = len(z_cols)
        masks = sample_z_masks(
            n_z = n_z,
            n_masks = z_specs_sample_size,
            seed = getattr(self, "seed", None)
            )
        z_specs_sample = [
            tuple(z_cols[i] for i in range(n_z) if (m >> i) & 1)
            for m in masks
            ]
        space_n_sample = len(z_specs_sample)
        return space_n_sample, z_specs_sample
    
    def fit(
        self,
        *,
        controls: List[str],
        group: Optional[str] = None,
        draws: int = None,
        kfold: int = None,
        oos_metric: str = None,
        n_cpu: Optional[int] = None,
        seed: Optional[int] = None,
        composite_sample: Optional[int] = None,
        z_specs_sample_size: Optional[int] = None,
        rescale_y: Optional[bool] = False,
        rescale_x: Optional[bool] = False,
        rescale_z: Optional[bool] = False,
        threshold: int = 1000000
    ) -> 'OLSRobust':
        """
        Fit the OLS models across the specification space and over bootstrap resamples.

        This method explores a variety of control variable specifications by constructing
        different combinations (z-specs) and optionally sampling from this space. It performs
        bootstrap resampling and/or cross-validation depending on the arguments provided.

        Parameters
        ----------
        controls : list of str
            Candidate control variables to include in model specifications.
        group : str, optional
            Column name for grouping fixed effects. If provided, outcomes are de-meaned by group.
        draws : int, optional
            Number of bootstrap resamples per specification. If None, bootstrapping is skipped.
        kfold : int, optional
            Number of folds for out-of-sample evaluation. Requires oos_metric to be specified.
        oos_metric : {'pseudo-r2', 'rmse'}, optional
            Metric to evaluate out-of-sample performance when kfold is set.
        n_cpu : int, optional
            Number of parallel processes to use. Defaults to all available CPUs minus one if None.
                seed : int, optional
            Random seed for reproducibility. Propagated to all random operations.
        composite_sample : int, optional
            If set, draw this many bootstrap samples **before** applying specification variation;
            used to reduce total computation while capturing uncertainty in the full composite model.
        z_specs_sample_size : int, optional
            Number of z specifications to randomly sample from the full set of possible combinations.
            If None, the full specification space is used.
        rescale_y : bool, default=False
            If True, rescale the dependent variable to have mean 0 and standard deviation 1.
        rescale_x : bool, default=False
            If True, rescale the x variable(s) to have mean 0 and standard deviation 1.
        rescale_z : bool, default=False
            If True, rescale the z variable(s) to have mean 0 and standard deviation 1.
        threshold : int, default=1_000_000
            If the total number of model fits (specs × draws × folds) exceeds this number, a warning is raised.

        Returns
        -------
        self : OLSRobust
            The fitted model instance. Results are stored in the .results attribute.

        Notes
        -----
        - At least one of draws or kfold must be set to perform model fitting.
        - If both composite_sample and z_specs_sample_size are set, the combination is applied in that order.
        - This method may be computationally intensive; parallelisation is recommended via n_cpu.
        """

        # Validate control variable input
        if not isinstance(controls, list):
            raise TypeError(f"'controls' must be a list. Received types: {type(controls).__name__}.")

        # Validate z_specs_sample_size input
        if z_specs_sample_size is not None:
            if not isinstance(z_specs_sample_size, int) or z_specs_sample_size <= 0:
                raise ValueError("'z_specs_sample_size' must be an integer greater than 0 if provided.")


        # Optionally rescale y, x, and z variables
        if rescale_y:
            if len(self.y) > 1:
                warnings.warn("Rescaling of the dependent variable already occurs with multiple y variables. Skipping additional rescaling.", UserWarning)
            else:
                self.data[self.y] = rescale(self.data[self.y])

        if rescale_x:
            for column in self.x:
                self.data[column] = rescale(self.data[column])

        if rescale_z:
            for column in controls:
                self.data[column] = rescale(self.data[column])

        # Standardize and propagate user arguments
        draws, kfold, oos_metric, n_cpu, seed = make_inquiry(
            self.model_name, self.y, self.data, draws, kfold, oos_metric, n_cpu, seed
        )

        self.seed = seed
        self.composite_sample = composite_sample

        # Ensure constant term and prepare x, controls

        if group:
            # With group demeaning, an intercept is not identified. Drop any existing constant.
            if "const" in self.data.columns:
                self.data = self.data.drop(columns=["const"])
            if "const" in self.x:
                self.x = [c for c in self.x if c != "const"]
            if "const" in controls:
                controls = [c for c in controls if c != "const"]
            warnings.warn(
                "[OLSRobust] Group fixed effects requested: dropping constant term "
                "before demeaning so t/p are identified.",
                UserWarning
            )
            invariant = _find_group_invariant_columns(self.data, self.x + controls, group)
            inv_x = [c for c in invariant if c in self.x]
            inv_controls = [c for c in invariant if c in controls]
            if inv_x or inv_controls:
                warnings.warn(
                    "[OLSRobust] These variables are constant within each group and "
                    "are not identified under group demeaning. t/p will be NaN for them: "
                    f"x={inv_x!r}, controls={inv_controls!r}.",
                    UserWarning
                )
        else:
            self.x, controls = _ensure_single_constant(self.data, self.y, self.x, controls)


        # Handle multiple y case
        if len(self.y) > 1:
            self.multiple_y()

        # Check that provided arguments are valid
        self._validate_fit_args(
            controls=controls,
            group=group,
            draws=draws,
            kfold=kfold,
            oos_metric=oos_metric,
            n_cpu=n_cpu,
            seed=self.seed,
            valid_oos_metrics=['pseudo-r2', 'rmse'],
            threshold=threshold
        )

        # Set global seed
        np.random.seed(self.seed)

        # Inform user of configuration
        print(
            f"OLSRobust is running with n_cpu={n_cpu}, draws={draws}, folds={kfold}, seed={seed}.\n"
            f"We're evaluating our out-of-sample predictions with the {oos_metric} metric.\n"
            f"The estimand of interest is {self.x[0]}. Let's begin the calculations..."
        )

        # Determine sample size and metric
        sample_size = self.data.shape[0]
        self.oos_metric_name = oos_metric

        # Optionally sample from the specification space
        if z_specs_sample_size:
            space_n, z_specs = self.sample_z_specs(controls=controls, z_specs_sample_size=z_specs_sample_size)
        else:
            space_n = space_size(controls)
            z_specs = all_subsets(controls)

        # Count y-composite specifications (1 if single y)
        n_y_comps = len(getattr(self, "y_specs", [None]))

        # Warn if model fit will be too large
        self._warn_if_large_draws(
            draws=draws,
            n_control_specs=space_n,
            n_y_composites=n_y_comps,
            threshold=threshold
        )

        # Delegate to single or multiple y logic
        if len(self.y) > 1:
            z_specs = list(z_specs)
            # Initialize containers to collect results across all composite outcomes
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
            all_predictors = []
            b_all_list = []
            p_all_list = []
            
            # Loop over each composite y variable
            for y, y_name in track(zip(self.y_composites,
                                 self.y_specs), total=len(self.y_composites)):
                
                # Initialize arrays to store draws for this y-spec
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
                
                # Loop over all z-specs (control combinations)
                for spec, index in zip(z_specs, range(0, space_n)):
                    
                    # Build dataframe with x + spec + optional group
                    if len(spec) == 0:
                        comb = self.data[self.x]
                    else:
                        comb = self.data[self.x + list(spec)]
                    if group:
                        comb = self.data[self.x + [group] + list(spec)]
                        
                    # Add the outcome column and clean
                    comb = pd.concat([y, comb], axis=1)
                    comb = comb.dropna()
                    comb = comb.reset_index(drop=True).copy()

                    # Demean by group if required
                    if group:
                        comb = group_demean(comb, group=group)
                    X_design = comb.drop(columns=comb.columns[0])
                    try:
                        self._check_colinearity(X_design)
                    except ValueError as e:
                        warnings.warn(
                            f"[OLSRobust] Spec {spec!r} may be unstable due to perfect collinearity:\n{e}\n"
                            "Proceeding with pseudoinverse; coefficients may not be uniquely identified.",
                            UserWarning
                        )
                        
                    # Fit the full sample OLS mode
                    (b_all, p_all, r2_i, ll_i,
                     aic_i, bic_i, hqic_i,
                     av_k_metric_i) = self._full_sample_OLS(comb,
                                                            kfold=kfold,
                                                            group=group,
                                                            oos_metric_name=self.oos_metric_name)
                    # Calculate pseudo-outcome residuals for bootstrap
                    y_star = comb.iloc[:, [0]] - np.dot(comb.iloc[:, [1]], b_all[0][0])
                    seeds = np.random.randint(0, 2**31, size=draws, dtype=np.int64)
                    # Run bootstrap regressions in parallel
                    b_list, p_list, r2_list, b_list_ystar, p_list_ystar = zip(*Parallel(n_jobs=n_cpu)
                    (delayed(self._strap_OLS)
                     (comb,
                      group,
                      sample_size,
                      seed,
                      y_star
                      )
                     for seed in seeds))
                    
                    # Collect and store results for this spec
                    y_names.append(y_name)
                    specs.append(frozenset(list(y_name) + list(spec)))
                    all_predictors.append(self.x + list(spec))
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

                # Store arrays across all y-specs
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

            # Assemble final result object across all composite outcomes
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
            # Initialize containers for storing results across z-specs
            specs = []
            all_predictors = []
            b_all_list = []
            p_all_list = []
            
            # Preallocate arrays for bootstrap and model stats
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
            
            # Prepare SHAP dataset (includes y, x, z, and optional group)
            if group:
                SHAP_comb = self.data[self.y + self.x + [group] + controls]
                SHAP_comb = group_demean(SHAP_comb, group=group)
            else:
                SHAP_comb = self.data[self.y + self.x + controls]
            SHAP_comb = SHAP_comb.dropna()
            SHAP_comb = SHAP_comb.reset_index(drop=True).copy()
            
            # Fit linear model for SHAP values
            x_train, x_test, y_train, _ = train_test_split(SHAP_comb[self.x + controls].drop(columns=['const'], errors='ignore'),
                                                           SHAP_comb[self.y],
                                                           test_size=0.2,
                                                           random_state=self.seed
                                                           )
            model = sklearn.linear_model.LinearRegression()
            model.fit(x_train, y_train)
            
            # Explain model predictions with SHAP
            explainer = shap.LinearExplainer(model, x_train)
            shap_return = [explainer.shap_values(x_test), x_test]
            
            # Generate shared bootstrap seeds ONCE for all specifications
            # This ensures all specs use the same bootstrap samples, creating proper correlation
            shared_seeds = np.random.randint(0, 2 ** 32 - 1, size=draws, dtype=np.int64)
            
            # Loop over all z-specs (combinations of control variables)
            for spec, index in track(zip(z_specs, range(0, space_n)), total=space_n):
                if 0 == len(spec):
                    comb = self.data[self.y + self.x]
                else:
                    comb = self.data[self.y + self.x + list(spec)]
                if group:
                    comb = self.data[self.y + self.x + [group] + list(spec)]

                 # Clean combined dataset
                comb = comb.dropna()
                comb = comb.reset_index(drop=True).copy()
                
                # Demean by group if required
                if group:
                    comb = group_demean(comb, group=group)
                
                # Fit model on full sample
                (b_all, p_all, r2_i, ll_i,
                 aic_i, bic_i, hqic_i,
                 av_k_metric_i) = self._full_sample_OLS(comb,
                                                        kfold=kfold,
                                                        group=group,
                                                        oos_metric_name=self.oos_metric_name)
                 
                # Calculate pseudo-outcome residuals for bootstrapping
                y_star = comb.iloc[:, [0]] - np.dot(comb.iloc[:, [1]], b_all[0][0])
                
                # Bootstrap OLS model draws in parallel using SHARED seeds
                # All specs use the same seeds to create proper correlation structure
                b_list, p_list, r2_list, b_list_ystar, p_list_ystar  = zip(*Parallel(n_jobs=n_cpu)
                (delayed(self._strap_OLS)
                 (comb,
                  group,
                  sample_size,
                  seed,
                  y_star)
                 for seed in shared_seeds))
                
                # Store results for current spec
                specs.append(frozenset(spec))
                all_predictors.append(self.x + list(spec))
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
                
            # Assemble final results object
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


    def _full_sample_OLS(
        self,
        comb_var,
        kfold,
        group,
        oos_metric_name
    ):
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
            Out-of-sample metric ('pseudo-r2' or 'rmse')..

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

        # Separate response and predictors
        y = comb_var.iloc[:, [0]]
        x_temp = comb_var.drop(comb_var.columns[0], axis=1)

        # Drop group column if provided
        if group:
            x = x_temp.drop(columns=group)
        else:
            x = x_temp

        # Fit full sample OLS
        out = simple_ols(y=y, x=x)

        # Initialize cross-validated metric
        av_k_metric = None

        # Cross-validation (with or without grouping)
        if kfold:

            metric = []

            if group:
                k_fold = GroupKFold(kfold)

                for k, (train, test) in enumerate(k_fold.split(x, y, groups=x_temp[group])):
                    out_k = simple_ols(y=y.loc[train], x=x.loc[train])
                    y_pred = self._predict(x.loc[test], out_k['b'])
                    y_true = y.loc[test]

                    if oos_metric_name == 'rmse':
                        k_rmse = root_mean_squared_error(y_true, y_pred)
                        metric.append(k_rmse)

                    elif oos_metric_name == 'pseudo-r2':
                        k_r2 = pseudo_r2(y_true, y_pred, np.mean(y.loc[train]))
                        metric.append(k_r2)

                    else:
                        raise ValueError('No valid OOS metric provided.')

            else:
                k_fold = KFold(kfold)

                for k, (train, test) in enumerate(k_fold.split(x, y)):
                    out_k = simple_ols(y=y.loc[train], x=x.loc[train])
                    y_pred = self._predict(x.loc[test], out_k['b'])
                    y_true = y.loc[test]

                    if oos_metric_name == 'rmse':
                        k_rmse = root_mean_squared_error(y_true, y_pred)
                        metric.append(k_rmse)

                    elif oos_metric_name == 'pseudo-r2':
                        k_r2 = pseudo_r2(y_true, y_pred, np.mean(y.loc[train]))
                        metric.append(k_r2)

                    else:
                        raise ValueError('No valid OOS metric provided.')

            av_k_metric = np.mean(metric)

        # Return model fit statistics and OOS metric
        return (
            out['b'],
            out['p'],
            out['r2'],
            out['ll'][0][0],
            out['aic'][0][0],
            out['bic'][0][0],
            out['hqic'][0][0],
            av_k_metric
        )
            
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
            # Sample randomly from full data (no group structure)
            samp_df = temp_data.sample(n=sample_size, replace=True, random_state=seed)

            y = samp_df.iloc[:, [0]]
            y_star = samp_df.iloc[:, [-1]]

            x = samp_df.drop(['y_star'], axis=1)
            x = x.drop(samp_df.columns[0], axis=1)

        else:
            # Stratified sampling at the group level
            np.random.seed(seed)
            idx = np.random.choice(temp_data[group].unique(), sample_size)

            select = temp_data[temp_data[group].isin(idx)]

            # Remove singleton groups (with only one observation)
            no_singleton = select[select.groupby(group).transform('size') > 1]

            if len(no_singleton) < 5:
                warnings.warn(
                    f"Bootstrap sample size is only {len(no_singleton)} after removing singleton groups "
                    f"(groups with a single observation). This may lead to unstable or unreliable estimates.",
                    UserWarning
                )

            # Drop grouping variable and extract y, y_star, and x
            no_singleton = no_singleton.drop(columns=[group])

            y = no_singleton.iloc[:, [0]]

            y_star = no_singleton.iloc[:, no_singleton.columns.get_loc('y_star')].to_frame()

            x = no_singleton.drop(['y_star'], axis=1)
            x = x.drop(no_singleton.columns[0], axis=1)

        # Fit standard OLS and y_star model
        # When using group-demeaning (fixed effects), don't add const since intercept is absorbed
        output = stripped_ols(y=y, x=x, add_const=(group is None))
        output_ystar = stripped_ols(y=y_star, x=x, add_const=(group is None))

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
        Run logistic regression over the full dataset (with y, x, controls), 
        and optionally compute cross-validated out-of-sample metrics.

        Parameters
        ----------
        comb_var : pandas.DataFrame
            DataFrame with columns [y, x, controls...].

        kfold : int
            Number of folds for cross-validation.

        group : str or None
            Grouping variable name (reserved for future use with fixed effects).

        oos_metric_name : str
            Out-of-sample metric to compute (if kfold > 1). Options:
            'mcfadden-r2', 'pseudo-r2', 'rmse', 'cross-entropy', or 'imv'.

        Returns
        -------
        beta : ndarray
            Coefficient estimates from full sample regression.

        p : ndarray
            P-value estimates from full sample regression.

        r2 : float
            In-sample pseudo-R² from full model.

        ll : float
            Log-likelihood.

        aic : float
            Akaike Information Criterion.

        bic : float
            Bayesian Information Criterion.

        hqic : float
            Hannan-Quinn Information Criterion.

        av_k_metric : float or None
            Average k-fold cross-validation metric, or None if no cross-validation.
        """

        # TODO: Implement Fixed Effects Logistic Regression (FE)
        
        # Split into response (y) and predictors (x)
        y = comb_var.iloc[:, [0]]
        x = comb_var.drop(comb_var.columns[0], axis=1)

        # Fit logistic regression on the full sample
        out = logistic_regression_sm(y=y, x=x)

        # Initialize output for cross-validation metric
        av_k_metric = None

        # Perform k-fold cross-validation if requested
        if kfold:
            k_fold = KFold(kfold)
            metric = []

            for k, (train, test) in enumerate(k_fold.split(x, y)):
                # Fit model on training fold
                out_k = logistic_regression_sm(y=y.loc[train], x=x.loc[train])

                # Predict on test fold
                y_pred = self._predict_LR(x.loc[test], out_k['b'])
                y_true = y.loc[test]

                # Evaluate prediction accuracy using selected OOS metric
                if oos_metric_name == 'rmse':
                    k_rmse = root_mean_squared_error(y_true, y_pred)
                    metric.append(k_rmse)

                elif oos_metric_name == 'mcfadden-r2':
                    k_r2 = mcfadden_r2(y_true, y_pred, np.mean(y.loc[train]))
                    metric.append(k_r2)

                elif oos_metric_name == 'pseudo-r2':
                    k_r2 = pseudo_r2(y_true, y_pred, np.mean(y.loc[train]))
                    metric.append(k_r2)

                elif oos_metric_name == 'cross-entropy':
                    k_cross_entropy = log_loss(y_true, y_pred)
                    metric.append(k_cross_entropy)

                elif oos_metric_name == 'imv':
                    imv = calculate_imv_score(y_true, y_pred)
                    metric.append(imv)

                else:
                    raise ValueError('No valid OOS metric provided.')

            # Average metric across all folds
            av_k_metric = np.mean(metric)

        # Return model statistics and cross-validation result
        return (
            out['b'],        # Coefficients
            out['p'],        # P-values
            out['r2'],       # In-sample pseudo-R²
            out['ll'],       # Log-likelihood
            out['aic'],      # AIC
            out['bic'],      # BIC
            out['hqic'],     # HQIC
            av_k_metric      # Cross-validation metric (if any)
        )

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
        return 1 / (1 + np.exp(-x_test.dot(betas)))


    def fit(
        self,
        *,
        controls: List[str],
        group: Optional[str] = None,
        draws: int = None,
        sample_size: Optional[int] = None,
        kfold: int = None,
        oos_metric: str = None,
        n_cpu: Optional[int] = None,
        seed: Optional[int] = None,
        rescale_x: Optional[bool] = False,
        rescale_y: Optional[bool] = False,
        rescale_z: Optional[bool] = False,
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
        draws : int, default=None
            Number of bootstrap resamples per specification.
        sample_size : int, optional
            Number of observations per bootstrap draw; defaults to full dataset.
        kfold : int, default=None
            Folds for out-of-sample CV; set to 0 to disable.
        oos_metric : default None.
            Options:    {'pseudo-r2', 'mcfadden-r2', 'imv', 'rmse','cross-entropy'}.
            Metric to compute on held-out folds.
        n_cpu : int, optional
            Number of parallel jobs; defaults to all available.
        seed : int, optional
            Random seed for reproducibility.
        rescale_y  : bool, default=False
            Rescale the dependent variable.
        rescale_x  : bool, default=False
            Rescale the x variable.
        rescale_z  : bool, default=False
            Rescale the z variables.
        threshold : int, default=1000000
            Warn if `draws * n_specs` exceeds this.

        Returns
        -------
        self : LRobust
            Self, with `.results` populated as an `OLSResult`.
        """
        # Rescale dependent variable (y) if needed
        if rescale_y is True:
            if len(self.y) > 1:
                warnings.warn(
                    "Rescaling of the dependent variable already occurs with multiple "
                    "y variables. Skipping additional rescaling.",
                    UserWarning
                )
            else:
                self.data[self.y] = rescale(self.data[self.y])

        # Rescale independent variables (x)
        if rescale_x is True:
            for column in self.x:
                self.data[column] = rescale(self.data[column])

        # Rescale control variables (z)
        if rescale_z is True:
            for column in controls:
                self.data[column] = rescale(self.data[column])

        # Prepare modeling setup (includes seed, CPU count, metric, etc.)
        draws, kfold, oos_metric, n_cpu, seed = make_inquiry(
            self.model_name,
            self.y,
            self.data,
            draws,
            kfold,
            oos_metric,
            n_cpu,
            seed
        )

        self.seed = seed
        self.data = self.data.copy()

        # Ensure a constant term in the design matrix

        if group:
            # With group demeaning, an intercept is not identified. Drop any existing constant.
            if "const" in self.data.columns:
                self.data = self.data.drop(columns=["const"])
            if "const" in self.x:
                self.x = [c for c in self.x if c != "const"]
            if "const" in controls:
                controls = [c for c in controls if c != "const"]
            warnings.warn(
                "[LRobust] Group fixed effects requested: dropping constant term "
                "before demeaning so t/p are identified.",
                UserWarning
            )
            invariant = _find_group_invariant_columns(self.data, self.x + controls, group)
            inv_x = [c for c in invariant if c in self.x]
            inv_controls = [c for c in invariant if c in controls]
            if inv_x or inv_controls:
                warnings.warn(
                    "[LRobust] These variables are constant within each group and "
                    "are not identified under group demeaning. t/p will be NaN for them: "
                    f"x={inv_x!r}, controls={inv_controls!r}.",
                    UserWarning
                )
        else:
            self.x, controls = _ensure_single_constant(self.data, self.y, self.x, controls)


        # Validate modeling parameters
        self._validate_fit_args(
            controls=controls,
            group=group,
            draws=draws,
            kfold=kfold,
            oos_metric=oos_metric,
            n_cpu=n_cpu,
            seed=seed,
            valid_oos_metrics=['pseudo-r2', 'mcfadden-r2', 'rmse', 'cross-entropy', 'imv'],
            threshold=threshold
        )

        # Set seed for reproducibility
        np.random.seed(self.seed)

        # Store the selected out-of-sample evaluation metric
        self.oos_metric_name = oos_metric

        # Log model configuration
        print(
            f"LRobust is running with n_cpu={n_cpu}, draws={draws}, "
            f"folds={kfold}, seed={seed}.\n"
            f"We're evaluating our out-of-sample predictions with the "
            f"{oos_metric} metric.\n"
            f"The target variable of interest is {self.x[0]}. "
            f"Let's begin the calculations..."
        )

        # Default to using the full dataset size if sample_size is not provided
        if sample_size is None:
            sample_size = self.data.shape[0]

        # Currently only supports one dependent variable for logistic regression
        if len(self.y) > 1:
            raise NotImplementedError("Not implemented yet for logistic regression")  # TODO: Add multi-y logistic support

        else:
            space_n = space_size(controls)  # Number of control variable combinations
            specs = []
            all_predictors = []
            b_all_list = []
            p_all_list = []

            # Initialize arrays to store model statistics across all control sets
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

            # Preprocess data for SHAP values
            if group:
                SHAP_comb = self.data[self.y + self.x + [group] + controls]
                SHAP_comb = group_demean(SHAP_comb, group=group)
            else:
                SHAP_comb = self.data[self.y + self.x + controls]

            SHAP_comb = SHAP_comb.dropna().reset_index(drop=True).copy()

            # Split data for training and testing
            x_train, x_test, y_train, _ = train_test_split(
                SHAP_comb[self.x + controls].drop(columns=['const']),
                SHAP_comb[self.y],
                test_size=0.2,
                random_state=self.seed
            )

            # Train logistic regression for SHAP explainability
            model = sklearn.linear_model.LogisticRegression(penalty="l2", C=0.1)
            model.fit(x_train, y_train.squeeze())
            explainer = shap.LinearExplainer(model, x_train)
            shap_return = [explainer.shap_values(x_test), x_test]

            # Loop through all subsets of control variables
            for spec, index in track(zip(all_subsets(controls), range(0, space_n)), total=space_n):

                # Subset the data according to the current control combination
                if len(spec) == 0:
                    comb = self.data[self.y + self.x]
                else:
                    comb = self.data[self.y + self.x + list(spec)]

                if group:
                    comb = self.data[self.y + self.x + [group] + list(spec)]

                comb = comb.dropna().reset_index(drop=True).copy()

                if group:
                    comb = group_demean(comb, group=group)

                X_design = comb.drop(columns=comb.columns[0])  # Drop y column
                self._check_colinearity(X_design)

                # Estimate full-sample model metrics
                (b_all, p_all, r2_i, ll_i,
                aic_i, bic_i, hqic_i,
                av_k_metric_i) = self._full_sample(
                    comb,
                    kfold=kfold,
                    group=group,
                    oos_metric_name=self.oos_metric_name
                )

                # Bootstrap sampling (parallelized)
                seeds = np.random.randint(0, 2**32 - 1, size=draws, dtype=np.int64)
                (b_list, p_list, r2_list) = zip(*Parallel(n_jobs=n_cpu)(
                    delayed(self._strap_regression)(
                        comb,
                        group,
                        sample_size,
                        seed
                    )
                    for seed in seeds
                ))

                # Save results
                specs.append(frozenset(spec))
                all_predictors.append(self.x + list(spec))
                b_array[index, :] = b_list
                p_array[index, :] = p_list
                b_array_ystar[index, :] = np.nan * len(b_list)
                p_array_ystar[index, :] = np.nan * len(p_list)
                r2_array[index, :] = r2_list
                r2i_array[index] = r2_i
                ll_array[index] = ll_i
                aic_array[index] = aic_i
                bic_array[index] = bic_i
                hqic_array[index] = hqic_i
                av_k_metric_array[index] = av_k_metric_i
                b_all_list.append(b_all)
                p_all_list.append(p_all)

            # Store all modeling results in an OLSResult object
            self.results = OLSResult(
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
