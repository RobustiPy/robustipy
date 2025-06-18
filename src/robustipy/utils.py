# Module containing utility functions for the library
from typing import Optional, List, Tuple, Iterable, Sequence, Union
import numpy as np
import random
import warnings
import scipy
import matplotlib
import pandas as pd
from itertools import chain, combinations
import statsmodels.api as sm
import matplotlib.pyplot as plt
from scipy.optimize import minimize


def concat_results(objs: List["OLSResult"], de_dupe=True) -> "OLSResult":
    """
     Core routine: take a list of OLSResult objects and stack them into one OLSResult.
     All per‐spec fields (estimates, p_values, specs_names, etc.) are concatenated,
     and any exact duplicates in (y_name, x_name, spec) are dropped in lockstep.
     accepts: de_dupe: bool, default True
        If True, drop exact duplicates in (y_name, x_name, spec) triplets.
     This function assumes each element of `objs` is already an OLSResult (not the wrapper class).
     We defer the import of OLSResult until inside the function to avoid circular‐import errors.
     """

    # ───────────────────────────────────────────────────────────
    # 1) Delay import of OLSResult until we actually need it
    #    (avoids “partially initialized module” / circular‐import errors).
    # ───────────────────────────────────────────────────────────
    from robustipy.models import OLSResult

    # 1a) Sanity‐check: each entry must be OLSResult
    if not all(isinstance(o, OLSResult) for o in objs):
        raise ValueError(
            "`_merge_results` expects a list of OLSResult objects (no OLSRobust/LRobust wrappers)."
        )
    if len(objs) == 0:
        raise ValueError("`_merge_results` got an empty list!")

    # ───────────────────────────────────────────────────────────
    # 2) Start by “copying” the first OLSResult into `merged`
    # ───────────────────────────────────────────────────────────
    merged: OLSResult = objs[0]

    # Helper: for a given OLSResult `r`, build a list of length = number of specs
    # that repeats r.y_name (whether scalar or list) once per spec.
    def _expand_yx(field, n_specs):
        """
        If `field` is a list of length == n_specs, return it unchanged.
        If `field` is a scalar (string), return [field] * n_specs.
        Otherwise, error.
        """
        if isinstance(field, list):
            if len(field) != n_specs:
                raise ValueError(
                    f"Length of list field {field!r} does not match number of specs ({n_specs})."
                )
            return field.copy()
        else:
            # assume scalar (e.g. "logpgp95"); repeat it once per spec
            return [field] * n_specs

    # 2a) “Number of specs” in the first object
    n0 = len(merged.specs_names)

    # 2b) Build y_list and x_list of length = n0
    merged_y_list = _expand_yx(merged.y_name, n0)
    merged_x_list = _expand_yx(merged.x_name, n0)

    # 2c) Grab all the “per‐spec” fields from the first object
    merged_specs = merged.specs_names.copy().reset_index(drop=True)
    merged_draws = [merged.draws] * n0
    merged_kfold = [merged.kfold] * n0
    merged_estimates = merged.estimates.copy().reset_index(drop=True)
    merged_estimates_ystar = merged.estimates_ystar.copy().reset_index(drop=True)
    merged_p_values = merged.p_values.copy().reset_index(drop=True)
    merged_p_values_ystar = merged.p_values_ystar.copy().reset_index(drop=True)
    merged_r2_values = merged.r2_values.copy().reset_index(drop=True)
    merged_all_b = list(merged.all_b)
    merged_all_p = list(merged.all_p)
    merged_all_predictors = list(merged.all_predictors)
    merged_summary_df = merged.summary_df.copy().reset_index(drop=True)

    # 2d) “Global” fields (just keep from the first object, but warn later if mismatch)
    merged_controls = list(merged.controls)
    merged_model_name = merged.model_name
    merged_name_av_k = merged.name_av_k_metric
    merged_shap_return = merged.shap_return
    # (We do NOT merge `inference` here.  If the user wants fresh inference after merging,
    #  they can call `merged._compute_inference()` manually.)

    # ───────────────────────────────────────────────────────────
    # 3) Append everything from the remaining OLSResult objects
    # ───────────────────────────────────────────────────────────
    for other in objs[1:]:
        # 3a) Expand other.y_name / other.x_name to lists of length = len(other.specs_names)
        n_o = len(other.specs_names)
        other_y_expanded = _expand_yx(other.y_name, n_o)
        other_x_expanded = _expand_yx(other.x_name, n_o)
        merged_y_list.extend(other_y_expanded)
        merged_x_list.extend(other_x_expanded)

        # 3b) specs_names is a pd.Series of length = n_o; concatenate
        merged_specs = pd.concat(
            [merged_specs, other.specs_names.reset_index(drop=True)],
            ignore_index=True
        )

        # 3c) draws and kfold become lists repeated once per spec
        merged_draws.extend([other.draws] * n_o)
        merged_kfold.extend([other.kfold] * n_o)

        # 3d) concat all per‐spec DataFrames
        merged_estimates = pd.concat(
            [merged_estimates, other.estimates.reset_index(drop=True)],
            ignore_index=True
        )
        merged_estimates_ystar = pd.concat(
            [merged_estimates_ystar, other.estimates_ystar.reset_index(drop=True)],
            ignore_index=True
        )
        merged_p_values = pd.concat(
            [merged_p_values, other.p_values.reset_index(drop=True)],
            ignore_index=True
        )
        merged_p_values_ystar = pd.concat(
            [merged_p_values_ystar, other.p_values_ystar.reset_index(drop=True)],
            ignore_index=True
        )
        merged_r2_values = pd.concat(
            [merged_r2_values, other.r2_values.reset_index(drop=True)],
            ignore_index=True
        )

        merged_all_b.extend(other.all_b)
        merged_all_p.extend(other.all_p)
        merged_all_predictors.extend(other.all_predictors)
        merged_summary_df = pd.concat(
            [merged_summary_df, other.summary_df.reset_index(drop=True)],
            ignore_index=True
        )

        # 3e) If controls or model_name differ, warn (but keep the first)
        if other.controls != merged_controls:
            warnings.warn(
                "Controls differ between merged OLSResult objects; keeping only the first set.",
                UserWarning
            )
        if other.model_name != merged_model_name:
            warnings.warn(
                "model_name differs between merged OLSResult objects; keeping only the first.",
                UserWarning
            )

    # ───────────────────────────────────────────────────────────
    # 4) Overwrite `merged` fields with our stacked versions
    # ───────────────────────────────────────────────────────────
    merged.y_name = merged_y_list
    merged.x_name = merged_x_list
    merged.specs_names = merged_specs.reset_index(drop=True)
    merged.draws = merged_draws
    merged.kfold = merged_kfold

    merged.estimates = merged_estimates
    merged.estimates_ystar = merged_estimates_ystar
    merged.p_values = merged_p_values
    merged.p_values_ystar = merged_p_values_ystar
    merged.r2_values = merged_r2_values

    merged.all_b = merged_all_b
    merged.all_p = merged_all_p
    merged.all_predictors = merged_all_predictors
    merged.summary_df = merged_summary_df.reset_index(drop=True)

    merged.controls = merged_controls
    merged.model_name = merged_model_name
    merged.name_av_k_metric = merged_name_av_k
    merged.shap_return = merged_shap_return

    if de_dupe:
        # ───────────────────────────────────────────────────────────
        # 5) Drop any exact‐duplicate (y_name, x_name, spec) triplets
        #    so that each unique combination survives only once.
        # ───────────────────────────────────────────────────────────
        n_specs_total = len(merged.specs_names)

        # 5a) Build a tiny DataFrame of length = number of specs
        Ys = merged.y_name
        Xs = merged.x_name
        df_dup = pd.DataFrame({
            "y": Ys,
            "x": Xs,
            "spec": merged.specs_names.values  # e.g. frozenset([...])
        })

        # 5b) Reset index → the old row‐indices land in a column called "index"
        temp = df_dup.reset_index()  # columns = ["index", "y", "x", "spec"]

        # 5c) Drop duplicates by (“y”, “x”, “spec”), keep only the first appearance
        dedup = temp.drop_duplicates(subset=["y", "x", "spec"], keep="first")

        # 5d) The “index” column in dedup tells us which rows we keep
        keep_idx = sorted(dedup["index"].tolist())

        # 5e) Now slice every per‐spec attribute by keep_idx → remove exact duplicates
        merged.specs_names = merged.specs_names.iloc[keep_idx].reset_index(drop=True)

        # y_name and x_name are lists; keep only those at indices in keep_idx
        merged.y_name = [merged.y_name[i] for i in keep_idx]
        merged.x_name = [merged.x_name[i] for i in keep_idx]

        merged.draws = [merged.draws[i] for i in keep_idx]
        merged.kfold = [merged.kfold[i] for i in keep_idx]

        merged.estimates = merged.estimates.iloc[keep_idx].reset_index(drop=True)
        merged.estimates_ystar = merged.estimates_ystar.iloc[keep_idx].reset_index(drop=True)
        merged.p_values = merged.p_values.iloc[keep_idx].reset_index(drop=True)
        merged.p_values_ystar = merged.p_values_ystar.iloc[keep_idx].reset_index(drop=True)
        merged.r2_values = merged.r2_values.iloc[keep_idx].reset_index(drop=True)

        merged.all_b = [merged.all_b[i] for i in keep_idx]
        merged.all_p = [merged.all_p[i] for i in keep_idx]
        merged.all_predictors = [merged.all_predictors[i] for i in keep_idx]

        merged.summary_df = merged.summary_df.iloc[keep_idx].reset_index(drop=True)

    if isinstance(merged.y_name, list):
        merged.y_name = [
            t if isinstance(t, tuple) else (t,)
            for t in merged.y_name
        ]
    else:
        # If it ended up as a single string (or something else), wrap it once:
        if not isinstance(merged.y_name, tuple):
            merged.y_name = (merged.y_name,)

    # ----------------- ENSURE EACH x_name IS A ONE‐ELEMENT TUPLE -----------------
    if isinstance(merged.x_name, list):
        merged.x_name = [
            t if isinstance(t, tuple) else (t,)
            for t in merged.x_name
        ]
    else:
        if not isinstance(merged.x_name, tuple):
            merged.x_name = (merged.x_name,)

    return merged


def space_size(iterable) -> int:
    """
    Calculate the size of the power set of the given iterable.

    Parameters
    ----------
    iterable: iterable
        Input iterable.

    Returns
    ----------
    int:
        Size of the power set of the input iterable.
    """
    n = len(iterable)
    return int(2**n)


def all_subsets(ss):
    """
    Generate all subsets of a given iterable.

    Parameters
    ----------
    ss: iterable
        Input iterable.

    Returns
    ----------
    itertools.chain:
        A chain object containing all subsets of the input iterable.
    """
    return chain(*map(lambda x: combinations(ss, x),
                      range(0, len(ss) + 1)))


def make_aic(ll: float, n: int) -> float:
    return (-2.0 * ll) + (2 * n)

def make_bic(ll: float, n: int, k: int) -> float:
    return (-2.0 * ll) + (k * np.log(n))

def make_hqic(ll: float, n: int, k: int) -> float:
    return -2 * ll + 2 * k * np.log(np.log(n))


def logistic_regression_sm(y, x) -> dict:
    """
    Perform logistic regression based on statsmodels.Logit.

    Parameters
    ----------
    y : array-like
        Dependent variable values.
    x : array-like
        Independent variable values. The matrix should be shaped as (number of observations, number of independent variables).

    Returns
    ----------
    dict: Dictionary containing regression results, including coefficients, p-values, log-likelihood,
          AIC, BIC, and HQIC.
    """
    model = sm.Logit(y, x)
    result = model.fit(method='newton', tol=1e-7, disp=0)
    n = result.nobs
    k = result.df_model + 1
    ll = result.llf

    null_model = sm.Logit(y, np.ones_like(y))  # only intercept
    result_null = null_model.fit(method='newton', tol=1e-7, disp=0)
    ll_null = result_null.llf
    r2 = 1 - (ll / ll_null)
    return {'b': [[x] for x in result.params.values],
            'p': [[x] for x in result.pvalues.values],
            'r2': r2,
            'll': ll,
            'aic': make_aic(ll, n),
            'bic': make_bic(ll, n, k),
            'hqic': make_hqic(ll, n, k)
            }


def logistic_regression_sm_stripped(y, x) -> dict:
    """
    Perform logistic regression using statsmodels with stripped output.

    Parameters
    ----------
    y : array-like
        Dependent variable values.
    x: array-like
        Independent variable values. The matrix should be shaped as
                   (number of observations, number of independent variables).

    Returns
    ----------
    dict: A dictionary containing regression coefficients ('b') and corresponding
          p-values ('p') for each independent variable.
    """
    X_const = sm.add_constant(x, prepend=False)
    model = sm.Logit(y, X_const)
    result = model.fit(method='newton', tol=1e-8, disp=0)
#    n = result.nobs
#    k = result.df_model + 1
    ll = result.llf
    null_model = sm.Logit(y, np.ones_like(y))  # only intercept
    result_null = null_model.fit(method='newton', tol=1e-8, disp=0)
    ll_null = result_null.llf
    r2 = 1 - (ll / ll_null)
    return {'b': [[x] for x in result.params.values],
            'p': [[x] for x in result.pvalues.values],
            'r2': r2,
#            'll': ll#,
#            'aic': make_aic(ll, n),
#            'bic': make_bic(ll, n, k),
#            'hqic': make_hqic(ll, n, k)
            }



def simple_ols(y, x) -> dict:
    """
    Perform simple ordinary least squares regression.

    Parameters
    ----------
    y : array-like
        Dependent variable.
    x : array-like
        Independent variables.

    Returns
    ----------
    dict: Dictionary containing regression results, including coefficients, p-values, log-likelihood,
          AIC, BIC, and HQIC.
    """
    x = np.asarray(x)
    y = np.asarray(y)
    if x.size == 0 or y.size == 0:
        raise ValueError("Inputs must not be empty.")
    try:
        inv_xx = np.linalg.inv(np.dot(x.T, x))
    except np.linalg.LinAlgError:
        inv_xx = np.linalg.pinv(np.dot(x.T, x))
    xy = np.dot(x.T, y)
    b = np.dot(inv_xx, xy)  # estimate coefficients
    nobs = y.shape[0]  # number of observations
    ncoef = x.shape[1]  # number of coef.
    df_e = nobs - ncoef  # degrees of freedom, error
    # df_r = ncoef - 1  # degrees of freedom, regression
    e = y - np.dot(x, b)  # residuals
    sse = np.dot(e.T, e) / df_e  # SSE
    se = np.sqrt(np.diagonal(sse * inv_xx))  # coef. standard errors
    t = b / se  # coef. t-statistics
    p = (1 - scipy.stats.t.cdf(abs(t), df_e)) * 2  # coef. p-values
    R2 = 1 - e.var() / y.var()  # model R-squared
    R2adj = 1 - (1 - R2) * ((nobs - 1) / (nobs - ncoef))  # adjusted R-square
    ll = (-(nobs * 1 / 2) * (1 + np.log(2 * np.pi)) - (nobs / 2)
          * np.log(abs(np.dot(e.T, e) / nobs)))
    return {'b': b,
            'p': p,
            'r2': R2adj,
            'll': ll,
            'aic': make_aic(ll, nobs),
            'bic': make_bic(ll, nobs, ncoef),
            'hqic': make_hqic(ll, nobs, ncoef)
            }


def group_demean(x: pd.DataFrame, group: Optional[str] = None) -> pd.DataFrame:
    """
    Demean the input data within groups.

    Parameters
    ----------
    x:  pd.DataFrame
        Input DataFrame.
    group : str, optional
        Column name for grouping. Default is None.

    Returns
    ----------
    pd.DataFrame: Demeaned DataFrame.
    """
    data = x.copy()
    if group is None:
        return data - np.mean(data)
    data_gm = data.groupby([group]).transform('mean')
    out = data.drop(columns=group) - data_gm
    return pd.concat([out, data[group]], axis=1)


def decorator_timer(func: callable) -> callable:
    """
    Decorator to time function execution.

    Parameters
    ----------
    func : callable
        Function to wrap.

    Returns
    -------
    callable
        Wrapped function returning (result, elapsed_seconds).
    """
    from time import time

    def wrapper(*args, **kwargs):
        t1 = time()
        result = callable(*args, **kwargs)
        end = time()-t1
        return result, end
    return wrapper


def get_selection_key(specs: List[List[str]]) -> List[frozenset]:
    """
    Convert list of spec lists into list of frozensets.

    Parameters
    ----------
    specs : list of list of str
        Each inner list is one specification.

    Returns
    -------
    list of frozenset
        Immutable keys for each specification.

    Raises
    ------
    ValueError
        If `specs` is not list of lists.
    """
    if all(isinstance(ele, list) for ele in specs):
        target = [frozenset(x) for x in specs]
        return target
    else:
        raise ValueError('Argument `specs` must be a list of list.')

'''
def get_colormap_colors(
    colormap_name: str,
    num_colors: int = 3,
    brightness_threshold: float = 0.7
) -> List[Tuple[float, float, float, float]]:
    """
    Extract a set of colors from a Matplotlib colormap, adjusting for brightness.

    Parameters
    ----------
    colormap_name : str
        Name of the Matplotlib colormap to sample from.
    num_colors : int, default 3
        Number of distinct colors to return.
    brightness_threshold : float, default 0.7
        Maximum allowed average brightness; colors brighter than this are darkened.

    Returns
    -------
    List[Tuple[float, float, float, float]]
        A list of RGBA tuples representing the selected colors.
    """
    colormap = plt.get_cmap(colormap_name)

    # Generate evenly spaced intervals between 0 and 1
    indices = np.linspace(0, 1, num_colors)

    # Extract the corresponding colors from the colormap
    colors = []
    for i in indices:
        color = colormap(i)

        # Calculate brightness (using a simple average of the RGB values)
        brightness = sum(color[:3]) / 3

        # If brightness is too high, move to a darker color
        while brightness > brightness_threshold and i > 0:
            i -= 0.05  # Move to a slightly darker color in the colormap
            color = colormap(i)
            brightness = sum(color[:3]) / 3

        colors.append(color)

    return colors
'''

def get_colormap_colors(num_colors: int = 3) -> List[str]:
    r"""
    Return the first \(\texttt{num_colors}\) entries of a predetermined color palette.

    Parameters
    ----------
    num_colors : int, optional
        The number of colors to return. Must satisfy
        \[
            1 \;\le\; \texttt{num_colors} \;\le\; N,
        \]
        where \(N = 4\) is the total number of available colors in the palette.
        Defaults to 3.

    Returns
    -------
    List[str]
        A list of hexadecimal color strings of length exactly \(\texttt{num_colors}\).

    Raises
    ------
    TypeError
        If `num_colors` is not an integer.
    ValueError
        If `num_colors < 1` or `num_colors > 4`.
    """

    palette: List[str] = ["#345995", "#B80C09", "#D4AF37", '#2E6F40', "#955196", "#3C4CAD"]

    # Validate type
    if not isinstance(num_colors, int):
        raise TypeError(f"num_colors must be an integer, got {type(num_colors).__name__!r}")

    N = len(palette)
    # Validate bounds
    if num_colors < 1 or num_colors > N:
        raise ValueError(f"num_colors must be between 1 and {N}, inclusive (got {num_colors}).")

    return palette[:num_colors]

def get_colors(specs: List[List[str]], color_set_name: Optional[str] = 'Set1') -> List[Tuple[float, float, float, float]]:
    """
    Generate a palette of colors for a list of specifications using a categorical colormap.

    Parameters
    ----------
    specs : list of list of str
        Each inner list represents one specification (set of variable names).
    color_set_name : str, optional
        Name of a Matplotlib qualitative colormap (default 'Set1').

    Returns
    -------
    List[Tuple[float, float, float, float]]
        A list of RGBA tuples, one per specification.

    Raises
    ------
    ValueError
        If `specs` is not a list of lists.
    """
    if color_set_name is None:
        color_set_name = 'Set1'
    if all(isinstance(ele, list) for ele in specs):
        colorset = matplotlib.colormaps[color_set_name]
        colorset = colorset.resampled(len(specs))
        return colorset.colors
    else:
        raise ValueError('Argument `specs` must be a list of list.')


def join_sig_test(*,
                  results_target,
                  results_shuffled,
                  sig_level,
                  positive):
    """
    Calculate joint significance test for the entire specification curve.

    Parameters
    ----------
    results_target : OLSResult
        Results object from the original analysis.
    results_shuffled : OLSResult
        Results object from shuffled analysis.
    sig_level : float
        Significance level threshold for specifications.
    positive : bool
        Direction of the joint significance test.

    Returns
    ----------
    float:
        Estimated p-value for the joint significance test.
    """
    if positive:
        target_sig_n = sum(results_target.summary_df.ci_down > 0)
        n_draws = []
        for col in results_shuffled.estimates:
            idx = results_shuffled.estimates[col] > 0
            n_draw = sum(results_shuffled.p_values[col][idx] < sig_level)
            n_draws.append(n_draw)
        shuffle_sig_n = sum(np.array(n_draws) >= target_sig_n)
        return shuffle_sig_n/results_shuffled.estimates.shape[1]
    else:
        target_sig_n = sum(results_target.summary_df.ci_up < 0)
        n_draws = []
        for col in results_shuffled.estimates:
            idx = results_shuffled.estimates[col] < 0
            n_draw = sum(results_shuffled.p_values[col][idx] < sig_level)
            n_draws.append(n_draw)
        shuffle_sig_n = sum(np.array(n_draws) >= target_sig_n)
        return shuffle_sig_n/results_shuffled.estimates.shape[1]


def prepare_union(path_to_union: str) -> Tuple[str, List[str], str, pd.DataFrame]:
    """
    Load and preprocess the classic union dataset for example analyses.

    Parameters
    ----------
    path_to_union : str
        Path to the Stata (.dta) file containing union data.

    Returns
    -------
    tuple
        y (str): Dependent variable name ('log_wage').
        c (List[str]): Control variable names.
        x (str): Treatment variable name ('union').
        final_data (pd.DataFrame): Cleaned DataFrame ready for modeling.

    Raises
    ------
    FileNotFoundError
        If the specified file does not exist.
    """
    union_df = pd.read_stata(path_to_union)
    union_df[['smsa','collgrad','married','union']]= union_df[['smsa','collgrad','married','union']].astype('str')
    union_df.loc[:, 'log_wage'] = np.log(union_df['wage'].copy()) * 100
    union_df = union_df[union_df['union'].notnull()].copy()
    union_df.loc[:, 'union'] = np.where(union_df['union'] == 'union', 1, 0)
    union_df.loc[:, 'married'] = np.where(union_df['married'] == 'married', 1, 0)
    union_df.loc[:, 'collgrad'] = np.where(union_df['collgrad'] == 'college grad', 1, 0)
    union_df.loc[:, 'smsa'] = np.where(union_df['smsa'] == 'SMSA', 1, 0)
    union_df[['smsa', 'collgrad', 'married', 'union']] = union_df[['smsa', 'collgrad', 'married', 'union']].astype('int')
    indep_list = ['hours',
                  'age',
                  'grade',
                  'collgrad',
                  'married',
                  'south',
                  'smsa',
                  'c_city',
                  'ttl_exp',
                  'tenure']
    for var in indep_list:
        union_df = union_df[union_df[var].notnull()]
    y = 'log_wage'
    c = indep_list
    x = 'union'
    final_data = pd.merge(union_df[y],
                          union_df[x],
                          how='left',
                          left_index=True,
                          right_index=True)
    final_data = pd.merge(final_data,
                          union_df[indep_list],
                          how='left',
                          left_index=True,
                          right_index=True)
    final_data = final_data.reset_index(drop=True)
    return y, c, x, final_data


def prepare_asc(asc_path: str) -> Tuple[str, List[str], List[str], str, pd.DataFrame]:
    """
    Load and preprocess the ASC example dataset for illustration.

    Parameters
    ----------
    asc_path : str
        Path to the Stata (.dta) file containing ASC data.

    Returns
    -------
    tuple
        y (str): Dependent variable name.
        x (List[str]): Continuous predictor names.
        c (List[str]): Control variable names.
        group (str): Grouping variable name ('pidp').
        ASC_df (pd.DataFrame): Cleaned DataFrame.
    """
    ASC_df = pd.read_stata(asc_path, convert_categoricals=False)
    one_hot = pd.get_dummies(ASC_df['year'])
    ASC_df = ASC_df.join(one_hot)
    ASC_df['dcareNew*c.lrealgs'] = ASC_df['dcareNew'] * ASC_df['lrealgs']
    #ASC_df['constant'] = 1
    ASC_df = ASC_df[['wellbeing_kikert', 'lrealgs', 'dcareNew*c.lrealgs', 'dcareNew',
                     'DR', 'lgva', 'Mtotp', 'ddgree', 'age',
                     2005, 2006.0, 2007.0, 2009.0,
                     2010.0, 2011.0, 2012.0, 2013.0, 2014.0,
                     2015.0, 2016.0, 2017.0, 2018.0,
                     'married', 'widowed', 'disable', 'lrealtinc_m',
                     'house_ownership', 'hhsize', 'work', 'retired',
                     #'constant'
                     'pidp'
                     ]]
    ASC_df = ASC_df.dropna()
    y = 'wellbeing_kikert'
    x = ['lrealgs', 'dcareNew*c.lrealgs', 'dcareNew',
         'DR', 'lgva', 'Mtotp', 'ddgree', 'age',
         2005, 2006.0, 2007.0, 2009.0,
         2010.0, 2011.0, 2012.0, 2013.0, 2014.0,
         2015.0, 2016.0, 2017.0, 2018.0]
    c = ['married', 'widowed', 'disable', 'lrealtinc_m',
         'house_ownership', 'hhsize', 'work', 'retired'
         ]
    group = 'pidp'
    ASC_df['pidp'] = ASC_df['pidp'].astype(int)
    return y, c, x, group, ASC_df


def reservoir_sampling(generator: Iterable, k: int) -> List:
    """
    Uniformly sample k items from a streaming generator (reservoir sampling).

    Parameters
    ----------
    generator : Iterable
        An iterator or generator yielding items.
    k : int
        Number of samples to retain.

    Returns
    -------
    List
        A list of k sampled items.
    """
    reservoir = []
    for i, item in enumerate(generator):
        if i < k:
            reservoir.append(item)
        else:
            j = random.randint(0, i)  # Randomly choose an index from 0 to i
            if j < k:
                reservoir[j] = item  # Replace element at the chosen index with new item
    return reservoir


import numpy as np
from typing import Sequence

def pseudo_r2(
    y_true: Sequence,
    y_pred: Sequence,
    mean_y_train: float
) -> float:
    """
    Compute the pseudo-R² (1 - MSE_model / MSE_null), coercing inputs to floats.

    Parameters
    ----------
    y_pred : Sequence
        Model predictions (can be list/array of floats or strings convertible to float).
    y_true : Sequence
        True target values (same length as y_pred).
    mean_y_train : float
        The baseline prediction (e.g. the training‐set mean of y).

    Returns
    -------
    float
        Pseudo‐R² = 1 - (MSE_model / MSE_null).

    Raises
    ------
    ValueError
        If lengths differ, if mean‐square‐null is zero, or if conversion to float fails.
    """
    # --- Coerce to numpy arrays of floats ---
    try:
        y_pred_arr = np.asarray(y_pred, dtype=float)
        y_true_arr = np.asarray(y_true, dtype=float)
    except Exception as e:
        raise ValueError(f"Could not convert inputs to floats: {e!s}")

    if y_pred_arr.shape != y_true_arr.shape:
        raise ValueError(
            f"Length mismatch: y_pred has shape {y_pred_arr.shape}, "
            f"y_true has shape {y_true_arr.shape}"
        )

    n = y_true_arr.size

    # --- Compute mean‐squared errors ---
    mse_model = np.mean((y_true_arr - y_pred_arr) ** 2)
    mse_null  = np.mean((y_true_arr - float(mean_y_train)) ** 2)

    if mse_null == 0.0:
        raise ValueError(
            "MSE_null is zero → pseudo-R^2 undefined (division by zero)."
        )

    return 1.0 - mse_model / mse_null


def mcfadden_r2(y_true, y_prob, insample_mean):
    """
    Compute McFadden's pseudo R-squared for logistic regression.
    """
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    # Compute log-likelihood for the fitted model
    eps = 1e-15  # to avoid log(0)
    y_prob = np.clip(y_prob, eps, 1 - eps)
    log_l_model = np.sum(y_true * np.log(y_prob) + (1 - y_true) * np.log(1 - y_prob))
    # Compute log-likelihood for the null model
    p_null = insample_mean
    log_l_null = np.sum(y_true * np.log(p_null) + (1 - y_true) * np.log(1 - p_null))
    return 1 - (log_l_model / log_l_null)


def calculate_imv_score(y_true, y_enhanced):
    """
    Calculates the IMV (Information Metric Value) score.

    Parameters:
    - y_true: array-like of binary true labels (0 or 1)
    - y_enhanced: array-like of predicted probabilities from an enhanced model

    Returns:
    - IMV score: relative improvement of enhanced model over the null model
    """
    y_true,y_enhanced = np.asarray(y_true),np.asarray(y_enhanced)

    def ll(x, p):
        epsilon = 1e-4  # avoid log(0)
        z = (np.log(p + epsilon) * x) + (np.log(1 - p + epsilon) * (1 - x))
        return np.exp(np.sum(z) / len(z))

    def minimize_me(p, a):
        return abs((p * np.log(p)) + ((1 - p) * np.log(1 - p)) - np.log(a))

    def get_w(a, guess=0.5, bounds=[(0.5, 0.999)]):
        res = minimize(minimize_me, guess, args=(a,),
                       options={'ftol': 0, 'gtol': 1e-9},
                       method='L-BFGS-B', bounds=bounds)
        return res.x[0]

    # Null model: always predict mean(y_true)
    y_null = np.full_like(y_true, fill_value=np.mean(y_true), dtype=float)

    # Compute likelihoods
    ll_null = ll(y_true, y_null)
    ll_enhanced = ll(y_true, y_enhanced)

    # Transform to entropy space via get_w
    w_null = get_w(ll_null)
    w_enhanced = get_w(ll_enhanced)

    # Compute IMV
    imv = (w_enhanced - w_null) / w_null
    return imv


def sample_y_masks(
    n_y: int,                      # how many raw outcome variables
    n_masks: int,                  # how many composites you want
    seed: Optional[int] = None
) -> List[int]:
    """
    Uniformly sample `n_masks` bit-masks from the non-empty power-set of
    `n_y` items **without** enumerating the 2^n_y possibilities.

    Returns
    -------
    list[int]   each mask is an `int` whose binary representation tells
                which outcomes enter the composite.
    """
    full_space = (1 << n_y) - 1        # ignore the 0/empty mask
    if n_masks >= full_space:
        # exhaustive: 1 … (2^n_y − 1)
        return list(range(1, full_space + 1))

    rng   = np.random.default_rng(seed)
    masks = rng.choice(
        full_space,
        size=n_masks,
        replace=False
    ) + 1           # shift from 0..full_space-1 to 1..full_space
    return masks.tolist()


def sample_z_masks(
    n_z: int,                      # how many raw outcome variables
    n_masks: int,                  # how many composites you want
    seed: Optional[int] = None
) -> List[int]:
    """
    Uniformly sample `n_masks` bit-masks from the non-empty power-set of
    `n_y` items **without** enumerating the 2^n_y possibilities.

    Returns
    -------
    list[int]   each mask is an `int` whose binary representation tells
                which outcomes enter the composite.
    """
    full_space = (1 << n_z)
    if n_masks >= full_space:
        # exhaustive: 1 … (2^n_y − 1)
        return list(range(1, full_space))

    rng   = np.random.default_rng(seed)
    masks = rng.choice(
        full_space,
        size=n_masks,
        replace=False
    )
    return masks.tolist()
