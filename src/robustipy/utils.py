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
from multiprocessing import cpu_count
import sys


class IntegerRangeValidator:
    """
    Validator that checks if an input value is an integer within a specified range.

    Args:
        min_value (int): The minimum allowed integer value (inclusive).
        max_value (int): The maximum allowed integer value (inclusive).

    Raises:
        ValidationError: If the input is not an integer or is outside the specified range.

    Usage:
        validator = IntegerRangeValidator(1, 10)
        validator(_, current_value)  # Returns True if valid, raises ValidationError otherwise.
    """
    def __init__(self, min_value, max_value):
        self.min_value = min_value
        self.max_value = max_value

    def __call__(self, _, current):
        try:
            value = int(current)
        except ValueError:
            raise ValidationError('', reason="Input must be an integer.")
        if not (self.min_value <= value <= self.max_value):
            raise ValidationError(
                '',
                reason=f"Input must be between {self.min_value} and {self.max_value} (inclusive)."
            )
        return True


def _running_in_jupyter() -> bool:
    """
    Return True exactly if this process is running inside a Jupyter (ZMQInteractiveShell) kernel.
    """
    try:
        from IPython import get_ipython
        shell = get_ipython().__class__.__name__
        return shell == "ZMQInteractiveShell"
    except (ImportError, AttributeError):
        return False

def _is_real_tty() -> bool:
    """
    Return True exactly if both sys.stdin and sys.stdout are real TTYs.
    (That is, neither has been redirected to a file, and we are not inside Jupyter.)
    """
    return sys.stdin.isatty() and sys.stdout.isatty()

def is_interactive() -> bool:
    """
    Return True if either:
      (a) we are inside a Jupyter notebook/lab, OR
      (b) we are running from a real terminal (both stdin and stdout are TTYs).
    """
    return _running_in_jupyter() or _is_real_tty()

# ───────────────────────────────────────────────────────────────────────────────
#  Revised make_inquiry: use three‐way branching
# ───────────────────────────────────────────────────────────────────────────────

def make_inquiry(
    model_name,
    y,
    data,
    draws,
    kfolds,
    oos_metric,
    n_cpu,
    seed
):
    """
    Prompt the user for missing inputs if in an interactive environment;
    otherwise, silently fall back to default values.

    Returns
    -------
    tuple[int, int, str, int, int]
        (draws, kfolds, oos_metric, n_cpu, seed)
    """

    # 1) Determine which metrics apply to this model_name:
    if model_name == 'OLS Robust':
        oos_metric_choices = ['pseudo-r2', 'rmse']
    elif model_name == 'Logistic Regression Robust':
        oos_metric_choices = ['mcfadden-r2', 'pseudo-r2', 'rmse', 'cross-entropy']
    else:
        raise ValueError(f"Unknown model_name: {model_name}")

    # 2) Check interactive status:
    in_jupyter = _running_in_jupyter()
    in_real_tty = _is_real_tty()

    # 3) Helper: pure‐Python loop for integer input in [min_val, max_val]:
    def _ask_integer(prompt_text: str, min_val: int, max_val: int) -> int:
        """
        Repeatedly call input() until user enters a valid integer in [min_val, max_val].
        Raise ValidationError otherwise. Return the integer once valid.
        """
        while True:
            raw = input(f"{prompt_text} [{min_val}–{max_val}]: ").strip()
            try:
                val = int(raw)
            except ValueError:
                print(f"Invalid: '{raw}' is not an integer.")
                continue
            if val < min_val or val > max_val:
                print(f"Invalid: must be between {min_val} and {max_val}.")
                continue
            return val

    # 4) Helper: pure‐Python loop for selecting one choice from choices list:
    def _ask_choice(prompt_text: str, choices: list[str]) -> str:
        """
        Print all choices with numeric indices, then loop until the user types
        either a valid index (1–len(choices)) or a choice name exactly.
        Return the chosen string.
        """
        # Display a numbered list:
        print(prompt_text)
        for idx, choice in enumerate(choices, start=1):
            print(f"  {idx}. {choice}")
        while True:
            raw = input(f"Type number (1–{len(choices)}) or exact choice: ").strip()
            # Try interpreting as integer index:
            if raw.isdigit():
                idx = int(raw)
                if 1 <= idx <= len(choices):
                    return choices[idx - 1]
                else:
                    print(f"Number out of range. Must be 1–{len(choices)}.")
                    continue
            # Otherwise check if it exactly matches one of the choices:
            if raw in choices:
                return raw
            print(f"Invalid: type one of {choices}, or a valid index.")

    # ───────────────────────────────────────────────────────────────
    # 5) If draws is missing, decide which prompt to run:
    # ───────────────────────────────────────────────────────────────
    if draws is None:
        if in_real_tty:
            # (a) Real TTY: use inquirer.Text with IntegerRangeValidator
            question_draws = [
                inquirer.Text(
                    'draws_inq',
                    message="Enter number of bootstrap draws",
                    validate=IntegerRangeValidator(2, 1000000)
                )
            ]
            # inquirer.prompt(...) returns a dict; we extract and convert to int
            draws = int(inquirer.prompt(question_draws, theme=GreenPassion())['draws_inq'])
        elif in_jupyter:
            # (b) Jupyter: fall back to pure input() + validator
            draws = _ask_integer("Enter number of bootstrap draws", 2, 1000000)
        else:
            # (c) Neither TTY nor Jupyter: use default
            draws = 1000

    # ───────────────────────────────────────────────────────────────
    # 6) If kfolds is missing, prompt for number of folds:
    # ───────────────────────────────────────────────────────────────
    if kfolds is None:
        # The maximum valid folds is len(data[y]); we compute it now:
        max_k = len(data[y])
        if in_real_tty:
            question_kfolds = [
                inquirer.Text(
                    'kfolds_inq',
                    message="Enter number of folds for cross-validation",
                    validate=IntegerRangeValidator(2, max_k)
                )
            ]
            kfolds = int(inquirer.prompt(question_kfolds, theme=GreenPassion())['kfolds_inq'])
        elif in_jupyter:
            kfolds = _ask_integer("Enter number of folds for cross-validation", 2, max_k)
        else:
            kfolds = 10

    # ───────────────────────────────────────────────────────────────
    # 7) If oos_metric is missing, prompt the user to choose one:
    # ───────────────────────────────────────────────────────────────
    if oos_metric is None:
        if in_real_tty:
            question_oos = [
                inquirer.List(
                    'oos_inq',
                    message="Select the out-of-sample evaluation metric",
                    choices=oos_metric_choices,
                    carousel=True
                )
            ]
            oos_metric = inquirer.prompt(question_oos, theme=GreenPassion())['oos_inq']
        elif in_jupyter:
            oos_metric = _ask_choice("Select the out-of-sample evaluation metric:", oos_metric_choices)
        else:
            oos_metric = 'pseudo-r2'

    # ───────────────────────────────────────────────────────────────
    # 8) If n_cpu is missing, ask about CPU count:
    # ───────────────────────────────────────────────────────────────
    if n_cpu is None:
        default_cpus = max(1, cpu_count() - 1)
        if in_real_tty:
            question_ncpu = [
                inquirer.List(
                    'ncpu_inq',
                    message=f"You haven’t specified the number of CPUs. Is {default_cpus} ok?",
                    choices=['Yes', 'No'],
                    carousel=True
                )
            ]
            n_cpu_answer = inquirer.prompt(question_ncpu, theme=GreenPassion())['ncpu_inq']
            if n_cpu_answer == 'Yes':
                n_cpu = default_cpus
            else:
                question_ncpu2 = [
                    inquirer.Text(
                        'ncpu_inq',
                        message="Enter number of CPUs to use",
                        validate=IntegerRangeValidator(1, cpu_count())
                    )
                ]
                n_cpu = int(inquirer.prompt(question_ncpu2, theme=GreenPassion())['ncpu_inq'])
        elif in_jupyter:
            # In Jupyter, ask via input() if default is OK:
            ans = input(f"You haven’t specified the number of CPUs. Is {default_cpus} okay? (yes/no): ").strip().lower()
            if ans in ('y', 'yes'):
                n_cpu = default_cpus
            else:
                n_cpu = _ask_integer("Enter number of CPUs to use", 1, cpu_count())
        else:
            n_cpu = default_cpus

        # ───────────────────────────────────────────────────────────────
        #  9) If seed is missing, prompt the user (TTY, Jupyter, or fallback)
        # ───────────────────────────────────────────────────────────────
    SEED_MIN = 0
    SEED_MAX = 2 ** 31 - 1
    in_jupyter = _running_in_jupyter()
    in_real_tty = _is_real_tty()
    def _ask_seed_input(prompt_text: str, min_val: int, max_val: int) -> int:
        """
        Repeatedly call input() until user enters a valid integer in [min_val, max_val].
        """
        while True:
            raw = input(f"{prompt_text} [{min_val}–{max_val}]: ").strip()
            try:
                val = int(raw)
            except ValueError:
                print(f"Invalid: '{raw}' is not an integer.")
                continue
            if val < min_val or val > max_val:
                print(f"Invalid: must be between {min_val} and {max_val}.")
                continue
            return val
    if seed is None:
        if in_real_tty:
            question_seed = [
                inquirer.Text(
                    'seed_inq',
                    message="Enter integer seed for reproducibility",
                    validate=IntegerRangeValidator(SEED_MIN, SEED_MAX)
                )
            ]
            seed = int(inquirer.prompt(question_seed, theme=GreenPassion())['seed_inq'])
        elif in_jupyter:
            seed = _ask_seed_input("Enter integer seed for reproducibility", SEED_MIN, SEED_MAX)
        else:
            seed = 192735
    return draws, kfolds, oos_metric, n_cpu, seed


def rescale(variable):
    """
    Rescales the input variable to have zero mean and unit standard deviation.

    Parameters
    ----------
    variable : array-like
        Input data to be rescaled. Can be a list, NumPy array, or similar structure.

    Returns
    -------
    out : ndarray
        The rescaled array with mean 0 and standard deviation 1 along the specified axis.
        NaN values are ignored in the computation of mean and standard deviation.

    Notes
    -----
    This function uses `np.nanmean` and `np.nanstd` to ignore NaN values during scaling.
    """
    variable = np.asarray(variable, dtype=np.float64)
    mean = np.nanmean(variable, axis=0, keepdims=True)
    std = np.nanstd(variable, axis=0, keepdims=True)
    out = (variable - mean) / std
    return out


def _ensure_single_constant(
        dataframe: pd.DataFrame,
        y_list: List[str],
        x_list: List[str],
        controls: List[str],
) -> Tuple[List[str], List[str]]:
    """
    Revised version: if a zero‐variance column appears in both x_list and controls,
    it will be kept in 'controls' (not moved into x_list).  More precisely:

    1) If there is already a column literally named "const", drop it and remove it
       from y_list, x_list, and controls (to avoid stale duplicates).
    2) Among the columns in (y_list ∪ x_list ∪ controls) ∩ dataframe.columns,
       identify those with zero variance.
    3) If none exist, create a brand‐new "const" column (all 1.0) and append it to x_list.
    4) If one or more exist, choose exactly one to keep according to the new priority:
         (a) Any zero‐variance column in controls
         (b) Otherwise any zero‐variance column in x_list
         (c) Otherwise any zero‐variance column in y_list
         (d) Otherwise the alphabetically first among them
       If there are multiple, warn that only one will be kept and the others dropped.
    5) Drop all zero‐variance columns except the chosen one, and remove them from
       whichever of y_list, x_list, controls they appeared in.
    6) For the chosen column "keep":
       • If it was originally in controls (regardless of whether also in x_list), then
         after renaming it to "const" it stays in controls and is removed from x_list (if present).
       • Else if it was originally only in x_list, then after renaming it to "const" it stays in x_list.
       • Else if it was originally only in y_list, then after renaming it to "const", it is moved into x_list.
       In all cases, remove "keep" from any list where it does not belong post‐rename.
    7) Return the updated (x_list, controls).

    This ensures that if a constant was specified as a control, it remains a control
    (now called "const") rather than being moved into x_list.
    """

    y_list = y_list.copy()
    x_list = x_list.copy()
    controls = controls.copy()

    # ──────────────────────────────────────────────────────────────────────────
    # 1) Drop any pre‐existing "const" column:
    # ──────────────────────────────────────────────────────────────────────────
    if "const" in dataframe.columns and "const" not in controls:
        dataframe.drop(columns=["const"], inplace=True)
        if "const" in y_list:
            y_list.remove("const")
        if "const" in x_list:
            x_list.remove("const")
    # ──────────────────────────────────────────────────────────────────────────
    # 2) Identify zero‐variance columns among the “relevant” ones:
    # ──────────────────────────────────────────────────────────────────────────
    relevant_cols = [
        col for col in (y_list + x_list + controls)
        if col in dataframe.columns
    ]
    zero_var_cols = [
        col
        for col in relevant_cols
        if dataframe[col].nunique(dropna=False) == 1
    ]

    # ──────────────────────────────────────────────────────────────────────────
    # 3) If no zero‐variance column, create "const" = 1.0 and append to x_list:
    # ──────────────────────────────────────────────────────────────────────────
    if len(zero_var_cols) == 0:
        dataframe.loc[:, "const"] = 1.0
        x_list.append("const")
        return x_list, controls

    # ──────────────────────────────────────────────────────────────────────────
    # 4) Choose exactly one zero‐variance column to keep with new priority:
    #    (controls > x_list > y_list > alphabetical)
    # ──────────────────────────────────────────────────────────────────────────
    in_ctrl = [c for c in zero_var_cols if c in controls]
    in_x = [c for c in zero_var_cols if (c not in in_ctrl) and (c in x_list)]
    in_y = [c for c in zero_var_cols if (c not in in_ctrl) and (c not in in_x) and (c in y_list)]

    if in_ctrl:
        keep = in_ctrl[0]
    elif in_x:
        keep = in_x[0]
    elif in_y:
        keep = in_y[0]
    else:
        keep = sorted(zero_var_cols)[0]

    if len(zero_var_cols) > 1:
        warnings.warn(
            f"More than one constant‐valued column detected among {relevant_cols!r}: "
            f"{zero_var_cols!r}. Keeping only '{keep}' and dropping the others.",
            UserWarning
        )

    # ──────────────────────────────────────────────────────────────────────────
    # Record original membership of `keep`:
    orig_in_ctrl = (keep in controls)
    orig_in_x = (keep in x_list)
    orig_in_y = (keep in y_list)

    # ──────────────────────────────────────────────────────────────────────────
    # 5) Drop all zero‐variance columns except the one chosen (`keep`),
    #    and remove them from y_list, x_list, controls:
    # ──────────────────────────────────────────────────────────────────────────
    for c in zero_var_cols:
        if c == keep:
            continue
        if c in dataframe.columns:
            dataframe.drop(columns=[c], inplace=True)
        if c in y_list:
            y_list.remove(c)
        if c in x_list:
            x_list.remove(c)
        if c in controls:
            controls.remove(c)

    # ──────────────────────────────────────────────────────────────────────────
    # 6) Remove `keep` from any list it should no longer be in:
    #    We will rename `keep` to "const" below.
    # ──────────────────────────────────────────────────────────────────────────
    #    If `keep` was originally in controls, we want it to remain a control after renaming.
    #    So: remove it from x_list (if present) and from y_list. Do NOT remove from controls.
    if orig_in_ctrl:
        if keep in x_list:
            x_list.remove(keep)
        if keep in y_list:
            y_list.remove(keep)
        # `keep` remains in controls for now.
    else:
        # If `keep` was not originally in controls, remove it from controls if it somehow is there,
        # so that after renaming "const" goes to the correct list.
        if keep in controls:
            controls.remove(keep)

        # If `keep` was in x_list (but not in controls), we will keep it as a regressor.
        # If `keep` was only in y_list, we remove it from y_list and later add "const" to x_list.
        if keep in x_list:
            # We will replace keep→"const" in x_list below.
            pass
        else:
            # If it was in y_list only (and not in x_list or controls), remove it here.
            if keep in y_list:
                y_list.remove(keep)

    # ──────────────────────────────────────────────────────────────────────────
    # 7) Rename the single kept column (`keep`) to "const" (unless it already was "const"):
    # ──────────────────────────────────────────────────────────────────────────
    if keep != "const":
        # Rename the column in the DataFrame:
        dataframe.rename(columns={keep: "const"}, inplace=True)

        # Now place "const" into the correct list:
        if orig_in_ctrl:
            # The constant should remain in controls.  Replace `keep`→"const" in controls:
            controls[controls.index(keep)] = "const"
        elif orig_in_x:
            # The constant should remain among regressors.  Replace `keep`→"const" in x_list:
            x_list[x_list.index(keep)] = "const"
        else:
            # `keep` was originally only in y_list.  Now we add "const" to x_list:
            x_list.append("const")
            # (No need to modify controls in this branch.)

    if "const" in x_list:
        x_list = [c for c in x_list if c != "const"] + ["const"]
    if "const" in controls:
        controls = [c for c in controls if c != "const"] + ["const"]

    return x_list, controls



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
    # Calculate Akaike Information Criterion (AIC).
    return (-2.0 * ll) + (2 * n)

def make_bic(ll: float, n: int, k: int) -> float:
    # Calculate Bayesian Information Criterion (BIC).
    return (-2.0 * ll) + (k * np.log(n))

def make_hqic(ll: float, n: int, k: int) -> float:
    # Calculate Hannan-Quinn Information Criterion (HQIC).
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

    # Compute (X'X)^(-1) or use pseudo-inverse if singular
    try:
        inv_xx = np.linalg.inv(np.dot(x.T, x))
    except np.linalg.LinAlgError:
        inv_xx = np.linalg.pinv(np.dot(x.T, x))

    # Compute X'Y
    xy = np.dot(x.T, y)

    # Estimate coefficients: b = (X'X)^(-1) X'Y
    b = np.dot(inv_xx, xy)

    # Basic dimensions
    nobs = y.shape[0]       # number of observations
    ncoef = x.shape[1]      # number of coefficients
    df_e = nobs - ncoef     # degrees of freedom (residual)

    # Compute residuals
    e = y - np.dot(x, b)

    # Sum of squared errors divided by df: residual variance
    sse = np.dot(e.T, e) / df_e

    # Standard errors of coefficients
    se = np.sqrt(np.diagonal(sse * inv_xx))

    # T-statistics and p-values
    t = b / se
    p = (1 - scipy.stats.t.cdf(abs(t), df_e)) * 2

    # R² and adjusted R²
    R2 = 1 - e.var() / y.var()
    R2adj = 1 - (1 - R2) * ((nobs - 1) / (nobs - ncoef))

    # Log-likelihood of model under Gaussian errors
    ll = (-(nobs / 2) * (1 + np.log(2 * np.pi)) -
          (nobs / 2) * np.log(abs(np.dot(e.T, e) / nobs)))

    return {
        'b': b,
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
        result = func(*args, **kwargs)
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
        # Count significant positive coefficients in the actual results
        target_sig_n = sum(results_target.summary_df.ci_down > 0)

        n_draws = []

        # For each bootstrap draw in shuffled results
        for col in results_shuffled.estimates:
            idx = results_shuffled.estimates[col] > 0
            n_draw = sum(results_shuffled.p_values[col][idx] < sig_level)
            n_draws.append(n_draw)

        # Count how often shuffled draws meet or exceed target
        shuffle_sig_n = sum(np.array(n_draws) >= target_sig_n)

        return shuffle_sig_n / results_shuffled.estimates.shape[1]

    else:
        # Count significant negative coefficients in the actual results
        target_sig_n = sum(results_target.summary_df.ci_up < 0)

        n_draws = []

        # For each bootstrap draw in shuffled results
        for col in results_shuffled.estimates:
            idx = results_shuffled.estimates[col] < 0
            n_draw = sum(results_shuffled.p_values[col][idx] < sig_level)
            n_draws.append(n_draw)

        # Count how often shuffled draws meet or exceed target
        shuffle_sig_n = sum(np.array(n_draws) >= target_sig_n)

        return shuffle_sig_n / results_shuffled.estimates.shape[1]

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
        Or if MSE_null is zero (division by zero for pseudo-R²).
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
            "MSE_null is zero → pseudo-R² undefined (division by zero)."
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
        # Log-likelihood function for binary outcomes
        epsilon = 1e-4  # avoid log(0)
        z = (np.log(p + epsilon) * x) + (np.log(1 - p + epsilon) * (1 - x))
        return np.exp(np.sum(z) / len(z))

    def minimize_me(p, a):
        # Objective function to minimize
        return abs((p * np.log(p)) + ((1 - p) * np.log(1 - p)) - np.log(a))

    def get_w(a, guess=0.5, bounds=[(0.5, 0.999)]):
        # Find the value of p that minimizes the objective function
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
    `n_z` items **without** enumerating the 2^n_z possibilities.

    Returns
    -------
    list[int]   each mask is an `int` whose binary representation tells
                which specifications enter the composite.
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
