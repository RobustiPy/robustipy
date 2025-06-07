from abc import ABC, abstractmethod
import warnings
from typing import Optional, List
from multiprocessing import cpu_count
import numpy as np
import pandas as pd

from robustipy.utils import all_subsets, space_size, sample_y_masks


class Protomodel(ABC):
    """
    Prototype class, intended to be used in inheritance,
    not to be called.
    """
    def __init__(self):
        # upon instantiation calling data loading methods
        # and general sanity checks.
        self.y = None
        self.x = None
        self.results = None

    @abstractmethod
    def fit(self):
        # Public method to fit model
        pass


class Protoresult(ABC):
    """
    Prototype class for results object, intended to be used in inheritance,
    not to be called.
    """
    @abstractmethod
    def summary(self):
        # Public method to print summary
        pass

    @abstractmethod
    def plot(self):
        # Public method to plot general results
        pass


class MissingValueWarning(UserWarning):
    pass

def _check_numeric_columns(data, cols):
    """Check that all specified columns in the DataFrame are numeric."""
    non_numeric = data[cols].select_dtypes(exclude=[np.number]).columns.tolist()
    if non_numeric:
        raise ValueError(f"The following columns are not numeric and must be converted before fitting: {non_numeric}")


class BaseRobust(Protomodel):
    """
    Base class for robust model estimation, including OLS and logistic.

    Provides shared validation, bootstrapping, cross-validation,
    and composite outcome support.

    Attributes
    ----------
    y : list of str
        Dependent variable column names.
    x : list of str
        Independent variable column names.
    data : pandas.DataFrame
        Input dataset containing variables in y, x, controls.
    model_name : str
        Custom label for the model run.
    results : object
        Fitted result object populated after fit().
    parameters : dict
        Stores initialization parameters and any derived settings.
    """

    def __init__(
        self,
        *,
        y: list[str],
        x: list[str],
        data: pd.DataFrame,
        model_name: str = "BaseRobust"
    ) -> None:
        """
        Initialize the base robust model, validating inputs.

        Parameters
        ----------
        y : list of str
            Names of the dependent variable columns.
        x : list of str
            Names of the independent variable columns.
        data : pandas.DataFrame
            Dataset containing all necessary columns.
        model_name : str, default "BaseRobust"
            Label for this model, used in outputs.

        Raises
        ------
        TypeError, ValueError
            If inputs fail type or membership checks.
        """
        super().__init__()
        if not isinstance(y, list) or not isinstance(x, list):
            raise TypeError(
                "parameters 'y' and 'x' must each be a list of strings "
                "corresponding to DataFrame column names. "
                f"Received types: y={type(y).__name__}, x={type(x).__name__}. "
                "Hint: Try wrapping your variable names in square brackets, like y=['target'] and x=['treatment', 'covariate1']"
            )
        if not all(isinstance(col, str) for col in y + x):
            raise TypeError(
                "All elements in 'y' and 'x' must be strings corresponding to DataFrame columns."
            )
        if not isinstance(data, pd.DataFrame):
            raise TypeError(
                f"'data' must be a pandas DataFrame. Received type: {type(data).__name__}."
            )
        missing_vars = [col for col in (y + x) if col not in data.columns]
        if missing_vars:
            raise ValueError(
                "the following specified columns were not found in the DataFrame: "
                f"{missing_vars}. Ensure exact name matches (including case sensitivity)."
            )
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
    
#    def multiple_y(self) -> None:
#        """
#        Build `self.y_composites` and `self.y_specs`.
#
#        If `self._selected_y_masks` is not None, restrict to those masks.
#        """
#        self.y_specs = []
#        self.y_composites = []
#        print("Calculating Composite Ys")
#        subsets = list(all_subsets(self.y))
#        iterator = subsets 
#        for spec in iterator:
#            if len(spec) > 0:
#                subset = self.data[list(spec)]
#                subset = (subset - subset.mean()) / subset.std()
#                self.y_composites.append(subset.mean(axis=1))
#                self.y_specs.append(spec)
#        self.parameters['y_specs'] = self.y_specs
#        self.parameters['y_composites'] = self.y_composites

    def multiple_y(self) -> None:
        """
        Build the lists
            * self.y_composites  – pandas Series, one per composite Y
            * self.y_specs       – tuple[str], names that form that composite

        If `self.composite_sample` is a positive int, draw that many random
        non-empty subsets of the raw Y columns *before* we create any Series.
        Otherwise enumerate **all** non-empty subsets (original behaviour).
        """
        print("Calculating Composite Ys")
        y_cols = self.y                               # list[str] of raw outcome vars
        n_y    = len(y_cols)

        # ------------------------------------------------------------------
        # Decide which subsets to build
        # ------------------------------------------------------------------
        if getattr(self, "composite_sample", None) and self.composite_sample > 0:
            masks = sample_y_masks(
                n_y=n_y,
                n_masks=self.composite_sample,
                seed=getattr(self, "seed", None)
            )
            subset_iter = [
                tuple(y_cols[i] for i in range(n_y) if (m >> i) & 1)
                for m in masks
            ]
        else:
            # Exhaustive: use generator but skip the very first (empty) subset
            subset_iter = (
                spec for spec in all_subsets(y_cols) if spec  # truthy -> non-empty
            )
        # ------------------------------------------------------------------

        self.y_composites = []
        self.y_specs      = []

        for spec in subset_iter:
            subset = self.data[list(spec)]
            subset = (subset - subset.mean()) / subset.std(ddof=0)  # z-score
            self.y_composites.append(subset.mean(axis=1))
            self.y_specs.append(spec)

        # keep for reproducibility
        self.parameters["y_specs"]      = self.y_specs
        self.parameters["y_composites"] = self.y_composites
    
    def fit(self, *, controls: List[str], group: Optional[str] = None, draws: int = 500,
            kfold: int = 5, oos_metric: str = 'r-squared', n_cpu: Optional[int] = None,
            seed: Optional[int] = None) -> None:
        """
        Abstract fit method; must be overridden by subclasses.

        Parameters
        ----------
        controls : List[str]
            Optional control variable names to include in specifications.
        group : str, optional
            Column name for grouping (fixed effects) variable.
        draws : int, default=500
            Number of bootstrap draws.
        kfold : int, default=5
            Number of cross-validation folds.
        oos_metric : str, default='r-squared'
            Out-of-sample metric ('r-squared', 'rmse', etc.).
        n_cpu : int, optional
            Number of CPU cores for parallel computation.
        seed : int, optional
            Random seed for reproducibility.

        Raises
        ------
        NotImplementedError
            Always, since this method must be implemented by subclasses.
        """
        raise NotImplementedError("This method should be implemented in subclasses.")
    
    def _warn_if_large_draws(self, draws: int, controls: List[str], threshold: int = 10_000) -> None:
        """
        Warn if the total number of bootstrap models exceeds a threshold.

        Parameters
        ----------
        draws : int
            Number of draws per specification.
        controls : List[str]
            Control variable names to define specification space.
        threshold : int, default=10000
            Maximum safe number of total model runs.
        """
        est_specs = space_size(controls)
        y_space = len(getattr(self, "y_specs", [None]))  
        total_models = est_specs * y_space * draws

        if total_models > threshold:
            warnings.warn(
                f"You've requested {draws} bootstrap draws across {est_specs} control specs "
                f"and {y_space} y-composites, which is roughly {total_models:,} total model runs.\n\n"
                "This might lead to extended runtime or high memory usage.",
                UserWarning,
                stacklevel=2
            )

    def _check_numeric_columns_for_fit(self, controls: List[str], group: Optional[str]) -> None:
        """
        Validate that all required columns are numeric before fitting.

        Parameters
        ----------
        controls : List[str]
            Control variable names.
        group : str or None
            Grouping variable name, if any.
        """
        cols_to_check = self.y + self.x + ( [group] if group else [] ) + controls
        _check_numeric_columns(self.data, cols_to_check)

    def _check_colinearity(self, X: pd.DataFrame):
        """
        Check for perfect multicollinearity in X. Warn about all involved columns.
        """
        mat = X.values
        n_cols = mat.shape[1]
        rank  = np.linalg.matrix_rank(mat)

        if rank < n_cols:
            # Identify perfectly collinear pairs by checking correlation matrix
            corr = np.corrcoef(mat.T)
            problematic_pairs = []
            for i in range(n_cols):
                for j in range(i+1, n_cols):
                    if np.isclose(abs(corr[i, j]), 1.0, atol=1e-10):
                        problematic_pairs.append((X.columns[i], X.columns[j]))

            flat_problem_vars = sorted(set(var for pair in problematic_pairs for var in pair))

            raise ValueError(
                f"Perfect collinearity detected (rank={rank} < {n_cols}).\n"
                f"Variables involved in exact linear dependence: {flat_problem_vars}\n"
                f"Collinear pairs detected: {problematic_pairs}\n"
                "Please remove or merge these variables."
            )

    def _validate_fit_args(
        self,
        controls: List[str],
        group: Optional[str],
        draws: int,
        kfold: int,
        oos_metric: str,
        n_cpu: Optional[int],
        seed: Optional[int],
        valid_oos_metrics: List[str],
        threshold: int = 10_000
    ) -> int:
        """
        Shared validation for `fit` arguments across model subclasses.

        Parameters
        ----------
        controls : List[str]
            Control variable names.
        group : str or None
            Grouping variable name.
        draws : int
            Number of bootstrap draws.
        kfold : int
            Number of cross-validation folds; must be ≥2.
        oos_metric : str
            Must be one of `valid_oos_metrics`.
        n_cpu : int or None
            Number of CPU cores to use; if None, defaults to max(1, cpu_count()-1).
        seed : int or None
            Random seed for reproducibility.
        valid_oos_metrics : List[str]
            Permitted out-of-sample metrics.
        threshold : int, default=10000
            Threshold for total model runs warning.

        Returns
        -------
        int
            Validated `n_cpu` value to use.

        Raises
        ------
        TypeError, ValueError
            If any argument is of wrong type or out of allowed range.
        """
        all_vars = set(self.data.columns)
        # Check controls type
        if not isinstance(controls, list):
            raise TypeError(f"'controls' must be a list. Received types: {type(controls).__name__}.")
        if not all(isinstance(col, str) for col in controls):
            raise TypeError("All elements in 'controls' must be strings.")
        
        missing_ctrl = [var for var in controls if var not in all_vars]
        if missing_ctrl:
            raise ValueError(
                "Variable names in 'controls' must exist in the provided DataFrame 'data'.\n"
                f"The following controls were not found in the DataFrame: "
                f"{missing_ctrl}."
            )

        # Group validation
        if group is not None:
            if group not in all_vars:
                raise ValueError(f"Grouping variable '{group}' not found in your DataFrame.")
            if not isinstance(group, str):
                raise TypeError(f"'group' must be a string. Received types: {type(group).__name__}.")

        # K-fold & draws
        if (kfold < 2) or (kfold>len(self.data)-1):
            raise ValueError(f"kfold values must be between 2 and {len(self.data)-1}, current value is {kfold}.")
        if draws < 1:
            raise ValueError(f"Draws value must be 1 or above, current value is {draws}.")

        # OOS metric
        if oos_metric not in valid_oos_metrics:
            raise ValueError(f"OOS Metric must be one of {valid_oos_metrics}.")

        # n_cpu
        if n_cpu is None:
            raise ValueError(f"CPU count is currently {n_cpu}.")
        else:
            if not isinstance(n_cpu, int):
                raise TypeError("n_cpu must be an integer")
            else:
                if (n_cpu <= 0) or (n_cpu > cpu_count()):
                    raise ValueError(f"n_cpu not in a valid range: pick between 0 and {cpu_count()}.")

        if seed is not None:
            if not isinstance(seed, int):
                raise TypeError("seed must be an integer")
            if (seed<0) or (seed>2 ** 31 - 1):
                raise ValueError(f"seed must be between 0 and {seed>2 ** 31 - 1}, current value is {seed}.")
            np.random.seed(seed)

        # numeric columns check
        cols_to_check = self.y + self.x + ([group] if group else []) + controls
        _check_numeric_columns(self.data, cols_to_check)

        # 9. warn if large draws
        self._warn_if_large_draws(draws, controls, threshold)
        
        # Disallow overlap between x and controls
        overlap = set(self.x).intersection(controls)
        if overlap:
            raise ValueError(
                "Configuration conflict: the following variables appear in both 'x' and 'controls': "
                f"{sorted(overlap)}. Please ensure treatment (x) and control sets are disjoint."
            )
        # Disallow y appearing in x
        if any(col in self.y for col in self.x):
            raise ValueError(
                "Invalid configuration: dependent variable(s) in 'y' must not also appear in 'x'."
            )

        # check for empty x
        if len(self.x) == 0:
            raise ValueError("No independent variables (x) provided.")

        return n_cpu