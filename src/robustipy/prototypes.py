from abc import ABC, abstractmethod
import warnings
from multiprocessing import cpu_count
import numpy as np
import pandas as pd
from rich.progress import track

from robustipy.utils import all_subsets, space_size


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
    A base class factoring out the repeated logic in OLSRobust and LRobust:
      - Basic validation (controls, group, etc.)
      - multiple_y support
      - parallel bootstrapping loops
      - SHAP logic
    """

    def __init__(self, *, y, x, data, model_name="BaseRobust"):
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
    
    def multiple_y(self, progress: bool = False):
        """
        Computes composite y variables from the indicators in self.y.
        Use progress=True to enable progress tracking.
        """
        self.y_specs = []
        self.y_composites = []
        print("Calculating Composite Ys")
        subsets = list(all_subsets(self.y))
        iterator = subsets 
        for spec in iterator:
            if len(spec) > 0:
                subset = self.data[list(spec)]
                subset = (subset - subset.mean()) / subset.std()
                self.y_composites.append(subset.mean(axis=1))
                self.y_specs.append(spec)
        self.parameters['y_specs'] = self.y_specs
        self.parameters['y_composites'] = self.y_composites
    
    def fit(self, *, controls, group=None, draws=500, kfold=5, oos_metric='r-squared', n_cpu=None, seed=None):
        raise NotImplementedError("This method should be implemented in subclasses.")
    
    def _warn_if_large_draws(self, draws, controls, threshold=500):
        """
        Issues a warning if 'draws' × #specs × #y_composites is large.
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
        if kfold < 2:
            raise ValueError(f"kfold values must be 2 or above, current value is {kfold}.")
        if draws < 1:
            raise ValueError(f"Draws value must be 1 or above, current value is {draws}.")

        # OOS metric
        if oos_metric not in valid_oos_metrics:
            raise ValueError(f"OOS Metric must be one of {valid_oos_metrics}.")

        # n_cpu
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

        # numeric columns check
        cols_to_check = self.y + self.x + ([group] if group else []) + controls
        _check_numeric_columns(self.data, cols_to_check)

        # 9. warn if large draws
        self._warn_if_large_draws(draws, controls, threshold=500)
        
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