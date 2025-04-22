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
            raise TypeError("'y' and 'x' must be lists.")
        if not isinstance(data, pd.DataFrame):
            raise TypeError("'data' must be a pandas DataFrame.")
        all_vars = set(data.columns)
        if not all(var in all_vars for var in y) or not all(var in all_vars for var in x):
            raise ValueError("Variable names in 'y' and 'x' must exist in the provided DataFrame 'data'.")
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
    
    def multiple_y(self, progress: bool = True):
        """
        Computes composite y variables from the indicators in self.y.
        Use progress=True to enable progress tracking.
        """
        self.y_specs = []
        self.y_composites = []
        print("Calculating Composite Ys")
        subsets = list(all_subsets(self.y))
        iterator = track(subsets, total=len(subsets)) if progress else subsets
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
        
        if len(set(controls) & set(self.x)) > 0:
            raise ValueError("Some control variables overlap with independent variables (x). Please ensure 'x' and 'controls' are disjoint sets.")

        if any(var in self.y for var in self.x):
            raise ValueError("Dependent variable(s) must not be included in the independent variables (x).")
        
        if len(self.x) == 0:
            raise ValueError("No independent variables (x) provided.")
        return n_cpu