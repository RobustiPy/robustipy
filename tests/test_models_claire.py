"""
test_models.py

This file contains unit tests for the robustipy models, including OLSRobust,
LRobust, and OLSResult. Tests cover basic functionality, input validation,
error handling, and methods such as merge, summary, save/load, and saving CSV.
"""

import os
import pytest
import pandas as pd
import numpy as np
from robustipy.models import (
    OLSRobust, LRobust, OLSResult,
    stouffer_method, MergedResult
)
from robustipy.prototypes import MissingValueWarning

# ----------------------------------------------------------------------------
#                              Test Fixtures
# ----------------------------------------------------------------------------

@pytest.fixture
def simple_data():
    """
    Returns a simple DataFrame of random data for testing OLSRobust and LRobust models.
    """
    np.random.seed(0)
    df = pd.DataFrame({
        'y': np.random.randn(100),
        'x1': np.random.randn(100),
        'x2': np.random.randn(100),
        'control1': np.random.randn(100),
        'control2': np.random.randn(100),
        'group': np.random.randint(1, 5, 100)
    })
    return df

@pytest.fixture
def binary_data(simple_data):
    """
    Returns the same data as simple_data but adds binary y-columns for testing LRobust.
    """
    df = simple_data.copy()
    df['binary_y'] = (df['y'] > 0).astype(int)
    df['some_other_y'] = (df['x1'] > 0).astype(int)
    return df

# ----------------------------------------------------------------------------
#                             Basic OLSRobust Tests
# ----------------------------------------------------------------------------

def test_olsrobust_init(simple_data):
    """
    Test that OLSRobust initialization sets attributes correctly.
    """
    model = OLSRobust(y=['y'], x=['x1'], data=simple_data)
    assert model.y == ['y']
    assert model.x == ['x1']
    assert model.data.equals(simple_data)

def test_olsrobust_fit(simple_data):
    """
    Test that OLSRobust.fit() runs and returns an OLSResult object with a summary.
    """
    model = OLSRobust(y=['y'], x=['x1'], data=simple_data)
    model.fit(controls=['control1', 'control2'], kfold=2, draws=10, n_cpu=1)
    results = model.get_results()
    assert results is not None
    assert hasattr(results, 'summary_df')
    assert isinstance(results, OLSResult)

def test_model_merge(simple_data):
    """
    Test merging of two OLSResult objects via OLSResult.merge().
    """
    model1 = OLSRobust(y=['y'], x=['x1'], data=simple_data)
    model1.fit(controls=['control1'], kfold=2, draws=5, n_cpu=1)
    res1 = model1.get_results()

    model2 = OLSRobust(y=['y'], x=['x1'], data=simple_data)
    model2.fit(controls=['control2'], kfold=2, draws=5, n_cpu=1)
    res2 = model2.get_results()

    merged = res1.merge(res2, left_prefix='A', right_prefix='B')
    assert merged is not None
    assert hasattr(merged, 'summary_df')
    assert isinstance(merged, MergedResult)
    # The merged summary should have rows equal to the sum of the two original results.
    assert merged.summary_df.shape[0] == res1.summary_df.shape[0] + res2.summary_df.shape[0]

def test_invalid_seed(simple_data):
    """
    Test that passing a non-integer seed to OLSRobust.fit() raises a TypeError.
    """
    model = OLSRobust(y=['y'], x=['x1'], data=simple_data)
    with pytest.raises(TypeError):
        model.fit(controls=['control1'], kfold=2, draws=10, n_cpu=1, seed="not_an_integer")

# ----------------------------------------------------------------------------
#                          Basic LRobust (Logistic) Tests
# ----------------------------------------------------------------------------

def test_lrobust_init(binary_data):
    """
    Test that LRobust initialization sets attributes correctly.
    """
    model = LRobust(y=['binary_y'], x=['x1'], data=binary_data)
    assert model.y == ['binary_y']
    assert model.x == ['x1']
    assert model.data.equals(binary_data)

def test_lrobust_fit(binary_data):
    """
    Test that LRobust.fit() runs and returns results with a summary.
    """
    model = LRobust(y=['binary_y'], x=['x1'], data=binary_data)
    model.fit(controls=['control1', 'control2'], kfold=2, draws=10, n_cpu=1)
    results = model.get_results()
    assert results is not None
    assert hasattr(results, 'summary_df')

def test_invalid_seed_lrobust(binary_data):
    """
    Test that passing a non-integer seed to LRobust.fit() raises a TypeError.
    """
    model = LRobust(y=['binary_y'], x=['x1'], data=binary_data)
    with pytest.raises(TypeError):
        model.fit(controls=['control1'], kfold=2, draws=10, n_cpu=1, seed=3.14)

# ----------------------------------------------------------------------------
#                       Extended Error Handling Tests
# ----------------------------------------------------------------------------

def test_invalid_y_type(simple_data):
    """
    Test that passing a non-list for y raises a TypeError.
    """
    with pytest.raises(TypeError):
        OLSRobust(y="y", x=['x1'], data=simple_data)

def test_invalid_x_type(simple_data):
    """
    Test that passing a non-list for x raises a TypeError.
    """
    with pytest.raises(TypeError):
        OLSRobust(y=['y'], x="x1", data=simple_data)

def test_invalid_data_type():
    """
    Test that passing a non-DataFrame for data raises a TypeError.
    """
    with pytest.raises(TypeError):
        OLSRobust(y=['y'], x=['x1'], data=[1, 2, 3])

def test_missing_column_error(simple_data):
    """
    Test that if y or x columns are missing from the DataFrame, a ValueError is raised.
    """
    with pytest.raises(ValueError):
        OLSRobust(y=['nonexistent'], x=['x1'], data=simple_data)

def test_invalid_controls_type(simple_data):
    """
    Test that passing a non-list for controls in fit() raises a TypeError.
    """
    model = OLSRobust(y=['y'], x=['x1'], data=simple_data)
    with pytest.raises(TypeError):
        model.fit(controls="control1", kfold=2, draws=10, n_cpu=1)

def test_invalid_kfold(simple_data):
    """
    Test that a kfold value less than 2 raises a ValueError.
    """
    model = OLSRobust(y=['y'], x=['x1'], data=simple_data)
    with pytest.raises(ValueError):
        model.fit(controls=['control1'], kfold=1, draws=10, n_cpu=1)

def test_invalid_draws(simple_data):
    """
    Test that a draws value less than 1 raises a ValueError.
    """
    model = OLSRobust(y=['y'], x=['x1'], data=simple_data)
    with pytest.raises(ValueError):
        model.fit(controls=['control1'], kfold=2, draws=-5, n_cpu=1)

def test_invalid_n_cpu(simple_data):
    """
    Test that passing a non-integer for n_cpu raises a TypeError.
    """
    model = OLSRobust(y=['y'], x=['x1'], data=simple_data)
    with pytest.raises(TypeError):
        model.fit(controls=['control1'], kfold=2, draws=10, n_cpu="1")

def test_invalid_group_column(simple_data):
    """
    Test that if a group string is passed but not present in the DataFrame, a ValueError is raised.
    """
    model = OLSRobust(y=['y'], x=['x1'], data=simple_data)
    with pytest.raises(ValueError):
        model.fit(controls=['control1'], group='nonexistent', kfold=2, draws=10, n_cpu=1)

def test_missing_values_warning(simple_data):
    """
    Test that if the DataFrame contains missing values, a MissingValueWarning is issued.
    """
    simple_data.loc[0, 'x1'] = np.nan
    with pytest.warns(MissingValueWarning):
        OLSRobust(y=['y'], x=['x1'], data=simple_data)

def test_non_numeric_column_error(simple_data):
    """
    Test that if controls include non-numeric columns, a ValueError is raised.
    """
    simple_data['control_non_numeric'] = ['a'] * len(simple_data)
    model = OLSRobust(y=['y'], x=['x1'], data=simple_data)
    with pytest.raises(ValueError):
        model.fit(controls=['control_non_numeric'], kfold=2, draws=10, n_cpu=1)

# ----------------------------------------------------------------------------
#                 Additional LRobust Error Cases
# ----------------------------------------------------------------------------

def test_lrobust_multiple_y_not_implemented(binary_data):
    """
    Test that attempting multiple y for logistic regression raises NotImplementedError.
    """
    with pytest.raises(NotImplementedError):
        LRobust(y=['binary_y', 'some_other_y'], x=['x1'], data=binary_data).fit(controls=['control1'])

def test_lrobust_invalid_oos_metric(binary_data):
    """
    Test that an invalid oos_metric for LRobust.fit() raises a ValueError.
    """
    model = LRobust(y=['binary_y'], x=['x1'], data=binary_data)
    with pytest.raises(ValueError):
        model.fit(controls=['control1'], kfold=2, draws=10, n_cpu=1, oos_metric='bad_metric')

# ----------------------------------------------------------------------------
#                Tests for Additional Methods
# ----------------------------------------------------------------------------

def test_compute_bma(simple_data):
    """
    Test that compute_bma() returns a DataFrame with expected columns.
    """
    model = OLSRobust(y=['y'], x=['x1'], data=simple_data)
    model.fit(controls=['control1'], kfold=2, draws=10, n_cpu=1)
    results = model.get_results()
    bma_df = results.compute_bma()
    expected_columns = {'control_var', 'probs', 'average_coefs'}
    assert expected_columns.issubset(bma_df.columns)

def test_stouffer_method():
    """
    Test that stouffer_method() returns a combined p-value less than the maximum input p-value.
    """
    p_values = [0.05, 0.1, 0.2]
    z, combined_p = stouffer_method(p_values)
    assert combined_p < max(p_values)

# ----------------------------------------------------------------------------
#                 Tests for .summary() and .get_results()
# ----------------------------------------------------------------------------

def test_summary_runs(simple_data, capsys):
    """
    Test that .summary() runs without error and prints model information.
    """
    model = OLSRobust(y=['y'], x=['x1'], data=simple_data)
    model.fit(controls=['control1'], kfold=2, draws=10, n_cpu=1)
    results = model.get_results()
    results.summary()
    captured = capsys.readouterr()
    assert "Model: OLS Robust" in captured.out

def test_summary_lrobust_runs(binary_data, capsys):
    """
    Test that .summary() runs for logistic regression and prints model information.
    """
    model = LRobust(y=['binary_y'], x=['x1'], data=binary_data)
    model.fit(controls=['control1'], kfold=2, draws=10, n_cpu=1)
    results = model.get_results()
    results.summary()
    captured = capsys.readouterr()
    assert "Model: Logistic Regression Robust" in captured.out

def test_get_results_return_object(simple_data):
    """
    Test that get_results() returns an OLSResult instance after fit().
    """
    model = OLSRobust(y=['y'], x=['x1'], data=simple_data)
    model.fit(controls=['control1'], kfold=2, draws=5, n_cpu=1)
    result = model.get_results()
    assert isinstance(result, OLSResult)

# ----------------------------------------------------------------------------
#          Tests for .save(), .load(), and .save_to_csv()
# ----------------------------------------------------------------------------

def test_save_and_load(simple_data, tmp_path):
    """
    Test that saving and loading the results using pickle preserves key attributes.
    """
    model = OLSRobust(y=['y'], x=['x1'], data=simple_data)
    model.fit(controls=['control1'], kfold=2, draws=10, n_cpu=1)
    results = model.get_results()
    file_path = tmp_path / "results.pkl"
    results.save(str(file_path))
    loaded = OLSResult.load(str(file_path))
    pd.testing.assert_frame_equal(loaded.summary_df, results.summary_df)

def test_save_to_csv(simple_data, tmp_path):
    """
    Test that saving summary_df to a CSV file works correctly.
    """
    model = OLSRobust(y=['y'], x=['x1'], data=simple_data)
    model.fit(controls=['control1'], kfold=2, draws=10, n_cpu=1)
    results = model.get_results()
    csv_path = tmp_path / "summary.csv"
    results.save_to_csv(str(csv_path))
    assert os.path.exists(csv_path)
    df_loaded = pd.read_csv(str(csv_path))
    assert not df_loaded.empty

# ----------------------------------------------------------------------------
#          Test for empty controls (valid case)
# ----------------------------------------------------------------------------

def test_empty_controls(simple_data):
    """
    Test that fit() works correctly when controls is an empty list.
    """
    model = OLSRobust(y=['y'], x=['x1'], data=simple_data)
    model.fit(controls=[], kfold=2, draws=10, n_cpu=1)
    results = model.get_results()
    assert results is not None
    assert hasattr(results, 'summary_df')

# ----------------------------------------------------------------------------
#         Thorough Input Validation Tests for .fit() Method
# ----------------------------------------------------------------------------

def test_fit_invalid_kfold_type(simple_data):
    """
    Ensure that passing a non-integer for kfold raises a TypeError.
    """
    model = OLSRobust(y=['y'], x=['x1'], data=simple_data)
    with pytest.raises(TypeError):
        model.fit(controls=['control1'], kfold='two', draws=10, n_cpu=1)

def test_fit_invalid_draws_type(simple_data):
    """
    Ensure that passing a non-integer for draws raises a TypeError.
    """
    model = OLSRobust(y=['y'], x=['x1'], data=simple_data)
    with pytest.raises(TypeError):
        model.fit(controls=['control1'], kfold=2, draws='ten', n_cpu=1)

def test_fit_invalid_seed_type(simple_data):
    """
    Ensure that passing a non-integer for seed raises a TypeError.
    """
    model = OLSRobust(y=['y'], x=['x1'], data=simple_data)
    with pytest.raises(TypeError):
        model.fit(controls=['control1'], kfold=2, draws=10, n_cpu=1, seed=3.14)

def test_fit_invalid_group_column(simple_data):
    """
    Ensure that if group is a string but not a column in the data, a ValueError is raised.
    """
    model = OLSRobust(y=['y'], x=['x1'], data=simple_data)
    with pytest.raises(ValueError):
        model.fit(controls=['control1'], group='nonexistent', kfold=2, draws=10, n_cpu=1)



# ----------------------------------------------------------------------------
#           Failed test
# ----------------------------------------------------------------------------

# Failed as it raise: ValueError: 'group' variable must exist in the provided DataFrame 'data'.
# should raise typeerror instead of valueerror?
# revised the models.py from the original code 
        #  if group is not None:
        #     if not isinstance(group, str) or not group in all_vars:
        #         raise ValueError("'group' variable must exist in the provided DataFrame 'data'.")

# to:
        # if group is not None:
        #     if group not in all_vars:
        #         raise ValueError("'group' variable must exist in the provided DataFrame 'data'.")
        #     if not isinstance(group, str):
        #         raise TypeError("'group' must be a string.")
def test_fit_group_column_name_is_non_string_and_group_not_string_raises_typeerror():
    """
    Test case where the DataFrame contains a group column whose name is not a string (e.g., an int),
    and the 'group' parameter is passed as an int. This should raise a TypeError because
    the 'group' argument must be a string, regardless of the column names present in the DataFrame.
    """
    df = pd.DataFrame({
        'y': np.random.randn(100),
        'x1': np.random.randn(100),
        'control1': np.random.randn(100),
        123: np.random.randint(1, 5, 100),  # Column named with an int
    })
    model = OLSRobust(y=['y'], x=['x1'], data=df)

    with pytest.raises(TypeError, match="'group' must be a string"):
        model.fit(controls=['control1'], group=123, kfold=2, draws=10, n_cpu=1)

