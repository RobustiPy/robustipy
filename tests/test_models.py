import pytest
import pandas as pd
import numpy as np
from matplotlib.figure import Figure

from robustipy.models import OLSRobust, LRobust, OLSResult, stouffer_method
from robustipy.prototypes import MissingValueWarning


# Fixture to provide a simple DataFrame for testing
@pytest.fixture
def simple_data():
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

# ----------------------------
# Tests for correct (regular) usage
# ----------------------------
def test_olsrobust_init(simple_data):
    model = OLSRobust(y=['y'], x=['x1'], data=simple_data)
    assert model.y == ['y']
    assert model.x == ['x1']
    assert model.data.equals(simple_data)

def test_lrobust_init(simple_data):
    binary_y = (simple_data['y'] > 0).astype(int)
    simple_data['binary_y'] = binary_y
    model = LRobust(y=['binary_y'], x=['x1'], data=simple_data)
    assert model.y == ['binary_y']
    assert model.x == ['x1']

def test_olsrobust_fit(simple_data):
    model = OLSRobust(y=['y'], x=['x1'], data=simple_data)
    model.fit(controls=['control1', 'control2'], kfold=2, draws=10, n_cpu=1)
    results = model.get_results()
    assert results is not None
    assert hasattr(results, 'summary_df')

def test_lrobust_fit(simple_data):
    simple_data['binary_y'] = (simple_data['y'] > 0).astype(int)
    model = LRobust(y=['binary_y'], x=['x1'], data=simple_data)
    model.fit(controls=['control1', 'control2'], kfold=2, draws=10, n_cpu=1)
    results = model.get_results()
    assert results is not None
    assert hasattr(results, 'summary_df')

def test_model_merge(simple_data):
    model1 = OLSRobust(y=['y'], x=['x1'], data=simple_data)
    model1.fit(controls=['control1'], kfold=2, draws=5, n_cpu=1)
    res1 = model1.get_results()

    model2 = OLSRobust(y=['y'], x=['x1'], data=simple_data)
    model2.fit(controls=['control2'], kfold=2, draws=5, n_cpu=1)
    res2 = model2.get_results()

    merged = res1.merge(res2, left_prefix='A', right_prefix='B')
    assert merged is not None
    assert hasattr(merged, 'summary_df')
    # The merged summary should have rows equal to the sum of the two original results.
    assert merged.summary_df.shape[0] == res1.summary_df.shape[0] + res2.summary_df.shape[0]

# ----------------------------
# Tests for invalid inputs and error messages
# ----------------------------

def test_invalid_y_type(simple_data):
    # y must be a list; providing a string should raise a TypeError.
    with pytest.raises(TypeError, match="'y' and 'x' must be lists."):
        OLSRobust(y="y", x=['x1'], data=simple_data)

def test_invalid_x_type(simple_data):
    # x must be a list; providing a string should raise a TypeError.
    with pytest.raises(TypeError, match="'y' and 'x' must be lists."):
        OLSRobust(y=['y'], x="x1", data=simple_data)

def test_invalid_data_type():
    # data must be a pandas DataFrame; providing a list should raise a TypeError.
    with pytest.raises(TypeError, match="'data' must be a pandas DataFrame."):
        OLSRobust(y=['y'], x=['x1'], data=[1, 2, 3])

def test_missing_column_error(simple_data):
    # When y or x column names are not present in the DataFrame, a ValueError should be raised.
    with pytest.raises(ValueError, match="Variable names in 'y' and 'x' must exist"):
        OLSRobust(y=['nonexistent'], x=['x1'], data=simple_data)

def test_invalid_controls_type(simple_data):
    # The controls argument in fit() must be a list.
    model = OLSRobust(y=['y'], x=['x1'], data=simple_data)
    with pytest.raises(TypeError, match="'controls' must be a list."):
        model.fit(controls="control1", kfold=2, draws=10, n_cpu=1)

def test_invalid_kfold(simple_data):
    # kfold must be 2 or above. Using 1 should raise a ValueError.
    model = OLSRobust(y=['y'], x=['x1'], data=simple_data)
    with pytest.raises(ValueError, match="kfold values must be 2 or above"):
        model.fit(controls=['control1'], kfold=1, draws=10, n_cpu=1)

def test_invalid_draws(simple_data):
    # draws must be 1 or above. Using a negative number should raise a ValueError.
    model = OLSRobust(y=['y'], x=['x1'], data=simple_data)
    with pytest.raises(ValueError, match="Draws value must be 1 or above"):
        model.fit(controls=['control1'], kfold=2, draws=-5, n_cpu=1)

def test_invalid_n_cpu(simple_data):
    # n_cpu must be an integer. Passing a string should raise a TypeError.
    model = OLSRobust(y=['y'], x=['x1'], data=simple_data)
    with pytest.raises(TypeError, match="n_cpu must be an integer"):
        model.fit(controls=['control1'], kfold=2, draws=10, n_cpu="1")

def test_invalid_group_column(simple_data):
    # If group is provided but not present in the DataFrame, a ValueError should be raised.
    with pytest.raises(ValueError, match="'group' variable must exist in the provided DataFrame"):
        OLSRobust(y=['y'], x=['x1'], data=simple_data).fit(controls=['control1'], group='nonexistent', kfold=2, draws=10, n_cpu=1)

def test_missing_values_warning(simple_data, caplog):
    # Introduce some missing values
    simple_data.loc[0, 'x1'] = np.nan
    with pytest.warns(MissingValueWarning):
        OLSRobust(y=['y'], x=['x1'], data=simple_data)

def test_non_numeric_column_error(simple_data):
    # Add a non-numeric column in the controls
    simple_data['control_non_numeric'] = ['a'] * len(simple_data)
    with pytest.raises(ValueError, match="The following columns are not numeric"):
        OLSRobust(y=['y'], x=['x1'], data=simple_data).fit(controls=['control_non_numeric'], kfold=2, draws=10, n_cpu=1)

def test_compute_bma(simple_data):
    model = OLSRobust(y=['y'], x=['x1'], data=simple_data)
    model.fit(controls=['control1'], kfold=2, draws=10, n_cpu=1)
    results = model.get_results()
    bma_df = results.compute_bma()
    expected_columns = {'control_var', 'probs', 'average_coefs'}
    assert expected_columns.issubset(bma_df.columns)

def test_save_and_load(simple_data, tmp_path):
    model = OLSRobust(y=['y'], x=['x1'], data=simple_data)
    model.fit(controls=['control1'], kfold=2, draws=10, n_cpu=1)
    results = model.get_results()
    file_path = tmp_path / "results.pkl"
    results.save(str(file_path))
    loaded = OLSResult.load(str(file_path))
    # Check that key attributes match.
    pd.testing.assert_frame_equal(loaded.summary_df, results.summary_df)

def test_stouffer_method():
    p_values = [0.05, 0.1, 0.2]
    z, combined_p = stouffer_method(p_values)
    # Simple check: combined p-value should be less than the maximum of the input p-values.
    assert combined_p < max(p_values)

def test_empty_controls(simple_data):
    model = OLSRobust(y=['y'], x=['x1'], data=simple_data)
    # Empty list for controls should be acceptable.
    model.fit(controls=[], kfold=2, draws=10, n_cpu=1)
    results = model.get_results()
    assert results is not None