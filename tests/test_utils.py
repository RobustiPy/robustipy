import numpy as np
import pandas as pd
import pytest
import statsmodels.api as sm

from robustipy.utils import (
    space_size,
    all_subsets,
    simple_ols,
    logistic_regression_sm,
    group_demean,
    get_colormap_colors,
    get_colors,
    decorator_timer
)


@pytest.fixture
def dummy_data():
    df = pd.DataFrame({
        'y': np.random.randn(100),
        'x1': np.random.randn(100),
        'x2': np.random.randn(100),
        'group': np.repeat(['A', 'B', 'C', 'D'], 25)
    })
    return df


def test_space_size():
    assert space_size([1, 2, 3]) == 8
    assert space_size([]) == 1


def test_all_subsets():
    items = ['a', 'b']
    subsets = list(all_subsets(items))
    expected = [(), ('a',), ('b',), ('a', 'b')]
    assert subsets == expected


def test_simple_ols_against_statsmodels(dummy_data):
    y = dummy_data[['y']]
    x = dummy_data[['x1', 'x2']].copy()

    result_custom = simple_ols(y, x.copy())

    # Prepare data for statsmodels by adding constant.
    x['const'] = 1
    sm_model = sm.OLS(y, x).fit()

    # Compare coefficient estimates:
    # simple_ols returns b as an array of shape (k,1); flatten to (k,)
    np.testing.assert_allclose(
        np.array(result_custom['b']).ravel(),
        sm_model.params.values,
        rtol=1e-6
    )

    # Compare p-values: extract only the diagonal of the (k,k) array
    p_custom = np.array(result_custom['p'])
    if p_custom.ndim == 2 and p_custom.shape[0] == p_custom.shape[1]:
        p_to_check = np.diag(p_custom)
    else:
        p_to_check = p_custom.ravel()
    np.testing.assert_allclose(
        p_to_check,
        sm_model.pvalues.values,
        rtol=1e-6
    )

    # Information criteria:
    # nobs = number of observations, ncoef = number of coefficients (after adding 'const')
    nobs = y.shape[0]              # e.g., 100
    ncoef = x.shape[1]             # e.g., 3 (x1, x2, const)
    offset = 2 * nobs - 2 * ncoef  # For example: 2*100 - 2*3 = 194

    # AIC from simple_ols must have the offset subtracted:
    aic_custom  = result_custom['aic'][0][0] - offset
    np.testing.assert_allclose(aic_custom, sm_model.aic, rtol=1e-6)

    # BIC can be compared directly:
    bic_custom  = result_custom['bic'][0][0]
    np.testing.assert_allclose(bic_custom, sm_model.bic, rtol=1e-6)



def test_logistic_regression_sm_returns_expected_keys():
    y = np.random.binomial(1, 0.5, 100)
    x = pd.DataFrame({'x1': np.random.randn(100), 'x2': np.random.randn(100)})
    result = logistic_regression_sm(y, x)
    assert set(result.keys()) == {'b', 'p', 'r2', 'll', 'aic', 'bic', 'hqic'}
    assert isinstance(result['r2'], float)

def test_empty_input_raises():
    y = pd.DataFrame([])
    x = pd.DataFrame([])
    with pytest.raises(ValueError):
        simple_ols(y, x)

def test_output_keys(dummy_data):
    y = dummy_data[['y']]
    x = dummy_data[['x1', 'x2']]
    result = simple_ols(y, x)
    expected_keys = {'b', 'p', 'r2', 'll', 'aic', 'bic', 'hqic'}
    assert set(result.keys()) == expected_keys

def test_output_shapes(dummy_data):
    y = dummy_data[['y']]
    x = dummy_data[['x1', 'x2']]
    result = simple_ols(y, x)
    assert result['b'].shape == (3, 1)
    p_shape = result['p'].shape
    assert p_shape[0] == 3
    assert p_shape[1] in [1, 3], f"Unexpected p-value shape: {p_shape}"


def test_handles_singular_matrix():
    # x1 and x2 are perfectly collinear
    y = pd.DataFrame(np.random.randn(100, 1), columns=['y'])
    x = pd.DataFrame(np.ones((100, 1)), columns=['x1'])
    x['x2'] = x['x1']  # duplicate column
    result = simple_ols(y, x)
    assert 'b' in result  # should still return output, using pinv fallback


def test_group_demean_with_group(dummy_data):
    result = group_demean(dummy_data[['x1', 'x2', 'group']], group='group')
    group_means = result.groupby(dummy_data['group'])[['x1', 'x2']].mean()
    assert group_means.abs().max().max() < 1e-10


def test_get_colormap_colors_brightness():
    colors = get_colormap_colors('viridis', num_colors=5, brightness_threshold=0.7)
    assert isinstance(colors, list)
    assert len(colors) == 5
    for color in colors:
        brightness = sum(color[:3]) / 3
        assert brightness <= 0.7


def test_get_colors_valid_input():
    specs = [['a'], ['a', 'b'], ['b']]
    colors = get_colors(specs, color_set_name='Set1')
    assert len(colors) == len(specs)


def test_get_colors_invalid_input_raises():
    with pytest.raises(ValueError):
        get_colors(['a', 'b'])  # should be list of lists


def test_decorator_timer_runtime():
    @decorator_timer
    def dummy_func(x):
        return x**2

    result, elapsed = dummy_func(3)
    assert result == 9
    assert isinstance(elapsed, float)
    assert elapsed >= 0



df = pd.DataFrame({
    'x1': [1, 2, 3],
    'x2': [4, 5, 6]
})

df - df.mean()
df - np.mean(df)