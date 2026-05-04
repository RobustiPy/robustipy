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
    stouffer_method, MergedResult,
    _cluster_bootstrap_by_rows,
    _make_group_bootstrap_lookup,
    _prepare_non_group_ols_bootstrap_arrays,
    _prepare_ols_bootstrap_data,
    _strap_non_group_OLS_arrays,
    _run_parallel_seed_batches
)
from robustipy.prototypes import MissingValueWarning, BaseRobust

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

def test_ols_loglikelihood_scale_invariant(simple_data):
    """
    OLS likelihood metric used for reporting should be invariant to scaling y.
    Raw log-likelihood should still shift under scaling.
    """
    data_1 = simple_data.copy()
    data_2 = simple_data.copy()
    data_2['y'] = 10.0 * data_2['y']

    model_1 = OLSRobust(y=['y'], x=['x1'], data=data_1)
    model_1.fit(controls=['control1', 'control2'], kfold=2, draws=5, n_cpu=1, seed=123)
    res_1 = model_1.get_results()

    model_2 = OLSRobust(y=['y'], x=['x1'], data=data_2)
    model_2.fit(controls=['control1', 'control2'], kfold=2, draws=5, n_cpu=1, seed=123)
    res_2 = model_2.get_results()

    for col in ['ll', 'll_raw', 'll_null', 'll_gain', 'll_gain_per_obs', 'nobs']:
        assert col in res_1.summary_df.columns
        assert col in res_2.summary_df.columns

    assert np.allclose(
        res_1.summary_df['ll_gain_per_obs'].to_numpy(dtype=float),
        res_2.summary_df['ll_gain_per_obs'].to_numpy(dtype=float),
        rtol=1e-10,
        atol=1e-10
    )
    assert np.allclose(
        res_1.summary_df['ll'].to_numpy(dtype=float),
        res_2.summary_df['ll'].to_numpy(dtype=float),
        rtol=1e-10,
        atol=1e-10
    )
    assert not np.allclose(
        res_1.summary_df['ll_raw'].to_numpy(dtype=float),
        res_2.summary_df['ll_raw'].to_numpy(dtype=float),
        rtol=1e-10,
        atol=1e-10
    )

def test_cluster_bootstrap_samples_whole_groups_by_position():
    """
    Grouped bootstrap should match the previous concat-based implementation
    exactly for a fixed seed while avoiding concat in production code.
    """
    data = pd.DataFrame({
        'group': ['a', 'a', 'b', 'c', 'c', 'c'],
        'row_id': [0, 1, 2, 3, 4, 5],
        'value': [10, 11, 12, 13, 14, 15],
    })

    sample = _cluster_bootstrap_by_rows(
        temp_data=data,
        group='group',
        seed=123,
        target_rows=10,
    )
    unique_groups = data['group'].unique()
    rng = np.random.default_rng(123)
    group_lookup = {
        group_name: group_df
        for group_name, group_df in data.groupby('group', sort=False, observed=True)
    }
    sampled_frames = []
    n_rows = 0
    while n_rows < 10:
        group_name = rng.choice(unique_groups)
        group_df = group_lookup[group_name]
        sampled_frames.append(group_df)
        n_rows += len(group_df)
    expected = pd.concat(sampled_frames, ignore_index=True)

    pd.testing.assert_frame_equal(sample, expected)
    assert len(sample) >= 10
    assert list(sample.columns) == list(data.columns)
    assert sample.index.tolist() == list(range(len(sample)))

    original_rows = {
        group_name: tuple(group_df['row_id'])
        for group_name, group_df in data.groupby('group', sort=False)
    }
    for group_name, group_df in sample.groupby('group', sort=False):
        rows = tuple(group_df['row_id'])
        original = original_rows[group_name]
        assert len(rows) % len(original) == 0
        for start in range(0, len(rows), len(original)):
            assert rows[start:start + len(original)] == original

def test_cluster_bootstrap_precomputed_lookup_matches_old_concat():
    """
    Precomputing group row positions should not change grouped bootstrap samples.
    """
    data = pd.DataFrame({
        'group': ['a', 'a', 'b', 'c', 'c', 'c'],
        'row_id': [0, 1, 2, 3, 4, 5],
        'value': [10, 11, 12, 13, 14, 15],
    })

    group_lookup = _make_group_bootstrap_lookup(data, 'group')
    sample = _cluster_bootstrap_by_rows(
        temp_data=data,
        group='group',
        seed=321,
        target_rows=10,
        group_lookup=group_lookup,
    )

    unique_groups = data['group'].unique()
    rng = np.random.default_rng(321)
    old_lookup = {
        group_name: group_df
        for group_name, group_df in data.groupby('group', sort=False, observed=True)
    }
    sampled_frames = []
    n_rows = 0
    while n_rows < 10:
        group_name = rng.choice(unique_groups)
        group_df = old_lookup[group_name]
        sampled_frames.append(group_df)
        n_rows += len(group_df)
    expected = pd.concat(sampled_frames, ignore_index=True)

    pd.testing.assert_frame_equal(sample, expected)

def test_ols_bootstrap_optimized_path_matches_legacy(simple_data):
    """
    Prebuilt OLS bootstrap data and precomputed group lookup should not change
    one-draw bootstrap outputs.
    """
    comb = simple_data[['y', 'x1', 'group', 'control1']].reset_index(drop=True).copy()
    y_star = pd.DataFrame(
        comb.iloc[:, 0].to_numpy() - (0.25 * comb.iloc[:, 1].to_numpy())
    )
    model = OLSRobust(y=['y'], x=['x1'], data=simple_data)

    legacy = model._strap_OLS(
        comb,
        group='group',
        sample_size=len(comb),
        seed=123,
        y_star=y_star,
    )

    bootstrap_data = _prepare_ols_bootstrap_data(comb, y_star)
    group_lookup = _make_group_bootstrap_lookup(bootstrap_data, 'group')
    optimized = model._strap_OLS(
        bootstrap_data,
        group='group',
        sample_size=len(comb),
        seed=123,
        y_star=None,
        group_bootstrap_lookup=group_lookup,
        min_rows_after_filter=5,
    )

    np.testing.assert_allclose(optimized, legacy, equal_nan=True)

def test_ols_non_group_bootstrap_optimized_path_matches_legacy(simple_data):
    """
    The non-grouped OLS bootstrap path should produce the same one-draw output
    after replacing pandas sample and prebuilding y_star.
    """
    comb = simple_data[['y', 'x1', 'control1']].reset_index(drop=True).copy()
    y_star = pd.DataFrame(
        comb.iloc[:, 0].to_numpy() - (0.25 * comb.iloc[:, 1].to_numpy())
    )
    model = OLSRobust(y=['y'], x=['x1'], data=simple_data)

    legacy = model._strap_OLS(
        comb,
        group=None,
        sample_size=len(comb),
        seed=123,
        y_star=y_star,
    )

    bootstrap_data = _prepare_ols_bootstrap_data(comb, y_star)
    optimized = model._strap_OLS(
        bootstrap_data,
        group=None,
        sample_size=len(comb),
        seed=123,
        y_star=None,
    )

    np.testing.assert_allclose(optimized, legacy, equal_nan=True)

def test_ols_non_group_array_bootstrap_matches_pandas_path(simple_data):
    """
    Precomputed-array non-grouped OLS bootstrap should match the pandas sample
    path exactly for fixed seeds.
    """
    comb = simple_data[['y', 'x1', 'control1']].reset_index(drop=True).copy()
    y_star = pd.DataFrame(
        comb.iloc[:, 0].to_numpy() - (0.25 * comb.iloc[:, 1].to_numpy())
    )
    bootstrap_data = _prepare_ols_bootstrap_data(comb, y_star)
    bootstrap_arrays = _prepare_non_group_ols_bootstrap_arrays(bootstrap_data)
    model = OLSRobust(y=['y'], x=['x1'], data=simple_data)

    for seed in [0, 1, 123, 192735]:
        pandas_path = model._strap_OLS(
            bootstrap_data,
            group=None,
            sample_size=len(comb),
            seed=seed,
            y_star=None,
        )
        array_path = _strap_non_group_OLS_arrays(
            *bootstrap_arrays,
            sample_size=len(comb),
            seed=seed,
        )

        np.testing.assert_allclose(array_path, pandas_path, rtol=0, atol=0, equal_nan=True)

def test_logistic_non_group_bootstrap_matches_pandas_sample(binary_data):
    """
    Logistic non-grouped bootstrap should match the previous pandas sample
    row draw for a fixed seed.
    """
    from robustipy.utils import logistic_regression_sm

    comb = binary_data[['binary_y', 'x1', 'control1']].reset_index(drop=True).copy()
    model = LRobust(y=['binary_y'], x=['x1'], data=binary_data)

    actual = model._strap_regression(
        comb,
        group=None,
        sample_size=len(comb),
        seed=123,
    )

    samp_df = comb.sample(n=len(comb), replace=True, random_state=123)
    y = samp_df.iloc[:, [0]]
    x = samp_df.drop(samp_df.columns[0], axis=1)
    expected_output = logistic_regression_sm(y, x)
    expected = (
        expected_output['b'][0][0],
        expected_output['p'][0][0],
        expected_output['r2'],
    )

    np.testing.assert_allclose(actual, expected, equal_nan=True)

def test_logistic_group_bootstrap_precomputed_lookup_matches_default(binary_data):
    """
    Logistic grouped bootstrap should not change when the group lookup is
    precomputed once per spec.
    """
    comb = binary_data[['binary_y', 'x1', 'group', 'control1']].reset_index(drop=True).copy()
    model = LRobust(y=['binary_y'], x=['x1'], data=binary_data)

    default = model._strap_regression(
        comb,
        group='group',
        sample_size=len(comb),
        seed=123,
    )

    group_lookup = _make_group_bootstrap_lookup(comb, 'group')
    optimized = model._strap_regression(
        comb,
        group='group',
        sample_size=len(comb),
        seed=123,
        group_bootstrap_lookup=group_lookup,
        min_rows_after_filter=5,
    )

    np.testing.assert_allclose(optimized, default, equal_nan=True)

def test_parallel_seed_runner_matches_old_batch_order():
    """
    Streaming joblib dispatch should preserve the old batched output order.
    """
    seeds = np.arange(17, dtype=np.int64) + 10

    def run_one_seed(seed):
        return (seed, seed * seed)

    def old_batched_runner():
        outputs = []
        batch_size = max(8, 2)
        for start in range(0, len(seeds), batch_size):
            seed_batch = seeds[start:start + batch_size]
            batch_output = [
                run_one_seed(int(seed))
                for seed in seed_batch
            ]
            outputs.extend(batch_output)
        return outputs

    expected = old_batched_runner()
    actual = _run_parallel_seed_batches(
        seeds=seeds,
        n_cpu=2,
        run_one_seed=run_one_seed,
    )
    batched = _run_parallel_seed_batches(
        seeds=seeds,
        n_cpu=2,
        run_one_seed=run_one_seed,
        dispatch_mode="batched",
    )
    chunked = _run_parallel_seed_batches(
        seeds=seeds,
        n_cpu=2,
        run_one_seed=run_one_seed,
        dispatch_mode="chunked",
        task_batch_size=5,
    )
    serial = _run_parallel_seed_batches(
        seeds=seeds,
        n_cpu=1,
        run_one_seed=run_one_seed,
    )

    assert actual == expected
    assert batched == expected
    assert chunked == expected
    assert serial == expected

def test_parallel_seed_runner_throttles_progress_updates():
    """
    Progress updates should be batched so notebooks do not receive one IOPub
    message per bootstrap draw.
    """
    seeds = np.arange(17, dtype=np.int64) + 10

    class DummyBar:
        def __init__(self):
            self.updates = []
            self.refresh_count = 0

        def update(self, n):
            self.updates.append(n)

        def refresh(self):
            self.refresh_count += 1

    bar = DummyBar()
    out = _run_parallel_seed_batches(
        seeds=seeds,
        n_cpu=2,
        run_one_seed=lambda seed: seed,
        draws_bar=bar,
    )

    assert out == [int(seed) for seed in seeds]
    assert sum(bar.updates) == len(seeds)
    assert bar.updates == [8, 8, 1]
    assert bar.refresh_count == 0

    batched_bar = DummyBar()
    batched_out = _run_parallel_seed_batches(
        seeds=seeds,
        n_cpu=2,
        run_one_seed=lambda seed: seed,
        draws_bar=batched_bar,
        dispatch_mode="batched",
    )

    assert batched_out == [int(seed) for seed in seeds]
    assert sum(batched_bar.updates) == len(seeds)
    assert batched_bar.updates == [8, 8, 1]
    assert batched_bar.refresh_count == 0

    chunked_bar = DummyBar()
    chunked_out = _run_parallel_seed_batches(
        seeds=seeds,
        n_cpu=2,
        run_one_seed=lambda seed: seed,
        draws_bar=chunked_bar,
        dispatch_mode="chunked",
        task_batch_size=5,
    )

    assert chunked_out == [int(seed) for seed in seeds]
    assert sum(chunked_bar.updates) == len(seeds)
    assert chunked_bar.updates == [5, 5, 5, 2]
    assert chunked_bar.refresh_count == 0

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


def test_stouffer_method_null_calibrated():
    """
    When null draws are supplied, stouffer_method should use null-calibrated
    Monte Carlo p-values (not asymptotic-only p-values).
    """
    p_obs = np.array([0.01, 0.02, 0.03], dtype=float)
    b_obs = np.array([1.0, 1.0, 1.0], dtype=float)

    # n_specs x n_draws
    p_null = np.array([
        [0.4, 0.5, 0.6, 0.7, 0.8],
        [0.3, 0.6, 0.5, 0.7, 0.2],
        [0.2, 0.4, 0.6, 0.8, 0.9],
    ], dtype=float)
    b_null = np.array([
        [1.0, -1.0, 1.0, -1.0, 1.0],
        [1.0, 1.0, -1.0, -1.0, 1.0],
        [1.0, -1.0, 1.0, 1.0, -1.0],
    ], dtype=float)

    z_null_cal, p_null_cal = stouffer_method(
        p_obs,
        two_sided=True,
        betas=b_obs,
        p_values_ystar=p_null,
        betas_ystar=b_null,
        warn=False,
    )
    z_asym, p_asym = stouffer_method(
        p_obs,
        two_sided=True,
        betas=b_obs,
        warn=False,
    )

    # Monte Carlo p must be on the (B+1) grid with B=5 -> step size 1/6.
    assert p_null_cal == pytest.approx(1.0 / 6.0)
    # Null-calibrated and asymptotic p-values should differ on this setup.
    assert abs(p_null_cal - p_asym) > 1e-3
    # Both runs should produce finite Z.
    assert np.isfinite(z_null_cal)
    assert np.isfinite(z_asym)

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




def test_data_is_numpy_array_not_dataframe():
    """
    OLSRobust should raise an error when data is a NumPy array, even if it has shape like a DataFrame.
    """
    data = np.random.rand(100, 5)
    with pytest.raises(TypeError, match="must be a pandas DataFrame"):
        OLSRobust(y=['y'], x=['x1'], data=data)

def test_data_without_column_names_raises():
    """
    Test that passing a DataFrame without column names raises a ValueError during model initialization.
    """
    data = pd.DataFrame(np.random.rand(100, 3))
    data.columns = [None, None, None]  # This means the DataFrame does not have proper column names.
    with pytest.raises(ValueError):
        OLSRobust(y=['y'], x=['x1'], data=data)

def test_data_with_non_string_column_names_raises():
    """
    Test that passing a DataFrame with non-string column names raises a ValueError during model initialization.
    """
    # Build a DataFrame whose columns are numbers.
    df = pd.DataFrame({
        0: np.random.rand(100),
        1: np.random.rand(100),
        2: np.random.rand(100)
    })
    # The model will check if the variables in y and x exist in df.columns.
    # Since the error message is the same, we adjust our match string accordingly.
    with pytest.raises(ValueError):
        OLSRobust(y=['0'], x=['1'], data=df)

def test_controls_passed_as_dataframe_raises(simple_data):
    """
    Even if controls is a valid DataFrame, the model requires a list of column names.
    """
    controls_df = simple_data[['control1', 'control2']]
    model = OLSRobust(y=['y'], x=['x1'], data=simple_data)
    with pytest.raises(TypeError):
        model.fit(controls=controls_df, kfold=2, draws=10, n_cpu=1)        

def test_controls_passed_as_numpy_array_raises(simple_data):
    """
    Even if the array is shape (n,2), controls must be a list of strings,
    not a NumPy array.
    """
    controls_np = simple_data[['control1','control2']].values
    model = OLSRobust(y=['y'], x=['x1'], data=simple_data)
    with pytest.raises(TypeError, match="'controls' must be a list"):
        model.fit(controls=controls_np, kfold=2, draws=10, n_cpu=1)


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
