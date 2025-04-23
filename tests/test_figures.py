# test_figures.py

import pytest
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from unittest.mock import patch

# Force matplotlib to not use any X-server; must happen before pyplot import
matplotlib.use("Agg")

from robustipy.figures import (
    axis_formatter,
    plot_hexbin_r2,
    plot_hexbin_log,
    shap_violin,
    plot_curve,
    plot_ic,
    plot_bdist,
    plot_kfolds,
    plot_bma,
    plot_results
)
# -----------------------------------------------------------------------------
#                            FIXTURES / UTILITIES
# -----------------------------------------------------------------------------

@pytest.fixture
def mock_results_object():
    """
    Create a mock results_object that has only the minimal attributes/functions
    needed for the figures.py plotting functions to run successfully.
    """
    class MockResults:
        def __init__(self):
            # Minimal attributes used by various plots:
            # 1) .estimates and .r2_values for plot_hexbin_r2
            self.estimates = pd.DataFrame(np.random.randn(5, 10))  # shape => (num_specs, draws)
            self.r2_values = pd.DataFrame(np.random.rand(5, 10))   # shape => (num_specs, draws)
            
            # 2) .all_b and .summary_df for plot_hexbin_log
            self.all_b = [
                [[np.random.randn(1)] for _ in range(1)] for _ in range(5)
            ]
            
            # summary_df needed to have 'll', 'spec_name', etc.
            self.summary_df = pd.DataFrame({
                'll': np.random.randn(5)*10,
                'spec_name': [frozenset([f'spec_{i}']) for i in range(5)],
                'r2': np.random.rand(5),
                'av_k_metric': np.random.randn(5)*10,
                'aic': np.random.randn(5)*100,
                'bic': np.random.randn(5)*100,
                'hqic': np.random.randn(5)*100,
                'median': np.random.randn(5),
                'min': np.random.randn(5),
                'max': np.random.randn(5),
            })
            
            # 3) .inference used for the text inside plot_curve
            self.inference = {'median': 0.123, 'median_p': 0.456, 'Stouffers': (0.0, 1.0)}
            
            # 4) draws, kfold => used in plot_curve text
            self.draws = 50
            self.kfold = 5
            
            # 5) .specs_names => used in multiple plots
            self.specs_names = pd.Series(self.summary_df['spec_name'])
            
            # 6) .name_av_k_metric => used in plot_kfolds
            self.name_av_k_metric = "rmse"
            
            # 7) .shap_return => for shap_violin and plot_results
            self.x_name = "X"
            shap_vals = np.random.randn(20, 4)  # e.g., shape (20, 4)
            self.shap_return = [
                shap_vals, 
                pd.DataFrame(
                    shap_vals, 
                    columns=[self.x_name, "featA", "featB", "featC"]
                )
            ]
            
            # 8) Minimal .controls, used by compute_bma or other references
            self.controls = ["featA", "featB", "featC"]
        
        def compute_bma(self):
            """
            Minimal stub for plot_bma usage:
            Returns a DataFrame with 'control_var', 'probs', 'average_coefs' columns.
            """
            return pd.DataFrame({
                'control_var': self.controls,
                'probs': [0.5, 0.3, 0.2],
                'average_coefs': [1.0, 2.0, 3.0]
            })
    
    return MockResults()


@pytest.fixture
def fig_ax():
    """Provide a fresh figure, ax tuple for tests that need them."""
    fig, ax = plt.subplots()
    yield fig, ax
    plt.close(fig)


# -----------------------------------------------------------------------------
#                            TESTS FOR figures.py
# -----------------------------------------------------------------------------

def test_axis_formatter(fig_ax):
    """
    Verify that axis_formatter sets tick parameters, labels, and grid properties.
    """
    fig, ax = fig_ax
    axis_formatter(ax, "Y Label", "X Label", "Test Title")
    # Check that labels have been set.
    assert ax.get_ylabel() == "Y Label"
    assert ax.get_xlabel() == "X Label"
    # Check that grid is enabled (axis is below).
    assert ax._axisbelow is True

def test_plot_hexbin_r2(mock_results_object):
    """
    Smoke test for plot_hexbin_r2: ensure it runs and creates a colorbar.
    """
    fig, ax = plt.subplots(figsize=(6, 4))
    plot_hexbin_r2(mock_results_object, ax, fig, "Spectral_r", title="R2 Hexbin Test")
    # Verify that a colorbar is present (by checking for an Axes with title 'Count').
    colorbars = [child for child in fig.get_children() if isinstance(child, plt.Axes) and child.get_title() == 'Count']
    assert len(colorbars) >= 1
    plt.close(fig)

def test_plot_hexbin_log(mock_results_object):
    """
    Smoke test for plot_hexbin_log: ensure it runs and creates a colorbar.
    """
    fig, ax = plt.subplots(figsize=(6, 4))
    plot_hexbin_log(mock_results_object, ax, fig, "Spectral_r", title="Log Hexbin Test")
    colorbars = [child for child in fig.get_children() if isinstance(child, plt.Axes) and child.get_title() == 'Count']
    assert len(colorbars) >= 1
    plt.close(fig)

def test_shap_violin_basic(mock_results_object):
    """
    Test shap_violin returns a valid feature order list when provided with matching shap values and features.
    """
    fig, ax = plt.subplots(figsize=(6, 4))
    shap_vals = np.delete(mock_results_object.shap_return[0], 0, axis=1)  # Adjust shape: (20, 3)
    shap_x = mock_results_object.shap_return[1].drop(mock_results_object.x_name, axis=1)
    feature_order = shap_violin(ax, shap_vals, features=shap_x, feature_names=shap_x.columns,
                                max_display=5, title="SHAP Violin Test", clear_yticklabels=True)
    assert isinstance(feature_order, list)
    assert len(feature_order) <= shap_x.shape[1]
    plt.close(fig)

def test_shap_violin_shape_mismatch():
    """
    Test that shap_violin raises a ValueError when the shape of shap_values
    does not match the provided features.
    """
    fig, ax = plt.subplots(figsize=(6, 4))
    shap_vals = np.random.randn(10, 5)  # 10 samples, 5 features.
    features = np.random.randn(10, 4)   # Mismatch: 4 columns.
    with pytest.raises(ValueError, match="shape of the shap_values matrix does not match"):
        shap_violin(ax, shap_vals, features=features)
    plt.close(fig)

def test_plot_curve_no_specs(mock_results_object):
    """
    Test that plot_curve executes without error when an empty specs list is provided.
    """
    fig, ax = plt.subplots(figsize=(6, 4))
    returned_ax = plot_curve(mock_results_object, loess=True, specs=[], ax=ax, colormap="Spectral_r", title="Curve Test")
    assert hasattr(returned_ax, "get_xlim")
    plt.close(fig)

def test_plot_curve_with_specs(mock_results_object):
    """
    Test plot_curve with highlighted specifications.
    """
    fig, ax = plt.subplots(figsize=(6, 4))
    # Use a valid specification extracted from specs_names as a list-of-lists.
    specs_to_highlight = [[next(iter(mock_results_object.specs_names.iloc[0]))]]
    plot_curve(mock_results_object, loess=False, specs=specs_to_highlight, ax=ax, colormap="Spectral_r")
    plt.close(fig)

def test_plot_ic_basic(mock_results_object):
    """
    Test that plot_ic executes without error when provided with an empty specs list.
    """
    fig, ax = plt.subplots(figsize=(6, 4))
    plot_ic(mock_results_object, ic='aic', specs=[], ax=ax, colormap="Spectral_r", title="IC Test", despine_left=True)
    plt.close(fig)

def test_plot_ic_invalid_ic(mock_results_object):
    """
    Test that plot_ic raises a KeyError when an invalid IC name is provided.
    """
    fig, ax = plt.subplots(figsize=(6, 4))
    with pytest.raises(KeyError):
        plot_ic(mock_results_object, ic='invalid_ic_name', specs=[], ax=ax, colormap="Spectral_r", title="IC Invalid Test")
    plt.close(fig)

def test_plot_bdist_no_specs(mock_results_object):
    """
    Test that plot_bdist executes without error when an empty specs list is provided.
    """
    fig, ax = plt.subplots(figsize=(6, 4))
    plot_bdist(mock_results_object, specs=[], ax=ax, colormap="Spectral_r", title="Bdist Test", despine_left=True, legend_bool=False)
    plt.close(fig)

def test_plot_bdist_with_specs(mock_results_object):
    """
    Test plot_bdist with highlighted specifications and with legend enabled.
    """
    fig, ax = plt.subplots(figsize=(6, 4))
    specs_to_highlight = [[next(iter(mock_results_object.specs_names.iloc[0]))]]
    plot_bdist(mock_results_object, specs=specs_to_highlight, ax=ax, colormap="Spectral_r", title="Bdist Spec Test", despine_left=True, legend_bool=True)
    plt.close(fig)

def test_plot_kfolds(mock_results_object):
    """
    Test that plot_kfolds executes without error and sets x-axis limits.
    """
    fig, ax = plt.subplots(figsize=(6, 4))
    plot_kfolds(mock_results_object, colormap="Spectral_r", ax=ax, title="Kfolds Test", despine_left=True)
    assert ax.get_xlim() is not None
    plt.close(fig)

def test_plot_bma_runs_and_sets_labels(mock_results_object):
    """
    Ensure plot_bma executes without error and that the axis has correct x-axis label.
    """
    fig, ax = plt.subplots(figsize=(6, 4))
    feature_order = mock_results_object.controls
    plot_bma(mock_results_object, colormap="Spectral_r", ax=ax, feature_order=feature_order, title="BMA Test")
    
    # Check basic axis labels and ticks
    assert ax.get_xlabel() == "BMA Probabilities"
    assert len(ax.patches) == len(feature_order)  # one bar per feature
    plt.close(fig)


@pytest.mark.parametrize("ic", ["aic", "bic", "hqic"])
def test_plot_results_creates_files(tmp_path, mock_results_object, ic):
    """
    Integration test for plot_results.
    Patches os.getcwd() to direct output to a temporary directory.
    Verifies that the expected output file is created.
    """
    test_proj_name = "pytest_figures_test"
    out_dir = tmp_path / "figures" / test_proj_name
    out_dir.mkdir(parents=True, exist_ok=True)

    with patch("os.getcwd", return_value=str(tmp_path)):
        plot_results(
            results_object=mock_results_object,
            loess=True,
            specs=[],
            ic=ic,
            colormap="Spectral_r",
            figsize=(8, 8),
            ext='png',
            project_name=test_proj_name
        )
        main_plot_path = out_dir / f"{test_proj_name}_all.png"
        assert main_plot_path.exists(), f"Expected {main_plot_path} to be created."