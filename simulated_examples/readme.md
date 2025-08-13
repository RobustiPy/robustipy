# Simulated Examples

This folder contains a suite of standalone Python simulations and a profiling utility that demonstrate how **RobustiPy** can be used on synthetic data to estimate effects, perform cross-validated model selection, and visualize robustness, as well as to profile runtime performance.

Each script runs from the command line (no notebook required) and saves figures to a `../figures` directory.

---

## What’s inside

- **sim1_vanilla.py**  
  Basic OLS robustness demo on simulated data.  
  - **Data**: `n=100` observations; 1 main regressor `x1` + 4 controls `z1…z4`.  
  - **Outcome**: Linear in `x1` and controls with Gaussian noise.  
  - **Model**: `OLSRobust` with bootstrapping (`draws=1000`) and 10-fold CV.  
  - **Output**: SVG plots and summary table.

- **sim2_longitudinal.py**  
  Panel-style simulation with group-level heterogeneity.  
  - **Data**: 1,000 groups × 10 obs each; correlated covariates from a factor model + shocks; group-specific coefficients.  
  - **Model**: `OLSRobust` with `group='group'`, `rescale_y=True`, `rescale_z=True`.  
  - **Output**: HQIC-based selection plots (PDF) and summary table.

- **sim3_constants.py**  
  Multiple focal regressors with fixed coefficients.  
  - **Data**: `n=1000`; `x1` and `z1` as focal regressors, `z2…z7` as controls; correlated multivariate normal design.  
  - **Model**: `OLSRobust` with `draws=1000`, 10-fold CV.  
  - **Output**: HQIC plots (PDF) and summary.

- **sim4_binary.py**  
  Binary outcome example using logistic-style robustness.  
  - **Data**: `n=10,000`; covariates from a factor-structured covariance; latent index thresholded at its median to form `y1 ∈ {0,1}`.  
  - **Model**: `LRobust` with `x=['x1']`, controls `z1…z7`, `draws=1000`, 10-fold CV.  
  - **Output**: HQIC plots for several specs (PDF) and summary.

- **sim5_multipley.py**  
  Multiple dependent variables in a robust OLS framework.  
  - **Data**: `n=1000`; 5 covariates (`x1…x5`); 4 outcomes (`y1…y4`) generated from stacked β-vectors; 4 controls (`z1…z4`).  
  - **Model**: `OLSRobust` with multiple outcomes, `draws=1000`, 10-fold CV, and rescaling of y/x/z.  
  - **Output**: Plots (two-panel layout) showing robustness across specs, plus summary.

- **time_profiler.py**  
  Performance benchmarking tool for **RobustiPy**.  
  - **Purpose**: Tests runtime under different numbers of controls, draws, and CV folds for both `OLSRobust` and `LRobust`.  
  - **Features**:  
    - Skips already-completed runs by tracking them in CSV files.  
    - Runs both OLS and logistic variants with the same synthetic data generation.  
    - Varies control set size (`CONTROL_VARS` subsets), draws (logarithmic sequence), and folds (`FOLDS_LIST`).  
    - Handles segmentation faults gracefully by restarting the process.  
  - **Output**: Figures (PDF) and CSV logs of runtime per configuration.

---

## Requirements

Install **RobustiPy** and basic scientific Python packages:

```bash
pip install robustipy numpy pandas matplotlib
