# Empirical Examples

This folder contains a curated set of empirical Jupyter notebooks showcasing how **RobustiPy** can replicate classic results and explore robust inference in applied economics and related fields. The examples emphasize bootstrapping, specification variation, and model‑agnostic diagnostics, using modern Python tooling (e.g., `statsmodels`, `pandas`) together with RobustiPy’s robust pipelines.

## What’s inside

Brief summaries of each notebook:

- **empirical_type1_acemoglu.ipynb** — Replicates *The Colonial Origins of Comparative Development*; uses `statsmodels` + **RobustiPy** to probe specification sensitivity.
- **empirical_type1_ehrlich.ipynb** — Replicates *Participation in Illegitimate Activities: Ehrlich Revisited (1960)* using **RobustiPy** for robust inference.
- **empirical_type1_union.ipynb** — Uses `union.dta` (NLS Young Women 1968–1988; missing 1974, 1976, 1979, 1981, 1984, 1986) to study wages vs. union membership with robustness checks.
- **empirical_type2_mrw.ipynb** — Revisits MRW (1992) with **PWT v10.01**; tests Solow covariates `log(n+g+δ)` and `log(I/Y)` via two RobustiPy runs and inspects adjusted R² distributions.
- **empirical_type3_asc.ipynb** — Adult Social Care example based on Zhang, Bennett & Yeandle (BMJ Open). Data are not public due to UKHLS–ASC linkage; contact the authors for replication materials.
- **empirical_type3_ukhls.ipynb** — Example using Understanding Society: Longitudinal Teaching Dataset (Waves 1–9, 2009–2018). Registration is immediate; documentation is provided with the dataset. Demonstrates applying **RobustiPy** to longitudinal survey data.
- **empirical_type4_airline.ipynb** — Predicts airline customer satisfaction (129,880 rows, 22 features). Factors include punctuality, service quality, and other operational metrics; demonstrates robust modelling for service improvement insights.
- **empirical_type4_framingham.ipynb** — Uses the Framingham heart disease dataset (4,240 records, 16 columns) to predict 10‑year CHD risk. Demonstrates the `LRobust` class for binary outcomes, leveraging a large feature space and focusing on out‑of‑sample accuracy.
- **empirical_type5_gino.ipynb** — Shows use of **RobustiPy** with multiple dependent variables by replicating the retracted Gino et al. (2020) study. References Data Colada’s blog discussions of the case.
- **empirical_type5_orben.ipynb** — Multiple dependent variables; replicates Orben & Przybylski (Nature Human Behaviour, 2019). Data via the UK Data Service; reference code available on Amy Orben’s GitHub.

## Requirements

To run these notebooks, you must have the **Jupyter Notebook stack** installed alongside **RobustiPy** and its dependencies. This typically includes:
- `jupyter` / `notebook` (or `jupyterlab`)
- `robustipy`
