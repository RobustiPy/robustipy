# Getting Started with RobustiPy

> A practical, end‑to‑end guide for fitting robust specifications, managing compute, and interpreting outputs.

---

## At a Glance

1. Prepare a clean `pandas.DataFrame` with your outcome(s), main predictor(s), and candidate controls.
2. Fit a model across many specifications with `draws` and/or `kfold`.
3. Review `results.summary()` for key robustness statistics.
4. Generate the multi‑panel figure with `results.plot()`.
5. Interpret the panels using the guide linked at the end.

---

## Installation

```bash
pip install robustipy
```

For the latest development version:

```bash
git clone https://github.com/RobustiPy/robustipy.git
cd robustipy
pip install .
```

---

## Data Preparation

RobustiPy expects a tidy `pandas.DataFrame` where each column corresponds to a variable.

Recommendations:
- Make sure your `y`, `x`, and `controls` columns are numeric or properly encoded.
- Handle missing values in advance (drop, impute, or encode).
- Keep a small subset of rows for quick smoke tests before scaling up.

---

## Model Types

- **OLSRobust**: continuous outcomes.
- **LRobust**: binary outcomes (logistic regression).

Both accept single outcomes or lists of outcomes (run separately) and produce a unified results object.

---

## Quick Start (OLS)

```python
import pandas as pd
from robustipy.models import OLSRobust

# data = pd.read_csv("your_data.csv")

y = "your_outcome"
x = "your_key_predictor"
controls = ["control_1", "control_2", "control_3", "control_4"]

model = OLSRobust(y=y, x=x, data=data)
model.fit(
    controls=controls,
    draws=1000,
    kfold=10,
    oos_metric="pseudo-r2",
    seed=192735,
    n_cpu=4,
)

results = model.get_results()
results.summary(digits=3)
results.plot(ic="aic", project_name="union_example", figpath="./figures")
```

---

## Quick Start (Logistic Regression)

```python
from robustipy.models import LRobust

model = LRobust(y="outcome_binary", x="treatment", data=data)
model.fit(
    controls=controls,
    draws=500,
    kfold=5,
    oos_metric="pseudo-r2",
    seed=123,
)

results = model.get_results()
results.summary()
results.plot(ic="bic", oddsratio=True)
```

---

## Specification Space Basics

RobustiPy builds many “reasonable” specifications by varying which controls are included. The control list you pass to `fit()` defines the candidate set. The full specification space can be large, so RobustiPy lets you sample or downscale.

---

## Core Fit Parameters (Reference)

| Parameter | Purpose | Notes |
| --- | --- | --- |
| `controls` | Candidate control variables | Required list of column names |
| `draws` | Bootstrap resamples per spec | If `None`, bootstrapping is skipped |
| `kfold` | Folds for cross‑validation | Requires `oos_metric` |
| `oos_metric` | OOS metric | `'pseudo-r2'` or `'rmse'` |
| `n_cpu` | Parallel processes | Defaults to all available CPUs minus one if not provided |
| `seed` | Reproducibility | Propagated to all random ops |
| `group` | Fixed effects grouping | De‑means outcomes by group |
| `z_specs_sample_size` | Sample specs | Randomly samples control‑set combinations |
| `composite_sample` | Composite bootstrap | Reduces compute by sampling before specs |
| `rescale_y/x/z` | Standardization | Rescales variables to mean 0, sd 1 |
| `threshold` | Workload warning | Warns if specs × draws × folds is too large |

---

## Compute‑Friendly Run Example

```python
model.fit(
    controls=controls,
    draws=200,
    kfold=5,
    z_specs_sample_size=300,
    threshold=200000,
    n_cpu=4,
)
```

---

## Results Object: What You Get

After `get_results()`, the results object includes:

- `results.summary()` for a compact robustness overview.
- `results.plot()` for the multi‑panel figure.
- `results.summary_df` with per‑spec metrics (ICs, OOS performance, etc.).
- `results.estimates`, `results.p_values`, `results.r2_values` for full matrices of outcomes.

---

## Plotting Options You’ll Use Most

- `ic`: `'aic'`, `'bic'`, or `'hqic'`.
- `specs`: list of specific control sets to highlight (max 3).
- `ci`: confidence interval width.
- `loess`: smooth CI bounds.
- `colormap`: colormap for highlights and colorbars.
- `highlights`: toggle full‑model and null‑model highlights.
- `oddsratio`: show exponentiated coefficients for logistic models.
- `figpath`, `project_name`, `ext`: output location and file format.

---

## Highlighting Specific Specifications

```python
specs = [
    ["age", "tenure", "hours"],
    ["age", "tenure", "hours", "collgrad"],
]

results.plot(specs=specs, ic="hqic")
```

---

## Output Files

`results.plot(...)` saves a combined multi‑panel figure and a set of individual panels. Use `figpath` and `project_name` to control the output directory and filename prefix.

---

## Common Pitfalls

- The x‑axis in the spec curve is **rank‑ordered**, not time.
- Cross‑validation requires `oos_metric`.
- Large control sets can explode the spec space. Use sampling parameters early.
- If you change colormap or highlights, verify the legend colors line up across panels.

---

## Next Steps

- For a full subfigure‑by‑subfigure interpretation, read **`INTERPRETATION_GUIDE.MD`**.
- Browse the empirical example notebooks on GitHub: [empirical_examples](https://github.com/RobustiPy/robustipy/tree/main/empirical_examples)
