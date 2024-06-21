<img src="https://github.com/RobustiPy/RobustiPy.github.io/blob/main/assets/robustipy_logo_transparent_large_trimmed.png?raw=true" width="700"/>

![coverage](https://img.shields.io/badge/Purpose-Research-yellow)
[![Generic badge](https://img.shields.io/badge/Python-3.11-red.svg)](https://shields.io/)
[![Generic badge](https://img.shields.io/badge/R-brightgreen.svg)](https://shields.io/)
[![Generic badge](https://img.shields.io/badge/License-GNU3.0-purple.svg)](https://shields.io/)

Welcome to the home of `RobustiPy`, a library for the creation of a more robust and stable model space. Kindly note: **this project is in the early stages of development. Its functionally and API might change without notice**!

`RobustiPy` performs Multiversal/Specification Curve Analysis. Multiversal/Specification Curve Analysis attempts to compute most or all reasonable specifications of a statistical model, understanding a specification as a single attempt to estimate an estimand of interest, whether through a particular choice of covariates, hyperparameters, data cleaning decisions, and so forth.

More formally, lets assume we have a general model of the form:

$$
y = f(x, \textbf{z}) + \epsilon .
$$

We are essentially attempting to model a dependent variable $y$ using some kind of function $f()$, some predictor(s) $x$, some covariates $z$, and random error $\epsilon$. For all of these elements, different estimates of the coefficient of interest are produced. Lets assume $y$, $x$ and $z$ are imperfect latent variables or a collection of latent variables. Researchers can come up with _reasonable_ operationalisations of $y$, $x$ and $z$, running the analysis most usually with one or a small number of combinations of them. Ideally -- in an age of vast computational resources -- we should take all such _reasonable_ operationalisations, and store them in sets:

```math
Y = \{y_{1}, y_{2}, \dots, y_{n}\}
```
```math
X = \{x_{1}, x_{2}, \dots, x_{n}\}
```
```math
Z = \{z_{1}, z_{2}, \dots, z_{n}\}
```

`RobustiPy` will then:

```math
\Pi = \left\{ \overline{S_i} \mid S_i \in \mathcal{P}(Y) \text{ and } S_i \neq \emptyset \right\} \times X \times \mathcal{P}(Z)
```

In words, it creates a set contaning the aritmentic mean of the elements of the powerset $\mathcal{P}$ (all possible combination of any length) of $Y$, the set $X$ and the powerset of $Z$ to then produce the Cartesian product of these sets, creating the full set of possible model specifications $\Pi$. `RobustiPy` then takes these specifications, fits them against observable (tabular) data, and produces coefficients and relevant metrics for each version of the predictor $x$ in the set $X$.

## Installation

To install directly (in `Python`) from GitHub, run:

```
git clone https://github.com/RobustiPy/robustipy.git
cd robustipy
pip install .
```

## Usage

In a Python script (or Jupyter Notebook), import the `OLSRobust` class by running:

```python
from robustipy.models import OLSRobust
model_robust = OLSRobust(y=y, x=x, data=data)
model_robust.fit(controls=c,
	         draws=100,
                 sample_size=100)
model_results = model_robust.get_results()
```

Where `y` is a list of variable names used to create your dependent variable, and `x` is a list of variable names used as predictors.

## Example

A working usage example script -- `replication_example.py` -- is provided at the root of this repository. You can also find a number of empirical examples [here](https://github.com/RobustiPy/Empirical-Examples) and some simulated examples [here](Simulated-Examples).

![Union dataset example](./figures/union_example/union_curve.svg)

## Website

We have a website made with [jekkyl-theme-minimal](https://github.com/pages-themes/minimal) that you can find [here](https://robustipy.github.io/). It also contains details of a Hackathon!

## Contributing and Code of Conduct

Please kindly see our [guide for contributors](https://github.com/RobustiPy/robustipy/blob/main/contributing.md) file as well as our [code of conduct](https://github.com/RobustiPy/robustipy/blob/main/CODE-OF-CONDUCT.md). If you would like to become a formal project maintainer, please simply contact the team to discuss!

## License

This work is free. You can redistribute it and/or modify it under the terms of the GNU GPL 3.0 license. The two dataset which is bundled with the library comes with it's own licensing conditions, and should be treatedly accordingly.

## Acknowledgements
We are grateful to the extensive comments made by various academic communities over the course of our thinking about this work, not least the members of the [ESRC Centre for Care](https://centreforcare.ac.uk/) and the [Leverhulme Centre for Demographic Science](https://demography.ox.ac.uk/).

<div style="display: flex; justify-content: space-between;">
    <img src="https://github.com/RobustiPy/RobustiPy.github.io/blob/main/assets/cfc_logo.png?raw=true" alt="CfC" style="width: 200px; height: auto; margin-right: 20px;">
    <img src="https://github.com/RobustiPy/RobustiPy.github.io/blob/main/assets/lcds_logo.png?raw=true" alt="LCDS" style="width: 280px; height: auto;">
</div>
