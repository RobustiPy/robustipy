# nrobust

A set of packages for a robust, certain and stable model space.
This project is in early stages of development and its functionally and API might change without notice.

# Installation

## From PyPI

On a terminal run:

```
pip install nrobust
```

## From GitHub

To install directly from Github run:

```
git glone https://github.com/centre-for-care/nrobust.git
cd nrobust
pip install .
```

# Usage

In a Python script import OLSRobust class running:
```python
from nrobust.models import OLSRobust

my_ols_robust = OLSrobust(y=y, x=x)

```
Where `y` is a 1D array containing your target variable data, and `x` is a 1D array containing your predictor data.

# Example

A working usage example `replication_example.py` is provided cloning this repository. 


@TODO next: implement weighting and selection over the loaded outcome spaces
