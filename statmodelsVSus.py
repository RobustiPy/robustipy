import pandas as pd
import statsmodels.api as sm
import linearmodels.panel as lm
from nrobust.utils import simple_ols
from nrobust.utils import simple_panel_ols
from linearmodels.datasets import wage_panel


# Simple OLS comparison
# Statsmodels:

data = sm.datasets.get_rdataset('Duncan', 'carData')

y = data.data['income']
x = data.data['education']
x = sm.add_constant(X)
x

model = sm.OLS(y, x)
results = model.fit()

# Nrobust simple_ols

our_results = simple_ols(y,x)


# comparison:
our_results['b'].round(6) == results.params.round(6)  # Results are identical
                                                      # simple_ols works as planned

our_results['b'].round(6)
results.params.round(6)


# Fixed effects comparison
# Linearmodels:

data = wage_panel.load()
data = data.set_index(['nr', 'year'])

data

dependent = data.lwage
exog = sm.add_constant(data[['expersq','married','union']])
mod = lm.PanelOLS(dependent, exog, entity_effects=False)
res = mod.fit(cov_type='unadjusted')
res

# Nrobust simple_panel_ols
data = wage_panel.load()
yy = data.lwage
g = data.nr
y_g = pd.concat([yy, g], axis=1)
x_g = sm.add_constant(data[['expersq', 'married', 'union', 'nr']])

simple_panel_ols(y_g, x_g, 'nr')

from nrobust.utils import panel_ols

panel_ols(dependent, exog)['b']
