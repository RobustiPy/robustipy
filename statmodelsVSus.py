import pandas as pd
import numpy as np
import statsmodels.api as sm
import linearmodels.panel as lm
from nrobust.utils import simple_ols
from nrobust.utils import simple_panel_ols
from nrobust.utils import group_demean
from linearmodels.datasets import wage_panel


# Simple OLS comparison
# Statsmodels:

data = sm.datasets.get_rdataset('Duncan', 'carData')

y = data.data['income']
x = data.data['education']
x = sm.add_constant(x)
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
exog = sm.add_constant(data[['expersq', 'married']])
exog

mod = lm.PanelOLS(dependent, exog, entity_effects=True)
res = mod.fit(cov_type='unadjusted')
res

# Nrobust simple_panel_ols
data = wage_panel.load()
yy = data.lwage
g = data.nr
y_g = pd.concat([yy, g], axis=1)
x_g = sm.add_constant(data[['expersq', 'nr']])

simple_panel_ols(y_g, x_g, 'nr')

from nrobust.utils import panel_ols

panel_ols(dependent, exog)['b']


# mock example

df = wage_panel.load()

df_demean = df.copy()

# calculate the entity(state) mean beer tax
df_demean['lwage_mean'] = df_demean.groupby('nr').lwage.transform(np.mean)
df_demean['expersq_mean'] = df_demean.groupby('nr').expersq.transform(np.mean)
df_demean['married_mean'] = df_demean.groupby('nr').married.transform(np.mean)

# demean, subtract each row by the entity-mean
df_demean['lwage_c'] = df_demean['lwage'] - df_demean['lwage_mean']
df_demean['expersq_c'] = df_demean['expersq'] - df_demean['expersq_mean']
df_demean['married_c'] = df_demean['married'] - df_demean['married_mean']
df_demean['const'] = 1


yyy = df_demean['lwage_c']
xxx = df_demean[['const', 'expersq_c', 'married_c']]

model = sm.OLS(yyy, xxx)
results2 = model.fit(cov_type='fixed scale')
results2.summary()

yyy
simple_ols(yyy, xxx)['b'].round(4) # this works! why does not work in the library??????

group_demean(df_demean[['lwage', 'nr']], group='nr') # demean is working fine

df_demean[['expersq_c']]


data = wage_panel.load()
yy = data.lwage
g = data.nr
y_g = pd.concat([yy, g], axis=1)
x_g = sm.add_constant(data[['expersq', 'married', 'nr']])


yyyy = group_demean(y_g, group='nr')
xxxx = group_demean(x_g, group='nr')
xxxx['const'] = 1

simple_ols(yyyy, xxxx)['b'].round(4)

simple_panel_ols(y_g, x_g, 'nr')['b'].round(4)
