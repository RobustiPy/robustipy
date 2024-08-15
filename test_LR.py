"""
# Created by valler at 18/06/2024
Feature: 

"""


import os
from robustipy.utils import prepare_union, prepare_asc
import matplotlib.pyplot as plt
from robustipy.models import OLSRobust,LRobust

y, c, x, data = prepare_union(os.path.join('data',
                                           'input',
                                           'nlsw88.dta'))


# OLS
union_robust = OLSRobust(y=[y], x=[x], data=data)
union_robust.fit(controls=c,
                 draws=10)
union_results = union_robust.get_results()

union_results.plot(specs=[['hours', 'collgrad'],
                          ['collgrad'],
                          ['hours', 'age']],
                   ic='hqic',
                   figsize=(16, 10))
plt.show()

union_results.summary()

# LR
y = 'union'
x = 'log_wage'

union_robust = LRobust(y=[y], x=[x], data=data)
union_robust.fit(controls=c,
                 draws=10)
union_results = union_robust.get_results()

union_results.plot(specs=[['hours', 'collgrad'],
                          ['collgrad'],
                          ['hours', 'age']],
                   ic='hqic',
                   figsize=(16, 10))

df_summary = union_results.summary_df
union_results.summary()



df_summary = union_results.summary_df

import requests
