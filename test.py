import os
from nrobust.replication_data_prep import prepare_union,\
    prepare_asc
import matplotlib.pyplot as plt
from nrobust.utils import decorator_timer
from nrobust.models import OLSRobust, OLSRobust_fast

y, c, x, data = prepare_union(os.path.join('data',
                                           'input',
                                           'nlsw88.dta'))
data = data.dropna()


@decorator_timer
def old():
    union_robust = OLSRobust(y=[y], x=[x], data=data)
    union_robust.fit(controls=c,
                     draws=20,
                     sample_size=500,
                     replace=True)

@decorator_timer
def new():
    union_robust = OLSRobust_fast(y=[y], x=[x], data=data)
    union_robust.fit(controls=c,
                     draws=20,
                     sample_size=500,
                     replace=True)

old()
new()

y, c, x, g, data = prepare_asc(os.path.join('data',
                                            'input',
                                            'CleanData_LASpending.dta'))

c

myrobust_panel = OLSRobust_fast(y=[y], x=x, data=data)
myrobust_panel.fit(controls=c,
                   draws=30,
                   group=g,
                   sample_size=10000,
                   replace=True)
results = myrobust_panel.get_results()

fig, ax1, ax2, ax3 = results.plot()

plt.show()
