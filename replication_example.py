import os
import pandas as pd
import numpy as np
import statsmodels.api as sm
from linearmodels.panel import PanelOLS
from nrobust.models import OLSRobust
from nrobust.replication_data_prep import prepare_union,\
    prepare_asc
from nrobust.utils import save_myrobust,\
    load_myrobust, full_curve, save_spec,\
    load_spec
import matplotlib.pyplot as plt


def make_union_example():
    y, c, x, data = prepare_union(os.path.join('data',
                                         'input',
                                         'nlsw88.dta'))
    data = data.dropna()
    union_robust = OLSRobust(y=y, x=x, data=data)
    union_robust.fit(controls=c,
                     draws=5,
                     sample_size=100,
                     replace=True)

    union_results = union_robust.get_results()

    fig, ax1, ax2, ax3 = union_results.plot(specs=[['hours', 'collgrad'],
                                                   ['collgrad'],
                                                   ['hours', 'age']],
                                            figsize=(36, 12))
    plt.savefig(os.path.join(os.getcwd(), 'figures',
                             'union_example',
                             'curve_exp1.png'))


make_union_example()


def make_ASC_example():
    y, c, x, g, data = prepare_asc(os.path.join('data',
                                                'input',
                                                'CleanData_LASpending.dta'))
    myrobust_panel = OLSRobust(y=y, x=x, data=data)
    myrobust_panel.fit(controls=c,
                       draws=100,
                       group=g,
                       sample_size=100,
                       replace=True)
    results = myrobust_panel.get_results()

    fig, ax1, ax2, ax3 = results.plot()
    plt.savefig(os.path.join(os.getcwd(), 'figures',
                             'ASC_example',
                             'curve.png'))


make_ASC_example()


y, c, x, g, data = prepare_asc(os.path.join('data',
                                            'input',
                                            'CleanData_LASpending.dta'))

if __name__ == "__main__":
    make_union_example()
    #make_ASC_example()
