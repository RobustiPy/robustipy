import os
import pandas as pd
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
    y, c, x = prepare_union(os.path.join('data',
                                         'input',
                                         'nlsw88.dta'))

    union_robust = OLSRobust(y=y, x=x)
    union_robust.fit(controls=c,
                     draws=100,
                     sample_size=1000,
                     replace=True)

    union_results = union_robust.get_results()

    fig, ax1, ax2, ax3 = union_results.plot(specs=[['hours', 'collgrad'],
                                                   ['collgrad'],
                                                   ['hours', 'age']],
                                            figsize=(36, 12))
    plt.savefig(os.path.join(os.getcwd(), 'figures',
                             'union_example',
                             'curve.png'))


make_union_example()


def make_ASC_example():
    y, c, x, g = prepare_asc(os.path.join('data',
                                          'input',
                                          'CleanData_LASpending.dta'))

    # @TODO handle this missingvalue warning:
    #  dropping nans results in singularity issues
    #comb = pd.DataFrame(pd.merge(x, c, how='left',
    #                             left_index=True,
    #                             right_index=True))
    #mod = PanelOLS(y, pd.merge(x, c, how='left',
    #                           left_index=True,
    #                           right_index=True),
    #               drop_absorbed=True,
    #               entity_effects=True)
    #full_beta = mod.fit(cov_type='clustered', cluster_entity=True).params[0]
    #b_spec, p_spec, aic_spec, bic_spec = full_curve(y, x, c, 'panel')
    myrobust_panel = OLSRobust(y=y, x=x)
    myrobust_panel.fit(controls=c,
                       draws=1,
                       group=g,
                       sample_size=50000,
                       replace=True)
    results = myrobust_panel.get_results()

    fig, ax1, ax2, ax3 = results.plot()
    plt.savefig(os.path.join(os.getcwd(), 'figures',
                             'ASC_example',
                             'curve.png'))


make_ASC_example()


if __name__ == "__main__":
    make_union_example()
    #make_ASC_example()
