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
                     draws=10,
                     mode='simple',
                     sample_size=100,
                     replace=True)

    union_results = union_robust.get_results()

    fig, ax1, ax2, ax3 = union_results.plot(specs=[['hours', 'collgrad'],
                                                   ['collgrad'],
                                                   ['hours', 'age']],
                                            figsize=(36, 12))
    plt.savefig(os.path.join(os.getcwd(), 'figures',
                             'union_example',
                             'curve.png'))


def make_ASC_example():
    y, c, x = prepare_asc(os.path.join('data',
                                       'input',
                                       'CleanData_LASpending.dta'))

    # @TODO handle this missingvalue warning:
    #  dropping nans results in singularity issues
    comb = pd.DataFrame(pd.merge(x, c, how='left',
                                 left_index=True,
                                 right_index=True))
    mod = PanelOLS(y, pd.merge(x, c, how='left',
                               left_index=True,
                               right_index=True),
                   drop_absorbed=True,
                   entity_effects=True)
    full_beta = mod.fit(cov_type='clustered', cluster_entity=True).params[0]
    b_spec, p_spec, aic_spec, bic_spec = full_curve(y, x, c, 'panel')
    myrobust_panel = OLSRobust(y=y, x=x)
    beta, p, aic, bic = myrobust_panel.fit(controls=c,
                                           draws=20,
                                           mode='panel',
                                           sample_size=50000,
                                           replace=True)
    ASC_path = os.path.join('data', 'intermediate', 'ASC_example')
    save_myrobust(beta, p, aic, bic, ASC_path)
    save_spec(b_spec, p_spec, aic_spec, bic_spec, ASC_path)
    beta, summary_df, list_df = load_myrobust(ASC_path)
    b_spec, p_spec, aic_spec, bic_spec = load_spec(ASC_path)


if __name__ == "__main__":
    make_union_example()
    #make_ASC_example()
