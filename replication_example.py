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
from nrobust.figures import main_plotter


def make_union_example():
    y, c, x = prepare_union(os.path.join('data',
                                         'input',
                                         'nlsw88.dta'))
    full_beta = sm.OLS(y, pd.merge(x, c, how='left',
                                   left_index=True,
                                   right_index=True),
                       hasconst=True).fit().params[0]

    b_spec, p_spec, aic_spec, bic_spec = full_curve(y, x, c, 'simple')
    myrobust = OLSRobust(y=y, x=x)
    beta, p, aic, bic = myrobust.fit(controls=c,
                                     draws=1000,
                                     mode='simple',
                                     sample_size=100,
                                     replace=True)
    union_path = os.path.join('data',
                              'intermediate',
                              'union_example')

    # @TODO Some of these can probably be combined
    save_myrobust(beta, p, aic, bic, union_path)
    save_spec(b_spec, p_spec, aic_spec, bic_spec, union_path)
    beta, summary_df, list_df = load_myrobust(union_path)
    b_spec, p_spec, aic_spec, bic_spec = load_spec(union_path)
    main_plotter(beta, b_spec, full_beta, summary_df,
                 os.path.join(os.getcwd(), 'figures', 'union_path'))

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

    main_plotter(beta, b_spec, full_beta, summary_df,
                 os.path.join(os.getcwd(), 'figures', 'ASC_example'))



if __name__ == "__main__":
#    make_union_example()
    make_ASC_example()