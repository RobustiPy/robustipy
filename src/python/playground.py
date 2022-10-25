import os
from src.python.models import OLSRobust
from src.python.replication_data_prep import prepare_union, prepare_asc
from src.python.utils import save_myrobust, load_myrobust
from src.python.figures import main_plotter

from src.python.utils import simple_ols


# union example

y, c, x = prepare_union(os.path.join('data', 'input', 'nlsw88.dta'))
myrobust = OLSRobust(x, y)
beta, p, aic, bic = myrobust.fit(controls=c, samples=10, mode='simple')
union_example = os.path.join('data', 'intermediate', 'union_example')
save_myrobust(beta, p, aic, bic, union_example)
beta, summary_df, list_df = load_myrobust(union_example)

summary_df.mean()



main_plotter(beta, summary_df, os.path.join(os.getcwd(),
                                            'figures',
                                            'union_example'))
# asc example

y, c, x = prepare_asc(os.path.join('data', 'input', 'CleanData_LASpending.dta'))
myrobust_panel = OLSRobust(x, y)
beta, p, aic, bic = myrobust_panel.fit(controls=c,
                                       samples=10,
                                       mode='panel')
ASC_example = os.path.join('data', 'intermediate', 'ASC_example')
save_myrobust(beta, p, aic, bic, ASC_example)
beta, summary_df, list_df = load_myrobust(ASC_example)
main_plotter(beta, summary_df, os.path.join(os.getcwd(),
                                            'figures',
                                            'ASC_example'))
simple_ols(y, x)
