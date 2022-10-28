import os
from nrobust.models import OLSRobust
from nrobust.replication_data_prep import prepare_union, prepare_asc
from nrobust.utils import save_myrobust, load_myrobust
from nrobust.figures import main_plotter
from nrobust.figures import plot_joyplot, plot_curve

# union example

y, c, x = prepare_union(os.path.join('data', 'input', 'nlsw88.dta'))
myrobust = OLSRobust(y=y, x=x)
beta, p, aic, bic = myrobust.fit(controls=c,
                                 draws=1000,
                                 mode='simple',
                                 sample_size=100,
                                 replace=True)
union_example = os.path.join('data', 'intermediate', 'union_example')
save_myrobust(beta, p, aic, bic, union_example)
beta, summary_df, list_df = load_myrobust(union_example)

plot_joyplot(beta, './')
plot_curve(summary_df=summary_df, fig_path='./')


main_plotter(beta, summary_df, os.path.join(os.getcwd(),
                                            'figures',
                                            'union_example'))
# asc example

y, c, x = prepare_asc(os.path.join('data', 'input', 'CleanData_LASpending.dta'))
myrobust_panel = OLSRobust(y=y, x=x)
beta, p, aic, bic = myrobust_panel.fit(controls=c,
                                       draws=20,
                                       mode='panel',
                                       sample_size=50000,
                                       replace=True)

ASC_example = os.path.join('data', 'intermediate', 'ASC_example')
save_myrobust(beta, p, aic, bic, ASC_example)
beta, summary_df, list_df = load_myrobust(ASC_example)
main_plotter(beta, summary_df, os.path.join(os.getcwd(),
                                            'figures',
                                            'ASC_example'))

plot_joyplot(beta, './')

plot_curve(summary_df=summary_df, fig_path='./')
