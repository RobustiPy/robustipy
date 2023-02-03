import os
from nrobust.replication_data_prep import prepare_union,\
    prepare_asc
import matplotlib.pyplot as plt
from nrobust.models import OLSRobust


def union_example():
    y, c, x, data = prepare_union(os.path.join('data',
                                               'input',
                                               'nlsw88.dta'))

    union_robust = OLSRobust(y=[y], x=[x], data=data)
    union_robust.fit(controls=c,
                     draws=100,
                     sample_size=100,
                     replace=True)
    union_results = union_robust.get_results()

    fig, ax1, ax2, ax3 = union_results.plot(specs=[['hours', 'collgrad'],
                                                   ['collgrad'],
                                                   ['hours', 'age']],
                                            figsize=(36, 12))
    plt.savefig(os.path.join(os.getcwd(), 'figures',
                             'union_example',
                             'union_curve.png'))


def asc_example():
    y, c, x, g, data = prepare_asc(os.path.join('data',
                                                'input',
                                                'CleanData_LASpending.dta'))

    myrobust_panel = OLSRobust(y=[y], x=x, data=data)
    myrobust_panel.fit(controls=c,
                       draws=10,
                       group=g,
                       sample_size=20000,
                       replace=True)
    asc_results = myrobust_panel.get_results()

    fig, ax1, ax2, ax3 = asc_results.plot()

    plt.savefig(os.path.join(os.getcwd(), 'figures',
                             'ASC_example',
                             'asc_curve.png'))


if __name__ == "__main__":
    union_example()
    asc_example()
