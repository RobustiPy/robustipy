import os
from robustipy.utils import prepare_union, prepare_asc
import matplotlib.pyplot as plt
from robustipy.models import OLSRobust, LRobust

y, c, x, data = prepare_union(os.path.join('data',
                                               'input',
                                               'nlsw88.csv'))
union_robust = OLSRobust(y=[y], x=[x], data=data)
union_robust.fit(controls=c,
                     draws=100)
union_results = union_robust.get_results()



def union_example():
    y, c, x, data = prepare_union(os.path.join('data',
                                               'input',
                                               'nlsw88.csv'))
    union_robust = OLSRobust(y=[y], x=[x], data=data)
    union_robust.fit(controls=c,
                     draws=100)
    union_results = union_robust.get_results()

    union_results.plot(specs=[['hours', 'collgrad'],
                              ['collgrad'],
                              ['hours', 'age']],
                       ic='hqic',
                       figsize=(16, 8))
    plt.savefig(os.path.join(os.getcwd(), 'figures',
                             'union_example',
                             'union_curve.png'))

y, c, x, data = prepare_union(os.path.join('data',
                                           'input',
                                            'nlsw88.csv'))

# LR
y = 'union'
x = 'log_wage'

union_robust = LRobust(y=[y], x=[x], data=data)
union_robust.fit(controls=c,
                 draws=10)
union_results = union_robust.get_results()

union_results.name_av_k_metric

union_results.plot(specs=[['hours', 'collgrad'],
                          ['collgrad'],
                          ['hours', 'age']],
                   ic='hqic',
                   figsize=(16, 10))
plt.show()


def asc_example():
    y, c, x, g, data = prepare_asc(os.path.join('data',
                                                'input',
                                                'CleanData_LASpending.dta'))

    myrobust_panel = OLSRobust(y=[y], x=x, data=data)
    myrobust_panel.fit(controls=c,
                       draws=100,
                       group=g)
    asc_results = myrobust_panel.get_results()

    asc_results.plot(
        specs=[['married'],
               ['widowed', 'disable']],
        ic='bic',
        figsize=(16, 8)
    )

    plt.savefig(os.path.join(os.getcwd(), 'figures',
                             'ASC_example',
                             'asc_curve.png'))


if __name__ == "__main__":
    union_example()
    asc_example()
