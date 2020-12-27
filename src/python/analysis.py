import statsmodels.api as sm
import pandas as pd
from itertools import chain, combinations


def get_mspace(varnames) -> list:
    model_space = []

    def all_subsets(ss):
        return chain(*map(lambda x: combinations(ss, x),
                          range(0, len(ss) + 1)))

    for subset in all_subsets(varnames):
        model_space.append(subset)
    return model_space


def run_ols_sm(y, x):
    x.loc[:, 'constant'] = 1
    model = sm.OLS(y, x, hasconst=True).fit()
    print(model.summary())
    return model.params[0]


def make_robust(union_y, union_x, union_control, space):
    beta_mat = pd.DataFrame(index=range(0, len(space)),
                            columns=range(0, len(space)))
    ic_mat = pd.DataFrame(index=range(0, len(space)),
                          columns=range(0, len(space)))
    p_mat = pd.DataFrame(index=range(0, len(space)),
                         columns=range(0, len(space)))
    for spec in space:
        comb_var = pd.merge(union_x, union_control[list(space)],
                           how='left', left_index=True,
                           right_index=True)
        comb_var.loc[:, 'constant'] = 1
    return beta_mat, ic_mat, p_mat
