import statsmodels.api as sm
import pandas as pd
import numpy as np
from itertools import chain, combinations
from tqdm import tqdm


def bootstrapper(comb_var):
    samp_df = comb_var[np.random.choice(comb_var.shape[0], 800, replace=True)]
    # @TODO generalize the frac to the function call
    Y = samp_df[:, 0]
    X = samp_df[:, 1:]
    beta = np.dot(np.linalg.inv(np.dot(X.T, X)), np.dot(X.T, Y))
    return beta[0]#model.params[0], model.pvalues[0], model.bic
    # @TODO generalise this


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


def make_robust(y, x, control,
                space, info, straps=1000):
    beta = np.empty([len(space), straps])
    ic = np.empty([len(space), straps])
    p = np.empty([len(space), straps])
    for spec in tqdm(range(len(space))):
        if spec == 0:
            comb_var = x
        else:
            comb_var = pd.merge(x,
                                control[list(space[spec])],
                                how='left', left_index=True,
                                right_index=True)
        comb_var = pd.merge(y, comb_var,
                            how='left', left_index=True,
                            right_index=True)
        comb_var.loc[:, 'constant'] = 1
        beta_list=[]
#        p_list=[]
#        ic_list=[]
        for boot in range(straps):
            beta_out = bootstrapper(comb_var.to_numpy())
            beta_list.append(beta_out)
        beta[:, spec] = beta_list
#        p[:, spec] = p_list
#        ic[:, spec] = ic_list
    return beta
