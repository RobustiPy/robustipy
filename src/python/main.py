import os
import numpy as np
import pandas as pd
from load_data import load_union_data, save_union_data
from analysis import run_full_ols, get_mspace, make_robust


def run_union_example(d_path):
    y, x, c = load_union_data(os.path.join(d_path, 'input'))
    beta_full = run_full_ols(y, pd.merge(x, c,
                             how='left', left_index=True,
                             right_index=True))
    control_list = c.columns.to_list()
    model_space = get_mspace(control_list)
    beta, p, aic, bic, ll = make_robust(y, x, c, model_space,
                                        len(model_space))
    save_union_data(beta, p, aic, bic, ll, d_path)


def main():
    d_path = os.path.join(os.getcwd(), '..', '..', 'data')
    run_union_example(d_path)
    # @TODO: new replication one
    # @TODO: new replication two
    # @TODO: new replication three


if __name__ == "__main__":
    main()
