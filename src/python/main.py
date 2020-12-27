import os
import numpy as np
import pandas as pd
from load_data import load_union_data
from analysis import run_ols_sm, get_mspace, make_robust


def run_union_example(d_path):
    union_y, union_x, union_control = load_union_data(os.path.join(d_path, 'input'))
    beta_full = run_ols_sm(union_y, pd.merge(union_x, union_control,
                           how='left', left_index=True,
                           right_index=True))
    control_list = union_control.columns.to_list()

    model_space = get_mspace(control_list)
    beta = make_robust(union_y, union_x,
                       union_control, model_space,
                       'bic', len(model_space))
    np.savetxt(os.path.join(d_path, 'output', 'union_betas.csv',
                            beta, delimiter=","))

def main():
    d_path = os.path.join(os.getcwd(), '..', '..', 'data')
    run_union_example(d_path)
    # @TODO: new replication one
    # @TODO: new replication two
    # @TODO: new replication three


if __name__ == "__main__":
    main()
