import pandas as pd
import os
import numpy as np


def save_union_data(beta, p, aic, bic, d_path):
    union_out = os.path.join(d_path, 'output', 'union')
    np.savetxt(os.path.join(union_out, 'union_betas.csv'),
               beta, delimiter=",")
    np.savetxt(os.path.join(union_out, 'union_p.csv'),
               p, delimiter=",")
    np.savetxt(os.path.join(union_out, 'union_aic.csv'),
               aic, delimiter=",")
    np.savetxt(os.path.join(union_out, 'union_bic.csv'),
               bic, delimiter=",")


def load_union_data(d_path):
    union_df = pd.read_stata(os.path.join(d_path,
                                          'nlsw88.dta'))
    union_df.loc[:, 'log_wage'] = np.log(union_df['wage'].copy())*100
    union_df = union_df[union_df['union'].notnull()].copy()
    union_df.loc[:, 'union'] = np.where(union_df['union'] == 'union', 1, 0)
    union_df.loc[:, 'married'] = np.where(union_df['married'] == 'married', 1, 0)
    union_df.loc[:, 'collgrad'] = np.where(union_df['collgrad'] == 'college grad', 1, 0)
    union_df.loc[:, 'smsa'] = np.where(union_df['smsa'] == 'SMSA', 1, 0)
    indep_list = ['hours', 'age',
                  'grade', 'collgrad', 'married',
                  'south', 'smsa', 'c_city', 'ttl_exp',
                  'tenure']
    for var in indep_list:
        union_df = union_df[union_df[var].notnull()]
    union_y = pd.DataFrame(union_df['log_wage'])
    union_control = union_df[indep_list]
    union_x = pd.DataFrame(union_df['union'])
    return union_y, union_x, union_control
