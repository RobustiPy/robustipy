import pandas as pd
import numpy as np


def prepare_union(path_to_union):
    union_df = pd.read_stata(path_to_union)
    union_df.loc[:, 'log_wage'] = np.log(union_df['wage'].copy()) * 100
    union_df = union_df[union_df['union'].notnull()].copy()
    union_df.loc[:, 'union'] = np.where(union_df['union'] == 'union', 1, 0)
    union_df.loc[:, 'married'] = np.where(union_df['married'] == 'married', 1, 0)
    union_df.loc[:, 'collgrad'] = np.where(union_df['collgrad'] == 'college grad', 1, 0)
    union_df.loc[:, 'smsa'] = np.where(union_df['smsa'] == 'SMSA', 1, 0)
    indep_list = ['hours',
                  'age',
                  'grade',
                  'collgrad',
                  'married',
                  'south',
                  'smsa',
                  'c_city',
                  'ttl_exp',
                  'tenure']
    for var in indep_list:
        union_df = union_df[union_df[var].notnull()]
    y = pd.DataFrame(union_df['log_wage'])
    c = union_df[indep_list]
    x = pd.DataFrame(union_df['union'])
    return y, c, x


def prepare_asc(asc_path):
    ASC_df = pd.read_stata(asc_path, convert_categoricals=False)
    one_hot = pd.get_dummies(ASC_df['year'])
    ASC_df = ASC_df.join(one_hot)
    ASC_df = ASC_df.set_index(['pidp', 'year'])
    ASC_df['dcareNew*c.lrealgs'] = ASC_df['dcareNew'] * ASC_df['lrealgs']
    ASC_df['constant'] = 1
    y = ASC_df['wellbeing_kikert']
    x = ASC_df['lrealgs']
    c = ASC_df[['dcareNew*c.lrealgs',
                'dcareNew',
                'DR',
                'lgva',
                'hhsize',
                'work',
                'retired',
                2005.0,
                ]]
    return y, c, x
