import pandas as pd
import os
import numpy as np


def load_union_data(d_path):
    union_df = pd.read_stata(os.path.join(d_path,
                                          'nlsw88.dta'))
    union_df['log_wage'] = np.log(union_df['wage'])*100
    return union_df
