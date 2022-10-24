import pandas as pd
from src.python.models import OLSRobust
import warnings
import os
from src.python.replication_data_prep import prepare_union, prepare_asc
from src.python.utils import save_myrobust, load_myrobust
from itertools import combinations, chain
from tqdm import tqdm

# union example

# run once, comment out for prototyping

#y, c, x, control_list = prepare_union(os.path.join('data', 'input', 'nlsw88.dta'))
#myrobust = OLSRobust(x, y)
#beta, p, aic, bic = myrobust.fit(controls=c, samples=100, mode='simple')
union_example = os.path.join('data', 'intermediate', 'union_example')
#save_myrobust(beta, p, aic, bic, union_example)
beta, p, aic, bic = load_myrobust(union_example)

# asc example
#y, c, x, control_list = prepare_asc(os.path.join(d_path, 'data', 'input', 'CleanData_LASpending.dta'))
#myrobust_panel = OLSRobust(x, y)
#beta, p, aic, bic = myrobust_panel.fit(c=c, space=model_space, s=10, mode='panel')
ASC_example = os.path.join('data', 'intermediate', 'ASC_example')
#save_myrobust(beta, p, aic, bic, ASC_example)
beta, p, aic, bic = load_myrobust(ASC_example)