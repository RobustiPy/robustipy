import pandas as pd
from models import OLSRobust
import warnings
import os
from prepare_data import prepare_union, prepare_asc
from utils import save_myrobust
warnings.filterwarnings("ignore")

d_path = os.path.join('..', '..',)
# union example

# run once, comment out for prototyping

# y, c, x, control_list, model_space = prepare_union(os.path.join(d_path, 'data', 'input', 'nlsw88.dta'))
# myrobust = OLSRobust(x, y)
# beta, p, aic, bic = myrobust.fit(c=c, space=model_space, s=100, mode='simple')
# save_myrobust(beta, p, aic, bic, os.path.join(d_path, 'intermediate'), 'union_example')

# load arrays to prevent the above having to be run multiple times
# '@TODO write .utils func to load this in and then do selection and weighting

# asc example
#y, c, x, control_list, model_space = prepare_asc(os.path.join(d_path, 'data', 'input', 'CleanData_LASpending.dta'))
#myrobust_panel = OLSRobust(x, y)
#beta, p, aic, bic = myrobust_panel.fit(c=c, space=model_space, s=10, mode='panel')
#save_myrobust(beta, p, aic, bic, os.path.join(d_path, 'intermediate'), 'ASC_example')
# '@TODO write .utils func to load this in and then do selection and weighting