import os
import numpy as np
from robustipy.models import OLSRobust
import pandas as pd
import matplotlib.pyplot as plt



def sim1(project_name):

    beta1 = np.array([.2, .5, -.4, -.7, .2])
    beta2 = np.array([.3, .4, -.35, -.8, .2])
    beta3 = np.array([.15, .6, -.45, -.1, .2])
    beta4= np.array([.4, .3, -.5, -.1, .2])

    cov_matrix = np.array([[1, 0.7, 0.5, 0.7, 0.1],
                           [0.7, 1, 0.5, 0.3, 0.2],
                           [0.5, 0.5, 1, 0.4, 0.2],
                           [0.7, 0.3, 0.4, 1, 0.6],
                           [0.1, 0.2, 0.2, 0.6, 1]])
    num_samples = 1000
    mean_vector = np.zeros(5)
    X = np.random.multivariate_normal(mean=mean_vector,
                                      cov=cov_matrix,
                                      size=num_samples)
    X_i = np.column_stack((np.ones(num_samples), X[:, 0:4]))
    errors = np.random.normal(loc=0.0, scale=1.0, size=num_samples)
    Y1 = np.dot(X_i, beta1) + errors
    np_data = np.column_stack((Y1, X))
    data = pd.DataFrame(np_data, columns=['y1','x1', 'z1', 'z2', 'z3', 'z4'])

    y = ['y1'] # input names must always be enclosed in a list.
    x = ['x1']
    c= ['z1', 'z2', 'z3', 'z4']

    sim1 = OLSRobust(y=y, x=x, data=data)
    sim1.fit(controls=c, draws=1000, kfold=10)
    sim1_results = sim1.get_results()

    sim1_results.plot(specs=[['z1', 'z2', 'z3']],
                      ic='hqic',
                      figsize=(16, 12),
                      ext = 'pdf',
                      project_name = project_name,
                      )

if __name__ == "__main__":
    sim1('sim1_example')