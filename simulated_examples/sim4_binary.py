import numpy as np
import pandas as pd
from robustipy.models import LRobust

def sim4(project_name):
    np.random.seed(192735)

    # Adjusted beta1: smaller and mixed-sign coefficients
    beta1 = np.array([0.05, 0.1, -0.6, -0.35, 0.05, 0.1, 0.05, 0.1, 0.05])

    # Construct covariance matrix using a two-factor model
    L_factor = np.array([
        [0.8,  0.2],
        [0.6, -0.5],
        [0.7,  0.1],
        [0.5, -0.6],
        [0.4,  0.7],
        [0.3, -0.4],
        [0.2,  0.3],
        [0.1, -0.2]
    ])
    D = np.diag([0.3]*8)
    cov_matrix = L_factor.dot(L_factor.T) + D

    # Reduced sample size → wider confidence intervals
    num_samples = 10000
    mean_vector = np.zeros(8)
    X = np.random.multivariate_normal(mean=mean_vector, cov=cov_matrix, size=num_samples)
    X_i = np.column_stack((np.ones(num_samples), X))

    # Increased noise → wider confidence intervals
    errors = np.random.normal(0.0, 1, num_samples)
    Y1 = np.dot(X_i, beta1) + errors

    # Binary outcome: median split
    threshold = np.median(Y1)
    Y1 = (Y1 > threshold).astype(int)

    np_data = np.column_stack((Y1, X))
    data = pd.DataFrame(np_data, columns=['y1','x1','z1','z2','z3','z4','z5','z6','z7'])

    y = ['y1']
    x = ['x1']
    c = ['z1','z2','z3','z4','z5','z6','z7']
    sim4 = LRobust(y=y, x=x, data=data)
    sim4.fit(controls=c, draws=1000, kfold=10, seed=192735)

    sim4_results = sim4.get_results()
    sim4_results.plot(specs=[['z1'], ['z2','z4'],
                             ['z3','z5']],
                      ic='hqic', figsize=(16,16),
                      ext='pdf', project_name=project_name)
    sim4_results.summary()

if __name__ == "__main__":
    sim4('sim4_example')
