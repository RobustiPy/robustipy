import numpy as np
import pandas as pd
from robustipy.models import LRobust

def sim4(project_name):
    """Binary outcome simulation using logistic-style robustness (LRobust)."""
    # 1) Setup & seeds
    np.random.seed(192735)

    # 2) Coefficients and covariance specification
    beta1 = np.array([0.05, 0.1, -0.6, -0.35, 0.05,
                      0.1, 0.05, 0.1, 0.05])
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

    # 3) Draw covariates and latent index
    num_samples = 10000
    mean_vector = np.zeros(8)
    X = np.random.multivariate_normal(mean=mean_vector,
                                      cov=cov_matrix,
                                      size=num_samples)
    X_i = np.column_stack((np.ones(num_samples), X))
    errors = np.random.normal(0.0, 1, num_samples)
    Y1 = np.dot(X_i, beta1) + errors

    # 4) Threshold latent variable into a binary target
    threshold = np.median(Y1)
    Y1 = (Y1 > threshold).astype(int)

    # 5) Build DataFrame
    np_data = np.column_stack((Y1, X))
    data = pd.DataFrame(np_data, columns=['y1','x1','z1','z2',
                                          'z3','z4','z5','z6','z7'])

    
    # 6) Specify model variables and fit LRobust
    y = ['y1']
    x = ['x1']
    z = ['z1','z2','z3','z4','z5','z6','z7']
    sim4 = LRobust(y=y, x=x, data=data)
    sim4.fit(controls=z, draws=1000, kfold=10, seed=192735)

    # 7) Retrieve, plot, and summarize results
    sim4_results = sim4.get_results()
    sim4_results.plot(specs=[['z1'], ['z2','z4'],
                             ['z3','z5']],
                      ic='hqic', figsize=(16,16),
                      figpath='../figures',
                      ext='pdf', project_name=project_name)
    sim4_results.summary()

if __name__ == "__main__":
    sim4('sim4_example')
