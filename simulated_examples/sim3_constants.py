import numpy as np
import pandas as pd
from robustipy.models import OLSRobust


def sim3(project_name):
    """Simulation with fixed coefficients and multiple focal regressors."""
    # 1) Setup & seeds
    np.random.seed(192735)

    # 2) Coefficients and covariance specification
    beta1 = np.array([0.2, 0.5, -0.4, -0.7, 0.2,
                      0.5, 0.2, 0.5, 0.3])
    L = np.array([
        [0.8, 0.2], [0.6, -0.5], [0.7, 0.1],
        [0.5, -0.6], [0.4, 0.7], [0.3, -0.4],
        [0.2, 0.3], [0.1, -0.2]])
    D = np.diag([5] * 8)
    cov_matrix = L.dot(L.T) + D

    # 3) Sample size and covariate generation
    num_samples = 1000
    mean_vector = np.zeros(8)
    X = np.random.multivariate_normal(mean=mean_vector,
                                      cov=cov_matrix, size=num_samples)
    X_i = np.column_stack((np.ones(num_samples), X))

    # 4) Outcome generation
    errors = np.random.normal(0.0, 1.0, num_samples)
    Y1 = np.dot(X_i, beta1) + errors

    # 5) Build DataFrame
    np_data = np.column_stack((Y1, X))
    data = pd.DataFrame(np_data, columns=['y1', 'x1', 'z1', 'z2',
                                          'z3', 'z4', 'z5', 'z6',
                                          'z7'])
    
    # 6) Specify model variables and fit robust OLS
    y = ['y1']
    x = ['x1', 'z1']     # multiple focal regressors
    z = ['z2', 'z3','z4', 'z5', 'z6', 'z7']
    sim3 = OLSRobust(y=y, x=x, data=data)
    sim3.fit(controls=z, draws=1000, kfold=10, seed=192735)

    # 7) Retrieve, plot, and summarize results
    sim3_results = sim3.get_results()
    sim3_results.plot(specs=[['z4', 'z5']],
                      ic='hqic', figsize=(16, 16),
                      figpath='../figures',
                      ext='pdf', project_name=project_name)
    sim3_results.summary()


if __name__ == "__main__":
    sim3('sim3_example')
