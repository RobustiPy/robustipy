import numpy as np
import pandas as pd
from robustipy.models import OLSRobust


def sim1(project_name):
    np.random.seed(192735)
    beta1 = np.array([0.05, 0.1, -0.6, -0.35, 0.05,
                      0.1, 0.05, 0.1, 0.05])
    L_factor = np.array([
        [0.8,  0.2], [0.6, -0.5], [0.7,  0.1], [0.5, -0.6],
        [0.4,  0.7], [0.3, -0.4], [0.2,  0.3], [0.1, -0.2]])
    D = np.diag([0.3]*8)
    cov_matrix = L_factor.dot(L_factor.T) + D
    num_samples = 10000
    mean_vector = np.zeros(8)
    X = np.random.multivariate_normal(mean=mean_vector,
                                      cov=cov_matrix,
                                      size=num_samples)
    X_i = np.column_stack((np.ones(num_samples), X))
    errors = np.random.normal(0.0, 1.0, num_samples)
    Y1 = np.dot(X_i, beta1) + errors
    np_data = np.column_stack((Y1, X))
    data = pd.DataFrame(np_data, columns=['y1', 'x1', 'z1', 'z2',
                                          'z3', 'z4', 'z5', 'z6',
                                          'z7'])
    y = ['y1']
    x = ['x1']
    c = ['z1', 'z2', 'z3', 'z4', 'z5', 'z6', 'z7']
    sim1 = OLSRobust(y=y, x=x, data=data)
    sim1.fit(controls=c, draws=1000, kfold=10, seed=192735)
    sim1_results = sim1.get_results()
    sim1_results.plot(specs=[['z1', 'z2', 'z3']],
                      ic='hqic', figsize=(16, 16),
                      ext='pdf', project_name=project_name)
    sim1_results.summary()

if __name__ == "__main__":
    sim1('sim1_example')