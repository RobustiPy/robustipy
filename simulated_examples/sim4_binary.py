import numpy as np
import pandas as pd
from robustipy.models import LRobust


def sim4(project_name):
    np.random.seed(192735)
    beta1 = np.array([0.2, 0.5, -0.4, -0.7, 0.2, 0.5, 0.2, 0.5, 0.3])
    L = np.array([
        [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.3, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.5, 0.5, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.9, 0.3, 0.4, 1.0, 0.0, 0.0, 0.0, 0.0],
        [0.3, 0.2, 0.2, 0.25, 1.0, 0.0, 0.0, 0.0],
        [0.2, 0.3, 0.1, 0.1, 0.1, 1.0, 0.0, 0.0],
        [0.2, 0.3, 0.1, 0.1, 0.1, 0.5, 1.0, 0.0],
        [0.1, 0.2, 0.3, 0.4, 0.2, 0.1, 0.2, 1.0]
    ])
    cov_matrix = L + L.T - np.diag(np.diag(L))
    num_samples = 10000
    mean_vector = np.zeros(8)
    X = np.random.multivariate_normal(mean=mean_vector, cov=cov_matrix,
                                      size=num_samples)
    X_i = np.column_stack((np.ones(num_samples), X))
    errors = np.random.normal(loc=0.0, scale=1.0, size=num_samples)
    Y1 = np.dot(X_i, beta1) + errors

    threshold = np.median(Y1)
    Y1 = (Y1 > threshold).astype(int)
    np_data = np.column_stack((Y1, X))
    data = pd.DataFrame(np_data, columns=['y1', 'x1', 'z1', 'z2', 'z3',
                                          'z4', 'z5', 'z6', 'z7'])

    y = ['y1']
    x = ['x1']
    c = ['z1', 'z2', 'z3', 'z4', 'z5', 'z6', 'z7']
    sim1 = LRobust(y=y, x=x, data=data)
    sim1.fit(controls=c, draws=1000, kfold=10, seed=192735)
    sim1_results = sim1.get_results()
    sim1_results.plot(specs=[['z1', 'z2', 'z3']],
                      ic='hqic', figsize=(16, 16),
                      ext='pdf', project_name=project_name)


if __name__ == "__main__":
    sim4('sim4_example')
