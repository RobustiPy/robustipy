import numpy as np
import pandas as pd
from robustipy.models import OLSRobust

def sim2(project_name):
    np.random.seed(192735)
    beta_mean = [0.9, 0.4, -0.7, -0.1, 0.2, 0.3, -0.2, 0.5, 0.1]
    beta_std = 2
    L_factor = np.array([[0.8,  0.2], [0.6, -0.5],
                         [0.7,  0.1], [0.5, -0.6],
                         [0.4,  0.7], [0.3, -0.4],
                         [0.2,  0.3], [0.1, -0.2]])
    D = np.diag([0.3] * 8)
    cov_matrix = L_factor.dot(L_factor.T) + D
    num_samples, num_groups = 10, 1000
    mean_vec = np.zeros(8)
    X_list, Y_list, group_list = [], [], []
    for group in range(1, num_groups + 1):
        beta = np.random.normal(beta_mean, beta_std, len(beta_mean))
        mi, ma, factor = np.argmin(beta), np.argmax(beta), 5
        beta[mi] = np.random.normal(beta[mi], beta_std * factor)
        beta[ma] = np.random.normal(beta[ma], beta_std * factor)
        X = np.random.multivariate_normal(mean_vec,
                                          cov_matrix, num_samples) \
            + np.random.normal(0, 5, (num_samples, 8))
        X_i = np.column_stack((np.ones(num_samples), X))
        errors = np.random.normal(0, 20, num_samples)
        Y = np.dot(X_i, beta) + errors
        X_list.append(X)
        Y_list.append(Y)
        group_list.extend([group] * num_samples)
    data = pd.DataFrame(np.column_stack((np.concatenate(Y_list),
                                         np.vstack(X_list))),
                        columns=['y', 'x1', 'z1', 'z2', 'z3',
                                 'z4', 'z5', 'z6', 'z7'])
    data['group'] = group_list
    sim = OLSRobust(y=['y'], x=['x1'], data=data)
    sim.fit(controls=['z1', 'z2', 'z3', 'z4', 'z5', 'z6', 'z7'],
            group='group',
            #draws=1000,
            #kfold=10,
            rescale_y=True,
            rescale_z=True,
            seed=192735)
    results = sim.get_results()
    results.plot(specs=[['z1', 'z2', 'z3']], ic='hqic',
                 figsize=(16, 16), ext='pdf', project_name=project_name)
    results.summary()

if __name__ == '__main__':
    sim2('sim2_example')