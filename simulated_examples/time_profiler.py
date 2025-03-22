import time, numpy as np, pandas as pd, matplotlib as mpl
from robustipy.models import OLSRobust, LRobust
from tqdm import tqdm

PROJECT_NAME = 'sim1_example'
SEED = 192735
BETA1 = np.array([0.2, 0.5, -0.4, -0.7, 0.2, 0.5, 0.2, 0.5, 0.3])
L_MATRIX = np.array([[0.8, 0.2], [0.6, -0.5], [0.7, 0.1], [0.5, -0.6],
                     [0.4, 0.7], [0.3, -0.4], [0.2, 0.3], [0.1, -0.2]])
D_DIAG, MEAN_VECTOR, NUM_SAMPLES = np.diag([0.3] * 8), np.zeros(8), 10000
Y_VARS, X_VARS = ['y1'], ['x1']
CONTROL_VARS, FOLDS_LIST = (['z1', 'z2', 'z3', 'z4', 'z5', 'z6', 'z7'],
                            [2, 5, 20, 25, 10])
NUM_RUNS, START_VAL, END_VAL, NUM_POINTS = 10, 10, 10000, 25
log_sequence = sorted(set(map(int, np.logspace(np.log2(START_VAL),
                                               np.log2(END_VAL),
                                               num=NUM_POINTS, base=2))))
control_sets = [CONTROL_VARS[:k] for k in range(3, len(CONTROL_VARS) + 1)]

def runner(run, c_array, draws, estimator, folds):
    start_time = time.time()
    if estimator == 'OLS':
        model = OLSRobust(y=Y_VARS, x=X_VARS, data=data)
    elif estimator == 'LR':
        model = LRobust(y=Y_VARS, x=X_VARS, data=data)
    model.fit(controls=c_array, draws=draws, kfold=folds, seed=SEED, n_cpu=20)
    results = model.get_results()
    results.plot(specs=[['z1']], ic='hqic', figsize=(16, 16),
                 ext='pdf', project_name=PROJECT_NAME)
    results.summary()
    mpl.pyplot.close()
    saver(run, folds, draws, len(c_array), time.time() - start_time, estimator)

def saver(run, folds, draws, c_len, run_time, estimator):
    pd.DataFrame([{'Run Number': run, 'Folds': folds, 'Draws': draws,
                   'Length of c': c_len, 'Time Taken (s)': run_time
    }]).to_csv(f'{PROJECT_NAME}_{estimator}_simulation_results.csv', index=False)


def generate_data():
    cov = L_MATRIX @ L_MATRIX.T + D_DIAG
    X = np.random.multivariate_normal(mean=MEAN_VECTOR, cov=cov, size=NUM_SAMPLES)
    Y = (np.column_stack((np.ones(NUM_SAMPLES), X)) @ BETA1 +
         np.random.normal(0.0, 1.0, NUM_SAMPLES))
    return pd.DataFrame(np.column_stack((Y, X)),
                        columns=['y1', 'x1'] + CONTROL_VARS)

def time_profiler(estimator):
    global data
    np.random.seed(SEED)
    data = generate_data()
    if estimator == 'LR':
        data['y1'] = (data['y1'] > np.median(data['y1'])).astype(int)
    for c_array in control_sets:
        for draws in log_sequence:
            for run in range(1, NUM_RUNS + 1):
                runner(run, c_array, draws, estimator, FOLDS_LIST[-1])
        for folds in FOLDS_LIST[:-1]:
            runner(run, c_array, draws, estimator, folds)

if __name__ == "__main__":
    time_profiler('OLS')
    time_profiler('LR')