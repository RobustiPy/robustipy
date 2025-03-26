import os
import time
import signal
import numpy as np
import pandas as pd
import matplotlib as mpl
from robustipy.models import OLSRobust, LRobust
import multiprocessing

PROJECT_NAME = 'time_profiler'
SEED = 192735
BETA1 = np.array([0.05, 0.1, -0.6, -0.35, 0.05, 0.1, 0.05, 0.1, 0.05])
L_MATRIX = np.array([[0.8,  0.2], [0.6, -0.5], [0.7,  0.1], [0.5, -0.6],
                     [0.4,  0.7], [0.3, -0.4], [0.2,  0.3], [0.1, -0.2]])
D_DIAG, MEAN_VECTOR, NUM_SAMPLES = np.diag([0.3] * 8), np.zeros(8), 10000
Y_VARS, X_VARS = ['y1'], ['x1']
CONTROL_VARS = ['z1', 'z2', 'z3', 'z4', 'z5', 'z6', 'z7']
FOLDS_LIST = [2, 5, 20, 25, 10]  # For default we use FOLDS_LIST[-1]

NUM_RUNS = 5
START_VAL, END_VAL, NUM_POINTS = 10, 10000, 25
log_sequence = sorted(set(map(int, np.logspace(np.log2(START_VAL),
                                               np.log2(END_VAL),
                                               num=NUM_POINTS, base=2))))
control_sets = [CONTROL_VARS[:k] for k in range(3, len(CONTROL_VARS) + 1)]


def load_completed(result_file):
    """
    Reads the CSV file if it exists and returns a set of tuples representing
    completed iterations: (control_set_index, draws, run_number, folds).
    """
    if os.path.exists(result_file):
        df = pd.read_csv(result_file)
        completed = set()
        for _, row in df.iterrows():
            key = (int(row['Control_Set_Index']),
                   int(row['Draws']),
                   int(row['Run_Number']),
                   int(row['Folds']))
            completed.add(key)
        return completed
    else:
        return set()


def runner(control_index, run, c_array, draws, estimator, folds):
    """
    Runs a single iteration of the model fitting and saves the result.
    """
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
    run_time = time.time() - start_time
    saver(control_index, run, c_array, draws, estimator, folds, run_time)


def saver(control_index, run, c_array, draws, estimator, folds, run_time):
    """
    Saves the parameters and runtime of the current iteration into a CSV.
    The unique iteration is recorded by the following columns:
    Control_Set_Index, Draws, Run_Number, Folds.
    """
    filename = f'{PROJECT_NAME}_{estimator}_results.csv'
    write_header = not os.path.exists(filename)
    result_dict = {
        'Control_Set_Index': control_index,
        'Control_Set_Length': len(c_array),
        'Draws': draws,
        'Run_Number': run,
        'Folds': folds,
        'Time_Taken_s': run_time
    }
    pd.DataFrame([result_dict]).to_csv(filename, mode='a', index=False, header=write_header)


def generate_data():
    """
    Generates the dataset based on the provided covariance structure.
    """
    cov = L_MATRIX @ L_MATRIX.T + D_DIAG
    X = np.random.multivariate_normal(mean=MEAN_VECTOR, cov=cov, size=NUM_SAMPLES)
    Y = (np.column_stack((np.ones(NUM_SAMPLES), X)) @ BETA1 +
         np.random.normal(0.0, 1.0, NUM_SAMPLES))
    return pd.DataFrame(np.column_stack((Y, X)),
                        columns=['y1', 'x1'] + CONTROL_VARS)


def time_profiler(estimator):
    """
    Iterates over all combinations of control sets, draws, and runs.
    For each iteration, it first checks if the combination (as defined by
    control_set_index, draws, run_number, and folds) is already saved in the CSV.
    If so, it skips that iteration.
    """
    global data
    np.random.seed(SEED)
    data = generate_data()
    if estimator == 'LR':
        data['y1'] = (data['y1'] > np.median(data['y1'])).astype(int)
    result_file = f'{PROJECT_NAME}_{estimator}_results.csv'
    completed = load_completed(result_file)
    for control_index, c_array in enumerate(control_sets):
        for draws in log_sequence:
            # Run NUM_RUNS iterations with the default folds (last element)
            for run in range(1, NUM_RUNS + 1):
                key = (control_index, draws, run, FOLDS_LIST[-1])
                if key in completed:
                    continue
                runner(control_index, run, c_array, draws, estimator, FOLDS_LIST[-1])
            # Run a single iteration for each alternative fold value (using run=1)
            for folds in FOLDS_LIST[:-1]:
                key = (control_index, draws, 1, folds)
                if key in completed:
                    continue
                runner(control_index, 1, c_array, draws, estimator, folds)


def main():
    """
    Main routine that runs the time profiling for each estimator.
    """
    for estimator in ['OLS', 'LR']:
        time_profiler(estimator)


if __name__ == "__main__":
    segfault_signal = signal.SIGSEGV
    while True:
        p = multiprocessing.Process(target=main)
        p.start()
        p.join()
        if p.exitcode is not None and p.exitcode < 0:
            if abs(p.exitcode) == segfault_signal:
                print(f"Segmentation fault detected (exit code: {p.exitcode}). Restarting...")
                time.sleep(1)  # Wait briefly before restarting.
                continue
        break
