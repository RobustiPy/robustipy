import os
import time
import sys
import queue
import contextlib

import numpy as np
import pandas as pd
from robustipy.models import OLSRobust, LRobust
import multiprocessing
from tqdm.auto import tqdm

PROJECT_NAME = 'time_profiler'
SEED = 192735
N_CPU = 24
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
RESULT_DIR = os.path.join(REPO_ROOT, 'data', PROJECT_NAME)
BETA1 = np.array([0.05, 0.1, -0.6, -0.35, 0.05, 0.1, 0.05, 0.1, 0.05])
L_MATRIX = np.array([[0.8,  0.2], [0.6, -0.5], [0.7,  0.1], [0.5, -0.6],
                     [0.4,  0.7], [0.3, -0.4], [0.2,  0.3], [0.1, -0.2]])
D_DIAG, MEAN_VECTOR, NUM_SAMPLES = np.diag([0.3] * 8), np.zeros(8), 10000
Y_VARS, X_VARS = ['y1'], ['x1']
CONTROL_VARS = ['z1', 'z2', 'z3', 'z4', 'z5', 'z6', 'z7']
FOLDS_LIST = [2, 5, 15, 20, 25, 10]  # For default we use FOLDS_LIST[-1]

NUM_RUNS = 10
START_VAL, END_VAL, NUM_POINTS = 10, 10000, 25
log_sequence = [int(START_VAL * ((END_VAL / START_VAL) ** (i / (NUM_POINTS - 1))))
                for i in range(NUM_POINTS)]
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


def result_path(estimator):
    os.makedirs(RESULT_DIR, exist_ok=True)
    return os.path.join(RESULT_DIR, f'{PROJECT_NAME}_{estimator}_results.csv')


def runner(control_index, run, c_array, draws, estimator, folds):
    """
    Runs a single iteration of the model fitting and saves the result.
    """
    start_time = time.time()
    if estimator == 'OLS':
        model = OLSRobust(y=Y_VARS, x=X_VARS, data=data)
    elif estimator == 'LR':
        model = LRobust(y=Y_VARS, x=X_VARS, data=data)
    model.fit(
        controls=c_array,
        draws=draws,
        kfold=folds,
        seed=SEED,
        n_cpu=N_CPU,
        compute_shap=False,
    )
    run_time = time.time() - start_time
    saver(control_index, run, c_array, draws, estimator, folds, run_time)


def saver(control_index, run, c_array, draws, estimator, folds, run_time):
    """
    Saves the parameters and runtime of the current iteration into a CSV.
    The unique iteration is recorded by the following columns:
    Control_Set_Index, Draws, Run_Number, Folds.
    """
    filename = result_path(estimator)
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


def iter_jobs(estimator):
    for control_index, c_array in enumerate(control_sets):
        for run in range(1, NUM_RUNS + 1):
            for draws in log_sequence:
                key = (control_index, draws, run, FOLDS_LIST[-1])
                yield key, control_index, run, c_array, draws, estimator, FOLDS_LIST[-1]
            for folds in FOLDS_LIST[:-1]:
                key = (control_index, draws, run, folds)
                yield key, control_index, run, c_array, draws, estimator, folds


def count_total_jobs():
    return sum(1 for estimator in ['OLS', 'LR'] for _ in iter_jobs(estimator))


def count_completed_jobs():
    completed_count = 0
    for estimator in ['OLS', 'LR']:
        completed = load_completed(result_path(estimator))
        completed_count += sum(1 for key, *_ in iter_jobs(estimator) if key in completed)
    return completed_count


def time_profiler(estimator, progress_queue=None):
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
    result_file = result_path(estimator)
    completed = load_completed(result_file)
    for key, control_index, run, c_array, draws, estimator, folds in iter_jobs(estimator):
        if key in completed:
            continue
        runner(control_index, run, c_array, draws, estimator, folds)
        completed.add(key)
        if progress_queue is not None:
            progress_queue.put(1)


def main(progress_queue=None):
    """
    Main routine that runs the time profiling for each estimator.
    """
    for estimator in ['OLS', 'LR']:
        time_profiler(estimator, progress_queue=progress_queue)


def child_main(progress_queue):
    with open(os.devnull, 'w') as devnull:
        with contextlib.redirect_stdout(devnull), contextlib.redirect_stderr(devnull):
            try:
                main(progress_queue=progress_queue)
            except Exception:
                sys.exit(1)


def drain_progress_queue(progress_queue, progress_bar):
    drained = 0
    while True:
        try:
            drained += progress_queue.get_nowait()
        except queue.Empty:
            break
    if drained:
        progress_bar.update(drained)


if __name__ == "__main__":
    total_jobs = count_total_jobs()
    initial_completed = count_completed_jobs()
    with tqdm(
        total=total_jobs,
        initial=initial_completed,
        desc="Profiler jobs",
        unit="job",
        dynamic_ncols=True,
    ) as progress_bar:
        while progress_bar.n < total_jobs:
            progress_queue = multiprocessing.Queue()
            p = multiprocessing.Process(target=child_main, args=(progress_queue,))
            p.start()
            while p.is_alive():
                drain_progress_queue(progress_queue, progress_bar)
                time.sleep(0.25)
            p.join()
            drain_progress_queue(progress_queue, progress_bar)

            completed_now = count_completed_jobs()
            if completed_now > progress_bar.n:
                progress_bar.update(completed_now - progress_bar.n)

            if p.exitcode == 0:
                break
            time.sleep(1)
