import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.model import MetMulDagma
from utilities.tuning.data import make_sachs_cases
from utilities.tuning.reporting import run_tuning


# ---------------------------------------------------------------------------
# Experiment controls
# ---------------------------------------------------------------------------
SEED = 10
N_CPUS = 1
PATH_SACHS = ROOT / 'datasets' / 'sachs'
OUTPUT_DIR = ROOT / 'results' / 'tuning' / 'sachs' / 'fista'

VERBOSE = True
THRESHOLD = .3
STANDARDIZE_X_OPTIONS = [False]  # Use [False], [True], or [False, True].
FIX_LAMBDA = False
RESCALE_WEIGHTS = False
SACHS_TARGET = 'strong'  # 'strong' or 'all'
SAVE_W_EST = True
# Retain all configurations in the two best distinct SHD levels. Use 1 to
# retain only the best level or None to retain every successful W_est. Keeping
# all matrices is safest for an exhaustive threshold sweep, but two levels are
# a much smaller and usually sufficient exploratory archive.
W_EST_SHD_LEVELS = 2


### SETTINGS TO TEST:
##### RESCALE_WEIGHTS = False
##### RESCALE_WEIGHTS = TRUE 
##### STANDARDIZE_X_OPTIONS = False
##### STANDARDIZE_X_OPTIONS = True
##### Fista vs Adam

# ---------------------------------------------------------------------------
# Model and grid controls
# ---------------------------------------------------------------------------
MODEL_CONST = MetMulDagma

MODEL_ARGS_GRID = {
    'primal_opt': ['fista'],
    'acyclicity': ['logdet'],
    'restart': [True],
}

HYPERPARAMS = {
    'stepsize': [5e-6, 7.5e-6, 1e-5, 2.5e-5, 5e-5],
        'alpha_0': [.001, .01, 0.5],
        'rho_0': [.001, .01, .05],
        'beta': [1.2, 1.5, 2, 5],
        's': [.7, .8, .9, 1, 1.1, 1.2],
        'lamb': [.01, .05],
        'iters_in': [500, 1000, 5000, 10000],
        'iters_out': [5, 10, 50],
        'tol': [1e-6],
        'h_tol': [1e-4],
        'step_type': ['fixed'],
    'local_lipschitz_scale': [1],
    'min_stepsize': [1e-12],
    'max_stepsize': [None],
    'domain_bt_factor': [.5],
    'domain_bt_max_iters': [20],
    'domain_bt_tol': [1e-12],
}


def main():
    np.random.seed(SEED)
    cases = make_sachs_cases(PATH_SACHS, target=SACHS_TARGET)
    for standardize_x in STANDARDIZE_X_OPTIONS:
        std_label = 'std' if standardize_x else 'raw'
        output_dir = OUTPUT_DIR / std_label
        print(f'Running Sachs tuning with standardize_x={standardize_x}')
        run_tuning(
            cases=cases,
            model_const=MODEL_CONST,
            model_args_grid=MODEL_ARGS_GRID,
            hyperparams=HYPERPARAMS,
            output_dir=output_dir,
            n_jobs=N_CPUS,
            std_x=standardize_x,
            fix_lamb=FIX_LAMBDA,
            thr=THRESHOLD,
            rescale=RESCALE_WEIGHTS,
            verb=VERBOSE,
            save_w_est=SAVE_W_EST,
            w_est_shd_levels=W_EST_SHD_LEVELS,
        )


if __name__ == '__main__':
    main()
