import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.model import MetMulDagma
from utilities.tuning.data import make_synthetic_cases
from utilities.tuning.reporting import run_tuning


# ---------------------------------------------------------------------------
# Experiment controls
# ---------------------------------------------------------------------------
SEED = 10
N_CPUS = max(1, (os.cpu_count() or 1) // 2)
OUTPUT_DIR = ROOT / 'results' / 'tuning' / 'synthetic'

VERBOSE = True
THRESHOLD = .2
N_DAGS = 30
STANDARDIZE_X = False
FIX_LAMBDA = False
RESCALE_WEIGHTS = False

N_NODES = 200
DATA_PARAMS = {
    'n_nodes': N_NODES,
    'n_samples': 1000,
    'graph_type': 'er',
    'edges': 4 * N_NODES,
    'edge_type': 'positive',
    'w_range': (.5, 2),
    'var': 1,
}


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
    'stepsize': [5e-6, 1e-5, 5e-5, 1e-4],
    'alpha_0': [.01],
    'rho_0': [.01],
    'beta': [1.5],
    's': [1],
    'lamb': [.2, .5, 1, 2, 5, 10],
    'iters_in': [5000, 10000, 25000, 50000],
    'iters_out': [50, 100],
    'tol': [1e-6],
    'h_tol': [1e-4],
    'step_type': [
        'fixed',
    ],
    'local_lipschitz_scale': [1],
    'min_stepsize': [1e-12],
    'max_stepsize': [None],
    'domain_bt_factor': [.5],
    'domain_bt_max_iters': [20],
    'domain_bt_tol': [1e-12],
}


def main():
    np.random.seed(SEED)
    cases = make_synthetic_cases(DATA_PARAMS, N_DAGS, SEED)
    run_tuning(
        cases=cases,
        model_const=MODEL_CONST,
        model_args_grid=MODEL_ARGS_GRID,
        hyperparams=HYPERPARAMS,
        output_dir=OUTPUT_DIR,
        n_jobs=N_CPUS,
        std_x=STANDARDIZE_X,
        fix_lamb=FIX_LAMBDA,
        thr=THRESHOLD,
        rescale=RESCALE_WEIGHTS,
        verb=VERBOSE,
    )


if __name__ == '__main__':
    main()
