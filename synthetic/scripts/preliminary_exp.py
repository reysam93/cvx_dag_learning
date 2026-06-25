#!/usr/bin/env python
# coding: utf-8

#

# In[ ]:


import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ.setdefault("PYTHONWARNINGS", "ignore::FutureWarning:sklearn.linear_model._base")

import signal
import numpy as np
import pandas as pd
from numpy import linalg as la
import matplotlib.pyplot as plt
from IPython.display import display
from sklearn.metrics import f1_score
import cvxpy as cp
import random
import time
from copy import deepcopy
from datetime import datetime
from joblib import Parallel, delayed

from src.model import Nonneg_dagma, MetMulDagma, MetMulColide
import src.utils as utils

from baselines.colide import colide_ev, colide_nv
from baselines.dagma_linear import DAGMA_linear
from baselines.nonnegative_dagma_linear import NonnegativeDAGMA_linear
from baselines.golem import GOLEM_EV, GOLEM_NV, GOLEM_TF_EV, GOLEM_TF_NV
from baselines.nofears import NoFearsLinear
from baselines.notears import notears_linear
from baselines.sortnregress import VarSortNRegress, R2SortNRegress, RandomSortNRegress
from baselines.daguerreotype import DAGuerreotype

import warnings
warnings.filterwarnings("ignore", category=FutureWarning, module=r"sklearn\.linear_model\._base")

SEED = 10
N_CPUS = max(1, int(os.environ.get("N_CPUS", os.cpu_count() or 1)))
JOBLIB_VERBOSE = max(0, int(os.environ.get("JOBLIB_VERBOSE", 10)))
SELECTED_SCENARIOS = {
    scenario.strip()
    for scenario in os.environ.get("SCENARIOS", "").split(",")
    if scenario.strip()
}
LOAD = False
SAVE_RESULTS = True
RESULTS_PATH = ROOT / 'results' / 'preliminary'
os.makedirs(RESULTS_PATH, exist_ok=True)

np.random.seed(SEED)


def log_status(message):
    timestamp = datetime.now().isoformat(timespec='seconds')
    print(f'[{timestamp} pid={os.getpid()}] {message}', flush=True)


def _handle_termination(signum, frame):
    raise KeyboardInterrupt(f"Received signal {signum}; stopping experiments")


signal.signal(signal.SIGTERM, _handle_termination)
random.seed(SEED)


# In[2]:


Exps = [
    # Simple Gradient Descent
    # {'model': Nonneg_dagma, 'args': {'stepsize': 5e-3, 'alpha': 2, 's': 1, 'lamb': 1e-1, 'max_iters': 1000000, 'tol': 1e-6},
    #  'init': {'acyclicity': 'logdet', 'primal_opt': 'pgd'}, 'standarize': False, 'fix_lamb': False, 'leg': 'PGD'},

    # {'model': Nonneg_dagma, 'args': {'stepsize': 5e-3, 'alpha': 2, 's': 1, 'lamb': 1e-1, 'max_iters': 1000000, 'tol': 1e-6},
    #  'init': {'acyclicity': 'logdet', 'primal_opt': 'adam'}, 'standarize': False, 'fix_lamb': False, 'leg': 'PGD-Adam'},

    # NOMAD
    # {'model': MetMulDagma, 'args': {'stepsize': 3e-4, 'alpha_0': .01, 'rho_0': .05, 's': 1, 'lamb': .1, 'iters_in': 10000, 'step_type': 'fixed',
    #  'iters_out': 10, 'beta': 2}, 'init': {'acyclicity': 'logdet', 'primal_opt': 'adam'}, 'standarize': False,
    #  'fix_lamb': False, 'leg': 'NOMAD-adam-ORIG'},

    {'model': MetMulDagma, 'args': {'stepsize': 5e-3, 'alpha_0': .1, 'rho_0': .1, 's': 1, 'lamb': .2, 'iters_in': 5000, 'step_type': 'fixed',
     'iters_out': 10, 'beta': 1.5}, 'init': {'acyclicity': 'logdet', 'primal_opt': 'adam'}, 'standarize': False,
     'fix_lamb': False, 'leg': 'NOMAD-adam'},

    {'model': MetMulDagma, 'args': {'stepsize': 3e-4, 'alpha_0': .1, 'rho_0': .05, 's': 1, 'lamb': .5, 'iters_in': 20000, 'step_type': 'fixed',
     'iters_out': 50, 'beta': 1.5}, 'init': {'acyclicity': 'logdet', 'primal_opt': 'adam'}, 'standarize': False,
     'fix_lamb': False, 'leg': 'NOMAD-adam-200'},

    {'model': MetMulDagma, 'args': {'stepsize': 1e-5, 'alpha_0': .01, 'rho_0': .01, 's': 1, 'lamb': .2, 'iters_in': 5000, 'step_type': 'fixed',
     'iters_out': 50, 'beta': 1.5}, 'init': {'acyclicity': 'logdet', 'primal_opt': 'fista', 'restart': True}, 'standarize': False,
     'fix_lamb': False, 'leg': 'NOMAD-fista'},

    # {'model': MetMulDagma, 'args': {'stepsize': 1e-4, 'alpha_0': .05, 'rho_0': .05, 's': 1, 'lamb': .2, 'iters_in': 10000, 'step_type': 'fixed',
    #  'iters_out': 50, 'beta': 1.5}, 'init': {'acyclicity': 'logdet', 'primal_opt': 'fista', 'restart': True}, 'standarize': False,
    #  'fix_lamb': False, 'leg': 'NOMAD-fista-100'},

    {'model': MetMulDagma, 'args': {'stepsize': 5e-5, 'alpha_0': .01, 'rho_0': .01, 's': 1, 'lamb': .5, 'iters_in': 50000, 'step_type': 'fixed',
     'iters_out': 100, 'beta': 1.5}, 'init': {'acyclicity': 'logdet', 'primal_opt': 'fista', 'restart': True}, 'standarize': False,
     'fix_lamb': False, 'leg': 'NOMAD-fista-200'},

    # # COLIDE - NOMAD
    # {'model': MetMulColide, 'args': {'stepsize': 5e-5, 'alpha_0': .1, 'rho_0': .1, 's': 1, 'lamb': .1, 'iters_in': 30000,
    #  'iters_out': 10, 'beta': 1.2}, 'init': {'acyclicity': 'logdet', 'primal_opt': 'fista', 'restart': True}, 'standarize': False,
    #  'fix_lamb': False, 'leg': 'MM-Col-fista-r'},

    # {'model': MetMulColide, 'args': {'stepsize': 5e-5, 'alpha_0': .1, 'rho_0': .1, 's': 1, 'lamb': .1, 'iters_in': 30000,
    #  'iters_out': 10, 'beta': 1.2}, 'init': {'acyclicity': 'logdet', 'primal_opt': 'adam', 'restart': True}, 'standarize': False,
    #  'fix_lamb': False, 'leg': 'MM-Col-adam'},

    # {'model': MetMulColide, 'args': {'stepsize': 3e-4, 'alpha_0': .01, 'rho_0': .05, 's': 1, 'lamb': .1, 'iters_in': 20000,
    #  'iters_out': 10, 'beta': 2, 'sca_adam': True}, 'init': {'acyclicity': 'logdet', 'primal_opt': 'sca'}, 'standarize': False,
    #  'fix_lamb': False, 'leg': 'MM-Col-sca'},

    ### BASELINES ###
    # NoTears
    {'model': notears_linear, 'args': {'loss_type': 'l2', 'lambda1': .1, 'max_iter': 10}, 'standarize': False, 'leg': 'NoTears'},

    # NoFears: NOTEARS followed by the official KKTS local search
    {'model': NoFearsLinear, 'args': {'lambda1': .1, 'w_threshold': .3, 'max_iter': 100, 'h_tol': 1e-10,
     'rho_init': 1., 'rho_factor': 10., 'rho_max': 1e16, 'h_progress_rate': .25, 'w_tol': 1e-10,
     'pen_tol': 0., 'rev_edges': 'alt-full', 'minimize_z': True, 'init_no_pen': True, 'no_pen': False},
     'standarize': False, 'fix_lamb': True, 'leg': 'NoFears'},

    # DAGMA
    {'model': DAGMA_linear, 'init': {'loss_type': 'l2'}, 'args': {'lambda1': .05, 'T': 4, 's': [1.0, .9, .8, .7],
     'warm_iter': 2e4, 'max_iter': 7e4, 'lr': .0003}, 'standarize': False, 'leg': 'DAGMA'},

    # NonnegDAGMA
    {'model': NonnegativeDAGMA_linear, 'init': {'loss_type': 'l2'}, 'args': {'lambda1': .05, 'T': 4, 's': [1.0, .9, .8, .7],
     'warm_iter': 2e4, 'max_iter': 7e4, 'lr': .0003}, 'standarize': False, 'leg': 'NonDAGMA'},

    {'model': NonnegativeDAGMA_linear, 'init': {'loss_type': 'l2'}, 'args': {'lambda1': .74, 'T': 4, 's': [1.0, .9, .8, .7],
     'warm_iter': 2e4, 'max_iter': 7e4, 'lr': .0003}, 'fix_lamb': False, 'standarize': False, 'leg': 'NonDAGMA-nofix'},

    # Colide
    {'model': colide_ev, 'args': {'lambda1': .05, 'T': 4, 's': [1.0, .9, .8, .7], 'warm_iter': 2e4,
     'max_iter': 7e4, 'lr': .0003}, 'standarize': False, 'leg': 'CoLiDE-EV'},

    {'model': colide_nv, 'args': {'lambda1': .05, 'T': 4, 's': [1.0, .9, .8, .7], 'warm_iter': 2e4,
     'max_iter': 7e4, 'lr': .0003}, 'standarize': False, 'leg': 'CoLiDE-NV'},

    # GOLEM
    {'model': GOLEM_EV, 'args': {'lambda1': 2e-2, 'lambda2': 5.0, 'num_iter': 100000, 'learning_rate': 1e-3, 'w_threshold': 0.3,
     'postprocess': True, 'checkpoint': None}, 'standarize': False, 'fix_lamb': True, 'leg': 'GOLEM-EV-np'},

    {'model': GOLEM_NV, 'init': {'init_with_ev': True}, 'args': {'lambda1': 2e-3, 'lambda2': 5.0, 'lambda1_ev': 2e-2, 'lambda2_ev': 5.0,
     'num_iter': 100000, 'num_iter_ev': 100000, 'learning_rate': 1e-3, 'learning_rate_ev': 1e-3, 'w_threshold': 0.3, 'postprocess': True,
     'checkpoint': None}, 'standarize': False, 'fix_lamb': True, 'leg': 'GOLEM-NV-np'},

    # Similar performance but way slower than np implementation
    # {'model': GOLEM_TF_EV, 'args': {'lambda1': 2e-2, 'lambda2': 5.0, 'num_iter': 100000, 'learning_rate': 1e-3, 'w_threshold': 0.3,
    #  'postprocess': True, 'checkpoint': None}, 'standarize': False, 'fix_lamb': True, 'leg': 'GOLEM-EV'},

    # {'model': GOLEM_TF_NV, 'init': {'init_with_ev': True}, 'args': {'lambda1': 2e-3, 'lambda2': 5.0, 'lambda1_ev': 2e-2, 'lambda2_ev': 5.0,
    #  'num_iter': 100000, 'num_iter_ev': 100000, 'learning_rate': 1e-3, 'learning_rate_ev': 1e-3, 'w_threshold': 0.3, 'postprocess': True,
    #  'checkpoint': None}, 'standarize': False, 'fix_lamb': True, 'leg': 'GOLEM-NV'},

    # Regress
    {'model': VarSortNRegress, 'args': {'w_threshold': 0.3}, 'standarize': False, 'fix_lamb': True, 'leg': 'SortNRegress'},

    # # {'model': R2SortNRegress, 'args': {}, 'standarize': False, 'fix_lamb': True, 'leg': 'R2-SortNRegress'},

    # # # DAGuerreotype
    # # {'model': DAGuerreotype, 'init': {'structure': 'sp_map', 'sparsifier': 'l0_ber_ste', 'equations': 'linear', 'loss': 'nll_ev',
    # #  'joint': False, 'nogpu': True, 'standardize': False, 'init_theta': 'zeros', 'num_epochs': 5000, 'num_inner_iters': 200, 'lr': 1e-1,
    # #  'lr_theta': 1e-1, 'pruning_reg': 0.001, 'l2_theta': 0.0005, 'l2_eq': 0.0005}, 'args': {}, 'standarize': False, 'fix_lamb': True, 'leg': 'DAGuerreotype'}
]

# Keep model definitions and hyperparameters identical; only the input scale changes.
Exps_standardized = [{**exp, 'standarize': True} for exp in Exps]


def get_exp_by_leg(exps, leg):
    for exp in exps:
        if exp['leg'] == leg:
            return exp
    raise ValueError(f'Experiment not found: {leg}')


def copy_with_iteration_budget(exps, base_leg, reference_leg, new_leg):
    exp = deepcopy(get_exp_by_leg(exps, base_leg))
    reference = get_exp_by_leg(exps, reference_leg)
    for key in ('iters_in', 'iters_out'):
        exp['args'][key] = reference['args'][key]
    exp['leg'] = new_leg
    return exp


def add_n200_iteration_controls(exps):
    adam_control = copy_with_iteration_budget(
        exps,
        base_leg='NOMAD-adam',
        reference_leg='NOMAD-adam-200',
        new_leg='NOMAD-adam-iters200',
    )
    fista_control = copy_with_iteration_budget(
        exps,
        base_leg='NOMAD-fista',
        reference_leg='NOMAD-fista-200',
        new_leg='NOMAD-fista-iters200',
    )

    exps_n200 = []
    for exp in deepcopy(exps):
        exps_n200.append(exp)
        if exp['leg'] == 'NOMAD-adam-200':
            exps_n200.append(adam_control)
        elif exp['leg'] == 'NOMAD-fista-200':
            exps_n200.append(fista_control)
    return exps_n200


# In[3]:


def get_lamb_value(n_nodes, n_samples, times=1):
    return np.sqrt(np.log(n_nodes) / n_samples) * times


def run_parallel_exps(data_p, exps, n_dags, scenario_name, thr=.3, verb=False):
    n_jobs = max(1, min(N_CPUS, n_dags))
    log_status(
        f'RUN scenario={scenario_name} dags={n_dags} workers={n_jobs} '
        f'joblib_verbose={JOBLIB_VERBOSE}'
    )
    t_init = time.time()

    parallel = Parallel(n_jobs=n_jobs, verbose=JOBLIB_VERBOSE)
    try:
        results = parallel(
            delayed(run_exps)(g, data_p, exps, thr=thr, verb=verb)
            for g in range(n_dags)
        )
    except (KeyboardInterrupt, SystemExit):
        backend = getattr(parallel, "_backend", None)
        if backend is not None and hasattr(backend, "terminate"):
            backend.terminate()
        raise
    finally:
        backend = getattr(parallel, "_backend", None)
        if backend is not None and hasattr(backend, "terminate"):
            backend.terminate()

    elapsed_time = (time.time() - t_init)/60
    log_status(f'DONE scenario={scenario_name} elapsed_minutes={elapsed_time:.3f}')
    return results

def run_exps(g, data_p, exps, thr=.3, verb=False):
    A_true, _, X = utils.simulate_sem(**data_p)
    A_true_bin = utils.to_bin(A_true, thr)
    X_std = utils.standarize(X)
    X_scale = X.std(axis=0)
    A_true_std = A_true * X_scale[:, None] / X_scale[None, :]

    M, N = X.shape

    Z = X - X @ A_true
    fidelity = 1/data_p['n_samples']*la.norm(Z, 'fro')**2
    fidelity_std = 1/data_p['n_samples']*la.norm(X_std - X_std @ A_true_std, 'fro')**2

    Sigma_hat = la.norm(Z, axis=0) / np.sqrt(M)

    log_status(
        f'DAG {g}: fidelity={fidelity:.3f} fidelity_std={fidelity_std:.3f} '
        f'sigma_min={np.min(Sigma_hat):.4f} sigma_max={np.max(Sigma_hat):.4f}'
    )

    shd, fscore, sid_norm, err, acyc, runtime = [np.zeros(len(exps))  for _ in range(6)]
    for i, exp in enumerate(exps):
        standardized = exp.get('standarize', False)
        X_aux = X_std if standardized else X
        A_true_aux = A_true_std if standardized else A_true

        args = exp['args'].copy()
        if 'fix_lamb' in exp.keys() and not exp['fix_lamb']:
            if 'lamb' in args:
                args['lamb'] = get_lamb_value(N, M, args['lamb'])
            elif 'lambda1' in args:
                args['lambda1'] = get_lamb_value(N, M, args['lambda1'])

        if 'know_var' in exp.keys() and exp['know_var']:
            args['Sigma'] = data_p['var']

        model = None
        try:
            if exp['model'] == notears_linear:
                t_i = time.time()
                A_est = notears_linear(X_aux, **args)
                t_solved = time.time() - t_i
            else:
                model = exp['model'](**exp['init']) if 'init' in exp.keys() else exp['model']()
                t_i = time.time()
                model.fit(X_aux, **args)
                t_solved = time.time() - t_i

                A_est = model.W_est
                if getattr(model, 'cycle_repair_applied_', False):
                    log_status(
                        f'DAG {g} method={exp["leg"]} exact-DAG safeguard removed '
                        f'{model.cycle_repair_count_} edge(s): '
                        f'{model.cycle_repair_edges_}'
                    )
        except Exception as exc:
            raise RuntimeError(
                f'DAG {g} method={exp["leg"]} failed: '
                f'{type(exc).__name__}: {exc}'
            ) from exc

        A_est_bin = utils.to_bin(A_est, thr)
        try:
            shd[i], _, _, sid_norm[i] = utils.count_accuracy(
                A_true_bin,
                A_est_bin,
                compute_sid=True,
                sid_normalize=True,
            )
        except ValueError:
            shd[i], _, _ = utils.count_accuracy(A_true_bin, A_est_bin)
            sid_norm[i] = np.nan
        fscore[i] = f1_score(A_true_bin.flatten(), A_est_bin.flatten())
        err[i] = utils.compute_norm_sq_err(A_true_aux, A_est)
        acyc[i] = model.dagness(A_est) if model is not None and hasattr(model, 'dagness') else float(not utils.is_dag(A_est_bin))
        runtime[i] = t_solved

        if verb and (g % N_CPUS == 0):
            sid_text = f'{sid_norm[i]:.3f}' if np.isfinite(sid_norm[i]) else 'nan'
            log_status(
                f'DAG {g} method={exp["leg"]} standardized={standardized} '
                f'shd={shd[i]} sid={sid_text} err={err[i]:.3f} time={runtime[i]:.3f}'
            )

    return shd, fscore, sid_norm, err, acyc, runtime

def preliminary_results_prefix(scenario_name):
    return f'{RESULTS_PATH}/preliminary_{scenario_name}'


def load_preliminary_results(scenario_name):
    tables = {}
    exps_leg = None
    for agg in ('mean', 'median'):
        file_name = f'{preliminary_results_prefix(scenario_name)}_{agg}.csv'
        if not os.path.exists(file_name):
            raise FileNotFoundError(f'Results file not found: {file_name}')
        tables[agg] = pd.read_csv(file_name)
        if 'leg' not in tables[agg].columns:
            raise ValueError(f'Results file has no experiment legend column: {file_name}')

        table_exps_leg = tables[agg]['leg'].astype(str).tolist()
        if exps_leg is None:
            exps_leg = table_exps_leg
        elif table_exps_leg != exps_leg:
            raise ValueError(f'Experiment legends differ between saved tables for {scenario_name}')

        print(f'Loaded {agg} results from {file_name}')
        display(tables[agg])
    return tables, exps_leg


def run_or_load_preliminary_results(data_p, exps, n_dags, scenario_name, thr=.3, verb=False):
    if SELECTED_SCENARIOS and scenario_name not in SELECTED_SCENARIOS:
        log_status(f'SKIP scenario={scenario_name} selected={sorted(SELECTED_SCENARIOS)}')
        return None, None, None

    standardized_count = sum(exp.get('standarize', False) for exp in exps)
    mode = 'load' if LOAD else 'run'
    log_status(
        f'START scenario={scenario_name} mode={mode} nodes={data_p["n_nodes"]} '
        f'samples={data_p["n_samples"]} edges={data_p["edges"]} dags={n_dags} '
        f'standardized_methods={standardized_count}/{len(exps)}'
    )

    if LOAD:
        tables, exps_leg = load_preliminary_results(scenario_name)
        log_status(f'LOADED scenario={scenario_name} methods={len(exps_leg)}')
        return None, tables, exps_leg

    results = run_parallel_exps(
        data_p,
        exps,
        n_dags,
        scenario_name=scenario_name,
        thr=thr,
        verb=verb,
    )
    shd, fscore, sid_norm, err, acyc, runtime = zip(*results)
    metrics = {'shd': shd, 'fscore': fscore, 'sid_norm': sid_norm, 'err': err, 'acyc': acyc, 'time': runtime}

    exps_leg = [exp['leg'] for exp in exps]
    file_prefix = preliminary_results_prefix(scenario_name) if SAVE_RESULTS else None
    utils.display_results(exps_leg, metrics, agg='mean', file_name=f'{file_prefix}_mean' if file_prefix else None)
    utils.display_results(exps_leg, metrics, agg='median', file_name=f'{file_prefix}_median' if file_prefix else None)
    log_status(f'SAVED scenario={scenario_name} prefix={file_prefix}')
    return metrics, None, exps_leg



# ## CASE 0 - N=200, Weights - [0.5, 2]

# In[4]:


N = 200
SCENARIO_NAME = 'er4_N200_var1'

n_dags = 50
verb = True
data_params = {
    'n_nodes': N,
    'n_samples': 1000, # 1000,
    'graph_type': 'er',
    'edges': 4*N,
    'edge_type': 'positive',
    'w_range': (.5, 2),  # (.5, 1)
    'var': 1
}
Exps_N200 = add_n200_iteration_controls(Exps)
metrics, tables, exps_leg = run_or_load_preliminary_results(data_params, Exps_N200, n_dags, SCENARIO_NAME, thr=.3, verb=verb)


# In[5]:


# Results are displayed by run_or_load_preliminary_results.


# ## CASE 1 - N=50, Homocedastic

# In[6]:


N = 50
SCENARIO_NAME = 'er4_N50_var1'

n_dags = 50
verb = True
data_params = {
    'n_nodes': N,
    'n_samples': 1000, # 1000,
    'graph_type': 'er',
    'edges': 4*N,
    'edge_type': 'positive',
    'w_range': (.5, 1),  # (.5, 1)
    'var': 1
}
metrics, tables, exps_leg = run_or_load_preliminary_results(data_params, Exps, n_dags, SCENARIO_NAME, thr=.3, verb=verb)


# In[7]:


# Results are displayed by run_or_load_preliminary_results.


# ## CASE 2 - N=100, Homocedastic

# In[8]:


N = 100
SCENARIO_NAME = 'er4_N100_var1'

n_dags = 50
verb = True
data_params = {
    'n_nodes': N,
    'n_samples': 1000, # 1000,
    'graph_type': 'er',
    'edges': 4*N,
    'edge_type': 'positive',
    'w_range': (.5, 1),  # (.5, 1)
    'var': 1
}
metrics, tables, exps_leg = run_or_load_preliminary_results(data_params, Exps, n_dags, SCENARIO_NAME, thr=.3, verb=verb)


# In[9]:


# Results are displayed by run_or_load_preliminary_results.


# ## CASE 3 - N=100, Heterocedastic

# In[10]:


N = 100
SCENARIO_NAME = 'er4_N100_hetero_var'

var = np.random.uniform(low=0.5, high=5.0, size=N)

n_dags = 50
verb = True
data_params = {
    'n_nodes': N,
    'n_samples': 1000, # 1000,
    'graph_type': 'er',
    'edges': 4*N,
    'edge_type': 'positive',
    'w_range': (.5, 1),  # (.5, 1)
    'var': var**2
}
metrics, tables, exps_leg = run_or_load_preliminary_results(data_params, Exps, n_dags, SCENARIO_NAME, thr=.3, verb=verb)


# In[11]:


# Results are displayed by run_or_load_preliminary_results.


# ## CASE 4 - N=100, ER4, Standardized

# In[ ]:


N = 100
SCENARIO_NAME = 'er4_N100_var1_standardized'

n_dags = 50
verb = True
data_params = {
    'n_nodes': N,
    'n_samples': 1000, # 1000,
    'graph_type': 'er',
    'edges': 4*N,
    'edge_type': 'positive',
    'w_range': (.5, 1),  # (.5, 1)
    'var': 1
}
metrics, tables, exps_leg = run_or_load_preliminary_results(data_params, Exps_standardized, n_dags, SCENARIO_NAME, thr=.3, verb=verb)


# ## CASE 5 - N=100, ER2, Homocedastic

# In[12]:


N = 100
SCENARIO_NAME = 'er2_N100_var1'

n_dags = 50
verb = True
data_params = {
    'n_nodes': N,
    'n_samples': 1000, # 1000,
    'graph_type': 'er',
    'edges': 2*N,
    'edge_type': 'positive',
    'w_range': (.5, 1),  # (.5, 1)
    'var': 1
}
metrics, tables, exps_leg = run_or_load_preliminary_results(data_params, Exps, n_dags, SCENARIO_NAME, thr=.3, verb=verb)


# In[13]:


# Results are displayed by run_or_load_preliminary_results.
