#!/usr/bin/env python
# coding: utf-8

# 

# In[1]:


import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import signal
import numpy as np
from numpy import linalg as la
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score
import cvxpy as cp
import random
import time
from joblib import Parallel, delayed

from src.model import Nonneg_dagma, MetMulDagma, MetMulColide
import src.utils as utils

from baselines.colide import colide_ev, colide_nv
from baselines.dagma_linear import DAGMA_linear
from baselines.nonnegative_dagma_linear import NonnegativeDAGMA_linear
from baselines.golem import GOLEM_EV, GOLEM_TF_EV, GOLEM_TF_NV
from baselines.notears import notears_linear
from baselines.sortnregress import VarSortNRegress, R2SortNRegress, RandomSortNRegress
from baselines.daguerreotype import DAGuerreotype

import warnings
warnings.filterwarnings("ignore", category=FutureWarning, module="sklearn.linear_model")

SEED = 10
N_CPUS = max(1, int(os.environ.get("N_CPUS", os.cpu_count() or 1)))
SAVE_RESULTS = True
RESULTS_PATH = './results/preliminary'
os.makedirs(RESULTS_PATH, exist_ok=True)

np.random.seed(SEED)
random.seed(SEED)


# In[ ]:


def _handle_termination(signum, frame):
    raise KeyboardInterrupt(f"Received signal {signum}; stopping experiments")


signal.signal(signal.SIGTERM, _handle_termination)


Exps = [
    # Simple Gradient Descent
    # {'model': Nonneg_dagma, 'args': {'stepsize': 5e-3, 'alpha': 2, 's': 1, 'lamb': 1e-1, 'max_iters': 1000000, 'tol': 1e-6},
    #  'init': {'acyclicity': 'logdet', 'primal_opt': 'pgd'}, 'standarize': False, 'fix_lamb': False, 'leg': 'PGD'},

    # {'model': Nonneg_dagma, 'args': {'stepsize': 5e-3, 'alpha': 2, 's': 1, 'lamb': 1e-1, 'max_iters': 1000000, 'tol': 1e-6},
    #  'init': {'acyclicity': 'logdet', 'primal_opt': 'adam'}, 'standarize': False, 'fix_lamb': False, 'leg': 'PGD-Adam'},
    
    # NOMAD
    {'model': MetMulDagma, 'args': {'stepsize': 3e-4, 'alpha_0': .01, 'rho_0': .05, 's': 1, 'lamb': .1, 'iters_in': 10000, 'step_type': 'fixed',
     'iters_out': 10, 'beta': 2}, 'init': {'acyclicity': 'logdet', 'primal_opt': 'adam'}, 'standarize': False,
     'fix_lamb': False, 'leg': 'NOMAD-adam'},

    {'model': MetMulDagma, 'args': {'stepsize': 1e-5, 'alpha_0': .01, 'rho_0': .01, 's': 1, 'lamb': .2, 'iters_in': 5000, 'step_type': 'fixed',
     'iters_out': 50, 'beta': 1.5}, 'init': {'acyclicity': 'logdet', 'primal_opt': 'fista', 'restart': True}, 'standarize': False,
     'fix_lamb': False, 'leg': 'NOMAD-fista'},


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
    
    # DAGMA
    {'model': DAGMA_linear, 'init': {'loss_type': 'l2'}, 'args': {'lambda1': .05, 'T': 4, 's': [1.0, .9, .8, .8],
     'warm_iter': 2e4, 'max_iter': 7e4, 'lr': .0003}, 'standarize': False, 'leg': 'DAGMA'},

    {'model': NonnegativeDAGMA_linear, 'init': {'loss_type': 'l2'}, 'args': {'lambda1': .05, 'T': 4, 's': [1.0, .9, .8, .8],
     'warm_iter': 2e4, 'max_iter': 7e4, 'lr': .0003}, 'standarize': False, 'leg': 'NonDAGMA'},

    # Colide
    {'model': colide_ev, 'args': {'lambda1': .05, 'T': 4, 's': [1.0, .9, .8, .7], 'warm_iter': 2e4,
     'max_iter': 7e4, 'lr': .0003}, 'standarize': False, 'leg': 'CoLiDE-EV'},
    
    {'model': colide_nv, 'args': {'lambda1': .05, 'T': 4, 's': [1.0, .9, .8, .7], 'warm_iter': 2e4,
     'max_iter': 7e4, 'lr': .0003}, 'standarize': False, 'leg': 'CoLiDE-NV'},

    # GOLEM
    {'model': GOLEM_EV, 'args': {'lambda1': 2e-2, 'lambda2': 5.0, 'num_iter': 100000, 'learning_rate': 1e-3, 'w_threshold': 0.3,
     'postprocess': True, 'checkpoint': None}, 'standarize': False, 'fix_lamb': True, 'leg': 'GOLEM-EV-np'},

    {'model': GOLEM_TF_EV, 'args': {'lambda1': 2e-2, 'lambda2': 5.0, 'num_iter': 100000, 'learning_rate': 1e-3, 'w_threshold': 0.3,
     'postprocess': True, 'checkpoint': None}, 'standarize': False, 'fix_lamb': True, 'leg': 'GOLEM-EV'},
     
    {'model': GOLEM_TF_NV, 'init': {'init_with_ev': True}, 'args': {'lambda1': 2e-3, 'lambda2': 5.0, 'lambda1_ev': 2e-2, 'lambda2_ev': 5.0,
     'num_iter': 100000, 'num_iter_ev': 100000, 'learning_rate': 1e-3, 'learning_rate_ev': 1e-3, 'w_threshold': 0.3, 'postprocess': True,
     'checkpoint': None}, 'standarize': False, 'fix_lamb': True, 'leg': 'GOLEM-NV'},

    # Regress
    {'model': VarSortNRegress, 'args': {'w_threshold': 0.3}, 'standarize': False, 'fix_lamb': True, 'leg': 'SortNRegress'},

    # {'model': R2SortNRegress, 'args': {}, 'standarize': False, 'fix_lamb': True, 'leg': 'R2-SortNRegress'},

    # # DAGuerreotype
    # {'model': DAGuerreotype, 'init': {'structure': 'sp_map', 'sparsifier': 'l0_ber_ste', 'equations': 'linear', 'loss': 'nll_ev',
    #  'joint': False, 'nogpu': True, 'standardize': False, 'init_theta': 'zeros', 'num_epochs': 5000, 'num_inner_iters': 200, 'lr': 1e-1,
    #  'lr_theta': 1e-1, 'pruning_reg': 0.001, 'l2_theta': 0.0005, 'l2_eq': 0.0005}, 'args': {}, 'standarize': False, 'fix_lamb': True, 'leg': 'DAGuerreotype'}
]


# In[3]:


def get_lamb_value(n_nodes, n_samples, times=1):
    return np.sqrt(np.log(n_nodes) / n_samples) * times 


def run_parallel_exps(data_p, exps, n_dags, thr=.2, verb=False):
    n_jobs = max(1, min(N_CPUS, n_dags))
    print('CPUs employed:', n_jobs)
    t_init = time.time()

    parallel = Parallel(n_jobs=n_jobs)
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

    ellapsed_time = (time.time() - t_init)/60
    print(f'----- Solved in {ellapsed_time:.3f} minutes -----')
    return results


def run_exps(g, data_p, exps, thr=.2, verb=False):
    A_true, _, X = utils.simulate_sem(**data_p)
    A_true_bin = utils.to_bin(A_true, thr)
    X_std = utils.standarize(X)

    M, N = X.shape

    Z = X - X @ A_true
    fidelity = 1/data_p['n_samples']*la.norm(Z, 'fro')**2
    fidelity_norm = 1/data_p['n_samples']*la.norm(X_std - X_std @ A_true, 'fro')**2

    Sigma_hat = la.norm(Z, axis=0) / np.sqrt(M)

    print(f'{g}: Fidelity: {fidelity:.3f}  -  Fidelity (norm): {fidelity_norm:.3f}' +
          f'  -  Max Sigma: {np.max(Sigma_hat):.4f}  -  Min Sigma: {np.min(Sigma_hat):.4f}')

    shd, fscore, sid_norm, err, acyc, runtime = [np.zeros(len(exps))  for _ in range(6)]
    for i, exp in enumerate(exps):
        X_aux = X_std if 'standarize' in exp.keys() and exp['standarize'] else X

        args = exp['args'].copy()
        if 'fix_lamb' in exp.keys() and not exp['fix_lamb']:
            args['lamb'] = get_lamb_value(N, M, args['lamb'])

        if 'know_var' in exp.keys() and exp['know_var']:
            args['Sigma'] = data_p['var']

        model = None
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
        err[i] = utils.compute_norm_sq_err(A_true, A_est)
        acyc[i] = model.dagness(A_est) if model is not None and hasattr(model, 'dagness') else float(not utils.is_dag(A_est_bin))
        runtime[i] = t_solved

        if verb and (g % N_CPUS == 0):
            sid_text = f'{sid_norm[i]:.3f}' if np.isfinite(sid_norm[i]) else 'nan'
            print(f'	-{g+1}: {exp["leg"]}: shd {shd[i]}  -  sid {sid_text}  -  err: {err[i]:.3f}  -  time: {runtime[i]:.3f}')

    return shd, fscore, sid_norm, err, acyc, runtime


# ## CASE 1 - N=50, Homocedastic

# In[ ]:


N = 50
SCENARIO_NAME = 'er4_N50_var1'

n_dags = 25
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
results = run_parallel_exps(data_params, Exps, n_dags, thr=.2, verb=verb)

# Extract results
shd, fscore, sid_norm, err, acyc, runtime = zip(*results)
metrics = {'shd': shd, 'fscore': fscore, 'sid_norm': sid_norm, 'err': err, 'acyc': acyc, 'time': runtime}


# In[ ]:


exps_leg = [exp['leg'] for exp in Exps]
file_prefix = f'{RESULTS_PATH}/preliminary_{SCENARIO_NAME}' if SAVE_RESULTS else None
utils.display_results(exps_leg, metrics, agg='mean', file_name=f'{file_prefix}_mean' if file_prefix else None)
utils.display_results(exps_leg, metrics, agg='median', file_name=f'{file_prefix}_median' if file_prefix else None)


# ## CASE 2 - N=100, Homocedastic

# In[ ]:


N = 100
SCENARIO_NAME = 'er4_N100_var1'

n_dags = 25
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
results = run_parallel_exps(data_params, Exps, n_dags, thr=.2, verb=verb)

# Extract results
shd, fscore, sid_norm, err, acyc, runtime = zip(*results)
metrics = {'shd': shd, 'fscore': fscore, 'sid_norm': sid_norm, 'err': err, 'acyc': acyc, 'time': runtime}


# In[ ]:


exps_leg = [exp['leg'] for exp in Exps]
file_prefix = f'{RESULTS_PATH}/preliminary_{SCENARIO_NAME}' if SAVE_RESULTS else None
utils.display_results(exps_leg, metrics, agg='mean', file_name=f'{file_prefix}_mean' if file_prefix else None)
utils.display_results(exps_leg, metrics, agg='median', file_name=f'{file_prefix}_median' if file_prefix else None)


# ## CASE 3 - N=100, Heterocedastic

# In[ ]:


N = 100
SCENARIO_NAME = 'er4_N100_hetero_var'

var = np.random.uniform(low=0.5, high=5.0, size=N)

n_dags = 25
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
results = run_parallel_exps(data_params, Exps, n_dags, thr=.2, verb=verb)

# Extract results
shd, fscore, sid_norm, err, acyc, runtime = zip(*results)
metrics = {'shd': shd, 'fscore': fscore, 'sid_norm': sid_norm, 'err': err, 'acyc': acyc, 'time': runtime}


# In[ ]:


exps_leg = [exp['leg'] for exp in Exps]
file_prefix = f'{RESULTS_PATH}/preliminary_{SCENARIO_NAME}' if SAVE_RESULTS else None
utils.display_results(exps_leg, metrics, agg='mean', file_name=f'{file_prefix}_mean' if file_prefix else None)
utils.display_results(exps_leg, metrics, agg='median', file_name=f'{file_prefix}_median' if file_prefix else None)


# ## CASE 4 - N=100, ER2, Homocedastic

# In[ ]:


N = 100
SCENARIO_NAME = 'er2_N100_var1'

n_dags = 25
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
results = run_parallel_exps(data_params, Exps, n_dags, thr=.2, verb=verb)

# Extract results
shd, fscore, sid_norm, err, acyc, runtime = zip(*results)
metrics = {'shd': shd, 'fscore': fscore, 'sid_norm': sid_norm, 'err': err, 'acyc': acyc, 'time': runtime}


# In[ ]:


exps_leg = [exp['leg'] for exp in Exps]
file_prefix = f'{RESULTS_PATH}/preliminary_{SCENARIO_NAME}' if SAVE_RESULTS else None
utils.display_results(exps_leg, metrics, agg='mean', file_name=f'{file_prefix}_mean' if file_prefix else None)
utils.display_results(exps_leg, metrics, agg='median', file_name=f'{file_prefix}_median' if file_prefix else None)
