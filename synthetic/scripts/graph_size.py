# %%
from time import perf_counter
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score
from joblib import Parallel, delayed
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import src.utils as utils
from src.model import MetMulDagma, MetMulColide

from baselines.colide import colide_ev
from baselines.dagma_linear import DAGMA_linear
from baselines.notears import notears_linear


PATH = str(ROOT / 'results' / 'size') + os.sep
SAVE = True
SEED = 10
N_CPUS = os.cpu_count()
np.random.seed(SEED)

# %% [markdown]
# ## Auxiliary functions

# %%
# Experiment function
def get_lamb_value(n_nodes, n_samples, times=1):
    return np.sqrt(np.log(n_nodes) / n_samples) * times

def run_size_exp(g, data_p, Sizes, exps, thr=.2, verb=False):
    shd, tpr, fdr, fscore, err, acyc, runtime, dag_count = [np.zeros((len(Sizes), len(exps)))  for _ in range(8)]
    for i, n_nodes in enumerate(Sizes):
        if g % N_CPUS == 0:
            print(f'Graph: {g+1}, Nodes: {n_nodes}')

        # Create data
        data_p_aux = data_p.copy()
        data_p_aux['n_nodes'] = n_nodes
        data_p_aux['edges'] *= n_nodes
        data_p_aux['var'] = 1/np.sqrt(n_nodes) if data_p_aux['var'] is None else data_p_aux['var']
        data_p_aux['n_samples'] = 10*n_nodes if data_p_aux['n_samples'] is None else data_p_aux['n_samples']

        W_true, _, X = utils.simulate_sem(**data_p_aux)
        X_std = utils.standarize(X)
        W_true_bin = utils.to_bin(W_true, thr)
        norm_W_true = np.linalg.norm(W_true)

        for j, exp in enumerate(exps):
            X_aux = X_std if 'standarize' in exp.keys() and exp['standarize'] else X


            arg_aux = exp['args'].copy()
            if 'adapt_lamb' in exp.keys() and exp['adapt_lamb']:
                if 'lamb' in arg_aux.keys():
                    arg_aux['lamb'] = get_lamb_value(n_nodes, data_p_aux['n_samples'], arg_aux['lamb'])
                elif 'lambda1' in arg_aux.keys():
                    arg_aux['lambda1'] = get_lamb_value(n_nodes, data_p_aux['n_samples'], arg_aux['lambda1'])

            if exp['model'] == notears_linear:
                t_init = perf_counter()
                W_est = notears_linear(X_aux, **arg_aux)
                t_end = perf_counter()
            else:
                model = exp['model'](**exp['init']) if 'init' in exp.keys() else exp['model']()
                t_init = perf_counter()
                model.fit(X_aux, **arg_aux)
                t_end = perf_counter()

                W_est = model.W_est

            if np.isnan(W_est).any():
                W_est = np.zeros_like(W_est)
                W_est_bin = np.zeros_like(W_est)
            else:
                W_est_bin = utils.to_bin(W_est, thr)

            shd[i,j], tpr[i,j], fdr[i,j] = utils.count_accuracy(W_true_bin, W_est_bin)
            shd[i,j] /= n_nodes
            fscore[i,j] = f1_score(W_true_bin.flatten(), W_est_bin.flatten())
            err[i,j] = utils.compute_norm_sq_err(W_true, W_est, norm_W_true)
            acyc[i,j] = model.dagness(W_est) if hasattr(model, 'dagness') else 1
            runtime[i,j] = t_end - t_init
            dag_count[i,j] += 1 if utils.is_dag(W_est_bin) else 0

            if verb and (g % N_CPUS == 0):
                print(f'\t\t-{exp["leg"]}: shd {shd[i,j]}  -  err: {err[i,j]:.3f}  -  time: {runtime[i,j]:.3f}')

    return shd, tpr, fdr, fscore, err, acyc, runtime, dag_count

# %%
n_dags = 50
thr = .2
verb = True
Sizes = np.array( [50, 75, 100, 250, 500] )

# DEFINE EXPERIMENTS
Exps = [
  # MM + Cvx DAGMA
  {'model': MetMulDagma, 'args': {'stepsize': 3e-4, 'alpha_0': .01, 'rho_0': .05, 's': 1, 'lamb': 2e-1,
   'iters_in': 10000, 'iters_out': 10, 'beta': 2}, 'init': {'primal_opt': 'adam', 'acyclicity': 'logdet'},
   'adapt_lamb': True, 'standarize': False, 'fmt': 'o-', 'leg': 'MM-adam'},

  {'model': MetMulDagma, 'args': {'stepsize': 1e-5, 'alpha_0': .01, 'rho_0': .01, 's': 1, 'lamb': .2, 'iters_in': 5000,
     'iters_out': 50, 'beta': 1.5}, 'init': {'acyclicity': 'logdet', 'primal_opt': 'fista', 'restart': True},
     'adapt_lamb': True, 'standarize': False, 'fmt': 'o--', 'leg': 'MM-fista'},

  # MM + Cvx COLIDE
  {'model': MetMulColide, 'args': {'stepsize': 3e-4, 'alpha_0': .01, 'rho_0': .05, 's': 1, 'lamb': .1, 'iters_in': 20000,
     'iters_out': 10, 'beta': 2, 'sca_adam': True}, 'init': {'acyclicity': 'logdet', 'primal_opt': 'sca'},
     'adapt_lamb': True, 'standarize': False, 'fmt': 's-', 'leg': 'MM-Col-sca'},

  # DAGMA
  {'model': DAGMA_linear, 'init': {'loss_type': 'l2'}, 'args': {'lambda1': .05, 'T': 4, 's': [1.0, .9, .8, .8],
   'warm_iter': 2e4, 'max_iter': 7e4, 'lr': .0003}, 'standarize': False, 'fmt': '^-', 'leg': 'DAGMA'},
]

# %% [markdown]
# ## 4N Edges - weights (.5, 1)

# %%
# NOTE: larger beta required for larger weights
data_p = {
    'edges': 4,
    'edge_type': 'positive',
    'w_range': (.5, 1),
    'var': 1,
    'n_samples': 5000, # 1000
}

# %% [markdown]
# ### Estimate graphs

# %% [markdown]
# #### Erdos Renyi

# %%
data_p['graph_type'] = 'er'

shd, tpr, fdr, fscore, err, acyc, runtime, dag_count =\
      [np.zeros((n_dags, len(Sizes), len(Exps)))  for _ in range(8)]

print('CPUs employed:', N_CPUS)

t_init = perf_counter()
results_er4 = Parallel(n_jobs=N_CPUS)(delayed(run_size_exp)
                                  (g, data_p, Sizes, Exps, thr, verb) for g in range(n_dags))
t_end = perf_counter()
print(f'----- Solved in {(t_end-t_init)/60:.3f} minutes -----')

shd_er4, tpr_er4, fdr_er4, fscore_er4, err_er4, acyc_er4, runtime_er4, dag_count_er4 = zip(*results_er4)

# %%
if SAVE:
    file_name = PATH + f'size_ERgraph_{data_p["edges"]}N_{data_p["w_range"][1]}w'
    np.savez(file_name, shd=shd_er4, tpr=tpr_er4, fdr=fdr_er4, fscore=fscore_er4, err=err_er4,
             acyc=acyc_er4, runtime=runtime_er4, dag_count=dag_count_er4, exps=Exps,
             xvals=Sizes)
    print('SAVED in file:', file_name)

# %% [markdown]
# #### Scale free

# %%
Exps = [
  # MM + Cvx DAGMA
  {'model': MetMulDagma, 'args': {'stepsize': 3e-4, 'alpha_0': .01, 'rho_0': .05, 's': 1, 'lamb': 2e-1,
   'iters_in': 10000, 'iters_out': 10, 'beta': 2}, 'init': {'primal_opt': 'adam', 'acyclicity': 'logdet'},
   'adapt_lamb': True, 'standarize': False, 'fmt': 'o-', 'leg': 'MM-adam'},

  {'model': MetMulDagma, 'args': {'stepsize': 1e-5, 'alpha_0': .01, 'rho_0': .01, 's': 1, 'lamb': .2, 'iters_in': 50000,
     'iters_out': 50, 'beta': 1.5}, 'init': {'acyclicity': 'logdet', 'primal_opt': 'fista', 'restart': True},
     'adapt_lamb': True, 'standarize': False, 'fmt': 'o--', 'leg': 'MM-fista'},

  # MM + Cvx COLIDE
  {'model': MetMulColide, 'args': {'stepsize': 3e-4, 'alpha_0': .01, 'rho_0': .05, 's': 1, 'lamb': .1, 'iters_in': 20000,
     'iters_out': 10, 'beta': 2, 'sca_adam': True}, 'init': {'acyclicity': 'logdet', 'primal_opt': 'sca'},
     'adapt_lamb': True, 'standarize': False, 'fmt': 's-', 'leg': 'MM-Col-sca'},

  # DAGMA
  {'model': DAGMA_linear, 'init': {'loss_type': 'l2'}, 'args': {'lambda1': .05, 'T': 4, 's': [1.0, .9, .8, .8],
   'warm_iter': 2e4, 'max_iter': 7e4, 'lr': .0003}, 'standarize': False, 'fmt': '^-', 'leg': 'DAGMA'},
]

# %%
data_p['graph_type'] = 'sf'

shd, tpr, fdr, fscore, err, acyc, runtime, dag_count =\
      [np.zeros((n_dags, len(Sizes), len(Exps)))  for _ in range(8)]

print('CPUs employed:', N_CPUS)

t_init = perf_counter()
results_sf4 = Parallel(n_jobs=N_CPUS)(delayed(run_size_exp)
                                  (g, data_p, Sizes, Exps, thr, verb) for g in range(n_dags))
t_end = perf_counter()
print(f'----- Solved in {(t_end-t_init)/60:.3f} minutes -----')

shd_sf4, tpr_sf4, fdr_sf4, fscore_sf4, err_sf4, acyc_sf4, runtime_sf4, dag_count_sf4 = zip(*results_sf4)

# %%
if SAVE:
    file_name = PATH + f'size_SFgraph_{data_p["edges"]}N_{data_p["w_range"][1]}w'
    np.savez(file_name, shd=shd_sf4, tpr=tpr_sf4, fdr=fdr_sf4, fscore=fscore_sf4, err=err_sf4,
             acyc=acyc_sf4, runtime=runtime_sf4, dag_count=dag_count_sf4, exps=Exps,
             xvals=Sizes)
    print('SAVED in file:', file_name)

# %% [markdown]
# ### Plot results

# %%
### Weights (.5, 1.5) ###
# # Load Data
# file_name = "./results/size/size_ERgraph_4N_1w.npz"
# data = np.load(file_name, allow_pickle=True)
# Exps, shd_er4, tpr_er4, fdr_er4, fscore_er4, err_er4, acyc_er4, runtime_er4, dag_count_er4, Sizes = data['exps'], data['shd'], data['tpr'], \
#     data['fdr'], data['fscore'], data['err'], data['acyc'], data['runtime'], data['dag_count'], data['xvals']

# file_name = "./results/size/size_SFgraph_4N_1w.npz"
# data = np.load(file_name, allow_pickle=True)
# Exps, shd_sf4, tpr_sf4, fdr_sf4, fscore_sf4, err_sf4, acyc_sf4, runtime_sf4, dag_count_sf4, Sizes = data['exps'], data['shd'], data['tpr'], \
#     data['fdr'], data['fscore'], data['err'], data['acyc'], data['runtime'], data['dag_count'], data['xvals']

skip = []  # [0, 3]
Exps_er = [{'leg': exp['leg'] + '-ER', 'fmt': exp['fmt'][0] + '-'} for exp in Exps]
Exps_sf = [{'leg': exp['leg'] + '-SF', 'fmt': exp['fmt'][0] + '--'} for exp in Exps]
Exps_joint = Exps_er + Exps_sf

shd = np.concatenate((shd_er4, shd_sf4), axis=2)
err = np.concatenate((err_er4, err_sf4), axis=2)

if SAVE:
    agg_shd = np.mean(shd, axis=0)
    utils.data_to_csv(f'{PATH}size_shd_mean.csv', Exps_joint, Sizes, agg_shd)
    std_shd = np.std(err, axis=0)
    shd_std_up = agg_shd + std_shd
    utils.data_to_csv(f'{PATH}size_shd_std_up.csv', Exps_joint, Sizes, shd_std_up)
    shd_std_down = agg_shd - std_shd
    utils.data_to_csv(f'{PATH}size_shd_std_down.csv', Exps_joint, Sizes, shd_std_down)

fig, axes = plt.subplots(1, 2, figsize=(10, 5))
utils.plot_data(axes[0], shd, Exps_joint, Sizes, 'Number of nodes', 'Normalized SDH', skip,
                agg='mean', deviation='prctile', alpha=0.25)
utils.plot_data(axes[1], err, Exps_joint, Sizes, 'Number of nodes', 'Fro Error', skip,
                agg='mean', deviation='prctile', alpha=0.25, plot_func='loglog')
plt.tight_layout()

fig, axes = plt.subplots(1, 2, figsize=(10, 5))
utils.plot_data(axes[0], shd, Exps_joint, Sizes, 'Number of nodes', 'Normalized SDH', skip,
                agg='median', deviation='prctile', alpha=0.25)
utils.plot_data(axes[1], err, Exps_joint, Sizes, 'Number of nodes', 'Fro Error', skip,
                agg='median', deviation='prctile', alpha=0.25, plot_func='loglog')
plt.tight_layout()
# plt.savefig(PATH + f'size.png')

# %%
# Load Data
# file_name = "./results/size/size_SFgraph_4N_1w.npz"
data = np.load(file_name, allow_pickle=True)
Exps, shd_er4, tpr_er4, fdr_er4, fscore_er4, err_er4, acyc_er4, runtime_er4, dag_count_er4, Sizes = data['exps'], data['shd'], data['tpr'], \
    data['fdr'], data['fscore'], data['err'], data['acyc'], data['runtime'], data['dag_count'], data['xvals'],

skip = [] # [2]

utils.plot_all_metrics(shd_er4, tpr_er4, fdr_er4, fscore_er4, err_er4, acyc_er4, runtime_er4, dag_count_er4, Sizes, Exps,
                 skip_idx=skip, agg='median', xlabel='Number of nodes')
# utils.plot_all_metrics(shd_er4, tpr_er4, fdr_er4, fscore_er4, err_er4, acyc_er4, runtime_er4, dag_count_er4, Sizes, Exps,
#                  skip_idx=skip, agg='median', xlabel='Number of nodes')


# %%
# # Load Data
# file_name = "./results/size/size_SFgraph_4N.npz"
# data = np.load(file_name, allow_pickle=True)
# Exps, shd_sf4, tpr_sf4, fdr_sf4, fscore_sf4, err_sf4, acyc_sf4, runtime_sf4, dag_count_sf4, Sizes = data['exps'], data['shd'], data['tpr'], \
#     data['fdr'], data['fscore'], data['err'], data['acyc'], data['runtime'], data['dag_count'], data['xvals'],

skip = [] # [2]

utils.plot_all_metrics(shd_sf4, tpr_sf4, fdr_sf4, fscore_sf4, err_sf4, acyc_sf4, runtime_sf4, dag_count_sf4, Sizes, Exps,
                 skip_idx=skip, agg='median', xlabel='Number of nodes')
