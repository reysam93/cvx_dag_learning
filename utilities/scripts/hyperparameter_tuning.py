# %%
import numpy as np
from numpy import linalg as la
from time import perf_counter
import os
import sys
from pathlib import Path
from joblib import Parallel, delayed
import inspect
import warnings

import itertools

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.model import Nonneg_dagma, MetMulDagma, MetMulColide
import src.utils as utils

PATH = str(ROOT / 'results' / 'tuning') + os.sep
PATH_SACHS = str(ROOT / 'datasets' / 'sachs') + os.sep
SEED = 10
N_CPUS = max(1, (os.cpu_count() or 1) // 2)
np.random.seed(SEED)
os.makedirs(PATH, exist_ok=True)

DATASET = "SYNTH"  # SYNTH, SACHS

# %%
def get_lamb_value(n_nodes, n_samples, times=1):
    return np.sqrt(np.log(n_nodes) / n_samples) * times

def cartesian_product(hyperparams):
    """
    Generate all combinations of hyperparameters.
    """
    param_names = list(hyperparams.keys())
    param_values = list(hyperparams.values())
    param_combinations = [dict(zip(param_names, values)) for values in itertools.product(*param_values)]

    return param_combinations

args2str = lambda arguments: ''.join([f'{key[:3]}={val} ' for key, val in arguments.items()])

LOCAL_LIPSCHITZ_STEP_TYPES = {
    'local_lipschitz',
    'local_lipschitz_domain_backtracking',
}
DOMAIN_BACKTRACKING_STEP_TYPES = {
    'domain_backtracking',
    'local_lipschitz_domain_backtracking',
}
ADAPTIVE_STEP_TYPES = LOCAL_LIPSCHITZ_STEP_TYPES | DOMAIN_BACKTRACKING_STEP_TYPES
DEFAULT_LOCAL_LIPSCHITZ_STEPSIZE = 1e-5


def graph_seed(base_seed, graph_index):
    return int(np.random.SeedSequence([base_seed, graph_index]).generate_state(1)[0])


def completed_config_mask(success):
    success = np.asarray(success, dtype=bool)
    if success.ndim != 2:
        raise ValueError('success must have shape (n_dags, n_configs).')
    return np.all(success, axis=0)


def mask_incomplete_metrics(metrics, success):
    valid_configs = completed_config_mask(success)
    masked_metrics = {}
    for metric_key, values in metrics.items():
        values = np.asarray(values, dtype=float).copy()
        values[:, ~valid_configs] = np.nan
        masked_metrics[metric_key] = values
    return masked_metrics


def aggregate_metrics(metrics, agg_funct='mean', success=None):
    metrics = mask_incomplete_metrics(metrics, success) if success is not None else metrics
    agg_func = getattr(np, f'nan{agg_funct}')
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        return {
            metric_key: agg_func(np.asarray(value), axis=0)
            for metric_key, value in metrics.items()
        }


def normalize_step_args(fit_args):
    fit_args = fit_args.copy()
    step_type = fit_args.get('step_type', 'fixed')

    if step_type not in LOCAL_LIPSCHITZ_STEP_TYPES:
        fit_args.pop('local_lipschitz_scale', None)

    if step_type not in DOMAIN_BACKTRACKING_STEP_TYPES:
        fit_args.pop('domain_bt_factor', None)
        fit_args.pop('domain_bt_max_iters', None)
        fit_args.pop('domain_bt_tol', None)

    if step_type not in ADAPTIVE_STEP_TYPES:
        fit_args.pop('min_stepsize', None)
        fit_args.pop('max_stepsize', None)
    elif step_type in LOCAL_LIPSCHITZ_STEP_TYPES:
        fit_args['stepsize'] = None

    return fit_args


def materialize_fit_args(fit_args):
    fit_args = fit_args.copy()
    step_type = fit_args.get('step_type', 'fixed')

    if step_type in {'fixed', 'domain_backtracking'}:
        if fit_args.get('stepsize') is None:
            raise ValueError(f'{step_type} step_type requires a non-null stepsize.')
    elif fit_args.get('stepsize') is None:
        fit_args['stepsize'] = DEFAULT_LOCAL_LIPSCHITZ_STEPSIZE

    return fit_args


def build_search_configs(model_args_grid, fit_hyperparams, step_rule_grid=None):
    model_arg_combs = cartesian_product(model_args_grid)
    fit_arg_combs = cartesian_product(fit_hyperparams)
    step_rule_grid = step_rule_grid or [{}]

    configs = []
    seen = set()
    for model_args, fit_args, step_args in itertools.product(model_arg_combs, fit_arg_combs, step_rule_grid):
        fit_args_full = fit_args.copy()
        fit_args_full.update(step_args)
        label_fit_args = normalize_step_args(fit_args_full)
        config_key = (
            tuple(sorted(model_args.items())),
            tuple(sorted(label_fit_args.items())),
        )
        if config_key in seen:
            continue
        seen.add(config_key)
        configs.append({
            'model_args': model_args.copy(),
            'fit_args': materialize_fit_args(label_fit_args),
            'label_args': {**model_args, **label_fit_args},
        })
    return configs


def _unsupported_kwargs(callable_obj, kwargs):
    parameters = inspect.signature(callable_obj).parameters.values()
    if any(param.kind == inspect.Parameter.VAR_KEYWORD for param in parameters):
        return set()
    accepted = {
        param.name for param in parameters
        if param.name not in {'self', 'X'}
    }
    return set(kwargs) - accepted


def validate_search_configs(model_const, configs):
    for idx, config in enumerate(configs):
        unsupported_model_args = _unsupported_kwargs(model_const, config['model_args'])
        unsupported_fit_args = _unsupported_kwargs(model_const.fit, config['fit_args'])
        if unsupported_model_args or unsupported_fit_args:
            details = []
            if unsupported_model_args:
                details.append(f'model args: {sorted(unsupported_model_args)}')
            if unsupported_fit_args:
                details.append(f'fit args: {sorted(unsupported_fit_args)}')
            raise ValueError(
                f'Configuration {idx} is incompatible with {model_const.__name__} '
                f'({"; ".join(details)}).'
            )


def config2str(config):
    return args2str(config['label_args'])

def print_best(key, metrics, configs, agg_funct='mean', all_best=False, success=None):
    agg_metric = aggregate_metrics(metrics, agg_funct, success=success)

    finite_values = np.isfinite(agg_metric[key])
    if not np.any(finite_values):
        print(f'No finite values found for {key} (agg: {agg_funct}).')
        return

    best_value = np.min(agg_metric[key][finite_values])
    best_idxs = np.where(finite_values & np.isclose(agg_metric[key], best_value))[0]

    if not all_best:
        best_idxs = [best_idxs[0]]

    print(f'Combination{"s" if all_best else ""} with best {key} (agg: {agg_funct}):')
    for idx in best_idxs:
        print(configs[idx]['label_args'])
        print(f'shd: {agg_metric["shd"][idx]:.2f} | err: {agg_metric["err"][idx]:.4f} |' +
              f'acyc: {agg_metric["acyc"][idx]:.6f} | time: {agg_metric["time"][idx]:.2f}')


def print_best_by_step_type(key, metrics, configs, agg_funct='mean', success=None):
    agg_metric = aggregate_metrics(metrics, agg_funct, success=success)

    step_types = sorted({
        config['label_args'].get('step_type', 'fixed')
        for config in configs
    })

    print(f'Best {key} by step_type (agg: {agg_funct}):')
    for step_type in step_types:
        step_idxs = np.array([
            idx for idx, config in enumerate(configs)
            if config['label_args'].get('step_type', 'fixed') == step_type
        ])
        finite_step_idxs = step_idxs[np.isfinite(agg_metric[key][step_idxs])]
        if len(finite_step_idxs) == 0:
            print(f'{step_type}: no finite values found.')
            continue

        best_idx = finite_step_idxs[np.argmin(agg_metric[key][finite_step_idxs])]
        print(step_type)
        print(configs[best_idx]['label_args'])
        print(f'shd: {agg_metric["shd"][best_idx]:.2f} | err: {agg_metric["err"][best_idx]:.4f} |' +
              f'acyc: {agg_metric["acyc"][best_idx]:.6f} | time: {agg_metric["time"][best_idx]:.2f}')


def print_failure_summary(success, failures, configs):
    success = np.asarray(success, dtype=bool)
    failures = np.asarray(failures, dtype=object)
    failed_counts = np.sum(~success, axis=0)
    failed_configs = np.flatnonzero(failed_counts)
    print(
        f'Complete configurations: {len(configs) - len(failed_configs)}/{len(configs)}'
    )
    for idx in failed_configs:
        messages = [
            str(message) for message in failures[:, idx]
            if str(message)
        ]
        example = messages[0] if messages else 'unknown failure'
        print(
            f'  Config {idx}: failed on {failed_counts[idx]}/{success.shape[0]} '
            f'DAGs. Example: {example}'
        )



def run_grid_search_tuning(g, seed, data_p, configs, model_const, std_x, fix_lamb, thr, verb=False):
    # Create data
    if DATASET == "SACHS":
        W_true = np.load(PATH_SACHS + "sachs_A_matrix.npy")
        X = np.load(PATH_SACHS + "sachs_X.npy")
    else:
        np.random.seed(seed)
        W_true, _, X = utils.simulate_sem(**data_p)

    # X = X/np.linalg.norm(X, axis=1, keepdims=True) if std_x else X
    X = utils.standarize(X) if std_x else X
    norm_W_true = np.linalg.norm(W_true)
    W_true_bin = utils.to_bin(W_true, thr)
    M, N = X.shape

    fidelity = 1/data_p['n_samples']*la.norm(X - X @ W_true, 'fro')**2

    print(f'Graph {g+1}: Fidelity: {fidelity:.3f}')

    shd, err, acyc, runtime = [
        np.full(len(configs), np.nan, dtype=float) for _ in range(4)
    ]
    success = np.zeros(len(configs), dtype=bool)
    failures = np.full(len(configs), '', dtype=object)
    for i, config in enumerate(configs):
        t_init = perf_counter()
        try:
            args_aux = config['fit_args'].copy()
            args_aux['lamb'] = (
                args_aux['lamb']
                if fix_lamb
                else get_lamb_value(N, M, args_aux['lamb'])
            )
            model = model_const(**config['model_args'])
            if isinstance(model, MetMulColide):
                W_est, _ = model.fit(X, **args_aux)
            else:
                W_est = model.fit(X, **args_aux)

            W_est = np.asarray(W_est)
            if W_est.shape != W_true.shape:
                raise ValueError(
                    f'W_est has shape {W_est.shape}; expected {W_true.shape}.'
                )
            if not np.all(np.isfinite(W_est)):
                raise FloatingPointError('W_est contains NaN or Inf.')

            W_est_bin = utils.to_bin(W_est, thr)
            shd_i, _, _ = utils.count_accuracy(W_true_bin, W_est_bin)
            err_i = utils.compute_norm_sq_err(W_true, W_est, norm_W_true)
            acyc_i = model.dagness(W_est)
            if not np.all(np.isfinite([shd_i, err_i, acyc_i])):
                raise FloatingPointError('Computed metrics contain NaN or Inf.')

            shd[i] = shd_i
            err[i] = err_i
            acyc[i] = acyc_i
            success[i] = True
        except Exception as exc:
            failures[i] = f'{type(exc).__name__}: {exc}'
        finally:
            runtime[i] = perf_counter() - t_init

        if verb and g == 0:
            text = config2str(config)
            if success[i]:
                print(f'\t- {text}: shd {shd[i]}  -  err: {err[i]:.3f}  -  acyc: {acyc[i]:.5g}  -  time: {runtime[i]:.3f}')
            else:
                print(f'\t- {text}: FAILED - {failures[i]} - time: {runtime[i]:.3f}')

    return shd, err, acyc, runtime, success, failures

# %% [markdown]
# ## Experiment parameters

# %%
model_const = MetMulDagma

ModelArgsGrid = {
    'primal_opt': ['fista'],  # Add 'adam' here if you want to include it.
    'acyclicity': ['logdet'],
    'restart': [True],  # Only used in FISTA
}

verb = True
thr = .2
n_dags = 30 if DATASET != "SACHS" else 1
std_x = False
fix_lamb = False
N = 100
data_params = {
    'n_nodes': N,
    'n_samples': 1000, # 1000,
    'graph_type': 'er',
    'edges': 4*N,
    'edge_type': 'positive',
    'w_range': (.5, 1),
    'var': 1, # 1/np.sqrt(N),
}

### CURRENT BEST
# Combination with best err (agg: mean):
# {'primal_opt': 'fista', 'acyclicity': 'logdet', 'restart': True, 'stepsize': 1e-05, 'alpha_0': 0.01, 'rho_0': 0.01, 'beta': 1.5, 's': 1, 'lamb': 0.2, 'iters_in': 5000, 'iters_out': 50, 'tol': 1e-06, 'h_tol': 0.0001, 'step_type': 'fixed'}
# shd: 1.07 | err: 0.0069 |acyc: 0.000033 | time: 28.72

Hyperparams = {
    'stepsize': [5e-6, 1e-5, 5e-5, 1e-4],
    'alpha_0': [.01, .1, 2],
    'rho_0': [.001, .01, .05],
    'beta': [1.5],
    's': [1],
    'lamb': [.2],
    'iters_in': [500, 5000, 10000],
    'iters_out': [10, 50, 100],
    'tol': [1e-6],
    'h_tol': [1e-4],
    'step_type': [
        'fixed',
        'local_lipschitz',
    ],
    'local_lipschitz_scale': [0.75, .8, .9, 1, 1.1, 1.2, 1.5],
    'min_stepsize': [1e-12],
    'max_stepsize': [None],
    'domain_bt_factor': [.5],
    'domain_bt_max_iters': [20],
    'domain_bt_tol': [1e-12],
}

if DATASET == "SACHS":
    N_CPUS = 1

def main():
    print('CPUs employed:', N_CPUS)
    print('Looking hyperparameters for dataset', DATASET)
    # Get combinations of model initialization args, fit hyperparams, and step rules.
    search_configs = build_search_configs(ModelArgsGrid, Hyperparams)
    validate_search_configs(model_const, search_configs)
    print('Number of combinations:', len(search_configs))
    graph_seeds = np.asarray(
        [graph_seed(SEED, graph_index) for graph_index in range(n_dags)],
        dtype=np.uint32,
    )

    t_init = perf_counter()
    results = Parallel(n_jobs=N_CPUS)(delayed(run_grid_search_tuning)
                      (g, int(graph_seeds[g]), data_params, search_configs,
                       model_const, std_x, fix_lamb, thr, verb)
                      for g in range(n_dags))
    t_end = perf_counter()
    print(f'Total tuning time: {t_end - t_init:.2f}s')

    shd, err, acyc, runtime, success, failures = zip(*results)
    metrics = {'shd': shd, 'err': err, 'acyc': acyc, 'time': runtime}
    success = np.asarray(success, dtype=bool)
    failures = np.asarray(failures, dtype=str)
    masked_metrics = mask_incomplete_metrics(metrics, success)

    leg = [config2str(config) for config in search_configs]
    np.savez(
        f'{PATH}tuning_raw_metrics.npz',
        shd=np.asarray(shd),
        err=np.asarray(err),
        acyc=np.asarray(acyc),
        time=np.asarray(runtime),
        success=success,
        failures=failures,
        graph_seeds=graph_seeds,
        legends=np.asarray(leg),
    )
    print_failure_summary(success, failures, search_configs)
    utils.display_results(leg, masked_metrics, agg='mean', file_name=f'{PATH}tuning_mean')
    utils.display_results(leg, masked_metrics, agg='median', file_name=f'{PATH}tuning_med')

    print_best(
        'err', metrics, search_configs, agg_funct='mean', success=success
    )
    print_best(
        'err', metrics, search_configs, agg_funct='median', success=success
    )
    print_best(
        'shd', metrics, search_configs, agg_funct='mean',
        all_best=True, success=success
    )
    print_best_by_step_type('err', metrics, search_configs, success=success)
    print()


if __name__ == "__main__":
    main()
