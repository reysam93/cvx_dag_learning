seed = 10

exp_name = "noco_vvo0"
# exp_name = "noco_vvo_SMOKE"
HETERO_SIGMA_RANGE = (.5, 5.)
VAR_RATIOS = [.3,.5]

SAVE = True
LOAD = not SAVE
PATH = "results/nonneg_colide/vary_var_outliers/"

n_samples = 1000
n_nodes = 100
node_ord_type = "rand"
n_dags = 100
# var_range = [ 1. ,  2.5,  4. ,  5.5,  7. ,  8.5, 10. ]
var_range = [1.,10.]
thr = .2
verb = True

# ------------------------------------------------------------------------------------------------

import os, sys
import pandas as pd
from pathlib import Path
from IPython.display import display

# ROOT = Path(__file__).resolve().parents[2]
ROOT = '.'
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import random
import signal
from datetime import datetime
from time import perf_counter

import matplotlib
# if __name__ == "__main__":
#     matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from joblib import Parallel, delayed
from sklearn.metrics import f1_score
import warnings

# Methods without Sigma leave NaN columns in err_sig; silence the nan-aggregation warnings.
for _msg in ("All-NaN slice encountered", "Mean of empty slice", "Degrees of freedom <= 0 for slice"):
    warnings.filterwarnings("ignore", message=_msg, category=RuntimeWarning)

import utils
from model import MetMulDagma, MetMulColide

from baselines.colide import colide_ev, colide_nv
from baselines.dagma_linear import DAGMA_linear

SEED = seed
N_CPUS = max(1, int(os.environ.get("N_CPUS", os.cpu_count() or 1)))
JOBLIB_VERBOSE = max(0, int(os.environ.get("JOBLIB_VERBOSE", 5)))
SELECTED_SCENARIOS = {
    scenario.strip()
    for scenario in os.environ.get("SCENARIOS", "").split(",")
    if scenario.strip()
}
SMOKE_TEST = os.environ.get("SMOKE_TEST", "0") not in ("", "0", "false", "False")
N_DAGS_OVERRIDE = os.environ.get("N_DAGS")

np.random.seed(SEED)
random.seed(SEED)  # networkx graph generators use python's random module
os.makedirs(PATH, exist_ok=True)

def log_status(message):
    timestamp = datetime.now().isoformat(timespec="seconds")
    print(f"[{timestamp} pid={os.getpid()}] {message}", flush=True)


def _handle_termination(signum, frame):
    raise KeyboardInterrupt(f"Received signal {signum}; stopping experiments")


signal.signal(signal.SIGTERM, _handle_termination)


def get_lamb_value(n_nodes, n_samples, times=1):
    return np.sqrt(np.log(n_nodes) / n_samples) * times


def seed_task(g, i=0):
    """
    Seed the numpy and python RNGs for DAG `g` and x-value index `i` inside the worker.
    Module-level seeds are re-executed by every joblib worker that imports this module,
    which made all workers generate the same DAG; per-task seeds give independent and
    reproducible data regardless of the worker assignment. They also neutralize the
    baselines that reset the global numpy RNG in their constructor (colide/dagma seed=0).
    """
    seed = (SEED * 1_000_003 + g * 1_009 + i) % (2 ** 32)
    np.random.seed(seed)
    random.seed(seed)

# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
METRIC_NAMES = ("shd", "tpr", "fdr", "fscore", "err", "rel_err", "err_sig", "rel_err_sig",
                "acyc", "runtime", "dag_count")


def compute_sigma_errors(sigma_true, sigma_est):
    """
    Errors of the estimated noise std. `sigma_est` may be a scalar (EV methods) or a
    vector (NV methods); it is broadcast to the N nodes. `err_sig` normalizes both
    vectors before comparing them (same scale-invariant criterion as the Frobenius
    error of W), `rel_err_sig` is the plain relative squared error.
    """
    sigma_true = np.asarray(sigma_true, dtype=float).ravel()
    sigma_est = np.broadcast_to(np.asarray(sigma_est, dtype=float).ravel(), sigma_true.shape)
    if not np.all(np.isfinite(sigma_est)):
        return np.nan, np.nan
    err_sig = utils.compute_norm_sq_err(sigma_true, sigma_est)
    rel_err_sig = np.linalg.norm(sigma_true - sigma_est) ** 2 / np.linalg.norm(sigma_true) ** 2
    return err_sig, rel_err_sig


def fit_model(exp, X, args):
    """Fit one method and return (model, W_est, Sigma_est or None, runtime)."""
    model = exp["model"](**exp["init"]) if "init" in exp else exp["model"]()
    t_init = perf_counter()
    out = model.fit(X, **args)
    runtime = perf_counter() - t_init

    W_est = model.W_est
    Sigma_est = None
    if exp.get("sigma", False):
        if isinstance(out, tuple) and len(out) == 2:
            Sigma_est = out[1]
        elif hasattr(model, "sig_est"):
            Sigma_est = model.sig_est
        elif hasattr(model, "Sigma"):
            Sigma_est = model.Sigma
    return model, W_est, Sigma_est, runtime


def run_var_exp(g, data_p, var_range, exps, node_ord_type:str="rand", thr=.2, verb=False):
    assert node_ord_type in [None, "rand", "order", "reverse"], f"Invalid node ordering {node_ord_type}."

    if node_ord_type is None:
        node_ord_type = "rand"
    
    node_order = np.arange(data_p["n_nodes"])
    if node_ord_type=='reverse':
        node_order = np.flip(node_order)
    permute = node_ord_type=='rand'

    metrics = {name: np.zeros((len(var_range), len(VAR_RATIOS), len(exps))) for name in METRIC_NAMES}
    metrics["err_sig"][:] = np.nan
    metrics["rel_err_sig"][:] = np.nan

    var_base = np.broadcast_to(np.asarray(data_p["var"], dtype=float), data_p["n_nodes"])
    # node_order = np.random.permutation(data_p["n_nodes"])
    # node_order = np.arange(data_p["n_nodes"])
    node_order = np.arange(data_p["n_nodes"])
    # sigma_true = np.sqrt(np.broadcast_to(np.asarray(data_p["var"], dtype=float), data_p["n_nodes"]))

    for i, v in enumerate(var_range):
        var = var_base.copy()
        
        for k, vr in enumerate(VAR_RATIOS):
            if g % N_CPUS == 0:
                log_status(f"Graph: {g + 1}, var: {v}, ratio: {vr}")
            num_diff = round( data_p["n_nodes"] * vr )
            idx = node_order[:num_diff]
            var[idx] = v
            sigma_true = np.sqrt(np.broadcast_to(np.asarray(var, dtype=float), len(var)))

            seed_task(g, i)
            data_p_aux = data_p.copy()
            data_p_aux["var"] = var
            data_p_aux["permute"] = permute

            W_true, _, X = utils.simulate_sem(**data_p_aux)
            X_std = utils.standarize(X)
            W_true_bin = utils.to_bin(W_true, thr)
            norm_W_true = np.linalg.norm(W_true)

            for j, exp in enumerate(exps):
                X_aux = X_std if exp.get("standarize", False) else X

                arg_aux = exp["args"].copy()
                if exp.get("adapt_lamb", False):
                    if "lamb" in arg_aux:
                        arg_aux["lamb"] = get_lamb_value(data_p["n_nodes"], data_p["n_samples"], arg_aux["lamb"])
                    elif "lambda1" in arg_aux:
                        arg_aux["lambda1"] = get_lamb_value(data_p["n_nodes"], data_p["n_samples"], arg_aux["lambda1"])

                try:
                    model, W_est, Sigma_est, runtime = fit_model(exp, X_aux, arg_aux)
                except Exception as exc:
                    raise RuntimeError(
                        f"DAG {g} var={v} method={exp['leg']} failed: "
                        f"{type(exc).__name__}: {exc}"
                    ) from exc

                if np.isnan(W_est).any():
                    W_est = np.zeros_like(W_est)
                    W_est_bin = np.zeros_like(W_est)
                else:
                    W_est_bin = utils.to_bin(W_est, thr)

                metrics["shd"][i, k, j], metrics["tpr"][i, k, j], metrics["fdr"][i, k, j] = \
                    utils.count_accuracy(W_true_bin, W_est_bin)
                metrics["fscore"][i, k, j] = f1_score(W_true_bin.flatten(), W_est_bin.flatten())
                metrics["err"][i, k, j] = utils.compute_norm_sq_err(W_true, W_est, norm_W_true)
                metrics["rel_err"][i, k, j] = np.linalg.norm(W_true - W_est, "fro") ** 2 / norm_W_true ** 2
                if Sigma_est is not None:
                    metrics["err_sig"][i, k, j], metrics["rel_err_sig"][i, k, j] = \
                        compute_sigma_errors(sigma_true, Sigma_est)
                metrics["acyc"][i, k, j] = model.dagness(W_est) if hasattr(model, "dagness") else 1
                metrics["runtime"][i, k, j] = runtime
                metrics["dag_count"][i, k, j] += 1 if utils.is_dag(W_est_bin) else 0

                if verb and (g % N_CPUS == 0):
                    sig_text = (f"  -  err_sig: {metrics['err_sig'][i, k, j]:.4f}"
                                if Sigma_est is not None else "")
                    log_status(
                        f"\t-{exp['leg']}: shd {metrics['shd'][i, k, j]}  -  err: {metrics['err'][i, k, j]:.4f}"
                        f"{sig_text}  -  time: {runtime:.2f}"
                    )

    return tuple(metrics[name] for name in METRIC_NAMES)


# ---------------------------------------------------------------------------
# Saving / loading
# ---------------------------------------------------------------------------
def var_results_prefix(scenario_name):
    return f"{PATH}var_{scenario_name}"

def save_var_results(file_prefix, metrics, exps, var_range, scenario_name):
    metrics = dict(zip(METRIC_NAMES, metrics))
    np.savez(file_prefix, exps=exps, xvals=var_range, scenario=scenario_name, **metrics)
    log_status(f"SAVED in file: {file_prefix}.npz")

    def save_stats(metric, tag, prctiles=True):
        data = metrics[metric]
        utils.data_to_csv(f"{file_prefix}_{tag}_mean.csv", exps, var_range, np.nanmean(data, axis=0))
        utils.data_to_csv(f"{file_prefix}_{tag}_std.csv", exps, var_range, np.nanstd(data, axis=0))
        if prctiles:
            utils.data_to_csv(f"{file_prefix}_{tag}_med.csv", exps, var_range, np.nanmedian(data, axis=0))
            utils.data_to_csv(f"{file_prefix}_{tag}_prctile25.csv", exps, var_range,
                              np.nanpercentile(data, 25, axis=0))
            utils.data_to_csv(f"{file_prefix}_{tag}_prctile75.csv", exps, var_range,
                              np.nanpercentile(data, 75, axis=0))

def load_var_results(scenario_name):
    file_name = f"{var_results_prefix(scenario_name)}.npz"
    data = np.load(file_name, allow_pickle=True)
    log_status(f"Loaded var results from {file_name}")
    metrics = tuple(data[name] for name in METRIC_NAMES)
    return (*metrics, data["exps"].tolist(), data["xvals"])


def run_or_load_var_results(data_p, var_range, exps, n_dags, scenario_name, thr=.2, verb=False):
    if SELECTED_SCENARIOS and scenario_name not in SELECTED_SCENARIOS:
        log_status(f"SKIP scenario={scenario_name} selected={sorted(SELECTED_SCENARIOS)}")
        return None

    if LOAD:
        return load_var_results(scenario_name)

    var = np.asarray(data_p["var"], dtype=float)
    var_text = f"{var:.3g}" if var.ndim == 0 else f"hetero[{var.min():.3g},{var.max():.3g}] mean={var.mean():.3g}"
    n_jobs = max(1, min(N_CPUS, n_dags))
    log_status(
        f"START scenario={scenario_name} nodes={data_p['n_nodes']} edges={data_p['edges']} "
        f"var={var_text} dags={n_dags} variances={list(var_range)} "
        f"methods={[exp['leg'] for exp in exps]} workers={n_jobs}"
    )

    t_init = perf_counter()
    parallel = Parallel(n_jobs=n_jobs, verbose=JOBLIB_VERBOSE)
    try:
        results = parallel(
            # delayed(run_var_exp)(g, data_p, var_range, exps, thr, verb)
            delayed(run_var_exp)(g, data_p, var_range, exps, node_ord_type, thr, verb)
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

    log_status(f"DONE scenario={scenario_name} elapsed_minutes={(perf_counter() - t_init) / 60:.3f}")

    metrics = tuple(np.asarray(metric) for metric in zip(*results))
    if SAVE:
        save_var_results(var_results_prefix(scenario_name), metrics, exps, var_range, scenario_name)

    return (*metrics, exps, var_range)


# ---------------------------------------------------------------------------
# Experiments
# ---------------------------------------------------------------------------
def build_experiments():
    """
    Nonnegative CoLiDE (SCA+Adam, NV and EV) with the hyperparameters of
    nonneg_colide/scripts/preliminary_exp.py; NOMAD, DAGMA and CoLiDE with the
    configuration of synthetic/scripts/number_samples.py. 'sigma': True marks
    the methods that estimate the noise std.
    """
    nn_colide_args = {
        "stepsize": 3e-4,
        "step_type": "fixed",
        "alpha_0": .01,
        "rho_0": .05,
        "s": 1,
        "lamb": .1,
        "iters_in": 30000,
        "iters_out": 10,
        "beta": 2,
        "sca_adam": True,
    }
    colide_args = {"lambda1": .05, "T": 4, "s": [1.0, .9, .8, .7], "warm_iter": 2e4, "max_iter": 7e4, "lr": .0003}

    exps = [
        ### NONNEGATIVE COLIDE ###
        {
            "model": MetMulColide,
            "args": nn_colide_args.copy(),
            "init": {"primal_opt": "sca", "acyclicity": "logdet", "equal_var": False},
            "adapt_lamb": True,
            "standarize": False,
            "sigma": True,
            "fmt": "o-",
            "leg": "NN-CoLiDE-NV",
        },
        {
            "model": MetMulColide,
            "args": nn_colide_args.copy(),
            "init": {"primal_opt": "sca", "acyclicity": "logdet", "equal_var": True},
            "adapt_lamb": True,
            "standarize": False,
            "sigma": True,
            "fmt": "o--",
            "leg": "NN-CoLiDE-EV",
        },
        ### NOMAD (previous paper) ###
        {
            "model": MetMulDagma,
            "args": {
                "stepsize": 5e-3,
                "step_type": "fixed",
                "alpha_0": .1,
                "rho_0": .1,
                "s": 1,
                "lamb": .2,
                "iters_in": 5000,
                "iters_out": 10,
                "beta": 1.5,
            },
            "init": {"primal_opt": "adam", "acyclicity": "logdet"},
            "adapt_lamb": True,
            "standarize": False,
            "sigma": False,
            "fmt": "s-",
            "leg": "NOMAD-adam",
        },
        ### BASELINES ###
        {
            "model": DAGMA_linear,
            "init": {"loss_type": "l2"},
            "args": colide_args.copy(),
            "adapt_lamb": False,
            "standarize": False,
            "sigma": False,
            "fmt": "^-",
            "leg": "DAGMA",
        },
        {
            "model": colide_ev,
            "args": colide_args.copy(),
            "adapt_lamb": False,
            "standarize": False,
            "sigma": True,
            "fmt": "v--",
            "leg": "CoLiDE-EV",
        },
        {
            "model": colide_nv,
            "args": colide_args.copy(),
            "adapt_lamb": False,
            "standarize": False,
            "sigma": True,
            "fmt": "v-",
            "leg": "CoLiDE-NV",
        },
    ]

    if SMOKE_TEST:
        for exp in exps:
            if exp["model"] is MetMulColide or exp["model"] is MetMulDagma:
                exp["args"].update({"iters_in": 200, "iters_out": 2})
            else:
                exp["args"].update({"T": 2, "warm_iter": 100, "max_iter": 200})
    return exps


def build_scenarios(n_nodes=100):
    base = {
        "n_samples": n_samples,
        "n_nodes": n_nodes,
        "graph_type": "er",
        "edges": 4 * n_nodes,
        "edge_type": "positive",
        "w_range": (.5, 1),
    }
    rng = np.random.default_rng(SEED)
    hetero_var = rng.uniform(low=HETERO_SIGMA_RANGE[0], high=HETERO_SIGMA_RANGE[1], size=n_nodes) ** 2

    return [
        {"name": f"{exp_name}_N{n_nodes}_var1", "title": "var = 1", "data_params": {**base, "var": 1}},
        # {"name": f"noco_N{n_nodes}_var5", "title": "var = 5", "data_params": {**base, "var": 5}},
        # {"name": f"noco_N{n_nodes}_var10", "title": "var = 10", "data_params": {**base, "var": 10}},
        # {"name": f"noco_N{n_nodes}_hetero", "title": "heteroscedastic", "data_params": {**base, "var": hetero_var}},
    ]


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------
def sigma_skip_idx(exps, skip_idx=()):
    """Indices to skip in the Sigma plots: methods without Sigma plus the given ones."""
    return sorted(set(skip_idx) | {i for i, exp in enumerate(exps) if not exp.get("sigma", False)})

def plot_sigma_error(err_sig, exps, var_range, skip_idx=(), agg="median", deviation="prctile",
                     alpha=.25, title=None, figsize=(6, 4.5), ylabel="Sigma error (normalized)"):
    """Single panel with the scale-invariant Sigma error of the methods that estimate it."""
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    plot_data(ax, err_sig, exps, var_range, "Variance of outliers", ylabel,
                    sigma_skip_idx(exps, skip_idx), agg=agg, deviation=deviation, alpha=alpha,
                    plot_func="semilogy")
    if title is not None:
        ax.set_title(title)
    fig.tight_layout()
    return fig, ax

def plot_data(axes, data, exps, x_vals, xlabel, ylabel, skip_idx=[], agg='mean', deviation=None,
              alpha=.25, plot_func='semilogx', legend=True):
    if agg == 'median':
        agg_data = np.median(data, axis=0)
    else:
        agg_data = np.mean(data, axis=0)

    num_vars, num_ratios, num_exps = agg_data.shape

    std = np.std(data, axis=0)
    prctile25 = np.percentile(data, 25, axis=0)
    prctile75 = np.percentile(data, 75, axis=0)

    for k, vr in enumerate(VAR_RATIOS):
        for i, exp in enumerate(exps):
            if i in skip_idx:
                continue
            getattr(axes, plot_func)(x_vals, agg_data[:,k,i], exp['fmt'], label=exp['leg'] + f" r={vr}")

            if deviation == 'prctile':
                up_ci = prctile25[:,k,i]
                low_ci = prctile75[:,k,i]
                axes.fill_between(x_vals, low_ci, up_ci, alpha=alpha)
            elif deviation == 'std':
                up_ci = agg_data[:,k,i] + std[:,k,i]
                low_ci = np.maximum(agg_data[:,k,i] - std[:,k,i], 0)
                axes.fill_between(x_vals, low_ci, up_ci, alpha=alpha)

    axes.set_xlabel(xlabel)
    axes.set_ylabel(ylabel)
    axes.grid(True)
    if legend:
        axes.legend()

def plot_shd_error_pair(shd, err, exps, x_vals, xlabel, shd_ylabel, err_ylabel, skip_idx=[],
                        agg='mean', deviation=None, alpha=.25, shd_plot_func='semilogy',
                        err_plot_func='semilogy', title=None, figsize=(8, 4),
                        legend_ncol=4, shd_agg=None, err_agg=None, shd_deviation=None,
                        err_deviation=None):
    n_labels = len([i for i in range(len(exps)) if i not in set(skip_idx)])
    legend_rows = utils.shared_legend_rows(n_labels, legend_ncol)
    if legend_rows:
        figsize = (figsize[0], figsize[1] + 0.25 * legend_rows)

    fig, axes = plt.subplots(1, 2, figsize=figsize)
    shd_agg = agg if shd_agg is None else shd_agg
    err_agg = agg if err_agg is None else err_agg
    shd_deviation = deviation if shd_deviation is None else shd_deviation
    err_deviation = deviation if err_deviation is None else err_deviation

    plot_data(axes[0], shd, exps, x_vals, xlabel, shd_ylabel, skip_idx,
              agg=shd_agg, deviation=shd_deviation, alpha=alpha, plot_func=shd_plot_func,
              legend=False)
    plot_data(axes[1], err, exps, x_vals, xlabel, err_ylabel, skip_idx,
              agg=err_agg, deviation=err_deviation, alpha=alpha, plot_func=err_plot_func,
              legend=False)

    if title is not None:
        fig.suptitle(title)

    utils.shared_legend(fig, axes, ncol=legend_ncol)
    fig.tight_layout(rect=(0, 0, 1, utils.shared_legend_top(n_labels, legend_ncol)))
    return fig, axes

def plot_all_metrics(shd, tpr, fdr, fscore, err, acyc, runtime, dag_count, x_vals, exps, 
                     agg='mean', skip_idx=[], dev=False, alpha=.25, xlabel='Number of samples'):
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    plot_data(axes[0], shd, exps, x_vals, xlabel, 'SDH', skip_idx,
              agg=agg, deviation=dev, alpha=alpha)
    plot_data(axes[1], tpr, exps, x_vals, xlabel, 'TPR', skip_idx,
              agg=agg, deviation=dev, alpha=alpha)
    plot_data(axes[2], fdr, exps, x_vals, xlabel, 'FDR', skip_idx,
              agg=agg, deviation=dev, alpha=alpha)
    plot_data(axes[3], fscore, exps, x_vals, xlabel, 'F1', skip_idx,
              agg=agg, deviation=dev, alpha=alpha)
    plt.tight_layout()

    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    plot_data(axes[0], err, exps, x_vals, xlabel, 'Fro Error', skip_idx, agg=agg,
              deviation=dev, alpha=alpha, plot_func='semilogy')
    plot_data(axes[1], acyc, exps, x_vals, xlabel, 'Acyclity', skip_idx, agg=agg,
              deviation=dev)
    plot_data(axes[2], runtime, exps, x_vals, xlabel, 'Running time (seconds)',
              skip_idx, agg=agg, deviation=dev, alpha=alpha, plot_func='semilogy')
    plot_data(axes[3], dag_count, exps, x_vals, xlabel, 'Graph is DAG', skip_idx,
              agg=agg)
    plt.tight_layout()

def plot_results(metrics, exps, var_range, scenario, skip_idx=(), save=True):
    metrics = dict(zip(METRIC_NAMES, metrics))
    prefix = var_results_prefix(scenario["name"])
    skip = list(skip_idx)

    fig, _ = plot_shd_error_pair(
        metrics["shd"], metrics["err"], exps, var_range,
        xlabel="Variance of outliers",
        shd_ylabel="SHD",
        err_ylabel="Fro Error",
        skip_idx=skip,
        alpha=0.25,
        shd_agg="mean",
        shd_deviation="std",
        err_agg="median",
        err_deviation="prctile",
        err_plot_func="semilogy",
        figsize=(10, 5),
        title=scenario["title"],
    )
    if save:
        fig.savefig(f"{prefix}_summary.png", bbox_inches="tight")

    fig, _ = plot_sigma_error(metrics["err_sig"], exps, var_range, skip_idx=skip,
                              title=f"{scenario['title']} - Sigma error")
    if save:
        fig.savefig(f"{prefix}_sigma_error.png", bbox_inches="tight")

    for agg in ("mean", "median"):
        plot_all_metrics(metrics["shd"], metrics["tpr"], metrics["fdr"], metrics["fscore"], metrics["err"],
                               metrics["acyc"], metrics["runtime"], metrics["dag_count"], var_range, exps,
                               skip_idx=skip, agg=agg)
        if save:
            plt.savefig(f"{prefix}_all_metrics_{agg}.png", bbox_inches="tight")
    if save:
        plt.close("all")

def run_or_load(name, skip=()):
    sc = scenarios[name]
    out = run_or_load_var_results(sc["data_params"], var_range, exps, n_dags, sc["name"], thr=thr, verb=verb)
    *metrics, exps_out, xvals = out
    results[name] = {"metrics": dict(zip(METRIC_NAMES, metrics)), "exps": exps_out, "xvals": xvals, "scenario": sc}
    return results[name]


def show_scenario(name, skip=()):
    res = run_or_load(name)
    m, exps_out, xvals, sc = res["metrics"], res["exps"], res["xvals"], res["scenario"]
    skip = list(skip)
    prefix = var_results_prefix(sc["name"])

    # SHD (mean +- std) and Fro error (median, prctiles 25-75)
    fig, _ = plot_shd_error_pair(m["shd"], m["err"], exps_out, xvals, xlabel="Variance of outliers",
                                       shd_ylabel="SHD", err_ylabel="Fro Error", skip_idx=skip, alpha=.25,
                                       shd_agg="mean", shd_deviation="std", err_agg="median", err_deviation="prctile",
                                       err_plot_func="semilogy", figsize=(10, 5), title=sc["title"])
    if SAVE_FIGS:
        fig.savefig(f"{prefix}_summary.png", bbox_inches="tight")

    # Sigma error (only methods that estimate Sigma)
    fig, _ = plot_sigma_error(m["err_sig"], exps_out, xvals, skip_idx=skip, title=f'{sc["title"]} - Sigma error')
    if SAVE_FIGS:
        fig.savefig(f"{prefix}_sigma_error.png", bbox_inches="tight")

    # Tables at each number of samples
    legs = [e["leg"] for e in exps_out]
    for metric, agg, label in [("shd", np.mean, "SHD (mean)"), ("err", np.median, "Fro error (median)"),
                               ("err_sig", np.nanmedian, "Sigma error (median)"), ("runtime", np.mean, "time (mean, s)")]:
        with np.errstate(all="ignore"):
            m_agg = agg(m[metric], axis=0)
            R = m_agg.shape[1]
            for k in range(R):
                table = pd.DataFrame(m_agg[:,k,:], index=xvals, columns=legs).rename_axis("variances")
        print(f"\n{sc['title']} - {label}")
        display(table.round(4))
    return res


def show_all_metrics(name, agg="mean", skip=()):
    m, exps_out, xvals = results[name]["metrics"], results[name]["exps"], results[name]["xvals"]
    plot_all_metrics(m["shd"], m["tpr"], m["fdr"], m["fscore"], m["err"], m["acyc"], m["runtime"],
                           m["dag_count"], xvals, exps_out, skip_idx=list(skip), agg=agg)


# ---------------------------------------------------------------


exps = build_experiments()
scenarios = build_scenarios(n_nodes=n_nodes)
for scenario in scenarios:
    out = run_or_load_var_results(
        scenario["data_params"],
        var_range, 
        exps, 
        n_dags, 
        scenario["name"],
        thr=thr,
        verb=verb
    )
    if out is None:
        continue
    *metrics, exps_out, xvals = out
    plot_results(metrics, exps_out, xvals, scenario)

SAVE = False
LOAD = True
SAVE_FIGS=  False
# RESULTS_DIR = None

scenarios = {sc["name"]: sc for sc in build_scenarios(n_nodes=n_nodes)}

pd.DataFrame([{"leg": e["leg"], "model": e["model"].__name__, "sigma": e["sigma"], "adapt_lamb": e["adapt_lamb"],
               **e.get("init", {}), **e["args"]} for e in exps]).set_index("leg")

results = {}




skip = []
res = show_scenario(f"{exp_name}_N{n_nodes}_var1", skip=skip)

show_all_metrics(f"{exp_name}_N{n_nodes}_var1", agg="mean", skip=skip)
show_all_metrics(f"{exp_name}_N{n_nodes}_var1", agg="median", skip=skip)


# skip = []
# res = show_scenario(f"noco_N{n_nodes}_var5", skip=skip)

# show_all_metrics(f"noco_N{n_nodes}_var5", agg="mean", skip=skip)
# show_all_metrics(f"noco_N{n_nodes}_var5", agg="median", skip=skip)



# skip = []
# res = show_scenario(f"noco_N{n_nodes}_hetero", skip=skip)

# show_all_metrics(f"noco_N{n_nodes}_hetero", agg="mean", skip=skip)
# show_all_metrics(f"noco_N{n_nodes}_hetero", agg="median", skip=skip)

print("DONE")