#!/usr/bin/env python3
# coding: utf-8

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import signal
from time import perf_counter

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from joblib import Parallel, delayed
from sklearn.metrics import f1_score

import src.utils as utils
from src.model import MetMulDagma, MetMulColide

from baselines.colide import colide_ev
from baselines.dagma_linear import DAGMA_linear
from baselines.notears import notears_linear
from baselines.golem import GOLEM_TF_EV
from baselines.nonnegative_dagma_linear import NonnegativeDAGMA_linear


PATH = str(ROOT / "results" / "samples") + os.sep
SAVE = True
LOAD = False
SEED = 10
N_CPUS = max(1, int(os.environ.get("N_CPUS", os.cpu_count() or 1)))

np.random.seed(SEED)
os.makedirs(PATH, exist_ok=True)


def _handle_termination(signum, frame):
    raise KeyboardInterrupt(f"Received signal {signum}; stopping experiments")


signal.signal(signal.SIGTERM, _handle_termination)


def get_lamb_value(n_nodes, n_samples, times=1):
    return np.sqrt(np.log(n_nodes) / n_samples) * times


def run_samples_exp(g, data_p, n_samples_values, exps, thr=.2, verb=False):
    shd, tpr, fdr, fscore, err, acyc, runtime, dag_count = [
        np.zeros((len(n_samples_values), len(exps))) for _ in range(8)
    ]
    for i, n_samples in enumerate(n_samples_values):
        if g % N_CPUS == 0:
            print(f"Graph: {g + 1}, samples: {n_samples}", flush=True)

        data_p_aux = data_p.copy()
        data_p_aux["n_samples"] = n_samples

        W_true, _, X = utils.simulate_sem(**data_p_aux)
        X_std = utils.standarize(X)
        W_true_bin = utils.to_bin(W_true, thr)
        norm_W_true = np.linalg.norm(W_true)

        for j, exp in enumerate(exps):
            X_aux = X_std if exp.get("standarize", False) else X

            arg_aux = exp["args"].copy()
            if exp.get("adapt_lamb", False):
                if "lamb" in arg_aux:
                    arg_aux["lamb"] = get_lamb_value(data_p["n_nodes"], n_samples, arg_aux["lamb"])
                elif "lambda1" in arg_aux:
                    arg_aux["lambda1"] = get_lamb_value(data_p["n_nodes"], n_samples, arg_aux["lambda1"])

            model = None
            if exp["model"] == notears_linear:
                t_init = perf_counter()
                W_est = notears_linear(X_aux, **arg_aux)
                t_end = perf_counter()
            else:
                model = exp["model"](**exp["init"]) if "init" in exp else exp["model"]()
                t_init = perf_counter()
                model.fit(X_aux, **arg_aux)
                t_end = perf_counter()
                W_est = model.W_est

            if np.isnan(W_est).any():
                W_est = np.zeros_like(W_est)
                W_est_bin = np.zeros_like(W_est)
            else:
                W_est_bin = utils.to_bin(W_est, thr)

            shd[i, j], tpr[i, j], fdr[i, j] = utils.count_accuracy(W_true_bin, W_est_bin)
            fscore[i, j] = f1_score(W_true_bin.flatten(), W_est_bin.flatten())
            err[i, j] = utils.compute_norm_sq_err(W_true, W_est, norm_W_true)
            acyc[i, j] = model.dagness(W_est) if model is not None and hasattr(model, "dagness") else 1
            runtime[i, j] = t_end - t_init
            dag_count[i, j] += 1 if utils.is_dag(W_est_bin) else 0

            if verb and (g % N_CPUS == 0):
                print(
                    f'\t-{exp["leg"]}: shd {shd[i, j]}  -  err: {err[i, j]:.3f}'
                    f"  -  time: {runtime[i, j]:.3f}",
                    flush=True,
                )

    return shd, tpr, fdr, fscore, err, acyc, runtime, dag_count


def samples_results_prefix(data_p):
    n_nodes = data_p["n_nodes"]
    return f'{PATH}samples_{n_nodes}N_{int(data_p["edges"] / n_nodes)}'


def save_samples_results(file_prefix, metrics, exps, n_samples_values):
    shd, tpr, fdr, fscore, err, acyc, runtime, dag_count = metrics
    np.savez(
        file_prefix,
        shd=shd,
        tpr=tpr,
        fdr=fdr,
        fscore=fscore,
        err=err,
        acyc=acyc,
        runtime=runtime,
        dag_count=dag_count,
        exps=exps,
        xvals=n_samples_values,
    )
    print("SAVED in file:", file_prefix, flush=True)

    agg_error = np.median(err, axis=0)
    utils.data_to_csv(f"{PATH}samples_err_med.csv", exps, n_samples_values, agg_error)
    prctile25 = np.percentile(err, 25, axis=0)
    utils.data_to_csv(f"{PATH}samples_err_prctile25.csv", exps, n_samples_values, prctile25)
    prctile75 = np.percentile(err, 75, axis=0)
    utils.data_to_csv(f"{PATH}samples_err_prctile75.csv", exps, n_samples_values, prctile75)

    agg_shd = np.mean(shd, axis=0)
    utils.data_to_csv(f"{PATH}samples_shd_mean.csv", exps, n_samples_values, agg_shd)
    std_shd = np.std(shd, axis=0)
    utils.data_to_csv(f"{PATH}samples_shd_std.csv", exps, n_samples_values, std_shd)


def load_samples_results(file_prefix):
    file_name = f"{file_prefix}.npz"
    data = np.load(file_name, allow_pickle=True)
    print("Loaded samples results from", file_name, flush=True)
    return (
        data["shd"],
        data["tpr"],
        data["fdr"],
        data["fscore"],
        data["err"],
        data["acyc"],
        data["runtime"],
        data["dag_count"],
        data["exps"].tolist(),
        data["xvals"],
    )


def run_or_load_samples_results(data_p, n_samples_values, exps, n_dags, thr=.2, verb=False):
    file_prefix = samples_results_prefix(data_p)

    if LOAD:
        return load_samples_results(file_prefix)

    n_jobs = max(1, min(N_CPUS, n_dags))
    print("CPUs employed:", n_jobs, flush=True)

    t_init = perf_counter()
    parallel = Parallel(n_jobs=n_jobs)
    try:
        results = parallel(
            delayed(run_samples_exp)(g, data_p, n_samples_values, exps, thr, verb)
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

    t_end = perf_counter()
    print(f"----- Solved in {(t_end - t_init) / 60:.3f} minutes -----", flush=True)

    metrics = tuple(np.asarray(metric) for metric in zip(*results))
    if SAVE:
        save_samples_results(file_prefix, metrics, exps, n_samples_values)

    return (*metrics, exps, n_samples_values)


def build_experiments():
    return [
        # {
        #     "model": MetMulDagma,
        #     "args": {
        #         "stepsize": 3e-4,
        #         "step_type": "fixed",
        #         "alpha_0": .01,
        #         "rho_0": .05,
        #         "s": 1,
        #         "lamb": 1e-1,
        #         "iters_in": 10000,
        #         "iters_out": 10,
        #         "beta": 2,
        #     },
        #     "init": {"primal_opt": "adam", "acyclicity": "logdet"},
        #     "adapt_lamb": True,
        #     "standarize": False,
        #     "fmt": "o-",
        #     "leg": "NOMAD-adam-ORIG",
        # },
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
            "fmt": "o-",
            "leg": "NOMAD-adam",
        },
        {
            "model": MetMulDagma,
            "args": {
                "stepsize": 1e-5,
                "step_type": "fixed",
                "alpha_0": .01,
                "rho_0": .01,
                "s": 1,
                "lamb": .2,
                "iters_in": 10000,
                "iters_out": 50,
                "beta": 1.5,
            },
            "init": {"acyclicity": "logdet", "primal_opt": "fista", "restart": True},
            "adapt_lamb": True,
            "standarize": False,
            "fmt": "o--",
            "leg": "NOMAD-fista",
        },
        ##### BASELINES ####
        ### NoTears
        {
            "model": notears_linear,
            "args": {"loss_type": "l2", "lambda1": .1, "max_iter": 10},
            "standarize": False,
            "fmt": "D-",
            "leg": "NoTears",
        },
        ### DAGMA
        {
            "model": DAGMA_linear,
            "init": {"loss_type": "l2"},
            "args": {"lambda1": .05, "T": 4, "s": [1.0, .9, .8, .7], "warm_iter": 2e4, "max_iter": 7e4, "lr": .0003},
            "standarize": False,
            "fmt": "^-",
            "leg": "DAGMA",
        },
        # {
        #     "model": MetMulDagma,
        #     "args": {
        #         "stepsize": 1e-4,
        #         "step_type": "fixed",
        #         "alpha_0": .05,
        #         "rho_0": .05,
        #         "s": 1,
        #         "lamb": .2,
        #         "iters_in": 10000,
        #         "iters_out": 50,
        #         "beta": 1.5,
        #     },
        #     "init": {"acyclicity": "logdet", "primal_opt": "fista", "restart": True},
        #     "adapt_lamb": True,
        #     "standarize": False,
        #     "fmt": "o--",
        #     "leg": "NOMAD-fista-100",
        # },
        ### Nonnegative DAGMA
        {
            "model": NonnegativeDAGMA_linear,
            "init": {"loss_type": "l2"},
            "args": {"lambda1": .05, "T": 4, "s": [1.0, .9, .8, .7], "warm_iter": 2e4, "max_iter": 7e4, "lr": .0003},
            "standarize": False,
            "adapt_lamb": False,
            "fmt": "s--",
            "leg": "NonDAGMA-Fix",
        },
        {
            "model": NonnegativeDAGMA_linear,
            "init": {"loss_type": "l2"},
            "args": {"lambda1": .74, "T": 4, "s": [1.0, .9, .8, .7], "warm_iter": 2e4, "max_iter": 7e4, "lr": .0003},
            "standarize": False,
            "adapt_lamb": True,
            "fmt": "s-",
            "leg": "NonDAGMA",
        },

        ### CoLiDE
        {
            "model": colide_ev,
            "args": {"lambda1": .05, "T": 4, "s": [1.0, .9, .8, .7], "warm_iter": 2e4, "max_iter": 7e4, "lr": .0003},
            "standarize": False,
            "fmt": "v--",
            "leg": "CoLiDE-Fix",
        },
        {
            "model": colide_ev,
            "args": {"lambda1": .74, "T": 4, "s": [1.0, .9, .8, .7], "warm_iter": 2e4, "max_iter": 7e4, "lr": .0003},
            "standarize": False,
            "adapt_lamb": True,
            "fmt": "v-",
            "leg": "CoLiDE",
        },
        ### GOLEM
        # {
        #     "model": GOLEM_TF_EV,
        #     "args": {
        #         "lambda1": 2e-2,
        #         "lambda2": 5.0,
        #         "num_iter": 100000,
        #         "learning_rate": 1e-3,
        #         "w_threshold": 0.3,
        #         "postprocess": True,
        #         "checkpoint": None,
        #     },
        #     "standarize": False,
        #     "adapt_lamb": True,
        #     "fmt": ">--",
        #     "leg": "GOLEM-EV-Fix",
        # },
        {
            "model": GOLEM_TF_EV,
            "args": {
                "lambda1": 0.3,
                "lambda2": 5.0,
                "num_iter": 100000,
                "learning_rate": 1e-3,
                "w_threshold": 0.3,
                "postprocess": True,
                "checkpoint": None,
            },
            "standarize": False,
            "adapt_lamb": True,
            "fmt": ">-",
            "leg": "GOLEM-EV",
        },

    ]


def plot_results(metrics, exps, n_samples_values):
    shd, tpr, fdr, fscore, err, acyc, runtime, dag_count = metrics

    skip = []
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    utils.plot_data(axes[0], shd, exps, n_samples_values, "Number of samples", "SDH", skip,
                    agg="mean", deviation="std", alpha=0.25)
    utils.plot_data(axes[1], err, exps, n_samples_values, "Number of samples", "Fro Error", skip,
                    agg="median", deviation="prctile", alpha=0.25, plot_func="loglog")
    plt.tight_layout()
    fig.savefig(f"{PATH}samples_summary.png", bbox_inches="tight")
    plt.close(fig)

    utils.plot_all_metrics(shd, tpr, fdr, fscore, err, acyc, runtime, dag_count, n_samples_values, exps,
                           skip_idx=skip, agg="mean")
    plt.savefig(f"{PATH}samples_all_metrics_mean.png", bbox_inches="tight")
    plt.close("all")

    utils.plot_all_metrics(shd, tpr, fdr, fscore, err, acyc, runtime, dag_count, n_samples_values, exps,
                           skip_idx=skip, agg="median")
    plt.savefig(f"{PATH}samples_all_metrics_median.png", bbox_inches="tight")
    plt.close("all")


def main():
    n_dags = 50 # 100
    n_samples_values = np.array([50, 60, 80, 100, 200, 500, 1000, 5000, 10000])
    exps = build_experiments()

    n_nodes = 100
    thr = .2
    verb = True
    data_p = {
        "n_nodes": n_nodes,
        "graph_type": "er",
        "edges": 4 * n_nodes,
        "edge_type": "positive",
        "w_range": (.5, 1),
        "var": 1,
    }

    *metrics, exps, n_samples_values = run_or_load_samples_results(
        data_p, n_samples_values, exps, n_dags, thr=thr, verb=verb
    )
    plot_results(metrics, exps, n_samples_values)


if __name__ == "__main__":
    main()
