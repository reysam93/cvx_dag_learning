#!/usr/bin/env python3
# coding: utf-8

import os
import signal
import sys
from pathlib import Path
from time import perf_counter

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("MPLCONFIGDIR", str(Path("/tmp") / "matplotlib"))
os.makedirs(os.environ["MPLCONFIGDIR"], exist_ok=True)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from joblib import Parallel, delayed
from sklearn.metrics import f1_score

import src.utils as utils
from src.model import MetMulDagma, MetMulColide

from baselines.colide import colide_ev, colide_nv
from baselines.dagma_linear import DAGMA_linear
from baselines.golem import GOLEM_EV, GOLEM_NV
from baselines.notears import notears_linear
from baselines.nonnegative_dagma_linear import NonnegativeDAGMA_linear


PATH = str(ROOT / "results" / "var") + os.sep
SAVE = True
LOAD = False
SEED = 10
N_CPUS = max(1, int(os.environ.get("N_CPUS", os.cpu_count() or 1)))
JOBLIB_VERBOSE = max(0, int(os.environ.get("JOBLIB_VERBOSE", 0)))

N_DAGS = 50
THR = .2
VERB = False
VAR_VALUES = np.array([1, 5, 10, 20, 30])
HETERO_VAR_RANGE = (.5, 5.0)
JOINT_AGGS = ("mean", "median")
SKIP_IDX = []

BASE_DATA_PARAMS = {
    "graph_type": "er",
    "n_nodes": 100,
    "edges": 4,  # Edges per node; converted to total edges inside run_var_exp.
    "edge_type": "positive",
    "w_range": (.5, 1),
    "n_samples": 1000,
}

# Set to None to run every experiment from build_experiments().
# Example: SELECTED_EXPERIMENT_LEGS = ["MM-adam", "MM-fista", "DAGMA"]
SELECTED_EXPERIMENT_LEGS = None

np.random.seed(SEED)
os.makedirs(PATH, exist_ok=True)


def _handle_termination(signum, frame):
    raise KeyboardInterrupt(f"Received signal {signum}; stopping experiments")


signal.signal(signal.SIGTERM, _handle_termination)


def get_lamb_value(n_nodes, n_samples, times=1):
    return np.sqrt(np.log(n_nodes) / n_samples) * times


def build_scenarios(data_p):
    rng = np.random.default_rng(SEED)
    hetero_profile = rng.uniform(
        low=HETERO_VAR_RANGE[0],
        high=HETERO_VAR_RANGE[1],
        size=data_p["n_nodes"],
    ) ** 2

    return [
        {
            "name": "homocedastic",
            "suffix": "hom",
            "data_params": data_p.copy(),
            "noise_profile": None,
            "line_style": "-",
        },
        {
            "name": "heterocedastic",
            "suffix": "hetero",
            "data_params": data_p.copy(),
            "noise_profile": hetero_profile,
            "line_style": "--",
        },
    ]


def build_experiments():
    return [
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
        # {
        #     "model": MetMulDagma,
        #     "args": {
        #         "stepsize": 3e-4,
        #         "alpha_0": .01,
        #         "rho_0": .05,
        #         "s": 1,
        #         "lamb": .05,
        #         "iters_in": 10000,
        #         "iters_out": 10,
        #         "beta": 2,
        #     },
        #     "init": {"primal_opt": "adam", "acyclicity": "logdet"},
        #     "adapt_lamb": True,
        #     "sigma_known": True,
        #     "standarize": False,
        #     "fmt": "o:",
        #     "leg": "MM-Logdet-Sigma",
        # },

        ##### BASELINES ####
        {
            "model": DAGMA_linear,
            "init": {"loss_type": "l2"},
            "args": {"lambda1": .05, "T": 4, "s": [1.0, .9, .8, .7], "warm_iter": 2e4, "max_iter": 7e4, "lr": .0003},
            "standarize": False,
            "fmt": "^-",
            "leg": "DAGMA",
        },
        {
            "model": NonnegativeDAGMA_linear,
            "init": {"loss_type": "l2"},
            "args": {"lambda1": .05, "T": 4, "s": [1.0, .9, .8, .7], "warm_iter": 2e4, "max_iter": 7e4, "lr": .0003},
            "standarize": False,
            "adapt_lamb": False,
            "fmt": "s--",
            "leg": "NonDAGMA",
        },
        ### CoLiDE
        {
            "model": colide_ev,
            "args": {"lambda1": .05, "T": 4, "s": [1.0, .9, .8, .7], "warm_iter": 2e4, "max_iter": 7e4, "lr": .0003},
            "standarize": False,
            "fmt": "v--",
            "leg": "CoLiDE-EV",
        },
        {
            "model": colide_nv,
            "args": {"lambda1": .05, "T": 4, "s": [1.0, .9, .8, .7], "warm_iter": 2e4, "max_iter": 7e4, "lr": .0003},
            "standarize": False,
            "fmt": "v-",
            "leg": "CoLiDE-NV",
        },
        ### GOLEM
        {
            "model": GOLEM_EV,
            "args": {
                "lambda1": 2e-2,
                "lambda2": 5.0,
                "num_iter": 100000,
                "learning_rate": 1e-3,
                "w_threshold": 0.3,
                "postprocess": True,
                "checkpoint": None,
            },
            "standarize": False,
            "fmt": ">--",
            "leg": "GOLEM-EV",
        },
        {
            "model": GOLEM_NV,
            "init": {"init_with_ev": True},
            "args": {
                "lambda1": 2e-3,
                "lambda2": 5.0,
                "lambda1_ev": 2e-2,
                "lambda2_ev": 5.0,
                "num_iter": 100000,
                "num_iter_ev": 100000,
                "learning_rate": 1e-3,
                "learning_rate_ev": 1e-3,
                "w_threshold": 0.3,
                "postprocess": True,
                "checkpoint": None,
            },
            "standarize": False,
            "fmt": ">-",
            "leg": "GOLEM-NV",
        },
    ]


def filter_experiments(exps, selected_legs):
    if selected_legs is None:
        return exps

    selected_legs = list(selected_legs)
    by_leg = {exp["leg"]: exp for exp in exps}
    missing = [leg for leg in selected_legs if leg not in by_leg]
    if missing:
        raise ValueError(f"Unknown experiment legend(s): {missing}")
    return [by_leg[leg] for leg in selected_legs]


def run_var_exp(g, data_p, var_values, exps, noise_profile=None, thr=.2, verb=False):
    shd, tpr, fdr, fscore, err, acyc, runtime, dag_count = [
        np.zeros((len(var_values), len(exps))) for _ in range(8)
    ]

    for i, var in enumerate(var_values):
        if g % N_CPUS == 0:
            print(f"Graph: {g + 1}, variance: {var}", flush=True)

        data_p_aux = data_p.copy()
        data_p_aux["edges"] *= data_p_aux["n_nodes"]
        data_p_aux["var"] = var if noise_profile is None else noise_profile * var
        data_p_aux["n_samples"] = (
            10 * data_p_aux["n_nodes"]
            if data_p_aux["n_samples"] is None
            else data_p_aux["n_samples"]
        )

        W_true, _, X = utils.simulate_sem(**data_p_aux)
        X_std = utils.standarize(X)
        W_true_bin = utils.to_bin(W_true, thr)
        norm_W_true = np.linalg.norm(W_true)

        for j, exp in enumerate(exps):
            X_aux = X_std if exp.get("standarize", False) else X

            arg_aux = exp["args"].copy()
            if exp.get("adapt_lamb", False):
                if "lamb" in arg_aux:
                    arg_aux["lamb"] = get_lamb_value(data_p_aux["n_nodes"], data_p_aux["n_samples"], arg_aux["lamb"])
                elif "lambda1" in arg_aux:
                    arg_aux["lambda1"] = get_lamb_value(data_p_aux["n_nodes"], data_p_aux["n_samples"], arg_aux["lambda1"])

            if exp.get("sigma_known", False) or exp.get("know_var", False):
                arg_aux["Sigma"] = data_p_aux["var"]

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
            shd[i, j] /= data_p_aux["n_nodes"]
            fscore[i, j] = f1_score(W_true_bin.flatten(), W_est_bin.flatten())
            err[i, j] = utils.compute_norm_sq_err(W_true, W_est, norm_W_true)
            acyc[i, j] = (
                model.dagness(W_est)
                if model is not None and hasattr(model, "dagness")
                else float(not utils.is_dag(W_est_bin))
            )
            runtime[i, j] = t_end - t_init
            dag_count[i, j] += 1 if utils.is_dag(W_est_bin) else 0

            if verb and (g % N_CPUS == 0):
                print(
                    f'\t-{exp["leg"]}: shd {shd[i, j]}  -  err: {err[i, j]:.3f}'
                    f"  -  time: {runtime[i, j]:.3f}",
                    flush=True,
                )

    return shd, tpr, fdr, fscore, err, acyc, runtime, dag_count


def vars_results_prefix(data_p, scenario_suffix):
    return f'{PATH}var_{scenario_suffix}_{data_p["graph_type"].upper()}graph_{data_p["edges"]}N'


def save_vars_results(file_prefix, metrics, exps, var_values, scenario_suffix):
    os.makedirs(PATH, exist_ok=True)
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
        xvals=var_values,
    )
    print("SAVED in file:", file_prefix, flush=True)

    agg_error = np.median(err, axis=0)
    utils.data_to_csv(f"{PATH}vars_{scenario_suffix}_err_med.csv", exps, var_values, agg_error)
    prctile25 = np.percentile(err, 25, axis=0)
    utils.data_to_csv(f"{PATH}vars_{scenario_suffix}_err_prctile25.csv", exps, var_values, prctile25)
    prctile75 = np.percentile(err, 75, axis=0)
    utils.data_to_csv(f"{PATH}vars_{scenario_suffix}_err_prctile75.csv", exps, var_values, prctile75)

    agg_shd = np.mean(shd, axis=0)
    utils.data_to_csv(f"{PATH}vars_{scenario_suffix}_shd_mean.csv", exps, var_values, agg_shd)
    std_shd = np.std(shd, axis=0)
    utils.data_to_csv(f"{PATH}vars_{scenario_suffix}_shd_std.csv", exps, var_values, std_shd)


def load_vars_results(file_prefix):
    file_name = f"{file_prefix}.npz"
    data = np.load(file_name, allow_pickle=True)
    print("Loaded variance results from", file_name, flush=True)
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


def run_or_load_vars_results(scenario, var_values, exps, n_dags, thr=.2, verb=False):
    data_p = scenario["data_params"]
    file_prefix = vars_results_prefix(data_p, scenario["suffix"])

    if LOAD:
        return load_vars_results(file_prefix)

    n_jobs = max(1, min(N_CPUS, n_dags))
    print(f'Running scenario={scenario["name"]}. CPUs employed: {n_jobs}', flush=True)

    t_init = perf_counter()
    parallel = Parallel(n_jobs=n_jobs, verbose=JOBLIB_VERBOSE)
    try:
        results = parallel(
            delayed(run_var_exp)(
                g,
                data_p,
                var_values,
                exps,
                scenario["noise_profile"],
                thr,
                verb,
            )
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
        save_vars_results(file_prefix, metrics, exps, var_values, scenario["suffix"])

    return (*metrics, exps, var_values)


def plot_results(metrics, exps, var_values, scenario_suffix, skip_idx=None):
    os.makedirs(PATH, exist_ok=True)
    shd, tpr, fdr, fscore, err, acyc, runtime, dag_count = metrics

    skip = [] if skip_idx is None else list(skip_idx)
    fig, axes = plt.subplots(1, 2, figsize=(8, 4))
    utils.plot_data(axes[0], shd, exps, var_values, "Noise variance", "Normalized SHD", skip,
                    agg="mean", deviation="std", alpha=0.25, plot_func="plot")
    utils.plot_data(axes[1], err, exps, var_values, "Noise variance", "Fro Error", skip,
                    agg="mean", deviation="std", alpha=0.25, plot_func="semilogy")
    fig.suptitle(f"{scenario_suffix} - mean")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(f"{PATH}vars_{scenario_suffix}_summary_mean.png", bbox_inches="tight")
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(8, 4))
    utils.plot_data(axes[0], shd, exps, var_values, "Noise variance", "Normalized SHD", skip,
                    agg="median", deviation="prctile", alpha=0.25, plot_func="plot")
    utils.plot_data(axes[1], err, exps, var_values, "Noise variance", "Fro Error", skip,
                    agg="median", deviation="prctile", alpha=0.25, plot_func="semilogy")
    fig.suptitle(f"{scenario_suffix} - median")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(f"{PATH}vars_{scenario_suffix}_summary_median.png", bbox_inches="tight")
    plt.close(fig)

    utils.plot_all_metrics(shd, tpr, fdr, fscore, err, acyc, runtime, dag_count, var_values, exps,
                           skip_idx=skip, agg="mean", dev="std", xlabel="Noise variance")
    plt.gcf().suptitle(f"{scenario_suffix} - all metrics - mean")
    plt.savefig(f"{PATH}vars_{scenario_suffix}_all_metrics_mean.png", bbox_inches="tight")
    plt.close("all")

    utils.plot_all_metrics(shd, tpr, fdr, fscore, err, acyc, runtime, dag_count, var_values, exps,
                           skip_idx=skip, agg="median", dev="prctile", xlabel="Noise variance")
    plt.gcf().suptitle(f"{scenario_suffix} - all metrics - median")
    plt.savefig(f"{PATH}vars_{scenario_suffix}_all_metrics_median.png", bbox_inches="tight")
    plt.close("all")


def scenario_experiments(exps, suffix, line_style):
    return [
        {
            "leg": f'{exp["leg"]}-{suffix}',
            "fmt": exp.get("fmt", "o-")[0] + line_style,
        }
        for exp in exps
    ]


def plot_joint_results(scenario_results, agg="mean", skip_idx=None):
    os.makedirs(PATH, exist_ok=True)
    if len(scenario_results) < 2:
        return

    skip = set([] if skip_idx is None else skip_idx)
    reference_xvals = scenario_results[0]["var_values"]
    for result in scenario_results[1:]:
        if not np.array_equal(reference_xvals, result["var_values"]):
            raise ValueError("Cannot plot joint results with different variance grids")

    joint_exps = []
    shd_parts = []
    err_parts = []
    for result in scenario_results:
        keep_idx = [i for i in range(len(result["exps"])) if i not in skip]
        shd_parts.append(result["metrics"][0][:, :, keep_idx])
        err_parts.append(result["metrics"][4][:, :, keep_idx])
        joint_exps.extend(
            scenario_experiments(
                [result["exps"][i] for i in keep_idx],
                result["scenario"]["suffix"],
                result["scenario"]["line_style"],
            )
        )

    shd_joint = np.concatenate(shd_parts, axis=2)
    err_joint = np.concatenate(err_parts, axis=2)

    deviation = "std" if agg == "mean" else "prctile"

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    utils.plot_data(axes[0], shd_joint, joint_exps, reference_xvals, "Noise variance", "Normalized SHD", [],
                    agg=agg, deviation=deviation, alpha=0.25, plot_func="plot")
    utils.plot_data(axes[1], err_joint, joint_exps, reference_xvals, "Noise variance", "Fro Error", [],
                    agg=agg, deviation=deviation, alpha=0.25, plot_func="semilogy")
    fig.suptitle(f"hom vs hetero - {agg}")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(f"{PATH}vars_hom_hetero_joint_{agg}.png", bbox_inches="tight")
    plt.close(fig)


def main():
    exps = filter_experiments(build_experiments(), SELECTED_EXPERIMENT_LEGS)
    var_values = np.asarray(VAR_VALUES)
    scenario_results = []

    for scenario in build_scenarios(BASE_DATA_PARAMS):
        *metrics, scenario_exps, scenario_var_values = run_or_load_vars_results(
            scenario,
            var_values,
            exps,
            N_DAGS,
            thr=THR,
            verb=VERB,
        )
        plot_results(metrics, scenario_exps, scenario_var_values, scenario["suffix"], skip_idx=SKIP_IDX)
        scenario_results.append({
            "scenario": scenario,
            "metrics": metrics,
            "exps": scenario_exps,
            "var_values": scenario_var_values,
        })

    for agg in JOINT_AGGS:
        plot_joint_results(scenario_results, agg=agg, skip_idx=SKIP_IDX)


if __name__ == "__main__":
    main()
