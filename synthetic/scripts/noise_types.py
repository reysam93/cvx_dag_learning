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
from src.model import MetMulDagma

from baselines.colide import colide_ev, colide_nv
from baselines.dagma_linear import DAGMA_linear
from baselines.golem import GOLEM_EV
from baselines.notears import notears_linear
from baselines.nonnegative_dagma_linear import NonnegativeDAGMA_linear


PATH = str(ROOT / "results" / "noise") + os.sep
SAVE = True
LOAD = False
SEED = 10
N_CPUS = max(1, int(os.environ.get("N_CPUS", os.cpu_count() or 1)))
JOBLIB_VERBOSE = max(0, int(os.environ.get("JOBLIB_VERBOSE", 0)))

N_DAGS = 50
THR = .2
VERB = False
LOG_BASELINE_RESULTS = True
RUN_EXPERIMENTS = ("samples", "variance")

N_SAMPLES_VALUES = np.array([50, 60, 80, 100, 200, 500, 1000, 5000, 10000])
VAR_VALUES = np.array([1, 5, 10, 15, 20, 25, 30])
NOISE_TYPES = ("normal", "exp", "gumbel", "laplace")
JOINT_AGGS = ("mean", "median")
SKIP_IDX = []

BASE_DATA_PARAMS = {
    "graph_type": "er",
    "n_nodes": 100,
    "edges": 4,  # Edges per node; converted to total edges inside run_noise_exp.
    "edge_type": "positive",
    "w_range": (.5, 1),
    "n_samples": 1000,
    "var": 1,
}

NOISE_SCENARIOS = [
    {"name": "Gaussian", "suffix": "normal", "noise_type": "normal", "line_style": "-"},
    {"name": "Exponential", "suffix": "exp", "noise_type": "exp", "line_style": "--"},
    {"name": "Gumbel", "suffix": "gumbel", "noise_type": "gumbel", "line_style": ":"},
    {"name": "Laplace", "suffix": "laplace", "noise_type": "laplace", "line_style": "-."},
]

# Set to None to run every noise scenario from NOISE_SCENARIOS.
SELECTED_NOISE_TYPES = list(NOISE_TYPES)

# Set to None to run every experiment from build_experiments().
SELECTED_EXPERIMENT_LEGS = None
# SELECTED_EXPERIMENT_LEGS = [
#     "NOMAD-adam",
#     "NOMAD-fista",
#     "NonDAGMA",
#     "CoLiDE-EV",
#     "GOLEM-EV",
# ]

np.random.seed(SEED)
os.makedirs(PATH, exist_ok=True)


def _handle_termination(signum, frame):
    raise KeyboardInterrupt(f"Received signal {signum}; stopping experiments")


signal.signal(signal.SIGTERM, _handle_termination)


def get_lamb_value(n_nodes, n_samples, times=1):
    return np.sqrt(np.log(n_nodes) / n_samples) * times


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
        {
            "model": NonnegativeDAGMA_linear,
            "init": {"loss_type": "l2"},
            "args": {"lambda1": .05, "T": 4, "s": [1.0, .9, .8, .7], "warm_iter": 2e4, "max_iter": 7e4, "lr": .0003},
            "standarize": False,
            "adapt_lamb": False,
            "fmt": "s--",
            "leg": "NonDAGMA",
        },
        {
            "model": colide_ev,
            "args": {"lambda1": .05, "T": 4, "s": [1.0, .9, .8, .7], "warm_iter": 2e4, "max_iter": 7e4, "lr": .0003},
            "standarize": False,
            "fmt": "v--",
            "leg": "CoLiDE-EV",
        },
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


def filter_noise_scenarios(scenarios, selected_noise_types):
    if selected_noise_types is None:
        return scenarios

    selected_noise_types = list(selected_noise_types)
    by_suffix = {scenario["suffix"]: scenario for scenario in scenarios}
    by_type = {scenario["noise_type"]: scenario for scenario in scenarios}
    selected = []
    missing = []
    for noise_type in selected_noise_types:
        scenario = by_suffix.get(noise_type, by_type.get(noise_type))
        if scenario is None:
            missing.append(noise_type)
        else:
            selected.append(scenario)
    if missing:
        raise ValueError(f"Unknown noise type(s): {missing}")
    return selected


def sweep_config(experiment_name):
    if experiment_name == "samples":
        return {
            "xvals": np.asarray(N_SAMPLES_VALUES),
            "xlabel": "Number of samples",
            "x_label": "samples",
            "shd_plot_func": "semilogx",
            "err_plot_func": "loglog",
        }
    if experiment_name == "variance":
        return {
            "xvals": np.asarray(VAR_VALUES),
            "xlabel": "Noise variance",
            "x_label": "variance",
            "shd_plot_func": "plot",
            "err_plot_func": "semilogy",
        }
    raise ValueError(f"Unknown experiment: {experiment_name}")


def data_params_for_xval(base_data_p, scenario, experiment_name, xval):
    data_p_aux = base_data_p.copy()
    data_p_aux["edges"] *= data_p_aux["n_nodes"]
    data_p_aux["noise_type"] = scenario["noise_type"]

    if experiment_name == "samples":
        data_p_aux["n_samples"] = int(xval)
        data_p_aux["var"] = 1
    elif experiment_name == "variance":
        data_p_aux["n_samples"] = (
            10 * data_p_aux["n_nodes"]
            if data_p_aux["n_samples"] is None
            else data_p_aux["n_samples"]
        )
        data_p_aux["var"] = xval
    else:
        raise ValueError(f"Unknown experiment: {experiment_name}")

    return data_p_aux


def run_noise_exp(g, base_data_p, scenario, experiment_name, xvals, exps, thr=.2, verb=False):
    shd, tpr, fdr, fscore, err, acyc, runtime, dag_count = [
        np.zeros((len(xvals), len(exps))) for _ in range(8)
    ]
    x_label = sweep_config(experiment_name)["x_label"]

    for i, xval in enumerate(xvals):
        if g % N_CPUS == 0:
            print(
                f'Graph: {g + 1}, experiment={experiment_name}, '
                f'noise={scenario["suffix"]}, {x_label}={xval}',
                flush=True,
            )

        data_p_aux = data_params_for_xval(base_data_p, scenario, experiment_name, xval)

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

            if (verb or LOG_BASELINE_RESULTS) and (g % N_CPUS == 0):
                print(
                    f'\t-{exp["leg"]}: shd {shd[i, j]}  -  err: {err[i, j]:.3f}'
                    f"  -  time: {runtime[i, j]:.3f}",
                    flush=True,
                )

    return shd, tpr, fdr, fscore, err, acyc, runtime, dag_count


def noise_results_prefix(experiment_name, scenario, data_p):
    return (
        f'{PATH}noise_{experiment_name}_{scenario["suffix"]}_'
        f'{data_p["graph_type"].upper()}graph_{data_p["edges"]}N'
    )


def save_noise_results(file_prefix, metrics, exps, xvals, experiment_name, scenario):
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
        xvals=xvals,
        experiment_name=experiment_name,
        noise_scenario=scenario,
    )
    print("SAVED in file:", file_prefix, flush=True)

    prefix = f"{PATH}noise_{experiment_name}_{scenario['suffix']}"
    utils.data_to_csv(f"{prefix}_err_mean.csv", exps, xvals, np.mean(err, axis=0))
    utils.data_to_csv(f"{prefix}_err_std.csv", exps, xvals, np.std(err, axis=0))
    utils.data_to_csv(f"{prefix}_err_med.csv", exps, xvals, np.median(err, axis=0))
    utils.data_to_csv(f"{prefix}_err_prctile25.csv", exps, xvals, np.percentile(err, 25, axis=0))
    utils.data_to_csv(f"{prefix}_err_prctile75.csv", exps, xvals, np.percentile(err, 75, axis=0))
    utils.data_to_csv(f"{prefix}_shd_mean.csv", exps, xvals, np.mean(shd, axis=0))
    utils.data_to_csv(f"{prefix}_shd_std.csv", exps, xvals, np.std(shd, axis=0))
    utils.data_to_csv(f"{prefix}_shd_med.csv", exps, xvals, np.median(shd, axis=0))
    utils.data_to_csv(f"{prefix}_shd_prctile25.csv", exps, xvals, np.percentile(shd, 25, axis=0))
    utils.data_to_csv(f"{prefix}_shd_prctile75.csv", exps, xvals, np.percentile(shd, 75, axis=0))


def load_noise_results(file_prefix):
    file_name = f"{file_prefix}.npz"
    data = np.load(file_name, allow_pickle=True)
    print("Loaded noise results from", file_name, flush=True)
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


def print_metric_summary(metrics, exps, experiment_name, scenario):
    shd, _, _, _, err, _, runtime, _ = metrics
    print(f'----- Summary experiment={experiment_name}, noise={scenario["suffix"]} -----', flush=True)
    for j, exp in enumerate(exps):
        print(
            f'\t-{exp["leg"]}: mean shd {np.nanmean(shd[:, :, j]):.4f}'
            f"  -  mean err {np.nanmean(err[:, :, j]):.4f}"
            f"  -  mean time {np.nanmean(runtime[:, :, j]):.3f}",
            flush=True,
        )


def run_or_load_noise_results(scenario, experiment_name, xvals, exps, n_dags, thr=.2, verb=False):
    file_prefix = noise_results_prefix(experiment_name, scenario, BASE_DATA_PARAMS)

    if LOAD:
        return load_noise_results(file_prefix)

    n_jobs = max(1, min(N_CPUS, n_dags))
    print(
        f'Running experiment={experiment_name}, noise={scenario["name"]}. CPUs employed: {n_jobs}',
        flush=True,
    )

    t_init = perf_counter()
    parallel = Parallel(n_jobs=n_jobs, verbose=JOBLIB_VERBOSE)
    try:
        results = parallel(
            delayed(run_noise_exp)(
                g,
                BASE_DATA_PARAMS,
                scenario,
                experiment_name,
                xvals,
                exps,
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
    print_metric_summary(metrics, exps, experiment_name, scenario)
    if SAVE:
        save_noise_results(file_prefix, metrics, exps, xvals, experiment_name, scenario)

    return (*metrics, exps, xvals)


def plot_results(metrics, exps, xvals, experiment_name, scenario, skip_idx=None):
    os.makedirs(PATH, exist_ok=True)
    shd, tpr, fdr, fscore, err, acyc, runtime, dag_count = metrics
    config = sweep_config(experiment_name)
    skip = [] if skip_idx is None else list(skip_idx)
    prefix = f"{PATH}noise_{experiment_name}_{scenario['suffix']}"
    title_prefix = f'{experiment_name} - {scenario["name"]}'

    fig, _ = utils.plot_shd_error_pair(
        shd, err, exps, xvals,
        xlabel=config["xlabel"],
        shd_ylabel="Normalized SHD",
        err_ylabel="Fro Error",
        skip_idx=skip,
        agg="mean",
        deviation="std",
        alpha=0.25,
        shd_plot_func=config["shd_plot_func"],
        err_plot_func=config["err_plot_func"],
        title=f"{title_prefix} - mean",
        figsize=(8, 4),
    )
    fig.savefig(f"{prefix}_summary_mean.png", bbox_inches="tight")
    plt.close(fig)

    fig, _ = utils.plot_shd_error_pair(
        shd, err, exps, xvals,
        xlabel=config["xlabel"],
        shd_ylabel="Normalized SHD",
        err_ylabel="Fro Error",
        skip_idx=skip,
        agg="median",
        deviation="prctile",
        alpha=0.25,
        shd_plot_func=config["shd_plot_func"],
        err_plot_func=config["err_plot_func"],
        title=f"{title_prefix} - median",
        figsize=(8, 4),
    )
    fig.savefig(f"{prefix}_summary_median.png", bbox_inches="tight")
    plt.close(fig)

    utils.plot_all_metrics(shd, tpr, fdr, fscore, err, acyc, runtime, dag_count, xvals, exps,
                           skip_idx=skip, agg="mean", dev="std", xlabel=config["xlabel"])
    plt.gcf().suptitle(f"{title_prefix} - all metrics - mean")
    plt.savefig(f"{prefix}_all_metrics_mean.png", bbox_inches="tight")
    plt.close("all")

    utils.plot_all_metrics(shd, tpr, fdr, fscore, err, acyc, runtime, dag_count, xvals, exps,
                           skip_idx=skip, agg="median", dev="prctile", xlabel=config["xlabel"])
    plt.gcf().suptitle(f"{title_prefix} - all metrics - median")
    plt.savefig(f"{prefix}_all_metrics_median.png", bbox_inches="tight")
    plt.close("all")


def scenario_experiments(exps, suffix, line_style):
    return [
        {
            "leg": f'{exp["leg"]}-{suffix}',
            "fmt": exp.get("fmt", "o-")[0] + line_style,
        }
        for exp in exps
    ]


def plot_joint_results(scenario_results, experiment_name, agg="mean", skip_idx=None):
    os.makedirs(PATH, exist_ok=True)
    if len(scenario_results) < 2:
        return

    skip = set([] if skip_idx is None else skip_idx)
    config = sweep_config(experiment_name)
    reference_xvals = scenario_results[0]["xvals"]
    for result in scenario_results[1:]:
        if not np.array_equal(reference_xvals, result["xvals"]):
            raise ValueError("Cannot plot joint noise results with different x-value grids")

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

    fig, _ = utils.plot_shd_error_pair(
        shd_joint, err_joint, joint_exps, reference_xvals,
        xlabel=config["xlabel"],
        shd_ylabel="Normalized SHD",
        err_ylabel="Fro Error",
        skip_idx=[],
        agg=agg,
        deviation=deviation,
        alpha=0.25,
        shd_plot_func=config["shd_plot_func"],
        err_plot_func=config["err_plot_func"],
        title=f"noise types - {experiment_name} - {agg}",
        figsize=(10, 5),
    )
    fig.savefig(f"{PATH}noise_{experiment_name}_joint_{agg}.png", bbox_inches="tight")
    plt.close(fig)


def validate_run_experiments(run_experiments):
    valid = {"samples", "variance"}
    selected = tuple(run_experiments)
    unknown = [experiment for experiment in selected if experiment not in valid]
    if unknown:
        raise ValueError(f"Unknown run experiment(s): {unknown}")
    return selected


def main():
    exps = filter_experiments(build_experiments(), SELECTED_EXPERIMENT_LEGS)
    noise_scenarios = filter_noise_scenarios(NOISE_SCENARIOS, SELECTED_NOISE_TYPES)
    run_experiments = validate_run_experiments(RUN_EXPERIMENTS)

    print(f"Selected experiments: {', '.join(run_experiments)}", flush=True)
    print(f"Selected noise types: {', '.join(scenario['suffix'] for scenario in noise_scenarios)}", flush=True)
    print(f"Selected baselines: {', '.join(exp['leg'] for exp in exps)}", flush=True)

    for experiment_name in run_experiments:
        config = sweep_config(experiment_name)
        xvals = np.asarray(config["xvals"])
        scenario_results = []

        for scenario in noise_scenarios:
            *metrics, scenario_exps, scenario_xvals = run_or_load_noise_results(
                scenario,
                experiment_name,
                xvals,
                exps,
                N_DAGS,
                thr=THR,
                verb=VERB,
            )
            plot_results(metrics, scenario_exps, scenario_xvals, experiment_name, scenario, skip_idx=SKIP_IDX)
            scenario_results.append({
                "scenario": scenario,
                "metrics": metrics,
                "exps": scenario_exps,
                "xvals": scenario_xvals,
            })

        for agg in JOINT_AGGS:
            plot_joint_results(scenario_results, experiment_name, agg=agg, skip_idx=SKIP_IDX)


if __name__ == "__main__":
    main()
