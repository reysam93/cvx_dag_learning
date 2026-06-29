#!/usr/bin/env python3
# coding: utf-8

import os
import signal
import sys
from copy import deepcopy
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

from baselines.colide import colide_ev
from baselines.dagma_linear import DAGMA_linear
from baselines.nonnegative_dagma_linear import NonnegativeDAGMA_linear
from baselines.notears import notears_linear
from baselines.sortnregress import VarSortNRegress


PATH = str(ROOT / "results" / "size") + os.sep
SAVE = True
LOAD = False
SEED = 10
N_CPUS = max(1, int(os.environ.get("N_CPUS", os.cpu_count() or 1)))
JOBLIB_VERBOSE = max(0, int(os.environ.get("JOBLIB_VERBOSE", 0)))

N_DAGS = 50
THR = .2
VERB = True
SIZES = np.array([50, 75, 100, 250, 500, 1000])
JOINT_AGGS = ("mean", "median")
SKIP_IDX = []
# Scenario filter. Use None to run all scenarios, or one/comma-separated values:
# "er", "erdos", "erdos-renyi", "erdos_renyi", "sf", "scale-free", "scale_free".
SELECTED_SCENARIOS = os.environ.get("GRAPH_SIZE_SCENARIOS")

BASE_DATA_PARAMS = {
    "edges": 4,  # Edges per node; converted to total edges inside run_size_exp.
    "edge_type": "positive",
    "w_range": (.5, 1),
    "var": 1,
    "n_samples": 5000,
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
                "iters_in": 10000,
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
                "stepsize": 3e-4,
                "step_type": "fixed",
                "alpha_0": .01,
                "rho_0": .05,
                "s": 1,
                "lamb": .1,
                "iters_in": 1000,
                "iters_out": 10,
                "beta": 2,
            },
            "init": {"primal_opt": "adam", "acyclicity": "logdet"},
            "adapt_lamb": True,
            "standarize": False,
            "fmt": "o-",
            "leg": "NOMAD-adam-v2",
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
                "iters_in": 30000,
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
            "model": DAGMA_linear,
            "init": {"loss_type": "l2"},
            "args": {
                "lambda1": .05,
                "T": 4,
                "s": [1.0, .9, .8, .8],
                "warm_iter": 2e4,
                "max_iter": 7e4,
                "lr": .0003,
            },
            "standarize": False,
            "fmt": "^-",
            "leg": "DAGMA",
        },
        {
            "model": VarSortNRegress,
            "args": {"w_threshold": 0.3},
            "standarize": False,
            "fmt": "D-",
            "leg": "SortNRegress",
        },
    ]


def build_experiments_for_scenario(scenario_suffix):
    exps = deepcopy(build_experiments())
    if scenario_suffix == "sf":
        for exp in exps:
            if exp["leg"] == "MM-fista":
                exp["args"]["iters_in"] = 50000
    return exps


def build_scenarios(data_p):
    return [
        {
            "name": "Erdos-Renyi",
            "suffix": "er",
            "file_tag": "ER",
            "data_params": {**data_p, "graph_type": "er"},
            "line_style": "-",
        },
        {
            "name": "Scale-free",
            "suffix": "sf",
            "file_tag": "SF",
            "data_params": {**data_p, "graph_type": "sf"},
            "line_style": "--",
        },
    ]


def filter_scenarios(scenarios, selected_scenarios):
    if not selected_scenarios:
        return scenarios

    aliases = {
        "er": "er",
        "erdos": "er",
        "erdos-renyi": "er",
        "erdos_renyi": "er",
        "sf": "sf",
        "scale-free": "sf",
        "scale_free": "sf",
    }
    selected = [
        aliases.get(item.strip().lower(), item.strip().lower())
        for item in selected_scenarios.split(",")
        if item.strip()
    ]
    by_suffix = {scenario["suffix"]: scenario for scenario in scenarios}
    missing = [suffix for suffix in selected if suffix not in by_suffix]
    if missing:
        raise ValueError(f"Unknown graph-size scenario(s): {missing}")
    return [by_suffix[suffix] for suffix in selected]


def filter_experiments(exps, selected_legs):
    if selected_legs is None:
        return exps

    selected_legs = list(selected_legs)
    by_leg = {exp["leg"]: exp for exp in exps}
    missing = [leg for leg in selected_legs if leg not in by_leg]
    if missing:
        raise ValueError(f"Unknown experiment legend(s): {missing}")
    return [by_leg[leg] for leg in selected_legs]


def run_single_size_exp(g, data_p, n_nodes, exps, thr=.2, verb=False):
    shd, tpr, fdr, fscore, err, acyc, runtime, dag_count = [
        np.zeros(len(exps)) for _ in range(8)
    ]

    if g % N_CPUS == 0:
        print(f"Graph: {g + 1}, nodes: {n_nodes}", flush=True)

    data_p_aux = data_p.copy()
    data_p_aux["n_nodes"] = int(n_nodes)
    data_p_aux["edges"] *= int(n_nodes)
    data_p_aux["var"] = 1 / np.sqrt(n_nodes) if data_p_aux["var"] is None else data_p_aux["var"]
    data_p_aux["n_samples"] = (
        10 * int(n_nodes)
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
                arg_aux["lamb"] = get_lamb_value(n_nodes, data_p_aux["n_samples"], arg_aux["lamb"])
            elif "lambda1" in arg_aux:
                arg_aux["lambda1"] = get_lamb_value(n_nodes, data_p_aux["n_samples"], arg_aux["lambda1"])

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

        shd[j], tpr[j], fdr[j] = utils.count_accuracy(W_true_bin, W_est_bin)
        shd[j] /= n_nodes
        fscore[j] = f1_score(W_true_bin.flatten(), W_est_bin.flatten())
        err[j] = utils.compute_norm_sq_err(W_true, W_est, norm_W_true)
        acyc[j] = (
            model.dagness(W_est)
            if model is not None and hasattr(model, "dagness")
            else float(not utils.is_dag(W_est_bin))
        )
        runtime[j] = t_end - t_init
        dag_count[j] += 1 if utils.is_dag(W_est_bin) else 0

        if verb and (g % N_CPUS == 0):
            print(
                f'\t-{exp["leg"]}: shd {shd[j]}  -  err: {err[j]:.3f}'
                f"  -  time: {runtime[j]:.3f}",
                flush=True,
            )

    return shd, tpr, fdr, fscore, err, acyc, runtime, dag_count


def size_results_prefix(scenario):
    data_p = scenario["data_params"]
    return f'{PATH}size_{scenario["file_tag"]}graph_{data_p["edges"]}N_{data_p["w_range"][1]}w'


def save_size_results(file_prefix, metrics, exps, sizes, scenario_suffix,
                      completed_sizes=None, write_tables=True):
    os.makedirs(PATH, exist_ok=True)
    shd, tpr, fdr, fscore, err, acyc, runtime, dag_count = metrics
    completed_sizes = [] if completed_sizes is None else list(completed_sizes)
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
        xvals=sizes,
        completed_sizes=np.asarray(completed_sizes),
    )
    print("SAVED in file:", file_prefix, flush=True)

    if not write_tables:
        return

    utils.data_to_csv(f"{PATH}size_{scenario_suffix}_err_med.csv", exps, sizes, np.median(err, axis=0))
    utils.data_to_csv(f"{PATH}size_{scenario_suffix}_err_prctile25.csv", exps, sizes, np.percentile(err, 25, axis=0))
    utils.data_to_csv(f"{PATH}size_{scenario_suffix}_err_prctile75.csv", exps, sizes, np.percentile(err, 75, axis=0))
    utils.data_to_csv(f"{PATH}size_{scenario_suffix}_shd_mean.csv", exps, sizes, np.mean(shd, axis=0))
    utils.data_to_csv(f"{PATH}size_{scenario_suffix}_shd_std.csv", exps, sizes, np.std(shd, axis=0))


def load_size_results(file_prefix):
    file_name = f"{file_prefix}.npz"
    data = np.load(file_name, allow_pickle=True)
    print("Loaded size results from", file_name, flush=True)
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


def run_or_load_size_results(scenario, sizes, exps, n_dags, thr=.2, verb=False):
    file_prefix = size_results_prefix(scenario)

    if LOAD:
        return load_size_results(file_prefix)

    n_jobs = max(1, min(N_CPUS, n_dags))
    print(f'Running scenario={scenario["name"]}. CPUs employed: {n_jobs}', flush=True)

    t_init = perf_counter()
    metrics = tuple(
        np.full((n_dags, len(sizes), len(exps)), np.nan, dtype=float)
        for _ in range(8)
    )
    completed_sizes = []

    for size_idx, n_nodes in enumerate(sizes):
        print(f'Running scenario={scenario["name"]}, nodes={n_nodes}', flush=True)
        parallel = Parallel(n_jobs=n_jobs, verbose=JOBLIB_VERBOSE)
        try:
            results = parallel(
                delayed(run_single_size_exp)(
                    g,
                    scenario["data_params"],
                    n_nodes,
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

        size_metrics = tuple(np.asarray(metric) for metric in zip(*results))
        for metric, size_metric in zip(metrics, size_metrics):
            metric[:, size_idx, :] = size_metric

        completed_sizes.append(n_nodes)
        if SAVE:
            save_size_results(
                file_prefix,
                metrics,
                exps,
                sizes,
                scenario["suffix"],
                completed_sizes=completed_sizes,
                write_tables=False,
            )

    t_end = perf_counter()
    print(f"----- Solved in {(t_end - t_init) / 60:.3f} minutes -----", flush=True)

    if SAVE:
        save_size_results(
            file_prefix,
            metrics,
            exps,
            sizes,
            scenario["suffix"],
            completed_sizes=completed_sizes,
            write_tables=True,
        )

    return (*metrics, exps, sizes)


def plot_results(metrics, exps, sizes, scenario_suffix, skip_idx=None):
    os.makedirs(PATH, exist_ok=True)
    shd, tpr, fdr, fscore, err, acyc, runtime, dag_count = metrics
    skip = [] if skip_idx is None else list(skip_idx)

    fig, axes = plt.subplots(1, 2, figsize=(8, 4))
    utils.plot_data(axes[0], shd, exps, sizes, "Number of nodes", "Normalized SHD", skip,
                    agg="mean", deviation="std", alpha=0.25)
    utils.plot_data(axes[1], err, exps, sizes, "Number of nodes", "Fro Error", skip,
                    agg="mean", deviation="std", alpha=0.25, plot_func="loglog")
    fig.suptitle(f"{scenario_suffix} - mean")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(f"{PATH}size_{scenario_suffix}_summary_mean.png", bbox_inches="tight")
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(8, 4))
    utils.plot_data(axes[0], shd, exps, sizes, "Number of nodes", "Normalized SHD", skip,
                    agg="median", deviation="prctile", alpha=0.25)
    utils.plot_data(axes[1], err, exps, sizes, "Number of nodes", "Fro Error", skip,
                    agg="median", deviation="prctile", alpha=0.25, plot_func="loglog")
    fig.suptitle(f"{scenario_suffix} - median")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(f"{PATH}size_{scenario_suffix}_summary_median.png", bbox_inches="tight")
    plt.close(fig)

    utils.plot_all_metrics(shd, tpr, fdr, fscore, err, acyc, runtime, dag_count, sizes, exps,
                           skip_idx=skip, agg="mean", dev="std", xlabel="Number of nodes")
    plt.gcf().suptitle(f"{scenario_suffix} - all metrics - mean")
    plt.savefig(f"{PATH}size_{scenario_suffix}_all_metrics_mean.png", bbox_inches="tight")
    plt.close("all")

    utils.plot_all_metrics(shd, tpr, fdr, fscore, err, acyc, runtime, dag_count, sizes, exps,
                           skip_idx=skip, agg="median", dev="prctile", xlabel="Number of nodes")
    plt.gcf().suptitle(f"{scenario_suffix} - all metrics - median")
    plt.savefig(f"{PATH}size_{scenario_suffix}_all_metrics_median.png", bbox_inches="tight")
    plt.close("all")


def scenario_experiments(exps, suffix, line_style):
    return [
        {
            "leg": f'{exp["leg"]}-{suffix.upper()}',
            "fmt": exp.get("fmt", "o-")[0] + line_style,
        }
        for exp in exps
    ]


def plot_joint_results(scenario_results, agg="mean", skip_idx=None):
    os.makedirs(PATH, exist_ok=True)
    if len(scenario_results) < 2:
        return

    skip = set([] if skip_idx is None else skip_idx)
    reference_xvals = scenario_results[0]["sizes"]
    for result in scenario_results[1:]:
        if not np.array_equal(reference_xvals, result["sizes"]):
            raise ValueError("Cannot plot joint results with different size grids")

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
    utils.plot_data(axes[0], shd_joint, joint_exps, reference_xvals, "Number of nodes", "Normalized SHD", [],
                    agg=agg, deviation=deviation, alpha=0.25)
    utils.plot_data(axes[1], err_joint, joint_exps, reference_xvals, "Number of nodes", "Fro Error", [],
                    agg=agg, deviation=deviation, alpha=0.25, plot_func="loglog")
    fig.suptitle(f"ER vs SF - {agg}")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(f"{PATH}size_er_sf_joint_{agg}.png", bbox_inches="tight")
    plt.close(fig)


def main():
    sizes = np.asarray(SIZES)
    scenario_results = []

    scenarios = filter_scenarios(build_scenarios(BASE_DATA_PARAMS), SELECTED_SCENARIOS)
    print(f"Selected scenarios: {', '.join(scenario['suffix'] for scenario in scenarios)}", flush=True)

    for scenario in scenarios:
        exps = filter_experiments(
            build_experiments_for_scenario(scenario["suffix"]),
            SELECTED_EXPERIMENT_LEGS,
        )
        *metrics, scenario_exps, scenario_sizes = run_or_load_size_results(
            scenario,
            sizes,
            exps,
            N_DAGS,
            thr=THR,
            verb=VERB,
        )
        plot_results(metrics, scenario_exps, scenario_sizes, scenario["suffix"], skip_idx=SKIP_IDX)
        scenario_results.append({
            "scenario": scenario,
            "metrics": metrics,
            "exps": scenario_exps,
            "sizes": scenario_sizes,
        })

    for agg in JOINT_AGGS:
        plot_joint_results(scenario_results, agg=agg, skip_idx=SKIP_IDX)


if __name__ == "__main__":
    main()
