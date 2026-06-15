import argparse
import os
import random
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib
matplotlib.use("Agg")
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from numpy import linalg as la
from sklearn.metrics import f1_score

from src.model import MetMulDagma, Nonneg_dagma
import src.utils as utils


SEED = 0
PATH_SACHS = str(ROOT / "datasets" / "sachs") + os.sep


def get_lamb_value(n_nodes, n_samples, times=1):
    return np.sqrt(np.log(n_nodes) / n_samples) * times


def generate_data(quick=False, dataset="SYNTH"):
    n_nodes = 100
    n_samples = 5000
    graph_type = "er"
    edge_type = "positive"
    w_range = (0.5, 1.0)
    variance = 1
    norm_x = False

    if quick:
        n_nodes = 25
        n_samples = 500

    if dataset == "SACHS":
        W_true = np.load(PATH_SACHS + "sachs_A_matrix.npy")
        X = np.load(PATH_SACHS + "sachs_X.npy")
        n_samples, n_nodes = X.shape
    else:
        W_true, _, X = utils.simulate_sem(
            n_nodes,
            n_samples,
            graph_type,
            4 * n_nodes,
            permute=False,
            edge_type=edge_type,
            w_range=w_range,
            noise_type="normal",
            var=variance,
        )

    if norm_x:
        X = X / la.norm(X, axis=1, keepdims=True)

    info = {
        "dataset": dataset,
        "n_nodes": n_nodes,
        "n_samples": n_samples,
        "graph_type": graph_type,
        "edges": 4 * n_nodes,
        "edge_type": edge_type,
        "w_range": w_range,
        "variance": variance,
        "norm_x": norm_x,
        "mean_degree": W_true.sum(axis=0).mean(),
        "mean_norm_X": la.norm(X, axis=1).mean(),
        "fidelity_error": la.norm(X - X @ W_true, "fro") ** 2 / n_samples,
    }
    return W_true, X, info


def get_algorithm_configs(
    n_nodes,
    n_samples,
    quick=False,
    include_mm_pgd=False,
    use_dummy=False,
    local_lipschitz_scale=1.0,
    min_stepsize=1e-12,
    max_stepsize=None,
    domain_bt_factor=0.5,
    domain_bt_max_iters=20,
    domain_bt_tol=1e-12,
):
    nonneg_cls = Nonneg_dagma
    mm_cls = MetMulDagma
    name_prefix = "STEP-" if use_dummy else ""

    pgd_lamb = get_lamb_value(n_nodes, n_samples, 1e-1)
    mm_fista_lamb = get_lamb_value(n_nodes, n_samples, 2e-1)
    mm_adam_lamb = get_lamb_value(n_nodes, n_samples, 1e-1)

    pgd_args = {
        "stepsize": 5e-3,
        "alpha": 2,
        "s": 1,
        "lamb": pgd_lamb,
        "max_iters": 1000000,
        "tol": 1e-6,
    }
    mm_fista_args = {
        "stepsize": 1e-5,
        "alpha_0": 0.01,
        "rho_0": 0.01,
        "s": 1,
        "lamb": mm_fista_lamb,
        "iters_in": 10000,
        "iters_out": 50,
        "tol": 1e-6,
        "beta": 1.5,
        "verb": True,
    }
    mm_adam_args = {
        "stepsize": 3e-4,
        "alpha_0": 0.01,
        "rho_0": 0.05,
        "s": 1,
        "lamb": mm_adam_lamb,
        "iters_in": 10000,
        "iters_out": 10,
        "tol": 1e-6,
        "beta": 5,
        "verb": True,
    }

    if quick:
        pgd_args.update({"max_iters": 1500, "tol": 1e-5})
        mm_fista_args.update({"iters_in": 300, "iters_out": 6, "verb": False})
        mm_adam_args.update({"iters_in": 300, "iters_out": 6, "verb": False})

    if use_dummy:
        local_step_args = {
            "local_lipschitz_scale": local_lipschitz_scale,
            "min_stepsize": min_stepsize,
            "max_stepsize": max_stepsize,
        }
        domain_bt_args = {
            "domain_bt_factor": domain_bt_factor,
            "domain_bt_max_iters": domain_bt_max_iters,
            "domain_bt_tol": domain_bt_tol,
        }

        pgd_local_args = pgd_args.copy()
        pgd_local_args.update({"step_type": "local_lipschitz", **local_step_args})

        mm_fista_local_args = mm_fista_args.copy()
        mm_fista_local_args.update({"step_type": "local_lipschitz", **local_step_args})

        mm_fista_bt_args = mm_fista_args.copy()
        mm_fista_bt_args.update({"step_type": "domain_backtracking", **domain_bt_args})

        mm_fista_local_bt_args = mm_fista_args.copy()
        mm_fista_local_bt_args.update({
            "step_type": "local_lipschitz_domain_backtracking",
            **local_step_args,
            **domain_bt_args,
        })

        return [
            {
                "name": "STEP-PGD-fixed",
                "model": lambda: nonneg_cls(acyclicity="logdet", primal_opt="pgd", restart=False),
                "args": pgd_args,
                "fit_kind": "direct",
            },
            {
                "name": "STEP-PGD-localL",
                "model": lambda: nonneg_cls(acyclicity="logdet", primal_opt="pgd", restart=False),
                "args": pgd_local_args,
                "fit_kind": "direct",
            },
            {
                "name": "STEP-MM-FISTA-fixed",
                "model": lambda: mm_cls(acyclicity="logdet", primal_opt="fista", restart=True),
                "args": mm_fista_args,
                "fit_kind": "mm",
            },
            {
                "name": "STEP-MM-FISTA-localL",
                "model": lambda: mm_cls(acyclicity="logdet", primal_opt="fista", restart=True),
                "args": mm_fista_local_args,
                "fit_kind": "mm",
            },
            {
                "name": "STEP-MM-FISTA-domainBT",
                "model": lambda: mm_cls(acyclicity="logdet", primal_opt="fista", restart=True),
                "args": mm_fista_bt_args,
                "fit_kind": "mm",
            },
            {
                "name": "STEP-MM-FISTA-localL-domainBT",
                "model": lambda: mm_cls(acyclicity="logdet", primal_opt="fista", restart=True),
                "args": mm_fista_local_bt_args,
                "fit_kind": "mm",
            },
        ]

    configs = [
        {
            "name": name_prefix + "PGD",
            "model": lambda: nonneg_cls(acyclicity="logdet", primal_opt="pgd", restart=False),
            "args": pgd_args,
            "fit_kind": "direct",
        },
        {
            "name": name_prefix + "MM-FISTA",
            "model": lambda: mm_cls(acyclicity="logdet", primal_opt="fista", restart=True),
            "args": mm_fista_args,
            "fit_kind": "mm",
        },
        {
            "name": name_prefix + "MM-Adam",
            "model": lambda: mm_cls(acyclicity="logdet", primal_opt="adam"),
            "args": mm_adam_args,
            "fit_kind": "mm",
        },
    ]

    if include_mm_pgd:
        mm_pgd_lamb = get_lamb_value(n_nodes, n_samples, 1e-1)
        mm_pgd_args = {
            "stepsize": 3e-4,
            "alpha_0": 0.01,
            "rho_0": 0.05,
            "s": 1,
            "lamb": mm_pgd_lamb,
            "iters_in": 40000,
            "iters_out": 10,
            "tol": 1e-6,
            "beta": 2,
            "verb": True,
        }
        if quick:
            mm_pgd_args.update({"iters_in": 300, "iters_out": 6, "verb": False})
        configs.insert(
            1,
            {
                "name": name_prefix + "MM-PGD",
                "model": lambda: mm_cls(acyclicity="logdet", primal_opt="pgd"),
                "args": mm_pgd_args,
                "fit_kind": "mm",
            },
        )

    return configs


def compute_metrics(W_true, W_est, threshold=0.2):
    W_est_bin = utils.to_bin(W_est, threshold)
    W_true_bin = utils.to_bin(W_true, threshold)
    shd, tpr, fdr = utils.count_accuracy(W_true_bin, W_est_bin)
    return {
        "err": utils.compute_norm_sq_err(W_true, W_est),
        "err_bin": utils.compute_norm_sq_err(W_true_bin, W_est_bin),
        "shd": shd,
        "tpr": tpr,
        "fdr": fdr,
        "fscore": f1_score(W_true_bin.flatten(), W_est_bin.flatten()),
        "is_dag": utils.is_dag(W_est_bin),
        "mean_est_value": W_est.mean(),
    }


def run_algorithm(config, X, W_true, threshold=0.2):
    model = config["model"]()
    fit_args = config["args"].copy()

    print(f"\n== {config['name']} ==")
    print(f"lambda: {fit_args['lamb']:.6g}")

    t_init = time.time()
    W_est = model.fit(X, **fit_args, track_seq=False, track_diagnostics=True)
    runtime = time.time() - t_init

    metrics = compute_metrics(W_true, W_est, threshold)
    metrics.update(
        {
            "algorithm": config["name"],
            "runtime": runtime,
            "final_h": model.dagness(W_est),
            "n_diff_W": len(model.diff_W),
            "n_diagnostics": len(model.diagnostics),
        }
    )
    print(
        "runtime: {runtime:.3f}s | h: {final_h:.4g} | err: {err:.4g} | "
        "shd: {shd} | fscore: {fscore:.3f}".format(**metrics)
    )
    return model, W_est, metrics


def diagnostics_frame(models):
    frames = []
    for name, model in models.items():
        if model.diagnostics:
            frame = pd.DataFrame(model.diagnostics)
        else:
            frame = pd.DataFrame()
        frame.insert(0, "algorithm", name)
        frames.append(frame)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def plot_diagnostics(diag, out_dir):
    if diag.empty:
        return

    plot_specs = [
        ("h", "Acyclicity h(W)", "log"),
        ("spectral_radius", "Spectral radius", "linear"),
        ("diag_norm", "Diagonal norm", "log"),
        ("grad_norm", "Gradient norm", "log"),
        ("prox_grad_norm", "Prox-gradient norm", "log"),
        ("aug_lagrangian", "Augmented Lagrangian", "log"),
        ("alpha", "Alpha", "linear"),
        ("rho", "Rho", "linear"),
        ("fista_restarts", "FISTA restarts", "linear"),
    ]

    fig, axes = plt.subplots(3, 3, figsize=(15, 10))
    axes = axes.ravel()

    for ax, (column, title, scale) in zip(axes, plot_specs):
        if column not in diag:
            ax.axis("off")
            continue
        for name, group in diag.groupby("algorithm"):
            values = group[column].to_numpy(dtype=float)
            x = group["outer_iter"].to_numpy(dtype=float)
            if np.all(np.isnan(values)):
                continue
            if scale == "log":
                values = np.where(values > 0, values, np.nan)
                ax.semilogy(x, values, marker="o", label=name)
            else:
                ax.plot(x, values, marker="o", label=name)
        ax.set_title(title)
        ax.set_xlabel("outer iteration")
        ax.grid(True)

    axes[0].legend()
    fig.tight_layout()
    fig.savefig(out_dir / "diagnostics_curves.png", dpi=180)
    plt.close(fig)


def plot_diff_w(models, out_dir):
    fig, ax = plt.subplots(figsize=(8, 5))
    for name, model in models.items():
        if model.diff_W:
            ax.semilogy(model.diff_W, label=name)
    ax.set_title("Relative change in W")
    ax.set_xlabel("inner iteration")
    ax.grid(True)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "diff_W_curves.png", dpi=180)
    plt.close(fig)


def plot_summary(summary, out_dir):
    fig, axes = plt.subplots(1, 4, figsize=(14, 4))
    for ax, column, title in zip(
        axes,
        ["err", "shd", "final_h", "runtime"],
        ["Normalized error", "SHD", "Final h(W)", "Runtime (s)"],
    ):
        ax.bar(summary["algorithm"], summary[column])
        ax.set_title(title)
        ax.grid(True, axis="y")
        ax.tick_params(axis="x", rotation=25)
        if column in ["err", "final_h", "runtime"]:
            ax.set_yscale("log")
    fig.tight_layout()
    fig.savefig(out_dir / "summary_bars.png", dpi=180)
    plt.close(fig)


def save_estimates(estimates, out_dir):
    for name, W_est in estimates.items():
        safe_name = name.lower().replace("-", "_")
        np.save(out_dir / f"W_est_{safe_name}.npy", W_est)


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Run the logdet algorithms from "
            "synthetic/notebooks/run_algorithms.ipynb and plot diagnostics."
        )
    )
    parser.add_argument("--quick", action="store_true", help="Use a smaller synthetic setup for a fast smoke test.")
    parser.add_argument("--dataset", choices=["SYNTH", "SACHS"], default="SYNTH")
    parser.add_argument(
        "--out-dir",
        default=str(ROOT / "results" / "diagnostics"),
    )
    parser.add_argument("--threshold", type=float, default=0.2)
    parser.add_argument(
        "--local-lipschitz-scale",
        type=float,
        default=1.0,
        help="Multiplier for the local-Lipschitz step, eta = scale / L.",
    )
    parser.add_argument("--min-stepsize", type=float, default=1e-12)
    parser.add_argument("--max-stepsize", type=float, default=None)
    parser.add_argument("--domain-bt-factor", type=float, default=0.5)
    parser.add_argument("--domain-bt-max-iters", type=int, default=20)
    parser.add_argument("--domain-bt-tol", type=float, default=1e-12)
    parser.add_argument(
        "--include-mm-pgd",
        action="store_true",
        help="Also run the method-of-multipliers variant with PGD as the inner solver.",
    )
    parser.add_argument("--dummy", action="store_true", help="Compare the trial step-size variants.")
    return parser.parse_args()


def main():
    args = parse_args()

    np.random.seed(SEED)
    random.seed(SEED)

    out_dir = Path(args.out_dir)
    if args.quick:
        out_dir = out_dir / "quick"
    if args.dummy:
        out_dir = out_dir / "dummy"
    out_dir.mkdir(parents=True, exist_ok=True)

    W_true, X, info = generate_data(quick=args.quick, dataset=args.dataset)
    print("Scenario:")
    for key, value in info.items():
        print(f"  {key}: {value}")

    configs = get_algorithm_configs(
        info["n_nodes"],
        info["n_samples"],
        quick=args.quick,
        include_mm_pgd=args.include_mm_pgd,
        use_dummy=args.dummy,
        local_lipschitz_scale=args.local_lipschitz_scale,
        min_stepsize=args.min_stepsize,
        max_stepsize=args.max_stepsize,
        domain_bt_factor=args.domain_bt_factor,
        domain_bt_max_iters=args.domain_bt_max_iters,
        domain_bt_tol=args.domain_bt_tol,
    )

    models = {}
    estimates = {}
    summary_rows = []
    for config in configs:
        model, W_est, metrics = run_algorithm(config, X, W_true, threshold=args.threshold)
        models[config["name"]] = model
        estimates[config["name"]] = W_est
        summary_rows.append(metrics)

    summary = pd.DataFrame(summary_rows)
    diag = diagnostics_frame(models)

    summary.to_csv(out_dir / "summary.csv", index=False)
    diag.to_csv(out_dir / "diagnostics.csv", index=False)
    pd.DataFrame([info]).to_csv(out_dir / "scenario.csv", index=False)
    save_estimates(estimates, out_dir)

    plot_diagnostics(diag, out_dir)
    plot_diff_w(models, out_dir)
    plot_summary(summary, out_dir)

    print("\nSaved:")
    print(f"  {out_dir / 'summary.csv'}")
    print(f"  {out_dir / 'diagnostics.csv'}")
    print(f"  {out_dir / 'diagnostics_curves.png'}")
    print(f"  {out_dir / 'diff_W_curves.png'}")
    print(f"  {out_dir / 'summary_bars.png'}")


if __name__ == "__main__":
    main()
