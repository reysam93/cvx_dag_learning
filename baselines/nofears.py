"""NoFears (NOTEARS-KKTS) baseline adapted from the official repository.

Original code:
https://github.com/skypea/DAG_No_Fear

The KKTS local search is imported unchanged from the official clone under
``code_aux/DAG_No_Fear``. This wrapper exposes the same ``fit(X, ...)``
interface as the other project baselines.
"""

import networkx as nx
import numpy as np
import scipy.optimize as sopt


def _load_kkts():
    try:
        from code_aux.DAG_No_Fear.local_search_given_matrix import (
            eval_h,
            local_search_given_W,
        )
    except ImportError as exc:
        raise ImportError(
            "NoFears requires the official repository at "
            "code_aux/DAG_No_Fear. Clone it with: "
            "git clone https://github.com/skypea/DAG_No_Fear.git "
            "code_aux/DAG_No_Fear"
        ) from exc
    return eval_h, local_search_given_W


class NoFearsLinear:
    """Linear NoFears baseline using NOTEARS followed by official KKTS.

    Defaults reproduce the settings reported by Wei, Gao, and Yu (2020):
    least-squares loss, L1 penalty 0.1, polynomial acyclicity function,
    acyclicity tolerance 1e-10, and edge threshold 0.3. An exact graph check
    removes the weakest edge from any residual numerical cycle.
    """

    def __init__(self, dtype=np.float64, verbose=False):
        self.dtype = dtype
        self.verbose = verbose
        self.vprint = print if verbose else lambda *args, **kwargs: None

    @staticmethod
    def is_dag(W):
        return nx.is_directed_acyclic_graph(nx.DiGraph(W))

    def dagness(self, W):
        eval_h, _ = _load_kkts()
        return float(eval_h(np.abs(np.asarray(W))))

    @staticmethod
    def _validate_data(X, dtype):
        X = np.asarray(X, dtype=dtype)
        if X.ndim != 2:
            raise ValueError("X must be a two-dimensional [n_samples, n_nodes] array.")
        if X.shape[0] < 2:
            raise ValueError("X must contain at least two samples.")
        if X.shape[1] < 2:
            raise ValueError("X must contain at least two variables.")
        if not np.isfinite(X).all():
            raise ValueError("X contains NaN or infinite values.")
        return X.copy()

    @staticmethod
    def _threshold(W, threshold):
        W = np.asarray(W).copy()
        W[np.abs(W) < threshold] = 0
        np.fill_diagonal(W, 0)
        return W

    @staticmethod
    def _break_cycles(W):
        """Remove the weakest edge in each detected cycle until W is a DAG."""
        W = np.asarray(W).copy()
        removed_edges = []
        graph = nx.DiGraph(W)

        while not nx.is_directed_acyclic_graph(graph):
            cycle = nx.find_cycle(graph)
            source, target = min(
                cycle,
                key=lambda edge: (abs(W[edge[0], edge[1]]), edge[0], edge[1]),
            )
            weight = W[source, target]
            W[source, target] = 0
            graph.remove_edge(source, target)
            removed_edges.append((int(source), int(target), float(weight)))

        return W, removed_edges

    @staticmethod
    def _notears_h(W):
        """Polynomial NOTEARS acyclicity value and gradient."""
        d = W.shape[0]
        M = np.eye(d, dtype=W.dtype) + W * W / d
        E = np.linalg.matrix_power(M, d - 1)
        h = (E.T * M).sum() - d
        G_h = 2 * E.T * W
        return h, G_h

    def _fit_notears(
        self,
        X,
        lambda1,
        max_iter,
        h_tol,
        rho_init,
        rho_factor,
        rho_max,
        h_progress_rate,
    ):
        """Run the polynomial NOTEARS initialization used by NoFears."""
        n, d = X.shape

        def _adj(w):
            return (w[: d * d] - w[d * d :]).reshape(d, d)

        def _loss(W):
            residual = X - X @ W
            loss = 0.5 / n * np.sum(residual * residual)
            gradient = -X.T @ residual / n
            return loss, gradient

        def _func(w):
            W = _adj(w)
            loss, G_loss = _loss(W)
            h, G_h = self._notears_h(W)
            objective = loss + 0.5 * rho * h * h + alpha * h + lambda1 * w.sum()
            G_smooth = G_loss + (rho * h + alpha) * G_h
            gradient = np.concatenate(
                (G_smooth + lambda1, -G_smooth + lambda1),
                axis=None,
            )
            return objective, gradient

        bounds = [
            (0, 0) if i == j else (0, None)
            for _ in range(2)
            for i in range(d)
            for j in range(d)
        ]
        w_est = np.zeros(2 * d * d, dtype=self.dtype)
        rho = float(rho_init)
        alpha = 0.0
        h = np.inf
        self.notears_history_ = []
        self.optimizer_results_ = []

        for iteration in range(int(max_iter)):
            w_new = None
            h_new = None
            while rho < rho_max:
                result = sopt.minimize(
                    _func,
                    w_est,
                    method="L-BFGS-B",
                    jac=True,
                    bounds=bounds,
                )
                self.optimizer_results_.append(
                    {
                        "success": bool(result.success),
                        "status": int(result.status),
                        "message": str(result.message),
                        "nit": int(result.nit),
                        "nfev": int(result.nfev),
                        "fun": float(result.fun),
                    }
                )
                if not np.isfinite(result.x).all():
                    raise FloatingPointError(
                        "NoFears NOTEARS initialization produced non-finite coefficients."
                    )

                w_new = result.x
                h_new, _ = self._notears_h(_adj(w_new))
                if h_new > h_progress_rate * h:
                    rho *= rho_factor
                else:
                    break

            if w_new is None or h_new is None:
                raise RuntimeError(
                    "NoFears NOTEARS initialization reached rho_max before an update."
                )

            w_est, h = w_new, float(h_new)
            alpha += rho * h
            self.notears_history_.append(
                {"iteration": iteration + 1, "h": h, "rho": rho, "alpha": alpha}
            )
            self.vprint(
                f"NOTEARS iteration {iteration + 1}: "
                f"h={h:.4e}, rho={rho:.4e}, alpha={alpha:.4e}"
            )
            if h <= h_tol or rho >= rho_max:
                break

        return _adj(w_est), h, rho, alpha

    def fit(
        self,
        X,
        lambda1=0.1,
        w_threshold=0.3,
        max_iter=100,
        h_tol=1e-10,
        rho_init=1.0,
        rho_factor=10.0,
        rho_max=1e16,
        h_progress_rate=0.25,
        w_tol=1e-10,
        pen_tol=0.0,
        rev_edges="alt-full",
        minimize_z=True,
        init_no_pen=True,
        no_pen=False,
    ):
        """Fit NOTEARS-KKTS and store the weighted DAG in ``W_est``."""
        if lambda1 < 0:
            raise ValueError("lambda1 must be non-negative.")
        if w_threshold < 0:
            raise ValueError("w_threshold must be non-negative.")
        if max_iter < 1:
            raise ValueError("max_iter must be at least one.")
        if h_tol <= 0 or w_tol <= 0:
            raise ValueError("h_tol and w_tol must be positive.")
        if rho_init <= 0 or rho_factor <= 1 or rho_max <= rho_init:
            raise ValueError("rho parameters must satisfy 0 < rho_init < rho_max.")

        X_work = self._validate_data(X, self.dtype)
        X_work -= X_work.mean(axis=0, keepdims=True)
        self.X_mean_ = np.asarray(X, dtype=self.dtype).mean(axis=0)

        W_raw, self.notears_h_, self.rho_final_, self.alpha_final_ = (
            self._fit_notears(
                X_work,
                lambda1=lambda1,
                max_iter=max_iter,
                h_tol=h_tol,
                rho_init=rho_init,
                rho_factor=rho_factor,
                rho_max=rho_max,
                h_progress_rate=h_progress_rate,
            )
        )
        self.W_notears_raw = W_raw.copy()
        self.W_notears = self._threshold(W_raw, w_threshold)

        _, local_search_given_W = _load_kkts()
        W_search, search_h, search_iterations = local_search_given_W(
            X_work,
            self.W_notears.copy(),
            Wtol=w_tol,
            penTol=pen_tol,
            tau=lambda1,
            hTol=h_tol,
            revEdges=rev_edges,
            noPen=no_pen,
            initNoPen=init_no_pen,
            minimizeZ=minimize_z,
        )

        if not np.isfinite(W_search).all():
            raise FloatingPointError("NoFears KKTS produced non-finite coefficients.")

        self.W_search_raw = W_search.copy()
        self.search_h_ = float(search_h)
        self.search_iterations_ = int(search_iterations)
        self.W_thresholded_ = self._threshold(W_search, w_threshold).astype(
            self.dtype,
            copy=False,
        )
        self.h_thresholded_ = self.dagness(self.W_thresholded_)

        self.W_est, self.cycle_repair_edges_ = self._break_cycles(
            self.W_thresholded_
        )
        self.cycle_repair_count_ = len(self.cycle_repair_edges_)
        self.cycle_repair_applied_ = self.cycle_repair_count_ > 0
        if self.cycle_repair_applied_:
            max_removed = max(abs(edge[2]) for edge in self.cycle_repair_edges_)
            self.vprint(
                "NoFears exact-DAG safeguard removed "
                f"{self.cycle_repair_count_} edge(s); "
                f"largest removed magnitude={max_removed:.4e}."
            )

        self.h_final = self.dagness(self.W_est)

        if not self.is_dag(self.W_est):
            raise RuntimeError(
                "NoFears exact-DAG safeguard failed to remove all cycles."
            )

        return self.W_est
